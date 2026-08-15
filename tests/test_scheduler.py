"""Tests for the observer tick scheduler and daemon (task 4.1).

The daemon's whole economic argument is one comparison: extract a session
only when its cursor moved. Everything here defends that claim and the two
properties that make it trustworthy rather than merely true in the happy
path.

What this module defends, test by test:

* **An idle store costs zero inference requests.** Ten ticks over a warm,
  unchanging fixture corpus record zero extraction requests — with the
  discovery count asserted non-zero in the same test, because "zero requests
  because nothing changed" and "zero requests because discovery found
  nothing" are the same number and completely different failures. The same
  test then appends one record and asserts the eleventh tick records exactly
  one request, so the counter that reported zero is proven able to count.
* **Warm cursors are what "static" means.** `Cursor()` defaults to offset 0,
  so a cold store makes *every* session look changed on the first tick. The
  idle test seeds each cursor to where a tail leaves it and asserts the
  seeded offsets are non-zero, so the zero it later reports is a real skip
  and not a discovery that never happened.
* **One request per tick per changed session**, not zero and not two — and a
  third tick with nothing appended drops back to zero, which is what
  separates "gated on change" from "gated on nothing".
* **One writer.** `sqlite3.connect` is patched with a spy that counts *live*
  connections, so `migrate()`'s own short-lived connections are counted too,
  not just the daemon's injected factory. Two ticks open none. The same test
  opens a second connection by hand and asserts the spy reports two, so the
  peak-of-one is a measurement rather than a blind spot.
* **INV-1's status channel.** One tick emits at least one status update, and
  — through the CLI, with the *default* channel rather than a recorder, so
  the assertion is not vacuous — stdout stays empty while stderr carries the
  progress.
* **At-least-once.** A failing extractor leaves the cursor where it was, so
  the next tick re-schedules the same session; a positive control then
  succeeds and shows the cursor does advance when extraction does.
* **Shrink recovery is not idleness.** A store that was truncated or swapped
  returns a *lower* cursor (task 1.3), which a `>` gate would read as idle
  and never repair. The gate is `!=`, and this pins it.

No real session store (`~/.claude/`, `~/.codex/`,
`~/.local/share/opencode/`) is opened, globbed, or read by this module —
every sample directory is built under pytest's `tmp_path`, either from
records invented in this file or copied byte-for-byte from the committed,
hand-authored corpus in `tests/fixtures/` (INV-3, INV-9). The only socket
any test here opens is to an in-process stub server bound to `127.0.0.1` on
an ephemeral port.
"""

import io
import json
import os
import socket
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from palaver.cli import observe as observe_cli
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import Cursor, CursorStore
from palaver.observer.daemon import (
    DaemonNotStartedError,
    ModelExtractor,
    ObserverDaemon,
    ensure_scope,
    extraction_schema,
)
from palaver.observer.scheduler import plan_tick
from palaver.observer.signals import FORBIDDEN_PAYLOAD_KEYS, REFINEMENT_PAYLOAD_KEYS

#: Fixed reference time; every fixture's mtime is set relative to it, so no
#: assertion in this module depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --- helpers -----------------------------------------------------------------


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def _human(text: str = "please check the deploy", session_id: str = "session-1") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant(text: str = "on it", session_id: str = "session-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _write_store(root: Path, project: str, session: str, records: list[dict]) -> Path:
    """Write one session store in the layout `ClaudeCodeAdapter` requires."""
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _set_mtime(path, timedelta(minutes=5))
    return path


def _append(path: Path, record: dict) -> None:
    """Append one complete record, the way an agent writing this store would."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _frozen_sample(tmp_path: Path) -> Path:
    """Copy the committed flat `tests/fixtures/` corpus into adapter shape.

    `glob("*.jsonl")` is deliberately non-recursive: the flat root is the
    Claude Code transcript namespace, and `tests/fixtures/labels/` holds
    task 7.1's label artifacts, which are not session stores.
    """
    project_dir = tmp_path / "projects" / "fixture-corpus"
    project_dir.mkdir(parents=True)
    for fixture in sorted(FIXTURES_DIR.glob("*.jsonl")):
        copy = project_dir / fixture.name
        copy.write_bytes(fixture.read_bytes())
        _set_mtime(copy, timedelta(minutes=5))
    return tmp_path / "projects"


def _warm_cursors(adapter: ClaudeCodeAdapter, cursors: CursorStore) -> list[str]:
    """Seed every session's cursor to where a tail leaves it, as if already read.

    This is what makes a store *static* rather than merely unchanging: a
    cold `CursorStore` hands back `Cursor(offset=0)` for an unknown session,
    which the scheduler correctly reads as "everything after byte zero is
    new". Seeding uses the adapter itself, not the scheduler, so the fixture
    is not defined in terms of the thing under test.

    Returns:
        Every seeded `session_key`.
    """
    keys = []
    for ref in adapter.discover_sessions(all=True):
        cursor = adapter.tail(ref.path, Cursor()).cursor
        assert cursor.offset > 0, f"{ref.session_key} tailed to offset 0; nothing was seeded"
        cursors.save(ref.session_key, cursor)
        keys.append(ref.session_key)
    return keys


class RecordingExtractor:
    """Stands in for the one inference request `ModelExtractor` issues per session.

    Counting calls here counts inference requests exactly, because the real
    extractor's `__call__` issues exactly one `ModelClient.complete` and the
    daemon calls the extractor exactly once per scheduled session. Every
    test that asserts a count of zero also, in the same test, drives this
    same class to a non-zero count — a counter that cannot count is not a
    measurement.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, work, *, conn, on_status) -> None:
        on_status(f"stub extraction for {work.ref.session_key}")
        self.calls.append(work.ref.session_key)


class FailingExtractor:
    """An extractor that always raises, standing in for an unreachable server."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.exc = RuntimeError("model server unreachable") if exc is None else exc

    def __call__(self, work, *, conn, on_status) -> None:
        self.calls.append(work.ref.session_key)
        raise self.exc


class _ConnectionSpy:
    """Wraps `sqlite3.connect` and reports how many connections are still live.

    Liveness is probed rather than tracked: a closed `sqlite3.Connection`
    raises `ProgrammingError` on any statement, which is a fact about the
    connection itself and cannot drift from what the code under test
    actually did with it.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.opened: list[sqlite3.Connection] = []

    def __call__(self, *args, **kwargs) -> sqlite3.Connection:
        conn = self._real(*args, **kwargs)
        self.opened.append(conn)
        return conn

    def live(self) -> int:
        count = 0
        for conn in self.opened:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue
            count += 1
        return count


def _daemon(tmp_path: Path, sample_root: Path, extractor, **kwargs) -> ObserverDaemon:
    return ObserverDaemon(
        db_path=tmp_path / "store" / "palaver.db",
        adapters=(ClaudeCodeAdapter(root=sample_root),),
        cursors=CursorStore(tmp_path / "cursors"),
        extractor=extractor,
        all=True,
        **kwargs,
    )


def _cli_args(tmp_path: Path, sample_root: Path, **overrides) -> SimpleNamespace:
    args = {
        "once": True,
        "dry_run": False,
        "interval": 0.0,
        "max_ticks": None,
        "db": tmp_path / "store" / "palaver.db",
        "cursors": tmp_path / "cursors",
        "sample": sample_root,
        "all": True,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


# --- the idle case: zero inference requests ----------------------------------


def test_ten_ticks_over_an_idle_store_record_zero_inference_requests(tmp_path):
    """Ten ticks, nothing changed, zero requests — then one append proves the counter counts.

    The zero is only meaningful alongside the two facts asserted with it:
    discovery found sessions, and every one of them was *skipped* rather
    than never looked at. The eleventh tick is the positive control — the
    same daemon, the same counter, one appended record, exactly one request.
    """
    sample_root = _frozen_sample(tmp_path)
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    seeded = _warm_cursors(adapter, cursors)
    assert seeded, "the frozen fixture corpus produced no sessions to seed"

    extractor = RecordingExtractor()
    with _daemon(tmp_path, sample_root, extractor) as daemon:
        for _ in range(10):
            result = daemon.tick(now=NOW)
            assert result.plan.discovered == len(seeded)
            assert len(result.plan.skipped) == len(seeded)
            assert result.plan.scheduled == ()

        assert extractor.calls == []

        # Positive control: the counter above can count.
        changed = sorted((sample_root / "fixture-corpus").glob("*.jsonl"))[0]
        _append(changed, _human("one more turn"))
        eleventh = daemon.tick(now=NOW)

    assert len(eleventh.plan.scheduled) == 1
    assert len(extractor.calls) == 1


def test_idle_ticks_do_not_rewrite_cursors(tmp_path):
    """A skipped session's cursor file is not touched — idle really is a no-op."""
    sample_root = _frozen_sample(tmp_path)
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    seeded = _warm_cursors(adapter, cursors)

    before = {key: cursors.load(key).offset for key in seeded}
    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        daemon.tick(now=NOW)
        daemon.tick(now=NOW)

    assert {key: cursors.load(key).offset for key in seeded} == before


# --- the changed case: exactly one request per tick --------------------------


def test_advanced_cursor_emits_exactly_one_extraction_request_per_tick(tmp_path):
    """One append, one request. Two appends over two ticks, two requests. Then nothing."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human(), _assistant()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    cursors.save(key, ClaudeCodeAdapter(root=sample_root).tail(path, Cursor()).cursor)

    extractor = RecordingExtractor()
    with _daemon(tmp_path, sample_root, extractor) as daemon:
        _append(path, _human("and now the other thing"))
        first = daemon.tick(now=NOW)
        assert len(first.plan.scheduled) == 1
        assert extractor.calls == [key]

        _append(path, _assistant("done"))
        second = daemon.tick(now=NOW)
        assert len(second.plan.scheduled) == 1
        assert extractor.calls == [key, key]

        # Nothing appended: the gate is change, not existence.
        third = daemon.tick(now=NOW)

    assert third.plan.scheduled == ()
    assert len(third.plan.skipped) == 1
    assert extractor.calls == [key, key]


def test_a_successful_extraction_advances_the_persisted_cursor(tmp_path):
    """The cursor the tick reported is the cursor the store holds afterwards."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    assert cursors.load(key).offset == 0

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        result = daemon.tick(now=NOW)

    assert result.extracted == (key,)
    assert cursors.load(key).offset == result.plan.scheduled[0].cursor_after.offset
    assert cursors.load(key).offset > 0


# --- one writer --------------------------------------------------------------


def test_two_ticks_never_open_two_write_connections(tmp_path, monkeypatch):
    """Across two ticks exactly one connection is live, and the ticks open none.

    The spy patches `sqlite3.connect` itself rather than the daemon's
    injected factory, so `migrate()`'s own connections are inside the
    measurement. The hand-opened connection near the end is the positive
    control: without it, `live() == 1` could equally mean "the spy sees
    nothing".
    """
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    spy = _ConnectionSpy(sqlite3.connect)
    monkeypatch.setattr(sqlite3, "connect", spy)

    daemon = _daemon(tmp_path, sample_root, RecordingExtractor())
    daemon.start()
    opened_after_start = len(spy.opened)
    assert spy.live() == 1, "migration left a connection open beside the daemon's writer"

    daemon.start()  # idempotent: a second start must not open a second writer
    assert len(spy.opened) == opened_after_start

    daemon.tick(now=NOW)
    assert spy.live() == 1
    daemon.tick(now=NOW)
    assert spy.live() == 1
    assert len(spy.opened) == opened_after_start, "a tick opened its own connection"

    # Positive control: the spy can see a second live connection.
    extra = sqlite3.connect(str(daemon.db_path))
    assert spy.live() == 2
    extra.close()
    assert spy.live() == 1

    daemon.close()
    assert spy.live() == 0


def test_tick_before_start_raises_rather_than_opening_a_connection(tmp_path):
    """No implicit connect: a tick without `start()` is a named error."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    daemon = _daemon(tmp_path, sample_root, RecordingExtractor())
    with pytest.raises(DaemonNotStartedError):
        daemon.tick(now=NOW)


# --- INV-1: the status channel -----------------------------------------------


def test_observer_tick_emits_status(tmp_path, capsys):
    """One tick emits progress, and the default channel keeps stdout clean.

    Split deliberately. The first half injects a recorder, which proves the
    channel is *called* — but makes "nothing on stdout" trivially true,
    since the real channel never ran. The second half goes through the CLI
    with the default channel and asserts stderr carries the progress while
    stdout carries only the result line. That pairing is INV-1's gate; the
    recorder alone is not.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    messages: list[str] = []
    with _daemon(tmp_path, sample_root, RecordingExtractor(), on_status=messages.append) as daemon:
        daemon.tick(now=NOW)
    assert messages, "a tick emitted no status update at all"
    # Both assertions name a *production* emitter. "session-1 appears
    # somewhere" would also be satisfied by `RecordingExtractor`'s own
    # status line, which is this test's fixture, not the daemon's behavior.
    assert any("tailing" in message and "session-1" in message for message in messages), (
        "the scheduler emitted no per-session progress"
    )
    assert any(message.startswith("tick 1:") for message in messages), (
        "the daemon emitted no tick-level progress"
    )

    # The default channel, through the CLI. Cursors are pre-warmed so the
    # tick schedules nothing: this test must never reach for a model server.
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cli-cursors")
    cursors.save(adapter.session_key_for(path), adapter.tail(path, Cursor()).cursor)

    capsys.readouterr()  # discard anything the daemon half emitted
    out = io.StringIO()
    args = _cli_args(tmp_path, sample_root, cursors=tmp_path / "cli-cursors")
    exit_code = observe_cli.run(args, out=out, now=NOW)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err != "", "the default status channel emitted nothing"
    assert captured.out == "", "the status channel wrote to stdout"
    assert out.getvalue().startswith("tick 1:")
    assert "changed=0 extracted=0 failed=0" in out.getvalue()


# --- at-least-once: a failed extraction is retried ---------------------------


def test_failed_extraction_leaves_the_cursor_for_the_next_tick(tmp_path):
    """A raising extractor does not consume the cursor, so the next tick retries.

    The swap to a succeeding extractor at the end is the positive control:
    without it, "the cursor did not advance" would also pass on a daemon
    that never advances cursors at all.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)

    failing = FailingExtractor()
    daemon = _daemon(tmp_path, sample_root, failing)
    with daemon:
        first = daemon.tick(now=NOW)
        assert first.extracted == ()
        assert first.failed == ((key, "RuntimeError: model server unreachable"),)
        assert cursors.load(key).offset == 0

        second = daemon.tick(now=NOW)
        assert len(second.plan.scheduled) == 1
        assert failing.calls == [key, key]

        # Positive control: the cursor does advance once extraction succeeds.
        succeeding = RecordingExtractor()
        daemon.extractor = succeeding
        third = daemon.tick(now=NOW)
        assert third.extracted == (key,)
        assert cursors.load(key).offset > 0

        fourth = daemon.tick(now=NOW)

    assert fourth.plan.scheduled == ()
    assert succeeding.calls == [key]


# --- shrink recovery is change, not idleness ---------------------------------


def test_a_shrunken_store_is_scheduled_rather_than_read_as_idle(tmp_path):
    """A truncated or swapped store returns a lower cursor and must still be scheduled.

    `read_complete_records` repairs a past-EOF cursor by re-reading from 0
    (task 1.3), which returns an offset *below* the one it was handed. A
    scheduler gating on `cursor > before` would classify that as idle and
    never persist the repair, leaving the session silently unread forever.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human(), _assistant(), _human("more")])
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    key = adapter.session_key_for(path)
    cursors.save(key, adapter.tail(path, Cursor()).cursor)

    assert plan_tick((adapter,), cursors, all=True, now=NOW).scheduled == ()

    _write_store(sample_root, "proj", "session-1", [_human("a fresh, shorter file")])
    plan = plan_tick((adapter,), cursors, all=True, now=NOW)

    assert len(plan.scheduled) == 1
    work = plan.scheduled[0]
    assert work.cursor_after.offset < work.cursor_before.offset
    assert work.cursor_after.offset > 0


# --- the CLI -----------------------------------------------------------------


def test_dry_run_reports_the_plan_without_saving_a_cursor(tmp_path, capsys):
    """`--dry-run` answers "what would this tick extract" and changes nothing."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    cursors = CursorStore(tmp_path / "cursors")

    out = io.StringIO()
    exit_code = observe_cli.run(_cli_args(tmp_path, sample_root, dry_run=True), out=out, now=NOW)
    capsys.readouterr()

    assert exit_code == 0
    assert "dry-run: discovered=1 changed=1 unchanged=0" in out.getvalue()
    assert key in out.getvalue()
    assert cursors.load(key).offset == 0
    assert not (tmp_path / "store" / "palaver.db").exists()


def test_once_runs_exactly_one_tick(tmp_path, capsys):
    """`--once` is one tick, not a loop with a short interval."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    out = io.StringIO()
    exit_code = observe_cli.run(
        _cli_args(tmp_path, sample_root, cursors=tmp_path / "warm"),
        out=out,
        now=NOW,
    )
    capsys.readouterr()

    assert exit_code == 0
    assert out.getvalue().count("tick ") == 1


def test_run_sleeps_between_ticks_and_never_after_the_last(tmp_path):
    """A bounded run returns as soon as its work is done, not one interval later."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    naps: list[float] = []
    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        results = daemon.run(interval=7.0, max_ticks=3, sleep=naps.append, now=NOW)

    assert len(results) == 3
    assert naps == [7.0, 7.0]


def test_run_stops_when_asked(tmp_path):
    """`stop()` ends the loop without needing `max_ticks`."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    ticks: list[int] = []

    def stop() -> bool:
        return len(ticks) >= 2

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        results = daemon.run(
            interval=0.0,
            stop=stop,
            sleep=lambda _: ticks.append(1),
            now=NOW,
        )

    assert len(results) == 2


# --- the real extractor, against a stub server -------------------------------


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, handle_post, *args, **kwargs):
        self._handle_post = handle_post
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *args) -> None:
        pass  # BaseHTTPRequestHandler logs to stderr by default; keep the suite quiet

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self._handle_post(self, self.rfile.read(length))


class _StubServer:
    """An in-process HTTP server bound to an ephemeral 127.0.0.1 port (INV-9)."""

    def __init__(self, handle_post):
        def _factory(*args, **kwargs):
            return _Handler(handle_post, *args, **kwargs)

        self._server = HTTPServer(("127.0.0.1", 0), _factory)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_server():
    """Yield a `start(handle_post) -> port` function; every server is closed after."""
    servers: list[_StubServer] = []

    def _start(handle_post) -> int:
        server = _StubServer(handle_post)
        servers.append(server)
        return server.port

    yield _start

    for server in servers:
        server.close()


def _prompts_seen(sink: list[str]):
    """A stub responder that records the prompt and returns a conforming payload."""

    def _respond(handler: BaseHTTPRequestHandler, body: bytes) -> None:
        sink.append(json.loads(body)["messages"][-1]["content"])
        content = json.dumps(
            {
                "current_task": "wire the observer daemon",
                "remaining_work": "hook up slot management",
                "blockers_now": None,
                "open_questions": "",
            }
        )
        payload = json.dumps(
            {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 11}}
        ).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    return _respond


def test_model_extractor_writes_current_state_and_no_memories(tmp_path, stub_server):
    """The production extractor persists the ephemeral half and nothing durable.

    INV-4: `memories` is append-only, so a regeneratable field written there
    would grow without bound on churn. `extraction_from_model_payload` reads
    only `REFINEMENT_PAYLOAD_KEYS`, which is why this path structurally
    cannot reach `memories` — asserted here rather than assumed.
    """
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human(), _assistant()])
    prompts: list[str] = []
    port = stub_server(_prompts_seen(prompts))

    extractor = ModelExtractor(port=port, timeout=10.0)
    with _daemon(tmp_path, sample_root, extractor) as daemon:
        result = daemon.tick(now=NOW)
        assert result.failed == ()
        assert len(result.extracted) == 1

        rows = dict(daemon.conn.execute("SELECT key, value FROM current_state").fetchall())
        memories = daemon.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        runs = daemon.conn.execute(
            "SELECT COUNT(*) FROM model_runs WHERE purpose = 'observer-extraction'"
        ).fetchone()[0]

    assert rows["current_task"] == "wire the observer daemon"
    assert rows["open_questions"] == ""  # affirmatively nothing, not absent
    assert "blockers_now" not in rows  # null means "no opinion", so no row is written
    assert memories == 0
    assert runs == 1
    assert len(prompts) == 1
    assert "please check the deploy" in prompts[0]


def test_extraction_prompt_never_asks_for_a_status(tmp_path, stub_server):
    """INV-7: status is derived, so the request must not solicit one.

    Asserted against the same forbidden-key vocabulary
    `extraction_from_model_payload` rejects on the response side, so the two
    halves of the invariant cannot drift apart.
    """
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    prompts: list[str] = []
    port = stub_server(_prompts_seen(prompts))

    with _daemon(tmp_path, sample_root, ModelExtractor(port=port, timeout=10.0)) as daemon:
        daemon.tick(now=NOW)

    schema = extraction_schema()
    assert set(schema["properties"]) == set(REFINEMENT_PAYLOAD_KEYS)
    assert schema["additionalProperties"] is False, "the schema permits a status field"
    instruction = prompts[0].split("Transcript:")[0].lower()
    for forbidden in FORBIDDEN_PAYLOAD_KEYS:
        assert forbidden not in instruction, f"the instruction names {forbidden!r}"
    # Positive control: the assertion above can fail. The four fields the
    # instruction *does* ask for are present in the same text.
    for asked_for in REFINEMENT_PAYLOAD_KEYS:
        assert asked_for in instruction


def test_unreachable_model_server_is_a_recorded_failure_not_a_dead_daemon(tmp_path):
    """One sick session must not stop the daemon from observing healthy ones."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    _write_store(sample_root, "proj", "session-2", [_human("second session")])

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    unused_port = sock.getsockname()[1]
    sock.close()

    with _daemon(tmp_path, sample_root, ModelExtractor(port=unused_port, timeout=2.0)) as daemon:
        result = daemon.tick(now=NOW)

    assert len(result.plan.scheduled) == 2
    assert result.extracted == ()
    assert len(result.failed) == 2
    assert all("ModelConnectionError" in reason for _, reason in result.failed)


# --- scoping -----------------------------------------------------------------


def test_ensure_scope_is_idempotent_across_ticks(tmp_path):
    """Two ticks over one session produce one project row and one session row."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    adapter = ClaudeCodeAdapter(root=sample_root)
    (ref,) = adapter.discover_sessions(all=True)

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        first = ensure_scope(daemon.conn, ref)
        second = ensure_scope(daemon.conn, ref)
        projects = daemon.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        sessions = daemon.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    assert first == second
    assert projects == 1
    assert sessions == 1
    assert path.exists()
