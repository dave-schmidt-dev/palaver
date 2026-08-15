"""Tests for the concurrent inference benchmark (task 4.4: `palaver.bench`).

**The vacuity this file is built around.** Every interesting assertion here —
"six requests were in flight at once", "the round fit inside the tick
interval" — passes trivially against a stub that answers instantly and a
harness that never actually overlaps anything. A serial `for` loop over six
fast requests reports six successes, a small total wall time, and six
`model_runs` rows. The only thing it cannot do is have two requests
outstanding at the same instant.

So the stub server holds every request at a `threading.Barrier` until all of
them have arrived. A concurrent harness releases the barrier and every request
succeeds; a serial one blocks the first request until the barrier times out,
which raises `BrokenBarrierError` *in the handler* and turns into a 500 the
harness reports as a failure. That is deliberate: without the timeout a serial
harness would deadlock, and a hung suite reads as a broken runner rather than
as a failed assertion. `test_six_sessions_are_driven_concurrently...` was
verified against a deliberately serialized harness, which fails it.

`ThreadingHTTPServer`, not `HTTPServer`: the single-threaded default
serializes the six requests inside the *stub*, which would fail a correct
harness for the stub's reasons.

This repository is public. Every prompt, label, and identifier here is invented
for the test; none of it is derived from a real observed session.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import palaver.bench as palaver_bench
from palaver.bench import (
    DEFAULT_PROMPT_FRACTION,
    FALLBACK_PROMPT_WORDS,
    SLOT_PATH_UNKNOWN_NOTE,
    TOKENS_PER_WORD,
    BenchReport,
    SessionTiming,
    SlotFileUsage,
    measure_slot_files,
    peak_rss_bytes,
    resolve_prompt_words,
    run_bench,
    synthesize_sessions,
    synthetic_prompt,
)
from palaver.cli import bench as bench_cli
from palaver.cli import build_parser
from palaver.observer.daemon import extraction_schema
from palaver.store.migrate import connect, migrate

#: How long the stub's barrier waits for every request to arrive. Long enough
#: that six threads reliably reach it on a loaded machine, short enough that a
#: serial harness fails the test in seconds instead of hanging the suite.
BARRIER_TIMEOUT = 4.0

#: Sessions every concurrency test drives, matching the plan's `--sessions 6`.
SESSIONS = 6

#: Kept tiny so no test's runtime depends on prompt size. Passed explicitly
#: wherever the derivation itself is not what is under test.
TEST_PROMPT_WORDS = 20

#: What the stub reports from `/props`, chosen so the derived per-slot budget
#: (4096 // 2 = 2048 tokens) is unmistakably smaller than the whole figure.
STUB_N_CTX = 4096
STUB_SLOTS = 2


class _Handler(BaseHTTPRequestHandler):
    """Answers `/v1/chat/completions` with a schema-conforming envelope."""

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        """Answer `/props` so the harness can size its prompt to one slot."""
        if self.path != "/props":
            self.send_error(404, "unknown path")
            return
        self._send_json(
            {
                "total_slots": STUB_SLOTS,
                "endpoint_slots": True,
                "model_path": "/fixture/invented.gguf",
                "model_alias": "fixture-model",
                "build_info": "fixture-build",
                "default_generation_settings": {"n_ctx": STUB_N_CTX},
            }
        )

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server = self.server
        request = json.loads(body)
        with server.lock:
            server.seen.append(request.get("model"))
            server.prompts.append(request["messages"][0]["content"])
        if server.barrier is not None:
            try:
                server.barrier.wait()
            except threading.BrokenBarrierError:
                with server.lock:
                    server.barrier_broken = True
                self.send_error(500, "requests did not arrive together")
                return
        if server.delay:
            time.sleep(server.delay)
        payload = json.dumps({key: None for key in extraction_schema()["required"]})
        self._send_json(
            {"choices": [{"message": {"content": payload}}], "usage": {"prompt_tokens": 123}}
        )

    def _send_json(self, obj: dict) -> None:
        envelope = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(envelope)))
        self.end_headers()
        self.wfile.write(envelope)

    def log_message(self, *args) -> None:
        """Silence the default stderr access log."""


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True


@dataclass
class _Handle:
    server: _StubServer
    host: str
    port: int

    @property
    def barrier_broke(self) -> bool:
        return self.server.barrier_broken

    @property
    def request_count(self) -> int:
        return len(self.server.seen)

    @property
    def prompts(self) -> list[str]:
        return list(self.server.prompts)


@pytest.fixture
def stub_server():
    """Factory for a threaded stub llama-server, torn down after each test."""
    started = []

    def build(*, barrier_parties: int | None = None, delay: float = 0.0) -> _Handle:
        server = _StubServer(("127.0.0.1", 0), _Handler)
        server.lock = threading.Lock()
        server.seen = []
        server.prompts = []
        server.delay = delay
        server.barrier_broken = False
        server.barrier = (
            None
            if barrier_parties is None
            else threading.Barrier(barrier_parties, timeout=BARRIER_TIMEOUT)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started.append((server, thread))
        host, port = server.server_address[0], server.server_address[1]
        return _Handle(server=server, host=host, port=port)

    yield build

    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _refused_port() -> int:
    """Return a port nothing is listening on.

    Bind, read the assigned port, close. A racing process could claim it in the
    window between, which would make the test fail rather than pass falsely —
    the safe direction for a test asserting a connection is refused.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run(handle, db_path: Path, **overrides) -> BenchReport:
    kwargs = {
        "db_path": db_path,
        "sessions": SESSIONS,
        "host": handle.host,
        "port": handle.port,
        "timeout": 30.0,
        "prompt_words": TEST_PROMPT_WORDS,
    }
    kwargs.update(overrides)
    return run_bench(**kwargs)


def _cli(argv: list[str], out) -> int:
    args = build_parser().parse_args(argv)
    return bench_cli.run(args, out=out, on_status=lambda message: None)


# =============================================================================
# Concurrency: the property a serial loop cannot fake
# =============================================================================


def test_six_sessions_are_driven_concurrently_not_serially(stub_server, tmp_path):
    """All six requests are outstanding at one instant, proven from both sides.

    The barrier proves the requests genuinely reached the server together; the
    harness's own peak-in-flight gauge proves it issued them together. Either
    alone is weaker: a gauge could count threads that never got a socket, and
    a barrier says nothing about what the harness reports.
    """
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, tmp_path / "bench.db")

    assert report.peak_in_flight == SESSIONS
    assert not handle.barrier_broke
    assert handle.request_count == SESSIONS
    assert report.ok
    assert len(report.successful_latencies_ms) == SESSIONS


def test_a_concurrent_round_that_arrives_late_is_reported_as_a_failure(stub_server, tmp_path):
    """The barrier's failure path is live, not merely configured.

    Five parties against six requests can never complete, so the barrier times
    out. Without this control, `not handle.barrier_broke` above would pass
    against a barrier that was never actually engaged.
    """
    handle = stub_server(barrier_parties=SESSIONS + 1)

    report = _run(handle, tmp_path / "bench.db", timeout=BARRIER_TIMEOUT + 20)

    assert handle.barrier_broke
    assert not report.ok
    assert not report.unreachable


def test_a_concurrent_round_fits_inside_a_configured_tick_interval(stub_server, tmp_path):
    """Bullet 4, with the paired overrun that makes it non-vacuous.

    Judged against a tick interval passed in, not against `DEFAULT_TICK_INTERVAL`
    — six instant requests fit inside 30 s no matter how the harness is
    written, so asserting against the default would prove nothing. The same
    stub and the same round are then judged against a budget they cannot meet,
    which proves the harness can report an overrun at all. A benchmark that can
    only ever say "fits" is not a measurement.
    """
    handle = stub_server(barrier_parties=SESSIONS, delay=0.2)

    report = _run(handle, tmp_path / "bench.db", tick_interval=10.0)

    assert report.fits_tick_interval
    assert report.tick_wall_s <= report.tick_interval_s
    # Concurrent, so the round costs about one delay, not six.
    assert report.tick_wall_s < 0.2 * SESSIONS

    overrun = BenchReport(
        sessions=report.sessions,
        tick_interval_s=0.01,
        tick_wall_s=report.tick_wall_s,
        peak_in_flight=report.peak_in_flight,
        timings=report.timings,
        rss_before_bytes=report.rss_before_bytes,
        rss_after_bytes=report.rss_after_bytes,
        slot_files=report.slot_files,
        unreachable=False,
    )
    assert not overrun.fits_tick_interval
    rendered = bench_cli.render_report(overrun, host="127.0.0.1", port=1, detailed=False)
    assert "OVER" in rendered


# =============================================================================
# Failing loudly: an unreachable server is not a zero
# =============================================================================


def test_bench_exits_non_zero_naming_an_unreachable_server(stub_server, tmp_path, capsys):
    port = _refused_port()

    code = _cli(
        [
            "bench",
            "--sessions",
            "2",
            "--report",
            "--port",
            str(port),
            "--prompt-words",
            str(TEST_PROMPT_WORDS),
            "--db",
            str(tmp_path / "unreachable.db"),
        ],
        sys.stdout,
    )
    captured = capsys.readouterr()

    assert code == 1
    assert f"127.0.0.1:{port}" in captured.err
    assert "unreachable" in captured.err

    # Positive control: the same command, the same flags, a server that is
    # actually there. Without this the exit code above could come from any
    # failure at all, including a broken argument parser.
    handle = stub_server()
    ok_code = _cli(
        [
            "bench",
            "--sessions",
            "2",
            "--report",
            "--host",
            handle.host,
            "--port",
            str(handle.port),
            "--prompt-words",
            str(TEST_PROMPT_WORDS),
            "--db",
            str(tmp_path / "reachable.db"),
        ],
        sys.stdout,
    )
    assert ok_code == 0


def test_an_unreachable_round_reports_every_session_as_a_connection_failure(tmp_path):
    port = _refused_port()

    report = run_bench(
        db_path=tmp_path / "bench.db",
        sessions=3,
        port=port,
        timeout=5.0,
        prompt_words=TEST_PROMPT_WORDS,
    )

    assert report.unreachable
    assert not report.ok
    assert [timing.error_kind for timing in report.timings] == ["connection"] * 3
    # And it did not quietly report a successful round of zero-latency work.
    assert report.successful_latencies_ms == ()


# =============================================================================
# The report: numbers a human reads, and units that are not silently wrong
# =============================================================================


def test_the_report_emits_peak_rss_and_per_tick_wall_time_as_numbers(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    rendered = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=True)

    wall = re.search(r"tick wall time: ([0-9]+\.[0-9]+) s", rendered)
    rss = re.search(r"peak RSS: ([0-9]+\.[0-9]+) MiB", rendered)
    assert wall and float(wall.group(1)) >= 0.0
    assert rss and float(rss.group(1)) > 0.0


def test_peak_rss_is_normalized_to_bytes_not_the_platform_unit(monkeypatch):
    """A 1024x-wrong memory number reads as a measurement, not as a bug.

    Both branches are exercised regardless of which platform the suite runs
    on. Comparing a live reading against a live reading would agree with
    whichever unit the module happened to pick, which is the whole bug.
    """
    monkeypatch.setattr(palaver_bench, "_RSS_IN_BYTES", True)
    assert palaver_bench.normalize_rss(4096) == 4096
    monkeypatch.setattr(palaver_bench, "_RSS_IN_BYTES", False)
    assert palaver_bench.normalize_rss(4096) == 4096 * 1024

    monkeypatch.undo()
    # And the flag itself is right for *this* platform, checked empirically
    # rather than by restating the source line. A pytest process occupies
    # somewhere between a few MiB and a few GiB; getting the unit backwards
    # puts the normalized figure a factor of 1024 outside that band in
    # whichever direction the mistake was made, on either platform.
    live = peak_rss_bytes()
    assert 4 * 1024 * 1024 < live < 4 * 1024 * 1024 * 1024, f"implausible peak RSS: {live} bytes"
    assert sys.platform in {"darwin", "linux"}


def test_the_rss_delta_is_the_attributable_number(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    assert report.rss_after_bytes >= report.rss_before_bytes
    assert report.rss_delta_bytes == report.rss_after_bytes - report.rss_before_bytes
    # `ru_maxrss` is a never-reset high-water mark, so the endpoint is much
    # larger than the growth. Asserting only on the endpoint would let a report
    # attribute the whole interpreter to the benchmark.
    assert report.rss_before_bytes > report.rss_delta_bytes


def test_the_report_flag_adds_the_per_session_table(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    detailed = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=True)
    summary = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=False)

    assert "bench-session-1" in detailed
    assert "bench-session-1" not in summary
    # Both still carry the measurement itself: the flag widens the output, it
    # does not gate the run.
    assert "peak in flight: 6" in summary


# =============================================================================
# Slot files: an honest "not measured" instead of a zero
# =============================================================================


def test_slot_file_usage_is_unavailable_when_no_path_is_supplied():
    usage = measure_slot_files(None)

    assert usage == SlotFileUsage("", False, 0, 0, SLOT_PATH_UNKNOWN_NOTE)
    assert "--slot-save-path" in usage.detail


def test_slot_file_usage_is_measured_when_a_path_is_supplied(tmp_path):
    slot_dir = tmp_path / "slots"
    slot_dir.mkdir()
    (slot_dir / "slot-0.bin").write_bytes(b"x" * 2048)
    (slot_dir / "slot-1.bin").write_bytes(b"y" * 1024)

    usage = measure_slot_files(slot_dir)

    assert usage.available
    assert usage.file_count == 2
    assert usage.total_bytes == 3072


def test_slot_file_usage_says_so_when_the_path_does_not_exist(tmp_path):
    usage = measure_slot_files(tmp_path / "absent")

    assert not usage.available
    assert usage.total_bytes == 0
    assert "no directory" in usage.detail


def test_the_rendered_report_never_shows_an_unmeasured_zero(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    rendered = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=False)

    assert "slot files: unavailable" in rendered
    assert "slot files: 0 file(s), 0.0 MiB" not in rendered


# =============================================================================
# Fixture synthesis and store bookkeeping
# =============================================================================


def test_synthesized_sessions_are_reused_rather_than_duplicated(tmp_path):
    db_path = tmp_path / "bench.db"
    migrate(db_path)
    conn = connect(db_path)
    try:
        first = synthesize_sessions(conn, SESSIONS)
        second = synthesize_sessions(conn, SESSIONS)
        conn.commit()
        assert first == second
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == SESSIONS
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
    finally:
        conn.close()


def test_synthesizing_zero_sessions_is_rejected(tmp_path):
    db_path = tmp_path / "bench.db"
    migrate(db_path)
    conn = connect(db_path)
    try:
        with pytest.raises(ValueError, match="must be positive"):
            synthesize_sessions(conn, 0)
        # Positive control: one session is accepted on the same connection.
        assert len(synthesize_sessions(conn, 1)) == 1
    finally:
        conn.close()


def test_every_session_records_a_model_runs_row(stub_server, tmp_path):
    db_path = tmp_path / "bench.db"
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, db_path)
    assert report.ok

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT session_id, status FROM model_runs WHERE purpose = 'bench-extraction'"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == SESSIONS
    assert {status for _, status in rows} == {"done"}
    assert {session_id for session_id, _ in rows} == {
        timing.session_id for timing in report.timings
    }


def test_a_zero_word_prompt_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        synthetic_prompt(0)
    # Positive control: a small prompt is built, and carries its label so two
    # concurrent requests never present the server with identical input.
    prompt = synthetic_prompt(TEST_PROMPT_WORDS, label="bench-session-3")
    assert prompt.startswith("bench-session-3")
    assert "invented benchmark transcript line" in prompt


def test_zero_sessions_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        run_bench(db_path=tmp_path / "bench.db", sessions=0)


def test_a_timing_reports_its_own_success(tmp_path):
    assert SessionTiming(label="a", session_id=1, latency_ms=5).ok
    assert not SessionTiming(
        label="a", session_id=1, latency_ms=5, error="boom", error_kind="response"
    ).ok


# =============================================================================
# Prompt sizing: the first live run 500'd because a fixed size cannot be right
# =============================================================================


def test_the_prompt_is_sized_to_one_slot_not_the_whole_reported_context():
    """`n_ctx` is the server's total, shared across its slots.

    Measured 2026-08-15 against the observed server: a ~19,700-token prompt to
    a four-slot server reporting `n_ctx: 32768` came back
    `500 Context size has been exceeded`, which is impossible against a
    32,768-token slot and expected against an 8,192-token one.
    """
    words = resolve_prompt_words(32768, 4)
    per_slot_tokens = 32768 // 4

    assert words * TOKENS_PER_WORD <= per_slot_tokens
    # And it is not merely small: it uses the share it was told it could.
    assert words * TOKENS_PER_WORD >= per_slot_tokens * DEFAULT_PROMPT_FRACTION * 0.9
    # The control that pins the division: the same context on one slot allows
    # roughly four times the prompt. Without it, a helper that ignored
    # `total_slots` and just returned something small would pass above.
    assert resolve_prompt_words(32768, 1) > words * 3
    # And the size that actually failed live is excluded.
    assert words < 12000


def test_an_unreadable_context_budget_falls_back_to_a_small_prompt():
    assert resolve_prompt_words(None, 4) == FALLBACK_PROMPT_WORDS
    assert resolve_prompt_words(0, 4) == FALLBACK_PROMPT_WORDS
    # Positive control: a real budget is not the fallback.
    assert resolve_prompt_words(32768, 4) != FALLBACK_PROMPT_WORDS


def test_run_bench_derives_the_prompt_size_from_the_server(stub_server, tmp_path):
    """The derivation is wired to the round, not merely available beside it."""
    handle = stub_server(barrier_parties=SESSIONS)

    report = run_bench(
        db_path=tmp_path / "bench.db",
        sessions=SESSIONS,
        host=handle.host,
        port=handle.port,
        timeout=30.0,
    )

    assert report.ok
    expected = resolve_prompt_words(STUB_N_CTX, STUB_SLOTS)
    sent = [len(prompt.split()) for prompt in handle.prompts]
    assert len(sent) == SESSIONS
    # Within a line's worth of words of the derived size, since the prompt is
    # built from whole repetitions of one line.
    assert all(abs(count - expected) < 20 for count in sent), sent
    # And it is the *derived* size, not the fallback the code uses when it
    # cannot read /props.
    assert expected != FALLBACK_PROMPT_WORDS


def test_an_explicit_prompt_size_overrides_the_derivation(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, tmp_path / "bench.db")

    assert report.ok
    sent = [len(prompt.split()) for prompt in handle.prompts]
    assert all(abs(count - TEST_PROMPT_WORDS) < 20 for count in sent), sent
