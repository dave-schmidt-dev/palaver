"""
The real ModelExtractor against an in-process stub HTTP server: prompts, schema, and
unreachable-server failure.
"""

import json
import socket
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.adapters.codex import CodexAdapter
from palaver.ingest.cursors import CursorStore
from palaver.observer.daemon import (
    ModelExtractor,
    ObserverDaemon,
    extraction_schema,
)
from palaver.observer.signals import FORBIDDEN_PAYLOAD_KEYS, REFINEMENT_PAYLOAD_KEYS
from tests._test_scheduler_support import NOW, _assistant, _daemon, _human, _set_mtime, _write_store


def _write_codex_store(root: Path, session: str = "codex-session") -> Path:
    """Write one invented Codex rollout in its date-partitioned layout."""
    path = root / "2026" / "08" / "14" / f"rollout-20260814-{session}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "session_meta",
            "payload": {
                "cwd": "/tmp/invented-codex-project",
                "id": session,
                "session_id": session,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "invented Codex work"}],
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _set_mtime(path, timedelta(minutes=5))
    return path


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


def test_one_writer_extracts_ephemeral_state_for_claude_and_codex(tmp_path, stub_server):
    """The supervised daemon handles both supported sources in one writer."""
    claude_root = tmp_path / "claude-projects"
    codex_root = tmp_path / "codex-sessions"
    _write_store(claude_root, "claude-project", "claude-session", [_human(), _assistant()])
    _write_codex_store(codex_root)
    prompts: list[str] = []
    port = stub_server(_prompts_seen(prompts))
    daemon = ObserverDaemon(
        db_path=tmp_path / "store" / "palaver.db",
        adapters=(
            ClaudeCodeAdapter(root=claude_root),
            CodexAdapter(root=codex_root),
        ),
        cursors=CursorStore(tmp_path / "cursors"),
        extractor=ModelExtractor(port=port, timeout=10.0),
        all=True,
    )

    with daemon:
        result = daemon.tick(now=NOW)
        sources = daemon.conn.execute("SELECT source FROM sessions ORDER BY source").fetchall()
        state_count = daemon.conn.execute("SELECT COUNT(*) FROM current_state").fetchone()[0]
        runs = daemon.conn.execute(
            "SELECT COUNT(*) FROM model_runs WHERE purpose = 'observer-extraction'"
        ).fetchone()[0]

    assert result.failed == ()
    assert len(result.extracted) == 2
    assert sources == [("claude-code",), ("codex",)]
    assert state_count == 6
    assert runs == 2
    assert len(prompts) == 2


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
