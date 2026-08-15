"""Tests for the llama-server HTTP client (task 3.2) and its model_runs migration.

Every test spins up its own stub HTTP server bound to an ephemeral port on
127.0.0.1. The plan is explicit that this suite must pass whether or not a
real llama-server is listening on 8090, so no test here ever points a
`ModelClient` at port 8090, and none starts or requires one. Stub payloads
are short strings invented for these tests, never real transcript prose
(INV-9) -- the client under test never touches an observed session's
content at all, but the discipline holds regardless.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from palaver.extract.client import (
    ModelClient,
    ModelClientError,
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from palaver.store.migrate import connect, migrate
from palaver.store.schema import LATEST_VERSION

# Self-contained (no external $ref, per the orchestrator's verified
# constraint on the server's schema converter), reused across tests.
_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["status", "confidence"],
}


def _write_json(handler: BaseHTTPRequestHandler, status_code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    _write_raw(handler, status_code, body, content_type="application/json")


def _write_raw(
    handler: BaseHTTPRequestHandler,
    status_code: int,
    body: bytes,
    content_type: str = "text/plain",
) -> None:
    try:
        handler.send_response(status_code)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    except OSError:
        # The client may already have closed its side (e.g. after its own
        # timeout fired) by the time a deliberately slow handler gets here.
        pass


def _conforming_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    """A schema-conforming stub response: the positive control for every failure-path test."""
    content = json.dumps({"status": "blocked", "confidence": 0.87})
    _write_json(
        handler,
        200,
        {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 4242}},
    )


def _malformed_body_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    """The whole HTTP body is not valid JSON."""
    _write_raw(handler, 200, b"not json at all {{{")


def _unconstrained_text_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    """A valid envelope whose `content` is plain text, not schema-conforming JSON.

    The specific failure the orchestrator's research flagged: a build or
    request-shape mismatch producing unconstrained text under a 200 status
    must not be mistaken for a valid response.
    """
    _write_json(
        handler,
        200,
        {"choices": [{"message": {"content": "sure, here is your answer: blocked"}}]},
    )


def _missing_required_field_response(handler: BaseHTTPRequestHandler, body: bytes) -> None:
    """A schema-shaped JSON object missing a field the schema marks `required`."""
    content = json.dumps({"status": "blocked"})  # no "confidence"
    _write_json(handler, 200, {"choices": [{"message": {"content": content}}]})


def _make_slow_response(delay_seconds: float):
    """Build a handler that sleeps before returning a conforming response."""

    def _respond(handler: BaseHTTPRequestHandler, body: bytes) -> None:
        time.sleep(delay_seconds)
        _conforming_response(handler, body)

    return _respond


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, handle_post, *args, **kwargs):
        self._handle_post = handle_post
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *args) -> None:
        pass  # keep test output quiet; BaseHTTPRequestHandler logs to stderr by default

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self._handle_post(self, body)


class _StubServer:
    """An in-process HTTP server bound to an ephemeral 127.0.0.1 port."""

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
    """Yield a `start(handle_post) -> port` function.

    Every server started through it is closed when the test ends.
    """
    servers = []

    def _start(handle_post) -> int:
        server = _StubServer(handle_post)
        servers.append(server)
        return server.port

    yield _start

    for server in servers:
        server.close()


@pytest.fixture
def store_conn(tmp_path):
    """A connection to a store migrated to the latest schema version."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()


def _unused_port() -> int:
    """A port nothing is listening on: bind, read the assigned port, then close."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# =============================================================================
# Positive control: a schema-conforming response parses to the stub's own values
# =============================================================================


def test_conforming_response_returns_parsed_object_matching_stub(stub_server, store_conn):
    """The positive control for every failure-path test below.

    Without this, a client that raises unconditionally would pass the
    malformed-JSON and unreachable-port tests vacuously, and a client that
    never successfully parses anything would be indistinguishable from a
    working one.
    """
    port = stub_server(_conforming_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    result = client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)

    assert result == {"status": "blocked", "confidence": 0.87}


# =============================================================================
# Negative: malformed / non-conforming responses raise typed errors
# =============================================================================


def test_malformed_response_body_raises_typed_error(stub_server, store_conn):
    """A response body that is not valid JSON at all raises ModelResponseError."""
    port = stub_server(_malformed_body_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    with pytest.raises(ModelResponseError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)


def test_unconstrained_text_content_raises_typed_error(stub_server, store_conn):
    """A 200 whose message content is plain text, not JSON, raises ModelResponseError."""
    port = stub_server(_unconstrained_text_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    with pytest.raises(ModelResponseError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)


def test_missing_required_field_raises_typed_error(stub_server, store_conn):
    """A schema-shaped object missing a `required` field raises ModelResponseError."""
    port = stub_server(_missing_required_field_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    with pytest.raises(ModelResponseError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)


# =============================================================================
# Negative: unreachable port / timeout raise typed errors within the timeout
# =============================================================================


def test_unreachable_port_raises_within_configured_timeout(store_conn):
    """A port with nothing listening raises within the configured timeout."""
    client = ModelClient(store_conn, port=_unused_port(), timeout=2.0)

    started = time.monotonic()
    with pytest.raises(ModelClientError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)
    elapsed = time.monotonic() - started

    assert elapsed <= client.timeout + 1.0


def test_unreachable_port_raises_connection_error_specifically(store_conn):
    """Positive control for the error hierarchy: connection refusal is ModelConnectionError."""
    client = ModelClient(store_conn, port=_unused_port(), timeout=2.0)

    with pytest.raises(ModelConnectionError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)


def test_slow_response_past_timeout_raises_model_timeout_error(stub_server, store_conn):
    """A server that responds slower than the configured timeout raises ModelTimeoutError.

    The stub still answers (after 1s); the client's 0.2s timeout must fire
    first, which is what distinguishes this from the connection-refused
    tests above -- this exercises the read-timeout path, not connect-refused.
    """
    port = stub_server(_make_slow_response(1.0))
    client = ModelClient(store_conn, port=port, timeout=0.2)

    started = time.monotonic()
    with pytest.raises(ModelTimeoutError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)
    elapsed = time.monotonic() - started

    assert elapsed <= client.timeout + 1.0


# =============================================================================
# model_runs: every request lands a row with a latency; successes carry a prompt token count
# =============================================================================


def test_model_runs_row_count_matches_request_count_with_latency_and_tokens(
    stub_server, store_conn
):
    """Row count equals request count; every row carries a latency and a prompt token count."""
    port = stub_server(_conforming_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    for _ in range(3):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)

    rows = store_conn.execute("SELECT status, latency_ms, prompt_tokens FROM model_runs").fetchall()

    assert len(rows) == 3
    for status, latency_ms, prompt_tokens in rows:
        assert status == "done"
        assert latency_ms is not None and latency_ms >= 0
        assert prompt_tokens == 4242


def test_failed_request_still_writes_a_model_runs_row(stub_server, store_conn):
    """'Every request and its latency lands in model_runs' holds for failures too.

    prompt_tokens stays NULL -- there was never a response to read a token
    count from -- which is why migration 6 made that column nullable.
    """
    port = stub_server(_malformed_body_response)
    client = ModelClient(store_conn, port=port, timeout=5.0)

    with pytest.raises(ModelResponseError):
        client.complete(model="test-model", purpose="extraction", prompt="hi", schema=_SCHEMA)

    row = store_conn.execute("SELECT status, latency_ms, prompt_tokens FROM model_runs").fetchone()

    assert row is not None
    status, latency_ms, prompt_tokens = row
    assert status == "error"
    assert latency_ms is not None and latency_ms >= 0
    assert prompt_tokens is None


# =============================================================================
# INV-1: on_status fires during a blocking request; the channel never touches stdout
# =============================================================================


def test_on_status_emits_during_blocking_request_and_stdout_stays_silent(
    stub_server, store_conn, capsys
):
    """Positive control (on_status fired) paired with the negative assertion (stdout silent).

    Without the positive half, "stdout has no output" would pass just as
    well on a client that never emits progress at all.
    """
    port = stub_server(_make_slow_response(0.3))
    client = ModelClient(store_conn, port=port, timeout=5.0, progress_interval=0.05)
    messages = []

    result = client.complete(
        model="test-model",
        purpose="extraction",
        prompt="hi",
        schema=_SCHEMA,
        on_status=messages.append,
    )

    assert result == {"status": "blocked", "confidence": 0.87}
    assert len(messages) >= 1
    captured = capsys.readouterr()
    assert captured.out == ""


# =============================================================================
# Schema migration 6: latency_ms / prompt_tokens land on an upgraded store
# =============================================================================


def test_migration_6_adds_columns_to_an_upgraded_store(tmp_path):
    """Migrate to version 5, then forward to latest, and prove the columns exist there.

    This is the only assertion that fails if migration 6's columns were
    added by editing migration 1 in place instead of appending a new
    migration: a database frozen at version 5 and then migrated forward is
    exactly the shape an already-deployed store would have, which a fresh
    from-latest database (what every other test in this suite builds) can
    never distinguish from a correctly-appended migration.
    """
    db_path = tmp_path / "palaver.db"

    reached_v5 = migrate(db_path, target_version=5)
    assert reached_v5 == 5

    reached_latest = migrate(db_path)
    # `LATEST_VERSION`, not a literal: what this test is about is that the
    # columns arrive by *upgrade* rather than by a rewritten migration 1, and
    # a hardcoded number turns every later migration into a spurious failure
    # here. The `>= 6` keeps the claim honest -- forward is still forward.
    assert reached_latest == LATEST_VERSION >= 6

    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO model_runs(model, purpose, status, latency_ms, prompt_tokens) "
            "VALUES ('m', 'p', 'done', 120, 4096)"
        )
        conn.commit()
        row = conn.execute("SELECT latency_ms, prompt_tokens FROM model_runs").fetchone()
    finally:
        conn.close()

    assert row == (120, 4096)


def test_migration_6_rejects_negative_latency(tmp_path):
    """The non-negative CHECK constraint on latency_ms rejects a negative value."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO model_runs(model, purpose, status, latency_ms) "
                "VALUES ('m', 'p', 'done', -1)"
            )
    finally:
        conn.close()
