"""Tests for `palaver.eval.harness` and `palaver.cli.eval` (task 3.5).

No test here ever points a client at the real port 8090 or 8091, and none
starts a real `llama-server` subprocess or needs a real GGUF file --
`managed_e2b_server`'s `popen`/`check_health` and `run_eval`'s `ModelClient`
arguments are all injectable for exactly this reason, the same discipline
`tests/test_model_client.py` documents for task 3.2's client.

Every negative assertion here is paired with a positive control on the same
input shape, per the plan's standing rule and task 3.5's own explicit
example: "no shutdown call against port 8090" is checked together with "at
least one ordinary inference request to port 8090" in the same test, because
the first half alone passes vacuously if client wiring never reaches port
8090 at all.

INV-9: every fixture referenced here is one of Phase 1's already-vetted
`tests/fixtures/*.jsonl` files or `tests/fixtures/eval/decision-database-choice.jsonl`,
which reuses only phrasebook strings from `palaver/cli/fixture_lint.py`'s
`SYNTHESIZED_TEXT` allowlist -- no prose here was copied from a real session.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from palaver.store.migrate import connect

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_CONFORMING_EXTRACTION = {
    "current_task": "",
    "blockers_now": [],
    "questions_for_user": [],
    "user_decisions": [],
    "session_complete": False,
}


# =============================================================================
# Stub HTTP server (mirrors tests/test_model_client.py's pattern)
# =============================================================================


class _RecordingHandler(BaseHTTPRequestHandler):
    def __init__(self, requests, response_body, *args, **kwargs):
        self._requests = requests
        self._response_body = response_body
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self._requests.append(("POST", self.path))
        body = json.dumps(self._response_body()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._requests.append(("GET", self.path))
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self) -> None:
        self._requests.append(("DELETE", self.path))
        self.send_response(404)
        self.end_headers()


class _StubServer:
    def __init__(self, requests, response_body):
        def _factory(*args, **kwargs):
            return _RecordingHandler(requests, response_body, *args, **kwargs)

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
    """Yield `start(requests, response_body) -> port`; every server is closed at teardown."""
    servers = []

    def _start(requests, response_body=lambda: _wrap(_CONFORMING_EXTRACTION)) -> int:
        server = _StubServer(requests, response_body)
        servers.append(server)
        return server.port

    yield _start

    for server in servers:
        server.close()


def _wrap(extraction: dict, *, prompt_tokens: int = 100) -> dict:
    content = json.dumps(extraction)
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens},
    }


@pytest.fixture
def store_conn(tmp_path):
    from palaver.store.migrate import migrate

    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()
