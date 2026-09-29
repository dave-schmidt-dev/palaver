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
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from palaver.bench import (
    BenchReport,
    run_bench,
)
from palaver.cli import bench as bench_cli
from palaver.cli import build_parser
from palaver.observer.daemon import extraction_schema

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
