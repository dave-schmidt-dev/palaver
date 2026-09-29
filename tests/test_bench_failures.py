"""
Failing loudly: an unreachable server exits non-zero and names the failure per session.
"""

from __future__ import annotations

import socket
import sys

from palaver.bench import (
    run_bench,
)
from tests._bench_support import TEST_PROMPT_WORDS, _cli
from tests._bench_support import stub_server as stub_server


def _refused_port() -> int:
    """Return a port nothing is listening on.

    Bind, read the assigned port, close. A racing process could claim it in the
    window between, which would make the test fail rather than pass falsely —
    the safe direction for a test asserting a connection is refused.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
