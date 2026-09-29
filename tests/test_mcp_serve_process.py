"""
Serving for real (not the selftest): port-in-use refusal, signal handling before/after
the server exists, clean exit.
"""

from __future__ import annotations

import asyncio
import errno
import signal
import subprocess
import sys
from pathlib import Path

import pytest

import palaver
from palaver.cli import mcp as mcp_cli
from palaver.mcp import server as mcp_server
from tests._mcp_support import _SERVE_SUBPROCESS, _conn, _endpoint_line, _with_timeout
from tests._mcp_support import store as store


def test_a_port_already_in_use_is_refused_before_a_url_is_printed(tmp_path):
    """The failure mode is the printed URL, not the failed bind.

    stdout is this command's result. A caller that reads a URL from it has
    no way to know the process died a moment later, so it holds a confident
    endpoint for a server that is not there — INV-7's failure, arrived at
    through a socket instead of a status.
    """
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    held = mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    try:
        port = held.getsockname()[1]
        # A subprocess, not an in-process `run()`: if the guard ever stops
        # working, `run()` does not return an error, it serves forever, and
        # an in-process call would hang the suite rather than fail it.
        # Measured — mutating the bind away cost a mutation battery its whole
        # budget twice before this test was moved out of process.
        proc = subprocess.Popen(
            [sys.executable, "-c", _SERVE_SUBPROCESS, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError(
                "the collision was never refused: the process is still serving a "
                "port another socket holds"
            ) from None
        assert proc.returncode == 1, f"exited {proc.returncode}\n{stderr}"
        assert "http://" not in stdout, stdout
        assert "cannot bind" in stdout, stdout
        assert "already serving it" in stdout, stdout
    finally:
        held.close()


def test_a_bound_listener_refuses_a_second_one(tmp_path):
    """The positive control for the test above, and for `SO_REUSEADDR`.

    `bind_listener` sets `SO_REUSEADDR` so a restart is not blocked by the
    previous process's sockets in `TIME_WAIT`. On a platform where that
    option also permitted a second live listener, the refusal above would
    never fire and the test would pass by never reaching its own subject.
    """
    first = mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    try:
        with pytest.raises(OSError) as excinfo:
            mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, first.getsockname()[1]).close()
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        first.close()


def test_the_stop_signals_are_claimed_before_asyncio_can_take_them():
    """`asyncio.run` only installs its handler if the slot is still default.

    That check is the whole mechanism `own_stop_signals` relies on, so this
    asserts the precondition rather than trusting it: SIGINT starts as
    `default_int_handler`, and afterwards it does not.
    """
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        stop = mcp_cli.own_stop_signals()
        for sig in (signal.SIGINT, signal.SIGTERM):
            installed = signal.getsignal(sig)
            assert installed is not signal.default_int_handler
            assert installed.__self__ is stop
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def test_a_stop_that_arrives_before_the_server_exists_is_not_dropped(tmp_path):
    """The window between claiming the signal and uvicorn claiming it back.

    uvicorn installs its handlers inside `serve()`, so everything before
    that — ASGI startup included — is covered only by the handler
    `own_stop_signals` left in the slot. A stop recorded there and then
    forgotten leaves a server running that was told to stop, which is the
    worse failure of the two: at least a traceback exits.
    """
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    _, app = mcp_server.build_app(db_path)
    stop = mcp_cli._StopRequest()
    stop.record(signal.SIGTERM, None)
    announced: list[str] = []

    # Returns rather than serving: with no timeout in sight, a test that
    # hangs here is the defect, not a slow test.
    asyncio.run(
        _with_timeout(
            mcp_cli._serve_forever(
                app,
                mcp_server.DEFAULT_HOST,
                mcp_cli._free_port(),
                stop=stop,
                announce=lambda: announced.append("up"),
            )
        )
    )
    assert announced == [], "announced an endpoint it never served"


def test_a_stop_that_arrives_after_the_server_exists_reaches_it(tmp_path):
    """The other half: once attached, a recorded stop is applied to uvicorn."""
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    _, app = mcp_server.build_app(db_path)
    stop = mcp_cli._StopRequest()
    server = mcp_cli._build_server(app, mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    stop.attach(server)
    assert server.should_exit is False
    stop.record(signal.SIGINT, None)
    assert server.should_exit is True


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_a_stop_signal_exits_cleanly_rather_than_as_a_traceback(tmp_path, stop_signal):
    """Ctrl-C is the normal way to stop a foreground server, not a crash.

    It printed a twenty-five line `KeyboardInterrupt` traceback, because
    `sse_starlette` chains uvicorn's signal handler to whatever was
    installed when the app started, and under `asyncio.run` that is the
    handler which cancels the main task. SIGTERM is here for the same
    reason at one remove: it is what launchd sends, and a service that
    exits non-zero on a deliberate stop gets treated as a crash and
    throttled.
    """
    db_path = tmp_path / "serve.db"
    session_key = mcp_cli.seed_selftest_db(db_path)
    port = mcp_cli._free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVE_SUBPROCESS, str(db_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for the endpoint on stdout rather than sleeping: the URL is
        # written after the bind, so its arrival means the port is held.
        url = _endpoint_line(proc)
        assert url == f"http://{mcp_server.DEFAULT_HOST}:{port}{mcp_server.DEFAULT_PATH}"
        # The URL means "serving", so a client acting on it the instant it
        # appears must be answered. Also the only check anywhere that the
        # real serve path — not the selftest's ephemeral one — answers a
        # client at all.
        seen = asyncio.run(_with_timeout(mcp_cli._one_client(url, session_key)))
        assert [m["statement"] for m in seen["recall"]["memories"]] == [mcp_cli.SELFTEST_STATEMENT]
        proc.send_signal(stop_signal)
        stdout, stderr = proc.communicate(timeout=30)
    except BaseException:
        proc.kill()
        raise
    assert proc.returncode == 0, f"exited {proc.returncode}\n{stderr}"
    assert "Traceback" not in stderr, stderr
    assert "KeyboardInterrupt" not in stderr, stderr
    # Reached the end of `run()` rather than merely failing to crash.
    assert "stopped" in stderr, stderr


def test_the_server_advertises_a_version_in_the_handshake(store):
    """An empty version tells a user comparing two machines nothing."""
    server = mcp_server.build_server(Path("unused.db"), connect=lambda _: _conn(store))
    assert server.version == palaver.__version__
    assert server.version
