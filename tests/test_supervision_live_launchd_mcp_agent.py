"""
Live launchd, the mcp agent: loads and is found, restarts and keeps serving across it,
and a held session does not survive the restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import time

import pytest

from palaver.cli import install_agent
from palaver.cli import mcp as mcp_cli
from palaver.cli.install_agent import (
    MCP_TEMPLATE_PATH,
    RESTART_WINDOW_SECONDS,
    bootstrap,
    mcp_program_arguments,
    print_service,
    render_plist,
    wait_for_new_pid,
)
from tests._supervision_support import (
    MCP_SELFTEST_LABEL,
    STARTUP_WINDOW_SECONDS,
    _await_pid,
    _bootout_and_settle,
    _seed_memory,
    live,
)


def _contains_assertion(exc: BaseException) -> bool:
    """Whether `exc` is, or groups, an `AssertionError`.

    anyio's task groups re-raise failures wrapped in a `BaseExceptionGroup`,
    so an `isinstance` check alone would miss a test's own failed assertion
    once it has crossed an `async with` boundary.
    """
    if isinstance(exc, AssertionError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_contains_assertion(inner) for inner in exc.exceptions)
    return False


# --- live launchd: the mcp agent (task 6.5) --------------------------------


def _accepting(port: int, *, window: float, host: str = "127.0.0.1") -> bool:
    """Wait until something actually accepts on the port.

    Not `wait_for_running_pid`, and the difference is the whole reason this
    helper exists. launchd publishes a pid the moment the process execs, but
    this one is a Python interpreter that must import, open the store, and
    only then bind. A client that proceeded on the pid alone would connect
    into that gap and read `ECONNREFUSED` as a failed restart.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


async def _recall_over_http(url: str) -> list[str]:
    """Read the seeded project through a *fresh* client, and return statements."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool("palaver_recall", {"scope": {"project": "demo"}})
            assert not result.is_error, result.content[0].text
            page = json.loads(result.content[0].text)
    return [memory["statement"] for memory in page["memories"]]


@pytest.fixture
def loaded_mcp_agent(tmp_path):
    """Bootstrap the MCP server under real launchd, on a port nobody else holds."""
    db_path = tmp_path / "mcp-selftest.db"
    _seed_memory(db_path)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    port = mcp_cli._free_port()

    plist = tmp_path / f"{MCP_SELFTEST_LABEL}.plist"
    plist.write_text(
        render_plist(
            label=MCP_SELFTEST_LABEL,
            program_arguments=mcp_program_arguments(
                executable=install_agent._default_executable(),
                db_path=db_path,
                host="127.0.0.1",
                port=port,
            ),
            stdout_path=logs / "out.log",
            stderr_path=logs / "err.log",
            working_directory=tmp_path,
            template_path=MCP_TEMPLATE_PATH,
        ),
        encoding="utf-8",
    )

    _bootout_and_settle(MCP_SELFTEST_LABEL)
    result = bootstrap(plist)
    if result.returncode != 0:
        _bootout_and_settle(MCP_SELFTEST_LABEL, strict=False)
        pytest.fail(f"launchctl bootstrap failed: {(result.stderr or result.stdout).strip()}")
    try:
        yield {"port": port, "url": f"http://127.0.0.1:{port}/mcp", "logs": logs}
    finally:
        # Settling at teardown rather than only at the next setup: teardown is
        # where the job actually is, and leaving it half-removed makes the
        # *following* test fail for a reason that has nothing to do with it.
        _bootout_and_settle(MCP_SELFTEST_LABEL, strict=False)


@live
def test_the_mcp_agent_loads_and_launchctl_print_reports_it(loaded_mcp_agent):
    """The done-when's load check: `launchctl print` on the MCP label."""
    result = print_service(MCP_SELFTEST_LABEL)
    assert result.returncode == 0, (result.stdout + result.stderr)[:2000]
    assert MCP_SELFTEST_LABEL in result.stdout

    # The same control the observer's load test carries: a `print` that
    # succeeded for any argument would pass the assertion above unchanged.
    absent = print_service(f"{MCP_SELFTEST_LABEL}.absent")
    assert absent.returncode != 0


@live
def test_killing_the_mcp_agent_brings_back_a_different_pid(loaded_mcp_agent):
    """The done-when's restart check, against the real server and real launchd."""
    assert _accepting(loaded_mcp_agent["port"], window=STARTUP_WINDOW_SECONDS), (
        f"the server never bound {loaded_mcp_agent['port']} within "
        f"{STARTUP_WINDOW_SECONDS}s of bootstrap"
    )
    original = _await_pid(MCP_SELFTEST_LABEL)
    assert original is not None

    os.kill(original, signal.SIGKILL)
    restarted = wait_for_new_pid(MCP_SELFTEST_LABEL, original, on_status=lambda _message: None)

    assert restarted is not None, (
        f"no restart within {RESTART_WINDOW_SECONDS}s of killing pid {original}"
    )
    assert restarted != original


@live
def test_a_client_reading_across_the_mcp_agent_restart_gets_its_answer(loaded_mcp_agent):
    """The done-when's reconnect check — corrected, because the premise was wrong.

    The plan and README both said a restart is "invisible" to clients because
    HTTP transports reconnect automatically. Half of that is true. The TCP
    connection is re-established, but the MCP *session* is not: the restarted
    server has never seen the `Mcp-Session-Id` the client is holding, answers
    404, and the SDK surfaces that as `Session terminated` rather than
    re-initializing (mcp/client/streamable_http.py). Re-establishing the
    session is the host application's job, not the transport's.

    So this asserts both halves, from one restart. The held session must fail
    — that is the negative control, and without it "a fresh client works"
    would pass even if the kill had never landed — and a client that
    reconnects must get the same answer as before, from the same store, with
    no connection error.
    """
    url = loaded_mcp_agent["url"]
    port = loaded_mcp_agent["port"]

    assert _accepting(port, window=STARTUP_WINDOW_SECONDS), "the server never bound its port"
    before = asyncio.run(_recall_over_http(url))
    assert before, "the fixture seeded no memory, so the reads below prove nothing"

    original = _await_pid(MCP_SELFTEST_LABEL)
    assert original is not None
    os.kill(original, signal.SIGKILL)
    restarted = wait_for_new_pid(MCP_SELFTEST_LABEL, original, on_status=lambda _message: None)
    assert restarted is not None and restarted != original

    # The pid is back; the socket need not be. Waiting on the port rather than
    # on the pid is what keeps this from being a flaky race.
    assert _accepting(port, window=RESTART_WINDOW_SECONDS), (
        f"pid {restarted} exists but nothing is accepting on {port}"
    )

    after = asyncio.run(_recall_over_http(url))
    assert after == before


@live
def test_a_held_session_does_not_survive_the_mcp_agent_restart(loaded_mcp_agent):
    """Documents what actually happens, so the README's claim stays honest.

    Kept separate from the reconnect test because it asserts the opposite
    outcome and would otherwise read as a bug. It is the measured behaviour of
    the SDK, and the reason the reconnect test opens a new client rather than
    reusing one.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = loaded_mcp_agent["port"]
    assert _accepting(port, window=STARTUP_WINDOW_SECONDS)

    async def hold_across_restart() -> tuple[str, str]:
        """Return which boundary surfaced the failure, and what it said.

        Both boundaries are caught, not just the call. Tearing down a session
        whose server is gone can raise on `__aexit__` instead of on the call,
        and with only the call guarded that escape becomes a pytest *error*
        with an unwind traceback — the same finding, reported as a broken
        test. Which boundary it came from is returned rather than flattened,
        because the message assertion below is only meaningful for one of
        them.
        """
        source, message = "", ""
        try:
            async with streamable_http_client(loaded_mcp_agent["url"]) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    first = await session.call_tool(
                        "palaver_recall", {"scope": {"project": "demo"}}
                    )
                    assert not first.is_error

                    original = _await_pid(MCP_SELFTEST_LABEL)
                    assert original is not None
                    os.kill(original, signal.SIGKILL)
                    assert (
                        wait_for_new_pid(
                            MCP_SELFTEST_LABEL, original, on_status=lambda _message: None
                        )
                        is not None
                    )
                    assert _accepting(port, window=RESTART_WINDOW_SECONDS)

                    try:
                        await session.call_tool("palaver_recall", {"scope": {"project": "demo"}})
                    except Exception as exc:  # noqa: BLE001 - the escape is the finding
                        source, message = "call", f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - so is an escape from the unwind
            # An assertion above is a real failure, not the finding. Without
            # this the guard would swallow "the first call errored" and report
            # it as "the session did not survive", passing for the wrong
            # reason. anyio wraps failures in groups, so the check recurses.
            if _contains_assertion(exc):
                raise
            if not source:
                source, message = "unwind", f"{type(exc).__name__}: {exc}"
        return source, message

    source, message = asyncio.run(hold_across_restart())
    assert source, "the held session survived the restart; the README claim may now be true"
    # Pinned only on the call path, where the SDK's own wording ("Session
    # terminated", from its 404 handler) is what proves the session — rather
    # than the socket — is what died. An unwind failure is the same finding
    # arriving by a different route and carries no such guaranteed wording.
    if source == "call":
        assert "session" in message.lower(), message
