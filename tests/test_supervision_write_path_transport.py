"""
The write path over the real socket and the real MCP transport: a correction reaching a
real daemon, elicitation, and the lock-only edge cases.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sqlite3
import subprocess
import sys

import pytest

from palaver.observer import socket as writer_socket
from tests._supervision_support import _HOLDER, _line_within, _raw_row, _row_count, _seed_memory
from tests._supervision_support import short_tmp as short_tmp

# ---------------------------------------------------------------------------
# The write path: what it will do, what it refuses, and what it never touches.
# ---------------------------------------------------------------------------

#: A daemon that actually serves. Holds the writer role, opens the one
#: writable connection, and answers requests until told to stop -- which is
#: the only way to test `request()` against something that can really reply.
_SERVER = """
import sys
from pathlib import Path
from palaver.observer.socket import single_writer, serve_request
from palaver.store.migrate import connect

db_path = Path(sys.argv[1])
conn = connect(db_path)
try:
    with single_writer(db_path) as server:
        print("HELD", flush=True)
        while True:
            serve_request(server, conn)
finally:
    conn.close()
"""


def test_a_correction_travels_over_the_real_socket_to_a_real_daemon(short_tmp):
    """End to end: a separate process holds the writer role and applies it.

    Every in-process test above shares this interpreter's connection, which
    is exactly the arrangement production does not have. This one has the
    writer in another process, reached only through the socket.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line_within(proc, 30.0) == "HELD"
        assert writer_socket.daemon_running(db_path) is True

        reply = writer_socket.request(
            db_path, {"op": "correct", "memory_id": memory_id, "statement": "over the wire"}
        )
        assert reply["ok"], reply
        assert _raw_row(db_path, reply["memory_id"])["statement"] == "over the wire"

        refused = writer_socket.request(db_path, {"op": "delete", "memory_id": memory_id})
        assert refused["ok"] is False
        assert refused["error"] == "UnsupportedOperationError"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_write_with_no_daemon_fails_loudly_rather_than_opening_a_second_writer(short_tmp):
    """The refusal is the feature.

    A fallback to a direct write would be invisible and would end the
    single-writer guarantee precisely when the daemon is already unhealthy
    -- the moment it is least safe to have two processes writing.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    assert writer_socket.daemon_running(db_path) is False

    with pytest.raises(writer_socket.DaemonUnavailableError, match="second"):
        writer_socket.request(
            db_path, {"op": "correct", "memory_id": memory_id, "statement": "no daemon"}
        )
    assert _row_count(db_path) == 1, "a write happened with no daemon running"


#: The MCP server, as `palaver mcp` runs it.
_MCP_SERVE = """
import sys
from palaver.cli import main
db, port = sys.argv[1], sys.argv[2]
sys.argv = ["palaver", "mcp", "--db", db, "--port", port]
sys.exit(main())
"""


async def _correct_over_the_wire(url, memory_id, statement, *, approve):
    """Drive `palaver_correct` the way a real client would, sign-off included."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import ElicitResult

    prompts = []

    async def elicitation_callback(_ctx, params):
        prompts.append(params.message)
        return ElicitResult(action="accept", content={"approved": approve, "note": "checked"})

    async with streamable_http_client(url) as streams:
        async with ClientSession(
            streams[0], streams[1], elicitation_callback=elicitation_callback
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "palaver_correct", {"memory_id": memory_id, "statement": statement}
            )
            return result, prompts


@pytest.mark.parametrize("approve", [True, False])
def test_a_correction_crosses_the_real_transport_and_the_real_socket(short_tmp, approve):
    """The whole path, with nothing stubbed: client, server, daemon, store.

    Every other test here replaces at least one link -- an in-process
    connection, a stub context. This one has a real client eliciting over a
    real streamable-HTTP back-channel, a real MCP server holding a `mode=ro`
    connection, and a real daemon in a third process applying the write. It
    is the only test that would catch a break in how those three meet.

    Parametrised on the answer because the negative case is the one that
    matters: a sign-off gate that writes regardless is worse than none, and
    it looks identical from the accept side.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    port = _free_port()

    daemon = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = None
    try:
        assert _line_within(daemon, 30.0) == "HELD"
        server = subprocess.Popen(
            [sys.executable, "-c", _MCP_SERVE, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        url = _line_within(server, 30.0)
        assert url.startswith("http://"), url

        result, prompts = asyncio.run(
            asyncio.wait_for(
                _correct_over_the_wire(url, memory_id, "corrected over the wire", approve=approve),
                timeout=60,
            )
        )
    finally:
        for proc in (server, daemon):
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)

    assert prompts, "the client was never asked to sign off"
    assert "the observer's original reading" in prompts[0], "the prompt hid what changes"

    if approve:
        assert not result.is_error, result.content[0].text
        assert _row_count(db_path) == 2
        successor = json.loads(result.content[0].text)
        assert _raw_row(db_path, successor["memory_id"])["statement"] == "corrected over the wire"
    else:
        assert result.is_error, "a refused sign-off returned success"
        assert _row_count(db_path) == 1, "the store was written despite a refusal"


def test_a_client_that_cannot_elicit_is_refused_rather_than_written_for(short_tmp):
    """No back-channel means no sign-off, and no sign-off means no write.

    The SDK reports this as `MCPError: Elicitation not supported` -- named
    and immediate, not a hang. Failing closed is the only safe direction: a
    memory rewritten at tier 1 because nobody could be asked is exactly the
    confidently-wrong state the tier system exists to prevent.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    port = _free_port()

    daemon = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = None
    try:
        assert _line_within(daemon, 30.0) == "HELD"
        server = subprocess.Popen(
            [sys.executable, "-c", _MCP_SERVE, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        url = _line_within(server, 30.0)

        async def call_without_elicitation():
            # No `elicitation_callback`, so the client declares no such
            # capability -- which is what an older or simpler client is.
            async with streamable_http_client(url) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "palaver_correct",
                        {"memory_id": memory_id, "statement": "written without consent"},
                    )

        result = asyncio.run(asyncio.wait_for(call_without_elicitation(), timeout=60))
        message = result.content[0].text
    finally:
        for proc in (server, daemon):
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)

    assert result.is_error, "a client that cannot ask its user got a write anyway"
    # The SDK's own wording is "Elicitation not supported", which names
    # neither the correction nor whether it landed. The refusal has to say
    # both, or a caller is left guessing about the state of their store.
    assert "was not corrected" in message, message
    assert "nothing was written" in message, message
    assert _row_count(db_path) == 1


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


#: Takes the writer lock and nothing else -- no socket, no listener. The only
#: thing that can refuse a daemon against this is the `flock` itself.
_LOCK_ONLY = """
import fcntl, os, sys, time
from pathlib import Path
from palaver.observer.socket import lock_path_for

db_path = Path(sys.argv[1])
db_path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(lock_path_for(db_path), os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("LOCKED", flush=True)
time.sleep(float(sys.argv[2]))
"""


def test_the_lock_alone_refuses_a_second_daemon_when_no_socket_exists_yet(short_tmp):
    """The case the lock exists for, and the one the other tests never reach.

    `test_a_second_daemon_refuses_to_start_while_the_first_still_serves` was
    passing on the *connect probe*: the first daemon was already listening,
    so the intruder was turned away by a successful connect and the `flock`
    was never the reason. Deleting the `flock` outright left that test green
    -- which mutation testing found and no amount of reading would have.

    Here there is no socket at all, which is the state two daemons racing
    from cold start are both in. The lock is the only thing that can refuse,
    so if it is not taken, the second daemon binds and there are two writers.
    """
    db_path = short_tmp / "palaver.db"
    socket_path = writer_socket.socket_path_for(db_path)

    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_ONLY, str(db_path), "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line_within(holder, 30.0) == "LOCKED"
        assert not socket_path.exists(), "the fixture must leave no socket to probe"

        second = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(db_path), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode != 0, "a second daemon started with the lock already held"
        assert "DaemonAlreadyRunningError" in second.stdout, second.stdout + second.stderr
        assert not socket_path.exists(), "the refused daemon bound a socket anyway"
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_daemon_starts_once_that_lock_is_released(short_tmp):
    """The positive control: the refusal above is the lock, not the path.

    Without this, a `single_writer` that refused every start for an
    unrelated reason -- a permissions problem on the lock file, say --
    would satisfy the test above perfectly.
    """
    db_path = short_tmp / "palaver.db"
    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_ONLY, str(db_path), "0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert _line_within(holder, 30.0) == "LOCKED"
    holder.wait(timeout=30)

    started = subprocess.run(
        [sys.executable, "-c", _HOLDER, str(db_path), "0.1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert started.returncode == 0, started.stdout + started.stderr
    assert "HELD" in started.stdout


def test_correcting_a_memory_with_no_evidence_names_the_evidence_as_the_problem(short_tmp):
    """INV-6 is refused either way; what is under test is the diagnosis.

    Without the explicit check the write still fails -- `write_memory`
    raises on empty evidence -- but it reports "evidence must not be empty"
    about a memory the caller never mentioned, which sends a reader looking
    at the wrong row. The store is equally safe and the message is useless.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    # INV-6 is enforced in Python, not by the schema (TASKS.md Task 13's
    # sibling), so a row with no evidence is reachable on a raw connection --
    # which is exactly the inconsistent store this branch reports on.
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DELETE FROM memory_evidence WHERE memory_id = ?", (memory_id,))
        raw.commit()
    finally:
        raw.close()

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "no evidence to inherit"}
        )
    finally:
        conn.close()

    assert reply["ok"] is False
    assert "evidence to inherit" in reply["detail"], reply
    assert str(memory_id) in reply["detail"], "the message names no memory"
    assert _row_count(db_path) == 1
