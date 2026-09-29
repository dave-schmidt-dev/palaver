"""Task 6.1: the MCP read surface, and the two ways it could quietly lie.

The tests here are organised around failures that are invisible from the
caller's side, because those are the ones an API-shaped test misses:

* **A defaulted scope.** A tool that answers a project-wide question when a
  session was meant returns something that reads exactly as authoritative as
  the right answer. Nothing downstream can tell. So the refusal is asserted
  at both layers — the parser, and a real tool call through the server.
* **A silently-resolved identifier.** `read_memories(session=...)` takes a
  rowid; an MCP caller holds a `session_key`. If the tool accepted either,
  an integer-looking key would resolve against an unrelated row and answer
  confidently. The rowid path is asserted *refused*, not merely unused.

The transport tests exist because the concurrency question Phase 6 is
accepted on is a transport question, and a test that called the tool
functions in-process would prove none of it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from palaver.mcp import server as mcp_server
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate

# =============================================================================
# Fixtures
# =============================================================================


def _seed(db_path: Path, *, sources=("claude-code",), external_id="session-aaa") -> dict:
    """Build a small database and return the ids a test needs to assert on.

    `sources` is a tuple so a test can create the same `external_id` under
    two sources — the collision `sessions`' own `UNIQUE (source, external_id)`
    permits and a key-only lookup therefore cannot resolve.
    """
    migrate(db_path)
    conn = connect(db_path)
    project_id = conn.execute(
        "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
        ("demo", str(db_path.parent)),
    ).fetchone()[0]

    session_ids = []
    for source in sources:
        session_id = conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
            (project_id, source, external_id),
        ).fetchone()[0]
        session_ids.append(session_id)

    chunk_id = conn.execute(
        "INSERT INTO transcript_chunks (session_id, seq, role, content) VALUES (?, ?, ?, ?) "
        "RETURNING id",
        (session_ids[0], 0, "assistant", "the evidence text"),
    ).fetchone()[0]
    memory_id = write_memory(
        conn,
        project_id=project_id,
        session_id=session_ids[0],
        statement="the first session decided to use SQLite",
        origin="observer",
        tier=4,
        evidence=[EvidenceAnchor(start_offset=0, end_offset=8, transcript_chunk_id=chunk_id)],
    )
    conn.commit()
    conn.close()
    return {
        "project_id": project_id,
        "session_ids": session_ids,
        "memory_id": memory_id,
        "external_id": external_id,
        "session_key": f"demo/{external_id}",
    }


@pytest.fixture
def short_store():
    """A seeded store on a path short enough for the daemon socket.

    pytest's `tmp_path` is ~90 bytes before a test name is appended, and
    `sun_path` holds 103 -- so any test that must reach the *real* socket
    path needs a shorter home than the default fixture provides.
    """
    directory = Path(tempfile.mkdtemp(prefix="plv", dir="/tmp"))  # noqa: S108
    try:
        db_path = directory / "palaver.db"
        yield db_path, _seed(db_path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def store(tmp_path):
    """A migrated database with one project, one session, and one memory."""
    db_path = tmp_path / "store.db"
    seeded = _seed(db_path)
    return db_path, seeded


def _conn(db_path: Path) -> sqlite3.Connection:
    return mcp_server.open_readonly(db_path)


def _call(server, name, arguments):
    """Call a tool the way a client does, through the server's dispatcher."""
    return asyncio.run(server.call_tool(name, arguments))


# =============================================================================
# Serving for real, which the selftest does not exercise
#
# `--selftest` picks a free ephemeral port, drives clients, and tears down
# inside one `asyncio.run`. The invocation a person actually types after
# `claude mcp add` does none of those things: it binds the *fixed* default
# port and runs until a signal stops it. Both differences produced a defect
# that every test in the sections above passed straight over.
# =============================================================================


async def _with_timeout(coro, seconds: float = 10.0):
    """Fail rather than hang; a serve call that never returns is the defect."""
    async with asyncio.timeout(seconds):
        return await coro


#: Runs `palaver mcp` the way a person does, in a process of its own, so a
#: real signal can be delivered to it. In-process this is untestable: the
#: defect was `asyncio.run` raising `KeyboardInterrupt` out of the serve
#: call, and pytest's own process cannot be interrupted to find out.
_SERVE_SUBPROCESS = """
import sys
from palaver.cli import main
db, port = sys.argv[1], sys.argv[2]
sys.argv = ["palaver", "mcp", "--db", db, "--port", port]
sys.exit(main())
"""


def _endpoint_line(proc, deadline: float = 30.0) -> str:
    """Read the endpoint from stdout, or fail — never wait forever.

    A bare `readline()` on a live child has no deadline, so a server that
    binds nothing and prints nothing turns a *failing* test into a *hanging*
    one. That is not hypothetical: a mutation battery over `bind_listener`
    lost its entire budget to the one mutant that hangs, and the run was
    killed mid-flight with a mutant still applied to the tree. A test that
    can hang is a test that cannot be run in bulk.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        line = pool.submit(proc.stdout.readline).result(timeout=deadline)
    except concurrent.futures.TimeoutError:
        proc.kill()  # Unblocks the reader thread by closing the pipe.
        raise AssertionError(f"no endpoint on stdout within {deadline}s") from None
    finally:
        pool.shutdown(wait=False)
    return line.strip()
