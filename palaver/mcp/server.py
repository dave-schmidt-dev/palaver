"""The `MCPServer` instance, its transport, and how a tool reaches SQLite.

**Every tool call opens its own read-only connection and closes it.** The
2026-07-28 protocol revision removed session state, so there is no
per-connection place to keep a handle even if holding one were wise — and it
is not: one long-lived connection shared across six concurrent clients
serialises them behind a single cursor, and a connection that outlives a
`palaver observe` restart reads a database file that has since been replaced
underneath it.

**Read-only at the SQLite layer, not merely by convention.** The tools here
only read, but "only reads" is a property of the code as written, which the
next task can change without noticing. Opening `mode=ro` makes the database
itself refuse a write, so the read surface stays a read surface even if a
future tool forgets. Task 6.3's `palaver_correct` writes through the
daemon's single-writer socket rather than through this connection, which is
exactly why this one can stay closed to writes.

A missing database is an error rather than an empty answer: `palaver status`
with no store behind it should say so, not report that this machine has no
memories.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context

from palaver import __version__
from palaver.mcp import tools_read, tools_write
from palaver.observer.socket import daemon_running

log = logging.getLogger(__name__)

#: The server name clients see, and the name `claude mcp add` registers.
SERVER_NAME = "palaver"

#: Bound to loopback and nowhere else. INV-9 permits exactly one local MCP
#: listener; a server reachable off-machine would make the aggregated store
#: of every observed session remotely queryable.
DEFAULT_HOST = "127.0.0.1"

#: The default port. Fixed rather than ephemeral because clients register a
#: URL once in `~/.claude.json` or `~/.codex/config.toml` and a port that
#: moved on every restart would break every registration.
DEFAULT_PORT = 8787

#: The HTTP path the Streamable HTTP transport is mounted at.
DEFAULT_PATH = "/mcp"

INSTRUCTIONS = (
    "Palaver observes coding-agent sessions on this machine and keeps "
    "evidence-backed memory about them. Every read tool requires an explicit "
    "scope of exactly one of {project: <name>} or {session: <session_key>}; "
    "scope is never defaulted, because a project-wide answer returned where a "
    "session was meant is indistinguishable from the right one. Session keys "
    "look like <project>/<session-id> and can be listed with palaver_sessions."
)


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open Palaver's database read-only.

    Args:
        db_path: The database file.

    Returns:
        A connection that will refuse writes.

    Raises:
        FileNotFoundError: `db_path` does not exist. Reported by name rather
            than as an empty result set — "no database" and "a database with
            nothing in it" are different answers to every tool here.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"no Palaver database at {db_path}. Run `palaver observe` first, or pass --db."
        )
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def build_server(
    db_path: Path,
    *,
    name: str = SERVER_NAME,
    connect: Callable[[Path], sqlite3.Connection] = open_readonly,
) -> MCPServer:
    """Build the server with every read tool registered.

    Args:
        db_path: The database each tool call reads.
        name: The server name clients see.
        connect: Connection factory, injected so a test can hand in an
            already-populated in-memory database.

    Returns:
        The configured `MCPServer`. The Streamable HTTP app is not built
        here — `streamable_http_app()` is what creates the session manager,
        and building it eagerly would tie server construction to a transport
        a caller might not want yet.
    """
    # The version travels in the initialize handshake and is what a client
    # reports back. Left unset it is the empty string, so a user comparing
    # what two machines are running has nothing to compare.
    server = MCPServer(name=name, version=__version__, instructions=INSTRUCTIONS)

    def _register(tool_name: str, handler: Callable[..., dict]) -> None:
        # Bound in a closure per tool rather than in the loop body directly:
        # a late-binding `handler` would register every tool against the last
        # one in the mapping, and every tool would answer identically.
        #
        # `cursor` defaults to None rather than being required, so the first
        # call of a sequence looks exactly like an unpaginated one. `scope`
        # has no default and never will — see `tools_read`.
        def _call(scope: dict[str, str], cursor: str | None = None) -> dict[str, Any]:
            conn = connect(db_path)
            try:
                # Passed *into* the tool rather than merged onto its
                # result: `paginate` asserts the assembled response fits the
                # byte budget, and keys added after that assertion are not
                # the ones it measured. Freshness is stamped here rather
                # than inside each tool because it is a property of the
                # store this server points at, not of the question asked —
                # so a tool added later cannot answer without saying how
                # current its answer is.
                return handler(conn, scope, cursor, freshness(conn, db_path))
            finally:
                conn.close()

        _call.__name__ = tool_name
        _call.__doc__ = handler.__doc__
        server.add_tool(_call, name=tool_name, structured_output=False)

    for tool_name, handler in tools_read.READ_TOOLS.items():
        _register(tool_name, handler)

    def _register_write(tool_name: str, handler: Callable[..., Any]) -> None:
        # Async, and given the context, because a write is gated on an
        # elicitation round trip to the client — see `tools_write`.
        async def _call(ctx: Context, memory_id: int, statement: str) -> dict[str, Any]:
            conn = connect(db_path)
            try:
                return await handler(conn, db_path, ctx, memory_id, statement)
            finally:
                conn.close()

        _call.__name__ = tool_name
        _call.__doc__ = handler.__doc__
        server.add_tool(_call, name=tool_name, structured_output=False)

    for tool_name, handler in tools_write.WRITE_TOOLS.items():
        _register_write(tool_name, handler)

    return server


def freshness(conn: sqlite3.Connection, db_path: Path) -> dict:
    """How current this store is, and whether anything is still maintaining it.

    Every read tool carries both. A crashed daemon and a quiet one return
    byte-identical results otherwise — same memories, same timestamps — so a
    reader has no way to tell "nothing has happened" from "nothing has been
    watching for two days". Taking the first for the second is INV-7's
    failure exactly, and it is the likelier reading, because a store that
    answers at all looks healthy.

    Args:
        conn: The tool's read-only connection.
        db_path: The store, which is where the daemon's socket is looked for.

    Returns:
        `observed_at`, the newest memory's timestamp — the honest upper
        bound on what this store has seen, and `None` for a store with no
        memories yet — and `daemon_running`, a live connect probe rather
        than a cached flag, because a flag written at startup says only that
        the daemon once existed. `daemon_running` is `null`, not `false`,
        when the store has no probeable socket: a daemon can run without one
        (see `palaver.observer.socket.daemon_running`), and reporting it as
        stopped would be the confident wrong answer INV-7 is about.
    """
    row = conn.execute("SELECT max(created_at) FROM memories").fetchone()
    return {"observed_at": row[0] if row else None, "daemon_running": daemon_running(db_path)}


def build_app(
    db_path: Path,
    *,
    host: str = DEFAULT_HOST,
    path: str = DEFAULT_PATH,
    connect: Callable[[Path], sqlite3.Connection] = open_readonly,
) -> tuple[MCPServer, Any]:
    """Build the server and its Streamable HTTP ASGI app together.

    Returns both because they are not independently useful: the session
    manager only exists once the app has been built, so a caller handed only
    the server cannot serve it, and a caller handed only the app cannot
    inspect what it registered.

    Args:
        db_path: The database each tool call reads.
        host: Interface to advertise; loopback by default.
        path: HTTP path to mount the transport at.
        connect: Connection factory, injected for tests.

    Returns:
        A `(server, app)` pair, where `app` is a Starlette application.
    """
    server = build_server(db_path, connect=connect)
    app = server.streamable_http_app(streamable_http_path=path, host=host)
    return server, app
