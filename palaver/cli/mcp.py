"""`palaver mcp` — serve the read surface, or prove it serves six clients.

The selftest is the interesting half. Phase 6's acceptance is not "the
server starts"; it is that **six concurrent Streamable HTTP clients against
one `MCPServer` process all receive correct responses with no dropped
connection**. That number is not arbitrary — it is how many agents this
machine actually runs at once, and the SDK documents no tested client count,
so the only way to know is to open six and look.

Two things make the check mean something rather than merely run:

* **The clients are real.** They speak the actual Streamable HTTP transport
  over a real loopback socket through `mcp.client.streamable_http`, not an
  in-process shortcut. An in-process call would exercise the tool functions
  and none of the transport, which is the layer the concurrency question is
  about.
* **Every response is compared to the expected answer, not merely counted.**
  Six connections that all return an error are still six connections. The
  check asserts each client got the seeded session back.

The selftest seeds its own temporary database rather than reading the
machine's real one. A fresh install has no store yet, and a selftest that
passes only after `palaver observe` has run is a selftest that cannot answer
"is this working" on the machine that needs to ask.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import socket
import sqlite3
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from palaver.cli.observe import DEFAULT_DB_PATH
from palaver.mcp import server as mcp_server
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate

NAME = "mcp"
HELP = "serve Palaver's memory to other agents over MCP, or check that it serves"

#: How many concurrent clients the selftest opens by default. Phase 6's
#: acceptance number, and the number of agents this machine runs at once.
DEFAULT_CLIENTS = 6

#: The selftest's seeded fixtures. Named rather than inline so the assertion
#: compares against the same constants the seed wrote.
SELFTEST_PROJECT = "palaver-selftest"
SELFTEST_SESSION_ID = "0e5f1a1c-selftest-0000-000000000000"
SELFTEST_STATEMENT = "The selftest seeded exactly one memory, at tier 3."
SELFTEST_TIER = 3


def seed_selftest_db(db_path: Path) -> str:
    """Create a small database the selftest can assert exact answers against.

    Args:
        db_path: Where to create it.

    Returns:
        The `session_key` of the seeded session, in the same
        `<project>/<session-id>` form `palaver status` prints.
    """
    migrate(db_path)
    conn = connect(db_path)
    try:
        project_id = conn.execute(
            "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
            (SELFTEST_PROJECT, str(db_path.parent)),
        ).fetchone()[0]
        session_id = conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
            (project_id, "claude-code", SELFTEST_SESSION_ID),
        ).fetchone()[0]
        chunk_id = conn.execute(
            "INSERT INTO transcript_chunks (session_id, seq, role, content) "
            "VALUES (?, ?, ?, ?) RETURNING id",
            (session_id, 0, "assistant", SELFTEST_STATEMENT),
        ).fetchone()[0]
        write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=SELFTEST_STATEMENT,
            origin="selftest",
            tier=SELFTEST_TIER,
            evidence=[
                EvidenceAnchor(
                    start_offset=0,
                    end_offset=len(SELFTEST_STATEMENT),
                    transcript_chunk_id=chunk_id,
                )
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return f"{SELFTEST_PROJECT}/{SELFTEST_SESSION_ID}"


def _free_port() -> int:
    """Ask the OS for a port nothing is listening on.

    Bound and released rather than guessed: a hard-coded selftest port
    collides with the running daemon on the one machine where both are most
    likely to be up at once.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((mcp_server.DEFAULT_HOST, 0))
        return probe.getsockname()[1]


async def _one_client(url: str, session_key: str) -> dict:
    """Connect one real client, call both read tools, and return what it saw.

    Args:
        url: The server's Streamable HTTP endpoint.
        session_key: The seeded session to ask about.

    Returns:
        A dict with the decoded payload of each call, for the caller to
        compare against what was seeded.
    """
    from mcp import ClientSession  # noqa: PLC0415 - keeps import cost off the serve path
    from mcp.client.streamable_http import streamable_http_client  # noqa: PLC0415

    async with streamable_http_client(url) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.call_tool(
                "palaver_sessions", {"scope": {"project": SELFTEST_PROJECT}}
            )
            recalled = await session.call_tool(
                "palaver_recall", {"scope": {"session": session_key}}
            )
    return {
        "sessions": json.loads(listed.content[0].text),
        "recall": json.loads(recalled.content[0].text),
    }


#: Third-party loggers that emit one INFO line per HTTP request. Quieted
#: during the selftest because stderr is INV-1's progress channel, and a
#: reader looking for "which check failed" should not have to find it among
#: sixty transport lines. Palaver's own logging is untouched.
_NOISY_LOGGERS = ("mcp", "httpx", "httpx2", "httpcore", "sse_starlette")


def _quiet_dependency_logging() -> None:
    """Raise third-party log levels to WARNING for the duration of the process."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _build_server(app, host: str, port: int):
    """Configure a uvicorn server for the app without starting it.

    Split from serving so the selftest can hold the instance and ask it to
    exit. Cancelling `serve()` instead tears the ASGI lifespan down mid-await
    and uvicorn logs the resulting `CancelledError` at ERROR with a full
    traceback — a clean shutdown that looks exactly like a crash, in the
    output of the command whose whole job is to report whether things work.
    """
    import uvicorn  # noqa: PLC0415 - transitive through mcp; imported where used

    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
    return uvicorn.Server(config)


async def _announce_when_started(server, announce: Callable[[], None]) -> None:
    """Call `announce` once uvicorn is accepting requests.

    The endpoint URL is stdout's result, so it should mean "this is
    serving", not "this bound a port and intends to". The difference is not
    cosmetic: a supervisor or a test that treats the URL as the ready
    signal and acts on it immediately would otherwise race the startup it
    was waiting for.
    """
    while not server.started:
        await asyncio.sleep(0.01)
    announce()


async def _serve_forever(
    app,
    host: str,
    port: int,
    *,
    sock: socket.socket | None = None,
    stop: _StopRequest | None = None,
    announce: Callable[[], None] | None = None,
) -> None:
    """Run the ASGI app under uvicorn until it is asked to stop.

    Args:
        app: The Starlette application.
        host: Interface, used for logging when `sock` is supplied.
        port: Port, likewise.
        sock: An already-bound listening socket. When given, uvicorn serves
            it instead of binding its own — see `bind_listener`.
        stop: The record from `own_stop_signals`, handed the server so a
            signal arriving before uvicorn installs its own handler still
            stops it.
        announce: Called once the server is actually accepting requests.
    """
    server = _build_server(app, host, port)
    if stop is not None:
        stop.attach(server)
        if stop.requested:
            # Signalled before there was anything to signal. Returning here
            # rather than serving is the difference between a stop that is
            # honoured and one that is dropped on the floor.
            return
    if announce is not None:
        asyncio.get_running_loop().create_task(_announce_when_started(server, announce))
    await server.serve(sockets=None if sock is None else [sock])


def bind_listener(host: str, port: int) -> socket.socket:
    """Take the listening socket before anything is announced.

    `run()` writes the endpoint URL to stdout, which is its result contract,
    and letting uvicorn bind means that write happens *before* the bind is
    known to have succeeded. A second `palaver mcp` on the fixed default port
    would then print a URL it does not serve and exit, which is exactly the
    confidently-stale answer INV-7 exists to prevent — the reader has a URL
    in hand and no reason to distrust it.

    Binding here also closes the check-then-bind race a pre-flight probe
    would leave open: the port is held from this call onward, not merely
    observed to be free a moment ago.

    `SO_REUSEADDR` is set so a restart is not blocked by the previous
    process's sockets in `TIME_WAIT`. Measured on macOS 15: it does **not**
    permit a second live listener — a real collision still raises
    `EADDRINUSE` — so it buys the restart without weakening the check.

    The INV-9 loopback check is repeated here rather than left to `run()`.
    `run()` refuses first so the CLI answers with an exit code instead of a
    traceback, but this is the single place a listening socket is created,
    and a caller reaching it directly must not be able to open the store to
    the network by skipping the front door.

    Args:
        host: Interface to bind. IPv4 loopback only — see `ensure_loopback`.
        port: Port to bind.

    Returns:
        A bound, listening socket for uvicorn to serve.

    Raises:
        NonLoopbackHost: `host` is not an IPv4 loopback literal.
        OSError: The address is in use or otherwise unbindable.
    """
    mcp_server.ensure_loopback(host)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(socket.SOMAXCONN)
    except OSError:
        sock.close()
        raise
    return sock


class _StopRequest:
    """A stop signal that arrived before uvicorn was ready to receive it.

    uvicorn installs its own signal handlers, but only once `serve()` is
    running. Between `own_stop_signals()` and that moment there is a window
    — short, but it covers the whole of ASGI startup — where a SIGINT or
    SIGTERM would land on a handler that does nothing, and the server would
    then run forever having been told twice to stop. This holds the request
    across that window and applies it as soon as there is a server to apply
    it to.
    """

    def __init__(self) -> None:
        self.requested = False
        self._server = None

    def attach(self, server) -> None:
        """Bind the server, so a later signal has somewhere to go.

        Deliberately does *not* apply an already-recorded request: mutation
        testing showed that branch survives every test, because
        `_serve_forever` checks `requested` immediately afterwards and
        returns without serving. Two mechanisms for one case means one of
        them is never the reason the stop worked, so the redundant one is
        gone rather than covered by a test that would only prove it is
        redundant.
        """
        self._server = server

    def record(self, signum: int, frame) -> None:
        """Signal handler: remember the request, and forward it if possible."""
        self.requested = True
        if self._server is not None:
            self._server.should_exit = True


def own_stop_signals() -> _StopRequest:
    """Claim SIGINT and SIGTERM before `asyncio.run` can.

    Ctrl-C is how a foreground server is stopped, so it is the normal exit
    path, and it printed a 25-line `KeyboardInterrupt` traceback. The cause
    is a three-way interaction worth writing down, because none of the three
    parties is doing anything wrong:

    1. `asyncio.run` installs its own SIGINT handler, but only if the current
       handler is still `signal.default_int_handler`. That handler cancels
       the main task and re-raises `KeyboardInterrupt`.
    2. uvicorn's `serve()` then installs *its* handler, which sets
       `should_exit` and shuts down gracefully.
    3. `sse_starlette` — transitive through `mcp` — replaces
       `uvicorn.Server.handle_exit` with one that chains to whatever handler
       was installed when the app started. Under `asyncio.run` that chain
       lands on asyncio's task-cancelling handler, so a single Ctrl-C both
       shuts uvicorn down cleanly *and* cancels the task out from under it.

    Installing a handler here defeats step 1's check, so asyncio installs
    nothing and step 3 chains to this no-op instead. Measured: the graceful
    shutdown still runs, and the process returns from `asyncio.run` normally
    rather than through an exception. SIGTERM gets the same treatment,
    because that is what launchd sends.

    Returns:
        The `_StopRequest` this installed, to be handed to `_serve_forever`
        so a signal arriving before uvicorn takes the slot is not lost.
    """
    stop = _StopRequest()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError, OSError):
            signal.signal(sig, stop.record)
    return stop


async def _await_listening(host: str, port: int, *, attempts: int = 100) -> bool:
    """Poll until the server accepts a TCP connection, or give up.

    A fixed sleep would either be too short on a loaded machine or waste
    time on an idle one; this returns as soon as the socket answers.
    """
    for _ in range(attempts):
        try:
            reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True
    return False


async def _run_selftest(
    db_path: Path,
    *,
    clients: int,
    host: str,
    out: TextIO,
    on_status: Callable[[str], None],
) -> int:
    """Serve the app on an ephemeral port and drive `clients` real clients."""
    _quiet_dependency_logging()
    session_key = seed_selftest_db(db_path)
    on_status(f"seeded {session_key}")

    _, app = mcp_server.build_app(db_path, host=host)
    port = _free_port()
    url = f"http://{host}:{port}{mcp_server.DEFAULT_PATH}"

    server = _build_server(app, host, port)
    serving = asyncio.create_task(server.serve())
    failures = 0
    try:
        if not await _await_listening(host, port):
            out.write(f"fail     listen: nothing accepted a connection on {url}\n")
            return 1
        out.write(f"ok       listen: serving Streamable HTTP at {url}\n")

        on_status(f"opening {clients} concurrent client(s)")
        results = await asyncio.gather(
            *(_one_client(url, session_key) for _ in range(clients)),
            return_exceptions=True,
        )

        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                out.write(f"fail     client {index}: {type(result).__name__}: {result}\n")
                failures += 1
                continue
            keys = [entry["session_key"] for entry in result["sessions"]["sessions"]]
            memories = result["recall"]["memories"]
            # Compared against what was seeded, not merely counted: six
            # clients that all returned an error are still six clients.
            if keys != [session_key]:
                out.write(f"fail     client {index}: listed {keys}, expected [{session_key!r}]\n")
                failures += 1
            elif [m["statement"] for m in memories] != [SELFTEST_STATEMENT]:
                out.write(
                    f"fail     client {index}: recalled {len(memories)} unexpected memory(ies)\n"
                )
                failures += 1
            elif [m["tier"] for m in memories] != [SELFTEST_TIER]:
                out.write(f"fail     client {index}: tier {[m['tier'] for m in memories]}\n")
                failures += 1
        if not failures:
            out.write(
                f"ok       concurrency: {clients} concurrent client(s) each got the seeded answer\n"
            )
    finally:
        # Asked to exit, not cancelled — see `_build_server`.
        server.should_exit = True
        with contextlib.suppress(asyncio.CancelledError):
            await serving

    out.write(f"{clients - failures} ok, {failures} failed\n")
    return 1 if failures else 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register `mcp`'s flags on its subparser."""
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="seed a temporary store, serve it, and drive concurrent real clients against it",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=DEFAULT_CLIENTS,
        help=f"how many concurrent clients the selftest opens (default: {DEFAULT_CLIENTS})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"database to serve (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--host",
        default=mcp_server.DEFAULT_HOST,
        help=f"interface to bind (default: {mcp_server.DEFAULT_HOST}; loopback only, per INV-9)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=mcp_server.DEFAULT_PORT,
        help=f"port to bind (default: {mcp_server.DEFAULT_PORT})",
    )


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver mcp`.

    Args:
        args: Parsed arguments.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel (INV-1), defaulting to stderr.

    Returns:
        0 when the server exits cleanly or the selftest passes, 1 when the
        selftest fails or the database is missing, 2 for a bad invocation.
    """
    out = sys.stdout if out is None else out
    if on_status is None:

        def on_status(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

    if args.clients < 1:
        out.write("palaver mcp: --clients must be at least 1\n")
        return 2

    # Ahead of the selftest branch on purpose. The selftest reaches uvicorn
    # through `_build_server` rather than `bind_listener`, so a check placed
    # only in the binder would leave `--selftest --host 0.0.0.0` open.
    try:
        mcp_server.ensure_loopback(args.host)
    except mcp_server.NonLoopbackHost as exc:
        out.write(f"palaver mcp: {exc}\n")
        return 2

    if args.selftest:
        with tempfile.TemporaryDirectory(prefix="palaver-mcp-selftest-") as tmp:
            return asyncio.run(
                _run_selftest(
                    Path(tmp) / "selftest.db",
                    clients=args.clients,
                    host=args.host,
                    out=out,
                    on_status=on_status,
                )
            )

    db_path = DEFAULT_DB_PATH if args.db is None else args.db
    try:
        mcp_server.open_readonly(db_path).close()
    except (FileNotFoundError, sqlite3.Error) as exc:
        out.write(f"palaver mcp: {exc}\n")
        return 1

    try:
        sock = bind_listener(args.host, args.port)
    except OSError as exc:
        out.write(
            f"palaver mcp: cannot bind {args.host}:{args.port}: {exc.strerror or exc}. "
            "Another Palaver MCP server is probably already serving it.\n"
        )
        return 1

    _, app = mcp_server.build_app(db_path, host=args.host)
    url = f"http://{args.host}:{args.port}{mcp_server.DEFAULT_PATH}"
    on_status(f"serving {db_path} at {url}")

    def _announce() -> None:
        # stdout is the result, and it goes out only once the server is
        # accepting requests — a URL printed by a process that is not yet
        # (or no longer) serving is worse than no URL at all.
        out.write(f"{url}\n")
        out.flush()

    stop = own_stop_signals()
    try:
        asyncio.run(
            _serve_forever(app, args.host, args.port, sock=sock, stop=stop, announce=_announce)
        )
    finally:
        sock.close()
    on_status("stopped")
    return 0
