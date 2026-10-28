"""Serving requests off the socket until told to stop."""

from __future__ import annotations

import json
import select
import socket as socket_module
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from .core import DEFAULT_TIMEOUT, log
from .protocol import apply_request

#: Largest request body the daemon will buffer. A correction is a sentence;
#: a client streaming past this without a newline is broken or hostile.
MAX_REQUEST_BYTES = 1024 * 1024

#: Wall-clock budget for receiving one whole request. `settimeout` alone
#: resets on every `recv`, so a client trickling one byte per interval would
#: hold the single tick-loop thread forever (INV-1: a silent stall).
REQUEST_DEADLINE_SECONDS = DEFAULT_TIMEOUT


class _RequestTooLargeError(Exception):
    """The request body outgrew `MAX_REQUEST_BYTES` before its newline arrived."""


def serve_request(server: socket_module.socket, conn: sqlite3.Connection) -> bool:
    """Accept one connection, apply its request, and reply.

    Args:
        server: The listening socket from `single_writer`.
        conn: The daemon's writable connection.

    Returns:
        True if a request was served. False if the connection carried
        nothing — which is what the startup liveness probe leaves behind,
        and what `daemon_running` does on every read. Those must not be
        logged as malformed requests; they are the mechanism working.

        A request that was applied but whose reply could not be delivered
        still counts as served. A query event is posted fire-and-forget
        (see `palaver.mcp.query_events`) and its sender is usually gone
        before the acknowledgement is written; returning False there would
        make the daemon's own count of what it did disagree with what is in
        the database.
    """
    try:
        client, _ = server.accept()
    except OSError as exc:
        # EMFILE, ECONNABORTED and the like: the pending connection is lost,
        # not the daemon.
        log.warning("accept failed: %s", exc)
        return False
    try:
        client.settimeout(DEFAULT_TIMEOUT)
        deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
        chunks = []
        received = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request not completed within its deadline")
            client.settimeout(min(DEFAULT_TIMEOUT, remaining))
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            received += len(chunk)
            if chunks[-1].endswith(b"\n"):
                break
            if received > MAX_REQUEST_BYTES:
                raise _RequestTooLargeError(
                    f"request exceeds {MAX_REQUEST_BYTES} bytes without a newline"
                )
        body = b"".join(chunks).strip()
        if not body:
            return False

        payload: Any = None
        if len(body) > MAX_REQUEST_BYTES:
            reply: dict = {
                "ok": False,
                "error": "RequestTooLarge",
                "detail": f"request exceeds {MAX_REQUEST_BYTES} bytes",
            }
        else:
            try:
                payload = json.loads(body)
            except (ValueError, RecursionError) as exc:
                # ValueError covers JSONDecodeError, UnicodeDecodeError and the
                # int-digit limit; RecursionError is a deeply nested body.
                reply = {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:200]}
            else:
                reply = apply_request(conn, payload)

        try:
            client.sendall(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
        except (ConnectionError, BrokenPipeError) as exc:
            # Narrower than the handler below, and deliberately after the
            # request has been applied and committed: the work is done and
            # durable, and only the receipt had nowhere to go.
            log.debug("reply to %r went unread: %s", payload, exc)
        return True
    except _RequestTooLargeError as exc:
        log.warning("write request abandoned: %s", exc)
        try:
            client.sendall(
                json.dumps({"ok": False, "error": "RequestTooLarge", "detail": str(exc)}).encode()
                + b"\n"
            )
        except OSError:
            pass
        return False
    except (TimeoutError, ConnectionError, BrokenPipeError) as exc:
        # One client that hangs up or stalls is not the daemon's failure.
        log.warning("write request abandoned: %s", exc)
        return False
    finally:
        client.close()


def serve_until(
    server: socket_module.socket | None,
    conn: sqlite3.Connection,
    seconds: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Answer write requests for `seconds`, then return.

    This is what the daemon does *instead of* sleeping between ticks. The
    alternative — a thread accepting requests alongside the tick loop —
    would put two threads on one SQLite connection, which is the same
    two-writer problem this module exists to prevent, moved inside the
    process where no lock would catch it. Serving in the idle window keeps
    every write on the tick loop's own thread by construction.

    The cost is that a request arriving mid-tick waits for the tick to
    finish. Ticks are seconds and corrections are a human typing, so that is
    the right trade; a correction that waits is fine, a correction racing an
    extraction on one connection is not.

    Args:
        server: The listening socket from `single_writer`.
        conn: The daemon's writable connection.
        seconds: How long to keep serving. The remaining time is recomputed
            after every request, so a busy window still ends on schedule
            rather than extending by one timeout per request.
        monotonic: Injected for tests. Monotonic rather than wall clock: a
            clock adjustment must not strand the daemon in an idle window
            for hours, or skip the window entirely.

    Returns:
        How many requests were served. Zero without a socket, which is not
        a failure — the daemon still ticks, it just has nothing to answer.
    """
    if server is None:
        # No request channel (see `single_writer`). Spend the window the way
        # a daemon with no socket at all would: asleep. Doing anything else
        # here would make the degraded configuration tick at a different
        # rate from the normal one, for no reason a reader could see.
        sleep(seconds)
        return 0

    deadline = monotonic() + seconds
    served = 0
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return served
        # `select` rather than a socket timeout, so the wait ends the moment
        # a request arrives instead of on the next timeout boundary.
        ready, _, _ = select.select([server], [], [], remaining)
        if not ready:
            return served
        if serve_request(server, conn):
            served += 1
