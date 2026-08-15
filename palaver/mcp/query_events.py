"""Recording what an agent actually retrieved, without becoming a writer.

The MCP server holds a `mode=ro` connection by design (see
`palaver.mcp.server`), and task 6.3 established that this machine has
exactly one writer. Both hold here: a query event is *posted* to the
`palaver observe` daemon over the socket 6.3 already listens on, and the
daemon writes it on its own connection. Nothing in this module opens a
database.

**Fire-and-forget, and that asymmetry with `palaver_correct` is deliberate.**
A correction that silently vanished would leave a human believing they had
fixed something they had not, so `palaver_correct` fails loudly when the
daemon is down. A query event is Palaver's own telemetry about itself. If
recording one fails, the honest cost is a gap in that telemetry — and the
alternative, failing the read, means an agent asking a question gets an
error because a *bookkeeping* channel was unavailable. That trade is
obviously wrong in one direction, so the post logs a WARNING and returns.

Consequently `post` never raises, and nothing downstream of it treats its
return value as required. The WARNING is the whole error channel; it exists
so a permanently-dead recording path is visible in the log rather than
inferred months later from an empty table.

The recorded volume is small enough not to need pruning: one row plus one
row per returned memory, per read call. A heavy day of agent use is on the
order of thousands of rows, against a store already holding transcript
chunks.
"""

from __future__ import annotations

import json
import logging
import socket as socket_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from palaver.observer.socket import SocketPathTooLongError, socket_path_for

log = logging.getLogger(__name__)

#: Seconds to allow for the whole post. Short, and short for a reason that
#: is not the daemon's speed: none of `connect`/`sendall`/`shutdown` waits
#: for the daemon to *accept*, so a healthy post costs microseconds no
#: matter how busy a tick is. The only way to spend this budget is a full
#: listen backlog, and in that case the right answer is to drop the event
#: rather than to hold up the read tool that is waiting on us.
POST_TIMEOUT = 1.0

#: The operation name the daemon dispatches on. Must match a member of
#: `palaver.observer.socket.WRITE_OPERATIONS`; the daemon refuses anything
#: else, which is what keeps the write protocol a closed set.
QUERY_OPERATION = "query"

#: Where each read tool puts its page, in the order they are looked for.
#:
#: A tool whose key is missing from here records `result_count: None` — "not
#: counted" — rather than zero. `tests/test_mcp.py` asserts every tool in
#: `READ_TOOLS` is covered, so the null is a bug report from a future tool,
#: not a state this suite tolerates.
ITEM_KEYS = ("memories", "sessions")


def describe(tool: str, result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Turn a read tool's answer into the query event that describes it.

    Args:
        tool: The tool name the client called.
        result: What the tool returned, before serialization.

    Returns:
        The request payload to post, or `None` if `result` carries no
        resolved scope. The scope is taken from the tool's *echo* rather
        than from the caller's argument because the echo is what the tool
        resolved and answered — a caller's `{"project": " demo "}` and the
        answer it received differ, and the useful record is the answer's.
    """
    scope = result.get("scope")
    if not isinstance(scope, Mapping) or len(scope) != 1:
        # Not raised: this is a read that already succeeded, and the reply is
        # on its way to the client. Refusing to record it cannot un-answer it.
        log.warning("query event for %s dropped: no single-key scope in its result", tool)
        return None

    ((scope_kind, scope_value),) = scope.items()

    items: list[Any] | None = None
    for key in ITEM_KEYS:
        candidate = result.get(key)
        if isinstance(candidate, list):
            items = candidate
            break

    return {
        "op": QUERY_OPERATION,
        "tool": tool,
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        # None, not 0, when no known page key was found. See `ITEM_KEYS`.
        "result_count": None if items is None else len(items),
        # Only memories are linked. `palaver_sessions` returns sessions, and
        # recording those ids here would put a session id in a column that
        # every reader will take for a memory id.
        "memory_ids": [
            item["id"]
            for item in (result.get("memories") or [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), int)
        ],
    }


def post(db_path: Path, payload: Mapping[str, Any], *, timeout: float = POST_TIMEOUT) -> bool:
    """Send one query event to the daemon and do not wait for an answer.

    The connection is shut down for writing and closed without a read. The
    daemon may not accept it for most of a tick interval; the bytes sit in
    its receive queue until it does, which is why closing early loses
    nothing.

    Args:
        db_path: The store whose daemon to reach.
        payload: The request body, from `describe`.
        timeout: Seconds to allow. See `POST_TIMEOUT`.

    Returns:
        True if the event was handed to the kernel, False if it was dropped.
        A caller is free to ignore this; every drop is already logged.

    Raises:
        Nothing. A read tool calls this after it has computed its answer,
        so an exception escaping here would turn a successful read into a
        failed one — precisely the outcome fire-and-forget exists to avoid.
    """
    try:
        socket_path = socket_path_for(db_path)
    except SocketPathTooLongError as exc:
        log.warning("query event dropped: %s", exc)
        return False

    try:
        # Constructed inside the `try`, not before it: a process out of file
        # descriptors fails here rather than at `connect`, and that failure
        # has to be a dropped event like any other. Outside, it would be an
        # exception escaping a function documented not to raise.
        with socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(socket_path))
            client.sendall(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            client.shutdown(socket_module.SHUT_WR)
    except OSError as exc:
        # OSError covers the lot: FileNotFoundError and ConnectionRefusedError
        # for a stopped daemon, TimeoutError for a full backlog, EPIPE for one
        # that died mid-post. They differ in cause and not in what we do about
        # them, so they are logged with the cause attached rather than sorted
        # into branches that all end the same way.
        log.warning(
            "query event for %s dropped: %s (%s). The read itself was unaffected.",
            payload.get("tool"),
            exc,
            type(exc).__name__,
        )
        return False
    return True


def record(db_path: Path, tool: str, result: Mapping[str, Any]) -> bool:
    """Describe a read tool's answer and post it. The one call a tool makes.

    Args:
        db_path: The store whose daemon to reach.
        tool: The tool name the client called.
        result: What the tool returned.

    Returns:
        True if the event reached the daemon's socket.
    """
    payload = describe(tool, result)
    if payload is None:
        return False
    return post(db_path, payload)
