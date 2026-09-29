"""
The client side: sending one write request to the daemon's socket and reading its reply.
"""

from __future__ import annotations

import json
import socket as socket_module
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import paths
from .core import DEFAULT_TIMEOUT, DaemonUnavailableError


def request(db_path: Path, payload: Mapping[str, Any], *, timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Send one write request to the daemon and read its reply.

    One request per connection, closed after. A pool would be faster and
    would also make a half-written request from a crashed client the next
    client's problem; write volume here is a human correcting a memory, so
    the simple framing is the right trade.

    Args:
        db_path: The database whose daemon to reach.
        payload: The request body, JSON-serializable.
        timeout: Seconds to allow for connect, send, and reply.

    Returns:
        The daemon's decoded reply.

    Raises:
        DaemonUnavailableError: No daemon is listening, or it closed without
            replying. Raised rather than falling back to a direct write:
            opening a second writer is the one thing this whole module
            exists to prevent, and a caller that silently got one would have
            no way to know.
    """
    socket_path = paths.socket_path_for(db_path)
    client = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        try:
            client.connect(str(socket_path))
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise DaemonUnavailableError(
                f"no palaver observe daemon is listening on {socket_path}, so this "
                "write cannot be made. Palaver has exactly one writer by design and "
                "will not open a second one to get around a stopped daemon. Start it "
                "with `palaver observe` and try again."
            ) from exc

        client.sendall(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        client.shutdown(socket_module.SHUT_WR)

        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    except TimeoutError as exc:
        raise DaemonUnavailableError(
            f"the daemon on {socket_path} accepted the connection but did not reply "
            f"within {timeout}s. The write may or may not have been applied; check "
            "with `palaver inspect` rather than retrying blindly."
        ) from exc
    finally:
        client.close()

    body = b"".join(chunks).strip()
    if not body:
        raise DaemonUnavailableError(
            f"the daemon on {socket_path} closed the connection without replying. "
            "The write was not acknowledged and must not be assumed to have landed."
        )
    return json.loads(body)
