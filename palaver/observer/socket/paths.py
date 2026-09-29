"""
Where the lock files and request socket for a store live, and the handshake that finds a
live socket without a stale one.
"""

from __future__ import annotations

import errno
import os
import socket as socket_module
from pathlib import Path

from .core import MAX_SOCKET_PATH_BYTES, SocketPathTooLongError


def lock_path_for(db_path: Path) -> Path:
    """The lock file that guards `db_path`'s writer role."""
    return db_path.parent / "palaver.lock"


def liveness_lock_path_for(db_path: Path) -> Path:
    """The queryable record-lock marker paired with the writer lock."""
    return db_path.parent / "palaver.liveness.lock"


def socket_path_for(db_path: Path) -> Path:
    """The socket the single writer accepts write requests on.

    Args:
        db_path: The database the socket sits beside.

    Returns:
        The socket path.

    Raises:
        SocketPathTooLongError: The path will not fit in `sun_path`. Checked
            here rather than at `bind`, because the kernel's `OSError:
            AF_UNIX path too long` names neither the limit, the length, nor
            which of the several paths in play was the problem.
    """
    path = db_path.parent / "palaver.sock"
    encoded = len(os.fsencode(path))
    if encoded > MAX_SOCKET_PATH_BYTES:
        raise SocketPathTooLongError(
            f"the write socket would be at {path}, which is {encoded} bytes — over the "
            f"{MAX_SOCKET_PATH_BYTES}-byte limit this platform's `sun_path` imposes. "
            "The socket has to sit beside the database, so point --db at a shorter "
            "path."
        )
    return path


def _probe(path: Path, timeout: float) -> bool:
    """Is somebody listening on this socket path right now?

    Args:
        path: The socket node to try.
        timeout: Seconds to wait for the connect.

    Returns:
        True if a connect succeeded, meaning a live listener owns the path.
        False if the node is absent or refuses — the two states that make it
        safe to unlink.

    Raises:
        OSError: Any other failure. A `EACCES` or a timeout is *not* evidence
            of staleness, and treating it as such would unlink a path that
            might still be served. Unknown means stop.
    """
    probe = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    probe.settimeout(timeout)
    try:
        probe.connect(str(path))
    except FileNotFoundError, ConnectionRefusedError:
        return False
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ECONNREFUSED):
            return False
        raise
    else:
        return True
    finally:
        probe.close()
