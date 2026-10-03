"""The single-writer boundary: one lock, one socket, one daemon.

Palaver's whole architecture rests on there being exactly one process writing
`palaver.db`. The MCP server opens the database `mode=ro` and posts writes
here instead. This module is what makes "exactly one" true rather than
hoped for.

**Why two checks and not one.** The `flock` is authoritative for daemons
that take it; the connect probe catches the one that does not — an older
build, or a process someone started by hand. Neither alone is enough:

* A lock without a probe would let this daemon unlink a socket node that a
  lock-less process is still serving on, and bind a second listener at the
  same path. Clients would then be split across two writers with no error
  anywhere.
* A probe without a lock is a time-of-check/time-of-use race. Two daemons
  starting together both probe a stale node, both see `ECONNREFUSED`, both
  unlink, and both bind — the second silently stealing the path from the
  first.

**The order is the design, not an implementation detail.** Take the
exclusive `flock` first, and hold it unbroken through the probe, the unlink,
and the bind. Everything between check and use is then serialized against
every other daemon that takes the lock, which closes the race above. A
release anywhere in the middle reopens it.

**Why a blind unlink is wrong.** A pathname socket stays live through its
owner's open descriptor no matter what happens to the filesystem name.
`unlink()` on a socket that a healthy daemon is serving does not disturb
that daemon at all — it keeps accepting on the descriptor it already holds,
while the name is now free for someone else to bind. The result is two live
listeners, the old one invisible to anything that looks the path up. So the
node is removed only after a connect proves nobody answers on it.

**Why the filesystem type is checked at all.** `flock` degrades to a silent
no-op on NFS without `lockd`, and on some FUSE and SMB mounts. Silent is the
problem: the call returns success, the daemon believes it holds an exclusive
lock, and a second daemon on another machine believes the same. The check is
an allowlist rather than a denylist, so a filesystem nobody here has tested
fails closed instead of being assumed to behave.

INV-1: every step that can block — the probe, the bind — reports through
`on_status`, so a daemon that cannot start says why rather than exiting
silently.

This repository is public. Nothing in this module is derived from a real
observed session (INV-9).
"""

from __future__ import annotations

from .client import request as request
from .core import _STATFS_SIZE as _STATFS_SIZE
from .core import LOCAL_FILESYSTEMS as LOCAL_FILESYSTEMS
from .core import MAX_SOCKET_PATH_BYTES as MAX_SOCKET_PATH_BYTES
from .core import DaemonAlreadyRunningError as DaemonAlreadyRunningError
from .core import DaemonUnavailableError as DaemonUnavailableError
from .core import NonLocalFilesystemError as NonLocalFilesystemError
from .core import SingleWriterError as SingleWriterError
from .core import SocketPathTooLongError as SocketPathTooLongError
from .core import UnsupportedOperationError as UnsupportedOperationError
from .filesystem import _Statfs as _Statfs
from .filesystem import filesystem_type as filesystem_type
from .paths import lock_path_for as lock_path_for
from .paths import socket_path_for as socket_path_for
from .protocol import CORRECTION_ORIGIN as CORRECTION_ORIGIN
from .protocol import apply_request as apply_request
from .serve import serve_request as serve_request
from .serve import serve_until as serve_until
from .writer import daemon_running as daemon_running
from .writer import single_writer as single_writer

__all__ = [
    "CORRECTION_ORIGIN",
    "LOCAL_FILESYSTEMS",
    "MAX_SOCKET_PATH_BYTES",
    "_STATFS_SIZE",
    "DaemonAlreadyRunningError",
    "DaemonUnavailableError",
    "NonLocalFilesystemError",
    "SingleWriterError",
    "SocketPathTooLongError",
    "UnsupportedOperationError",
    "_Statfs",
    "apply_request",
    "daemon_running",
    "filesystem_type",
    "lock_path_for",
    "request",
    "serve_request",
    "serve_until",
    "single_writer",
    "socket_path_for",
]
