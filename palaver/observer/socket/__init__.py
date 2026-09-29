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

import contextlib  # noqa: F401
import ctypes
import ctypes.util  # noqa: F401
import errno  # noqa: F401
import fcntl  # noqa: F401
import json  # noqa: F401
import logging  # noqa: F401
import os  # noqa: F401
import platform  # noqa: F401
import select  # noqa: F401
import socket as socket_module  # noqa: F401
import sqlite3  # noqa: F401
import time  # noqa: F401
from collections.abc import Callable, Iterator, Mapping  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from .client import request as request
from .core import _F_GETLK as _F_GETLK
from .core import _F_RDLCK as _F_RDLCK
from .core import _F_SETLK as _F_SETLK
from .core import _F_UNLCK as _F_UNLCK
from .core import _F_WRLCK as _F_WRLCK
from .core import _FSTYPENAME_LEN as _FSTYPENAME_LEN
from .core import _MNTNAME_LEN as _MNTNAME_LEN
from .core import _STATFS_SIZE as _STATFS_SIZE
from .core import DEFAULT_TIMEOUT as DEFAULT_TIMEOUT
from .core import LOCAL_FILESYSTEMS as LOCAL_FILESYSTEMS
from .core import MAX_SOCKET_PATH_BYTES as MAX_SOCKET_PATH_BYTES
from .core import DaemonAlreadyRunningError as DaemonAlreadyRunningError
from .core import DaemonUnavailableError as DaemonUnavailableError
from .core import NonLocalFilesystemError as NonLocalFilesystemError
from .core import SingleWriterError as SingleWriterError
from .core import SocketPathTooLongError as SocketPathTooLongError
from .core import StatusFn as StatusFn
from .core import UnsupportedOperationError as UnsupportedOperationError
from .core import log as log
from .filesystem import _Statfs as _Statfs
from .filesystem import filesystem_type as filesystem_type
from .filesystem import require_local_filesystem as require_local_filesystem
from .locks import _Flock as _Flock
from .locks import _lock_holder_pid as _lock_holder_pid
from .locks import _set_record_lock as _set_record_lock
from .paths import _probe as _probe
from .paths import liveness_lock_path_for as liveness_lock_path_for
from .paths import lock_path_for as lock_path_for
from .paths import socket_path_for as socket_path_for
from .protocol import CORRECTION_ORIGIN as CORRECTION_ORIGIN
from .protocol import WRITE_OPERATIONS as WRITE_OPERATIONS
from .protocol import _correct as _correct
from .protocol import _record_query as _record_query
from .protocol import apply_request as apply_request
from .serve import serve_request as serve_request
from .serve import serve_until as serve_until
from .writer import daemon_running as daemon_running
from .writer import single_writer as single_writer

__all__ = [
    "CORRECTION_ORIGIN",
    "DEFAULT_TIMEOUT",
    "LOCAL_FILESYSTEMS",
    "MAX_SOCKET_PATH_BYTES",
    "WRITE_OPERATIONS",
    "_FSTYPENAME_LEN",
    "_F_GETLK",
    "_F_RDLCK",
    "_F_SETLK",
    "_F_UNLCK",
    "_F_WRLCK",
    "_MNTNAME_LEN",
    "_STATFS_SIZE",
    "DaemonAlreadyRunningError",
    "DaemonUnavailableError",
    "NonLocalFilesystemError",
    "SingleWriterError",
    "SocketPathTooLongError",
    "StatusFn",
    "UnsupportedOperationError",
    "_Flock",
    "_Statfs",
    "_correct",
    "_lock_holder_pid",
    "_probe",
    "_record_query",
    "_set_record_lock",
    "apply_request",
    "daemon_running",
    "filesystem_type",
    "liveness_lock_path_for",
    "lock_path_for",
    "log",
    "request",
    "require_local_filesystem",
    "serve_request",
    "serve_until",
    "single_writer",
    "socket_path_for",
]
