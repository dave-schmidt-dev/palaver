"""
The raw fcntl record-lock primitives: the ctypes flock struct, taking a lock, and
reading who holds one.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import fcntl
import os

from .core import _F_GETLK, _F_SETLK, _F_UNLCK, _F_WRLCK


class _Flock(ctypes.Structure):
    """macOS `struct flock`, including the holder pid returned by F_GETLK."""

    _fields_ = (
        ("l_start", ctypes.c_int64),
        ("l_len", ctypes.c_int64),
        ("l_pid", ctypes.c_int32),
        ("l_type", ctypes.c_int16),
        ("l_whence", ctypes.c_int16),
    )


def _set_record_lock(fd: int, lock_type: int) -> None:
    """Set the lock-file's zero-byte liveness marker without reopening it."""
    lock = _Flock(l_start=0, l_len=1, l_pid=0, l_type=lock_type, l_whence=os.SEEK_SET)
    fcntl.fcntl(fd, _F_SETLK, bytes(lock))


def _lock_holder_pid(lock_fd: int) -> int | None:
    """Return the pid holding the queryable marker, without acquiring a lock."""
    requested = _Flock(l_start=0, l_len=1, l_pid=0, l_type=_F_WRLCK, l_whence=os.SEEK_SET)
    lock = _Flock.from_buffer_copy(fcntl.fcntl(lock_fd, _F_GETLK, bytes(requested)))
    return lock.l_pid if lock.l_type != _F_UNLCK and lock.l_pid > 0 else None
