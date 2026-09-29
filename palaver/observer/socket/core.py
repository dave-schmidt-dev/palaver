"""
The module logger, the filesystem/lock/timeout/socket-path constants, and the exception
types every other submodule raises.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger("palaver.observer.socket")

#: Filesystems where `flock` is known to be a real lock. An allowlist,
#: because the failure this guards against is a filesystem that *accepts*
#: `flock` and does nothing — which no probe can distinguish from a lock
#: that works, since both return success. Anything not named here stops the
#: daemon with an error naming the filesystem, which is a better outcome
#: than two writers on a share.
LOCAL_FILESYSTEMS = frozenset({"apfs", "hfs", "ufs"})

#: `MFSTYPENAMELEN` from `sys/mount.h`.
_FSTYPENAME_LEN = 16

#: `MAXPATHLEN`, the width of both name fields in `struct statfs`.
_MNTNAME_LEN = 1024

#: What `sizeof(struct statfs)` reports on this platform, checked against
#: the C compiler on 2026-08-15 (arm64 macOS 15): 2168 bytes, with
#: `f_fstypename` at offset 72 and `f_mntonname` at 88. A ctypes struct that
#: disagreed with the real layout would read a plausible-looking string from
#: the wrong offset and answer confidently, so the size is asserted rather
#: than assumed.
_STATFS_SIZE = 2168

#: The longest socket path this platform accepts, in bytes. `sun_path` is a
#: fixed 104-byte array in `sys/un.h` and the name must be NUL-terminated
#: within it, so 103 bytes is the most that can be bound. Bisected on
#: 2026-08-15: 103 binds, 104 raises `OSError: AF_UNIX path too long`.
#:
#: This is a real constraint on where a store may live, not a detail. A
#: deeply nested project directory produces a socket path over the limit,
#: and the kernel's error names neither the limit nor which path was too
#: long — so it is checked here, where both can be reported.
MAX_SOCKET_PATH_BYTES = 103

#: How long a probe or a request waits before giving up. A daemon that has
#: wedged mid-accept must not hang the MCP process indefinitely: the caller
#: needs a refusal it can report, not a stall.
DEFAULT_TIMEOUT = 5.0

# macOS `struct flock`, used only for a queryable liveness marker beside the
# authoritative BSD `flock`. Record locks release when *any* descriptor for
# the file closes, so `single_writer` owns exactly one descriptor for this
# path and no helper opens another.
_F_RDLCK = 1
_F_UNLCK = 2
_F_WRLCK = 3
_F_GETLK = 7
_F_SETLK = 8

StatusFn = Callable[[str], None]


class SingleWriterError(RuntimeError):
    """A daemon cannot take the writer role, and must not proceed."""


class DaemonAlreadyRunningError(SingleWriterError):
    """Another process already holds the writer role."""


class NonLocalFilesystemError(SingleWriterError):
    """The data directory is somewhere `flock` cannot be trusted."""


class SocketPathTooLongError(SingleWriterError):
    """The socket path exceeds what `sun_path` can hold."""


class DaemonUnavailableError(RuntimeError):
    """No daemon is listening, so a write cannot be performed at all."""


class UnsupportedOperationError(ValueError):
    """A request naming an operation the write path does not perform."""
