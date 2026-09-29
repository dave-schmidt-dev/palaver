"""
What filesystem a path lives on, and refusing to proceed on one the single-writer
guarantee cannot trust.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from pathlib import Path

from .core import (
    _FSTYPENAME_LEN,
    _MNTNAME_LEN,
    _STATFS_SIZE,
    LOCAL_FILESYSTEMS,
    NonLocalFilesystemError,
)


class _Statfs(ctypes.Structure):
    """macOS `struct statfs`, as of `sys/mount.h`.

    Declared in full rather than truncated at the field being read: a short
    struct still yields the right bytes for early fields, but `_STATFS_SIZE`
    then has nothing to check against, and a layout drift in a future SDK
    would go unnoticed until it moved a field this code does read.
    """

    _fields_ = (
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", ctypes.c_int32 * 2),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * _FSTYPENAME_LEN),
        ("f_mntonname", ctypes.c_char * _MNTNAME_LEN),
        ("f_mntfromname", ctypes.c_char * _MNTNAME_LEN),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    )


def filesystem_type(path: Path) -> str:
    """Name the filesystem `path` lives on.

    Args:
        path: A directory (or a file within one) to identify.

    Returns:
        The filesystem type as the kernel reports it — `"apfs"`, `"nfs"`,
        `"smbfs"`, and so on.

    Raises:
        NonLocalFilesystemError: This platform has no implementation here.
            Palaver targets macOS, and the honest answer on anything else is
            that the check has not been written and therefore has not been
            tested. Guessing would defeat the point of the check: the whole
            reason it exists is that an unverified filesystem must not be
            treated as safe.
        OSError: `statfs` failed — usually a path that does not exist.
    """
    if platform.system() != "Darwin":
        raise NonLocalFilesystemError(
            f"filesystem_type is implemented for Darwin only, not {platform.system()!r}. "
            "The single-writer lock cannot be trusted without knowing the filesystem, "
            "so startup stops here rather than assuming."
        )

    if ctypes.sizeof(_Statfs) != _STATFS_SIZE:  # pragma: no cover - layout drift
        raise NonLocalFilesystemError(
            f"struct statfs is {ctypes.sizeof(_Statfs)} bytes here, not {_STATFS_SIZE}; "
            "this build's layout does not match the one this code was checked against, "
            "so the filesystem name it would read cannot be trusted."
        )

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buffer = _Statfs()
    if libc.statfs(os.fsencode(path), ctypes.byref(buffer)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(path))
    return buffer.f_fstypename.decode()


def require_local_filesystem(path: Path) -> str:
    """Stop startup unless `flock` on `path` means what it says.

    Args:
        path: The directory the lock file will live in.

    Returns:
        The filesystem type, for a caller that wants to report it.

    Raises:
        NonLocalFilesystemError: The filesystem is not on `LOCAL_FILESYSTEMS`.
    """
    fstype = filesystem_type(path)
    if fstype not in LOCAL_FILESYSTEMS:
        raise NonLocalFilesystemError(
            f"{path} is on a {fstype!r} filesystem, and Palaver's single-writer "
            f"guarantee needs one of {sorted(LOCAL_FILESYSTEMS)}. `flock` is a silent "
            "no-op on NFS without lockd and on some FUSE and SMB mounts: it returns "
            "success while locking nothing, so two daemons would each believe they "
            "hold it. Point --db at a local disk."
        )
    return fstype
