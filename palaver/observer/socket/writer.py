"""
The single-writer boundary itself: holding the lock for the daemon's lifetime, and
telling a caller whether one is already holding it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import socket as socket_module
from collections.abc import Iterator
from pathlib import Path

from . import paths
from .core import (
    _F_UNLCK,
    _F_WRLCK,
    DEFAULT_TIMEOUT,
    DaemonAlreadyRunningError,
    SocketPathTooLongError,
    StatusFn,
    log,
)
from .filesystem import require_local_filesystem
from .locks import _lock_holder_pid, _set_record_lock
from .paths import _probe, liveness_lock_path_for, lock_path_for


@contextlib.contextmanager
def single_writer(
    db_path: Path,
    *,
    on_status: StatusFn | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    backlog: int = 128,
) -> Iterator[socket_module.socket | None]:
    """Claim the writer role, yielding the socket to accept requests on.

    The lock is taken first and released last. Everything that could race —
    probing, unlinking, binding — happens inside that window, which is what
    makes the sequence safe against another daemon doing the same thing at
    the same moment.

    Args:
        db_path: The database this daemon writes. The lock and socket are
            placed beside it, so `--db` moves all three together and a test
            store cannot collide with the real one.
        on_status: INV-1 progress channel. Called with a human-readable line
            at each step that can block or fail.
        timeout: Seconds to allow the connect probe.
        backlog: `listen()` backlog. Sized against the tick interval rather
            than against expected concurrency: this daemon accepts only in
            the idle window *between* ticks (see `serve_until`), so every
            request arriving during a 30-second extraction queues here until
            it finishes. Sixteen is a handful of MCP reads. A queue slot
            costs almost nothing, and the request that overflows it is
            dropped — silently, from the client's side, if it was a query
            event (task 6.4).

    Yields:
        A bound, listening `AF_UNIX` socket — or `None` when the store sits
        too deep for `sun_path` to hold a socket beside it. The writer role
        is held either way; only the request channel is missing. Callers
        pass the value straight to `serve_until`, which handles both.

    Raises:
        NonLocalFilesystemError: The data directory is somewhere `flock`
            cannot be trusted.
        DaemonAlreadyRunningError: Another daemon holds the lock, or is
            serving the socket without one.
        OSError: The probe failed in a way that is not evidence of
            staleness; see `_probe`.
    """
    say = on_status or (lambda _message: None)
    directory = db_path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fstype = require_local_filesystem(directory)
    say(f"data directory {directory} is {fstype}; flock is trustworthy here")

    lock_path = lock_path_for(db_path)
    liveness_path = liveness_lock_path_for(db_path)

    # Degrade here, do not refuse. The lock is what makes this the only
    # writer, and a lock path has no length limit; the socket only adds
    # corrections and a second liveness signal on top of it. Refusing to
    # observe at all because corrections cannot be accepted would trade a
    # missing feature for a total outage — the wrong direction for a process
    # whose entire job is to keep watching. The lock below still refuses a
    # second daemon, so nothing about single-writer safety rests on this.
    socket_path: Path | None
    try:
        socket_path = paths.socket_path_for(db_path)
    except SocketPathTooLongError as exc:
        socket_path = None
        disabled = str(exc)

    # Opened, never truncated: the file is a lock token, and its content is
    # nobody's business. `O_CREAT` without `O_TRUNC` so a concurrent holder's
    # descriptor is never disturbed by this open.
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    liveness_fd: int | None = None
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            try:
                liveness_fd = os.open(liveness_path, os.O_RDWR)
            except OSError:
                holder_pid = None
            else:
                try:
                    holder_pid = _lock_holder_pid(liveness_fd)
                finally:
                    os.close(liveness_fd)
                    liveness_fd = None
            holder = f" (holder pid {holder_pid})" if holder_pid is not None else ""
            raise DaemonAlreadyRunningError(
                f"another palaver observe holds {lock_path}{holder}. Exactly one daemon may "
                "write the store; this one is stopping rather than becoming a second "
                "writer. Stop the running daemon first, or point --db elsewhere."
            ) from exc
        say(f"took the exclusive writer lock on {lock_path}")
        liveness_fd = os.open(liveness_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _set_record_lock(liveness_fd, _F_WRLCK)
        except OSError as exc:
            os.close(liveness_fd)
            liveness_fd = None
            raise DaemonAlreadyRunningError(
                f"could not publish the queryable liveness lock for {lock_path}; "
                "this daemon stops rather than reporting an unknown writer state."
            ) from exc

        if socket_path is None:
            # Said at WARNING volume, not debug: the daemon runs, extracts,
            # and looks entirely healthy, while `palaver_correct` fails for
            # a reason nothing downstream can see. The one place that reason
            # is visible is here, at startup, where it can be acted on.
            say(f"write requests are disabled — {disabled}")
            log.warning("write requests are disabled: %s", disabled)
            yield None
            return

        # Under the lock from here to the bind. A daemon that released now
        # and re-acquired later would reopen the very race the lock closes.
        if _probe(socket_path, timeout):
            raise DaemonAlreadyRunningError(
                f"something is already serving {socket_path} without holding "
                f"{lock_path}. That is a daemon this build did not start — an older "
                "version, or one launched by hand. Unlinking the socket would not "
                "stop it; it would keep serving on the descriptor it already has "
                "while this process bound the same name. Stop it explicitly."
            )

        if socket_path.exists():
            # Proven stale by the probe above, under the lock, so no live
            # listener can be behind this name.
            socket_path.unlink()
            say(f"removed the stale socket node at {socket_path}")

        server = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
        try:
            server.bind(str(socket_path))
            os.chmod(socket_path, 0o600)  # noqa: S103 - owner-only is the point
            server.listen(backlog)
            say(f"listening for write requests on {socket_path}")
            yield server
        finally:
            server.close()
            with contextlib.suppress(FileNotFoundError):
                socket_path.unlink()
    finally:
        if liveness_fd is not None:
            with contextlib.suppress(OSError):
                _set_record_lock(liveness_fd, _F_UNLCK)
            os.close(liveness_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def daemon_running(db_path: Path, *, timeout: float = DEFAULT_TIMEOUT) -> bool | None:
    """Is a writer daemon serving this store right now, and can we tell?

    Read tools report this alongside their results. A crashed daemon and an
    idle one produce identical output otherwise — the same memories, the
    same timestamps — and a reader with no way to tell the difference will
    take a stale answer for a current one, which is INV-7's failure exactly.

    A store too deep for `sun_path` has no request socket, but its writer
    still publishes a POSIX record lock beside the authoritative `flock`.
    `F_GETLK` asks the kernel for that holder without acquiring any lock, so
    this probe cannot make a concurrently starting daemon falsely refuse.

    A *read* must never fail because of this probe, so nothing here raises;
    the conditions that would are the daemon's to report at startup, where
    they can be acted on, not a recall's to raise on a write path the caller
    never asked to use.

    Args:
        db_path: The database whose daemon to check.
        timeout: Seconds to allow the connect.

    Returns:
        True if a listener answered or the queryable lock is held, False if
        no listener and no lock holder exist, and None when the liveness
        marker itself cannot be queried.
    """
    try:
        socket_path = paths.socket_path_for(db_path)
    except SocketPathTooLongError:
        lock_path = liveness_lock_path_for(db_path)
        try:
            lock_fd = os.open(lock_path, os.O_RDWR)
        except FileNotFoundError:
            return False
        except OSError as error:
            log.warning("cannot open liveness lock for deep store: %s", error)
            return None
        try:
            return _lock_holder_pid(lock_fd) is not None
        except OSError as error:
            log.warning("cannot query liveness lock for deep store: %s", error)
            return None
        finally:
            os.close(lock_fd)
    try:
        return _probe(socket_path, timeout)
    except OSError:
        # `_probe` already narrowed "absent" and "refused" to False, so
        # anything still raising is an unexpected condition rather than
        # evidence of absence. Unknown, not dead.
        return None
