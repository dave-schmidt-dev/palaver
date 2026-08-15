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

import contextlib
import ctypes
import ctypes.util
import errno
import fcntl
import json
import logging
import os
import platform
import select
import socket as socket_module
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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


def lock_path_for(db_path: Path) -> Path:
    """The lock file that guards `db_path`'s writer role."""
    return db_path.parent / "palaver.lock"


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


@contextlib.contextmanager
def single_writer(
    db_path: Path,
    *,
    on_status: StatusFn | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    backlog: int = 16,
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
        backlog: `listen()` backlog.

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

    # Degrade here, do not refuse. The lock is what makes this the only
    # writer, and a lock path has no length limit; the socket only adds
    # corrections and a second liveness signal on top of it. Refusing to
    # observe at all because corrections cannot be accepted would trade a
    # missing feature for a total outage — the wrong direction for a process
    # whose entire job is to keep watching. The lock below still refuses a
    # second daemon, so nothing about single-writer safety rests on this.
    socket_path: Path | None
    try:
        socket_path = socket_path_for(db_path)
    except SocketPathTooLongError as exc:
        socket_path = None
        disabled = str(exc)

    # Opened, never truncated: the file is a lock token, and its content is
    # nobody's business. `O_CREAT` without `O_TRUNC` so a concurrent holder's
    # descriptor is never disturbed by this open.
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DaemonAlreadyRunningError(
                f"another palaver observe holds {lock_path}. Exactly one daemon may "
                "write the store; this one is stopping rather than becoming a second "
                "writer. Stop the running daemon first, or point --db elsewhere."
            ) from exc
        say(f"took the exclusive writer lock on {lock_path}")

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
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def daemon_running(db_path: Path, *, timeout: float = DEFAULT_TIMEOUT) -> bool | None:
    """Is a writer daemon serving this store right now, and can we tell?

    Read tools report this alongside their results. A crashed daemon and an
    idle one produce identical output otherwise — the same memories, the
    same timestamps — and a reader with no way to tell the difference will
    take a stale answer for a current one, which is INV-7's failure exactly.

    Three answers, not two, because a daemon at a path too deep for
    `sun_path` runs perfectly well while serving no socket to probe. Under
    the older design that combination could not exist — such a daemon
    refused to start — so `False` was the whole truth. Now it is reachable,
    and answering `False` there would report a *running* daemon as stopped:
    a confident wrong answer, which is the one thing INV-7 forbids. "I
    cannot tell" is not a confident wrong answer.

    A *read* must never fail because of this probe, so nothing here raises;
    the conditions that would are the daemon's to report at startup, where
    they can be acted on, not a recall's to raise on a write path the caller
    never asked to use.

    Args:
        db_path: The database whose daemon to check.
        timeout: Seconds to allow the connect.

    Returns:
        True if a listener answered, False if the socket is absent or
        refuses, and None if this store has no probeable socket at all.
    """
    try:
        socket_path = socket_path_for(db_path)
    except SocketPathTooLongError as exc:
        log.warning("cannot probe for a daemon: %s", exc)
        return None
    try:
        return _probe(socket_path, timeout)
    except OSError:
        # `_probe` already narrowed "absent" and "refused" to False, so
        # anything still raising is an unexpected condition rather than
        # evidence of absence. Unknown, not dead.
        return None


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
    socket_path = socket_path_for(db_path)
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


# ---------------------------------------------------------------------------
# The daemon side: what a request is allowed to ask for, and how it is served.
# ---------------------------------------------------------------------------

#: Every operation the write path performs, by name. A closed set, checked
#: before anything touches the database.
#:
#: The protocol carries an operation *name* and typed arguments — never SQL,
#: and never a table or column name. That is what makes "an UPDATE or DELETE
#: naming an existing memory row" unreachable from a caller rather than
#: merely discouraged: there is no request shape that expresses one. The
#: schema's own triggers (`memories_no_delete`, `memories_id_immutable`, and
#: the rest) are the second layer, and they are what would catch a future
#: operation added here carelessly.
WRITE_OPERATIONS = frozenset({"correct"})

#: What `palaver_correct` records as a memory's origin, so a correction is
#: distinguishable from an extraction in any later read.
CORRECTION_ORIGIN = "user-correction"


def apply_request(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> dict:
    """Perform one write request on the daemon's connection.

    Args:
        conn: The daemon's single writable connection.
        payload: A decoded request. `op` names the operation; the remaining
            keys are its arguments.

    Returns:
        A reply dict. `{"ok": True, ...}` on success; `{"ok": False,
        "error": ..., "detail": ...}` when the request was refused, so a
        refusal reaches the caller as data rather than as a dropped
        connection they would have to guess about.

    Raises:
        Nothing. Every failure is turned into a reply. A daemon that let an
        exception escape here would drop the connection, and the MCP process
        would report "the daemon did not reply" — which is what it says when
        the daemon has *crashed*. Two very different situations must not
        produce the same message (INV-7).
    """
    op = payload.get("op")
    try:
        if op not in WRITE_OPERATIONS:
            raise UnsupportedOperationError(
                f"{op!r} is not something the write path does. It performs exactly "
                f"{sorted(WRITE_OPERATIONS)}, and takes an operation name with typed "
                "arguments — never SQL, a table name, or a column name. Memories are "
                "append-only (INV-4): a correction is a new row that supersedes the "
                "old one, and the old row is never modified or removed."
            )
        return _correct(conn, payload)
    except Exception as exc:  # noqa: BLE001 - the reply *is* the error channel
        log.warning("write request %r refused: %s", op, exc)
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def _correct(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> dict:
    """Supersede one memory with a corrected statement at tier 1.

    The successor inherits the predecessor's evidence anchors rather than
    inventing new ones. That is the honest reading of a human correction:
    the same span of transcript is being pointed at, and what changed is the
    reading of it. Fabricating an anchor to satisfy INV-6 would put a
    citation in the store that leads somewhere the statement does not come
    from, which is worse than no citation at all.
    """
    from palaver.memory.evidence import EvidenceAnchor  # noqa: PLC0415 - cycle
    from palaver.memory.supersede import supersede_memory  # noqa: PLC0415 - cycle
    from palaver.memory.tiers import TIER_USER_INSTRUCTION  # noqa: PLC0415 - cycle

    memory_id = payload.get("memory_id")
    statement = payload.get("statement")
    if not isinstance(memory_id, int) or isinstance(memory_id, bool):
        raise ValueError(f"memory_id must be an integer, got {memory_id!r}")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")

    anchors = [
        EvidenceAnchor(
            start_offset=start,
            end_offset=end,
            transcript_chunk_id=chunk_id,
            event_id=event_id,
        )
        for start, end, chunk_id, event_id in conn.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ).fetchall()
    ]
    if not anchors:
        raise LookupError(
            f"memory {memory_id} has no evidence to inherit, so a correction of it "
            "would have none either (INV-6). Either the id names no memory, or the "
            "store is inconsistent."
        )

    successor_id = supersede_memory(
        conn,
        predecessor_id=memory_id,
        statement=statement.strip(),
        origin=CORRECTION_ORIGIN,
        tier=TIER_USER_INSTRUCTION,
        evidence=anchors,
    )
    conn.commit()
    return {"ok": True, "memory_id": successor_id, "supersedes": memory_id}


def serve_request(server: socket_module.socket, conn: sqlite3.Connection) -> bool:
    """Accept one connection, apply its request, and reply.

    Args:
        server: The listening socket from `single_writer`.
        conn: The daemon's writable connection.

    Returns:
        True if a request was served. False if the connection carried
        nothing — which is what the startup liveness probe leaves behind,
        and what `daemon_alive` does on every read. Those must not be logged
        as malformed requests; they are the mechanism working.
    """
    client, _ = server.accept()
    try:
        client.settimeout(DEFAULT_TIMEOUT)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if chunks[-1].endswith(b"\n"):
                break
        body = b"".join(chunks).strip()
        if not body:
            return False

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            reply: dict = {"ok": False, "error": "JSONDecodeError", "detail": str(exc)}
        else:
            reply = apply_request(conn, payload)

        client.sendall(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
        return True
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
