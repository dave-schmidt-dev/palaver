"""
Task 6.3: the single-writer lock, the socket, and the order between them -- one writer
at a time, a stale socket replaced but a live one never unlinked, the filesystem
allowlist, and the sun_path length limit.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import errno
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from palaver.observer import socket as writer_socket
from palaver.observer.socket import (
    DaemonAlreadyRunningError,
    NonLocalFilesystemError,
    single_writer,
)
from tests._supervision_support import _HOLDER, _holder, _raw_row, _seed_memory
from tests._supervision_support import short_tmp as short_tmp


def test_a_second_daemon_refuses_to_start_while_the_first_still_serves(short_tmp):
    """The property the whole architecture rests on.

    Two writers on one SQLite file is not a performance problem, it is a
    correctness one, and nothing downstream can detect it after the fact.
    """
    db_path = short_tmp / "palaver.db"
    first = _holder(db_path)
    try:
        second = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(db_path), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode != 0, "a second daemon started alongside the first"
        assert "DaemonAlreadyRunningError" in second.stdout
        assert f"holder pid {first.pid}" in second.stdout
        assert first.poll() is None, "the first daemon died, so this proved nothing"
    finally:
        first.kill()
        first.wait(timeout=10)


def test_the_writer_role_is_released_when_the_first_daemon_exits(short_tmp):
    """The positive control for the test above.

    Without it, "the second process exited non-zero" would be equally
    consistent with a lock that can never be taken by anyone.
    """
    db_path = short_tmp / "palaver.db"
    first = _holder(db_path, seconds=0.1)
    first.wait(timeout=30)

    second = subprocess.run(
        [sys.executable, "-c", _HOLDER, str(db_path), "0.1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 0, f"the role never came back: {second.stdout}{second.stderr}"
    assert "HELD" in second.stdout


def test_a_deep_store_reports_its_running_lock_holder_without_a_socket(short_tmp):
    """F_GETLK observes the liveness marker without taking any lock itself."""
    db_path = short_tmp / ("deep-" * 20) / "palaver.db"
    assert (
        len(os.fsencode(writer_socket.lock_path_for(db_path))) > writer_socket.MAX_SOCKET_PATH_BYTES
    )
    holder = _holder(db_path)
    try:
        for _ in range(20):
            assert writer_socket.daemon_running(db_path) is True
            second = subprocess.run(
                [sys.executable, "-c", _HOLDER, str(db_path), "0.1"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert second.returncode != 0
            assert f"holder pid {holder.pid}" in second.stdout
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_deep_store_liveness_logs_the_actual_lock_error(short_tmp, monkeypatch, caplog):
    """A failed liveness probe reports the OS error, not the path-length cause."""
    db_path = short_tmp / "palaver.db"

    def too_long(_path):
        raise writer_socket.SocketPathTooLongError("too long")

    def denied(*_args, **_kwargs):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(writer_socket.paths, "socket_path_for", too_long)
    monkeypatch.setattr(writer_socket.writer.os, "open", denied)

    assert writer_socket.daemon_running(db_path) is None
    assert "permission denied" in caplog.text


def test_a_stale_socket_node_is_unlinked_and_replaced_under_the_held_lock(short_tmp):
    """A crash leaves the node behind; the next daemon must not be blocked by it.

    The node is created by binding and abandoning a socket without ever
    listening, which is what a daemon killed between `bind` and `listen`
    leaves on disk: a filesystem entry that refuses every connect.
    """
    db_path = short_tmp / "palaver.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = writer_socket.socket_path_for(db_path)

    abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    abandoned.bind(str(socket_path))
    abandoned.close()  # the node outlives the socket
    assert socket_path.exists(), "the fixture did not leave a stale node"
    stale_inode = socket_path.stat().st_ino

    with single_writer(db_path) as server:
        assert socket_path.exists()
        assert socket_path.stat().st_ino != stale_inode, "the stale node was reused, not replaced"
        # Bound *and* listening: a node that exists but refuses is exactly the
        # state this test started from, so existence alone proves nothing.
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        try:
            client.connect(str(socket_path))
        finally:
            client.close()
        assert server.fileno() >= 0


def test_a_live_socket_is_never_unlinked_even_when_its_owner_holds_no_lock(short_tmp):
    """The reason the probe exists alongside the lock.

    A pathname socket keeps serving through its owner's descriptor no matter
    what happens to the name, so unlinking one that still has a listener does
    not stop it -- it just frees the name for a second listener nobody can
    see. This stands in for an older build, or a daemon started by hand.
    """
    db_path = short_tmp / "palaver.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = writer_socket.socket_path_for(db_path)

    squatter = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    squatter.bind(str(socket_path))
    squatter.listen(4)
    inode = socket_path.stat().st_ino
    try:
        with pytest.raises(DaemonAlreadyRunningError, match="without holding"):
            with single_writer(db_path):
                pass
        assert socket_path.stat().st_ino == inode, "the live socket's node was unlinked"
    finally:
        squatter.close()


def test_a_data_directory_on_an_unverified_filesystem_stops_startup(short_tmp, monkeypatch):
    """`flock` returning success is not evidence that it locked anything.

    On NFS without lockd, and on some FUSE and SMB mounts, it is a no-op --
    which is indistinguishable from a working lock at the call site. The
    filesystem name is the only signal available, so an unrecognised one has
    to fail closed.
    """
    db_path = short_tmp / "palaver.db"
    monkeypatch.setattr(writer_socket.filesystem, "filesystem_type", lambda _path: "nfs")

    with pytest.raises(NonLocalFilesystemError, match="nfs"):
        with single_writer(db_path):
            pass

    assert not writer_socket.socket_path_for(db_path).exists(), (
        "a socket was bound before the check"
    )


def test_the_filesystem_check_is_an_allowlist_not_a_denylist(short_tmp, monkeypatch):
    """A filesystem nobody here has tested must fail, not pass by omission.

    A denylist would admit every filesystem invented after this line was
    written, which is the population most likely to break `flock`.
    """
    db_path = short_tmp / "palaver.db"
    monkeypatch.setattr(writer_socket.filesystem, "filesystem_type", lambda _path: "somethingnew")
    with pytest.raises(NonLocalFilesystemError, match="somethingnew"):
        with single_writer(db_path):
            pass


def test_this_repository_lives_on_a_filesystem_the_allowlist_accepts():
    """The positive control for both tests above.

    They monkeypatch `filesystem_type`, so together they would still pass if
    the real one returned garbage for every path. This one calls it for real.
    """
    fstype = writer_socket.filesystem_type(Path(__file__).parent)
    assert fstype in writer_socket.LOCAL_FILESYSTEMS, f"unexpected filesystem {fstype!r}"


def test_the_statfs_struct_matches_the_layout_it_was_checked_against():
    """A ctypes layout that drifts reads a plausible string from the wrong offset.

    `statfs` would still return 0, and the wrong bytes would still decode --
    a confident answer from the wrong field, which is INV-7's shape. The size
    is the cheapest check that catches it.
    """
    assert ctypes.sizeof(writer_socket._Statfs) == writer_socket._STATFS_SIZE


def test_a_socket_path_over_the_sun_path_limit_is_named_rather_than_raised_raw(short_tmp):
    """`OSError: AF_UNIX path too long` names neither the limit nor the path.

    `sun_path` is a fixed 104-byte array, so 103 bytes is the most that can
    be bound -- bisected, not assumed. A user whose project sits deep enough
    to cross that gets a kernel error naming none of: which path, how long,
    or what the ceiling is. Reported here instead, with all three.
    """
    deep = short_tmp / ("d" * 120)
    with pytest.raises(writer_socket.SocketPathTooLongError) as caught:
        writer_socket.socket_path_for(deep / "palaver.db")
    message = str(caught.value)
    assert str(writer_socket.MAX_SOCKET_PATH_BYTES) in message
    assert "palaver.sock" in message


def test_a_path_at_the_limit_binds_and_one_byte_over_does_not(short_tmp):
    """The positive control: the limit is the measured one, off by nothing.

    Without this, `MAX_SOCKET_PATH_BYTES` could be any conservative number
    and the test above would still pass -- including one so low it refused
    paths that work perfectly well.
    """
    limit = writer_socket.MAX_SOCKET_PATH_BYTES
    # `<short_tmp>` + `/` + padding + `/x` + `/palaver.sock`
    padding = limit - len(str(short_tmp)) - len("/") - len("/x") - len("/palaver.sock")
    assert padding > 0, "the scratch directory is already too long to test the boundary"

    at_limit = short_tmp / ("x" * padding) / "x"
    at_limit.mkdir(parents=True)
    path = writer_socket.socket_path_for(at_limit / "palaver.db")
    assert len(str(path)) == limit

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))  # the kernel agrees this length is fine
    finally:
        server.close()

    with pytest.raises(writer_socket.SocketPathTooLongError):
        writer_socket.socket_path_for(at_limit / "yy" / "palaver.db")


def test_a_store_too_deep_for_a_socket_still_gets_a_writer(tmp_path):
    """Degrade the request channel; never degrade the observing.

    `palaver observe` exists to keep watching. A path too deep for
    `sun_path` costs it corrections and a liveness probe -- it must not cost
    it the daemon. Refusing to start here would convert a missing feature
    into a total outage, and it would do so on exactly the machines whose
    projects are nested deepest.

    `tmp_path`, deliberately: pytest's own scratch path is already over the
    limit, so this is the real configuration rather than a contrived one.
    """
    said: list[str] = []
    with pytest.raises(writer_socket.SocketPathTooLongError):
        writer_socket.socket_path_for(tmp_path / "palaver.db")  # the premise

    with single_writer(tmp_path / "palaver.db", on_status=said.append) as server:
        assert server is None, "a socket was bound at a path the kernel cannot hold"
        # The role is still held, which is the whole point -- proven from a
        # second process, since `flock` is per-open-file-description and a
        # same-process retry would conflict for the wrong reason.
        refused = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(tmp_path / "palaver.db"), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert refused.returncode != 0, "the writer role was not held"
        assert "DaemonAlreadyRunningError" in refused.stdout

    assert any("write requests are disabled" in line for line in said), (
        f"the daemon went quiet about its missing request channel: {said}"
    )
    assert any(str(writer_socket.MAX_SOCKET_PATH_BYTES) in line for line in said), (
        "the warning named no limit, so nobody reading it knows what to fix"
    )


def test_serving_without_a_socket_sleeps_the_window_out_and_serves_nothing(tmp_path):
    """The idle window has to behave the same either way.

    A degraded daemon that returned from its idle window immediately would
    spin the tick loop at whatever rate the CPU allows -- observation would
    survive, but the machine would not.
    """
    slept: list[float] = []
    served = writer_socket.serve_until(None, None, 7.5, sleep=slept.append)
    assert served == 0
    assert slept == [7.5], "the idle window was not spent asleep"


def test_serving_with_a_socket_still_answers_a_request(short_tmp):
    """The positive control for the two tests above.

    Without it, `serve_until` could return 0 unconditionally and both
    degradation tests would still pass -- while no correction ever landed.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    conn = connect(db_path)
    try:
        with single_writer(db_path) as server:
            assert server is not None
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                pending = pool.submit(
                    writer_socket.request,
                    db_path,
                    {"op": "correct", "memory_id": memory_id, "statement": "served while idle"},
                )
                # Scripted clock: `serve_until` serves for the *whole*
                # window, so a real 30-second one would cost 30 seconds
                # after the request it is meant to prove. Third reading is
                # past the deadline, which also exercises the recomputation.
                ticks = iter([0.0, 0.0, 100.0])
                served = writer_socket.serve_until(
                    server, conn, 30.0, monotonic=lambda: next(ticks)
                )
                assert served == 1
                reply = pending.result(timeout=10)
                assert reply["ok"], reply
            finally:
                pool.shutdown(wait=False)
    finally:
        conn.close()
    assert _raw_row(db_path, reply["memory_id"])["statement"] == "served while idle"
