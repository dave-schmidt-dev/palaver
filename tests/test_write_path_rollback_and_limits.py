"""
Review 2026-10-27 findings 1 and 4: a refused or failed write never leaves a half-written
transaction on the daemon's shared connection, and the socket handler is bounded and total.
"""

from __future__ import annotations

import socket
import sqlite3
import threading

import pytest

from palaver.observer import socket as writer_socket
from palaver.observer.socket import serve
from tests._supervision_support import _row_count, _seed_memory
from tests._supervision_support import short_tmp as short_tmp
from tests._test_scheduler_support import NOW, _daemon, _human, _write_store

# --- finding 1: rollback ------------------------------------------------------


def _evidence_count(db_path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0]
    finally:
        conn.close()


def test_a_refused_correction_leaves_no_open_transaction_and_no_stray_evidence(short_tmp):
    """The second correction of one memory is refused by the one-successor guard.

    It inserts evidence child-first before the guard fires. Without a rollback
    that evidence stays pending on the shared connection and the next commit
    attaches it to whichever memory takes the reused id.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    conn = connect(db_path)
    try:
        first = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "first"}
        )
        assert first["ok"] is True
        evidence_after_first = _evidence_count(db_path)

        second = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "second"}
        )
        assert second["ok"] is False
        assert conn.in_transaction is False, "the refusal left its transaction open"

        # Positive control: the connection still works, and the next memory
        # carries only its own evidence.
        third = writer_socket.apply_request(
            conn,
            {"op": "correct", "memory_id": first["memory_id"], "statement": "third"},
        )
        assert third["ok"] is True
    finally:
        conn.close()

    assert _row_count(db_path) == 3
    assert _evidence_count(db_path) == evidence_after_first + 1


def test_a_failed_extraction_rolls_back_its_partial_batch(tmp_path):
    """A session whose extractor raises midway must not leave rows pending."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])

    def half_writing_extractor(work, *, conn, on_status):
        conn.execute("INSERT INTO projects (name, path) VALUES ('partial', '/partial')")
        raise RuntimeError("extraction died midway")

    daemon = _daemon(tmp_path, sample_root, half_writing_extractor)
    with daemon:
        result = daemon.tick(now=NOW)
        assert len(result.failed) == 1
        assert daemon.conn.in_transaction is False
        daemon.conn.commit()
        assert (
            daemon.conn.execute("SELECT COUNT(*) FROM projects WHERE name='partial'").fetchone()[0]
            == 0
        )


# --- finding 4: the socket handler is bounded and total ------------------------


def _serve_one(short_tmp, send, *, shrink: bool = False):
    """Run `serve_request` against one client that calls `send(client)`."""
    from palaver.store.migrate import connect, migrate

    db_path = short_tmp / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    outcome: dict = {}
    try:
        with writer_socket.single_writer(db_path) as server:
            path = writer_socket.socket_path_for(db_path)

            def client_side() -> None:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(10)
                client.connect(str(path))
                try:
                    send(client)
                    outcome["reply"] = client.recv(65536)
                except OSError as exc:
                    outcome["client_error"] = exc
                finally:
                    client.close()

            thread = threading.Thread(target=client_side)
            thread.start()
            outcome["served"] = serve.serve_request(server, conn)
            thread.join(timeout=10)
    finally:
        conn.close()
    return outcome


@pytest.mark.parametrize(
    "body",
    [b"null\n", b"[]\n", b"42\n", b'"text"\n', b"\x80\xff\n", b"[" * 200_000 + b"\n"],
    ids=["null", "list", "int", "string", "bad-utf8", "deeply-nested"],
)
def test_a_malformed_body_gets_an_error_reply_instead_of_escaping(short_tmp, body):
    outcome = _serve_one(short_tmp, lambda c: c.sendall(body))
    assert outcome["served"] is True
    assert b'"ok":false' in outcome["reply"]


def test_a_body_without_a_newline_is_capped(short_tmp, monkeypatch):
    monkeypatch.setattr(serve, "MAX_REQUEST_BYTES", 1024)

    def send(client):
        client.sendall(b"x" * 4096)

    outcome = _serve_one(short_tmp, send)
    assert outcome["served"] is False
    assert b"RequestTooLarge" in outcome.get("reply", b"")


def test_a_trickling_client_hits_the_total_deadline(short_tmp, monkeypatch):
    """One byte per interval resets `settimeout`; the total deadline must not reset."""
    import time

    monkeypatch.setattr(serve, "REQUEST_DEADLINE_SECONDS", 0.6)

    def send(client):
        for _ in range(20):
            client.sendall(b" ")
            time.sleep(0.2)

    started = time.monotonic()
    outcome = _serve_one(short_tmp, send)
    assert time.monotonic() - started < 3
    assert outcome["served"] is False


def test_accept_failure_does_not_escape():
    class BrokenServer:
        def accept(self):
            raise OSError(24, "Too many open files")

    assert serve.serve_request(BrokenServer(), None) is False  # type: ignore[arg-type]


def test_a_well_formed_request_still_works_after_the_guards(short_tmp):
    """Positive control for the cases above."""
    outcome = _serve_one(short_tmp, lambda c: c.sendall(b'{"op": "nope"}\n'))
    assert outcome["served"] is True
    assert b"UnsupportedOperationError" in outcome["reply"]
