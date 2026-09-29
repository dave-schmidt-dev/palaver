"""
The write path applying a correction in-process: a new row, inherited evidence, and
refusing an update/delete naming an existing memory.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.observer import socket as writer_socket
from tests._supervision_support import _raw_row, _row_count, _seed_memory
from tests._supervision_support import short_tmp as short_tmp


def test_a_correction_writes_a_new_row_and_leaves_the_original_byte_identical(short_tmp):
    """INV-4: supersede, never edit. Asserted column by column.

    Checking only the statement would miss a correction that rewrote the
    predecessor's tier or origin in passing, and INV-5 makes tier the field
    most worth watching.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    before = _raw_row(db_path, memory_id)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "what really happened"}
        )
    finally:
        conn.close()

    assert reply["ok"], reply
    assert reply["supersedes"] == memory_id
    assert _raw_row(db_path, memory_id) == before, "the corrected row was modified in place"

    successor = _raw_row(db_path, reply["memory_id"])
    assert successor["statement"] == "what really happened"
    assert successor["supersedes"] == memory_id
    assert successor["tier"] == 1, "a correction is a user instruction, the highest tier"
    assert successor["origin"] == writer_socket.CORRECTION_ORIGIN


def test_the_correction_inherits_the_evidence_rather_than_inventing_any(short_tmp):
    """INV-6 without a fabricated citation.

    A correction reinterprets the same span of transcript, so it points at
    the same anchors. Synthesising an anchor to satisfy the non-empty rule
    would put a citation in the store leading somewhere the statement does
    not come from -- worse than no citation, because it reads as grounded.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "corrected"}
        )
    finally:
        conn.close()

    raw = sqlite3.connect(db_path)
    try:
        original = raw.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ).fetchall()
        inherited = raw.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (reply["memory_id"],),
        ).fetchall()
    finally:
        raw.close()
    assert inherited == original
    assert inherited, "a memory with no evidence would violate INV-6"


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "update", "memory_id": 1, "statement": "rewritten"},
        {"op": "delete", "memory_id": 1},
        {"op": "sql", "sql": "UPDATE memories SET statement = 'x' WHERE id = 1"},
        {"op": "sql", "sql": "DELETE FROM memories WHERE id = 1"},
        {"memory_id": 1, "statement": "no op at all"},
    ],
)
def test_an_update_or_delete_naming_a_memory_is_refused_by_the_write_path(short_tmp, payload):
    """There is no request shape that expresses an edit.

    The protocol carries an operation *name* and typed arguments, never SQL
    and never a table or column name, so this is not a filter that could be
    bypassed with different spelling -- there is nothing to spell.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    payload = {**payload, "memory_id": memory_id} if "memory_id" in payload else payload
    before = _raw_row(db_path, memory_id)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(conn, payload)
    finally:
        conn.close()

    assert reply["ok"] is False
    assert reply["error"] == "UnsupportedOperationError"
    assert _raw_row(db_path, memory_id) == before
    assert _row_count(db_path) == 1, "a refused request still wrote something"


def test_an_ordinary_correction_lands_on_that_same_connection(short_tmp):
    """The positive control for the refusals above.

    Without it, every one of them would pass against a write path that
    refused *everything* -- including a connection opened read-only by
    mistake, or a store with no writable table at all.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    conn = connect(db_path)
    try:
        refused = writer_socket.apply_request(conn, {"op": "delete", "memory_id": memory_id})
        accepted = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "this one lands"}
        )
    finally:
        conn.close()

    assert refused["ok"] is False
    assert accepted["ok"] is True
    assert _row_count(db_path) == 2, "the correction did not write its row"
