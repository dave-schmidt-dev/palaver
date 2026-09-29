"""
Task 3.4: ephemeral current_state rows versus durable memory rows, one row per key,
idempotent re-extraction.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.extract.persist import (
    BLOCKERS_NOW,
    CURRENT_TASK,
    EPHEMERAL_KEYS,
    OPEN_QUESTIONS,
    REMAINING_WORK,
    Extraction,
    GroundedClaim,
    persist_extraction,
    upsert_current_state,
)
from tests._extraction_support import _chunk_id, _memory_counts, _replayed, _user_record

# --- task 3.4: ephemeral state (current_state) versus durable memory --------


def _current_state_rows(
    conn: sqlite3.Connection, project_id: int, session_id: int | None
) -> dict[str, str]:
    """Return `{key: value}` for every `current_state` row in this scope.

    `session_id IS ?` rather than `= ?` so a `None` scope is queried the
    same way `upsert_current_state` writes it, instead of silently matching
    nothing (`session_id = NULL` is never true in SQL).
    """
    rows = conn.execute(
        "SELECT key, value FROM current_state WHERE project_id = ? AND session_id IS ?",
        (project_id, session_id),
    ).fetchall()
    return dict(rows)


def test_current_state_is_one_row_per_key_and_second_extraction_is_idempotent(tmp_path):
    """Extracting the same session twice upserts four `current_state` rows in place —
    one per `(project_id, session_id, key)`, never more — while a decision included in
    the same extraction keeps accumulating in `memories` on every call.

    The two destinations are asserted on the *same* two calls so the flat
    `current_state` count is legible as routing rather than as a persist layer that
    silently writes nothing on the second pass: if it wrote nothing, `memories` would
    stay flat too, and it does not.
    """
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("hold the release")])
    try:
        # `_replayed` itself writes one memory (replay.py's own "first observed" fact,
        # task 2.5) before this test's extraction runs at all, so counts below are
        # asserted relative to that baseline rather than against an absolute zero.
        baseline_memories = _memory_counts(conn)[0]
        chunk_id = _chunk_id(conn, 1)
        extraction = Extraction(
            current_task="write the migration",
            remaining_work="run the eval harness",
            blockers_now="waiting on the GGUF pair",
            open_questions="which port does the eval leg use",
            decisions=(
                GroundedClaim(
                    statement="hold the release",
                    quote="hold the release",
                    transcript_chunk_id=chunk_id,
                ),
            ),
        )

        first = persist_extraction(
            conn, project_id=project_id, session_id=session_id, extraction=extraction
        )
        assert set(first.current_state_keys_written) == set(EPHEMERAL_KEYS)
        rows = _current_state_rows(conn, project_id, session_id)
        assert rows == {
            CURRENT_TASK: "write the migration",
            REMAINING_WORK: "run the eval harness",
            BLOCKERS_NOW: "waiting on the GGUF pair",
            OPEN_QUESTIONS: "which port does the eval leg use",
        }
        assert _memory_counts(conn)[0] == baseline_memories + 1

        second = persist_extraction(
            conn, project_id=project_id, session_id=session_id, extraction=extraction
        )
        assert set(second.current_state_keys_written) == set(EPHEMERAL_KEYS)
        rows_after_second = _current_state_rows(conn, project_id, session_id)
        assert len(rows_after_second) == 4  # still one row per key, not eight
        assert rows_after_second == rows  # same values, upserted in place
        # decisions accumulate on every call: append-only (INV-4), unlike current_state.
        assert _memory_counts(conn)[0] == baseline_memories + 2
    finally:
        conn.close()


def test_changed_current_task_overwrites_its_current_state_row_and_writes_no_memories_row(
    tmp_path,
):
    """A changed `current_task` value overwrites its own `current_state` row and
    contributes nothing to `memories` — paired, in the same call and on the same
    connection, with a decision that DOES write a `memories` row, so the zero is
    the routing logic and not an inert persist layer.
    """
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("freeze the schema")])
    try:
        # See the idempotency test above: `_replayed` writes one memory of its own
        # before this test's first `persist_extraction` call.
        baseline_memories = _memory_counts(conn)[0]
        chunk_id = _chunk_id(conn, 1)

        persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=Extraction(current_task="draft the schema migration"),
        )
        assert _current_state_rows(conn, project_id, session_id) == {
            CURRENT_TASK: "draft the schema migration"
        }
        assert _memory_counts(conn)[0] == baseline_memories  # unchanged: no memories row yet

        result = persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=Extraction(
                current_task="review the schema migration",
                decisions=(
                    GroundedClaim(
                        statement="freeze the schema",
                        quote="freeze the schema",
                        transcript_chunk_id=chunk_id,
                    ),
                ),
            ),
        )
        # Still exactly one row for this key, overwritten rather than duplicated.
        assert _current_state_rows(conn, project_id, session_id) == {
            CURRENT_TASK: "review the schema migration"
        }
        assert result.current_state_keys_written == (CURRENT_TASK,)

        # The decision did write — the layer is live, and the zero above was routing.
        assert _memory_counts(conn)[0] == baseline_memories + 1
        assert len(result.decision_memory_ids) == 1
    finally:
        conn.close()


def test_resolved_question_writes_one_memories_row_and_current_state_gets_open_questions(
    tmp_path,
):
    """A newly resolved question writes exactly one `memories` row and no
    `current_state` row under its own key — paired, in the same call, with an
    `open_questions` value that DOES land in `current_state`, so both assertions
    measure routing rather than an all-or-nothing persist layer.
    """
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("ship on Friday")])
    try:
        baseline_memories = _memory_counts(conn)[0]
        chunk_id = _chunk_id(conn, 1)

        result = persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=Extraction(
                open_questions="does staging need a rehearsal run",
                resolved_questions=(
                    GroundedClaim(
                        statement="ship on Friday",
                        quote="ship on Friday",
                        transcript_chunk_id=chunk_id,
                    ),
                ),
            ),
        )

        assert _memory_counts(conn)[0] == baseline_memories + 1
        assert len(result.resolved_question_memory_ids) == 1
        assert result.decision_memory_ids == ()  # resolved questions are not decisions

        assert _current_state_rows(conn, project_id, session_id) == {
            OPEN_QUESTIONS: "does staging need a rehearsal run"
        }
        assert result.current_state_keys_written == (OPEN_QUESTIONS,)
    finally:
        conn.close()


@pytest.mark.inv4
def test_upsert_current_state_treats_null_session_id_as_a_single_row(tmp_path):
    """Two upserts under `session_id IS NULL` update the same row rather than
    duplicating it. The table's own `UNIQUE (project_id, session_id, key)`
    constraint does not catch this by itself — SQLite treats NULL as distinct
    from NULL in a UNIQUE index — so this pins `upsert_current_state`'s explicit
    `IS`-based lookup rather than the constraint. A concrete `session_id` for the
    same key is the positive control: it is a genuinely different row, so scoping
    still works once NULL is handled.
    """
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("freeze the schema")])
    try:
        upsert_current_state(
            conn, project_id=project_id, session_id=None, key=CURRENT_TASK, value="first"
        )
        upsert_current_state(
            conn, project_id=project_id, session_id=None, key=CURRENT_TASK, value="second"
        )
        assert _current_state_rows(conn, project_id, None) == {CURRENT_TASK: "second"}

        upsert_current_state(
            conn,
            project_id=project_id,
            session_id=session_id,
            key=CURRENT_TASK,
            value="scoped",
        )
        assert _current_state_rows(conn, project_id, session_id) == {CURRENT_TASK: "scoped"}

        (total,) = conn.execute(
            "SELECT COUNT(*) FROM current_state WHERE project_id = ? AND key = ?",
            (project_id, CURRENT_TASK),
        ).fetchone()
        assert total == 2  # one NULL-scoped row, one session-scoped row — never three
    finally:
        conn.close()


def test_current_state_updated_at_is_refreshed_on_overwrite_not_left_stale(tmp_path):
    """An upsert that overwrites an existing row always refreshes `updated_at`.

    The schema's `updated_at` DEFAULT only fires on INSERT; an UPDATE that does
    not name the column would leave a row that changed a moment ago reading as
    stale forever. A sentinel is written directly (bypassing `upsert_current_state`)
    to prove the *next* upsert actively replaces it rather than the column merely
    never having been touched by anything.
    """
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("freeze the schema")])
    try:
        upsert_current_state(
            conn, project_id=project_id, session_id=session_id, key=CURRENT_TASK, value="first"
        )
        (row_id,) = conn.execute(
            "SELECT id FROM current_state WHERE project_id = ? AND session_id = ? AND key = ?",
            (project_id, session_id, CURRENT_TASK),
        ).fetchone()

        stale = "2000-01-01T00:00:00.000Z"
        conn.execute("UPDATE current_state SET updated_at = ? WHERE id = ?", (stale, row_id))
        (confirmed_stale,) = conn.execute(
            "SELECT updated_at FROM current_state WHERE id = ?", (row_id,)
        ).fetchone()
        assert confirmed_stale == stale  # the sentinel really landed

        upsert_current_state(
            conn, project_id=project_id, session_id=session_id, key=CURRENT_TASK, value="second"
        )
        (updated_at,) = conn.execute(
            "SELECT updated_at FROM current_state WHERE id = ?", (row_id,)
        ).fetchone()
        assert updated_at != stale

        (row_count,) = conn.execute(
            "SELECT COUNT(*) FROM current_state WHERE project_id = ? AND session_id = ? "
            "AND key = ?",
            (project_id, session_id, CURRENT_TASK),
        ).fetchone()
        assert row_count == 1  # updated in place, not a second row inserted
    finally:
        conn.close()
