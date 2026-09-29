"""
Task 2.4: raw INSERT OR REPLACE / UPDATE OR REPLACE / DELETE attacks that try to bypass
supersede_memory or destroy a successor.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.memory.supersede import is_superseded, successor_of, supersede_memory
from palaver.memory.tiers import (
    TIER_USER_INSTRUCTION,
)
from tests._memory_support import (
    _reserved_memory_id,
    _seed_anchor,
    _seed_evidence_for,
    _seed_project_and_session,
    _write_evidenced_memory,
)
from tests._memory_support import db as db


@pytest.mark.inv4
def test_insert_or_replace_on_an_existing_memory_id_raises_so_supersedes_is_the_only_path(db):
    """`INSERT OR REPLACE` cannot overwrite a memory in place.

    Measured against a store at migration 4: this exact statement silently
    rewrote a row's tier from 4 to 1 and left no trace. `REPLACE` is a
    DELETE followed by an INSERT, so the migration-3 `BEFORE UPDATE OF tier`
    trigger never fired — a `BEFORE UPDATE` trigger cannot see an operation
    that is not an UPDATE. Migration 5's `memories_id_never_reused` is a
    `BEFORE INSERT` trigger, which fires for every insert shape including
    this one and independently of any per-connection pragma.

    Positive control, required by the plan: an ordinary `INSERT` of a NEW
    row still succeeds on this same connection, so the refusal above is
    about reusing an existing id and not about the connection or the table
    having become unwritable. Two further controls cover the `rowid`
    spelling of the same column and an `INSERT OR REPLACE` that does not
    collide.
    """
    project_id, session_id = _seed_project_and_session(db)
    memory_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an observer inference about cadence", 4
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()

    with pytest.raises(sqlite3.IntegrityError, match="never reused"):
        db.execute(
            "INSERT OR REPLACE INTO memories(id, project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                project_id,
                session_id,
                "fixture: a statement smuggled in over the original",
                "attacker",
                TIER_USER_INSTRUCTION,
            ),
        )

    # The `rowid` spelling of `memories.id` is the same column and is refused too.
    with pytest.raises(sqlite3.IntegrityError, match="never reused"):
        db.execute(
            "INSERT OR REPLACE INTO memories"
            "(rowid, project_id, session_id, statement, origin, tier) VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                project_id,
                session_id,
                "fixture: the same attack spelled rowid",
                "attacker",
                TIER_USER_INSTRUCTION,
            ),
        )

    assert db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone() == before

    # Positive control (plan requirement): an ordinary INSERT of a new row
    # still succeeds on this same connection.
    new_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an ordinary later observation", 4
    )
    assert new_id != memory_id
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2

    # Positive control: `INSERT OR REPLACE` itself is not banned — only the
    # collision is — so a non-colliding id still writes.
    fresh_id = memory_id + 10_000
    _seed_evidence_for(db, session_id, fresh_id)
    db.execute(
        "INSERT OR REPLACE INTO memories(id, project_id, session_id, statement, origin, tier) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (fresh_id, project_id, session_id, "fixture: a replace with a fresh id", "observer", 4),
    )
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 3


@pytest.mark.inv4
def test_insert_or_replace_cannot_destroy_a_successor_by_colliding_on_supersedes(db):
    """A REPLACE colliding on the supersedes unique index cannot delete the incumbent.

    Measured with the unique index in place but no `BEFORE INSERT` guard:
    this statement succeeded, the existing successor row vanished, and the
    row count was unchanged — a memory destroyed with no error and no gap in
    the count to notice. SQLite's REPLACE conflict resolution deletes
    conflicting rows without firing delete triggers unless `PRAGMA
    recursive_triggers` is ON, and a per-connection pragma is not an
    invariant, so the guard is a `BEFORE INSERT` trigger instead.

    Positive control: a supersession of a *different* predecessor still
    succeeds through the same `INSERT OR REPLACE` statement shape.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the original conclusion", 4
    )
    other_predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an unrelated original conclusion", 4
    )
    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the incumbent correction",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()
    count_before = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # The id is stated (and its evidence pre-written) only because migration
    # 8 requires it; it does not collide with anything, so the `supersedes`
    # unique index is still the only conflict this REPLACE has to resolve.
    with _reserved_memory_id(db, session_id) as evicting_id:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            db.execute(
                "INSERT OR REPLACE INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    evicting_id,
                    project_id,
                    session_id,
                    "fixture: a correction that would evict the incumbent",
                    "attacker",
                    4,
                    predecessor_id,
                ),
            )

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == count_before
    assert (
        db.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (successor_id,)).fetchone()[0] == 1
    )
    assert successor_of(db, predecessor_id) == successor_id

    # Positive control: the same statement shape against an unclaimed
    # predecessor writes normally.
    with _reserved_memory_id(db, session_id) as unrelated_correction_id:
        db.execute(
            "INSERT OR REPLACE INTO memories"
            "(id, project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                unrelated_correction_id,
                project_id,
                session_id,
                "fixture: a first correction of the unrelated conclusion",
                "observer",
                4,
                other_predecessor_id,
            ),
        )
    assert is_superseded(db, other_predecessor_id)


@pytest.mark.inv4
def test_update_or_replace_cannot_destroy_a_superseded_row_through_a_rowid_rewrite(db):
    """`UPDATE OR REPLACE ... SET rowid = <another row's id>` cannot delete that row.

    Measured with `BEFORE UPDATE OF id` as the guard: this statement
    succeeded and destroyed the row it collided with, because a trigger
    scoped `OF id` does not fire when the SET clause spells the same column
    `rowid` or `_rowid_`. Migration 5's `memories_id_immutable` is written
    as `BEFORE UPDATE ... WHEN NEW.id IS NOT OLD.id`, which is spelling-
    independent.

    Positive control: an UPDATE of a non-identity column on a row nothing
    supersedes still succeeds on the same connection.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the memory an attacker wants gone", 4
    )
    supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: its legitimate correction",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    bystander_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an unrelated bystander memory", 4
    )
    db.commit()
    count_before = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    for column in ("rowid", "_rowid_", "id"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(
                f"UPDATE OR REPLACE memories SET {column} = ? WHERE id = ?",
                (predecessor_id, bystander_id),
            )

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == count_before
    assert (
        db.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (predecessor_id,)).fetchone()[0]
        == 1
    )

    # Positive control: the bystander is an ordinary live row, and the
    # refusals above are the identity guard rather than a dead connection.
    # Since migration 8 the columns it froze are refused by their own rule —
    # a different message — while `created_at`, which no rule covers on a
    # live row, still writes.
    with pytest.raises(sqlite3.IntegrityError, match="supersede, never rewrite"):
        db.execute(
            "UPDATE memories SET origin = ? WHERE id = ?", ("observer-amended", bystander_id)
        )
    db.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2003-03-03T00:00:00.000Z", bystander_id),
    )
    assert (
        db.execute("SELECT created_at FROM memories WHERE id = ?", (bystander_id,)).fetchone()[0]
        == "2003-03-03T00:00:00.000Z"
    )


@pytest.mark.inv4
def test_delete_raises_so_writing_a_successor_that_supersedes_is_the_only_correction(db):
    """No `DELETE` against `memories` succeeds, whether it targets one row or all of them.

    `tests/test_memory.py::test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module`
    proves this project's own code never issues one; this proves the
    database refuses one issued by anything else.

    Positive control: an ordinary INSERT still succeeds on the same
    connection, and `DELETE` against a different table (`current_state`,
    which is regeneratable and deliberately outside INV-4) still works — so
    this is a guard on `memories`, not a broken connection.
    """
    project_id, session_id = _seed_project_and_session(db)
    memory_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a memory somebody would rather forget", 4
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM memories")

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    # Positive control 1: writes still work.
    assert (
        _write_evidenced_memory(db, project_id, session_id, "fixture: a later observation", 4)
        != memory_id
    )

    # Positive control 2: the regeneratable table is still deletable, so the
    # guard is scoped to durable memory rather than to the whole store.
    db.execute(
        "INSERT INTO current_state(project_id, session_id, key, value) VALUES (?, ?, ?, ?)",
        (project_id, session_id, "current_task", "fixture: an ephemeral summary"),
    )
    db.execute("DELETE FROM current_state")
    assert db.execute("SELECT COUNT(*) FROM current_state").fetchone()[0] == 0


@pytest.mark.inv4
def test_update_or_replace_cannot_destroy_a_successor_by_rewriting_supersedes_on_another_row(db):
    """A bystander row cannot seize a predecessor's supersedes link and evict its successor.

    Measured with the unique index in place but `memories.supersedes` still
    writable: `UPDATE OR REPLACE memories SET supersedes = <claimed> WHERE
    id = <bystander>` succeeded and the incumbent successor row was deleted
    to resolve the unique-index conflict — again with no error and no delete
    trigger firing. `memories_supersedes_immutable` closes it: the
    supersession edge is written once, at insert, exactly like `tier`.

    This is a distinct hole from
    `test_supersede_preserves_original_row`'s: the row being UPDATEd here is
    a *current* memory that nothing supersedes, so the superseded-row guard
    does not apply to it at all.

    Positive control: an UPDATE of `origin` on that same bystander succeeds,
    so the refusal is scoped to the `supersedes` column.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a conclusion with one correction", 4
    )
    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the incumbent correction",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    bystander_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a current, unrelated memory", 4
    )
    db.commit()
    count_before = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # Control for the premise: the bystander is not itself superseded, so
    # the superseded-row guard is not what refuses the statement below.
    assert not is_superseded(db, bystander_id)

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute(
            "UPDATE OR REPLACE memories SET supersedes = ? WHERE id = ?",
            (predecessor_id, bystander_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute(
            "UPDATE memories SET supersedes = ? WHERE id = ?", (predecessor_id, bystander_id)
        )

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == count_before
    assert successor_of(db, predecessor_id) == successor_id

    # Positive control: the refusal above is the `supersedes` rule, not a
    # dead connection. Since migration 8 `origin` is refused as well, but by
    # a different rule with a different message, and `created_at` — covered
    # by neither on a live row — still writes.
    with pytest.raises(sqlite3.IntegrityError, match="supersede, never rewrite"):
        db.execute(
            "UPDATE memories SET origin = ? WHERE id = ?", ("observer-amended", bystander_id)
        )
    db.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2004-04-04T00:00:00.000Z", bystander_id),
    )
    assert (
        db.execute("SELECT created_at FROM memories WHERE id = ?", (bystander_id,)).fetchone()[0]
        == "2004-04-04T00:00:00.000Z"
    )
