"""
Task 2.4: the supersedes-uniqueness guard (FK pragma, unique index, trigger) and
migration 5's addition/rollback of it on an existing store.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.memory.supersede import successor_of, supersede_memory
from palaver.memory.tiers import (
    TIER_OBSERVER_INFERENCE,
    TIER_USER_INSTRUCTION,
)
from palaver.store.migrate import MigrationError, connect, current_version, migrate
from palaver.store.schema import LATEST_VERSION, SCHEMA_MIGRATIONS
from tests._memory_support import (
    _reserved_memory_id,
    _seed_anchor,
    _seed_evidence_for,
    _seed_project_and_session,
    _write_evidenced_memory,
)
from tests._memory_support import db as db


def _seed_evidenced_memory_pre_v8(
    conn: sqlite3.Connection,
    project_id: int,
    session_id: int,
    statement: str,
    tier: int,
    origin: str = "observer",
) -> int:
    """Seed a memory and its evidence in the *pre*-migration-8 order.

    Below version 8 `memory_evidence.memory_id` is an immediate foreign key,
    so a child cannot name a parent that does not exist yet and
    `write_memory`'s child-first order is invalid there — parent first is
    the only order those versions accept. Used only by the staged-migration
    tests, which are the only ones that operate on a store below the latest
    schema version.
    """
    memory_id = conn.execute(
        "INSERT INTO memories(project_id, session_id, statement, origin, tier) "
        "VALUES (?, ?, ?, ?, ?)",
        (project_id, session_id, statement, origin, tier),
    ).lastrowid
    _seed_evidence_for(conn, session_id, memory_id)
    return memory_id


@pytest.mark.inv5
def test_supersedes_naming_no_existing_memory_raises_without_the_foreign_keys_pragma(tmp_path):
    """A dangling `supersedes` is refused on a connection that never enabled foreign keys.

    `palaver.store.migrate.connect` runs `PRAGMA foreign_keys=ON`, but a
    pragma is per connection: a bare `sqlite3.connect(path)` — the next
    module, another process, a human at the shell — gets foreign keys OFF.
    Measured at that setting before migration 5, a `supersedes` naming no
    row was accepted, which also made the tier-ordering comparison vacuous:
    `NEW.tier > (SELECT tier FROM memories WHERE id = NEW.supersedes)`
    evaluates to NULL, not true, when the subquery finds nothing, so INV-5's
    rule silently did not run. The trigger closes both.

    Positive controls on the same pragma-less connection: an INSERT with
    `supersedes` NULL succeeds, and a real supersession of an existing row
    succeeds.
    """
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    raw = sqlite3.connect(str(db_path))
    try:
        # Control for the premise: foreign keys really are off here.
        assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 0

        project_id, session_id = _seed_project_and_session(raw)
        with _reserved_memory_id(raw, session_id) as dangling_id:
            with pytest.raises(sqlite3.IntegrityError, match="must name an existing memory"):
                raw.execute(
                    "INSERT INTO memories"
                    "(id, project_id, session_id, statement, origin, tier, supersedes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        dangling_id,
                        project_id,
                        session_id,
                        "fixture: a successor pointing at a memory that never existed",
                        "attacker",
                        TIER_OBSERVER_INFERENCE,
                        999_999,
                    ),
                )

        # Positive control: unlinked insert, then a genuine supersession.
        with _reserved_memory_id(raw, session_id) as predecessor_id:
            raw.execute(
                "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (predecessor_id, project_id, session_id, "fixture: a real predecessor", "obs", 4),
            )
        with _reserved_memory_id(raw, session_id) as successor_id:
            raw.execute(
                "INSERT INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    successor_id,
                    project_id,
                    session_id,
                    "fixture: a real successor",
                    "observer",
                    4,
                    predecessor_id,
                ),
            )
        assert successor_id != predecessor_id
        raw.commit()
    finally:
        raw.close()


@pytest.mark.inv4
def test_a_second_successor_for_the_same_predecessor_raises_on_the_supersedes_uniqueness(db):
    """Two rows cannot claim the same predecessor.

    Issued as raw `sqlite3` SQL. Without uniqueness, "which memory is the
    current one?" has no answer for a predecessor with two successors, and
    `superseded_memories` would report the same predecessor twice.

    Positive control: a successor for a *different* predecessor succeeds on
    the same connection, so the refusal is about the duplicate claim rather
    than about second supersessions in general.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the first invented conclusion", 4
    )
    other_predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an unrelated invented conclusion", 4
    )
    supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the corrected first conclusion",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()

    with _reserved_memory_id(db, session_id) as rival_id:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            db.execute(
                "INSERT INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rival_id,
                    project_id,
                    session_id,
                    "fixture: a rival correction of the same conclusion",
                    "observer",
                    4,
                    predecessor_id,
                ),
            )

    # Positive control: a successor for a different predecessor still lands.
    second_successor_id = supersede_memory(
        db,
        predecessor_id=other_predecessor_id,
        statement="fixture: the corrected unrelated conclusion",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()
    assert successor_of(db, other_predecessor_id) == second_successor_id


@pytest.mark.inv4
def test_supersedes_uniqueness_is_a_real_unique_index_and_not_only_a_trigger(tmp_path):
    """Both layers of the one-successor rule are independently live.

    The `BEFORE INSERT` guard fires first, so a test that merely matched
    `"UNIQUE"` in the error would pass with the index deleted — it would be
    measuring the trigger's message, not the constraint. This asserts the
    index exists by name in `sqlite_master` *and*, with the trigger dropped
    on a throwaway database, that the index alone still refuses the insert
    with SQLite's own native wording. The two layers are not redundant: the
    trigger holds under `INSERT OR REPLACE`, where the index's conflict
    resolution would delete the incumbent successor instead of raising
    (`test_insert_or_replace_cannot_destroy_a_successor_by_colliding_on_supersedes`).
    """
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    try:
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            ("memories_one_successor_per_predecessor",),
        ).fetchone()
        assert index_sql is not None
        assert "UNIQUE" in index_sql[0].upper()
        assert "supersedes" in index_sql[0]

        project_id, session_id = _seed_project_and_session(conn)
        predecessor_id = _write_evidenced_memory(
            conn, project_id, session_id, "fixture: a conclusion to be corrected once", 4
        )
        supersede_memory(
            conn,
            predecessor_id=predecessor_id,
            statement="fixture: the single permitted correction",
            origin="observer",
            tier=4,
            evidence=[_seed_anchor(conn, session_id)],
        )
        conn.commit()

        # One reservation serves both attempts: neither insert lands, so the
        # id stays free and its evidence row is cleaned up on the way out.
        with _reserved_memory_id(conn, session_id) as rival_id:
            rival_correction = (
                rival_id,
                project_id,
                session_id,
                "fixture: a rival correction",
                "observer",
                4,
                predecessor_id,
            )
            insert_rival = (
                "INSERT INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            )

            # With both layers present, the trigger answers first.
            with pytest.raises(sqlite3.IntegrityError, match="at most one successor"):
                conn.execute(insert_rival, rival_correction)

            # With the trigger gone, the index alone still refuses it — in
            # SQLite's own words, which no trigger in this schema produces.
            conn.execute("DROP TRIGGER memories_one_successor_guard")
            with pytest.raises(
                sqlite3.IntegrityError, match=r"UNIQUE constraint failed: memories\.supersedes"
            ):
                conn.execute(insert_rival, rival_correction)
    finally:
        conn.close()


@pytest.mark.inv4
def test_supersede_guards_are_added_by_migration_5_not_migration_1(tmp_path):
    """The supersession guarantees come from migration 5, which an existing store receives.

    Every other test here migrates a fresh database straight to the latest
    version, which cannot tell "the guard exists" from "the guard was added
    by a migration an already-created store will actually get". This
    migrates only through version 4, proves the attacks genuinely succeed
    there — the same measurements that motivated this task, re-run as
    positive controls — then applies migration 5 to that same populated
    database and shows they now raise.

    Guards against the silent trap of folding this DDL into
    `_V1_STATEMENTS`: the suite builds fresh databases from the latest
    schema, so that edit passes every other test in this file while leaving
    every store created before it permanently unprotected.
    """
    db_path = tmp_path / "palaver.db"
    # `target_version=4` rather than a filtered migration tuple: the runner
    # still holds the real SCHEMA_MIGRATIONS and merely stops early, which is
    # what a store created before migration 5 shipped actually looks like.
    assert migrate(db_path, target_version=4) == 4

    conn = connect(db_path)
    try:
        views = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        }
        assert "superseded_memories" not in views

        project_id, session_id = _seed_project_and_session(conn)
        # Parent-first: at version 4 the evidence foreign key is immediate,
        # so `write_memory`'s child-first order cannot run against this store.
        memory_id = _seed_evidenced_memory_pre_v8(
            conn, project_id, session_id, "fixture: a pre-migration-5 observer inference", 4
        )
        conn.commit()

        # Positive control: at version 4 the REPLACE attack genuinely works,
        # silently rewriting tier 4 to tier 1 through the migration-3 trigger.
        conn.execute(
            "INSERT OR REPLACE INTO memories(id, project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                memory_id,
                project_id,
                session_id,
                "fixture: rewritten before migration 5 existed",
                "attacker",
                TIER_USER_INSTRUCTION,
            ),
        )
        conn.commit()
        assert (
            conn.execute("SELECT tier FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
            == TIER_USER_INSTRUCTION
        )
    finally:
        conn.close()

    migrate(db_path)  # apply migration 5

    conn = connect(db_path)
    try:
        # LATEST_VERSION, not the literal 5: this asserts the store migrated
        # all the way forward, and every migration added after this test was
        # written would otherwise break an assertion that has nothing to do
        # with what it is testing.
        assert current_version(conn) == LATEST_VERSION
        assert conn.execute("SELECT * FROM superseded_memories").fetchall() == []

        with pytest.raises(sqlite3.IntegrityError, match="never reused"):
            conn.execute(
                "INSERT OR REPLACE INTO memories"
                "(id, project_id, session_id, statement, origin, tier) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    project_id,
                    session_id,
                    "fixture: the same attack after migration 5",
                    "attacker",
                    TIER_OBSERVER_INFERENCE,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

        # Positive control: legitimate supersession works post-migration, on
        # this same already-populated store.
        project_id, session_id = conn.execute(
            "SELECT project_id, session_id FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        successor_id = supersede_memory(
            conn,
            predecessor_id=memory_id,
            statement="fixture: a correction written after migration 5",
            origin="user",
            tier=TIER_USER_INSTRUCTION,
            evidence=[_seed_anchor(conn, session_id)],
        )
        conn.commit()
        assert successor_of(conn, memory_id) == successor_id
        # The derived view now names the predecessor, on the upgraded store.
        assert conn.execute("SELECT memory_id FROM superseded_memories").fetchall() == [
            (memory_id,)
        ]

        # A second successor for that same predecessor is refused here too.
        with _reserved_memory_id(conn, session_id) as rival_id:
            second_correction = (
                rival_id,
                project_id,
                session_id,
                "fixture: a rival correction after migration 5",
                "observer",
                TIER_USER_INSTRUCTION,
                memory_id,
            )
            insert_rival = (
                "INSERT INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            with pytest.raises(sqlite3.IntegrityError, match="at most one successor"):
                conn.execute(insert_rival, second_correction)
            # ...and the refusal survives losing the trigger, because migration 5
            # gave this upgraded store the real partial unique index as well.
            conn.execute("DROP TRIGGER memories_one_successor_guard")
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
                conn.execute(insert_rival, second_correction)
    finally:
        conn.close()


def test_migration_5_rolls_back_a_store_that_already_has_two_rows_that_supersede_one_memory(
    tmp_path,
):
    """Migration 5 fails loudly, and rolls back, on a store the new uniqueness rejects.

    A store written before migration 5 could hold two successors for one
    predecessor, which `CREATE UNIQUE INDEX` cannot accept. Fresh-database
    tests structurally cannot see this. The runner's `VACUUM INTO` rollback
    is what keeps that failure recoverable: the database must come back at
    version 4 with both rows intact, not half-migrated.

    Positive control: an otherwise identical store *without* duplicates
    migrates to version 5 cleanly, so the failure is caused by the duplicate
    data and not by migration 5 being broken.
    """
    db_path = tmp_path / "duplicated.db"
    pre_v5 = tuple(m for m in SCHEMA_MIGRATIONS if m.version <= 4)
    migrate(db_path, migrations=pre_v5)

    conn = connect(db_path)
    try:
        project_id, session_id = _seed_project_and_session(conn)
        predecessor_id = _seed_evidenced_memory_pre_v8(
            conn, project_id, session_id, "fixture: a doubly-corrected conclusion", 4
        )
        for label in ("first", "second"):
            conn.execute(
                "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    session_id,
                    f"fixture: the {label} rival correction",
                    "observer",
                    4,
                    predecessor_id,
                ),
            )
        conn.commit()
        rows_before = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
    finally:
        conn.close()

    with pytest.raises(MigrationError, match="version 5"):
        migrate(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == 4
        assert conn.execute("SELECT * FROM memories ORDER BY id").fetchall() == rows_before
    finally:
        conn.close()

    # Positive control: the same migration succeeds on a store with no duplicates.
    clean_path = tmp_path / "clean.db"
    migrate(clean_path, migrations=pre_v5)
    clean = connect(clean_path)
    try:
        project_id, session_id = _seed_project_and_session(clean)
        predecessor_id = _seed_evidenced_memory_pre_v8(
            clean, project_id, session_id, "fixture: a singly-corrected conclusion", 4
        )
        # This store does migrate all the way to the latest version below, so
        # its successor needs evidence too: migration 8 refuses to grandfather
        # a memory that has none, whichever version wrote it.
        correction_id = clean.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, session_id, "fixture: its one correction", "observer", 4, predecessor_id),
        ).lastrowid
        _seed_evidence_for(clean, session_id, correction_id)
        clean.commit()
    finally:
        clean.close()

    assert migrate(clean_path) == LATEST_VERSION
    clean = connect(clean_path)
    try:
        assert clean.execute("SELECT memory_id FROM superseded_memories").fetchall() == [
            (predecessor_id,)
        ]
    finally:
        clean.close()
