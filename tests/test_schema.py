"""Tests for the SQLite schema, migration runner, and FTS5 search.

Every database lives under pytest's `tmp_path`; none of these tests touch
the repository root or the current working directory.
"""

import importlib
import sqlite3

import pytest

from palaver.store.migrate import MigrationError, connect, current_version, migrate
from palaver.store.schema import SCHEMA_MIGRATIONS, Migration, search

# palaver/store/__init__.py re-exports the migrate() function under the name
# `migrate`, which shadows the submodule `palaver.store.migrate` on the
# package object — `import palaver.store.migrate as x` resolves through that
# same shadowed attribute, so it would hand monkeypatch the function instead
# of the module. importlib.import_module reads sys.modules directly instead.
migrate_module = importlib.import_module("palaver.store.migrate")

EXPECTED_TABLES = {
    "projects",
    "sessions",
    "transcript_chunks",
    "events",
    "memories",
    "memory_evidence",
    "memory_relationships",
    "current_state",
    "model_runs",
}


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row[0] for row in rows}


def _insert_project(conn, name="proj", path="/tmp/proj"):
    cursor = conn.execute("INSERT INTO projects(name, path) VALUES (?, ?)", (name, path))
    return cursor.lastrowid


def _seed_evidence_chunk(conn, project_id):
    """A transcript chunk for fixture evidence to anchor into.

    Reuses one session per project and numbers chunks from what is already
    stored, so repeated calls stay inside `sessions`' and
    `transcript_chunks`' UNIQUE constraints.
    """
    session = conn.execute(
        "SELECT id FROM sessions WHERE project_id = ? AND external_id = ?",
        (project_id, "evidence-source"),
    ).fetchone()
    if session is None:
        session_id = conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
            (project_id, "fixture", "evidence-source"),
        ).lastrowid
    else:
        session_id = session[0]
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM transcript_chunks WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    return conn.execute(
        "INSERT INTO transcript_chunks(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
        (session_id, seq, "user", "fixture: an invented line for evidence to point at"),
    ).lastrowid


def _insert_memory(conn, project_id, statement, tier=3, origin="observer"):
    """Insert a memory in the order migration 8 requires: evidence first.

    Since migration 8 the database refuses a `memories` row that no
    `memory_evidence` row already names, and it will not accept an id SQLite
    picks at insert time either — `NEW.id` is not yet the rowid inside a
    BEFORE INSERT trigger. So the id is reserved here and stated explicitly,
    exactly as `palaver.memory.write.write_memory` does.
    """
    memory_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()[0]
    conn.execute(
        "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
        "VALUES (?, ?, ?, ?)",
        (memory_id, _seed_evidence_chunk(conn, project_id), 0, 7),
    )
    conn.execute(
        "INSERT INTO memories(id, project_id, statement, origin, tier) VALUES (?, ?, ?, ?, ?)",
        (memory_id, project_id, statement, origin, tier),
    )
    return memory_id


def test_migrate_creates_all_v1_tables(tmp_path):
    """Migrating a fresh database creates all nine schema v1 tables."""
    db_path = tmp_path / "palaver.db"

    migrate(db_path)

    conn = connect(db_path)
    try:
        assert EXPECTED_TABLES.issubset(_table_names(conn))
    finally:
        conn.close()


def test_migrate_reaches_latest_version(tmp_path):
    """migrate() with no target defaults to the highest known version."""
    db_path = tmp_path / "palaver.db"

    reached = migrate(db_path)

    assert reached == max(m.version for m in SCHEMA_MIGRATIONS)


def test_journal_mode_is_wal(tmp_path):
    """The database is configured for WAL journaling."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()

    assert mode == "wal"


def test_bak0_backup_created_on_first_migration(tmp_path):
    """The very first migration's rollback point is named .bak-0 (from_version=0)."""
    db_path = tmp_path / "palaver.db"

    migrate(db_path)

    assert (tmp_path / "palaver.db.bak-0").exists()


def test_fts5_search_finds_hit_inserted_after_migration(tmp_path):
    """A memory written after the FTS5 index exists is immediately searchable."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        _insert_memory(conn, project_id, "the deploy script needs a dryrun flag")
        conn.commit()

        results = search(conn, "dryrun")
    finally:
        conn.close()

    assert any(r["source"] == "memories" and "dryrun" in r["text"] for r in results)


def test_search_tags_hits_with_their_source_table(tmp_path):
    """search() unifies hits from all three FTS indexes, each tagged by source."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        session_id = conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
            (project_id, "claude-code", "sess-1"),
        ).lastrowid
        conn.execute(
            "INSERT INTO transcript_chunks(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
            (session_id, 1, "user", "please rotate the widget zzyxmarker"),
        )
        conn.execute(
            "INSERT INTO events(session_id, kind, payload) VALUES (?, ?, ?)",
            (session_id, "tool_result", "widget zzyxmarker rotated successfully"),
        )
        _insert_memory(conn, project_id, "user asked to rotate the widget zzyxmarker")
        conn.commit()

        results = search(conn, "zzyxmarker")
    finally:
        conn.close()

    sources = {r["source"] for r in results}
    assert sources == {"transcript_chunks", "events", "memories"}


def test_fts5_backfills_preexisting_rows(tmp_path):
    """Rows written before the FTS5 index exists are searchable once it is created.

    External-content FTS5 triggers only capture writes made after they are
    created; a migration that adds the index without an explicit backfill
    would leave this row unsearchable. Inserting before the index-creating
    migration, rather than after, is what makes this test fail without the
    backfill INSERT in schema.py's migration 2.
    """
    db_path = tmp_path / "palaver.db"
    v1_only = tuple(m for m in SCHEMA_MIGRATIONS if m.version == 1)

    migrate(db_path, migrations=v1_only)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        # Raw, and parent-first: at version 1 there is no evidence trigger to
        # satisfy, and `memory_evidence` still has the `quote` column that
        # migration 4 removes.
        memory_id = conn.execute(
            "INSERT INTO memories(project_id, statement, origin, tier) VALUES (?, ?, ?, ?)",
            (project_id, "backfilled memory about the archived spike notes", "observer", 3),
        ).lastrowid
        conn.commit()
    finally:
        conn.close()

    # Now apply the FTS5-creating migration against the already-populated
    # database. Stopping at 7 first: migration 4 rebuilds `memory_evidence`
    # and drops every quote-shaped row with it, so this memory reaches
    # migration 8 unanchored unless it is re-anchored in the new shape —
    # which migration 8 refuses to grandfather rather than quietly accept.
    migrate(db_path, target_version=7)

    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, _seed_evidence_chunk(conn, project_id), 0, 7),
        )
        conn.commit()
    finally:
        conn.close()

    migrate(db_path, migrations=SCHEMA_MIGRATIONS)

    conn = connect(db_path)
    try:
        results = search(conn, "archived")
    finally:
        conn.close()

    assert any(r["source"] == "memories" and "archived" in r["text"] for r in results)


def test_migration_failure_restores_backup_and_row_count(tmp_path):
    """A migration that fails partway is rolled back to the pre-attempt row count."""
    db_path = tmp_path / "palaver.db"
    v1 = SCHEMA_MIGRATIONS[0]
    migrate(db_path, migrations=(v1,))

    conn = connect(db_path)
    try:
        _insert_project(conn, name="p1", path="/tmp/p1")
        _insert_project(conn, name="p2", path="/tmp/p2")
        _insert_project(conn, name="p3", path="/tmp/p3")
        conn.commit()
        pre_count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    finally:
        conn.close()
    assert pre_count == 3

    broken_v2 = Migration(
        version=2,
        description="broken migration for testing rollback",
        statements=(
            # This INSERT commits under autocommit before the later statement fails,
            # so a correct restore must undo it too, not just the schema change.
            "INSERT INTO projects(name, path) VALUES ('temp-during-migration', '/tmp/temp')",
            "CREATE TABLE this_will_fail_twice (id INTEGER)",
            "CREATE TABLE this_will_fail_twice (id INTEGER)",
        ),
    )

    with pytest.raises(MigrationError):
        migrate(db_path, migrations=(v1, broken_v2))

    assert (tmp_path / "palaver.db.bak-1").exists()

    conn = connect(db_path)
    try:
        assert current_version(conn) == 1
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == pre_count
        assert "this_will_fail_twice" not in _table_names(conn)
    finally:
        conn.close()


def test_migration_raises_when_backup_absent_before_ddl(tmp_path, monkeypatch):
    """If the backup file never gets created, the runner raises before any DDL runs."""
    db_path = tmp_path / "palaver.db"

    monkeypatch.setattr(migrate_module, "_take_backup", lambda db_path, backup_path: None)

    with pytest.raises(MigrationError, match="missing"):
        migrate(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == 0
        assert "projects" not in _table_names(conn)
    finally:
        conn.close()


def test_migration_raises_when_backup_corrupt_before_ddl(tmp_path, monkeypatch):
    """If the backup file fails to open as a database, the runner raises before any DDL runs."""
    db_path = tmp_path / "palaver.db"

    def _write_garbage_backup(db_path, backup_path):
        backup_path.write_bytes(b"not a real sqlite file")

    monkeypatch.setattr(migrate_module, "_take_backup", _write_garbage_backup)

    with pytest.raises(MigrationError, match="unreadable"):
        migrate(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == 0
        assert "projects" not in _table_names(conn)
    finally:
        conn.close()


def test_migrate_is_idempotent_when_already_at_target(tmp_path):
    """Calling migrate() again once at the latest version applies nothing new."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        conn.commit()
    finally:
        conn.close()

    second_result = migrate(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == second_result
        assert conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
    finally:
        conn.close()


def test_memory_evidence_requires_a_transcript_or_event_link(tmp_path):
    """The memory_evidence CHECK constraint rejects a row with neither link set."""
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        memory_id = _insert_memory(conn, project_id, "orphan evidence attempt")

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO memory_evidence(memory_id, start_offset, end_offset) VALUES (?, ?, ?)",
                (memory_id, 0, 5),
            )
    finally:
        conn.close()


def _seed_v7_memory_with_evidence(conn, project_id, statement, evidence_count=1):
    """Seed a memory and its evidence in the pre-migration-8, parent-first order.

    Below version 8 `memory_evidence.memory_id` is an immediate foreign key,
    so the child cannot name a parent that does not exist yet — the reverse
    of the order the upgraded schema requires. Returns the memory's id.
    """
    memory_id = conn.execute(
        "INSERT INTO memories(project_id, statement, origin, tier) VALUES (?, ?, ?, ?)",
        (project_id, statement, "observer", 3),
    ).lastrowid
    for _ in range(evidence_count):
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
            "VALUES (?, ?, ?, ?)",
            (memory_id, _seed_evidence_chunk(conn, project_id), 0, 7),
        )
    return memory_id


def test_migration_8_carries_existing_evidence_rows_forward_with_their_ids(tmp_path):
    """The `memory_evidence` rebuild copies every row across unchanged, ids included.

    Migration 8 has to rebuild the table to make the child foreign key
    deferrable, and SQLite has no `ALTER TABLE` that changes a constraint in
    place. A rebuild that renumbered rows would be silently destructive in a
    way no schema assertion would catch: `palaver.memory.evidence.
    resolve_evidence` takes a `memory_evidence.id`, so every anchor a caller
    already holds would resolve to the wrong span, or to nothing.

    Also covers the one-to-many shape surviving the rebuild — one memory
    here carries three evidence rows, which a primary-anchor redesign could
    not have represented at all.
    """
    db_path = tmp_path / "palaver.db"
    assert migrate(db_path, target_version=7) == 7

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        single_id = _seed_v7_memory_with_evidence(conn, project_id, "one anchor", 1)
        multi_id = _seed_v7_memory_with_evidence(conn, project_id, "three anchors", 3)
        conn.commit()
        before = conn.execute("SELECT * FROM memory_evidence ORDER BY id").fetchall()
        memories_before = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
    finally:
        conn.close()
    assert len(before) == 4

    assert migrate(db_path) == max(m.version for m in SCHEMA_MIGRATIONS)

    conn = connect(db_path)
    try:
        assert conn.execute("SELECT * FROM memory_evidence ORDER BY id").fetchall() == before
        assert conn.execute("SELECT * FROM memories ORDER BY id").fetchall() == memories_before
        counts = dict(
            conn.execute(
                "SELECT memory_id, COUNT(*) FROM memory_evidence GROUP BY memory_id"
            ).fetchall()
        )
        assert counts == {single_id: 1, multi_id: 3}
        # The rebuilt table is the one now named memory_evidence, and it is
        # the deferrable one — asserted against the stored DDL, not inferred
        # from the copy having worked.
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_evidence'"
        ).fetchone()[0]
        assert "DEFERRABLE INITIALLY DEFERRED" in table_sql
        assert "memory_evidence_v8" not in _table_names(conn)
    finally:
        conn.close()


def test_migration_8_refuses_a_store_holding_a_memory_with_no_evidence(tmp_path):
    """An already-invalid store is reported and rolled back, never grandfathered.

    Triggers only see writes made after they exist, so a migration that
    added the evidence rule and stopped there would leave every unevidenced
    row already on disk permanently exempt — INV-6 true for new memories and
    quietly false for old ones. Migration 8 inventories them first and
    aborts, and the runner's `VACUUM INTO` rollback puts the store back at
    version 7 with both rows intact so the operator can fix the data.

    Positive control: the same migration succeeds against a store whose
    memories are all evidenced, so the abort is caused by the invalid row
    and not by migration 8 being broken.
    """
    db_path = tmp_path / "palaver.db"
    assert migrate(db_path, target_version=7) == 7

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        evidenced_id = _seed_v7_memory_with_evidence(conn, project_id, "properly anchored")
        unevidenced_id = conn.execute(
            "INSERT INTO memories(project_id, statement, origin, tier) VALUES (?, ?, ?, ?)",
            (project_id, "written before the rule existed", "observer", 3),
        ).lastrowid
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(MigrationError, match="version 8"):
        migrate(db_path)

    conn = connect(db_path)
    try:
        assert current_version(conn) == 7
        assert {row[0] for row in conn.execute("SELECT id FROM memories").fetchall()} == {
            evidenced_id,
            unevidenced_id,
        }
        # The scratch table the guard raises through leaves no residue.
        assert "memories_missing_evidence" not in _table_names(conn)
    finally:
        conn.close()

    # Positive control: anchor the offending row and the same migration runs.
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
            "VALUES (?, ?, ?, ?)",
            (unevidenced_id, _seed_evidence_chunk(conn, project_id), 0, 7),
        )
        conn.commit()
    finally:
        conn.close()

    assert migrate(db_path) == max(m.version for m in SCHEMA_MIGRATIONS)


def test_memory_evidence_child_foreign_key_is_deferred_not_dropped(tmp_path):
    """Evidence may name a memory that does not exist yet — until the commit.

    This is what makes the child-first write order legal, and it is only
    half a guarantee on its own: deferral moves the foreign key check to
    `COMMIT`, which is why the evidence *rule* is a trigger instead (see
    `tests/test_memory.py`). What this asserts is the other half — the
    constraint was made deferrable, not quietly dropped, so an orphan is
    still refused rather than committed.
    """
    db_path = tmp_path / "palaver.db"
    migrate(db_path)

    conn = connect(db_path)
    try:
        project_id = _insert_project(conn)
        chunk_id = _seed_evidence_chunk(conn, project_id)
        conn.commit()

        # Inside the transaction the orphan is accepted: that is the deferral.
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
            "VALUES (?, ?, ?, ?)",
            (999_999, chunk_id, 0, 7),
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.commit()
        conn.rollback()

        # Positive control: the same insert commits once its parent exists.
        memory_id = _insert_memory(conn, project_id, "a memory whose evidence is written first")
        conn.commit()
        assert conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0] == 1
    finally:
        conn.close()
