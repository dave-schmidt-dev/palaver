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


def _insert_memory(conn, project_id, statement, tier=3, origin="observer"):
    cursor = conn.execute(
        "INSERT INTO memories(project_id, statement, origin, tier) VALUES (?, ?, ?, ?)",
        (project_id, statement, origin, tier),
    )
    return cursor.lastrowid


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
        _insert_memory(conn, project_id, "backfilled memory about the archived spike notes")
        conn.commit()
    finally:
        conn.close()

    # Now apply the FTS5-creating migration against the already-populated database.
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
