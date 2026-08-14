"""SQLite schema v1 for Palaver's local store.

Migration 1 creates the nine core tables. Migration 2 adds FTS5
external-content search over `transcript_chunks`, `events`, and `memories`.

FTS5's `content=` option names exactly one source table, view, or virtual
table — a single external-content index cannot span three tables, and a
UNION view has no stable rowid to serve as `content_rowid`, which is the
condition FTS5's own documentation says produces unpredictable query
results. So this module builds three independent external-content indexes
instead, one per source table, and `search()` below is the single entry
point that queries all three and returns unified, source-tagged results.

External-content triggers only capture writes made after they exist; rows
already in a content table are not backfilled automatically. Migration 2
therefore ends each index's statements with an explicit
`INSERT INTO ..._fts(rowid, col) SELECT id, col FROM ...` so rows written
under migration 1 (or any migration before the index existed) are
searchable immediately, not just newly-written ones.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_CREATED_AT_DEFAULT = "DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"


@dataclass(frozen=True)
class Migration:
    """One schema migration: a monotonic version and its DDL statements.

    Statements run one at a time under autocommit (see
    `palaver.store.migrate.migrate`), so a failure partway through a
    migration can leave a subset already committed to disk. That is exactly
    the case the runner's VACUUM INTO backup exists to recover from — the
    statements are deliberately not wrapped in one transaction here.
    """

    version: int
    description: str
    statements: tuple[str, ...]


_V1_STATEMENTS = (
    f"""
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        path TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT}
    )
    """,
    f"""
    CREATE TABLE sessions (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id),
        source TEXT NOT NULL,
        external_id TEXT,
        started_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        ended_at TEXT,
        UNIQUE (source, external_id)
    )
    """,
    f"""
    CREATE TABLE transcript_chunks (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        UNIQUE (session_id, seq)
    )
    """,
    f"""
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        kind TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT}
    )
    """,
    # memories has no stored status/superseded column: INV-4 requires
    # supersession to never mutate or delete a row, and the plan's task 2.4
    # ("Supersession as a derived view, never a stored flag") derives current
    # status from whether some other row's `supersedes` points at it. `tier`
    # encodes the INV-5 provenance ordering (1 = highest, explicit user
    # instruction, down to 5 = observer speculation); the CHECK/trigger that
    # enforces a lower tier can never supersede a higher one is task 2.1's
    # responsibility in palaver/memory/, per the INV-5 gate test area.
    f"""
    CREATE TABLE memories (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id),
        session_id INTEGER REFERENCES sessions(id),
        statement TEXT NOT NULL,
        origin TEXT NOT NULL,
        tier INTEGER NOT NULL CHECK (tier BETWEEN 1 AND 5),
        supersedes INTEGER REFERENCES memories(id),
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT}
    )
    """,
    # INV-6: every memory carries at least one evidence link to stored raw
    # transcript. `quote` is NOT NULL and the CHECK requires a link to a
    # transcript_chunks row, an events row, or both; the substring-verified
    # quote-grounding check itself is task 2.2's job.
    f"""
    CREATE TABLE memory_evidence (
        id INTEGER PRIMARY KEY,
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        transcript_chunk_id INTEGER REFERENCES transcript_chunks(id),
        event_id INTEGER REFERENCES events(id),
        quote TEXT NOT NULL,
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        CHECK (transcript_chunk_id IS NOT NULL OR event_id IS NOT NULL)
    )
    """,
    f"""
    CREATE TABLE memory_relationships (
        id INTEGER PRIMARY KEY,
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        related_memory_id INTEGER NOT NULL REFERENCES memories(id),
        relationship TEXT NOT NULL,
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        CHECK (memory_id != related_memory_id)
    )
    """,
    # Regeneratable current-state summaries are stored separately from
    # durable memories precisely so INV-4's append-only rule can be absolute
    # on the memories table; this table is expected to be overwritten.
    f"""
    CREATE TABLE current_state (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL REFERENCES projects(id),
        session_id INTEGER REFERENCES sessions(id),
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        UNIQUE (project_id, session_id, key)
    )
    """,
    f"""
    CREATE TABLE model_runs (
        id INTEGER PRIMARY KEY,
        session_id INTEGER REFERENCES sessions(id),
        model TEXT NOT NULL,
        purpose TEXT NOT NULL,
        started_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        finished_at TEXT,
        status TEXT NOT NULL DEFAULT 'running'
    )
    """,
)


def _fts_statements(source_table: str, text_column: str) -> tuple[str, ...]:
    """Build the external-content FTS5 index, triggers, and backfill for one table."""
    fts_table = f"{source_table}_fts"
    return (
        f"""
        CREATE VIRTUAL TABLE {fts_table} USING fts5(
            {text_column}, content='{source_table}', content_rowid='id'
        )
        """,
        f"""
        CREATE TRIGGER {fts_table}_ai AFTER INSERT ON {source_table} BEGIN
            INSERT INTO {fts_table}(rowid, {text_column}) VALUES (new.id, new.{text_column});
        END
        """,
        f"""
        CREATE TRIGGER {fts_table}_ad AFTER DELETE ON {source_table} BEGIN
            INSERT INTO {fts_table}({fts_table}, rowid, {text_column})
            VALUES ('delete', old.id, old.{text_column});
        END
        """,
        f"""
        CREATE TRIGGER {fts_table}_au AFTER UPDATE ON {source_table} BEGIN
            INSERT INTO {fts_table}({fts_table}, rowid, {text_column})
            VALUES ('delete', old.id, old.{text_column});
            INSERT INTO {fts_table}(rowid, {text_column}) VALUES (new.id, new.{text_column});
        END
        """,
        # Backfill: rows already present when the index is created are not
        # captured by the triggers above, which only see writes from here on.
        f"INSERT INTO {fts_table}(rowid, {text_column}) "
        f"SELECT id, {text_column} FROM {source_table}",
    )


_FTS_SOURCES = (
    ("transcript_chunks", "content"),
    ("events", "payload"),
    ("memories", "statement"),
)

_V2_STATEMENTS = tuple(
    statement
    for source_table, text_column in _FTS_SOURCES
    for statement in _fts_statements(source_table, text_column)
)

SCHEMA_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description=(
            "Base schema: projects, sessions, transcript_chunks, events, memories, "
            "memory_evidence, memory_relationships, current_state, model_runs"
        ),
        statements=_V1_STATEMENTS,
    ),
    Migration(
        version=2,
        description=(
            "FTS5 external-content indexes over transcript_chunks, events, and "
            "memories, with backfill of pre-existing rows"
        ),
        statements=_V2_STATEMENTS,
    ),
)

LATEST_VERSION = max(migration.version for migration in SCHEMA_MIGRATIONS)


def search(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Search transcript_chunks, events, and memories, unified and source-tagged.

    A single external-content FTS5 index cannot span three source tables, so
    this queries the three independent indexes built by migration 2 and
    merges their hits. This is the one search entry point callers should use
    instead of querying an individual `*_fts` table directly.

    Args:
        conn: Open connection to a database migrated to at least version 2.
        query: An FTS5 MATCH query string, passed through unescaped. FTS5's
            query syntax treats an unquoted hyphen as the NOT operator even
            mid-word (`dry-run` parses as `dry NOT run`, not a single term),
            so callers searching for a literal hyphenated term must quote it,
            e.g. `'"dry-run"'`.
        limit: Maximum rows returned per source table (not overall).

    Returns:
        A list of {"source", "id", "text"} dicts, one per hit, grouped by
        source table in the fixed order transcript_chunks, events, memories.
    """
    results = []
    for source_table, text_column in _FTS_SOURCES:
        fts_table = f"{source_table}_fts"
        rows = conn.execute(
            f"SELECT rowid, {text_column} FROM {fts_table} "
            f"WHERE {fts_table} MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        for rowid, text in rows:
            results.append({"source": source_table, "id": rowid, "text": text})
    return results
