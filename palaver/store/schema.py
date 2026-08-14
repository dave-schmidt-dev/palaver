"""SQLite schema v1 for Palaver's local store.

Migration 1 creates the nine core tables. Migration 2 adds FTS5
external-content search over `transcript_chunks`, `events`, and `memories`.
Migration 3 adds `memories_tier_immutable`, a trigger enforcing INV-5's
"tier is assigned at insert and never changes" rule at the database layer
(see `palaver/memory/write.py` for why a Python-only guard is not enough).
Migration 4 rebuilds `memory_evidence` to carry `start_offset`/`end_offset`
span anchors instead of a copied `quote` string (see
`palaver/memory/evidence.py`). Migration 5 makes supersession a derived
view rather than a stored flag, and closes the `REPLACE`-shaped holes
through which a row could still be destroyed or rewritten (see
`palaver/memory/supersede.py`).

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
    # supersession to never mutate or delete a row, so current status is
    # derived from whether some other row's `supersedes` points at it —
    # migration 5's `superseded_memories` view. `tier` encodes the INV-5
    # provenance ordering (1 = highest, explicit user instruction, down to
    # 5 = observer speculation); migration 3 below adds the trigger that
    # makes `tier` immutable once written, and migration 5 adds the rule
    # that a lower-confidence tier can never *supersede* a higher-confidence
    # one, per
    # tests/test_memory.py::test_lower_tier_cannot_supersede_higher_tier.
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

# INV-5: tier is assigned once, at insert, and never changes. A Python-only
# guard (e.g. in palaver/memory/write.py) is bypassed by the next module
# that opens its own connection to the same database file, so the actual
# enforcement is this trigger, not application code. `BEFORE UPDATE OF tier`
# fires whenever an UPDATE statement's SET clause names `tier` at all —
# including a no-op UPDATE that sets it to its existing value — so there is
# no shape of UPDATE that can touch this column and succeed. It is
# deliberately scoped to `tier` alone: full-row immutability for a
# superseded predecessor (any column, once superseded) is task 2.4's
# trigger, layered on top of this one, not a replacement for it.
_V3_STATEMENTS = (
    """
    CREATE TRIGGER memories_tier_immutable
    BEFORE UPDATE OF tier ON memories
    BEGIN
        SELECT RAISE(ABORT, 'memories.tier is immutable; insert a superseding row instead (INV-5)');
    END
    """,
)

# INV-6: a memory's evidence is a pointer into stored raw transcript, never a
# copied string — task 2.2 (palaver/memory/evidence.py). `quote` is replaced
# by `start_offset`/`end_offset`, a span resolved live against the current
# content of the referenced transcript_chunks or events row, so a quote can
# never silently drift from its source between write time and read time.
# SQLite has no ALTER TABLE that swaps a NOT NULL TEXT column for two NOT
# NULL INTEGER columns in place, so this drops and recreates the table under
# its own name rather than copying rows forward: any memory_evidence row
# written before this migration carries only a `quote` string, with no
# offsets to backfill from, so there is no lossless copy path — and no real
# deployment exists yet for this pre-release project to make that a live
# concern. The `transcript_chunk_id IS NOT NULL OR event_id IS NOT NULL`
# CHECK carries forward unchanged from migration 1.
_V4_STATEMENTS = (
    "DROP TABLE memory_evidence",
    f"""
    CREATE TABLE memory_evidence (
        id INTEGER PRIMARY KEY,
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        transcript_chunk_id INTEGER REFERENCES transcript_chunks(id),
        event_id INTEGER REFERENCES events(id),
        start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
        end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
        created_at TEXT NOT NULL {_CREATED_AT_DEFAULT},
        CHECK (transcript_chunk_id IS NOT NULL OR event_id IS NOT NULL)
    )
    """,
)

# INV-4/INV-5, task 2.4: supersession is a derived view, never a stored flag.
# A stored `superseded` boolean on the predecessor would mean writing to a row
# INV-4 declares immutable, so the successor row carries the whole edge
# (`memories.supersedes`) and `superseded_memories` derives the predecessor set
# from it.
#
# Uniqueness is a partial index, not a rebuilt table. Adding a table-level
# `UNIQUE (supersedes)` to an existing SQLite table means the 12-step
# ALTER-TABLE rebuild: create a new table, copy every row, drop the old one,
# rename. That would DROP a table holding live `memories` rows — the exact
# operation INV-4 exists to forbid — and would invalidate the migration-2 FTS5
# external-content index, whose `content_rowid` points at those rowids. A
# `CREATE UNIQUE INDEX ... WHERE supersedes IS NOT NULL` gives the identical
# guarantee (SQLite already treats NULLs as distinct in a UNIQUE index, so the
# partial predicate only keeps the index small and its intent explicit) with no
# rebuild, no row copy, and no FTS invalidation.
#
# The trigger set below closes three `REPLACE`-shaped holes, all of them
# measured against a real migrated store rather than reasoned about:
#
#   1. `INSERT OR REPLACE` on an existing `memories.id` silently rewrote tier
#      from 4 to 1. `REPLACE` is a DELETE followed by an INSERT, so the
#      migration-3 `BEFORE UPDATE OF tier` trigger never fires.
#   2. `INSERT OR REPLACE` conflicting on the new `supersedes` unique index
#      deleted the existing successor row and took its place.
#   3. `UPDATE OR REPLACE ... SET rowid = <other row's id>` deleted the row it
#      collided with, and a trigger written as `BEFORE UPDATE OF id` does not
#      fire for the `rowid`/`_rowid_` spelling of the same column.
#
# A `BEFORE DELETE` trigger does NOT close any of them on its own: measured,
# `memories_no_delete` blocks a plain `DELETE` but does not fire during
# REPLACE conflict resolution unless `PRAGMA recursive_triggers` is ON, and
# that pragma is per connection — the next module that opens its own
# connection would silently escape the guarantee. So the enforcement is
# `BEFORE INSERT`/`BEFORE UPDATE` triggers, which fire regardless of any
# pragma; `memories_no_delete` is kept for the plain-DELETE case it does
# cover. For the same reason `memories_supersedes_must_exist` exists rather
# than leaning on the `REFERENCES memories(id)` foreign key: FK enforcement
# needs `PRAGMA foreign_keys=ON`, and on a connection without it a `supersedes`
# naming no row was accepted — which also made the tier comparison below
# vacuous, since `NEW.tier > (SELECT tier FROM ... )` is NULL, not true, when
# the subquery finds nothing.
_V5_STATEMENTS = (
    """
    CREATE UNIQUE INDEX memories_one_successor_per_predecessor
    ON memories(supersedes) WHERE supersedes IS NOT NULL
    """,
    # The derived view. One row per superseded predecessor; no DISTINCT is
    # needed because the index above already makes duplicates impossible.
    # Column named `memory_id` rather than `supersedes` because from the
    # view's side it *is* a memory id, and `SELECT ... WHERE supersedes = ?`
    # against a view of predecessors reads like the opposite of what it means.
    """
    CREATE VIEW superseded_memories AS
    SELECT supersedes AS memory_id FROM memories WHERE supersedes IS NOT NULL
    """,
    """
    CREATE TRIGGER memories_no_delete
    BEFORE DELETE ON memories
    BEGIN
        SELECT RAISE(ABORT, 'memories rows are append-only; supersede, never delete (INV-4)');
    END
    """,
    """
    CREATE TRIGGER memories_id_never_reused
    BEFORE INSERT ON memories
    WHEN EXISTS (SELECT 1 FROM memories WHERE id = NEW.id)
    BEGIN
        SELECT RAISE(ABORT, 'memories.id is never reused; REPLACE would destroy a row (INV-4)');
    END
    """,
    # Written against NEW.id/OLD.id rather than `BEFORE UPDATE OF id`, so it
    # fires for the `rowid` and `_rowid_` spellings of the same column too.
    """
    CREATE TRIGGER memories_id_immutable
    BEFORE UPDATE ON memories
    WHEN NEW.id IS NOT OLD.id
    BEGIN
        SELECT RAISE(ABORT, 'memories.id is immutable; a rowid rewrite destroys a row (INV-4)');
    END
    """,
    """
    CREATE TRIGGER memories_supersedes_immutable
    BEFORE UPDATE OF supersedes ON memories
    BEGIN
        SELECT RAISE(ABORT, 'memories.supersedes is immutable; write a successor row (INV-4)');
    END
    """,
    """
    CREATE TRIGGER memories_supersedes_must_exist
    BEFORE INSERT ON memories
    WHEN NEW.supersedes IS NOT NULL
     AND NOT EXISTS (SELECT 1 FROM memories WHERE id = NEW.supersedes)
    BEGIN
        SELECT RAISE(ABORT, 'memories.supersedes must name an existing memory (INV-4)');
    END
    """,
    # Says UNIQUE in its message because it enforces the same rule as the
    # partial index above, one step earlier: a BEFORE INSERT trigger fires
    # even under `INSERT OR REPLACE`, where the index's own conflict
    # resolution would delete the incumbent successor instead of raising.
    """
    CREATE TRIGGER memories_one_successor_guard
    BEFORE INSERT ON memories
    WHEN NEW.supersedes IS NOT NULL
     AND EXISTS (SELECT 1 FROM memories WHERE supersedes = NEW.supersedes)
    BEGIN
        SELECT RAISE(ABORT, 'UNIQUE constraint: a memory has at most one successor (INV-4)');
    END
    """,
    """
    CREATE TRIGGER memories_supersedes_tier_order
    BEFORE INSERT ON memories
    WHEN NEW.supersedes IS NOT NULL
     AND NEW.tier > (SELECT tier FROM memories WHERE id = NEW.supersedes)
    BEGIN
        SELECT RAISE(ABORT, 'a lower-confidence tier may not supersede a higher one (INV-5)');
    END
    """,
    # Tests `memories` directly rather than the `superseded_memories` view: an
    # INV-4 guard that depends on a view definition is one view edit away from
    # being silently disabled.
    """
    CREATE TRIGGER memories_superseded_row_immutable
    BEFORE UPDATE ON memories
    WHEN EXISTS (SELECT 1 FROM memories WHERE supersedes = OLD.id)
    BEGIN
        SELECT RAISE(ABORT, 'a superseded memory is immutable; see its successor (INV-4)');
    END
    """,
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
    Migration(
        version=3,
        description="memories_tier_immutable trigger: tier can never be UPDATEd (INV-5)",
        statements=_V3_STATEMENTS,
    ),
    Migration(
        version=4,
        description=(
            "memory_evidence rebuilt: start_offset/end_offset span anchors replace the "
            "copied quote column (INV-6)"
        ),
        statements=_V4_STATEMENTS,
    ),
    Migration(
        version=5,
        description=(
            "Supersession as a derived view: one-successor-per-predecessor unique index, "
            "superseded_memories view, and the triggers that make a memories row "
            "undeletable and unrewritable including via REPLACE (INV-4/INV-5)"
        ),
        statements=_V5_STATEMENTS,
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
