"""Tests for the append-only memory writer (2.1: INV-4/INV-5; 2.2: INV-6; 2.4: supersession).

Per the plan's standing rule, every negative assertion here is paired with a
positive control proving the same mechanism is live, not merely agreeing
with whatever the code already looks like. `tests/test_invariants.py`'s
INV-3 test is the reference for this pattern (prove *which* layer denied a
query); `test_update_tier_raises_at_the_database_layer` below follows it.

**INV-5 — tier is immutable.** `test_update_tier_raises_at_the_database_layer`
attacks a raw `sqlite3` connection with a hand-written `UPDATE`, never
calling into `palaver.memory.write`, so the failure proves the database
itself refuses the mutation — not a Python-level guard that a different
module opening its own connection could simply not go through.
`test_reclassification_writes_a_new_row_and_leaves_the_original_byte_identical`
is the companion positive path: a reclassification is a second row, and the
first is provably untouched.
`test_tier_immutable_trigger_is_added_by_migration_3_not_migration_1` proves
the guarantee comes from the versioned migration and not a v1-baked trigger,
by migrating an already-populated database from version 2 to version 3 and
showing the identical UPDATE only starts failing after that step. That test
seeds its version-2 `memories` row with raw SQL rather than `write_memory`,
because `write_memory` now targets the version-4 `memory_evidence` shape
(`start_offset`/`end_offset`) and cannot be called against an
older-than-version-4 database at all — a real constraint task 2.2 added,
not an oversight.

**INV-4 — no DELETE path.**
`test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module` statically
scans every `execute*` call under `palaver/memory/` for a `DELETE` or `DROP`
SQL string, with a positive control proving the detector is live. (This
does not extend to `palaver/store/schema.py`'s migrations 4 and 8, which
are allowed to `DROP TABLE memory_evidence` and recreate it — a schema
migration reshaping a table's columns or constraints is not the append-only
memory *write path* INV-4 governs, and each runs once per database.
Migration 8 copies every existing evidence row forward, ids included, and
neither migration ever rebuilds `memories` itself, which is the table INV-4
actually protects.)

**INV-6 — every memory carries at least one evidence link.**
`test_memory_without_evidence_is_rejected` is this invariant's charter gate
(named in `INVARIANTS.md`). It asserts `write_memory` itself raises when
called with no evidence anchors. This paragraph previously went on to
explain why that Python-level check was the *only* enforcement and why a
database-layer design had been rejected; migration 8 has since carried the
rule into the database, so that reasoning is corrected here rather than
left standing. The shape it landed in keeps what the rejected designs could
not: evidence rows are written first, naming a pre-reserved memory id that
migration 8's `DEFERRABLE INITIALLY DEFERRED` child foreign key permits, and
`memories_requires_evidence` — a `BEFORE INSERT` trigger, so it fires
whatever `PRAGMA foreign_keys` says and refuses at the write itself rather
than at `conn.commit()` — is what actually enforces.
`test_a_memory_with_no_evidence_is_refused_by_the_database_not_only_by_write_memory`
attacks that floor with raw SQL,
`test_evidence_and_mutation_rules_hold_on_a_connection_with_foreign_keys_off`
proves it does not depend on the pragma, and
`test_an_insert_letting_sqlite_pick_the_id_is_refused_even_with_evidence_at_minus_one`
pins the measured `NEW.id = -1` quirk that makes the explicit id mandatory.
A `memories.primary_evidence_id` FK remains rejected, for the reason it
always was: it restructures `memory_evidence` away from the 1-many
`memory_id` shape task 2.4's supersession work depends on. See
`palaver/memory/write.py`'s module docstring for the full comparison.
`test_write_memory_rejects_an_evidence_anchor_with_neither_chunk_nor_event_link`
is the companion proof for the *other* half of "at least one": even if a
caller assembles an `EvidenceAnchor` with neither id set, `write_memory`'s
`INSERT` still fails, because the schema's own CHECK constraint — written
in migration 1 and carried forward unchanged by both later rebuilds of the
table — rejects a `memory_evidence` row with both link columns NULL — the same
guarantee `resolve_evidence`'s otherwise-untested `else` branch assumes
holds.

The evidence itself is a pointer, not a copy: `EvidenceAnchor` names a
`(transcript_chunk_id | event_id, start_offset, end_offset)` span, and
`resolve_evidence` re-reads the source's *current* text on every call
rather than trusting a string captured at write time.
`test_evidence_anchor_resolves_to_the_exact_source_substring` and
`test_evidence_anchor_into_a_truncated_chunk_raises_instead_of_returning_a_shortened_span`
cover the round trip and the failure mode a copied string could never
exhibit. `test_memory_evidence_table_has_no_quote_column` asserts this
against `PRAGMA table_info`, the actual schema, rather than against
`write_memory`'s own behavior — the point is that copying a quote is
structurally impossible, not merely unused by this codebase's one writer.

**Task 2.4 — supersession is a derived view, never a stored flag.** The
`Supersession` section at the bottom of this file attacks migration 5 with
raw `sqlite3` SQL: `DELETE`, `UPDATE`, `INSERT OR REPLACE`, and `UPDATE OR
REPLACE`, including the `rowid` spelling of `memories.id`. Three of those
were measured to succeed against the pre-migration-5 store, silently
rewriting or destroying a row, so each negative assertion below is a
regression test for a hole that was open, not a hypothetical.
`test_supersede_guards_are_added_by_migration_5_not_migration_1` is the
proof they close because of a versioned migration an existing store will
actually receive.

This repository is public. Every statement, quote, and identifier in these
tests is invented for the test; none of it is derived from a real observed
session.
"""

from __future__ import annotations

import contextlib
import sqlite3

import pytest

from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate


def _seed_project_and_session(conn: sqlite3.Connection) -> tuple[int, int]:
    project_id = conn.execute(
        "INSERT INTO projects(name, path) VALUES (?, ?)",
        ("fixture-project", "/tmp/fixture-project"),
    ).lastrowid
    session_id = conn.execute(
        "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
        (project_id, "claude-code", "fixture-session-1"),
    ).lastrowid
    return project_id, session_id


def _seed_transcript_chunk(
    conn: sqlite3.Connection, session_id: int, content: str, seq: int = 1
) -> int:
    return conn.execute(
        "INSERT INTO transcript_chunks(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
        (session_id, seq, "user", content),
    ).lastrowid


def _seed_memory_directly(
    conn: sqlite3.Connection, project_id: int, session_id: int, statement: str, tier: int
) -> int:
    """Insert a bare `memories` row via raw SQL, bypassing `write_memory`.

    Only used where a test needs a `memories` row at a schema version older
    than `write_memory`'s minimum (version 4, as of task 2.2) — every other
    test in this file exercises `write_memory` itself at the latest schema
    version.
    """
    return conn.execute(
        "INSERT INTO memories(project_id, session_id, statement, origin, tier) "
        "VALUES (?, ?, ?, ?, ?)",
        (project_id, session_id, statement, "observer", tier),
    ).lastrowid


def _seed_anchor(conn: sqlite3.Connection, session_id: int) -> EvidenceAnchor:
    """Seed a fresh transcript chunk and return an anchor into it.

    Task 2.4's tests care about the `supersedes` edge, not about which text
    a memory cites, so this keeps every one of them from restating the same
    four lines of chunk-and-offset setup. The `seq` is derived from what is
    already stored, so repeated calls on one session stay UNIQUE-safe.
    """
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM transcript_chunks WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    content = f"fixture: transcript line {seq} recording an invented observation"
    chunk_id = _seed_transcript_chunk(conn, session_id, content, seq=seq)
    start, end = _span(content, "recording an invented observation")
    return EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)


def _seed_evidence_for(conn: sqlite3.Connection, session_id: int, memory_id: int) -> int:
    """Write a `memory_evidence` row naming `memory_id` before that row exists.

    Since migration 8 the database refuses a `memories` insert that no
    evidence row already names (INV-6), and the deferred child foreign key
    is what makes writing the child first legal. Every raw-SQL memory insert
    below therefore states an explicit id and seeds its evidence through
    here first — the same order `write_memory` itself now uses.
    """
    anchor = _seed_anchor(conn, session_id)
    return conn.execute(
        "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, start_offset, end_offset) "
        "VALUES (?, ?, ?, ?)",
        (memory_id, anchor.transcript_chunk_id, anchor.start_offset, anchor.end_offset),
    ).lastrowid


@contextlib.contextmanager
def _reserved_memory_id(conn: sqlite3.Connection, session_id: int):
    """Yield the next `memories.id`, with its evidence row already written.

    The raw-SQL tests below hand that id to an INSERT they expect some
    *other* guard to refuse. The evidence row would then be left naming a
    parent that never arrives, and the deferred foreign key would fail the
    next unrelated `commit()` — a failure in a later test's setup, which is
    the worst place to debug one. So the reservation is dropped on the way
    out whenever the memory did not land.
    """
    memory_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()[0]
    _seed_evidence_for(conn, session_id, memory_id)
    try:
        yield memory_id
    finally:
        if conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone() is None:
            conn.execute("DELETE FROM memory_evidence WHERE memory_id = ?", (memory_id,))


def _write_evidenced_memory(
    conn: sqlite3.Connection,
    project_id: int,
    session_id: int,
    statement: str,
    tier: int,
    origin: str = "observer",
) -> int:
    """`write_memory` with a freshly seeded evidence anchor (INV-6 satisfied)."""
    return write_memory(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement=statement,
        origin=origin,
        tier=tier,
        evidence=[_seed_anchor(conn, session_id)],
    )


def _span(content: str, substring: str) -> tuple[int, int]:
    """The `(start_offset, end_offset)` span of `substring` within `content`.

    A small helper so every test below anchors evidence at the substring it
    actually means, instead of hand-counted integer offsets that would
    silently go stale the moment a fixture string is edited.
    """
    start = content.index(substring)
    return start, start + len(substring)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()
