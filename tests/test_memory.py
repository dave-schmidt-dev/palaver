"""Tests for the append-only memory writer (task 2.1: INV-4/INV-5; task 2.2: INV-6).

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
does not extend to `palaver/store/schema.py`'s migration 4, which is
allowed to `DROP TABLE memory_evidence` and recreate it — a schema
migration reshaping a table's columns is not the append-only memory *write
path* INV-4 governs, and the migration only ever runs once per database,
before any `memories` row can exist at the new schema version.)

**INV-6 — every memory carries at least one evidence link.**
`test_memory_without_evidence_is_rejected` is this invariant's charter gate
(named in `INVARIANTS.md`). It asserts `write_memory` itself raises when
called with no evidence anchors — a Python-level check in
`palaver.memory.write.write_memory`, not a database trigger. That choice is
deliberate, not a shortcut: a database-layer design (a `memories.
primary_evidence_id` FK, written after the first evidence row) was
considered and would reject immediately, but it restructures
`memory_evidence` away from the 1-many `memory_id` shape task 2.4's
supersession work depends on. A deferred-FK variant avoids that
restructuring but only rejects at `conn.commit()` — after `write_memory`
has already returned — which cannot satisfy a gate test asserting the
*write call* raises. See `palaver/memory/write.py`'s module docstring for
the full design comparison.
`test_write_memory_rejects_an_evidence_anchor_with_neither_chunk_nor_event_link`
is the companion proof for the *other* half of "at least one": even if a
caller assembles an `EvidenceAnchor` with neither id set, `write_memory`'s
`INSERT` still fails, because the schema's own CHECK constraint (migration
1) rejects a `memory_evidence` row with both link columns NULL — the same
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

This repository is public. Every statement, quote, and identifier in these
tests is invented for the test; none of it is derived from a real observed
session.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

import palaver
from palaver.memory.evidence import EvidenceAnchor, EvidenceAnchorError, resolve_evidence
from palaver.memory.tiers import (
    ALL_TIERS,
    TIER_AGENT_CONCLUSION,
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_OBSERVER_SPECULATION,
    TIER_USER_INSTRUCTION,
    tier_name,
)
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from palaver.store.schema import SCHEMA_MIGRATIONS

MEMORY_DIR = Path(palaver.__file__).resolve().parent / "memory"


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


# =============================================================================
# palaver.memory.tiers
# =============================================================================


def test_all_tiers_is_the_five_tier_range_matching_the_schema_check():
    """ALL_TIERS matches the CHECK(tier BETWEEN 1 AND 5) range in schema.py.

    Catches the tier vocabulary drifting from the actual database
    constraint (e.g. someone adding a 6th tier here without a migration).
    """
    assert ALL_TIERS == (1, 2, 3, 4, 5)
    assert TIER_USER_INSTRUCTION == 1
    assert TIER_AGENT_CONCLUSION == 2
    assert TIER_OBSERVED_RESULT == 3
    assert TIER_OBSERVER_INFERENCE == 4
    assert TIER_OBSERVER_SPECULATION == 5


def test_tier_name_returns_the_expected_name_for_every_defined_tier():
    """Every tier in ALL_TIERS resolves to a distinct, non-empty name."""
    names = {tier_name(tier) for tier in ALL_TIERS}
    assert len(names) == len(ALL_TIERS)
    assert all(names)


def test_tier_name_raises_on_an_undefined_tier():
    """tier_name raises rather than silently returning a placeholder for tier 0 or 6.

    Without this, a caller passing a stray CHECK-violating tier value could
    get back `None` or an empty string instead of a clear error.
    """
    with pytest.raises(ValueError, match="unknown tier"):
        tier_name(0)
    with pytest.raises(ValueError, match="unknown tier"):
        tier_name(6)


# =============================================================================
# write_memory: basic shape
# =============================================================================


def test_write_memory_creates_a_memories_row_and_linked_evidence(db):
    """A basic write creates one memories row and one memory_evidence row, linked by id."""
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: the invented widget rotates nightly"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "invented widget rotates nightly")

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="the invented widget rotates on a nightly schedule",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()

    row = db.execute(
        "SELECT project_id, session_id, statement, origin, tier, supersedes "
        "FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert row == (
        project_id,
        session_id,
        "the invented widget rotates on a nightly schedule",
        "observer",
        TIER_OBSERVED_RESULT,
        None,
    )

    evidence_rows = db.execute(
        "SELECT memory_id, transcript_chunk_id, event_id, start_offset, end_offset "
        "FROM memory_evidence WHERE memory_id = ?",
        (memory_id,),
    ).fetchall()
    assert evidence_rows == [(memory_id, chunk_id, None, start, end)]


def test_write_memory_defaults_session_id_and_supersedes_to_null(db):
    """A write with no session_id or supersedes leaves both columns NULL."""
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: a project-level observation with no session attribution"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "project-level observation with no session attribution")

    memory_id = write_memory(
        db,
        project_id=project_id,
        statement="fixture: a project-level memory with no session",
        origin="observer",
        tier=TIER_AGENT_CONCLUSION,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()

    row = db.execute(
        "SELECT session_id, supersedes FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    assert row == (None, None)


# =============================================================================
# INV-6 — every memory carries at least one evidence anchor
# =============================================================================


def test_memory_without_evidence_is_rejected(db):
    """write_memory raises when called with no evidence anchors (INV-6).

    This is INV-6's charter gate test (`INVARIANTS.md`). It writes a
    memory carrying no evidence link at all and asserts the write itself
    raises, before anything reaches the database.

    LAYER PROOF: the raise happens in `palaver.memory.write.write_memory`,
    a plain Python `if not evidence: raise ValueError(...)` before the
    function's first `conn.execute`, not a `sqlite3.IntegrityError` from a
    trigger or constraint. The positive control below proves that guard
    truly blocked the `INSERT` rather than one silently succeeding and
    something else raising afterward: `memories` has zero rows immediately
    after the `pytest.raises` block, and exactly one after a properly
    evidenced write on the same connection.
    """
    project_id, session_id = _seed_project_and_session(db)

    with pytest.raises(ValueError, match="evidence"):
        write_memory(
            db,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a memory asserted with no evidence link at all",
            origin="observer",
            tier=TIER_OBSERVER_SPECULATION,
            evidence=[],
        )

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0

    # Positive control: the same connection still accepts a properly-evidenced write.
    content = "fixture: a corroborated observation worth remembering"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "corroborated observation worth remembering")
    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a properly evidenced observation",
        origin="observer",
        tier=TIER_OBSERVER_SPECULATION,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    assert memory_id is not None
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_write_memory_rejects_an_evidence_anchor_with_neither_chunk_nor_event_link(db):
    """A caller who builds an EvidenceAnchor with neither id set still fails, at the schema layer.

    `EvidenceAnchor` itself does not validate that at least one of
    `transcript_chunk_id`/`event_id` is set — nothing stops a caller from
    constructing one with both left `None`. This is the layer that catches
    that anyway: the CHECK constraint migration 1 puts on `memory_evidence`
    rejects the resulting row with `sqlite3.IntegrityError`.
    `resolve_evidence`'s `else` branch documents this CHECK as the reason
    that branch is unreachable in practice; without this test, that was an
    unverified comment rather than a proven fact.
    """
    project_id, session_id = _seed_project_and_session(db)

    with pytest.raises(sqlite3.IntegrityError):
        write_memory(
            db,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: evidence anchor built with neither source link set",
            origin="observer",
            tier=TIER_OBSERVER_SPECULATION,
            evidence=[EvidenceAnchor(start_offset=0, end_offset=5)],
        )


# =============================================================================
# Evidence anchoring and retrieval (task 2.2)
# =============================================================================


def test_evidence_anchor_resolves_to_the_exact_source_substring(db):
    """resolve_evidence returns exactly the source substring the anchor names.

    Asserts equality against the substring taken directly from the source
    fixture text, not a hand-typed copy of it, so a resolver that is off by
    one on either offset fails this.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: the deployment pipeline retried three times before succeeding"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    substring = "retried three times before succeeding"
    start, end = _span(content, substring)

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the deployment pipeline needed three retries",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()

    evidence_id = db.execute(
        "SELECT id FROM memory_evidence WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]

    assert resolve_evidence(db, evidence_id) == substring


def test_evidence_anchor_into_a_truncated_chunk_raises_instead_of_returning_a_shortened_span(db):
    """A truncated source makes a previously-valid anchor unresolvable.

    First resolves the anchor against the intact chunk and asserts the
    exact substring comes back — the positive control proving this test
    measures a resolver that can succeed, not one that always raises.
    Then truncates the chunk's stored content out from under the anchor,
    by direct UPDATE, and asserts resolution now raises `EvidenceAnchorError`
    rather than silently returning whatever fits in the shorter string.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: the archived migration script emitted a deprecation notice"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    substring = "emitted a deprecation notice"
    start, end = _span(content, substring)

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the archived migration script warned about deprecation",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()
    evidence_id = db.execute(
        "SELECT id FROM memory_evidence WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]

    # Positive control: intact, the anchor resolves to the exact substring.
    assert resolve_evidence(db, evidence_id) == substring

    db.execute(
        "UPDATE transcript_chunks SET content = ? WHERE id = ?",
        ("fixture: the archived migration script emitted a depre", chunk_id),
    )
    db.commit()

    with pytest.raises(EvidenceAnchorError, match="truncated"):
        resolve_evidence(db, evidence_id)


def test_resolve_evidence_raises_for_an_unknown_evidence_id(db):
    """resolve_evidence raises for an id with no matching memory_evidence row.

    Positive control: the id one less than it (guaranteed to have been a
    valid, resolvable row written just before) still resolves on the same
    connection, so the raise below is about the unknown id specifically,
    not a resolver that has stopped working.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: the scheduled backup completed without errors"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    substring = "completed without errors"
    start, end = _span(content, substring)

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the scheduled backup finished cleanly",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()
    evidence_id = db.execute(
        "SELECT id FROM memory_evidence WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]

    assert resolve_evidence(db, evidence_id) == substring

    unknown_id = evidence_id + 1_000_000
    with pytest.raises(EvidenceAnchorError, match="no memory_evidence row"):
        resolve_evidence(db, unknown_id)


def test_memory_evidence_table_has_no_quote_column(db):
    """The memory_evidence table's actual schema has no column that could hold a copied quote.

    Queries PRAGMA table_info directly rather than inspecting write_memory's
    behavior — the requirement is that copying a quote is structurally
    impossible, not merely that this codebase's one writer doesn't do it.
    Positive control: start_offset/end_offset are present, so this isn't
    passing because the table lookup itself silently found nothing.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(memory_evidence)").fetchall()}
    assert "quote" not in columns
    assert {"start_offset", "end_offset", "transcript_chunk_id", "event_id"} <= columns


def test_memory_evidence_offsets_replace_quote_by_migration_4_not_migration_1(tmp_path):
    """The quote-to-offsets schema change comes from migration 4, not a v1-baked shape.

    Migrates only through version 3 first and proves a legacy quote-based
    `INSERT` genuinely succeeds there, and that no `start_offset` column
    exists yet — the positive control that makes the later assertion
    meaningful. Then applies migration 4 against that already-populated
    database and shows the table has been rebuilt: the quote column and the
    row it held are both gone, and a new anchor-shaped row succeeds.
    Guards against the trap of folding this change into `_V1_STATEMENTS`
    instead of appending a new `Migration`, which would pass every other
    test here while leaving an already-created store's `memory_evidence`
    table permanently on the old, quote-copying shape.
    """
    db_path = tmp_path / "palaver.db"
    pre_v4 = tuple(m for m in SCHEMA_MIGRATIONS if m.version <= 3)
    migrate(db_path, migrations=pre_v4)

    conn = connect(db_path)
    try:
        columns_before = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_evidence)").fetchall()
        }
        assert "quote" in columns_before
        assert "start_offset" not in columns_before

        project_id, session_id = _seed_project_and_session(conn)
        memory_id = _seed_memory_directly(
            conn, project_id, session_id, "fixture: a pre-migration-4 memory", TIER_OBSERVED_RESULT
        )
        conn.execute(
            "INSERT INTO memory_evidence(memory_id, transcript_chunk_id, quote) VALUES (?, ?, ?)",
            (
                memory_id,
                _seed_transcript_chunk(conn, session_id, "fixture: pre-v4 evidence text"),
                "pre-v4 evidence text",
            ),
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 1
    finally:
        conn.close()

    migrate(db_path)  # apply migration 4

    conn = connect(db_path)
    try:
        columns_after = {
            row[1] for row in conn.execute("PRAGMA table_info(memory_evidence)").fetchall()
        }
        assert "quote" not in columns_after
        assert {"start_offset", "end_offset"} <= columns_after

        # The old quote-shaped row did not survive the rebuild — documented,
        # intentional, and asserted here rather than left implicit.
        assert conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()[0] == 0

        content = "fixture: a post-migration-4 evidence chunk"
        chunk_id = _seed_transcript_chunk(conn, session_id, content, seq=2)
        start, end = _span(content, "post-migration-4 evidence chunk")
        new_memory_id = write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a memory written after migration 4",
            origin="observer",
            tier=TIER_OBSERVED_RESULT,
            evidence=[
                EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)
            ],
        )
        assert new_memory_id != memory_id
    finally:
        conn.close()


# =============================================================================
# INV-5 — tier is immutable at the database layer
# =============================================================================


@pytest.mark.inv5
def test_update_tier_raises_at_the_database_layer(db):
    """A raw UPDATE naming `tier` raises via the schema's trigger, not this module's code.

    Attacks the connection directly with a hand-written UPDATE — never
    calling into palaver.memory.write — so this proves the *database*
    refuses the mutation. Two positive controls on the same connection,
    both after the failure: a fresh write_memory() call still succeeds
    (the connection/transaction is not simply broken), and an UPDATE naming
    a different column still succeeds (the trigger is scoped to `tier`
    specifically, not a blanket ban on ever touching a memories row).

    LAYER PROOF: `sqlite3.IntegrityError` is what SQLite's own
    `RAISE(ABORT, ...)` inside a trigger raises through the Python driver.
    A Python-side guard living in `write_memory` could never produce this
    from a raw `UPDATE` issued straight against the connection, since
    `write_memory` is never called in this test — the trigger created by
    schema.py migration 3 is the only thing that can be raising.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: the archived schedule runs nightly"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "archived schedule runs nightly")
    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the archived widget schedule runs nightly",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("UPDATE memories SET tier = ? WHERE id = ?", (TIER_USER_INSTRUCTION, memory_id))

    # Positive control 1: the row's tier truly did not change.
    unchanged_tier = db.execute("SELECT tier FROM memories WHERE id = ?", (memory_id,)).fetchone()[
        0
    ]
    assert unchanged_tier == TIER_OBSERVED_RESULT

    # Positive control 2: the same connection still accepts a legitimate write.
    second_content = "fixture: a second, unrelated invented transcript line"
    second_chunk_id = _seed_transcript_chunk(db, session_id, second_content, seq=2)
    second_start, second_end = _span(second_content, "unrelated invented transcript line")
    second_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a second, unrelated observation",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceAnchor(
                transcript_chunk_id=second_chunk_id,
                start_offset=second_start,
                end_offset=second_end,
            )
        ],
    )
    assert second_id != memory_id

    # Positive control 3: an UPDATE naming a different column still succeeds
    # on this same connection, proving the trigger is scoped to `tier`.
    db.execute("UPDATE memories SET origin = ? WHERE id = ?", ("observer-corrected", memory_id))
    corrected_origin = db.execute(
        "SELECT origin FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()[0]
    assert corrected_origin == "observer-corrected"


@pytest.mark.inv5
def test_tier_immutable_trigger_is_added_by_migration_3_not_migration_1(tmp_path):
    """The immutability guarantee comes from migration 3, not from a v1-baked trigger.

    Every other test in this file migrates a fresh database straight to the
    latest version, which cannot distinguish "the trigger exists" from "the
    trigger was added by a migration that an already-created store will
    actually receive." This test migrates only through version 2 first and
    proves the identical UPDATE genuinely succeeds there — the positive
    control that makes the later failure meaningful, rather than assuming
    version 2 already blocks it — then applies migrations 3 and 4 against
    that same already-populated database and re-issues the UPDATE, which
    now raises. Guards against the trap of folding the trigger's DDL into
    `_V1_STATEMENTS` instead of appending a new `Migration`: that change
    would pass every other test here while leaving every store created
    before the change permanently unmigrated and unprotected.

    The version-2 seed row is written with raw SQL
    (`_seed_memory_directly`), not `write_memory`: `write_memory` now
    targets the version-4 `memory_evidence` shape
    (`start_offset`/`end_offset`, task 2.2) and cannot be called against a
    database that hasn't been migrated that far yet. Only the post-migration
    positive control, which does run at the latest version, uses
    `write_memory` itself.
    """
    db_path = tmp_path / "palaver.db"
    pre_trigger = tuple(m for m in SCHEMA_MIGRATIONS if m.version <= 2)
    migrate(db_path, migrations=pre_trigger)

    conn = connect(db_path)
    try:
        project_id, session_id = _seed_project_and_session(conn)
        memory_id = _seed_memory_directly(
            conn,
            project_id,
            session_id,
            "fixture: a memory written before migration 3 exists",
            TIER_OBSERVER_INFERENCE,
        )
        conn.commit()

        # Positive control: at version 2, the identical UPDATE genuinely succeeds.
        conn.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (TIER_USER_INSTRUCTION, memory_id)
        )
        conn.commit()
        pre_trigger_tier = conn.execute(
            "SELECT tier FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        assert pre_trigger_tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()

    migrate(db_path)  # apply migration 3 (trigger) and migration 4 (evidence anchors)

    conn = connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE memories SET tier = ? WHERE id = ?", (TIER_OBSERVED_RESULT, memory_id)
            )

        # Positive control: a legitimate write still succeeds post-migration,
        # on this same connection, using the schema-v4 evidence-anchor shape.
        content = "fixture: a post-migration corroborating transcript line"
        second_chunk_id = _seed_transcript_chunk(conn, session_id, content)
        start, end = _span(content, "post-migration corroborating transcript line")
        second_id = write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a second memory written after migration 3 exists",
            origin="observer",
            tier=TIER_OBSERVED_RESULT,
            evidence=[
                EvidenceAnchor(
                    transcript_chunk_id=second_chunk_id, start_offset=start, end_offset=end
                )
            ],
        )
        assert second_id != memory_id
    finally:
        conn.close()


@pytest.mark.inv5
def test_update_tier_to_its_existing_value_still_raises(db):
    """The trigger fires on any UPDATE naming `tier`, even a no-op value-preserving one.

    `BEFORE UPDATE OF tier` fires because `tier` appears in the SET clause,
    independent of whether the new value differs from the old one. Without
    this test, a trigger written as `WHEN old.tier != new.tier` — which
    looks equivalent for every case exercised elsewhere in this file — would
    still pass every other assertion here while leaving a same-value UPDATE
    silently permitted.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: a no-op update fixture line"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "no-op update fixture line")
    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a memory whose tier will be set to itself",
        origin="observer",
        tier=TIER_OBSERVER_INFERENCE,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute(
            "UPDATE memories SET tier = ? WHERE id = ?", (TIER_OBSERVER_INFERENCE, memory_id)
        )


@pytest.mark.inv5
def test_reclassification_writes_a_new_row_and_leaves_the_original_byte_identical(db):
    """Reclassifying a memory's tier writes a second row; the first is fully untouched.

    Fetches every column of the original row (`SELECT *`) before and after
    the second `write_memory` call and asserts full equality — not just
    that `tier` didn't change — so a writer that touched `statement` or
    `created_at` while leaving `tier` alone would still fail this.
    """
    project_id, session_id = _seed_project_and_session(db)
    content = "fixture: an invented deploy-script warning"
    chunk_id = _seed_transcript_chunk(db, session_id, content)
    start, end = _span(content, "invented deploy-script warning")
    original_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the deploy script emitted an invented warning",
        origin="observer",
        tier=TIER_OBSERVER_INFERENCE,
        evidence=[EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=start, end_offset=end)],
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (original_id,)).fetchone()

    second_content = "fixture: a corroborating invented transcript line"
    second_chunk_id = _seed_transcript_chunk(db, session_id, second_content, seq=2)
    second_start, second_end = _span(second_content, "corroborating invented transcript line")
    reclassified_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the deploy script emitted an invented warning",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceAnchor(
                transcript_chunk_id=second_chunk_id,
                start_offset=second_start,
                end_offset=second_end,
            )
        ],
        supersedes=original_id,
    )
    db.commit()

    after = db.execute("SELECT * FROM memories WHERE id = ?", (original_id,)).fetchone()
    assert after == before

    assert reclassified_id != original_id
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    successor_row = db.execute(
        "SELECT tier, supersedes FROM memories WHERE id = ?", (reclassified_id,)
    ).fetchone()
    assert successor_row == (TIER_OBSERVED_RESULT, original_id)


# =============================================================================
# INV-4 — no DELETE path
# =============================================================================


def _executed_sql_strings(path: Path) -> list[str]:
    """String-literal arguments passed to any `execute*` call in `path`.

    An AST scan of the call sites that actually reach SQLite, not a text
    grep — so a docstring or comment discussing "no DELETE path" is never
    mistaken for a DELETE this module issues.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sql_strings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in (
            "execute",
            "executescript",
            "executemany",
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    sql_strings.append(arg.value)
    return sql_strings


@pytest.mark.inv4
def test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module(tmp_path):
    """No `execute*` call anywhere under palaver/memory/ issues a DELETE or DROP statement.

    Complements, rather than replaces, the plain-text acceptance check
    `rg -n 'DELETE|DROP' palaver/memory` — that check also catches the word
    appearing in a docstring (which is fine; this test only inspects strings
    actually handed to sqlite3's execute methods).
    """
    paths = sorted(MEMORY_DIR.rglob("*.py"))
    # Enumeration guard: an empty sweep would make the assertion below pass
    # vacuously regardless of what palaver/memory/ contains.
    assert any(path.name == "write.py" for path in paths)

    violations = {}
    for path in paths:
        hits = [
            sql
            for sql in _executed_sql_strings(path)
            if "DELETE" in sql.upper() or "DROP" in sql.upper()
        ]
        if hits:
            violations[path.name] = hits
    assert violations == {}

    # Positive control: prove the detector is live against a module that
    # does issue a DELETE.
    poisoned = tmp_path / "poisoned_writer.py"
    poisoned.write_text(
        "def wipe(conn):\n    conn.execute('DELETE FROM memories WHERE id = ?', (1,))\n"
    )
    assert _executed_sql_strings(poisoned) == ["DELETE FROM memories WHERE id = ?"]
