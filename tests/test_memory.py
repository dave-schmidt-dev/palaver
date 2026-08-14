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

import ast
import sqlite3
from pathlib import Path

import pytest

import palaver
from palaver.memory.evidence import EvidenceAnchor, EvidenceAnchorError, resolve_evidence
from palaver.memory.supersede import is_superseded, successor_of, supersede_memory
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
from palaver.store.migrate import MigrationError, connect, current_version, migrate
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


# =============================================================================
# Task 2.4 — supersession as a derived view, never a stored flag
# =============================================================================


@pytest.mark.inv4
def test_supersede_preserves_original_row(db):
    """A superseded memory is immutable: no UPDATE naming any of its columns succeeds.

    This is INV-4's charter gate test (`INVARIANTS.md`). Supersession
    records a correction on the *successor* row; the predecessor is
    evidence, and evidence that can be edited after the fact is not
    evidence. So the assertion is not "tier didn't change" but "no column
    changed, and every attempt to change one raised".

    Attacks the connection with hand-written `sqlite3` SQL — never through
    `palaver.memory.supersede` — and sweeps every column reported by
    `PRAGMA table_info`, so a column added by a later migration is covered
    the day it appears rather than the day someone remembers to extend this
    list.

    LAYER PROOF: `sqlite3.IntegrityError` is what SQLite's own
    `RAISE(ABORT, ...)` raises through the Python driver, and no code from
    this project runs during a raw `conn.execute("UPDATE ...")`. Two
    positive controls prove the guard is scoped rather than a blanket freeze
    on the table: the *successor* row — which nothing supersedes — still
    accepts an `origin` UPDATE on the same connection, and a fresh
    `write_memory` still succeeds.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the invented widget ships on Tuesdays", 4
    )
    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the invented widget ships on Thursdays",
        origin="observer",
        tier=3,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (predecessor_id,)).fetchone()

    columns = [row[1] for row in db.execute("PRAGMA table_info(memories)").fetchall()]
    # Enumeration guard: an empty sweep would pass this test vacuously.
    assert set(columns) == {
        "id",
        "project_id",
        "session_id",
        "statement",
        "origin",
        "tier",
        "supersedes",
        "created_at",
    }
    for column in columns:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(f"UPDATE memories SET {column} = {column} WHERE id = ?", (predecessor_id,))

    # The same refusal for UPDATEs carrying genuinely different values, not
    # only the value-preserving shape above.
    rewrites = (
        ("statement = ?", ("fixture: a statement rewritten in place",)),
        ("origin = ?", ("attacker",)),
        ("tier = ?", (TIER_USER_INSTRUCTION,)),
        ("supersedes = ?", (None,)),
        ("id = ?", (predecessor_id + 5000,)),
        ("created_at = ?", ("2000-01-01T00:00:00.000Z",)),
    )
    for clause, params in rewrites:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(f"UPDATE memories SET {clause} WHERE id = ?", (*params, predecessor_id))

    after = db.execute("SELECT * FROM memories WHERE id = ?", (predecessor_id,)).fetchone()
    assert after == before

    # Positive control 1: the successor is not superseded, so it is still
    # writable — the guard tracks the supersedes edge, it is not a freeze on
    # the whole table.
    db.execute("UPDATE memories SET origin = ? WHERE id = ?", ("observer-amended", successor_id))
    assert (
        db.execute("SELECT origin FROM memories WHERE id = ?", (successor_id,)).fetchone()[0]
        == "observer-amended"
    )

    # Positive control 2: the connection still accepts an ordinary write.
    third_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: an unrelated later observation", 4
    )
    assert third_id not in (predecessor_id, successor_id)


@pytest.mark.inv5
def test_lower_tier_cannot_supersede_higher_tier(db):
    """A tier-4 observer inference cannot supersede a tier-1 user instruction.

    This is INV-5's charter gate test (`INVARIANTS.md`). The insert is
    issued as raw `sqlite3` SQL, bypassing `palaver.memory.supersede`
    entirely, because the invariant's whole point is that the ordering holds
    regardless of which model wrote the row or which code path issued it.

    Positive control: the identical INSERT with `supersedes` left NULL
    succeeds on the same connection, so the refusal is about the
    supersession link and not about tier-4 rows being unwritable.
    """
    project_id, session_id = _seed_project_and_session(db)
    user_instruction_id = _write_evidenced_memory(
        db,
        project_id,
        session_id,
        "fixture: the user asked for nightly rotation, not hourly",
        TIER_USER_INSTRUCTION,
        origin="user",
    )
    db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="lower-confidence tier"):
        db.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                project_id,
                session_id,
                "fixture: the observer guessed hourly rotation was meant",
                "observer",
                TIER_OBSERVER_INFERENCE,
                user_instruction_id,
            ),
        )

    assert db.execute("SELECT COUNT(*) FROM superseded_memories").fetchone()[0] == 0

    # Positive control: the same tier-4 row is perfectly writable as its own
    # memory; only the supersession link was refused.
    unlinked_id = db.execute(
        "INSERT INTO memories(project_id, session_id, statement, origin, tier) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            project_id,
            session_id,
            "fixture: the observer guessed hourly rotation was meant",
            "observer",
            TIER_OBSERVER_INFERENCE,
        ),
    ).lastrowid
    assert unlinked_id is not None


@pytest.mark.inv5
def test_equal_or_higher_confidence_tier_may_supersede_a_lower_confidence_row(db):
    """Supersession in the permitted direction succeeds — the rule is an ordering, not a ban.

    Without this, `test_lower_tier_cannot_supersede_higher_tier` would still
    pass against a trigger that rejected every supersession outright, which
    would break correction entirely while looking like a satisfied invariant.
    """
    project_id, session_id = _seed_project_and_session(db)
    inference_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the observer inferred a nightly cadence", 4
    )
    same_tier_target_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a second observer inference", 4
    )
    db.commit()

    higher_confidence_id = supersede_memory(
        db,
        predecessor_id=inference_id,
        statement="fixture: the command output confirmed a nightly cadence",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[_seed_anchor(db, session_id)],
    )
    equal_confidence_id = supersede_memory(
        db,
        predecessor_id=same_tier_target_id,
        statement="fixture: a revised observer inference at the same tier",
        origin="observer",
        tier=TIER_OBSERVER_INFERENCE,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()

    assert {
        row[0] for row in db.execute("SELECT memory_id FROM superseded_memories").fetchall()
    } == {inference_id, same_tier_target_id}
    assert higher_confidence_id != equal_confidence_id


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
        with pytest.raises(sqlite3.IntegrityError, match="must name an existing memory"):
            raw.execute(
                "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    project_id,
                    session_id,
                    "fixture: a successor pointing at a memory that never existed",
                    "attacker",
                    TIER_OBSERVER_INFERENCE,
                    999_999,
                ),
            )

        # Positive control: unlinked insert, then a genuine supersession.
        predecessor_id = raw.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, session_id, "fixture: a real predecessor", "observer", 4),
        ).lastrowid
        successor_id = raw.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, session_id, "fixture: a real successor", "observer", 4, predecessor_id),
        ).lastrowid
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

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
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

        rival_correction = (
            project_id,
            session_id,
            "fixture: a rival correction",
            "observer",
            4,
            predecessor_id,
        )

        # With both layers present, the trigger answers first.
        with pytest.raises(sqlite3.IntegrityError, match="at most one successor"):
            conn.execute(
                "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rival_correction,
            )

        # With the trigger gone, the index alone still refuses it — in
        # SQLite's own words, which no trigger in this schema produces.
        conn.execute("DROP TRIGGER memories_one_successor_guard")
        with pytest.raises(
            sqlite3.IntegrityError, match=r"UNIQUE constraint failed: memories\.supersedes"
        ):
            conn.execute(
                "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rival_correction,
            )
    finally:
        conn.close()


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

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        db.execute(
            "INSERT OR REPLACE INTO memories"
            "(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
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
    db.execute(
        "INSERT OR REPLACE INTO memories"
        "(project_id, session_id, statement, origin, tier, supersedes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
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

    # Positive control: the bystander is still an ordinary, writable row.
    db.execute("UPDATE memories SET origin = ? WHERE id = ?", ("observer-amended", bystander_id))
    assert (
        db.execute("SELECT origin FROM memories WHERE id = ?", (bystander_id,)).fetchone()[0]
        == "observer-amended"
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


def test_superseded_memories_view_is_empty_until_a_supersession_names_the_predecessor(db):
    """The view holds no row before a supersession and exactly the predecessor's id after.

    This is the derived-view half of task 2.4: there is no stored flag to
    read, so "is this memory current?" is answered by whether any row points
    at it. The before-assertion is not decoration — a view defined over the
    wrong column, or one that listed every memory, would satisfy the
    after-assertion alone.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the original invented finding", 4
    )
    bystander_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a finding nobody corrects", 4
    )
    db.commit()

    assert db.execute("SELECT * FROM superseded_memories").fetchall() == []
    assert not is_superseded(db, predecessor_id)
    assert successor_of(db, predecessor_id) is None

    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the corrected invented finding",
        origin="observer",
        tier=3,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()

    assert db.execute("SELECT memory_id FROM superseded_memories").fetchall() == [(predecessor_id,)]
    assert is_superseded(db, predecessor_id)
    assert successor_of(db, predecessor_id) == successor_id
    # The successor itself, and the untouched bystander, are current.
    assert not is_superseded(db, successor_id)
    assert not is_superseded(db, bystander_id)


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
    pre_v5 = tuple(m for m in SCHEMA_MIGRATIONS if m.version <= 4)
    migrate(db_path, migrations=pre_v5)

    conn = connect(db_path)
    try:
        views = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'").fetchall()
        }
        assert "superseded_memories" not in views

        project_id, session_id = _seed_project_and_session(conn)
        memory_id = _write_evidenced_memory(
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
        assert current_version(conn) == 5
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
        predecessor_id = _write_evidenced_memory(
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
        predecessor_id = _write_evidenced_memory(
            clean, project_id, session_id, "fixture: a singly-corrected conclusion", 4
        )
        clean.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, session_id, "fixture: its one correction", "observer", 4, predecessor_id),
        )
        clean.commit()
    finally:
        clean.close()

    assert migrate(clean_path) == 5
    clean = connect(clean_path)
    try:
        assert clean.execute("SELECT memory_id FROM superseded_memories").fetchall() == [
            (predecessor_id,)
        ]
    finally:
        clean.close()


def test_supersede_memory_writes_a_successor_and_leaves_the_predecessor_untouched(db):
    """The Python helper inherits the predecessor's scope and never writes to it.

    `SELECT *` on the predecessor before and after is compared whole, so a
    helper that touched any column of it — not only `tier` — fails here.
    The inherited `project_id`/`session_id` is the one thing this helper
    adds over a bare `write_memory` call: a correction cannot silently land
    in a different project than the memory it corrects.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: the invented pipeline retries twice", 4
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (predecessor_id,)).fetchone()

    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: the invented pipeline retries three times",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()

    assert db.execute("SELECT * FROM memories WHERE id = ?", (predecessor_id,)).fetchone() == before
    assert db.execute(
        "SELECT project_id, session_id, tier, supersedes FROM memories WHERE id = ?",
        (successor_id,),
    ).fetchone() == (project_id, session_id, TIER_OBSERVED_RESULT, predecessor_id)
    assert (
        db.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id = ?", (successor_id,)
        ).fetchone()[0]
        == 1
    )


def test_supersede_memory_raises_for_a_predecessor_that_does_not_exist(db):
    """An unknown predecessor id raises before anything is written.

    Positive control: the same call against a real predecessor succeeds on
    the same connection, and the failed call left no partial row behind.
    """
    project_id, session_id = _seed_project_and_session(db)
    predecessor_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: a real memory to correct", 4
    )
    db.commit()

    with pytest.raises(LookupError, match="to supersede"):
        supersede_memory(
            db,
            predecessor_id=predecessor_id + 1_000_000,
            statement="fixture: a correction of nothing at all",
            origin="observer",
            tier=4,
            evidence=[_seed_anchor(db, session_id)],
        )
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    successor_id = supersede_memory(
        db,
        predecessor_id=predecessor_id,
        statement="fixture: a correction of something real",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()
    assert successor_of(db, predecessor_id) == successor_id


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

    # Positive control: a non-identity, non-link column on the same row is
    # still writable.
    db.execute("UPDATE memories SET origin = ? WHERE id = ?", ("observer-amended", bystander_id))
    assert (
        db.execute("SELECT origin FROM memories WHERE id = ?", (bystander_id,)).fetchone()[0]
        == "observer-amended"
    )
