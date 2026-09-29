"""
Evidence anchoring and retrieval (task 2.2): resolve_evidence re-reads the live source,
never a copy.
"""

from __future__ import annotations

import pytest

from palaver.memory.evidence import EvidenceAnchor, EvidenceAnchorError, resolve_evidence
from palaver.memory.tiers import (
    TIER_OBSERVED_RESULT,
)
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from palaver.store.schema import LATEST_VERSION, SCHEMA_MIGRATIONS
from tests._memory_support import (
    _seed_anchor,
    _seed_memory_directly,
    _seed_project_and_session,
    _seed_transcript_chunk,
    _span,
)
from tests._memory_support import db as db

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

    # Stops at 4 rather than running to the latest version: the rebuild
    # below leaves this store's one memory with no evidence at all, and
    # migration 8 refuses to grandfather exactly that (INV-6).
    assert migrate(db_path, target_version=4) == 4

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

        # Re-anchor the memory the rebuild orphaned, in the new shape. This
        # is the operator action migration 8's abort message asks for, and
        # doing it here is what lets this store migrate the rest of the way.
        content = "fixture: a post-migration-4 evidence chunk"
        chunk_id = _seed_transcript_chunk(conn, session_id, content, seq=2)
        start, end = _span(content, "post-migration-4 evidence chunk")
        conn.execute(
            "INSERT INTO memory_evidence"
            "(memory_id, transcript_chunk_id, start_offset, end_offset) VALUES (?, ?, ?, ?)",
            (memory_id, chunk_id, start, end),
        )
        conn.commit()
    finally:
        conn.close()

    assert migrate(db_path) == LATEST_VERSION

    conn = connect(db_path)
    try:
        new_memory_id = write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a memory written after migration 4",
            origin="observer",
            tier=TIER_OBSERVED_RESULT,
            evidence=[_seed_anchor(conn, session_id)],
        )
        assert new_memory_id != memory_id
    finally:
        conn.close()
