"""
The tier vocabulary (ALL_TIERS/tier_name) and write_memory's basic shape, including
INV-6's evidence-required gate.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.memory.evidence import EvidenceAnchor, resolve_evidence
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
from palaver.store.migrate import migrate
from tests._memory_support import (
    _seed_anchor,
    _seed_evidence_for,
    _seed_project_and_session,
    _seed_transcript_chunk,
    _span,
)
from tests._memory_support import db as db

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


@pytest.mark.inv6
def test_a_memory_with_no_evidence_is_refused_by_the_database_not_only_by_write_memory(db):
    """Raw SQL cannot write an unevidenced memory, whatever `write_memory` does.

    `test_memory_without_evidence_is_rejected` above proves the
    Python-level check; this proves the floor underneath it, added by
    migration 8. The INSERT is hand-written `sqlite3` SQL and no code from
    this project runs during it, so the refusal is the database's own — the
    case a second module opening its own connection would otherwise walk
    straight past.

    Positive control: the identical INSERT succeeds the moment an evidence
    row naming that id exists, so the refusal is about missing evidence and
    not about raw inserts being blocked outright.
    """
    project_id, session_id = _seed_project_and_session(db)
    memory_id = db.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError, match="evidence"):
        db.execute(
            "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, project_id, session_id, "fixture: an unevidenced claim", "attacker", 4),
        )

    _seed_evidence_for(db, session_id, memory_id)
    db.execute(
        "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (memory_id, project_id, session_id, "fixture: an evidenced claim", "observer", 4),
    )
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


@pytest.mark.inv6
def test_an_insert_letting_sqlite_pick_the_id_is_refused_even_with_evidence_at_minus_one(db):
    """The evidence check cannot be satisfied by an id the row will never have.

    Measured on SQLite 3.53: inside a `BEFORE INSERT` trigger, `NEW.id` for
    an unspecified INTEGER PRIMARY KEY reads as **-1** — not NULL, and not
    the rowid the row is about to receive. A trigger that only asked
    `EXISTS (... memory_id = NEW.id)` would therefore be satisfied for every
    implicit-id insert by a single evidence row planted at `memory_id = -1`.
    With foreign keys off nothing else would catch that, so the trigger also
    requires the id to be stated and positive.
    """
    project_id, session_id = _seed_project_and_session(db)
    _seed_evidence_for(db, session_id, -1)

    with pytest.raises(sqlite3.IntegrityError, match="explicit id"):
        db.execute(
            "INSERT INTO memories(project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, session_id, "fixture: a claim riding the -1 sentinel", "attacker", 4),
        )

    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


@pytest.mark.inv6
def test_evidence_and_mutation_rules_hold_on_a_connection_with_foreign_keys_off(tmp_path):
    """Neither rule depends on `PRAGMA foreign_keys`, which is per connection.

    `palaver.store.migrate.connect` turns foreign keys on; a bare
    `sqlite3.connect(path)` — the next module, another process, a human at
    the shell — does not. Migration 8's deferred child foreign key is inert
    on this connection, which is precisely why the enforcement is a trigger:
    a trigger fires whatever the pragma says.

    Positive control on the same pragma-less connection: an evidenced,
    explicitly-identified insert still succeeds, so the refusals are the
    rules and not the missing pragma breaking writes in general.
    """
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    raw = sqlite3.connect(str(db_path))
    try:
        # Control for the premise: foreign keys really are off here.
        assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        project_id, session_id = _seed_project_and_session(raw)

        memory_id = raw.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="evidence"):
            raw.execute(
                "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (memory_id, project_id, session_id, "fixture: unevidenced, no FKs", "attacker", 4),
            )

        _seed_evidence_for(raw, session_id, memory_id)
        raw.execute(
            "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, project_id, session_id, "fixture: evidenced, no FKs", "observer", 4),
        )
        raw.commit()

        for clause, params in (
            ("statement = ?", ("fixture: rewritten with foreign keys off",)),
            ("origin = ?", ("attacker",)),
            ("project_id = ?", (project_id,)),
            ("session_id = ?", (None,)),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                raw.execute(f"UPDATE memories SET {clause} WHERE id = ?", (*params, memory_id))

        assert (
            raw.execute("SELECT statement FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
            == "fixture: evidenced, no FKs"
        )
    finally:
        raw.close()


def test_write_memory_commits_one_and_many_evidence_rows_for_one_memory(db):
    """The ordinary path: one anchor and several, both landing on disk.

    Read back on a *second* connection after `commit()` rather than on the
    writing one, because the child-first order leaves the evidence rows
    naming a parent that does not exist yet for the length of the
    transaction. Only a committed store proves the deferred foreign key was
    satisfied rather than merely deferred.
    """
    project_id, session_id = _seed_project_and_session(db)
    single_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a memory resting on one line of transcript",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    multi_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a memory resting on three lines of transcript",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id) for _ in range(3)],
    )
    db.commit()
    db_path = db.execute("PRAGMA database_list").fetchone()[2]

    reader = sqlite3.connect(db_path)
    try:
        counts = dict(
            reader.execute(
                "SELECT memory_id, COUNT(*) FROM memory_evidence "
                "WHERE memory_id IN (?, ?) GROUP BY memory_id",
                (single_id, multi_id),
            ).fetchall()
        )
        assert single_id != multi_id
        assert counts == {single_id: 1, multi_id: 3}
        # Each row still resolves against the source it points at, so the
        # one-to-many link survived the inverted write order, not just the count.
        for (evidence_id,) in reader.execute(
            "SELECT id FROM memory_evidence WHERE memory_id = ? ORDER BY id", (multi_id,)
        ).fetchall():
            assert resolve_evidence(reader, evidence_id)
    finally:
        reader.close()
