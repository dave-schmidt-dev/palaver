"""
Task 2.4 core: the tier-ordering rule (both INV-4/INV-5 charter gates),
supersede_memory's positive path, and the derived superseded_memories view.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.memory.supersede import is_superseded, successor_of, supersede_memory
from palaver.memory.tiers import (
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_USER_INSTRUCTION,
)
from tests._memory_support import (
    _reserved_memory_id,
    _seed_anchor,
    _seed_project_and_session,
    _write_evidenced_memory,
)
from tests._memory_support import db as db

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

    # Positive control 1: the guard under test tracks the supersedes edge
    # rather than freezing the table. `created_at` is the one column
    # migration 8's live-row rule leaves alone, which is what makes it
    # discriminate the two rows: the predecessor's is refused because it is
    # superseded, and the identical UPDATE lands on the successor, which
    # nothing supersedes.
    amended_at = "2001-01-01T00:00:00.000Z"
    with pytest.raises(sqlite3.IntegrityError, match="see its successor"):
        db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (amended_at, predecessor_id))
    db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (amended_at, successor_id))
    assert (
        db.execute("SELECT created_at FROM memories WHERE id = ?", (successor_id,)).fetchone()[0]
        == amended_at
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

    with _reserved_memory_id(db, session_id) as rival_id:
        with pytest.raises(sqlite3.IntegrityError, match="lower-confidence tier"):
            db.execute(
                "INSERT INTO memories"
                "(id, project_id, session_id, statement, origin, tier, supersedes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    rival_id,
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
    with _reserved_memory_id(db, session_id) as unlinked_id:
        db.execute(
            "INSERT INTO memories(id, project_id, session_id, statement, origin, tier) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                unlinked_id,
                project_id,
                session_id,
                "fixture: the observer guessed hourly rotation was meant",
                "observer",
                TIER_OBSERVER_INFERENCE,
            ),
        )
    assert db.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (unlinked_id,)).fetchone()


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
