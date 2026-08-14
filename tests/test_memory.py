"""Tests for the append-only memory writer (task 2.1: INV-4 and INV-5).

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
showing the identical UPDATE only starts failing after that step.

**INV-4 — no DELETE path.**
`test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module` statically
scans every `execute*` call under `palaver/memory/` for a `DELETE` or `DROP`
SQL string, with a positive control proving the detector is live.

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
from palaver.memory.tiers import (
    ALL_TIERS,
    TIER_AGENT_CONCLUSION,
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_OBSERVER_SPECULATION,
    TIER_USER_INSTRUCTION,
    tier_name,
)
from palaver.memory.write import EvidenceInput, write_memory
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
    chunk_id = _seed_transcript_chunk(
        db, session_id, "fixture: the invented widget rotates nightly"
    )

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="the invented widget rotates on a nightly schedule",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceInput(quote="invented widget rotates nightly", transcript_chunk_id=chunk_id)
        ],
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
        "SELECT memory_id, transcript_chunk_id, event_id, quote "
        "FROM memory_evidence WHERE memory_id = ?",
        (memory_id,),
    ).fetchall()
    assert evidence_rows == [(memory_id, chunk_id, None, "invented widget rotates nightly")]


def test_write_memory_permits_zero_evidence_rows_pending_task_2_2(db):
    """write_memory does not itself enforce an evidence floor; task 2.2 will.

    Documents current scope: INV-6 ("every memory carries evidence") is
    task 2.2's gate (`test_memory_without_evidence_is_rejected`), built
    against the span-anchor shape that task replaces `quote` with. Asserting
    the opposite behavior here would have to be undone once that lands.
    """
    project_id, session_id = _seed_project_and_session(db)

    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a memory written with no evidence link, for now",
        origin="observer",
        tier=TIER_OBSERVER_SPECULATION,
    )
    db.commit()

    count = db.execute(
        "SELECT COUNT(*) FROM memory_evidence WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]
    assert count == 0


def test_write_memory_defaults_session_id_and_supersedes_to_null(db):
    """A write with no session_id or supersedes leaves both columns NULL."""
    project_id = _seed_project_and_session(db)[0]

    memory_id = write_memory(
        db,
        project_id=project_id,
        statement="fixture: a project-level memory with no session",
        origin="observer",
        tier=TIER_AGENT_CONCLUSION,
    )
    db.commit()

    row = db.execute(
        "SELECT session_id, supersedes FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    assert row == (None, None)


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
    chunk_id = _seed_transcript_chunk(db, session_id, "fixture: the archived schedule runs nightly")
    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the archived widget schedule runs nightly",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceInput(quote="archived schedule runs nightly", transcript_chunk_id=chunk_id)
        ],
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
    second_chunk_id = _seed_transcript_chunk(
        db, session_id, "fixture: a second, unrelated invented transcript line", seq=2
    )
    second_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a second, unrelated observation",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceInput(
                quote="unrelated invented transcript line", transcript_chunk_id=second_chunk_id
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
    version 2 already blocks it — then applies migration 3 against that same
    already-populated database and re-issues the UPDATE, which now raises.
    Guards against the trap of folding the trigger's DDL into
    `_V1_STATEMENTS` instead of appending a new `Migration`: that change
    would pass every other test here while leaving every store created
    before the change permanently unmigrated and unprotected.
    """
    db_path = tmp_path / "palaver.db"
    pre_trigger = tuple(m for m in SCHEMA_MIGRATIONS if m.version <= 2)
    migrate(db_path, migrations=pre_trigger)

    conn = connect(db_path)
    try:
        project_id, session_id = _seed_project_and_session(conn)
        chunk_id = _seed_transcript_chunk(
            conn, session_id, "fixture: a pre-migration-3 fixture line"
        )
        memory_id = write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a memory written before migration 3 exists",
            origin="observer",
            tier=TIER_OBSERVER_INFERENCE,
            evidence=[
                EvidenceInput(quote="pre-migration-3 fixture line", transcript_chunk_id=chunk_id)
            ],
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

    migrate(db_path)  # apply migration 3 against the already-populated database

    conn = connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE memories SET tier = ? WHERE id = ?", (TIER_OBSERVED_RESULT, memory_id)
            )

        # Positive control: a legitimate write still succeeds post-migration,
        # on this same connection.
        second_id = write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: a second memory written after migration 3 exists",
            origin="observer",
            tier=TIER_OBSERVED_RESULT,
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
    chunk_id = _seed_transcript_chunk(db, session_id, "fixture: a no-op update fixture line")
    memory_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: a memory whose tier will be set to itself",
        origin="observer",
        tier=TIER_OBSERVER_INFERENCE,
        evidence=[EvidenceInput(quote="no-op update fixture line", transcript_chunk_id=chunk_id)],
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
    chunk_id = _seed_transcript_chunk(db, session_id, "fixture: an invented deploy-script warning")
    original_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the deploy script emitted an invented warning",
        origin="observer",
        tier=TIER_OBSERVER_INFERENCE,
        evidence=[
            EvidenceInput(quote="invented deploy-script warning", transcript_chunk_id=chunk_id)
        ],
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (original_id,)).fetchone()

    second_chunk_id = _seed_transcript_chunk(
        db, session_id, "fixture: a corroborating invented transcript line", seq=2
    )
    reclassified_id = write_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        statement="fixture: the deploy script emitted an invented warning",
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        evidence=[
            EvidenceInput(
                quote="corroborating invented transcript line", transcript_chunk_id=second_chunk_id
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
