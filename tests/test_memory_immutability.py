"""
INV-5 (tier is immutable at the database layer) and INV-4 (no DELETE path), both proved
with hand-written sqlite3 SQL.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

import palaver
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.supersede import is_superseded, successor_of, supersede_memory
from palaver.memory.tiers import (
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_USER_INSTRUCTION,
)
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from palaver.store.schema import LATEST_VERSION, SCHEMA_MIGRATIONS
from tests._memory_support import (
    _seed_anchor,
    _seed_evidence_for,
    _seed_memory_directly,
    _seed_project_and_session,
    _seed_transcript_chunk,
    _span,
    _write_evidenced_memory,
)
from tests._memory_support import db as db

MEMORY_DIR = Path(palaver.__file__).resolve().parent / "memory"


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

    # Positive control 3: the trigger under test is still scoped to `tier`.
    # Since migration 8 an UPDATE of `origin` is refused as well, but by a
    # different rule carrying a different message — so this trigger is not
    # simply refusing every UPDATE the table receives. `created_at`, which
    # neither rule covers on a live row, still writes on this connection.
    with pytest.raises(sqlite3.IntegrityError, match="supersede, never rewrite"):
        db.execute("UPDATE memories SET origin = ? WHERE id = ?", ("observer-corrected", memory_id))
    db.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?", ("2002-02-02T00:00:00.000Z", memory_id)
    )
    assert (
        db.execute("SELECT created_at FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
        == "2002-02-02T00:00:00.000Z"
    )


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
    (`_seed_memory_directly`), not `write_memory`: `write_memory` targets
    the version-4 `memory_evidence` shape (`start_offset`/`end_offset`, task
    2.2) and, since migration 8, writes its evidence child-first against a
    deferred foreign key that older versions do not have — so it cannot be
    called against a store below version 8 at all. Only the post-migration
    positive control, which does run at the latest version, uses
    `write_memory` itself. The seed is re-anchored partway through, because
    migration 8 refuses to migrate a store whose memories have no evidence
    rather than grandfather them.
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

    # Stops at 4 for the same reason as the migration-4 test above: this
    # store's seed memory has no evidence, and migration 8 will not
    # grandfather it. The seed is re-anchored below, then migration
    # continues — so this test also covers the upgrade path an existing
    # store takes through both changes, not just through migration 3.
    assert migrate(db_path, target_version=4) == 4

    conn = connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE memories SET tier = ? WHERE id = ?", (TIER_OBSERVED_RESULT, memory_id)
            )
        _seed_evidence_for(conn, session_id, memory_id)
        conn.commit()
    finally:
        conn.close()

    assert migrate(db_path) == LATEST_VERSION

    conn = connect(db_path)
    try:
        # The trigger added at migration 3 is still the one refusing this at
        # the latest version, on the same upgraded store.
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE memories SET tier = ? WHERE id = ?", (TIER_OBSERVED_RESULT, memory_id)
            )

        # Positive control: a legitimate write still succeeds post-migration,
        # on this same connection, using the schema-v4 evidence-anchor shape.
        content = "fixture: a post-migration corroborating transcript line"
        second_chunk_id = _seed_transcript_chunk(conn, session_id, content, seq=99)
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


@pytest.mark.inv4
def test_a_live_memory_cannot_be_rewritten_in_place_by_raw_sql(db):
    """A memory nothing supersedes is still not editable: rewriting is not correcting.

    Migration 3 froze `tier` and migration 5 froze a *superseded* row
    entirely; until migration 8 a current memory's `statement`, `origin`,
    `project_id`, and `session_id` were rewritable by any raw `UPDATE` —
    the history-destroying edit supersession exists to replace. Attacked
    here with hand-written `sqlite3` SQL, never through `palaver.memory`.

    Positive control: `created_at` is deliberately outside this rule, so the
    same connection still writes it — the refusals are scoped rather than a
    table freeze — and the legitimate correction path still works.
    """
    project_id, session_id = _seed_project_and_session(db)
    memory_id = _write_evidenced_memory(
        db, project_id, session_id, "fixture: what the observer first recorded", 4
    )
    db.commit()
    before = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()

    # Control for the premise: nothing supersedes this row, so migration 5's
    # superseded-row guard is not what refuses the UPDATEs below.
    assert not is_superseded(db, memory_id)
    for clause, params in (
        ("statement = ?", ("fixture: a statement rewritten in place",)),
        ("origin = ?", ("attacker",)),
        ("project_id = ?", (project_id,)),
        ("session_id = ?", (None,)),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db.execute(f"UPDATE memories SET {clause} WHERE id = ?", (*params, memory_id))

    assert db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone() == before

    # Positive control: the row is still reachable and correctable.
    db.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?", ("2005-05-05T00:00:00.000Z", memory_id)
    )
    successor_id = supersede_memory(
        db,
        predecessor_id=memory_id,
        statement="fixture: what the observer recorded instead",
        origin="observer",
        tier=4,
        evidence=[_seed_anchor(db, session_id)],
    )
    db.commit()
    assert successor_of(db, memory_id) == successor_id
