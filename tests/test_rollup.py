"""Tests for retention pruning (task 4.3: `palaver.store.rollup`).

Per the plan's standing rule, every negative assertion here is paired with a
positive control proving the same mechanism is live rather than merely
agreeing with whatever the code already does. That rule bites unusually hard
on a pruning pass, because **the do-nothing implementation passes almost
every safety assertion in this file**. "The evidence-referenced chunk is still
there," "the memory still resolves," "`memories` was not written" are all true
of a `prune_ephemeral` whose body is `pass`. So every test that asserts
something survived also asserts, in the same test, that something else of the
same shape did not — that is what makes the survival meaningful.

`test_pruning_keeps_a_referenced_chunk_and_removes_an_unreferenced_one`
carries the plan's explicit version of that pairing, and deliberately keeps
both halves in one test with one fixture: split across two tests, a fixture
bug that seeds the wrong timestamps lets both pass while the pair proves
nothing.

**Why the timestamp helper here is not `rollup.retention_cutoff`.** Fixtures
below stamp rows using this module's own `_stamp`, written against the format
in `schema._CREATED_AT_DEFAULT` rather than by calling the function under
test. Seeding fixtures with the same formatter the cutoff uses would cancel a
format bug against itself — every comparison would still line up while both
sides drifted away from what SQLite actually writes.
`test_the_cutoff_format_matches_what_sqlite_writes` closes the loop by
comparing a real SQLite-written timestamp against the module's cutoff.

This repository is public. Every statement, name, and identifier in these
tests is invented for the test; none of it is derived from a real observed
session.
"""

from __future__ import annotations

import itertools
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from palaver.memory.evidence import EvidenceAnchor, resolve_evidence
from palaver.memory.tiers import TIER_OBSERVED_RESULT
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from palaver.store.rollup import (
    DEFAULT_RETENTION_DAYS,
    PruneReport,
    prune_ephemeral,
    retention_cutoff,
)
from palaver.store.schema import search

#: Fixed clock every test measures its retention window back from, so no
#: assertion here depends on wall time or on how long the suite takes to run.
NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)

#: The window every test uses unless it is specifically testing the default.
WINDOW_DAYS = 7.0


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()


def _stamp(days_ago: float) -> str:
    """Render `NOW` minus `days_ago` in the format SQLite's DEFAULT writes.

    Written independently of `rollup._TIMESTAMP_SECONDS_FORMAT` on purpose —
    see this module's docstring.
    """
    moment = NOW - timedelta(days=days_ago)
    return f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{moment.microsecond // 1000:03d}Z"


def _seed_project(conn: sqlite3.Connection, name: str) -> int:
    return conn.execute(
        "INSERT INTO projects(name, path) VALUES (?, ?)",
        (name, f"/tmp/fixture/{name}"),
    ).lastrowid


def _seed_session(conn: sqlite3.Connection, project_id: int, external_id: str) -> int:
    return conn.execute(
        "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
        (project_id, "claude-code", external_id),
    ).lastrowid


_next_seq = itertools.count(1).__next__


def _seed_chunk(conn: sqlite3.Connection, session_id: int, content: str, days_ago: float) -> int:
    """Insert a transcript chunk stamped `days_ago` before `NOW`.

    The globally unique `seq` counter satisfies `UNIQUE (session_id, seq)`
    without each call site tracking a per-session counter, matching the helper
    in `tests/test_memory_scope.py`.
    """
    return conn.execute(
        "INSERT INTO transcript_chunks(session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, _next_seq(), "user", content, _stamp(days_ago)),
    ).lastrowid


def _seed_state(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    session_id: int | None,
    key: str,
    value: str,
    days_ago: float,
) -> int:
    return conn.execute(
        "INSERT INTO current_state(project_id, session_id, key, value, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (project_id, session_id, key, value, _stamp(days_ago)),
    ).lastrowid


def _seed_event(conn: sqlite3.Connection, session_id: int, payload: str, days_ago: float) -> int:
    return conn.execute(
        "INSERT INTO events(session_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (session_id, "fixture", payload, _stamp(days_ago)),
    ).lastrowid


def _anchor_memory(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    session_id: int,
    chunk_id: int,
    content: str,
    statement: str,
    supersedes: int | None = None,
) -> int:
    return write_memory(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement=statement,
        origin="observer",
        tier=TIER_OBSERVED_RESULT,
        supersedes=supersedes,
        evidence=[
            EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=0, end_offset=len(content))
        ],
    )


def _evidence_ids(conn: sqlite3.Connection, memory_id: int) -> list[int]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id = ? ORDER BY id", (memory_id,)
        )
    ]


def _chunk_ids(conn: sqlite3.Connection, session_id: int) -> set[int]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT id FROM transcript_chunks WHERE session_id = ?", (session_id,)
        )
    }


def _prune(conn: sqlite3.Connection, **overrides) -> PruneReport:
    """Run a pass against the fixed clock and the short shared window."""
    kwargs = {"retention_days": WINDOW_DAYS, "now": NOW}
    kwargs.update(overrides)
    return prune_ephemeral(conn, **kwargs)


# =============================================================================
# The retention boundary itself
# =============================================================================


def test_the_cutoff_format_matches_what_sqlite_writes(db):
    """A cutoff SQLite cannot compare against is a prune that silently does nothing."""
    project_id = _seed_project(db, "cutoff-format")
    session_id = _seed_session(db, project_id, "session-cutoff")
    # No explicit created_at: let the schema DEFAULT write it.
    db.execute(
        "INSERT INTO transcript_chunks(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
        (session_id, _next_seq(), "user", "fixture: invented line"),
    )
    written = db.execute("SELECT created_at FROM transcript_chunks").fetchone()[0]
    cutoff = retention_cutoff(0, NOW)

    assert len(cutoff) == len(written)
    assert cutoff.endswith("Z")
    # Same shape, digit for digit, so lexicographic comparison is chronological.
    assert [c.isdigit() for c in cutoff] == [c.isdigit() for c in written]
    # Sub-second digits, pinned explicitly. Every other fixture in this module
    # lands on a whole second, so the length check above cannot see the
    # difference between SQLite's three fractional digits and Python's six —
    # and a six-digit cutoff sorts *after* every same-second stored row.
    assert retention_cutoff(0, NOW.replace(microsecond=123456)) == "2026-08-15T12:00:00.123Z"


def test_a_negative_retention_window_is_rejected(db):
    with pytest.raises(ValueError, match="must not be negative"):
        retention_cutoff(-1, NOW)
    # Positive control: zero is a legitimate window (prune everything written
    # before this instant), so the guard is rejecting the sign, not the edge.
    assert retention_cutoff(0, NOW).startswith("2026-08-15T12:00:00")


def test_the_default_retention_window_is_thirty_days():
    """Pin the policy constant, rather than asserting it equals itself.

    A test that only compared `retention_cutoff()`'s default against
    `DEFAULT_RETENTION_DAYS` would pass for any value the module happens to
    hold, which is precisely the vacuous shape a mutation run caught in
    `tests/test_slots.py` for `DEFAULT_SLOT_COUNT`.
    """
    assert DEFAULT_RETENTION_DAYS == 30.0
    assert retention_cutoff(now=NOW) == retention_cutoff(30.0, NOW)


def test_a_naive_now_is_read_as_utc():
    naive = NOW.replace(tzinfo=None)
    assert retention_cutoff(WINDOW_DAYS, naive) == retention_cutoff(WINDOW_DAYS, NOW)


# =============================================================================
# current_state: the retention window is the whole rule
# =============================================================================


def test_prune_leaves_only_current_state_rows_inside_the_retention_window(db):
    project_id = _seed_project(db, "state-window")
    session_id = _seed_session(db, project_id, "session-state")
    _seed_state(
        db,
        project_id=project_id,
        session_id=session_id,
        key="current_task",
        value="fixture: invented in-window value",
        days_ago=1,
    )
    _seed_state(
        db,
        project_id=project_id,
        session_id=session_id,
        key="blockers_now",
        value="fixture: invented out-of-window value",
        days_ago=30,
    )

    report = _prune(db)

    remaining = {row[0]: row[1] for row in db.execute("SELECT key, value FROM current_state")}
    assert remaining == {"current_task": "fixture: invented in-window value"}
    assert report.current_state_rows_deleted == 1


def test_prune_deletes_nothing_when_every_row_is_inside_the_window(db):
    """The do-nothing control, paired with a pass that does something."""
    project_id = _seed_project(db, "state-fresh")
    session_id = _seed_session(db, project_id, "session-fresh")
    _seed_state(
        db,
        project_id=project_id,
        session_id=session_id,
        key="current_task",
        value="fixture: invented value",
        days_ago=2,
    )
    _seed_chunk(db, session_id, "fixture: invented recent line", days_ago=2)

    quiet = _prune(db)
    assert quiet.total_rows_deleted == 0

    # Positive control: the same store, the same rows, a shorter window — if
    # this also deleted nothing the assertion above would be meaningless.
    busy = _prune(db, retention_days=1.0)
    assert busy.total_rows_deleted == 2


# =============================================================================
# The evidence exclusion: the reason this module is not an age query
# =============================================================================


def test_pruning_keeps_a_referenced_chunk_and_removes_an_unreferenced_one(db):
    """The plan's explicit pair, in one test, over one session and one fixture.

    Both chunks are out of the retention window and belong to the same
    session. The only difference between them is that one is named by a
    `memory_evidence` row. Split into two tests, the survival assertion alone
    passes against a prune that deletes nothing at all.
    """
    project_id = _seed_project(db, "evidence-exclusion")
    session_id = _seed_session(db, project_id, "session-mixed")
    referenced_text = "fixture: invented line the memory cites"
    referenced = _seed_chunk(db, session_id, referenced_text, days_ago=30)
    unreferenced = _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=30)
    _anchor_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        chunk_id=referenced,
        content=referenced_text,
        statement="fixture: invented durable claim",
    )

    report = _prune(db)

    surviving = _chunk_ids(db, session_id)
    assert referenced in surviving
    assert unreferenced not in surviving
    assert report.transcript_chunks_deleted == 1
    assert report.transcript_chunks_kept_for_evidence == 1


def test_every_memory_of_a_pruned_session_still_resolves_its_evidence(db):
    project_id = _seed_project(db, "resolvable")
    session_id = _seed_session(db, project_id, "session-resolvable")
    statements = ("fixture: invented claim one", "fixture: invented claim two")
    expected = {}
    for statement in statements:
        content = f"fixture: invented transcript line supporting '{statement}'"
        chunk_id = _seed_chunk(db, session_id, content, days_ago=45)
        memory_id = _anchor_memory(
            db,
            project_id=project_id,
            session_id=session_id,
            chunk_id=chunk_id,
            content=content,
            statement=statement,
        )
        expected[memory_id] = content
    _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=45)

    report = _prune(db)

    # Positive control: this session really was pruned. Without it, every
    # resolution below is trivially satisfied by a pass that did nothing.
    assert report.transcript_chunks_deleted == 1

    for memory_id, content in expected.items():
        evidence_ids = _evidence_ids(db, memory_id)
        assert evidence_ids, f"memory {memory_id} lost its evidence rows"
        for evidence_id in evidence_ids:
            assert resolve_evidence(db, evidence_id) == content


def test_a_superseded_memory_still_resolves_its_evidence_after_a_prune(db):
    """Supersession is derived, so it must not make a chunk look collectable.

    `memories` has no stored superseded flag (INV-4); the state is derived by
    the `superseded_memories` view. An exclusion written against non-superseded
    memories would leave this chunk collectable, and `resolve_evidence` would
    then raise for a memory the store still reports as present.
    """
    project_id = _seed_project(db, "superseded")
    session_id = _seed_session(db, project_id, "session-superseded")
    content = "fixture: invented line behind a later correction"
    chunk_id = _seed_chunk(db, session_id, content, days_ago=60)
    predecessor = _anchor_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        chunk_id=chunk_id,
        content=content,
        statement="fixture: invented superseded claim",
    )
    successor_text = "fixture: invented line carrying the correction"
    successor_chunk = _seed_chunk(db, session_id, successor_text, days_ago=1)
    _anchor_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        chunk_id=successor_chunk,
        content=successor_text,
        statement="fixture: invented superseding claim",
        supersedes=predecessor,
    )
    _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=60)

    # The fixture is what it claims to be: the predecessor really is superseded.
    assert db.execute(
        "SELECT 1 FROM superseded_memories WHERE memory_id = ?", (predecessor,)
    ).fetchone()

    report = _prune(db)
    assert report.transcript_chunks_deleted == 1

    (evidence_id,) = _evidence_ids(db, predecessor)
    assert resolve_evidence(db, evidence_id) == content


def test_prune_never_deletes_a_memory_evidence_row(db):
    """Deleting an anchor drops a memory below INV-6's one-link floor."""
    project_id = _seed_project(db, "evidence-rows")
    session_id = _seed_session(db, project_id, "session-evidence-rows")
    content = "fixture: invented cited line"
    chunk_id = _seed_chunk(db, session_id, content, days_ago=90)
    _anchor_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        chunk_id=chunk_id,
        content=content,
        statement="fixture: invented claim",
    )
    _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=90)
    before = db.execute("SELECT id, memory_id, transcript_chunk_id FROM memory_evidence").fetchall()

    report = _prune(db)
    assert report.transcript_chunks_deleted == 1

    after = db.execute("SELECT id, memory_id, transcript_chunk_id FROM memory_evidence").fetchall()
    assert after == before


def test_prune_never_deletes_an_events_row(db):
    """Events can carry evidence anchors too, and this pass does not touch them."""
    project_id = _seed_project(db, "events-untouched")
    session_id = _seed_session(db, project_id, "session-events")
    _seed_event(db, session_id, "fixture: invented event payload", days_ago=120)
    _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=120)

    report = _prune(db)
    assert report.transcript_chunks_deleted == 1

    assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


# =============================================================================
# memories is append-only, and the pass must not write it at all
# =============================================================================


def test_prune_issues_no_write_against_memories(db):
    """Bullet 2, guarded by triggers rather than by a row count.

    A before/after count comparison passes against a pass that deletes a
    memory and inserts a replacement — which INV-4 also forbids. The temporary
    triggers below abort on *either* operation, so the assertion is about
    statements issued, not about the net shape of the table afterward.
    """
    project_id = _seed_project(db, "memories-untouched")
    session_id = _seed_session(db, project_id, "session-memories")
    content = "fixture: invented cited line"
    chunk_id = _seed_chunk(db, session_id, content, days_ago=200)
    memory_id = _anchor_memory(
        db,
        project_id=project_id,
        session_id=session_id,
        chunk_id=chunk_id,
        content=content,
        statement="fixture: invented claim",
    )
    _seed_chunk(db, session_id, "fixture: invented uncited line", days_ago=200)
    _seed_state(
        db,
        project_id=project_id,
        session_id=session_id,
        key="current_task",
        value="fixture: invented stale value",
        days_ago=200,
    )
    before = db.execute("SELECT id, statement, tier, created_at FROM memories").fetchall()

    db.execute(
        "CREATE TEMP TRIGGER guard_memories_delete BEFORE DELETE ON memories "
        "BEGIN SELECT RAISE(ABORT, 'prune deleted a memory'); END"
    )
    db.execute(
        "CREATE TEMP TRIGGER guard_memories_insert BEFORE INSERT ON memories "
        "BEGIN SELECT RAISE(ABORT, 'prune inserted a memory'); END"
    )
    try:
        report = _prune(db)
    finally:
        db.execute("DROP TRIGGER guard_memories_delete")
        db.execute("DROP TRIGGER guard_memories_insert")

    # Positive control: the guarded pass was not a no-op, and the guards are
    # live — the same triggers abort a deliberate write.
    assert report.total_rows_deleted == 2
    assert db.execute("SELECT id, statement, tier, created_at FROM memories").fetchall() == before

    db.execute(
        "CREATE TEMP TRIGGER guard_memories_insert BEFORE INSERT ON memories "
        "BEGIN SELECT RAISE(ABORT, 'prune inserted a memory'); END"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="inserted a memory"):
            db.execute(
                "INSERT INTO memories(project_id, statement, origin, tier) VALUES (?, ?, ?, ?)",
                (project_id, "fixture: invented control", "test", TIER_OBSERVED_RESULT),
            )
    finally:
        db.execute("DROP TRIGGER guard_memories_insert")
    # And the delete half of the guarantee is live too, from the schema's own
    # `memories_no_delete` trigger — so "the pass issued no DELETE" is a claim
    # about a statement that would have aborted had it been issued.
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    assert db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
    assert memory_id in {row[0] for row in before}


# =============================================================================
# Index consistency and progress reporting
# =============================================================================


def test_a_pruned_chunk_leaves_no_orphan_in_the_full_text_index(db):
    """`schema.search` must not keep matching a chunk that no longer exists.

    The migration-2 `_ad` trigger is what removes the row from the FTS5
    external-content index on delete. If a prune ever bypassed it — a `DELETE`
    on a temp copy, a table rebuild — search would return a hit whose id
    resolves to nothing.
    """
    project_id = _seed_project(db, "fts-consistency")
    session_id = _seed_session(db, project_id, "session-fts")
    kept_text = "fixture: invented persistent zarquon line"
    kept = _seed_chunk(db, session_id, kept_text, days_ago=1)
    _seed_chunk(db, session_id, "fixture: invented expiring zarquon line", days_ago=90)
    assert len(search(db, "zarquon")) == 2

    report = _prune(db)
    assert report.transcript_chunks_deleted == 1

    hits = search(db, "zarquon")
    assert [(hit["source"], hit["id"]) for hit in hits] == [("transcript_chunks", kept)]
    assert hits[0]["text"] == kept_text


def test_prune_reports_progress_for_each_delete_phase(db):
    """INV-1: a pass over a large store is never a silent wait."""
    project_id = _seed_project(db, "progress")
    session_id = _seed_session(db, project_id, "session-progress")
    _seed_chunk(db, session_id, "fixture: invented expiring line", days_ago=90)
    _seed_state(
        db,
        project_id=project_id,
        session_id=session_id,
        key="current_task",
        value="fixture: invented stale value",
        days_ago=90,
    )
    messages: list[str] = []

    _prune(db, on_status=messages.append)

    assert any("current_state" in message for message in messages)
    assert any("transcript_chunks" in message for message in messages)
    # The cutoff is reported before any delete, so a slow pass says what it is
    # about to do rather than only what it did.
    assert messages[0].endswith(retention_cutoff(WINDOW_DAYS, NOW))


def test_prune_without_a_status_channel_still_runs(db):
    """`on_status` is optional; omitting it must not be an error path."""
    project_id = _seed_project(db, "no-status")
    session_id = _seed_session(db, project_id, "session-no-status")
    _seed_chunk(db, session_id, "fixture: invented expiring line", days_ago=90)

    report = prune_ephemeral(db, retention_days=WINDOW_DAYS, now=NOW)

    assert report.transcript_chunks_deleted == 1
