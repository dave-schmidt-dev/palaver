"""Tests for project/session memory scoping (task 2.3: `palaver.memory.scope`).

Per the plan's standing rule, every negative assertion here is paired with a
positive control proving the same mechanism is live, not merely agreeing
with whatever the code already looks like. `tests/test_memory.py` and
`tests/test_invariants.py` are the references for this pattern.

**The collapse this module exists to catch.** Project scope and session
scope are different questions; a helper that quietly answers one when the
other was asked is silently wrong, not loudly wrong. The trap a naive
implementation falls into is deriving project scope by joining through
`sessions` (`JOIN sessions ON memories.session_id = sessions.id WHERE
sessions.project_id = ?`) instead of reading `memories.project_id` directly.
That trap is invisible if a test only ever writes memories that carry a
`session_id` — `memories.session_id` is nullable (`write_memory` accepts
`session_id=None` for a project-level observation, per
`tests/test_memory.py::test_write_memory_defaults_session_id_and_supersedes_to_null`),
so a join-based implementation silently drops every project-level memory
from a project-scoped read. `test_project_scope_spans_two_sessions_and_a_session_less_memory`
below writes one memory per session under two distinct sessions of the same
project, plus a third memory with `session_id=None`, and asserts all three
come back — a test using only one session, or only session-attributed
memories, would pass against the join-based bug and prove nothing.

**The sibling-session trap.** A session-scoped test whose "sibling" session
belongs to a *different* project passes even against a helper that only
ever filters by project — that is the exact collapse under test, disguised
as a pass. `test_session_scope_excludes_a_sibling_session_of_the_same_project`
below puts the sibling under the *same* project as the target session, and
populates it with its own memory, so the assertion that it is absent is
meaningful.

**Unknown scope targets.** `read_memories` resolves `project`/`session`
against `projects`/`sessions` before ever touching `memories`, and raises
`LookupError` for a name/id that matches no row — rather than returning an
empty list indistinguishable from "this real project/session just has no
memories yet." `test_unknown_project_name_raises_lookup_error` and
`test_unknown_session_id_raises_lookup_error` cover that, each with a
positive control proving a *known* target on the same connection still
resolves.

This repository is public. Every statement, name, and identifier in these
tests is invented for the test; none of it is derived from a real observed
session.
"""

from __future__ import annotations

import itertools
import sqlite3

import pytest

from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.scope import read_memories
from palaver.memory.tiers import TIER_AGENT_CONCLUSION, TIER_OBSERVED_RESULT
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()


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


def _seed_chunk(conn: sqlite3.Connection, session_id: int, content: str) -> int:
    """Insert a transcript_chunks row at a fresh, globally-unique seq.

    A monotonically increasing counter shared across every call in this
    module trivially satisfies transcript_chunks' `UNIQUE (session_id,
    seq)` constraint even when a test seeds more than one chunk under the
    same session_id, without each call site having to track its own
    per-session counter.
    """
    return conn.execute(
        "INSERT INTO transcript_chunks(session_id, seq, role, content) VALUES (?, ?, ?, ?)",
        (session_id, _next_seq(), "user", content),
    ).lastrowid


def _write_fixture_memory(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    evidence_session_id: int,
    statement: str,
    session_id: int | None,
    tier: int = TIER_OBSERVED_RESULT,
) -> int:
    """Write a memory anchored to a fresh, invented transcript chunk.

    `evidence_session_id` names the session the evidence chunk is stored
    under; `session_id` is the memory's own scope attribution, which may
    legitimately be `None` even though the evidence itself always lives
    under some real session (transcript_chunks.session_id is NOT NULL).
    """
    content = f"fixture: invented transcript line supporting '{statement}'"
    chunk_id = _seed_chunk(conn, evidence_session_id, content)
    return write_memory(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement=statement,
        origin="observer",
        tier=tier,
        evidence=[
            EvidenceAnchor(transcript_chunk_id=chunk_id, start_offset=0, end_offset=len(content))
        ],
    )


# =============================================================================
# Scope argument validation: exactly one of project/session, never defaulted
# =============================================================================


def test_read_memories_raises_with_neither_scope(db):
    """Calling read_memories with neither project nor session raises ValueError.

    This is the acceptance-gate case: a helper that defaults to "everything"
    or to some implicit current project/session would return a confidently
    wrong answer instead of failing loudly.

    LAYER PROOF: this discriminates the specific `(project is None) ==
    (session is None)` check, not just "the function errors somehow." Delete
    that check and `session=None` falls through to the `else` branch, which
    queries `sessions WHERE id = NULL`, finds no row, and raises
    `LookupError` — a different exception that `pytest.raises(ValueError,
    match="exactly one")` does not accept. Only the guard clause satisfies
    this test.
    """
    with pytest.raises(ValueError, match="exactly one"):
        read_memories(db)


def test_read_memories_raises_with_both_scopes(db):
    """Calling read_memories with both project and session raises ValueError.

    The contract is "exactly one", not "project wins" or "session wins" — a
    caller passing both has a bug that should surface, not get silently
    resolved by an undocumented precedence rule.

    LAYER PROOF: without the `(project is None) == (session is None)`
    guard, this call falls through to the `if project is not None` branch
    and returns whatever the project alone resolves to — silently picking
    "project wins" rather than raising. Only the guard clause raises at all.
    """
    project_id = _seed_project(db, "fixture-both-scopes-project")
    session_id = _seed_session(db, project_id, "fixture-both-scopes-session")
    db.commit()

    with pytest.raises(ValueError, match="exactly one"):
        read_memories(db, project="fixture-both-scopes-project", session=session_id)


# =============================================================================
# Project scope: every session of the project, including session-less rows
# =============================================================================


def test_project_scope_spans_two_sessions_and_a_session_less_memory(db):
    """A project-scoped read returns memories from every session of that project.

    Writes one memory under each of two distinct sessions of the same
    project, plus a third memory with session_id=None (a project-level
    observation), and asserts all three come back. Proving this with two
    *different* session ids — not the same session twice — is the point:
    a helper that quietly collapsed project scope into "the current
    session" would still pass a test written against a single session.
    A session-less memory catches the other collapse: deriving project
    scope by joining through `sessions` instead of reading
    `memories.project_id` directly, which would silently drop this row.
    Also asserts the id order directly (they were written a, b,
    project-level, in that order) rather than only comparing as a set — the
    module docstring promises `ORDER BY id`, and every other multi-row
    assertion in this file uses a set or single element, so this is the
    one place that promise is actually locked in.
    """
    project_id = _seed_project(db, "fixture-multi-session-project")
    session_a = _seed_session(db, project_id, "fixture-session-a")
    session_b = _seed_session(db, project_id, "fixture-session-b")
    db.commit()

    memory_a = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_a,
        session_id=session_a,
        statement="fixture: observation recorded under session A",
    )
    memory_b = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_b,
        session_id=session_b,
        statement="fixture: observation recorded under session B",
    )
    memory_project_level = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_a,
        session_id=None,
        statement="fixture: a project-level observation with no owning session",
    )
    db.commit()

    results = read_memories(db, project="fixture-multi-session-project")

    assert [row["id"] for row in results] == [memory_a, memory_b, memory_project_level]
    assert {row["session_id"] for row in results} == {session_a, session_b, None}


def test_project_scope_excludes_a_memory_from_a_different_project(db):
    """A project-scoped read never returns a memory belonging to another project.

    Positive control for `test_project_scope_spans_two_sessions_and_a_session_less_memory`:
    without this, a helper that ignores project entirely and returns every
    `memories` row would still pass the "spans two sessions" test above.
    """
    project_id = _seed_project(db, "fixture-project-isolation-target")
    session_id = _seed_session(db, project_id, "fixture-project-isolation-session")
    other_project_id = _seed_project(db, "fixture-project-isolation-other")
    other_session_id = _seed_session(
        db, other_project_id, "fixture-project-isolation-other-session"
    )
    db.commit()

    target_memory = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=session_id,
        statement="fixture: an observation that belongs to the target project",
    )
    _write_fixture_memory(
        db,
        project_id=other_project_id,
        evidence_session_id=other_session_id,
        session_id=other_session_id,
        statement="fixture: an observation that belongs to a different project entirely",
    )
    db.commit()

    results = read_memories(db, project="fixture-project-isolation-target")

    assert [row["id"] for row in results] == [target_memory]


# =============================================================================
# Session scope: only the named session, never a sibling
# =============================================================================


def test_session_scope_excludes_a_sibling_session_of_the_same_project(db):
    """A session-scoped read never returns a row belonging to a sibling session.

    The sibling is a real, populated session of the *same* project, not an
    empty one and not a session of a different project — either shortcut
    would let a helper that only filters by project pass this test. Writing
    a memory into the sibling and asserting it does NOT come back is what
    proves the mechanism is live: a test against an empty sibling would
    pass even if session scoping were not implemented at all.
    """
    project_id = _seed_project(db, "fixture-session-isolation-project")
    target_session = _seed_session(db, project_id, "fixture-target-session")
    sibling_session = _seed_session(db, project_id, "fixture-sibling-session")
    db.commit()

    target_memory = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=target_session,
        session_id=target_session,
        statement="fixture: an observation that belongs to the target session",
        tier=TIER_AGENT_CONCLUSION,
    )
    sibling_memory = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=sibling_session,
        session_id=sibling_session,
        statement="fixture: an observation that belongs to a sibling session",
    )
    db.commit()

    results = read_memories(db, session=target_session)
    result_ids = {row["id"] for row in results}

    assert result_ids == {target_memory}
    assert sibling_memory not in result_ids


def test_session_scope_excludes_a_project_level_memory_with_no_session(db):
    """A session-scoped read never returns a project-level memory (session_id IS NULL).

    Companion to the project-scope test that asserts a session-less memory
    IS returned under project scope — here the same row must NOT appear
    under a session-scoped read of any session in that project, since it
    belongs to none of them specifically. The positive control — a second
    memory written *under* `session_id` — is not decoration: without it,
    the only row in this database is the session-less one, so a
    `read_memories(session=...)` that always returns `[]` (session scoping
    not implemented at all) would pass the bare `results == []` assertion
    for the wrong reason. Asserting the in-session memory comes back too is
    what makes the exclusion of the session-less one mean something.
    """
    project_id = _seed_project(db, "fixture-session-scope-nullness-project")
    session_id = _seed_session(db, project_id, "fixture-session-scope-nullness-session")
    db.commit()

    project_level_memory = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=None,
        statement="fixture: a project-level observation with no owning session",
    )
    in_session_memory = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=session_id,
        statement="fixture: an observation that does belong to this session",
    )
    db.commit()

    results = read_memories(db, session=session_id)
    result_ids = [row["id"] for row in results]

    assert result_ids == [in_session_memory]
    assert project_level_memory not in result_ids


# =============================================================================
# Unknown scope targets raise, rather than returning an empty list
# =============================================================================


def test_unknown_project_name_raises_lookup_error(db):
    """A project name matching no row raises LookupError, not an empty list.

    Positive control: a known project name, seeded on the same connection,
    still resolves and returns its memory — proving the raise above is
    about the specific unknown name, not a helper that has stopped working.
    """
    with pytest.raises(LookupError, match="fixture-nonexistent-project"):
        read_memories(db, project="fixture-nonexistent-project")

    project_id = _seed_project(db, "fixture-known-project")
    session_id = _seed_session(db, project_id, "fixture-known-project-session")
    db.commit()
    memory_id = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=session_id,
        statement="fixture: an observation under a project that does exist",
    )
    db.commit()

    assert [row["id"] for row in read_memories(db, project="fixture-known-project")] == [memory_id]


def test_unknown_session_id_raises_lookup_error(db):
    """A session id matching no row raises LookupError, not an empty list.

    Positive control: a known session id, seeded on the same connection,
    still resolves and returns its memory — proving the raise above is
    about the specific unknown id, not a helper that has stopped working.
    """
    project_id = _seed_project(db, "fixture-known-session-project")
    session_id = _seed_session(db, project_id, "fixture-known-session")
    db.commit()
    unknown_session_id = session_id + 1_000_000

    with pytest.raises(LookupError, match=str(unknown_session_id)):
        read_memories(db, session=unknown_session_id)

    memory_id = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=session_id,
        statement="fixture: an observation under a session that does exist",
    )
    db.commit()

    assert [row["id"] for row in read_memories(db, session=session_id)] == [memory_id]


# =============================================================================
# Result shape
# =============================================================================


def test_read_memories_returns_tier_and_statement_for_downstream_provenance_display(db):
    """Returned rows carry `tier` and `statement`, not just an id.

    Phase 6's read tools (task 6.1) must surface provenance tier on every
    result; a helper that returned bare ids would push that lookup back
    onto every caller. This only checks the shape this module promises in
    its own docstring, not task 6.1's tool-layer behavior.
    """
    project_id = _seed_project(db, "fixture-result-shape-project")
    session_id = _seed_session(db, project_id, "fixture-result-shape-session")
    db.commit()
    memory_id = _write_fixture_memory(
        db,
        project_id=project_id,
        evidence_session_id=session_id,
        session_id=session_id,
        statement="fixture: an observation whose full row shape is checked",
        tier=TIER_AGENT_CONCLUSION,
    )
    db.commit()

    (row,) = read_memories(db, project="fixture-result-shape-project")

    assert row["id"] == memory_id
    assert row["statement"] == "fixture: an observation whose full row shape is checked"
    assert row["tier"] == TIER_AGENT_CONCLUSION
    assert row["project_id"] == project_id
    assert row["session_id"] == session_id
