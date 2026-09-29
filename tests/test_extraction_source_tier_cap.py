"""
Task 7.3: the per-source tier cap (e.g. Codex), wired through admit_decision and
persist_extraction alike.
"""

from __future__ import annotations

import sqlite3

import pytest

from palaver.extract.persist import (
    Extraction,
    GroundedClaim,
    persist_extraction,
)
from palaver.extract.quote_gate import (
    SOURCE_TIER_CAPS,
    QuoteNotGroundedError,
    admit_decision,
    cap_tier_for_source,
)
from palaver.memory.tiers import (
    TIER_OBSERVER_INFERENCE,
    TIER_OBSERVER_SPECULATION,
    TIER_USER_INSTRUCTION,
)
from tests._extraction_support import ORIGIN, _chunk_id, _memory_counts, _replayed, _user_record

# --- task 7.3: the Codex tier cap, wired to the actual write path ------------
#
# Task 7.1 built the cap and reported, unprompted, that its "no other code path
# lifts the cap" condition was *vacuously* satisfied: nothing outside
# `codex.py` called the capping function at all, so no Codex-sourced write was
# capped end to end. These tests are that wiring's gate, and every one of them
# reads the tier back out of `memories` rather than off a return value — a
# function that returns 4 while writing 1 would pass the weaker check.


def _written_tier(conn: sqlite3.Connection, memory_id: int) -> int:
    (tier,) = conn.execute("SELECT tier FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return tier


def test_a_codex_sourced_decision_lands_at_tier_four_in_the_database(tmp_path):
    """The same statement, the same quote, the same human-channel chunk, written
    twice: from Claude Code it earns tier-1, from Codex it is written at tier-4.

    The Claude Code half is the control that makes the Codex half mean
    something. Without it, "the row says 4" also passes against a write path
    that lost the tier entirely, or one that caps everything — and capping
    everything would silently demote every memory this project has ever
    written, which is a defect the Codex assertion alone cannot see.
    """
    conn, project_id, session_id = _replayed(
        tmp_path, [_user_record("cap the retry budget at three attempts")]
    )
    try:
        chunk_id = _chunk_id(conn, 1)
        statement = "cap the retry budget at three attempts"

        from_claude_code = admit_decision(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=statement,
            quote=statement,
            transcript_chunk_id=chunk_id,
            origin=ORIGIN,
        )
        from_codex = admit_decision(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=statement,
            quote=statement,
            transcript_chunk_id=chunk_id,
            origin=ORIGIN,
            source="codex",
        )

        assert _written_tier(conn, from_claude_code.memory_id) == TIER_USER_INSTRUCTION
        assert _written_tier(conn, from_codex.memory_id) == TIER_OBSERVER_INFERENCE
        assert from_codex.grounded.tier == TIER_OBSERVER_INFERENCE
        assert "codex source cap" in from_codex.grounded.reason
    finally:
        conn.close()


def test_the_cap_binds_through_persist_extraction_not_only_at_the_gate(tmp_path):
    """The write path a real extraction pass takes — `persist_extraction`, which
    is what the observer daemon calls — carries the source through to the gate.

    Task 7.1's cap was enforced at a function nothing called. This asserts the
    connection exists at the layer above, so a Codex-sourced pass cannot reach
    `memories` through the persist API at a tier it did not earn.
    """
    conn, project_id, session_id = _replayed(
        tmp_path, [_user_record("route staging traffic through the canary")]
    )
    try:
        statement = "route staging traffic through the canary"
        claim = GroundedClaim(
            statement=statement, quote=statement, transcript_chunk_id=_chunk_id(conn, 1)
        )

        capped = persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=Extraction(decisions=[claim]),
            source="codex",
        )
        uncapped = persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=Extraction(resolved_questions=[claim]),
        )

        assert _written_tier(conn, capped.decision_memory_ids[0]) == TIER_OBSERVER_INFERENCE
        assert (
            _written_tier(conn, uncapped.resolved_question_memory_ids[0]) == TIER_USER_INSTRUCTION
        )
    finally:
        conn.close()


def test_the_cap_demotes_and_never_promotes(tmp_path):
    """A tier already weaker than the cap is left alone. The demotion is a `max`
    over tier numbers, so a tier-5 speculation from Codex stays tier-5 rather
    than being *promoted* to 4 by the thing meant to weaken it."""
    assert cap_tier_for_source(TIER_USER_INSTRUCTION, source="codex") == TIER_OBSERVER_INFERENCE
    assert cap_tier_for_source(TIER_OBSERVER_INFERENCE, source="codex") == TIER_OBSERVER_INFERENCE
    assert (
        cap_tier_for_source(TIER_OBSERVER_SPECULATION, source="codex") == TIER_OBSERVER_SPECULATION
    )
    assert cap_tier_for_source(TIER_USER_INSTRUCTION) == TIER_USER_INSTRUCTION


def test_an_unknown_source_is_not_silently_capped(tmp_path):
    """A source with no entry in `SOURCE_TIER_CAPS` writes at the tier it earned.

    The table is explicit precisely so this is a reviewable decision rather
    than an accident of import order: OpenCode is uncapped today because
    nobody has claimed it needs a cap, and that shows up here as a passing
    assertion someone has to change on purpose.
    """
    conn, project_id, session_id = _replayed(
        tmp_path, [_user_record("keep the nightly job on the old queue")]
    )
    try:
        statement = "keep the nightly job on the old queue"
        admitted = admit_decision(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=statement,
            quote=statement,
            transcript_chunk_id=_chunk_id(conn, 1),
            origin=ORIGIN,
            source="opencode",
        )

        assert "opencode" not in SOURCE_TIER_CAPS
        assert _written_tier(conn, admitted.memory_id) == TIER_USER_INSTRUCTION
    finally:
        conn.close()


def test_a_capped_source_still_has_to_ground_its_quote(tmp_path):
    """The cap is applied after grounding, not instead of it. An ungrounded
    Codex quote raises and writes nothing — a capped source is not a source
    whose claims are waved through at a lower tier."""
    conn, project_id, session_id = _replayed(tmp_path, [_user_record("freeze the schema")])
    try:
        before = _memory_counts(conn)

        with pytest.raises(QuoteNotGroundedError):
            admit_decision(
                conn,
                project_id=project_id,
                session_id=session_id,
                statement="freeze the schema",
                quote="freeze the schemas",
                transcript_chunk_id=_chunk_id(conn, 1),
                origin=ORIGIN,
                source="codex",
            )

        assert _memory_counts(conn) == before
    finally:
        conn.close()
