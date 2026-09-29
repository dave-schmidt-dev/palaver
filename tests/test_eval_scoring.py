"""
The scoring primitives: is_decision_grounded (incl. parity with quote_gate's
ground_quote), score_leg, and is_degenerate_extraction.
"""

from __future__ import annotations

from palaver.eval.harness import (
    Decision,
    Extraction,
    FixtureLabel,
    is_decision_grounded,
    is_degenerate_extraction,
    score_leg,
)
from palaver.extract.normalize import normalize_path
from palaver.extract.quote_gate import ground_quote
from palaver.replay import replay
from palaver.store.migrate import connect
from tests._eval_support import FIXTURES_DIR

# =============================================================================
# is_decision_grounded: fabrication / misattribution check
# =============================================================================


def test_is_decision_grounded_true_for_human_line():
    transcript = "HUMAN: refactor the auth module\nAGENT: the auth module is refactored\n"
    assert is_decision_grounded(transcript, "refactor the auth module") is True


def test_is_decision_grounded_false_for_injected_line():
    """Misattribution: the quote is real text, but on an INJECTED, not HUMAN, line."""
    transcript = "HUMAN: refactor the auth module\nINJECTED: <command-name>/status</command-name>\n"
    assert is_decision_grounded(transcript, "<command-name>/status</command-name>") is False


def test_is_decision_grounded_false_for_agent_line():
    """Fabrication: a real AGENT statement must never ground a user decision."""
    transcript = "HUMAN: refactor the auth module\nAGENT: the auth module is refactored\n"
    assert is_decision_grounded(transcript, "the auth module is refactored") is False


def test_is_decision_grounded_false_for_tool_result_line():
    transcript = "HUMAN: run the test suite\n  result[error]> command not found\n"
    assert is_decision_grounded(transcript, "command not found") is False


def test_is_decision_grounded_false_for_empty_quote():
    transcript = "HUMAN: refactor the auth module\n"
    assert is_decision_grounded(transcript, "") is False


def test_is_decision_grounded_parity_with_quote_gate_ground_quote(tmp_path):
    """Pins the harness's decoupled grounding check to the production gate.

    Replays two real fixtures through the actual pipeline (adapter ->
    `classify_channel` -> normalizer -> `transcript_chunks`), same pattern
    `tests/test_extraction.py` uses, so both checks run against a chunk the
    real classifier produced -- not a hand-typed string that could drift
    from what `classify_channel` actually does.
    """
    db_path = tmp_path / "palaver.db"

    human_result = replay(FIXTURES_DIR / "slash-command-after-reply.jsonl", db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, content FROM transcript_chunks WHERE session_id = ? ORDER BY seq",
            (human_result.session_id,),
        ).fetchall()
        human_chunk_id, human_content = rows[0]
        injected_chunk_id, injected_content = rows[2]
        assert human_content.startswith("HUMAN: ")
        assert injected_content.startswith("INJECTED: ")

        human_quote = "refactor the auth module"
        gate_verdict = ground_quote(
            conn, transcript_chunk_id=human_chunk_id, quote=human_quote, statement=human_quote
        )
        harness_verdict = is_decision_grounded(
            normalize_path(FIXTURES_DIR / "slash-command-after-reply.jsonl"), human_quote
        )
        assert gate_verdict.is_tier_one is True
        assert harness_verdict is True

        injected_quote = "<command-name>/status</command-name>"
        gate_verdict_injected = ground_quote(
            conn,
            transcript_chunk_id=injected_chunk_id,
            quote=injected_quote,
            statement=injected_quote,
        )
        harness_verdict_injected = is_decision_grounded(
            normalize_path(FIXTURES_DIR / "slash-command-after-reply.jsonl"), injected_quote
        )
        assert gate_verdict_injected.is_tier_one is False
        assert harness_verdict_injected is False
    finally:
        conn.close()


# =============================================================================
# score_leg
# =============================================================================


def _label(id_, **overrides):
    defaults = dict(
        path=f"{id_}.jsonl",
        expect_question=False,
        expect_blocker=False,
        expect_current_task=False,
        expect_decision=False,
        expect_completion=False,
        forbidden_quote_substrings=(),
    )
    defaults.update(overrides)
    return FixtureLabel(id=id_, **defaults)


def test_score_leg_perfect_extraction_scores_full_marks():
    labels = (
        _label("has-task", expect_current_task=True),
        _label("no-signal"),
    )
    extractions = {
        "has-task": Extraction("do the thing", (), (), (), False),
        "no-signal": Extraction("", (), (), (), False),
    }
    transcripts = {"has-task": "HUMAN: x\n", "no-signal": ""}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.current_task_accuracy == 1.0
    assert metrics.question_detection_accuracy == 1.0
    assert metrics.blocker_detection_accuracy == 1.0
    assert metrics.decision_retention_accuracy == 1.0
    assert metrics.completion_detection_accuracy == 1.0
    assert metrics.false_decision_rate == 0.0
    assert metrics.decisions_extracted == 0


def test_score_leg_penalizes_ungrounded_decision_as_false():
    labels = (_label("with-decision", expect_decision=True),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="chose postgres", quote="postgres"),), False
        )
    }
    # "postgres" never appears on a HUMAN line here -- ungrounded.
    transcripts = {"with-decision": "AGENT: we will use postgres\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.decisions_extracted == 1
    assert metrics.false_decisions == 1
    assert metrics.false_decision_rate == 1.0


def test_score_leg_grounded_decision_is_not_false():
    """Positive control for the test above: a correctly grounded decision must not be penalized."""
    labels = (_label("with-decision", expect_decision=True),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="chose postgres", quote="postgres"),), False
        )
    }
    transcripts = {"with-decision": "HUMAN: postgres\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.false_decisions == 0
    assert metrics.false_decision_rate == 0.0


def test_score_leg_forbidden_substring_is_false_even_if_grounded():
    """`forbidden_quote_substrings` catches a quote that is technically HUMAN-grounded
    but named as a known-bad attribution target for a specific fixture."""
    labels = (_label("with-decision", expect_decision=True, forbidden_quote_substrings=("no",)),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="declined", quote="no"),), False
        )
    }
    transcripts = {"with-decision": "HUMAN: no\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.false_decisions == 1


def test_score_leg_zero_decisions_extracted_gives_zero_rate_not_a_crash():
    labels = (_label("no-signal"),)
    extractions = {"no-signal": Extraction("", (), (), (), False)}
    transcripts = {"no-signal": ""}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.decisions_extracted == 0
    assert metrics.false_decision_rate == 0.0


# =============================================================================
# is_degenerate_extraction
# =============================================================================


def test_is_degenerate_extraction_flags_empty_field_when_demonstrably_present():
    labels = (_label("has-blocker", expect_blocker=True),)
    extractions = {"has-blocker": Extraction("", (), (), (), False)}

    offenders = is_degenerate_extraction(labels, extractions)

    assert offenders == ["has-blocker"]


def test_is_degenerate_extraction_empty_when_fields_are_present():
    """Positive control: a non-degenerate extractor produces no offenders."""
    labels = (_label("has-blocker", expect_blocker=True),)
    extractions = {"has-blocker": Extraction("", ("tests fail",), (), (), False)}

    offenders = is_degenerate_extraction(labels, extractions)

    assert offenders == []
