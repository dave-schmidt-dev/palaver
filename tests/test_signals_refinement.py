"""
Task 3.6: status refinement from an Extraction, across the signal and extraction space.
"""

import itertools
import json
import pathlib

from palaver.extract.client import ModelClientError, ModelTimeoutError
from palaver.extract.persist import Extraction
from palaver.observer.signals import (
    PHASE1_STATUS_RANGE,
    REFINED_STATUS_RANGE,
    Signals,
    Status,
    Tri,
    derive_status,
)
from palaver.observer.turn_boundary import derive_signals
from tests._signals_support import (
    EXPECTED_PHASE1_RANGE,
    _all_signal_combinations,
    _ended_turn,
    _signals,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


#: The range once an `Extraction` is supplied (task 3.6), written out the same
#: way and for the same reason.
EXPECTED_REFINED_RANGE = EXPECTED_PHASE1_RANGE | {
    Status.DONE,
    Status.WAITING_FOR_USER,
    Status.QUESTION,
    Status.BLOCKED,
}


#: Statuses unreachable even *with* an extraction: `IDLE` needs process
#: liveness (Phase 5.2), which no input to this module carries.
EXPECTED_DEFERRED_WITH_EXTRACTION = {Status.IDLE}


def _refined(**extraction_fields: str | None) -> Status:
    """Status for a clean ended turn refined by an extraction with these fields."""
    return derive_status(_ended_turn(), extraction=Extraction(**extraction_fields))


def _fixture_signals(name: str) -> Signals:
    """Read one committed fixture and compute its signals.

    `store_mtime` is withheld, matching how `tests/fixtures/README.md`
    measured every label it records: mtime is corroboration only and never
    moves a status, but a checked-out file's mtime is its checkout time, so
    leaving it out keeps the reading reproducible.
    """
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return derive_signals(records).signals


def _status_from_extractor(signals: Signals, extract) -> Status:
    """The caller contract task 4.1 must implement, exercised here as code.

    `derive_status()` deliberately calls no model and catches no model error
    — it opens no socket, which is what lets it promise anything at all — so
    the degradation lives with the caller: whatever the extractor raises,
    the status path is entered with `extraction=None` and falls back to
    Phase 1 behaviour rather than to a completion claim.
    """
    try:
        extraction = extract()
    except ModelClientError:
        extraction = None
    return derive_status(signals, extraction=extraction)


#: The value domain of each refinement field: absent (`None`, "no opinion"),
#: affirmatively empty (`""`), and carrying content. Three values because the
#: `None`/`""` distinction is the whole of the `DONE` rule — a two-valued
#: domain would make the defect this task fixes untestable.
EXTRACTION_FIELD_VALUES = (None, "", "some text")


def _all_extractions() -> list[Extraction | None]:
    """Enumerate the refinement space: no extraction, plus every field domain."""
    return [None] + [
        Extraction(remaining_work=remaining, blockers_now=blockers, open_questions=questions)
        for remaining, blockers, questions in itertools.product(EXTRACTION_FIELD_VALUES, repeat=3)
    ]


# --- task 3.6: refinement from extraction ------------------------------------


def test_refined_status_range_over_the_signal_and_extraction_space():
    """With an extraction the range is exactly the four Phase 1 statuses plus
    DONE, WAITING_FOR_USER, QUESTION, and BLOCKED — and `IDLE` is still
    unreachable, established by crossing the whole signal space with the whole
    refinement space rather than by reading the source.

    Both counts are asserted before the comparison, so a helper that
    enumerated nothing (or that quietly collapsed the `None`/`""` distinction
    to two values) fails here rather than making the range check vacuous. Set
    equality is asserted in both directions for the same reason as
    `test_phase1_status_range`: `issubset` would pass for an implementation
    that had lost a branch.
    """
    combinations = _all_signal_combinations()
    extractions = _all_extractions()

    assert len(combinations) == 81
    assert len(extractions) == len(EXTRACTION_FIELD_VALUES) ** 3 + 1 == 28

    returned = {
        derive_status(signals, extraction=extraction)
        for signals in combinations
        for extraction in extractions
    }

    assert returned == EXPECTED_REFINED_RANGE
    assert EXPECTED_REFINED_RANGE == returned
    assert set(REFINED_STATUS_RANGE) == EXPECTED_REFINED_RANGE
    assert set(PHASE1_STATUS_RANGE) < set(REFINED_STATUS_RANGE)

    # Unreachable, not merely unused: `IDLE` is a nameable member of `Status`
    # that 81 × 28 = 2268 calls cannot produce, because nothing here carries
    # process liveness (Phase 5.2).
    assert returned.isdisjoint(EXPECTED_DEFERRED_WITH_EXTRACTION)
    assert set(Status) - returned == EXPECTED_DEFERRED_WITH_EXTRACTION


def test_refinement_never_reports_done_without_an_affirmative_empty_remaining_work():
    """The defect this task exists to fix: a *missing* `remaining_work` is not a
    finished session.

    The spike's rule was `WAITING_FOR_USER if remaining_work else DONE`, under
    which `None` — the value task 3.4's `Extraction` uses for "this pass had no
    opinion" — falls through to DONE, so every extraction that failed to
    produce the field reported the session as complete. Here `None` yields
    AWAITING_HUMAN and only `""` yields DONE, which is the difference between
    an answer and a guess.

    The two calls differ in exactly one character of one field, so neither
    result can be an artefact of the rest of the signal set.
    """
    assert _refined(remaining_work=None, blockers_now="", open_questions="") is (
        Status.AWAITING_HUMAN
    )
    assert _refined(remaining_work=None, blockers_now="", open_questions="") is not Status.DONE

    # An extraction that returned nothing at all is the same case, and is what
    # a model that answered with an empty object produces.
    assert _refined() is Status.AWAITING_HUMAN

    # Positive control: the same shape with an affirmative "nothing remains".
    assert _refined(remaining_work="", blockers_now="", open_questions="") is Status.DONE


def test_refinement_treats_whitespace_only_fields_as_empty_and_interprets_no_prose():
    """Whitespace is stripped and nothing else is interpreted.

    `"   "` is empty because whitespace carries no claim. `"none"` is *not*,
    even though a human reading it would call the session finished: teaching
    Python to read model prose is unbounded and is the model deciding status
    by another route. The direction of that failure is what makes strip-only
    safe rather than lazy — an unrecognized prose form is non-empty, and
    non-empty falls toward WAITING_FOR_USER, never toward DONE.
    """
    assert _refined(remaining_work="   \n\t ") is Status.DONE
    assert _refined(remaining_work="none") is Status.WAITING_FOR_USER
    assert _refined(remaining_work="n/a") is Status.WAITING_FOR_USER
    assert _refined(blockers_now="  ", remaining_work="finish the migration") is (
        Status.WAITING_FOR_USER
    )
    assert _refined(blockers_now="none", remaining_work="finish the migration") is Status.BLOCKED


def test_refinement_with_blockers_now_is_blocked():
    """A non-empty `blockers_now` on an ended turn is BLOCKED.

    The control drops only that field, so BLOCKED cannot be an artefact of the
    ended-turn signal or of the other extraction content.
    """
    assert _refined(blockers_now="waiting on App Store reviewer access") is Status.BLOCKED

    # Positive controls: the same extraction with the blocker removed, and
    # with it affirmatively empty, both resolve elsewhere.
    assert _refined(blockers_now=None) is Status.AWAITING_HUMAN
    assert _refined(blockers_now="", remaining_work="") is Status.DONE


def test_refinement_orders_blocked_then_question_then_waiting_for_user_then_done():
    """The refinement order is the contract, so it is pinned by removing one
    field at a time and watching the answer change.

    Every step holds the other fields fixed at affirmatively-empty, so each
    transition isolates one rule. A reordering of any adjacent pair fails at
    the step where the two rules compete, rather than passing because a later
    rule happened to agree.
    """
    everything = {
        "blockers_now": "waiting on credentials",
        "open_questions": "which region should it deploy to?",
        "remaining_work": "finish the migration",
    }

    assert _refined(**everything) is Status.BLOCKED
    assert _refined(**{**everything, "blockers_now": ""}) is Status.QUESTION
    assert _refined(**{**everything, "blockers_now": "", "open_questions": ""}) is (
        Status.WAITING_FOR_USER
    )
    assert _refined(blockers_now="", open_questions="", remaining_work="") is Status.DONE


def test_refinement_never_overrides_a_deterministic_signal():
    """Model content refines a coarse structural answer; it never overturns a
    determinate one.

    An extraction that would produce BLOCKED on an ended turn leaves ERROR,
    WORKING, and every UNKNOWN untouched, because refinement is reached only
    from rule 5. This is the INV-7 boundary in its operational form: the
    measured finding is that a 4B model's own fields cannot be trusted as
    rule predicates, so they are allowed to split a branch and never to
    select one.
    """
    blocked = Extraction(blockers_now="waiting on credentials", remaining_work="")

    unchanged = {
        Status.ERROR: _signals(unresolved_tool_error=Tri.TRUE, agent_turn_ended=Tri.TRUE),
        Status.WORKING: _signals(agent_turn_ended=Tri.FALSE),
        Status.UNKNOWN: _signals(source_readable=Tri.FALSE, agent_turn_ended=Tri.TRUE),
    }
    for expected, signals in unchanged.items():
        assert derive_status(signals, extraction=blocked) is expected
        assert derive_status(signals, extraction=blocked) is derive_status(signals)

    for undeterminable in ("signal_records_parsed", "agent_turn_ended"):
        signals = _signals(**{undeterminable: Tri.UNKNOWN})
        assert derive_status(signals, extraction=blocked) is Status.UNKNOWN

    # Positive control: the same extraction on a clean ended turn does refine.
    assert derive_status(_ended_turn(), extraction=blocked) is Status.BLOCKED


def test_refinement_falls_back_to_awaiting_human_when_the_extraction_times_out():
    """A model outage degrades to Phase 1 behaviour, never to a completion claim.

    The extractor raises the real `ModelTimeoutError` that
    `palaver.extract.client` raises when llama-server does not answer within
    `timeout` — the exception type is the contract a caller degrades on, and
    no socket is needed to exercise it. The session under test is the
    finished-session fixture, i.e. the one most likely to be called DONE by a
    system that guesses: a timed-out extraction on a session that really has
    finished still reports AWAITING_HUMAN, because nothing observed it.

    The positive control runs the identical caller over the identical
    signals with an extractor that succeeds, so the fallback cannot be an
    inert path that always returns AWAITING_HUMAN.
    """
    signals = _fixture_signals("finished-session.jsonl")

    def timing_out() -> Extraction:
        raise ModelTimeoutError("request to 127.0.0.1:8090 timed out after 30.0s")

    status = _status_from_extractor(signals, timing_out)

    assert status is Status.AWAITING_HUMAN
    assert status is not Status.DONE

    # Positive control: same caller, same session, an extraction that arrived.
    assert _status_from_extractor(signals, lambda: Extraction(remaining_work="")) is Status.DONE


def test_refinement_of_the_finished_session_fixture_is_done_where_phase_1_was_awaiting_human():
    """`finished-session.jsonl` — the fixture the corpus keeps specifically to
    prove silence is not read as completion — reaches the corpus's recorded
    phase 3 target of DONE, and only through extraction.

    Both readings are asserted from the same signal set, which is the point:
    the structure did not change and cannot distinguish a finished session
    from a waiting one, so Phase 1's AWAITING_HUMAN was not a defect to be
    corrected but the honest answer for the inputs it had.
    """
    signals = _fixture_signals("finished-session.jsonl")

    assert signals.agent_turn_ended is Tri.TRUE
    assert derive_status(signals) is Status.AWAITING_HUMAN  # phase 1, unchanged
    assert derive_status(signals, extraction=Extraction(remaining_work="")) is Status.DONE


def test_refinement_of_the_ended_turn_fixtures_matches_their_recorded_phase_3_targets():
    """Every ended-turn fixture in the corpus reaches the finer label
    `tests/fixtures/README.md` records for it, from its real signals plus an
    extraction consistent with what the file shows.

    The last case is the one that shows refinement is doing the work:
    `waiting-for-user-reply.jsonl` and `question-askuserquestion-unresolved.jsonl`
    are both AWAITING_HUMAN structurally, and the same fixture resolves to
    WAITING_FOR_USER or QUESTION depending only on extraction content. That is
    the split Phase 1 could not make, and the README says so in its own
    derivation note.
    """
    cases = (
        ("waiting-for-user-reply.jsonl", Extraction(remaining_work="confirm the schema choice")),
        (
            "question-askuserquestion-unresolved.jsonl",
            Extraction(open_questions="which database should the worker use?"),
        ),
        ("finished-session.jsonl", Extraction(remaining_work="")),
    )
    expected = (Status.WAITING_FOR_USER, Status.QUESTION, Status.DONE)

    for (name, extraction), target in zip(cases, expected, strict=True):
        signals = _fixture_signals(name)
        assert derive_status(signals) is Status.AWAITING_HUMAN  # phase 1, all three
        assert derive_status(signals, extraction=extraction) is target

    # Same fixture, different extraction content, different status — the
    # refinement follows the extraction and not the file's structure.
    waiting = _fixture_signals("waiting-for-user-reply.jsonl")
    assert derive_status(waiting, extraction=Extraction(open_questions="which region?")) is (
        Status.QUESTION
    )
