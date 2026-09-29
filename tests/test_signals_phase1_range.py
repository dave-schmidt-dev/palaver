"""
Phase 1: the status range derivable from signals alone, and the never-model-supplied
gate test.
"""

import dataclasses
import inspect

import pytest

from palaver.extract.persist import Extraction
from palaver.observer.signals import (
    FORBIDDEN_PAYLOAD_KEYS,
    PHASE1_STATUS_RANGE,
    SIGNAL_NAMES,
    Status,
    Tri,
    derive_status,
)
from tests._signals_support import EXPECTED_PHASE1_RANGE, _all_signal_combinations, _signals

#: Statuses `derive_status()` cannot return when no `extraction` is passed —
#: the four the plan's §4.2 table stages for Phase 3.6, plus the one it stages
#: for Phase 5.2. This is the live contract for every caller in the tree that
#: predates task 3.6, not a historical note.
EXPECTED_DEFERRED_WITHOUT_EXTRACTION = {
    Status.DONE,
    Status.WAITING_FOR_USER,
    Status.QUESTION,
    Status.BLOCKED,
    Status.IDLE,
}


# --- the Phase 1 range, proved by exhausting the signal space ----------------


def test_phase1_status_range():
    """With no extraction, `derive_status()` returns exactly {WORKING,
    AWAITING_HUMAN, ERROR, UNKNOWN}, and the five deferred statuses are
    unreachable — established by evaluating every combination of every
    signal's full value domain, not by reading the source.

    Unchanged by task 3.6 on purpose, and re-run against the refined
    implementation for that reason: `extraction` is keyword-only and defaults
    to `None`, so every caller in the tree that predates refinement still
    sees this range and only this range.

    Set equality is asserted in both directions: `issubset` would pass for a
    `derive_status()` that had silently lost a branch and could only ever
    return `UNKNOWN`. The combination count is asserted first, so a helper
    that enumerated nothing (or sampled) fails here rather than producing a
    vacuous comparison downstream.
    """
    combinations = _all_signal_combinations()

    assert len(combinations) == len(Tri) ** len(SIGNAL_NAMES) == 81
    assert len(set(combinations)) == 81, "combinations must be distinct, not repeated"

    returned = {derive_status(s) for s in combinations}

    assert returned == EXPECTED_PHASE1_RANGE
    assert EXPECTED_PHASE1_RANGE == returned
    assert set(PHASE1_STATUS_RANGE) == EXPECTED_PHASE1_RANGE
    assert all(isinstance(status, Status) for status in returned)

    # Unreachable, not merely unused: these are nameable members of `Status`
    # that the whole signal space cannot produce without an extraction.
    assert returned.isdisjoint(EXPECTED_DEFERRED_WITHOUT_EXTRACTION)
    assert set(Status) - returned == EXPECTED_DEFERRED_WITHOUT_EXTRACTION


def test_status_is_never_model_supplied():
    """INV-7 gate: `derive_status()` accepts no model-supplied field.

    Task 3.6 gave this function refinement content and did not loosen this
    gate. `remaining_work` and `blockers_now` are still not parameters — they
    arrive as fields *inside* a typed `Extraction`, which declares no status
    of any kind and so cannot carry one — and passing either by name still
    raises `TypeError` naming that argument. The signature is inspected
    directly for a `**kwargs` that would swallow one and make this test pass
    for the wrong reason, and the baseline call is exercised as a positive
    control so a `derive_status()` that raised `TypeError` unconditionally
    could not pass.

    The signal set is built *outside* the `pytest.raises` block on purpose:
    `Signals.__post_init__` also raises `TypeError`, so constructing it
    inside would let a construction failure satisfy the assertion.
    """
    signals = _signals()

    assert isinstance(derive_status(signals), Status)  # positive control
    assert isinstance(derive_status(signals, extraction=Extraction()), Status)

    with pytest.raises(TypeError, match="remaining_work"):
        derive_status(signals, remaining_work=["finish the migration"])

    with pytest.raises(TypeError, match="blockers_now"):
        derive_status(signals, blockers_now=["waiting on credentials"])

    with pytest.raises(TypeError, match="status"):
        derive_status(signals, status="DONE")

    signature = inspect.signature(derive_status)
    parameters = signature.parameters

    assert "remaining_work" not in parameters
    assert "blockers_now" not in parameters
    assert "status" not in parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()), (
        "a **kwargs would silently swallow remaining_work"
    )
    assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values())
    assert list(parameters) == ["signals", "extraction"]
    assert parameters["extraction"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "a positional extraction would let a caller pass a raw payload by accident"
    )
    assert parameters["extraction"].default is None, (
        "every pre-3.6 caller must keep seeing PHASE1_STATUS_RANGE"
    )

    # The refinement input itself cannot spell a status: no field of the
    # dataclass is named for one, so there is nothing for a rule to read even
    # if one were written.
    extraction_fields = {field.name for field in dataclasses.fields(Extraction)}
    assert extraction_fields == {
        "current_task",
        "remaining_work",
        "blockers_now",
        "open_questions",
        "decisions",
        "resolved_questions",
    }
    assert not extraction_fields & FORBIDDEN_PAYLOAD_KEYS


def test_ended_turn_with_no_extraction_is_awaiting_human_never_done():
    """The brief's single named prohibition: lack of terminal output is not DONE.

    An ended turn is all Phase 1 has — there is no extraction, so nothing
    can distinguish "finished" from "waiting on you". The honest answer is
    `AWAITING_HUMAN`. `DONE` is a nameable member of `Status`, so asserting
    it is not returned is a real assertion rather than a property of the
    enum's size. The mid-turn control proves the ended-turn signal is what
    produced the result.
    """
    ended = _signals(agent_turn_ended=Tri.TRUE)

    status = derive_status(ended)

    assert status is Status.AWAITING_HUMAN
    assert status is not Status.DONE
    assert status not in EXPECTED_DEFERRED_WITHOUT_EXTRACTION

    # Positive control: the same signal set, turn not ended.
    assert derive_status(_signals(agent_turn_ended=Tri.FALSE)) is Status.WORKING
