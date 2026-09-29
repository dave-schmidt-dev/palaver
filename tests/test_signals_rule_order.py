"""
Rule ordering: unreadable source, unparsed records, error priority, and the
undeterminable turn boundary.
"""

from palaver.observer.signals import (
    Status,
    Tri,
    derive_status,
)
from tests._signals_support import _signals

# --- rule order --------------------------------------------------------------


def test_unreadable_source_is_unknown_however_decisive_the_other_signals():
    """Rule 1 outranks every later rule: a session Palaver could not read gets no
    status claim, even when the remaining signals would otherwise produce a
    confident ERROR, WORKING, or AWAITING_HUMAN.

    `FALSE` and `UNKNOWN` are both non-claims here — "we could not confirm we
    read it" is no better a footing than "we failed to read it". The
    positive-control block re-runs each identical signal set with
    `source_readable=TRUE`, proving the other signals really were decisive
    and that the UNKNOWNs above came from rule 1 rather than from an inert
    signal set.
    """
    decisive = {
        Status.ERROR: {"unresolved_tool_error": Tri.TRUE},
        Status.WORKING: {"agent_turn_ended": Tri.FALSE},
        Status.AWAITING_HUMAN: {"agent_turn_ended": Tri.TRUE},
    }

    for unreadable in (Tri.FALSE, Tri.UNKNOWN):
        for expected, overrides in decisive.items():
            assert (
                derive_status(_signals(source_readable=unreadable, **overrides)) is Status.UNKNOWN
            )
            # Positive control: same signals, source readable.
            assert derive_status(_signals(source_readable=Tri.TRUE, **overrides)) is expected


def test_unparsed_signal_records_are_unknown_however_decisive_the_other_signals():
    """Rule 2: signals derived from a record set that did not fully decode are
    computed over an incomplete view, and Palaver cannot know whether the
    record it lost was the decisive one — so it reports UNKNOWN rather than a
    status derived from a partial read.

    As with rule 1, `FALSE` and `UNKNOWN` are treated alike, and each case
    has a positive control asserting the other signals were decisive.
    """
    decisive = {
        Status.ERROR: {"unresolved_tool_error": Tri.TRUE},
        Status.WORKING: {"agent_turn_ended": Tri.FALSE},
        Status.AWAITING_HUMAN: {"agent_turn_ended": Tri.TRUE},
    }

    for unparsed in (Tri.FALSE, Tri.UNKNOWN):
        for expected, overrides in decisive.items():
            assert (
                derive_status(_signals(signal_records_parsed=unparsed, **overrides))
                is Status.UNKNOWN
            )
            assert derive_status(_signals(signal_records_parsed=Tri.TRUE, **overrides)) is expected


def test_error_outranks_the_turn_boundary_in_both_boundary_directions():
    """Rule 3 precedes rules 4 and 5, in both directions of the boundary signal.

    The turn boundary can only ever produce WORKING or AWAITING_HUMAN, so an
    ordering that consulted it first would make ERROR unreachable for every
    session whose boundary is determinable — which is nearly all of them.
    The controls flip only `unresolved_tool_error`, so the ERROR results
    cannot be an artefact of the boundary value.
    """
    for boundary, control_status in (
        (Tri.FALSE, Status.WORKING),
        (Tri.TRUE, Status.AWAITING_HUMAN),
    ):
        errored = _signals(unresolved_tool_error=Tri.TRUE, agent_turn_ended=boundary)

        assert derive_status(errored) is Status.ERROR

        # Positive control: identical signals, no tool error.
        clean = _signals(unresolved_tool_error=Tri.FALSE, agent_turn_ended=boundary)
        assert derive_status(clean) is control_status


def test_error_requires_positive_evidence_and_is_never_asserted_from_unknown():
    """An undeterminable tool-outcome signal never produces ERROR: rule 3 is a
    positive claim about an observed outcome, so absence of evidence falls
    through to the turn-boundary rules rather than manufacturing an error.

    This is the one place `UNKNOWN` is deliberately treated like `FALSE`, and
    it is the opposite direction from rules 1 and 2 — conservatism about a
    claim, not about a read. The `Tri.TRUE` control proves ERROR is reachable
    from this same shape when the evidence is actually there.
    """
    for boundary, expected in ((Tri.FALSE, Status.WORKING), (Tri.TRUE, Status.AWAITING_HUMAN)):
        unknown_error = _signals(unresolved_tool_error=Tri.UNKNOWN, agent_turn_ended=boundary)

        assert derive_status(unknown_error) is expected
        assert derive_status(unknown_error) is not Status.ERROR

        # Positive control: the same shape with real evidence does give ERROR.
        assert (
            derive_status(_signals(unresolved_tool_error=Tri.TRUE, agent_turn_ended=boundary))
            is Status.ERROR
        )


def test_undeterminable_turn_boundary_is_unknown_and_not_collapsed_to_either_side():
    """UNKNOWN is a first-class result, not a guess: when the source read cleanly
    but the turn boundary could not be determined, no status is claimed.

    This is the sharpest unknown-is-not-false test in the module — the same
    signal set with `FALSE` yields WORKING and with `TRUE` yields
    AWAITING_HUMAN, so a `derive_status()` that collapsed UNKNOWN into either
    boolean value fails here rather than silently reporting one of them.
    """
    assert derive_status(_signals(agent_turn_ended=Tri.UNKNOWN)) is Status.UNKNOWN

    # Positive controls: both determinate values produce a real status, so the
    # UNKNOWN above is not an inert signal set.
    assert derive_status(_signals(agent_turn_ended=Tri.FALSE)) is Status.WORKING
    assert derive_status(_signals(agent_turn_ended=Tri.TRUE)) is Status.AWAITING_HUMAN
