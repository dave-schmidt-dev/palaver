"""
The liveness layer: the two demotion rules, their exact range over the whole input
space, and observe_liveness's own pid/clock inputs.
"""

from __future__ import annotations

import itertools
import os
from datetime import timedelta

import pytest

from palaver.extract.persist import Extraction
from palaver.observer.signals import (
    LIVE_STATUS_RANGE,
    REFINED_STATUS_RANGE,
    SIGNAL_NAMES,
    Liveness,
    Signals,
    Status,
    Tri,
    apply_liveness,
    derive_status,
    derive_status_with_liveness,
)
from palaver.ui.pane_join import (
    DEFAULT_IDLE_WINDOW,
    observe_liveness,
)
from tests._pane_join_support import NOW

# --- liveness: the two rules -------------------------------------------------


def _signals(
    *,
    source_readable=Tri.TRUE,
    signal_records_parsed=Tri.TRUE,
    unresolved_tool_error=Tri.FALSE,
    agent_turn_ended=Tri.FALSE,
):
    """Build a signal set from a clean, mid-turn (`WORKING`) baseline."""
    return Signals(
        source_readable=source_readable,
        signal_records_parsed=signal_records_parsed,
        unresolved_tool_error=unresolved_tool_error,
        agent_turn_ended=agent_turn_ended,
    )


DEAD = Liveness(process_alive=Tri.FALSE, cursor_advanced_recently=Tri.UNKNOWN)
ALIVE_QUIET = Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=Tri.FALSE)
ALIVE_BUSY = Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=Tri.TRUE)
NO_PANE = Liveness(process_alive=Tri.UNKNOWN, cursor_advanced_recently=Tri.UNKNOWN)


def test_a_dead_process_with_a_stale_working_signal_returns_unknown():
    """Done-when: a dead process demotes `WORKING` to `UNKNOWN`.

    Not to `AWAITING_HUMAN`: that would claim the human is owed something,
    and a process killed mid-turn owes them nothing it can name. The
    controls pin both halves of the rule — the same signals with a live
    process stay `WORKING`, and the same dead process leaves every other
    status alone.
    """
    working = _signals(agent_turn_ended=Tri.FALSE)
    assert derive_status(working) is Status.WORKING

    assert derive_status_with_liveness(working, DEAD) is Status.UNKNOWN

    # Control 1: liveness is what changed the answer, not the signals.
    assert derive_status_with_liveness(working, ALIVE_BUSY) is Status.WORKING
    assert derive_status_with_liveness(working, NO_PANE) is Status.WORKING

    # Control 2: the same dead process does not rewrite anything else. A
    # dead process corroborates an ended-turn status rather than
    # contradicting it, and `ERROR` still names why to look at the session.
    for status in REFINED_STATUS_RANGE - {Status.WORKING}:
        assert apply_liveness(status, DEAD) is status


def test_a_live_process_with_an_ended_turn_and_no_cursor_advance_returns_idle():
    """Done-when: alive, turn ended, quiet for the window yields `IDLE`.

    Each control removes exactly one of the three conjuncts, so `IDLE`
    cannot be reached by any two of them.
    """
    ended = _signals(agent_turn_ended=Tri.TRUE)
    assert derive_status(ended) is Status.AWAITING_HUMAN

    assert derive_status_with_liveness(ended, ALIVE_QUIET) is Status.IDLE

    # Drop "quiet": a session written to inside the window is not idle.
    assert derive_status_with_liveness(ended, ALIVE_BUSY) is Status.AWAITING_HUMAN

    # Drop "alive": an unobservable process is not evidence of anything, and
    # a dead one is certainly not idle.
    assert (
        derive_status_with_liveness(ended, Liveness(Tri.UNKNOWN, Tri.FALSE))
        is Status.AWAITING_HUMAN
    )
    assert (
        derive_status_with_liveness(ended, Liveness(Tri.FALSE, Tri.FALSE)) is Status.AWAITING_HUMAN
    )

    # Drop "turn ended": a working session sitting quiet stays `WORKING`. A
    # ten-minute gap in writes is normal for an agent thinking or running a
    # long tool call, and calling that idle is the loudest way this rule can
    # be wrong.
    assert derive_status_with_liveness(_signals(agent_turn_ended=Tri.FALSE), ALIVE_QUIET) is (
        Status.WORKING
    )


def test_liveness_never_refines_a_status_that_already_says_more():
    """`BLOCKED`, `QUESTION`, `WAITING_FOR_USER`, and `DONE` outrank `IDLE`.

    Rewriting a blocked session to `IDLE` after ten quiet minutes would hide
    the blocker behind the one status meaning "nothing to do here" — and a
    session that has been blocked for ten minutes is more worth surfacing,
    not less.
    """
    ended = _signals(agent_turn_ended=Tri.TRUE)

    for extraction, expected in (
        (Extraction(blockers_now="waiting on a review"), Status.BLOCKED),
        (Extraction(open_questions="which database?"), Status.QUESTION),
        (Extraction(remaining_work="write the tests"), Status.WAITING_FOR_USER),
        (Extraction(remaining_work=""), Status.DONE),
    ):
        assert derive_status(ended, extraction=extraction) is expected
        assert derive_status_with_liveness(ended, ALIVE_QUIET, extraction=extraction) is expected


def test_liveness_cannot_resurrect_a_status_the_rules_withdrew():
    """An alive, quiet process over an unreadable source stays `UNKNOWN`.

    This is the case both rules fall through, and the one a mutant can flip
    invisibly. It is also what keeps `apply_liveness` compatible with
    `derive_status_for_source`'s coverage gate, which may only ever weaken an
    answer: a layer running after the gate that could manufacture a
    confident status would reverse that guarantee.
    """
    unreadable = _signals(source_readable=Tri.FALSE)
    assert derive_status(unreadable) is Status.UNKNOWN

    for liveness in (DEAD, ALIVE_QUIET, ALIVE_BUSY, NO_PANE):
        assert derive_status_with_liveness(unreadable, liveness) is Status.UNKNOWN

    unparsed = _signals(signal_records_parsed=Tri.UNKNOWN)
    assert derive_status(unparsed) is Status.UNKNOWN
    assert derive_status_with_liveness(unparsed, ALIVE_QUIET) is Status.UNKNOWN

    # And rule 6's `UNKNOWN`: the source read cleanly but the turn boundary
    # was not determinable. Liveness says a process exists; it never says the
    # observation succeeded.
    indeterminate = _signals(agent_turn_ended=Tri.UNKNOWN)
    assert derive_status(indeterminate) is Status.UNKNOWN
    assert derive_status_with_liveness(indeterminate, ALIVE_QUIET) is Status.UNKNOWN


def test_liveness_rejects_a_raw_bool():
    """A `bool` where a `Tri` belongs would skip every rule silently.

    `True is Tri.TRUE` is `False`, so the status would pass through unchanged
    and look deliberate. Same guard, same reason, as `Signals.__post_init__`.
    """
    with pytest.raises(TypeError, match="process_alive must be a Tri"):
        Liveness(process_alive=True, cursor_advanced_recently=Tri.FALSE)

    with pytest.raises(TypeError, match="cursor_advanced_recently must be a Tri"):
        Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=False)


# --- liveness: the range over the whole input space --------------------------


def _all_signal_combinations():
    """Every `Tri` value of every signal."""
    return [
        Signals(**dict(zip(SIGNAL_NAMES, combination, strict=True)))
        for combination in itertools.product(Tri, repeat=len(SIGNAL_NAMES))
    ]


def _all_extractions():
    """No extraction, plus every value of every refinement field."""
    return [None] + [
        Extraction(remaining_work=remaining, blockers_now=blockers, open_questions=questions)
        for remaining, blockers, questions in itertools.product((None, "", "some text"), repeat=3)
    ]


def _all_liveness():
    """Every `Tri` value of both liveness fields."""
    return [Liveness(alive, advanced) for alive, advanced in itertools.product(Tri, repeat=2)]


def test_the_live_status_range_over_the_signal_extraction_and_liveness_space():
    """With liveness in play the range is exactly `LIVE_STATUS_RANGE`.

    Established by crossing the whole signal space with the whole refinement
    space with the whole liveness space, rather than by reading the source.
    Every count is asserted before the comparison so a helper that enumerated
    nothing fails here instead of making the range check vacuous, and set
    equality is asserted in both directions because `issubset` would pass for
    an implementation that had lost a branch.
    """
    combinations = _all_signal_combinations()
    extractions = _all_extractions()
    livenesses = _all_liveness()

    assert len(combinations) == 81
    assert len(extractions) == 28
    assert len(livenesses) == 9

    returned = {
        derive_status_with_liveness(signals, liveness, extraction=extraction)
        for signals in combinations
        for extraction in extractions
        for liveness in livenesses
    }

    assert returned == set(LIVE_STATUS_RANGE)
    assert set(LIVE_STATUS_RANGE) == returned
    assert set(REFINED_STATUS_RANGE) < set(LIVE_STATUS_RANGE)

    # Every member of the enum is now reachable somewhere. Stated as a fact
    # the suite proves rather than as prose in a docstring — and it is the
    # assertion that would fail first if a later phase added a status without
    # a way to reach it.
    assert set(Status) - returned == frozenset()


def test_idle_is_reachable_only_from_an_alive_quiet_ended_turn():
    """`IDLE`'s reachability asserted as an exact count, not as membership.

    A rule that fired on `BLOCKED` too, or on any liveness where the process
    is merely not-dead, still passes a membership check over this space. The
    count does not: it is pinned to the exact conjunction, computed
    independently from `derive_status` rather than restated from
    `apply_liveness`.
    """
    cases = [
        (signals, extraction, liveness)
        for signals in _all_signal_combinations()
        for extraction in _all_extractions()
        for liveness in _all_liveness()
    ]
    assert len(cases) == 81 * 28 * 9 == 20_412

    idle = [
        (signals, extraction, liveness)
        for signals, extraction, liveness in cases
        if derive_status_with_liveness(signals, liveness, extraction=extraction) is Status.IDLE
    ]
    expected = [
        (signals, extraction, liveness)
        for signals, extraction, liveness in cases
        if liveness.process_alive is Tri.TRUE
        and liveness.cursor_advanced_recently is Tri.FALSE
        and derive_status(signals, extraction=extraction) is Status.AWAITING_HUMAN
    ]

    assert idle == expected
    assert len(idle) > 0, "IDLE is unreachable, so this test proves nothing"


def test_derive_status_still_cannot_return_idle_at_all():
    """The layering, asserted from the other side.

    `apply_liveness` is a separate step precisely so liveness is not a
    signal, and this is what would fail if someone folded it into the rule
    list to save a call — every existing caller in the tree passes no
    liveness and must keep its Phase 1 and Phase 3.6 ranges exactly.
    """
    returned = {
        derive_status(signals, extraction=extraction)
        for signals in _all_signal_combinations()
        for extraction in _all_extractions()
    }

    assert Status.IDLE not in returned
    assert returned == set(REFINED_STATUS_RANGE)


# --- liveness: building it from a pane and a clock ---------------------------


def test_observe_liveness_distinguishes_never_seen_from_quiet():
    """A session Palaver has not yet watched advance is not a quiet session.

    Collapsing `None` into "no advance" would report `IDLE` for every
    session on the first tick after a daemon restart — including ones
    actively working.
    """
    never = observe_liveness(os.getpid(), last_advance=None, now=NOW)
    assert never.cursor_advanced_recently is Tri.UNKNOWN
    assert never.process_alive is Tri.TRUE

    quiet = observe_liveness(
        os.getpid(), last_advance=NOW - DEFAULT_IDLE_WINDOW - timedelta(seconds=1), now=NOW
    )
    assert quiet.cursor_advanced_recently is Tri.FALSE

    busy = observe_liveness(os.getpid(), last_advance=NOW - timedelta(seconds=5), now=NOW)
    assert busy.cursor_advanced_recently is Tri.TRUE

    # The boundary itself counts as recent: exactly at the window edge is not
    # yet quiet.
    edge = observe_liveness(os.getpid(), last_advance=NOW - DEFAULT_IDLE_WINDOW, now=NOW)
    assert edge.cursor_advanced_recently is Tri.TRUE


def test_observe_liveness_distinguishes_no_pane_from_a_dead_process():
    """`pid=None` is `UNKNOWN`, never `FALSE`.

    `palaver observe` sees every session on the machine and only some of
    them have a pane. Reading "no join" as "dead" would demote every
    headless `WORKING` session to `UNKNOWN` — the whole status command,
    wrong, from one collapsed value.
    """
    no_pane = observe_liveness(None, last_advance=NOW, now=NOW)
    assert no_pane.process_alive is Tri.UNKNOWN

    dead = observe_liveness(999_999, last_advance=NOW, now=NOW, alive_probe=lambda pid: False)
    assert dead.process_alive is Tri.FALSE

    assert apply_liveness(Status.WORKING, no_pane) is Status.WORKING
    assert apply_liveness(Status.WORKING, dead) is Status.UNKNOWN
