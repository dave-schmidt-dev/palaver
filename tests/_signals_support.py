"""Tests for the deterministic signal set and the ordered status rule list.

These tests open no live session store and write no fixture: `Signals` is a
plain value and `derive_status()` is a pure function of it and an optional
`Extraction`, so almost every case here is built in memory (INV-3 is
satisfied trivially rather than by convention). The exceptions are the
task 3.6 refinement tests that read committed, sanitized fixtures from
`tests/fixtures/` to check a real session's structure against the finer
labels the corpus already records — never David's live `~/.claude/` (INV-9).

`tests/test_signals.py::test_status_is_never_model_supplied` is INV-7's gate
test, named as such in `INVARIANTS.md`.
"""

import itertools

from palaver.observer.signals import (
    SIGNAL_NAMES,
    Signals,
    Status,
    Tri,
)

#: The unrefined range, written out as a literal rather than imported, so this
#: module asserts the contract independently of the constant it is checking.
#: A single edit to `PHASE1_STATUS_RANGE` cannot move both sides at once.
EXPECTED_PHASE1_RANGE = {
    Status.WORKING,
    Status.AWAITING_HUMAN,
    Status.ERROR,
    Status.UNKNOWN,
}


def _signals(
    *,
    source_readable: Tri = Tri.TRUE,
    signal_records_parsed: Tri = Tri.TRUE,
    unresolved_tool_error: Tri = Tri.FALSE,
    agent_turn_ended: Tri = Tri.FALSE,
) -> Signals:
    """Build a signal set from a clean, mid-turn baseline.

    The baseline reads: the store was read fine, every signal record parsed,
    no tool error, agent still holds the turn — i.e. `WORKING`. Each test
    overrides only the signals it is actually about, so the delta under test
    is visible at the call site.
    """
    return Signals(
        source_readable=source_readable,
        signal_records_parsed=signal_records_parsed,
        unresolved_tool_error=unresolved_tool_error,
        agent_turn_ended=agent_turn_ended,
    )


def _ended_turn() -> Signals:
    """The clean ended-turn signal set — exactly the case rule 5 refines."""
    return _signals(agent_turn_ended=Tri.TRUE)


def _all_signal_combinations() -> list[Signals]:
    """Enumerate the entire signal space: every `Tri` value of every signal."""
    return [
        Signals(**dict(zip(SIGNAL_NAMES, combination, strict=True)))
        for combination in itertools.product(Tri, repeat=len(SIGNAL_NAMES))
    ]
