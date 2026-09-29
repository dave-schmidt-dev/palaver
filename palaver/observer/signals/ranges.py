"""Tri and Status enums, plus the derivable status-range constants."""

from __future__ import annotations

from enum import Enum


class Tri(Enum):
    """A three-valued signal: true, false, or not determinable.

    `UNKNOWN` means the observation could not be made — the store was
    unreadable, the record shape was unrecognized, the source offers no such
    evidence. It is never a synonym for `FALSE`.

    Boolean coercion raises rather than returning a value. Every `Enum`
    member is truthy by default, so `if signal:` would silently read
    `Tri.FALSE` and `Tri.UNKNOWN` as true; `if not signal:` would read all
    three as false. Both are the exact defect INV-7's rationale warns about,
    and both are invisible at review time. Comparing against a specific
    member (`is Tri.TRUE`) is the only supported test.
    """

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:
        """Refuse boolean coercion.

        Raises:
            TypeError: Always. Compare against a member instead, e.g.
                `signal is Tri.TRUE`.
        """
        raise TypeError(
            f"{type(self).__name__} is three-valued and has no boolean "
            f"meaning; compare against a member (e.g. `x is Tri.TRUE`) "
            f"rather than testing truthiness of {self!r}"
        )

    @classmethod
    def from_optional(cls, value: bool | None) -> Tri:
        """Lift an optional boolean into a `Tri`.

        For producers that already model absence as `None`. `None` becomes
        `UNKNOWN`, never `FALSE`.

        Args:
            value: `True`, `False`, or `None` for "could not determine".

        Returns:
            The corresponding `Tri` member.
        """
        if value is None:
            return cls.UNKNOWN
        return cls.TRUE if value else cls.FALSE


class Status(Enum):
    """Every status Palaver will ever report, across all phases.

    Which members are reachable depends on the inputs a caller supplies, and
    the two ranges below are the contract: `PHASE1_STATUS_RANGE` when no
    `extraction` is passed, `REFINED_STATUS_RANGE` when one is. `IDLE` is
    reachable from neither and is defined anyway, so that "nothing here can
    return `IDLE` yet" is an assertion a test can falsify rather than a
    property of the enum's size, and so the staging in the plan's §4.2 table
    is visible in one place instead of arriving as separate additions.

    Members:
        WORKING: The agent holds the turn and is doing something.
        AWAITING_HUMAN: The turn ended and control is back with the human,
            with nothing to say about why. The union of `DONE`,
            `WAITING_FOR_USER`, and `QUESTION`, and the answer whenever
            extraction is unavailable or had no opinion.
        ERROR: The most recent tool outcome is an unresolved error.
        UNKNOWN: No signal supports any status claim.
        DONE: The turn ended and extraction affirmatively reports no
            remaining work. Requires positive evidence (task 3.6).
        WAITING_FOR_USER: The turn ended with work still outstanding.
        QUESTION: The turn ended with an unanswered question.
        BLOCKED: The turn ended against something blocking progress now.
        IDLE: The process is alive, the turn ended, and nothing has been
            written to the session for the idle window. Reachable only
            through `apply_liveness` (task 5.2), never from
            `derive_status()`.
    """

    WORKING = "WORKING"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

    # Reachable only with an `Extraction` (task 3.6). See §4.2.
    DONE = "DONE"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    QUESTION = "QUESTION"
    BLOCKED = "BLOCKED"

    # Reachable only through `apply_liveness` (task 5.2). Unreachable from
    # `derive_status()` under every input it accepts.
    IDLE = "IDLE"


#: The exact set of statuses `derive_status()` may return when no
#: `extraction` is supplied. This is the whole of Phase 1's range, and it
#: stays the live contract for every existing caller: `extraction` is
#: keyword-only and defaults to `None`, so no caller written before task 3.6
#: can be handed a status it has never seen. It is also what a model outage
#: degrades to. Asserted by `tests/test_signals.py::test_phase1_status_range`
#: over the whole signal space, not by inspecting this module's source.
PHASE1_STATUS_RANGE = frozenset(
    {
        Status.WORKING,
        Status.AWAITING_HUMAN,
        Status.ERROR,
        Status.UNKNOWN,
    }
)

#: The exact set `derive_status()` may return once an `Extraction` is
#: supplied (task 3.6): the four above, plus the brief's three ended-turn
#: values and `BLOCKED`. A strict superset of `PHASE1_STATUS_RANGE` by
#: construction — refinement splits the ended-turn branch and removes
#: nothing, so every unrefined answer stays reachable. `IDLE` is excluded and
#: is asserted unreachable across the signal space crossed with the
#: extraction space.
REFINED_STATUS_RANGE = PHASE1_STATUS_RANGE | frozenset(
    {
        Status.DONE,
        Status.WAITING_FOR_USER,
        Status.QUESTION,
        Status.BLOCKED,
    }
)

#: The exact set `apply_liveness()` may return, and therefore the range of
#: `derive_status_with_liveness()` (task 5.2): everything refinement can
#: reach, plus `IDLE`. This is the whole `Status` enum, which is the point —
#: with liveness in play there is no member left that nothing can produce, so
#: `set(Status) - returned == frozenset()` is a fact `tests/test_pane_join.py`
#: states rather than one this module's source has to be read for.
LIVE_STATUS_RANGE = REFINED_STATUS_RANGE | frozenset({Status.IDLE})
