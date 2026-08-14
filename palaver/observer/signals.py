"""Deterministic status signals and the ordered rule list that consumes them.

This module is the whole of INV-7: *status is computed from deterministic
signals; the model never sets it*. `derive_status()` owns the rule list in
Python, takes no model output of any kind, and is a pure function of a
`Signals` value.

Why the rule list lives here rather than in a prompt. Spike run 1 gave a 4B
model an explicit ordered rule list and it still returned `IDLE` for a
plainly-paused session. Spike run 2 isolated the reason: the model honours
rules whose predicates are *computed signals* and ignores rules whose
predicates are *its own generated fields*. A rule list is only as strong as
the thing evaluating it, so this one is evaluated by the interpreter.

**Phase 1's range is exactly `PHASE1_STATUS_RANGE`** — `WORKING`,
`AWAITING_HUMAN`, `ERROR`, `UNKNOWN`. `Status` deliberately defines five more
members (`DONE`, `WAITING_FOR_USER`, `QUESTION`, `BLOCKED`, `IDLE`), which
Phase 3.6 and Phase 5.2 make reachable once semantic extraction and process
liveness exist. They are defined-but-unreachable on purpose: a status the
enum cannot spell is trivially unreachable, whereas a status that is
nameable, storable, and still never returned is a claim the range test can
actually falsify.

The single named prohibition, from the brief: **do not equate lack of
terminal output with `DONE`.** An ended turn with no extraction is
`AWAITING_HUMAN`, never `DONE`. Structure can prove that control returned to
the human; it cannot prove the work is finished. `AWAITING_HUMAN` is the
union of `DONE`, `WAITING_FOR_USER`, and `QUESTION`, and it is exactly as
much as Phase 1's inputs support. A confident wrong `DONE` tells the human a
session needs nothing when it may be waiting on them — the most costly error
this system can make.

`UNKNOWN` is a first-class value, not an error case. When no signal supports
any status, `derive_status()` returns `UNKNOWN` rather than guessing. There
is no fallthrough branch that picks a plausible default.

Three-valued signals. Every signal here is a `Tri` — true, false, or
*unknown* — because for every one of them absence is a real, observable
condition, and "could not determine" is not the same claim as "determined it
to be false". Collapsing unknown into false is how `UNKNOWN` stops being
reachable and how a guess gets reported as fact, so `Tri` refuses to be used
in a boolean context at all: `if signals.agent_turn_ended:` raises
`TypeError` instead of silently reading `UNKNOWN` as truthy.

Note what the ternary domain does and does not promise. It is the range of
what a *producer* may honestly assert about a signal; it is not a
requirement that each of a signal's three values map to a distinct status.
Under the rule list below, only `agent_turn_ended` discriminates all three.
`source_readable` and `signal_records_parsed` treat `FALSE` and `UNKNOWN`
alike, because "we could not confirm we read the session" is as weak a
footing for a status claim as "we failed to read it". `unresolved_tool_error`
treats `UNKNOWN` and `FALSE` alike in the other direction, because `ERROR`
is a positive claim that requires positive evidence. Both collapses are at
the rule level, deliberate, and named in the rules' own docstrings — the
signal values themselves stay distinct so a caller (and
`palaver diagnose --coverage`, task 1.6) can tell the cases apart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
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

    Only the members in `PHASE1_STATUS_RANGE` are reachable from
    `derive_status()` today. The rest are defined here so that "Phase 1
    cannot return `DONE`" is an assertion a test can falsify rather than a
    property of the enum's size, and so the staging in the plan's §4.2 table
    is visible in one place instead of arriving as five separate additions.

    Members:
        WORKING: The agent holds the turn and is doing something.
        AWAITING_HUMAN: The turn ended and control is back with the human.
            The Phase 1 union of `DONE`, `WAITING_FOR_USER`, and `QUESTION`.
        ERROR: The most recent tool outcome is an unresolved error.
        UNKNOWN: No signal supports any status claim.
        DONE: Unreachable until Phase 3.6 (needs semantic extraction).
        WAITING_FOR_USER: Unreachable until Phase 3.6.
        QUESTION: Unreachable until Phase 3.6.
        BLOCKED: Unreachable until Phase 3.6 (needs `blockers_now`).
        IDLE: Unreachable until Phase 5.2 (needs process liveness).
    """

    WORKING = "WORKING"
    AWAITING_HUMAN = "AWAITING_HUMAN"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

    # Staged, and unreachable from `derive_status()` in Phase 1. See §4.2.
    DONE = "DONE"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    QUESTION = "QUESTION"
    BLOCKED = "BLOCKED"
    IDLE = "IDLE"


#: The exact set of statuses `derive_status()` may return in Phase 1. Every
#: other `Status` member is unreachable until the phase that supplies its
#: input lands — asserted by `tests/test_signals.py::test_phase1_status_range`
#: over the whole signal space, not by inspecting this module's source.
PHASE1_STATUS_RANGE = frozenset(
    {
        Status.WORKING,
        Status.AWAITING_HUMAN,
        Status.ERROR,
        Status.UNKNOWN,
    }
)


@dataclass(frozen=True)
class Signals:
    """The deterministic signal set `derive_status()` reads.

    Every field is required. There are no defaults, because a default would
    let a caller that forgot a signal silently assert something about it —
    and the only safe default (`UNKNOWN`) would then hide the omission
    behind a status that looks deliberate. A producer that cannot determine
    a signal must say so by passing `Tri.UNKNOWN` explicitly.

    Nothing in this set comes from a model. Task 3.6 adds `remaining_work`
    and `blockers_now` as *content* inputs to `derive_status()`; even then
    the model supplies the content and Python still owns the rule list
    (INV-7).

    Attributes:
        source_readable: `TRUE` when the session store was opened and read
            without error on this observation. `FALSE` when the adapter
            raised, the path was missing, or permission was denied.
        signal_records_parsed: `TRUE` when every record the other signals
            were actually derived from decoded successfully. Scoped
            deliberately to *those* records — the message-bearing tail the
            turn boundary and tool-outcome signals read — and not to every
            record in the file. Claude Code transcripts are read from offset
            zero by `last_message_bearing_record`, so a file-wide definition
            would let one corrupt line anywhere in a session's history pin
            that session to `UNKNOWN` for the rest of its life. A trailing
            partial line from an in-flight write is *not* a corruption
            condition here: `read_complete_records` already withholds it and
            re-reads it whole, so it never reaches a parser.
        unresolved_tool_error: `TRUE` when the most recent tool outcome in
            the observed window is an error. A later successful tool outcome
            clears it back to `FALSE` — this is a claim about the session's
            latest outcome, not about whether an error ever occurred.
        agent_turn_ended: `TRUE` when the agent handed control back to the
            human, `FALSE` when it still holds the turn (including
            mid-`tool_use`). Derived structurally by
            `palaver.observer.turn_boundary` (task 1.6) from the last
            message-bearing record's role read *through* INV-8's channel
            classification, plus `tool_use`/`tool_result` pairing. Consumed
            here, never derived here.
    """

    source_readable: Tri
    signal_records_parsed: Tri
    unresolved_tool_error: Tri
    agent_turn_ended: Tri

    def __post_init__(self) -> None:
        """Reject any field that is not a `Tri`.

        A raw `bool` passed where a `Tri` belongs is the collapse this
        module exists to prevent, and it would otherwise fail silently:
        `True is Tri.TRUE` is `False`, so every rule would skip and the
        caller would get a plausible-looking `UNKNOWN`.

        Raises:
            TypeError: If any field's value is not a `Tri` member.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Tri):
                raise TypeError(
                    f"Signals.{field.name} must be a Tri member, got "
                    f"{type(value).__name__}: {value!r}"
                )


#: Every signal name in `Signals`, in rule-evaluation order. Task 1.6's
#: `palaver diagnose --coverage` reports one coverage percentage per entry.
SIGNAL_NAMES: tuple[str, ...] = tuple(field.name for field in fields(Signals))


def derive_status(signals: Signals) -> Status:
    """Compute a session's status from deterministic signals only.

    The ordered rule list. Each rule is stated with the reason it sits where
    it does, because the order is the contract — a wrong order ships a
    confident wrong status rather than a visible failure.

    1. **Unreadable source → `UNKNOWN`.** A component that could not read
       the session must not report on it. `UNKNOWN` and `FALSE` are treated
       alike here: "we cannot confirm we read it" is no better a footing for
       a status claim than "we failed to read it", and the alternative is to
       report a status derived from signals whose provenance is in doubt.

    2. **Unparsed signal records → `UNKNOWN`.** If a record the downstream
       signals were derived from did not decode, those signals were computed
       over an incomplete view and Palaver cannot know whether the missing
       record was the decisive one. Same `FALSE`/`UNKNOWN` treatment, same
       reason.

    3. **Unresolved tool error → `ERROR`.** Before the turn-boundary rules,
       not after. The turn boundary can only ever produce `WORKING` or
       `AWAITING_HUMAN`, so any ordering that consulted it first would make
       `ERROR` unreachable for every session whose boundary signal is
       determinable — which is nearly all of them. `ERROR` also strictly
       refines `AWAITING_HUMAN`: both say "look at this session", and this
       one says why. `UNKNOWN` is treated as `FALSE` here, in the opposite
       direction to rules 1 and 2: `ERROR` is a positive claim about an
       observed outcome and is never asserted without positive evidence.

    4. **Turn not ended → `WORKING`.** The agent holds the turn. Includes
       the mid-`tool_use` case, which task 1.6 resolves to "still working".

    5. **Turn ended → `AWAITING_HUMAN`.** Control is back with the human.
       Never `DONE`: structure proves the turn ended, and nothing more. This
       is the brief's single named prohibition, and it is the reason `DONE`
       is outside `PHASE1_STATUS_RANGE`.

    6. **Otherwise → `UNKNOWN`.** Reached only when the source read cleanly
       but the turn boundary was not determinable. This is a terminal rule,
       not a fallthrough default: there is deliberately no guess here.

    Args:
        signals: The deterministic signal set. Takes no model output — no
            `remaining_work`, no `blockers_now`, no `status` string, and no
            `**kwargs` that could swallow one (INV-7).

    Returns:
        A `Status` member, always drawn from `PHASE1_STATUS_RANGE`.
    """
    if signals.source_readable is not Tri.TRUE:
        return Status.UNKNOWN

    if signals.signal_records_parsed is not Tri.TRUE:
        return Status.UNKNOWN

    if signals.unresolved_tool_error is Tri.TRUE:
        return Status.ERROR

    if signals.agent_turn_ended is Tri.FALSE:
        return Status.WORKING

    if signals.agent_turn_ended is Tri.TRUE:
        return Status.AWAITING_HUMAN

    return Status.UNKNOWN
