"""Process liveness and its adjustment of a derived status."""

from __future__ import annotations

from dataclasses import dataclass, fields

from palaver.extract.persist import Extraction

from .ranges import Status, Tri
from .rules import derive_status
from .signal_set import Signals


@dataclass(frozen=True)
class Liveness:
    """What the pane join observed about the process behind a session.

    Both fields are `Tri` for the same reason every `Signals` field is: a
    session with no live pane, a pane whose join was refused, or a session
    Palaver has never yet seen advance are all *unobservable* rather than
    negative, and collapsing them into `FALSE` is what would turn a busy
    session into a confident `IDLE`.

    Attributes:
        process_alive: `TRUE` when a process for the joined agent was found
            running at observation time, `FALSE` when it was looked for and
            was gone, `UNKNOWN` when no join was made — no pane, a refused
            join, or a headless observation. `UNKNOWN` is the normal case,
            not a defect: `palaver observe` sees every session on the
            machine and only some of them have a pane.
        cursor_advanced_recently: `TRUE` when the session store grew within
            the idle window ending at observation time, `FALSE` when the
            last recorded advance is older than that window, `UNKNOWN` when
            no advance has ever been recorded for this session.

            Read the name forwards: `FALSE` says "nothing was written during
            the window", not "nothing was ever written". The distinction is
            the whole of rule L2 — a session Palaver has only just started
            watching has no advance on record, and reading that as a quiet
            session would report `IDLE` for one that is actively working.
    """

    process_alive: Tri
    cursor_advanced_recently: Tri

    def __post_init__(self) -> None:
        """Reject any field that is not a `Tri`.

        Raises:
            TypeError: If any field's value is not a `Tri` member. Same
                reason as `Signals.__post_init__`: a raw `bool` here would
                skip every rule silently and yield an unchanged status that
                looks deliberate.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Tri):
                raise TypeError(
                    f"Liveness.{field.name} must be a Tri member, got "
                    f"{type(value).__name__}: {value!r}"
                )


def apply_liveness(status: Status, liveness: Liveness) -> Status:
    """Adjust a derived status using what the pane join saw of the process.

    Runs after the rule list and after `derive_status_for_source`'s coverage
    gate, and is deliberately tiny — two rules, each requiring positive
    evidence, each touching exactly one input status:

    **L1. Dead process and `WORKING` → `UNKNOWN`.** `WORKING` is the only
    status that asserts something is happening *now*, and a process that no
    longer exists cannot be holding a turn. What the transcript recorded is
    then not wrong so much as stale, and Palaver cannot tell from it whether
    the turn ended or the agent was killed mid-turn — so `UNKNOWN`, not
    `AWAITING_HUMAN`, which would claim the human is owed something.

    No other status is demoted. `AWAITING_HUMAN`, `DONE`, `WAITING_FOR_USER`,
    `QUESTION`, and `BLOCKED` are claims about the state the transcript was
    left in, and a dead process corroborates them rather than contradicting
    them; `ERROR` likewise describes the last observed tool outcome. Demoting
    those would discard a good answer for a session the human still needs to
    look at.

    **L2. Live process, quiet for the idle window, and `AWAITING_HUMAN` →
    `IDLE`.** `AWAITING_HUMAN` is the contentless ended-turn status — the
    union of `DONE`, `WAITING_FOR_USER`, and `QUESTION` — and `IDLE` strictly
    adds to it: not merely "the turn ended" but "the turn ended and has
    stayed ended, untouched, for the whole window". Requires both halves
    affirmatively (`process_alive is TRUE`, `cursor_advanced_recently is
    FALSE`), so an unobservable process and a never-yet-seen cursor both
    leave the answer alone.

    Only `AWAITING_HUMAN` is refined. `BLOCKED`, `QUESTION`,
    `WAITING_FOR_USER`, and `DONE` all say more than `IDLE` does, and a quiet
    ten minutes is not evidence against any of them — a session that has been
    sitting on a blocker for an hour is *more* worth surfacing, not less, and
    rewriting it to `IDLE` would hide the blocker behind the one status that
    means "nothing to do here".

    **Everything else passes through unchanged, including `UNKNOWN`.** A
    session whose store could not be read is `UNKNOWN` no matter how alive
    and how quiet its process is: liveness says a process exists, never that
    the observation succeeded. That is what keeps this layer compatible with
    the coverage gate, which may only ever weaken an answer — `apply_liveness`
    can weaken `WORKING` and can sharpen `AWAITING_HUMAN`, and can do nothing
    at all to a status the rules already withdrew.

    Args:
        status: What `derive_status()` or `derive_status_for_source()`
            returned for this session.
        liveness: What the pane join observed. Pass
            `Liveness(Tri.UNKNOWN, Tri.UNKNOWN)` for a session with no pane;
            it is a no-op by construction rather than by special case.

    Returns:
        The adjusted status, drawn from `LIVE_STATUS_RANGE`.
    """
    if liveness.process_alive is Tri.FALSE and status is Status.WORKING:
        return Status.UNKNOWN

    if (
        liveness.process_alive is Tri.TRUE
        and liveness.cursor_advanced_recently is Tri.FALSE
        and status is Status.AWAITING_HUMAN
    ):
        return Status.IDLE

    return status


def derive_status_with_liveness(
    signals: Signals,
    liveness: Liveness,
    *,
    extraction: Extraction | None = None,
) -> Status:
    """Run the rule list, then adjust the result with process liveness.

    The composition `apply_liveness(derive_status(...), liveness)`, provided
    so callers do not each rewrite it and so the range over the full input
    space is one function's contract rather than a convention.

    Args:
        signals: The deterministic signal set. Takes no model output.
        liveness: What the pane join observed; see `apply_liveness`.
        extraction: Optional refinement content; see `derive_status()`.

    Returns:
        A `Status` member from `LIVE_STATUS_RANGE`.

    Raises:
        TypeError: `extraction` is neither `None` nor an `Extraction`.
    """
    return apply_liveness(derive_status(signals, extraction=extraction), liveness)
