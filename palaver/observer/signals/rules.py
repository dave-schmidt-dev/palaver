"""The ordered status rule list, its provenance, and the per-source coverage gate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from palaver.extract.persist import Extraction

from .ranges import Status, Tri
from .signal_set import DEFAULT_COVERAGE_THRESHOLD, Signals, SourceCoverage


def _has_content(value: str | None) -> bool:
    """Report whether an extraction field carries a non-empty claim.

    Whitespace is stripped and nothing else is interpreted — see the module
    docstring for why prose forms like `"none"` are deliberately left to read
    as content.
    """
    return value is not None and bool(value.strip())


def _is_affirmatively_empty(value: str | None) -> bool:
    """Report whether the pass had an opinion on this field and it was "nothing".

    `None` is not affirmatively empty: it is the absence of an opinion, and
    the distinction is the whole of the `DONE` rule.
    """
    return value is not None and not value.strip()


@dataclass(frozen=True)
class StatusDerivation:
    """One status, plus every signal the rule list read to reach it.

    `consulted` exists so the per-source coverage gate (task 7.3) can be
    computed rather than tabulated. A hand-written status-to-signals table
    would be a second copy of the rule order, maintained by hand, and wrong
    the first time a rule moved; this records what the interpreter actually
    read, in the order it read it.

    It is *every* signal examined up to and including the deciding one, not
    only the deciding one. A `WORKING` result consulted `unresolved_tool_error`
    and found it not `TRUE` — so if that signal is unreadable for most of a
    source's sessions, `WORKING` is exactly as suspect as `ERROR` would have
    been: the rule list may have skipped rule 3 for want of evidence rather
    than for want of an error.

    Attributes:
        status: What `derive_status()` returns, ungated.
        consulted: Names from `SIGNAL_NAMES`, in rule-evaluation order.
    """

    status: Status
    consulted: tuple[str, ...]


def under_covered(
    consulted: Sequence[str],
    coverage: SourceCoverage,
    *,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> tuple[str, ...]:
    """Name the consulted signals this source does not cover well enough.

    Args:
        consulted: Signal names to check, e.g. a `StatusDerivation.consulted`.
        coverage: Per-signal coverage percentages for one source. A signal
            absent from the mapping counts as 0% — unmeasured is not
            presumed adequate, for the same reason
            `CoverageReport.percentage` returns 0.0 over an empty sample
            rather than 100% by vacuity.
        threshold: Minimum percentage a signal must reach.

    Returns:
        The under-covered names, in `consulted` order. Empty when every
        consulted signal clears the threshold.
    """
    return tuple(name for name in consulted if coverage.get(name, 0.0) < threshold)


def derive_status(signals: Signals, *, extraction: Extraction | None = None) -> Status:
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

    5. **Turn ended → the refinement rules below.** Control is back with the
       human. Structure proves that much and nothing more, so without an
       extraction the answer is `AWAITING_HUMAN` — never `DONE`. This is the
       brief's single named prohibition, and it is the reason `DONE` is
       outside `PHASE1_STATUS_RANGE`.

    6. **Otherwise → `UNKNOWN`.** Reached only when the source read cleanly
       but the turn boundary was not determinable. This is a terminal rule,
       not a fallthrough default: there is deliberately no guess here.

    The ended-turn refinement (task 3.6), in order, reached only from rule 5
    and therefore unable to overturn rules 1 through 4:

    5a. **No extraction → `AWAITING_HUMAN`.** The model was unavailable,
        timed out, or raised, and the caller said so by passing `None`.
        Phase 1 behaviour exactly.

    5b. **`blockers_now` has content → `BLOCKED`.** First among the
        refinements: a blocker is the most actionable thing this system can
        tell a human, and it outranks a question because a session that is
        both blocked and curious needs the blocker cleared first.

    5c. **`open_questions` has content → `QUESTION`.** Ahead of
        `WAITING_FOR_USER` because it is the strict refinement of it: both
        say the human owes the session something, and this one says what.
        `open_questions` is the third discriminator the brief's three-way
        split requires — `remaining_work` and `blockers_now` alone cannot
        produce three ended-turn values, and this module has always
        documented `AWAITING_HUMAN` as the union of exactly these three.

    5d. **`remaining_work` has content → `WAITING_FOR_USER`.** The agent
        stopped with work outstanding.

    5e. **`remaining_work` is affirmatively empty → `DONE`.** The only
        status in this module that requires positive evidence rather than
        the absence of contrary evidence: the pass must have had an opinion
        on `remaining_work` (`""`, not `None`) and that opinion must be
        "nothing". See the module docstring for the spike defect this
        forbids.

    5f. **Otherwise → `AWAITING_HUMAN`.** An extraction that said nothing
        about remaining work refines nothing, so the coarse answer stands.

    Task 7.3's per-source coverage gate is **not** a parameter here, and
    that is deliberate rather than incidental: this function's contract is
    that it takes no mapping of any kind, which is checked by INV-7's gate
    test and is what makes "no status can reach it under any key" a property
    a reader can confirm from the signature alone. The gate lives in
    `derive_status_for_source`, which wraps this one.

    A note on what is deliberately *not* built here: an unresolved
    `AskUserQuestion` gives `turn_boundary` the basis
    `BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE`, which would corroborate
    `QUESTION` deterministically. Reaching it would mean adding the basis to
    `Signals`, which changes `SIGNAL_NAMES` and the coverage contract built
    on it. Recorded as available evidence, not taken.

    Args:
        signals: The deterministic signal set. Takes no model output.
        extraction: Optional refinement content from one extraction pass,
            keyword-only and defaulting to `None` so that no caller written
            before task 3.6 can receive a status it has never seen. Must be
            an `Extraction`; a raw model payload (a `dict`) is refused rather
            than read, which is why a response carrying a `status` key cannot
            reach this function at all — see
            `extraction_from_model_payload`. There is no `remaining_work`
            parameter, no `blockers_now` parameter, no `status` parameter,
            and no `**kwargs` that could swallow one (INV-7).

    Returns:
        A `Status` member, drawn from `PHASE1_STATUS_RANGE` when `extraction`
        is `None` and from `REFINED_STATUS_RANGE` otherwise.

    Raises:
        TypeError: `extraction` is neither `None` nor an `Extraction`.
    """
    return derive_status_with_provenance(signals, extraction=extraction).status


def derive_status_for_source(
    signals: Signals,
    coverage: SourceCoverage,
    *,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    extraction: Extraction | None = None,
) -> Status:
    """Compute a status, then withdraw it if its source cannot support it.

    `derive_status()` answers "what does this session's signal set say".
    This answers the narrower question a report has to answer: "what may
    this *source* be allowed to say". A source whose coverage for a
    consulted signal falls below `threshold` yields `UNKNOWN`, whatever the
    rules concluded.

    The gate runs last and can only ever weaken the answer — never turn one
    status into a different confident one. That ordering is the point: a
    coverage number is a property of a whole sample, not of the session in
    hand, so it may withdraw a status but must never manufacture one.

    Per-session, an undeterminable signal already yields `UNKNOWN` through
    rules 1, 2 and 6. The gate catches what those rules cannot see: the
    signal *was* determinable for this session, but the derivation behind it
    plainly does not fit the source the session came from, so the sessions it
    does answer for are as likely to be shape coincidences as observations.
    That is the plan's adapter-interface rollback point — a source that
    generalizes badly degrades honestly instead of degrading silently.

    Args:
        signals: The deterministic signal set. Takes no model output.
        coverage: Per-signal coverage percentages for the source this
            session came from, e.g.
            `palaver.cli.diagnose.CoverageReport.as_coverage()`. Positional,
            and with no default, because a caller reaching for the gated
            entry point without a measurement has nothing to gate with and
            should be calling `derive_status()` instead.
        threshold: Minimum coverage percentage a consulted signal must
            reach. A parameter rather than a module constant read directly,
            so a test can drive the gate from both sides without
            monkeypatching.
        extraction: Optional refinement content; see `derive_status()`.

    Returns:
        The status `derive_status()` would return, or `Status.UNKNOWN` when
        a consulted signal falls below `threshold` for this source.

    Raises:
        TypeError: `extraction` is neither `None` nor an `Extraction`.
    """
    derivation = derive_status_with_provenance(signals, extraction=extraction)
    if under_covered(derivation.consulted, coverage, threshold=threshold):
        return Status.UNKNOWN
    return derivation.status


def derive_status_with_provenance(
    signals: Signals, *, extraction: Extraction | None = None
) -> StatusDerivation:
    """Run `derive_status()`'s rule list, reporting which signals it read.

    The rule list itself, and the only copy of it. `derive_status()` is a
    thin wrapper that applies the coverage gate to this result; see its
    docstring for every rule, in order, with the reason it sits where it
    does.

    Args:
        signals: The deterministic signal set. Takes no model output.
        extraction: Optional refinement content; see `derive_status()`.

    Returns:
        The status and the signal names consulted to reach it, in
        rule-evaluation order.

    Raises:
        TypeError: `extraction` is neither `None` nor an `Extraction`.
    """
    if extraction is not None and not isinstance(extraction, Extraction):
        raise TypeError(
            f"derive_status() takes an Extraction or None, got "
            f"{type(extraction).__name__}: {extraction!r}. A raw model payload must cross "
            f"extraction_from_model_payload() first (INV-7)."
        )

    consulted: list[str] = ["source_readable"]
    if signals.source_readable is not Tri.TRUE:
        return StatusDerivation(status=Status.UNKNOWN, consulted=tuple(consulted))

    consulted.append("signal_records_parsed")
    if signals.signal_records_parsed is not Tri.TRUE:
        return StatusDerivation(status=Status.UNKNOWN, consulted=tuple(consulted))

    consulted.append("unresolved_tool_error")
    if signals.unresolved_tool_error is Tri.TRUE:
        return StatusDerivation(status=Status.ERROR, consulted=tuple(consulted))

    consulted.append("agent_turn_ended")
    if signals.agent_turn_ended is Tri.FALSE:
        return StatusDerivation(status=Status.WORKING, consulted=tuple(consulted))

    if signals.agent_turn_ended is Tri.TRUE:
        return StatusDerivation(status=_refine_ended_turn(extraction), consulted=tuple(consulted))

    return StatusDerivation(status=Status.UNKNOWN, consulted=tuple(consulted))


def _refine_ended_turn(extraction: Extraction | None) -> Status:
    """Split rule 5's `AWAITING_HUMAN` using one extraction pass's content.

    Rules 5a through 5f, in order; see `derive_status()` for why each sits
    where it does. Reached only from rule 5, so nothing here can overturn a
    deterministic signal.
    """
    if extraction is None:
        return Status.AWAITING_HUMAN

    if _has_content(extraction.blockers_now):
        return Status.BLOCKED

    if _has_content(extraction.open_questions):
        return Status.QUESTION

    if _has_content(extraction.remaining_work):
        return Status.WAITING_FOR_USER

    if _is_affirmatively_empty(extraction.remaining_work):
        return Status.DONE

    return Status.AWAITING_HUMAN
