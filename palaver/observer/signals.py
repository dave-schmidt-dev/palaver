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

**Without extraction the range is exactly `PHASE1_STATUS_RANGE`** —
`WORKING`, `AWAITING_HUMAN`, `ERROR`, `UNKNOWN`. That is not history: it is
the live contract for every caller that passes no `extraction`, which is
every caller in the tree today, and it is what a model outage degrades to.
With an extraction the range is `REFINED_STATUS_RANGE`, which adds `DONE`,
`WAITING_FOR_USER`, `QUESTION`, and `BLOCKED` (task 3.6). `IDLE` remains
defined-but-unreachable until Phase 5.2 supplies process liveness — a status
the enum cannot spell is trivially unreachable, whereas a status that is
nameable, storable, and still never returned is a claim the range test can
actually falsify.

The single named prohibition, from the brief: **do not equate lack of
terminal output with `DONE`.** An ended turn with no extraction is
`AWAITING_HUMAN`, never `DONE`. Structure can prove that control returned to
the human; it cannot prove the work is finished. `AWAITING_HUMAN` is the
union of `DONE`, `WAITING_FOR_USER`, and `QUESTION`, and it is exactly as
much as the structural signals support on their own. A confident wrong
`DONE` tells the human a session needs nothing when it may be waiting on
them — the most costly error this system can make.

**Task 3.6: refinement, and why it does not weaken any of the above.**
`derive_status()` takes an optional `Extraction` (task 3.4's dataclass, read
here and never modified) and splits the ended-turn branch into the brief's
three values, plus `BLOCKED`. Three properties keep INV-7 intact:

* The model supplies *content* — what work remains, what is blocking, what
  is unanswered — and never a status. `derive_status()` accepts no dict, no
  `status=` argument, and no `**kwargs`; the only refinement input is a
  typed `Extraction`, whose six fields do not include a status and cannot be
  made to. `extraction_from_model_payload()` is the boundary a raw model
  response crosses, and it refuses any status-like key outright, so a prompt
  regression that starts asking for a status is loud rather than silent.
* Refinement lives **inside** the ended-turn branch only. It never overrides
  an unreadable source, an unparsed record, an unresolved tool error, or an
  agent that still holds the turn. Model content refining a coarse structural
  answer is the design; model content overturning a deterministic one is the
  defect INV-7 names, and rule order is what forbids it.
* Absence degrades toward the coarse answer, never toward completion.
  `extraction=None` — the model was unavailable, timed out, or raised —
  returns `AWAITING_HUMAN`, i.e. exactly Phase 1 behaviour. So does an
  extraction that returned nothing about the fields that matter. The caller
  owns that degradation: catch whatever the extractor raises and pass `None`.
  This module opens no socket and calls no model, which is why it can make
  the guarantee at all.

`DONE` is the one status that requires positive evidence rather than the
absence of contrary evidence. The spike's rule was
`WAITING_FOR_USER if remaining_work else DONE`, which reads a *missing*
field as completion — every failed extraction becomes a finished session.
Here, `remaining_work=None` means "this pass had no opinion" (task 3.4's own
reading of the field) and yields `AWAITING_HUMAN`, while `remaining_work=""`
is an affirmative "nothing remains" and is the only thing that yields
`DONE`.

Field text is normalized by stripping whitespace and nothing else. No
`"none"`/`"n/a"`/`"TBD"` special-casing: inferring semantics from model prose
is unbounded, and it is the model deciding status by another route. Strip-only
is safe rather than lazy because every prose form it fails to recognize is
non-empty, and non-empty falls toward `WAITING_FOR_USER` or `BLOCKED` — never
toward `DONE`.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum

from palaver.extract.persist import Extraction

#: The share of a source's sampled sessions a signal must be determinable
#: for before a status derived from that signal may be asserted for that
#: source (task 7.3). Below it, `derive_status()` returns `UNKNOWN`.
#:
#: 50% is a policy constant, not a measurement, and it is deliberately
#: coarse. The claim it encodes is structural rather than statistical: a
#: derivation that cannot read a signal for most of a source's sessions has
#: not been shown to fit that source's format at all, and the minority it
#: does answer for are as likely to be shape coincidences — a record that
#: happens to look Claude-Code-like — as observations. Coverage is not
#: accuracy (see `palaver.cli.diagnose`), so this threshold cannot say a
#: source's answers are *right*; it can only refuse to pass on answers from
#: a derivation that demonstrably does not generalize. Every caller may pass
#: its own value.
DEFAULT_COVERAGE_THRESHOLD = 50.0

#: Per-signal coverage for one source, as percentages keyed by the names in
#: `SIGNAL_NAMES`. `palaver.cli.diagnose.CoverageReport` produces these.
SourceCoverage = Mapping[str, float]


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
        IDLE: Unreachable until Phase 5.2 (needs process liveness).
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

    # Staged for Phase 5.2, and unreachable from `derive_status()` today.
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


@dataclass(frozen=True)
class Signals:
    """The deterministic signal set `derive_status()` reads.

    Every field is required. There are no defaults, because a default would
    let a caller that forgot a signal silently assert something about it —
    and the only safe default (`UNKNOWN`) would then hide the omission
    behind a status that looks deliberate. A producer that cannot determine
    a signal must say so by passing `Tri.UNKNOWN` explicitly.

    Nothing in this set comes from a model, and task 3.6 did not change
    that. Refinement content arrives at `derive_status()` as a separate
    keyword-only `Extraction`, never as a signal: a signal is something
    Palaver observed for itself, and mixing a model's claim in among them
    would put model output behind the same `Tri` a rule trusts absolutely.
    The model supplies the content, Python still owns the rule list (INV-7).

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


class ExtractionPayloadError(ValueError):
    """A raw model response cannot be trusted as status-refinement input.

    A caller that degrades on this must pass `extraction=None` to
    `derive_status()` — i.e. fall back to `AWAITING_HUMAN` — and must not
    reach past the boundary for the field it wanted anyway.
    """


class ModelSuppliedStatusError(ExtractionPayloadError):
    """The model returned a status-like field. INV-7's tripwire.

    Not the primary enforcement: `derive_status()` reads no such field under
    any name, so a payload carrying one is already inert. It is raised
    anyway, because the day a prompt starts asking a 4B model for a status is
    the day this project's central measured finding has been forgotten, and
    that should surface as a failure rather than as a field nobody reads.
    """


#: Keys a model response may not carry, normalized (case-folded, `-` and
#: spaces to `_`). Deliberately broader than the literal `status`: the point
#: is to catch a prompt that started asking for a status under a synonym.
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "status",
        "state",
        "session_status",
        "session_state",
        "agent_status",
        "agent_state",
    }
)

#: The keys `extraction_from_model_payload` reads. Every other key is
#: ignored — including `decisions` and `resolved_questions`, which are
#: durable claims and belong to `palaver.extract.quote_gate`'s write
#: boundary, not to the status path.
REFINEMENT_PAYLOAD_KEYS: tuple[str, ...] = (
    "current_task",
    "remaining_work",
    "blockers_now",
    "open_questions",
)


def _normalized_key(key: object) -> str:
    """Case-fold and punctuation-fold one payload key for the forbidden check."""
    if not isinstance(key, str):
        return ""
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _payload_text(key: str, value: object) -> str | None:
    """Normalize one payload value to the `str | None` an `Extraction` field takes.

    A list is joined rather than passed through, because `Extraction`
    declares `str | None` and because `bool([""])` is `True` — a model that
    returned `["", ""]` for `remaining_work` would otherwise read as
    outstanding work under a truthiness test, which is the exact collapse
    this module exists to prevent.

    `None` (JSON `null`, or an absent key) survives as `None`: it means the
    pass had no opinion, and it is what keeps `DONE` from being the
    fallthrough for a failed extraction.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        if any(not isinstance(item, str) for item in value):
            raise ExtractionPayloadError(
                f"{key!r} is a sequence containing a non-string item: {value!r}"
            )
        return "\n".join(value)
    raise ExtractionPayloadError(
        f"{key!r} must be a string, null, or a sequence of strings, got "
        f"{type(value).__name__}: {value!r}"
    )


def extraction_from_model_payload(payload: Mapping[str, object]) -> Extraction:
    """Build the status path's `Extraction` from one raw model response object.

    This is the boundary a model response crosses on its way toward
    `derive_status()`, and the only place in the status path that touches an
    untyped mapping. It exists so a payload carrying a status is rejected
    *before* `derive_status()` is called rather than quietly ignored there.

    Only `REFINEMENT_PAYLOAD_KEYS` are read. Other keys are ignored rather
    than rejected — a real extraction pass also returns durable claims, which
    are `palaver.extract.quote_gate`'s business — so the object returned here
    is the ephemeral half of a pass, never a complete one. A caller that also
    needs decisions or resolved questions builds those through that gate; it
    must not read them off this result, which never carries any.

    Args:
        payload: One parsed model response object, e.g. what
            `palaver.extract.client.ModelClient.complete()` returns.

    Returns:
        An `Extraction` carrying only the four ephemeral fields, each either
        a normalized string (possibly empty, meaning "affirmatively nothing")
        or `None` (meaning "this pass had no opinion").

    Raises:
        ModelSuppliedStatusError: The payload carries a status-like key
            (INV-7).
        ExtractionPayloadError: The payload is not a mapping, or a field's
            value is neither a string, `null`, nor a sequence of strings.
    """
    if not isinstance(payload, Mapping):
        raise ExtractionPayloadError(
            f"model payload must be a mapping, got {type(payload).__name__}: {payload!r}"
        )

    for key in payload:
        if _normalized_key(key) in FORBIDDEN_PAYLOAD_KEYS:
            raise ModelSuppliedStatusError(
                f"model response carries a status-like field {key!r}; status is computed "
                f"from deterministic signals and is never model-supplied (INV-7)"
            )

    return Extraction(
        **{key: _payload_text(key, payload.get(key)) for key in REFINEMENT_PAYLOAD_KEYS}
    )


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
