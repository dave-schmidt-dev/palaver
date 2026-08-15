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

import ast
import dataclasses
import inspect
import itertools
import json
import pathlib

import pytest

from palaver.extract.client import ModelClientError, ModelTimeoutError
from palaver.extract.persist import Extraction
from palaver.observer import signals as signals_module
from palaver.observer.signals import (
    FORBIDDEN_PAYLOAD_KEYS,
    PHASE1_STATUS_RANGE,
    REFINED_STATUS_RANGE,
    SIGNAL_NAMES,
    ExtractionPayloadError,
    ModelSuppliedStatusError,
    Signals,
    Status,
    Tri,
    derive_status,
    extraction_from_model_payload,
)
from palaver.observer.turn_boundary import derive_signals

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

#: The unrefined range, written out as a literal rather than imported, so this
#: module asserts the contract independently of the constant it is checking.
#: A single edit to `PHASE1_STATUS_RANGE` cannot move both sides at once.
EXPECTED_PHASE1_RANGE = {
    Status.WORKING,
    Status.AWAITING_HUMAN,
    Status.ERROR,
    Status.UNKNOWN,
}

#: The range once an `Extraction` is supplied (task 3.6), written out the same
#: way and for the same reason.
EXPECTED_REFINED_RANGE = EXPECTED_PHASE1_RANGE | {
    Status.DONE,
    Status.WAITING_FOR_USER,
    Status.QUESTION,
    Status.BLOCKED,
}

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

#: Statuses unreachable even *with* an extraction: `IDLE` needs process
#: liveness (Phase 5.2), which no input to this module carries.
EXPECTED_DEFERRED_WITH_EXTRACTION = {Status.IDLE}

#: Import roots that would put a model call or a socket inside the status
#: path. `derive_status()` reaching any of these would breach INV-7 (the
#: model never sets status) or INV-9 (content never leaves this machine).
BANNED_IMPORT_ROOTS = frozenset(
    {"httpx", "requests", "urllib", "openai", "aiohttp", "socket", "http", "llama_cpp"}
)


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


def _all_signal_combinations() -> list[Signals]:
    """Enumerate the entire signal space: every `Tri` value of every signal."""
    return [
        Signals(**dict(zip(SIGNAL_NAMES, combination, strict=True)))
        for combination in itertools.product(Tri, repeat=len(SIGNAL_NAMES))
    ]


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


def _imported_modules(source: str) -> set[str]:
    """Return the full dotted module name of every import in `source`."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _imported_roots(source: str) -> set[str]:
    """Return the top-level module name of every import in `source`."""
    return {module.split(".")[0] for module in _imported_modules(source)}


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


# --- the three-valued signal type -------------------------------------------


def test_tri_refuses_boolean_coercion_so_unknown_cannot_collapse_to_false():
    """Every `Tri` member raises on boolean coercion, including `Tri.TRUE`.

    Enum members are truthy by default, so `if signal:` would read
    `Tri.FALSE` and `Tri.UNKNOWN` as true and `if not signal:` would read all
    three as false. Both are the collapse INV-7's rationale warns about and
    both are invisible at review. `Tri.TRUE` is included in the loop
    deliberately: an implementation that raised only for `UNKNOWN` would
    still let `if signal:` compile into a silent two-valued read.
    """
    for member in Tri:
        with pytest.raises(TypeError, match="three-valued"):
            bool(member)
        with pytest.raises(TypeError, match="three-valued"):
            if member:  # the implicit coercion is the thing under test
                pass

    # Positive control: identity comparison, the supported test, still works.
    assert Tri.UNKNOWN is Tri.UNKNOWN
    assert Tri.TRUE is not Tri.FALSE


def test_tri_from_optional_maps_none_to_unknown_not_false():
    """A producer that models absence as `None` lifts into `UNKNOWN`, never
    `FALSE` — the conversion boundary is where a three-valued signal is most
    likely to be silently flattened."""
    assert Tri.from_optional(None) is Tri.UNKNOWN
    assert Tri.from_optional(True) is Tri.TRUE
    assert Tri.from_optional(False) is Tri.FALSE


def test_signals_rejects_a_raw_bool_instead_of_silently_deriving_unknown():
    """A raw `bool` passed where a `Tri` belongs raises at construction.

    Without the check this fails silently and plausibly: `True is Tri.TRUE`
    is `False`, so every rule would skip and the caller would receive a
    confident-looking `UNKNOWN` for a session whose signals were fully
    determined. The valid construction below is the positive control.
    """
    with pytest.raises(TypeError, match="source_readable"):
        Signals(
            source_readable=True,
            signal_records_parsed=Tri.TRUE,
            unresolved_tool_error=Tri.FALSE,
            agent_turn_ended=Tri.TRUE,
        )

    with pytest.raises(TypeError, match="agent_turn_ended"):
        Signals(
            source_readable=Tri.TRUE,
            signal_records_parsed=Tri.TRUE,
            unresolved_tool_error=Tri.FALSE,
            agent_turn_ended=None,
        )

    assert derive_status(_signals()) is Status.WORKING


def test_signals_has_no_defaults_so_an_omitted_signal_cannot_pass_silently():
    """Every signal is required: a caller that forgets one gets a `TypeError`,
    not a default that quietly asserts something about a signal nobody
    measured."""
    with pytest.raises(TypeError, match="required positional argument"):
        Signals(source_readable=Tri.TRUE)  # type: ignore[call-arg]

    declared = dataclasses.fields(Signals)

    assert len(declared) == len(SIGNAL_NAMES) == 4  # not vacuous: fields were found
    assert all(field.default is dataclasses.MISSING for field in declared)
    assert all(field.default_factory is dataclasses.MISSING for field in declared)


def test_signal_names_matches_the_dataclass_fields_in_order():
    """`SIGNAL_NAMES` is the per-signal coverage contract task 1.6's
    `palaver diagnose --coverage` reports against, so it must track `Signals`
    exactly — a name that drifts would silently drop a signal from the
    coverage report."""
    assert SIGNAL_NAMES == (
        "source_readable",
        "signal_records_parsed",
        "unresolved_tool_error",
        "agent_turn_ended",
    )
    assert len(SIGNAL_NAMES) == len(set(SIGNAL_NAMES))
    # Constructible by name — proves these are the real field names.
    assert isinstance(Signals(**dict.fromkeys(SIGNAL_NAMES, Tri.UNKNOWN)), Signals)


# --- INV-7 / INV-9: nothing in the status path can reach a model or a socket --


def test_signals_module_imports_no_network_or_model_client():
    """The status path imports nothing that could reach a model or a socket.

    INV-7 (the model never sets status) and INV-9 (content never leaves this
    machine) both fail the moment a client library appears in this import
    list. The exact-set assertion is the tripwire: any new import here has to
    be a deliberate decision reviewed against both invariants, not a quiet
    addition.

    Asserted over full dotted module paths, not top-level roots. Task 3.6
    imports `Extraction` from `palaver.extract.persist`, and a root-set
    assertion would from then on admit the whole of `palaver` — including
    `palaver.extract.client`, the one module in this tree that opens a
    socket. Naming the exact module keeps the tripwire as tight as it was
    before the import existed.
    """
    source = pathlib.Path(signals_module.__file__).read_text(encoding="utf-8")
    modules = _imported_modules(source)

    assert modules == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "palaver.extract.persist",
    }
    assert "palaver.extract.client" not in modules
    assert not _imported_roots(source) & BANNED_IMPORT_ROOTS


def test_banned_import_detector_is_not_inert():
    """Positive control for the test above: the same extraction, run over source
    that really does import a network client, flags it.

    Without this, an `_imported_roots` that returned an empty set for every
    input would make the INV-7/INV-9 check pass unconditionally. The
    module-path extraction is controlled the same way, since the tripwire
    above now depends on it distinguishing `palaver.extract.persist` from
    `palaver.extract.client`.
    """
    source = "import httpx\nfrom urllib import request\nimport json\n"
    roots = _imported_roots(source)

    assert roots == {"httpx", "urllib", "json"}
    assert roots & BANNED_IMPORT_ROOTS == {"httpx", "urllib"}

    assert _imported_modules("from palaver.extract.client import ModelClient\n") == {
        "palaver.extract.client"
    }


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


def test_refinement_rejects_a_model_payload_carrying_a_status_key():
    """INV-7's tripwire: a model response that answers with a status is refused
    at the boundary and never reaches `derive_status()`.

    Spike run 1 measured a 4B model getting `status` wrong on exactly the
    sessions that mattered, and spike run 2 found the reason — it ignores rules
    whose predicates are its own generated fields. So a payload carrying one is
    a prompt regression, and it fails loudly here rather than being quietly
    dropped. Synonyms and casing are folded because a prompt that started
    asking for `session_state` would otherwise slip through a literal check.

    The positive control is the same payload with the status key removed: it
    parses, and its fields reach the rule list, so the raise above cannot come
    from a boundary that rejects everything.
    """
    for key in ("status", "Status", " STATUS ", "state", "session-state", "agent status"):
        with pytest.raises(ModelSuppliedStatusError, match="INV-7"):
            extraction_from_model_payload({key: "DONE", "remaining_work": ""})

    # Positive control: identical payload, status key removed.
    extraction = extraction_from_model_payload({"remaining_work": "", "current_task": "migrating"})

    assert extraction == Extraction(remaining_work="", current_task="migrating")
    assert derive_status(_ended_turn(), extraction=extraction) is Status.DONE

    # And the error is a `ValueError`, so a caller degrading on bad extraction
    # catches it with everything else the boundary raises.
    assert issubclass(ModelSuppliedStatusError, ExtractionPayloadError)
    assert issubclass(ExtractionPayloadError, ValueError)


def test_refinement_input_must_be_an_extraction_not_a_raw_payload():
    """`derive_status()` refuses a raw mapping outright.

    This is the second half of the guard above and the reason a status key
    "cannot reach `derive_status()`" is a property rather than a convention:
    even a caller that skipped the boundary function entirely gets a
    `TypeError`, because the only accepted refinement input is a typed
    `Extraction` that has no status field to read. Without the check the dict
    would sail through — `getattr` is never used, so every rule would simply
    find nothing and return a plausible AWAITING_HUMAN.
    """
    signals = _ended_turn()
    payload = {"status": "DONE", "remaining_work": ""}

    with pytest.raises(TypeError, match="Extraction"):
        derive_status(signals, extraction=payload)

    with pytest.raises(TypeError, match="Extraction"):
        derive_status(signals, extraction="DONE")

    # Positive control: the same content, correctly converted, is accepted.
    assert derive_status(signals, extraction=Extraction(remaining_work="")) is Status.DONE
    assert derive_status(signals, extraction=None) is Status.AWAITING_HUMAN


def test_refinement_payload_joins_sequences_rather_than_letting_a_list_reach_a_rule():
    """The brief's own state JSON models `remaining_work` as an array, so the
    boundary accepts one — and joins it, never passes it through.

    `bool([""])` is `True`, so a list of empty strings reaching a rule would
    read as outstanding work; a list is also not what `Extraction` declares.
    The empty-list case is the one that matters: an extractor reporting "no
    remaining items" must reach DONE, not AWAITING_HUMAN.
    """
    assert extraction_from_model_payload({"remaining_work": []}).remaining_work == ""
    assert extraction_from_model_payload({"remaining_work": [""]}).remaining_work == ""
    assert (
        extraction_from_model_payload(
            {"remaining_work": ["fix the failing test", "run the suite"]}
        ).remaining_work
        == "fix the failing test\nrun the suite"
    )

    for payload, expected in (
        ({"remaining_work": []}, Status.DONE),
        ({"remaining_work": [""]}, Status.DONE),
        ({"remaining_work": ["fix the failing test"]}, Status.WAITING_FOR_USER),
        ({"blockers_now": ["no credentials"], "remaining_work": []}, Status.BLOCKED),
    ):
        extraction = extraction_from_model_payload(payload)
        assert derive_status(_ended_turn(), extraction=extraction) is expected


def test_refinement_payload_rejects_untrustworthy_shapes_and_ignores_unread_keys():
    """The boundary is fail-loud on a value it cannot honestly normalize, and
    silent about keys it does not read.

    A number or a nested object in `remaining_work` means the response did not
    match the requested schema, and coercing it with `str()` would invent a
    claim; that raises. Keys the status path does not read — `decisions` and
    `resolved_questions`, which belong to the quote-grounding gate — are
    ignored rather than rejected, because a real extraction pass carries them
    and this function is only the ephemeral half of one. The returned object
    is asserted to carry no durable claim, so "ignored" is checked rather than
    assumed.
    """
    for bad in ({"remaining_work": 3}, {"remaining_work": {"a": 1}}, {"blockers_now": [1, 2]}):
        with pytest.raises(ExtractionPayloadError):
            extraction_from_model_payload(bad)

    with pytest.raises(ExtractionPayloadError, match="mapping"):
        extraction_from_model_payload(["remaining_work"])

    extraction = extraction_from_model_payload(
        {
            "remaining_work": "",
            "decisions": [{"statement": "use sqlite", "quote": "use sqlite"}],
            "confidence": 0.92,
        }
    )

    assert extraction.decisions == ()
    assert extraction.resolved_questions == ()
    assert extraction.remaining_work == ""
    assert derive_status(_ended_turn(), extraction=extraction) is Status.DONE

    # Positive control: `None` survives as `None` and is not coerced to `""`,
    # which is the distinction the DONE rule rests on.
    assert extraction_from_model_payload({"remaining_work": None}).remaining_work is None
    assert extraction_from_model_payload({}).remaining_work is None
    assert derive_status(_ended_turn(), extraction=extraction_from_model_payload({})) is (
        Status.AWAITING_HUMAN
    )
