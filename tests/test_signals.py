"""Tests for the deterministic signal set and the ordered status rule list.

These tests read no session store and construct no fixture files: `Signals`
is a plain value and `derive_status()` is a pure function of it, so every
case here is built in memory (INV-3 is satisfied trivially rather than by
convention).

`tests/test_signals.py::test_status_is_never_model_supplied` is INV-7's gate
test, named as such in `INVARIANTS.md`.
"""

import ast
import dataclasses
import inspect
import itertools
import pathlib

import pytest

from palaver.observer import signals as signals_module
from palaver.observer.signals import (
    PHASE1_STATUS_RANGE,
    SIGNAL_NAMES,
    Signals,
    Status,
    Tri,
    derive_status,
)

#: The Phase 1 range, written out as a literal rather than imported, so this
#: module asserts the contract independently of the constant it is checking.
#: A single edit to `PHASE1_STATUS_RANGE` cannot move both sides at once.
EXPECTED_PHASE1_RANGE = {
    Status.WORKING,
    Status.AWAITING_HUMAN,
    Status.ERROR,
    Status.UNKNOWN,
}

#: Statuses the plan's §4.2 staging table defers to Phase 3.6 (semantic
#: extraction) and Phase 5.2 (process liveness).
EXPECTED_DEFERRED_STATUSES = {
    Status.DONE,
    Status.WAITING_FOR_USER,
    Status.QUESTION,
    Status.BLOCKED,
    Status.IDLE,
}

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


def _all_signal_combinations() -> list[Signals]:
    """Enumerate the entire signal space: every `Tri` value of every signal."""
    return [
        Signals(**dict(zip(SIGNAL_NAMES, combination, strict=True)))
        for combination in itertools.product(Tri, repeat=len(SIGNAL_NAMES))
    ]


def _imported_roots(source: str) -> set[str]:
    """Return the top-level module name of every import in `source`."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


# --- the Phase 1 range, proved by exhausting the signal space ----------------


def test_phase1_status_range():
    """Phase 1 returns exactly {WORKING, AWAITING_HUMAN, ERROR, UNKNOWN}, and the
    five deferred statuses are unreachable — established by evaluating every
    combination of every signal's full value domain, not by reading the source.

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
    # that the whole signal space cannot produce.
    assert returned.isdisjoint(EXPECTED_DEFERRED_STATUSES)
    assert set(Status) - returned == EXPECTED_DEFERRED_STATUSES


def test_status_is_never_model_supplied():
    """INV-7 gate: `derive_status()` accepts no model-supplied field.

    Passing `remaining_work` — the spike's model-supplied status input, and
    the parameter task 3.6 adds only once a model exists — raises `TypeError`
    naming that argument. The signature is inspected directly for a
    `**kwargs` that would swallow it and make this test pass for the wrong
    reason, and the baseline call is exercised as a positive control so a
    `derive_status()` that raised `TypeError` unconditionally could not pass.

    The signal set is built *outside* the `pytest.raises` block on purpose:
    `Signals.__post_init__` also raises `TypeError`, so constructing it
    inside would let a construction failure satisfy the assertion.
    """
    signals = _signals()

    assert isinstance(derive_status(signals), Status)  # positive control

    with pytest.raises(TypeError, match="remaining_work"):
        derive_status(signals, remaining_work=["finish the migration"])

    signature = inspect.signature(derive_status)
    parameters = signature.parameters

    assert "remaining_work" not in parameters
    assert "blockers_now" not in parameters
    assert "status" not in parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()), (
        "a **kwargs would silently swallow remaining_work"
    )
    assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values())
    assert list(parameters) == ["signals"]


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
    assert status not in EXPECTED_DEFERRED_STATUSES

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
    """
    source = pathlib.Path(signals_module.__file__).read_text(encoding="utf-8")
    roots = _imported_roots(source)

    assert roots == {"__future__", "dataclasses", "enum"}
    assert not roots & BANNED_IMPORT_ROOTS


def test_banned_import_detector_is_not_inert():
    """Positive control for the test above: the same extraction, run over source
    that really does import a network client, flags it.

    Without this, an `_imported_roots` that returned an empty set for every
    input would make the INV-7/INV-9 check pass unconditionally.
    """
    roots = _imported_roots("import httpx\nfrom urllib import request\nimport json\n")

    assert roots == {"httpx", "urllib", "json"}
    assert roots & BANNED_IMPORT_ROOTS == {"httpx", "urllib"}
