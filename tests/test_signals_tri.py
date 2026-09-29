"""The three-valued Tri type and the Signals dataclass's no-defaults contract."""

import dataclasses

import pytest

from palaver.observer.signals import (
    SIGNAL_NAMES,
    Signals,
    Status,
    Tri,
    derive_status,
)
from tests._signals_support import _signals

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
