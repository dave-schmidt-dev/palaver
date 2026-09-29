"""
The tier-4 observer-inference cap: when it holds, when it lifts, and the committed
measurement that must keep holding it today.
"""

import json
from pathlib import Path

import pytest

from palaver.ingest.adapters.codex import (
    MEASUREMENT_PATH,
    REQUIRED_LABELLED_RECORDS,
    CodexTierCapError,
    RoleClassMeasurement,
    cap_codex_tier,
    codex_tier_cap_lifted,
    load_measurement,
    require_codex_tier,
)
from palaver.memory.tiers import (
    TIER_AGENT_CONCLUSION,
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_OBSERVER_SPECULATION,
    TIER_USER_INSTRUCTION,
)


def _write_measurement(path: Path, **fields) -> Path:
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _passing_measurement(tmp_path: Path) -> Path:
    """A synthetic measurement that legitimately lifts the cap.

    This is the positive control the whole cap suite rests on: without it,
    every "the cap holds" assertion below would also pass against a
    `cap_codex_tier` that ignored its measurement entirely.
    """
    return _write_measurement(
        tmp_path / "passing.json",
        n_records=REQUIRED_LABELLED_RECORDS,
        n_errors=0,
        threshold_met=True,
    )


# --- the tier cap -----------------------------------------------------------


def test_decision_is_tier_four_when_no_measurement_record_exists(tmp_path):
    """Done-when: a Codex-sourced decision returns tier-4 with no measurement."""
    absent = tmp_path / "does-not-exist.json"
    assert load_measurement(absent) is None
    assert codex_tier_cap_lifted(measurement_path=absent) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=absent) == (
        TIER_OBSERVER_INFERENCE
    )


def test_cap_remains_against_a_passing_measurement(tmp_path):
    """A diagnostic measurement cannot weaken Codex's permanent safety cap."""
    passing = _passing_measurement(tmp_path)
    assert codex_tier_cap_lifted(measurement_path=passing) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=passing) == (
        TIER_OBSERVER_INFERENCE
    )
    with pytest.raises(CodexTierCapError):
        require_codex_tier(TIER_USER_INSTRUCTION, measurement_path=passing)


@pytest.mark.parametrize(
    "fields,why",
    [
        (
            {"n_records": REQUIRED_LABELLED_RECORDS - 1, "n_errors": 0, "threshold_met": True},
            "sample one record short of the threshold",
        ),
        (
            {"n_records": REQUIRED_LABELLED_RECORDS, "n_errors": 1, "threshold_met": True},
            "one harness record classified as user",
        ),
        (
            {"n_records": REQUIRED_LABELLED_RECORDS, "n_errors": 0, "threshold_met": False},
            "threshold_met false despite passing counts",
        ),
    ],
    ids=["n_records_below_200", "n_errors_above_zero", "threshold_met_false"],
)
def test_each_failing_condition_independently_holds_the_cap(tmp_path, fields, why):
    """Done-when: tier-4 when `n_records` < 200, `n_errors` > 0, or `threshold_met` false.

    Each variant differs from the passing control in exactly one field, so a
    conjunction that dropped a term (`and self.threshold_met` deleted, say)
    fails on precisely the variant that term guards, and the failure names
    which condition stopped being enforced.
    """
    path = _write_measurement(tmp_path / "failing.json", **fields)
    assert codex_tier_cap_lifted(measurement_path=path) is False, why
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=path) == (TIER_OBSERVER_INFERENCE)


def test_requesting_tier_one_for_a_codex_source_raises(tmp_path):
    """Done-when: the tier cap raises when a caller requests tier-1.

    `cap_codex_tier` demotes silently, which is right for a tier the
    pipeline derived. An explicit tier-1 request is a caller asserting "the
    user said this" on the strength of a prefix heuristic, and silently
    handing it a 4 would hide that.
    """
    absent = tmp_path / "does-not-exist.json"
    with pytest.raises(CodexTierCapError) as excinfo:
        require_codex_tier(TIER_USER_INSTRUCTION, measurement_path=absent)
    message = str(excinfo.value)
    assert "tier 1" in message
    assert "user_instruction" in message
    assert "permanent" in message


@pytest.mark.parametrize(
    "tier", [TIER_AGENT_CONCLUSION, TIER_OBSERVED_RESULT], ids=["tier2", "tier3"]
)
def test_every_tier_above_the_cap_is_refused(tmp_path, tier):
    """The cap is not a tier-1 special case.

    Tier-2 ("the main agent concluded this") and tier-3 ("this tool result
    was observed") both rest on reading the transcript correctly, which for
    Codex means the same heuristic channel split.
    """
    absent = tmp_path / "does-not-exist.json"
    assert cap_codex_tier(tier, measurement_path=absent) == TIER_OBSERVER_INFERENCE
    with pytest.raises(CodexTierCapError):
        require_codex_tier(tier, measurement_path=absent)


def test_cap_never_promotes_a_weaker_tier(tmp_path):
    """Tier-5 speculation stays tier-5; the cap is a ceiling, not a floor."""
    absent = tmp_path / "does-not-exist.json"
    assert cap_codex_tier(TIER_OBSERVER_SPECULATION, measurement_path=absent) == (
        TIER_OBSERVER_SPECULATION
    )
    assert require_codex_tier(TIER_OBSERVER_SPECULATION, measurement_path=absent) == (
        TIER_OBSERVER_SPECULATION
    )
    assert require_codex_tier(TIER_OBSERVER_INFERENCE, measurement_path=absent) == (
        TIER_OBSERVER_INFERENCE
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[1, 2, 3]",
        '"a string"',
        "{}",
        '{"n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "threshold_met": true}',
        '{"n_records": 200, "n_errors": 0}',
        '{"n_records": "200", "n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "n_errors": 0, "threshold_met": 1}',
        '{"n_records": true, "n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "n_errors": false, "threshold_met": true}',
        '{"n_records": null, "n_errors": null, "threshold_met": null}',
    ],
    ids=[
        "empty",
        "not_json",
        "json_array",
        "json_string",
        "empty_object",
        "missing_n_records",
        "missing_n_errors",
        "missing_threshold_met",
        "n_records_as_string",
        "threshold_met_as_int",
        "n_records_as_bool",
        "n_errors_as_bool",
        "all_null",
    ],
)
def test_a_malformed_measurement_holds_the_cap(tmp_path, content):
    """Every unreadable measurement fails closed, and none of them lift the cap.

    `threshold_met: 1` and `n_records: true` are in here specifically because
    `isinstance(True, int)` is true in Python: a naive type check would admit
    both and lift a cap on a file that never passed a measurement.
    """
    path = tmp_path / "malformed.json"
    path.write_text(content, encoding="utf-8")
    assert load_measurement(path) is None
    assert codex_tier_cap_lifted(measurement_path=path) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=path) == (TIER_OBSERVER_INFERENCE)


def test_lifts_tier_cap_requires_all_three_conditions():
    """The conjunction, asserted directly on the dataclass.

    A truth table rather than a single happy path, so a term dropped from
    `lifts_tier_cap` cannot pass by coincidence.
    """
    n = REQUIRED_LABELLED_RECORDS
    assert RoleClassMeasurement(n, 0, True).lifts_tier_cap is True
    assert RoleClassMeasurement(n - 1, 0, True).lifts_tier_cap is False
    assert RoleClassMeasurement(n, 1, True).lifts_tier_cap is False
    assert RoleClassMeasurement(n, 0, False).lifts_tier_cap is False
    assert RoleClassMeasurement(0, 0, True).lifts_tier_cap is False


def test_the_committed_measurement_holds_the_cap_today():
    """The shipped state: 19 labelled records is not 200, so Codex stays tier-4.

    This is the honest negative result task 7.1 exists to record. It is not a
    failure of the mechanism — the mechanism is working, and this assertion
    is what proves it is wired to the real file rather than only to
    `tmp_path` fixtures.
    """
    measurement = load_measurement()
    assert measurement is not None, f"{MEASUREMENT_PATH} is missing or malformed"
    assert measurement.lifts_tier_cap is False
    assert codex_tier_cap_lifted() is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION) == TIER_OBSERVER_INFERENCE
    with pytest.raises(CodexTierCapError):
        require_codex_tier(TIER_USER_INSTRUCTION)
