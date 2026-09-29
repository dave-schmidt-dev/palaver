"""
The extraction_from_model_payload boundary: rejecting a model-supplied status and
untrustworthy shapes.
"""

import pytest

from palaver.extract.persist import Extraction
from palaver.observer.signals import (
    ExtractionPayloadError,
    ModelSuppliedStatusError,
    Status,
    derive_status,
    extraction_from_model_payload,
)
from tests._signals_support import _ended_turn


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
