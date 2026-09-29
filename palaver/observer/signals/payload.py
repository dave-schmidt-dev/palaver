"""
The model-payload boundary: forbidden keys, normalization, and Extraction construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from palaver.extract.persist import Extraction


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
