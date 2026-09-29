"""Ordinal-based ordering of a rollout's records."""

from __future__ import annotations

from collections.abc import Sequence


def record_ordinal(record: dict, line_index: int) -> int:
    """Return a record's cursor position: its `ordinal`, or its line index.

    Early rollout files add an `ordinal` to the envelope and later ones do
    not, so neither signal is available everywhere and the fallback is not
    an error path.

    Args:
        record: A decoded rollout record.
        line_index: The record's position in the batch it was read in.

    Returns:
        `record["ordinal"]` when it is a genuine integer, `line_index`
        otherwise. A boolean is rejected explicitly — `isinstance(True, int)`
        is true in Python, and `True` is not an ordinal.
    """
    ordinal = record.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        return line_index
    return ordinal


def order_records(records: Sequence[dict]) -> list[dict]:
    """Return `records` in cursor order: by `ordinal`, or by file order.

    The source's own numbering is believed over arrival order, but only when
    the whole batch carries it. A batch where some records have an `ordinal`
    and some do not has no single coordinate system to sort in — mixing real
    ordinals with fallback line indices would interleave them arbitrarily —
    so that batch keeps file order untouched.

    The sort is stable, so records sharing an ordinal keep their file order
    relative to each other.

    Args:
        records: Decoded rollout records, in file order.

    Returns:
        A new list in cursor order.
    """
    if not all(
        isinstance(record.get("ordinal"), int) and not isinstance(record.get("ordinal"), bool)
        for record in records
    ):
        return list(records)
    return sorted(records, key=lambda record: record_ordinal(record, 0))
