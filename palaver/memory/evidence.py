"""Evidence anchors: pointers into stored raw transcript, never copied text.

This module carries task 2.2's half of one invariant (`INVARIANTS.md`):

**INV-6 — every memory carries at least one evidence link to stored raw
transcript.** Task 2.1 deliberately left this unenforced (see
`palaver/memory/write.py`'s module docstring) because the anchor shape it
would have enforced against — a copied `quote` string on `memory_evidence`
— was itself scheduled to be replaced here. That replacement is the core of
this module: `EvidenceAnchor` names a `(transcript_chunk_id | event_id,
start_offset, end_offset)` span into `transcript_chunks.content` or
`events.payload` rather than a string copied out of either at write time, so
a quote cannot silently drift from its source between when a memory is
written and when it is read back. `palaver/store/schema.py` migration 4
rebuilds `memory_evidence` to have no column capable of holding a copied
quote at all — `start_offset`/`end_offset` INTEGER columns replace the old
`quote TEXT NOT NULL` column outright — so "no copied quote" is a schema
fact, not a convention `write_memory` happens to follow.

`resolve_evidence` is the retrieval half: given a `memory_evidence.id`, it
re-reads the *current* content of the row's referenced source and slices it
by the stored offsets, live, on every call. It never caches or trusts a
previously-resolved string. If the source has since been truncated short
enough that the stored offsets no longer fit, it raises `EvidenceAnchorError`
rather than silently returning a shortened, and therefore wrong, span.

INV-6's "at least one evidence link" half is enforced in
`palaver.memory.write.write_memory`, which raises `ValueError` before
issuing any `INSERT` if called with no evidence anchors — a Python-level
check, not a database trigger. `palaver/memory/write.py`'s module
docstring explains why a database-layer design was rejected rather than
just not attempted: a `memories.primary_evidence_id` FK column would
reject immediately, with no commit deferral, but it restructures
`memory_evidence` away from the 1-many `memory_id` shape task 2.4's
supersession work depends on, and privileges one evidence row as
"primary" among what can legitimately be several. A deferred-FK variant
avoids that restructuring but only raises at `conn.commit()`, after
`write_memory` has already returned — which cannot satisfy the charter
gate test's requirement that *the write call itself*
(`test_memory_without_evidence_is_rejected`) raises.

This repository is public. Nothing in this module's docstrings or examples
is derived from a real observed session.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class EvidenceAnchorError(Exception):
    """Raised when a stored evidence anchor can no longer be resolved to text.

    Covers both "the referenced source row no longer exists" and "the
    referenced source row exists but is now shorter than the stored
    `end_offset`" — the truncation case. Deliberately not a subclass of
    `sqlite3.Error`: this is a data-integrity failure this module detects
    by comparing an offset to a length, not a SQL error the driver raised.
    """


@dataclass(frozen=True)
class EvidenceAnchor:
    """One `memory_evidence` row to write alongside a memory.

    Attributes:
        start_offset: Start of the evidence span, inclusive, as a Python
            slice index into the referenced source's text column
            (`transcript_chunks.content` or `events.payload`).
        end_offset: End of the evidence span, exclusive. Must exceed
            `start_offset`; the schema's own CHECK constraints on
            `memory_evidence` reject a row that violates that, or a
            negative `start_offset`, independent of anything this
            dataclass does.
        transcript_chunk_id: Row id in `transcript_chunks` this evidence
            anchors to, if any.
        event_id: Row id in `events` this evidence anchors to, if any.

    At least one of `transcript_chunk_id`/`event_id` must be set — the
    schema's CHECK constraint on `memory_evidence` rejects a row with
    neither, independent of anything this dataclass does.

    Note what this dataclass does *not* do: it does not read the database
    to validate the offsets against the source's current length at
    construction time. That validation happens live, on every read, in
    `resolve_evidence` below — never once at write time and then trusted
    forever after.
    """

    start_offset: int
    end_offset: int
    transcript_chunk_id: int | None = None
    event_id: int | None = None


def resolve_evidence(conn: sqlite3.Connection, evidence_id: int) -> str:
    """Resolve a stored `memory_evidence` row back to its source substring.

    Re-reads the *current* content of the referenced `transcript_chunks` or
    `events` row on every call and slices it by the stored offsets — it
    never returns a value cached at write time. This is what makes a quote
    a live pointer rather than a copy: if the source is later edited or
    truncated, the next resolution reflects that, and raises rather than
    silently returning a shortened span.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 4.
        evidence_id: `memory_evidence.id` to resolve.

    Returns:
        The exact substring `source_content[start_offset:end_offset]` from
        the row's current source content.

    Raises:
        EvidenceAnchorError: `evidence_id` does not name a `memory_evidence`
            row, the row's referenced `transcript_chunks`/`events` row no
            longer exists, or the source's current content is shorter than
            `end_offset` (the source was truncated since the anchor was
            written).
    """
    anchor_row = conn.execute(
        "SELECT transcript_chunk_id, event_id, start_offset, end_offset "
        "FROM memory_evidence WHERE id = ?",
        (evidence_id,),
    ).fetchone()
    if anchor_row is None:
        raise EvidenceAnchorError(f"no memory_evidence row with id {evidence_id}")

    transcript_chunk_id, event_id, start_offset, end_offset = anchor_row

    if transcript_chunk_id is not None:
        source_table, id_column, text_column, source_id = (
            "transcript_chunks",
            "id",
            "content",
            transcript_chunk_id,
        )
    elif event_id is not None:
        source_table, id_column, text_column, source_id = "events", "id", "payload", event_id
    else:
        # Unreachable given the schema's own CHECK constraint (migration 1),
        # which rejects a memory_evidence row with neither id set — kept as
        # an explicit raise rather than an assertion so a future schema
        # change that weakens that CHECK fails loudly here too.
        raise EvidenceAnchorError(
            f"memory_evidence row {evidence_id} has neither transcript_chunk_id nor event_id set"
        )

    source_row = conn.execute(
        f"SELECT {text_column} FROM {source_table} WHERE {id_column} = ?",
        (source_id,),
    ).fetchone()
    if source_row is None:
        raise EvidenceAnchorError(
            f"evidence {evidence_id} anchors to {source_table}.{id_column} = {source_id}, "
            "which no longer exists"
        )

    content = source_row[0]
    if end_offset > len(content):
        raise EvidenceAnchorError(
            f"evidence {evidence_id} anchor [{start_offset}:{end_offset}] exceeds the current "
            f"{source_table} content length {len(content)} — the source was likely truncated "
            "since this anchor was written"
        )

    return content[start_offset:end_offset]
