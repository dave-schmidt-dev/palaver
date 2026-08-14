"""Append-only memory writer: `memories` rows plus their `memory_evidence` links.

This module carries task 2.1's half of two invariants (`INVARIANTS.md`):

**INV-4 — no DELETE path.** Nothing in this module, or anywhere under
`palaver/memory/`, issues a `DELETE` or `DROP`. `write_memory` is the only
way this module puts a row into `memories` or `memory_evidence`, and it
always runs `INSERT`. A reclassification — the observer deciding an
existing memory's provenance was wrong — is not a mutation of the existing
row; it is a second `write_memory` call whose `supersedes` argument names
the row being reclassified, leaving the predecessor byte-identical. Full
supersession semantics — the `UNIQUE` constraint on `supersedes`, the
`superseded_memories` view, and the tier-ordering rule that a
lower-confidence tier can never supersede a higher one — are task 2.4's;
`write_memory` only accepts and stores the `supersedes` link so that later
enforcement has a column to sit on top of.

**INV-5 — tier is immutable, enforced at the database layer.** `tier` is
assigned once, at insert, and this module never issues an `UPDATE` that
touches it. That is necessary but not sufficient: a Python-only rule is
bypassed by the next piece of code that opens its own connection to the same
database file. The actual enforcement is `palaver/store/schema.py`
migration 3, `memories_tier_immutable` — a `BEFORE UPDATE OF tier` trigger
that raises regardless of which process or language issues the `UPDATE`.
`tests/test_memory.py::test_update_tier_raises_at_the_database_layer`
attacks that trigger with a raw `sqlite3` connection, never calling into
this module, to prove the guarantee does not depend on going through
`write_memory` at all.

**INV-6 is deliberately not enforced here.** `write_memory` accepts and
stores whatever `EvidenceInput` rows a caller passes, including zero. The
anchor shape those rows carry today — a copied `quote` string — is exactly
what task 2.2 (`palaver/memory/evidence.py`) replaces with span offsets into
`transcript_chunks`; enforcing an evidence floor here now would have to be
redone against a different schema in that task anyway, so it is left to it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceInput:
    """One `memory_evidence` row to write alongside a memory.

    Attributes:
        quote: The evidence text, stored verbatim in `memory_evidence.quote`.
            Substring-verification against the cited source is task 2.2's
            job, not this module's.
        transcript_chunk_id: Row id in `transcript_chunks` this evidence
            anchors to, if any.
        event_id: Row id in `events` this evidence anchors to, if any.

    At least one of `transcript_chunk_id`/`event_id` must be set — the
    schema's own CHECK constraint on `memory_evidence` rejects a row with
    neither, independent of anything this dataclass does.
    """

    quote: str
    transcript_chunk_id: int | None = None
    event_id: int | None = None


def write_memory(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    statement: str,
    origin: str,
    tier: int,
    evidence: Sequence[EvidenceInput] = (),
    session_id: int | None = None,
    supersedes: int | None = None,
) -> int:
    """Insert a new `memories` row and its `memory_evidence` rows.

    Always an `INSERT`, never an `UPDATE` or `DELETE` — see the module
    docstring for how that keeps INV-4 and INV-5. Passing `supersedes` does
    not mutate the row it names; it only records, on this new row, which
    earlier row it reclassifies. The caller owns committing the connection.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 1.
        project_id: `projects.id` this memory belongs to.
        statement: The memory's text.
        origin: Free-text description of what produced this memory, e.g.
            `"observer"` or `"main-agent"`.
        tier: Provenance tier, 1 (highest confidence) through 5 (lowest);
            see `palaver.memory.tiers`. Enforced to 1-5 by the schema's own
            CHECK constraint.
        evidence: Zero or more `EvidenceInput` rows to link to this memory.
        session_id: `sessions.id` this memory was produced from, if any.
        supersedes: `memories.id` of the row this memory reclassifies, if
            any. Stored as-is; tier-ordering and uniqueness rules on this
            link are task 2.4's, not enforced here.

    Returns:
        The new row's `memories.id`.
    """
    cursor = conn.execute(
        """
        INSERT INTO memories(project_id, session_id, statement, origin, tier, supersedes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project_id, session_id, statement, origin, tier, supersedes),
    )
    memory_id = cursor.lastrowid
    for item in evidence:
        conn.execute(
            """
            INSERT INTO memory_evidence(memory_id, transcript_chunk_id, event_id, quote)
            VALUES (?, ?, ?, ?)
            """,
            (memory_id, item.transcript_chunk_id, item.event_id, item.quote),
        )
    return memory_id
