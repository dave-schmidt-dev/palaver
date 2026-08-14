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

**INV-6 — every memory carries at least one evidence anchor, enforced here.**
Task 2.1 deliberately left this unenforced: the anchor shape a floor would
have had to check — a copied `quote` string — was scheduled to be replaced
by task 2.2 (`palaver/memory/evidence.py`) with `start_offset`/`end_offset`
span anchors into `transcript_chunks` or `events`, so enforcing against the
old shape would only have been redone against the new one. That
replacement has landed: `write_memory` now takes `EvidenceAnchor` rows (see
`palaver.memory.evidence`) and raises `ValueError` before issuing any
`INSERT` if called with none. This is a Python-level check, not a database
trigger like INV-5's. A database-layer design was considered and rejected,
not skipped for lack of one: give `memories` a `primary_evidence_id
INTEGER NOT NULL REFERENCES memory_evidence(id)`, written after inserting
that memory's first evidence row, so a caller with no evidence fails
immediately on `NOT NULL`/the FK — no commit deferral needed. Rejected
because it restructures `memory_evidence` away from the 1-many `memory_id`
shape task 2.4's supersession work is built on, and it privileges one
evidence row as "primary" among what can legitimately be several. A second
variant — reserving the memory's id, inserting evidence first with a
`DEFERRABLE INITIALLY DEFERRED` FK back to it — avoids that restructuring,
but only raises at `conn.commit()`, after `write_memory` has already
returned, which cannot satisfy the charter gate test's requirement that
*the write call itself* raises (`test_memory_without_evidence_is_rejected`).
The Python-level check above is deliberately the one design that satisfies
that requirement without the first variant's schema cost.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from palaver.memory.evidence import EvidenceAnchor


def write_memory(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    statement: str,
    origin: str,
    tier: int,
    evidence: Sequence[EvidenceAnchor],
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
            version 4.
        project_id: `projects.id` this memory belongs to.
        statement: The memory's text.
        origin: Free-text description of what produced this memory, e.g.
            `"observer"` or `"main-agent"`.
        tier: Provenance tier, 1 (highest confidence) through 5 (lowest);
            see `palaver.memory.tiers`. Enforced to 1-5 by the schema's own
            CHECK constraint.
        evidence: One or more `EvidenceAnchor` rows to link to this memory.
            Must be non-empty — see INV-6 above.
        session_id: `sessions.id` this memory was produced from, if any.
        supersedes: `memories.id` of the row this memory reclassifies, if
            any. Stored as-is; tier-ordering and uniqueness rules on this
            link are task 2.4's, not enforced here.

    Returns:
        The new row's `memories.id`.

    Raises:
        ValueError: `evidence` is empty (INV-6).
    """
    if not evidence:
        raise ValueError(
            "write_memory requires at least one evidence anchor (INV-6): a memory with no "
            "evidence link is indistinguishable from a fabrication"
        )

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
            INSERT INTO memory_evidence(
                memory_id, transcript_chunk_id, event_id, start_offset, end_offset
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                item.transcript_chunk_id,
                item.event_id,
                item.start_offset,
                item.end_offset,
            ),
        )
    return memory_id
