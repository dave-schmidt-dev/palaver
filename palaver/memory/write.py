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

**INV-6 — every memory carries at least one evidence anchor, enforced in the
database.** This paragraph previously explained why the floor was a
Python-level `ValueError` here and *not* a database rule; that reasoning is
corrected rather than left standing, because schema migration 8 now carries
the rule. The `ValueError` stays — it names the invariant in the caller's
own language, before any SQL runs — but it is no longer the only thing
between a fabricated memory and the disk. The migration's
`memories_requires_evidence` trigger refuses any `memories` insert that no
`memory_evidence` row already names, and a trigger fires whatever process,
language, or `PRAGMA foreign_keys` setting issued the statement.

That inverts this function's write order. The parent row has to be able to
see its children at `INSERT` time, so evidence is written **first**, against
an id reserved for a `memories` row that does not exist yet, and the parent
row is inserted last with that id stated explicitly. Two things make that
legal and safe:

* Migration 8 rebuilds `memory_evidence`'s `memory_id` foreign key as
  `DEFERRABLE INITIALLY DEFERRED`, so a child may name a parent that has not
  been written yet — checked at commit, by which point the parent exists.
  The deferred FK is only what makes child-first *possible*; it is not the
  enforcement, since it would raise at `conn.commit()`, long after this
  function returned an id. Commit-time-only enforcement was rejected for
  exactly that reason, as were a circular `primary_evidence_id` FK and a
  primary-anchor redesign, both of which abandon the one-to-many `memory_id`
  shape supersession and `resolve_evidence` are built on.
* The id is reserved as `MAX(id) + 1` under a write lock this function takes
  if the caller is not already holding one. `memories` is append-only (INV-4)
  and `memories_id_never_reused` blocks reuse, so the maximum only ever
  climbs; the lock is what stops two writers reserving the same id and the
  loser's evidence rows being left pointing at a row it never got to write.
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
            version 8.
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

    # Reserve the id under a write lock. `BEGIN IMMEDIATE` is skipped when the
    # caller already holds a transaction — nesting one is an error, and the
    # transaction they hold serializes this reservation just as well. Callers
    # commonly do hold one: they have just inserted the transcript rows this
    # evidence anchors into.
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    memory_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM memories").fetchone()[0]

    # Child-first: these rows name a `memories` row that does not exist yet,
    # which migration 8's deferred foreign key permits inside a transaction.
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

    # `id` is stated rather than left to SQLite: `memories_requires_evidence`
    # runs BEFORE INSERT, where an unspecified INTEGER PRIMARY KEY reads as -1
    # and could never match the evidence written above.
    conn.execute(
        """
        INSERT INTO memories(id, project_id, session_id, statement, origin, tier, supersedes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, project_id, session_id, statement, origin, tier, supersedes),
    )
    return memory_id
