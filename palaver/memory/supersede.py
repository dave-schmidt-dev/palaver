"""Supersession: correcting a memory by writing its successor (task 2.4).

Supersession is a **derived view, never a stored flag**. The successor row
carries the entire edge in `memories.supersedes`; the predecessor is never
touched, because marking it `superseded = 1` would mean writing to a row
INV-4 declares immutable. "Is this memory still current?" is therefore
answered by asking whether any row points at it —
`palaver/store/schema.py` migration 5's `superseded_memories` view — not by
reading a column on the memory itself.

Every rule this module names is enforced in the database, by migration 5,
not here (INV-5's "in the database, not in prompt text" applies equally to
"not in Python"):

* at most one successor per predecessor — a partial unique index, plus a
  `BEFORE INSERT` trigger that also holds under `INSERT OR REPLACE`;
* `supersedes` must name an existing memory — a trigger rather than the
  foreign key, because FK enforcement needs `PRAGMA foreign_keys=ON` and a
  connection without it accepted a dangling link (measured);
* a lower-confidence tier may never supersede a higher-confidence one
  (INV-5);
* a superseded row cannot be UPDATEd, no memories row can be DELETEd, and
  neither `INSERT OR REPLACE` nor `UPDATE OR REPLACE` can destroy one
  through a rowid or unique-index collision.

`supersede_memory` below deliberately does **not** re-check those rules in
Python before writing. Duplicating them here would create a second, drifting
statement of the same contract while adding no guarantee: a caller that
opens its own `sqlite3` connection never runs this code, and the database
rejects it anyway. What this module adds is the one thing the database
cannot — inheriting the predecessor's scope so a correction cannot silently
land in a different project than the memory it corrects.

Nothing in this module issues a `DELETE` or `DROP`
(`tests/test_memory.py::test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module`
scans for that across the package).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory


def supersede_memory(
    conn: sqlite3.Connection,
    *,
    predecessor_id: int,
    statement: str,
    origin: str,
    tier: int,
    evidence: Sequence[EvidenceAnchor],
    session_id: int | None = None,
) -> int:
    """Write a successor memory that supersedes `predecessor_id`.

    The predecessor is read only to inherit its scope; it is never written
    to. The successor is an ordinary `write_memory` INSERT carrying
    `supersedes=predecessor_id`, so INV-4's append-only guarantee holds by
    construction rather than by care.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 5.
        predecessor_id: `memories.id` of the row being corrected.
        statement: The corrected statement.
        origin: Free-text description of what produced the correction, e.g.
            `"observer"` or `"user-correction"`.
        tier: Provenance tier of the *successor*, 1 (highest confidence)
            through 5. Must be at least as high-confidence as the
            predecessor's — i.e. numerically less than or equal to it —
            which migration 5's `memories_supersedes_tier_order` trigger
            enforces (INV-5).
        evidence: One or more `EvidenceAnchor` rows for the successor. Must
            be non-empty (INV-6).
        session_id: `sessions.id` to attribute the correction to. Defaults
            to the predecessor's `session_id`, so a correction stays in the
            session whose transcript produced the original unless a caller
            deliberately says otherwise.

    Returns:
        The successor's new `memories.id`.

    Raises:
        LookupError: `predecessor_id` names no row in `memories`. Raised
            before any write, so a typo'd id cannot land a memory in an
            unintended scope; the database would reject the dangling link
            regardless (`memories_supersedes_must_exist`).
        ValueError: `evidence` is empty (INV-6, from `write_memory`).
        sqlite3.IntegrityError: the database refused the supersession —
            `predecessor_id` already has a successor, or `tier` is lower
            confidence than the predecessor's (INV-5).
    """
    row = conn.execute(
        "SELECT project_id, session_id FROM memories WHERE id = ?", (predecessor_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no memory with id {predecessor_id!r} to supersede")
    project_id, predecessor_session_id = row

    return write_memory(
        conn,
        project_id=project_id,
        session_id=predecessor_session_id if session_id is None else session_id,
        statement=statement,
        origin=origin,
        tier=tier,
        evidence=evidence,
        supersedes=predecessor_id,
    )


def is_superseded(conn: sqlite3.Connection, memory_id: int) -> bool:
    """Report whether some other memory supersedes `memory_id`.

    Reads the `superseded_memories` view, which derives the answer from the
    successor's `supersedes` link. There is no stored flag to read, and
    deliberately so — see the module docstring.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 5.
        memory_id: `memories.id` to ask about. An id that names no memory
            is simply not superseded; this is a status question, not a
            lookup, so it does not raise.

    Returns:
        True if a successor row points at `memory_id`.
    """
    row = conn.execute(
        "SELECT 1 FROM superseded_memories WHERE memory_id = ?", (memory_id,)
    ).fetchone()
    return row is not None


def successor_of(conn: sqlite3.Connection, memory_id: int) -> int | None:
    """Return the id of the memory that supersedes `memory_id`, if any.

    At most one row can point at a given predecessor (migration 5's
    `memories_one_successor_per_predecessor` index and its `BEFORE INSERT`
    guard), so this is a single id and never a list.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 5.
        memory_id: `memories.id` whose successor is wanted.

    Returns:
        The successor's `memories.id`, or None if `memory_id` is still
        current (or names no memory at all).
    """
    row = conn.execute("SELECT id FROM memories WHERE supersedes = ?", (memory_id,)).fetchone()
    return None if row is None else row[0]
