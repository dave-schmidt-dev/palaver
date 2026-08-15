"""Ephemeral state versus durable memory: the extraction write boundary (Task 3.4).

This module owns the line the plan draws between two kinds of extracted
field, and it is a line `palaver/replay.py`'s module docstring already names
even though nothing wired it up before this task: **regeneratable
per-session fields — current task, remaining work, blockers now, open
questions — go to an upserted `current_state` row, one per `(project_id,
session_id, key)`. Only decisions and resolved questions ever reach the
append-only `memories` table.**

**Why (INV-4).** `memories` is append-only and nothing under
`palaver/memory/` ever issues a `DELETE`, so writing a regeneratable field
there makes growth unbounded on churn — a session whose "current task"
changes forty times would mint forty permanent rows for a fact that was
true for one tick each. Worse, it would make an ephemeral extraction
artifact permanent while the session that produced it is not: `INVARIANTS.md`
is explicit that "regeneratable current-state summaries are exempt and are
stored separately from durable memories precisely so this invariant can be
absolute." This module is that separation, enforced by routing rather than
by convention: `persist_extraction` never gives an ephemeral field a path
to `write_memory`, and never gives a decision or resolved question a path
to `current_state`.

**Durable claims are grounded, not merely inserted.** A decision or a
resolved question is a claim about what a quote in the transcript shows, so
it goes through `palaver.extract.quote_gate.admit_decision` (task 3.3) —
the same write boundary every other durable memory in this codebase goes
through — rather than calling `write_memory` directly. That gets this
module INV-6's evidence requirement, INV-8's channel-based tiering, and
INV-5's provenance ordering for free, and means a decision whose quote does
not ground raises `QuoteNotGroundedError` before anything is written, same
as everywhere else. This module adds no second, weaker path to `memories`.

**`current_state` is key/value: four fields, four rows, not one.** The
shipped schema is `id, project_id, session_id, key, value, updated_at` with
`UNIQUE (project_id, session_id, key)` — one session legitimately holds one
row *per key*. `upsert_current_state` never packs multiple fields into one
value to make a row count come out at one; `EPHEMERAL_KEYS` names the four
keys this module writes, and `persist_extraction` upserts each field
present in the `Extraction` under its own key, skipping any field left
`None` (a partial extraction pass has no opinion on a field it did not
return, and must not blank out a value a previous pass wrote).

**`session_id IS NULL` is not deduplicated by the UNIQUE constraint, so
`upsert_current_state` does not rely on it.** SQLite treats NULL as
DISTINCT from NULL in a UNIQUE index — two `INSERT`s with the same
`project_id`/`key` and `session_id IS NULL` both succeed under that
constraint alone, which would silently double a project-scoped row every
time one is written. `upsert_current_state` closes this itself, with an
explicit `SELECT ... WHERE session_id IS ?` (the `IS` operator, not `=`,
so the same query correctly matches either a NULL or a concrete id) before
deciding whether to `INSERT` or `UPDATE`. This makes the upsert correct
independent of whichever way the constraint behaves, rather than depending
on a database quirk this module would otherwise have to remember.

**`updated_at` is refreshed on every write, insert or update, deliberately.**
The column carries a wall-clock `DEFAULT`, which only ever fires on
`INSERT` — an `UPDATE` that does not name the column leaves it at the
original insert time, making a row that changed a moment ago read as
stale. `upsert_current_state` always sets `updated_at` to the current
timestamp explicitly, on both branches, so "when was this last true"
stays meaningful across repeated overwrites. `tests/test_extraction.py`
pins this by writing a stale sentinel directly, then asserting the next
upsert replaces it rather than leaving it untouched.

No `on_status` channel (INV-1): every operation here is a handful of
indexed `SELECT`/`INSERT`/`UPDATE` statements and one delegated call to
`admit_decision`, which is itself pure SQL with no network call,
subprocess, model inference, or stall-prone IO to surface progress for.
The caller owns committing the connection, matching every other write
module in this package (`write_memory`, `admit_decision`).

This repository is public. Nothing in this module's docstrings or examples
is derived from a real observed session (INV-9).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field

from palaver.extract.quote_gate import admit_decision

#: The session's current task, in the model's own words. Ephemeral: a new
#: value overwrites the old one in place.
CURRENT_TASK = "current_task"

#: What is left to do in the session, as of the extraction pass that
#: produced it. Ephemeral.
REMAINING_WORK = "remaining_work"

#: What is blocking forward progress right now. Ephemeral. Distinct from a
#: *resolved* question or decision, which is durable — see
#: `Extraction.resolved_questions` below.
BLOCKERS_NOW = "blockers_now"

#: Questions the session has not yet answered. Ephemeral until resolved;
#: once resolved they are no longer regeneratable per-session state and
#: belong in `Extraction.resolved_questions` instead, not here.
OPEN_QUESTIONS = "open_questions"

#: Every key `persist_extraction` may write to `current_state`, in the
#: fixed order it checks them. Exhaustive: nothing else in this module
#: writes that table under any other key.
EPHEMERAL_KEYS: tuple[str, ...] = (CURRENT_TASK, REMAINING_WORK, BLOCKERS_NOW, OPEN_QUESTIONS)

#: Always used in place of the schema's `updated_at` DEFAULT, on both the
#: insert and update branch of `upsert_current_state` — see the module
#: docstring for why the DEFAULT alone is not enough.
_UPDATED_AT_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


@dataclass(frozen=True)
class GroundedClaim:
    """One durable claim awaiting quote-grounding at the `memories` write boundary.

    Mirrors `admit_decision`'s per-claim arguments (task 3.3) rather than
    duplicating its grounding/tiering logic: this module's job is routing a
    claim to the right table, not re-deciding what tier a quote earns.

    Attributes:
        statement: The claim's text — what the model says the quote shows.
            Tier-1 admission requires this to *be* the anchored span; see
            `palaver.extract.quote_gate` for the comparison rule.
        quote: The verbatim text the model claims the transcript contains.
        transcript_chunk_id: `transcript_chunks.id` the claim cites.
        cited_span: `(start, end)` offsets narrowing the citation within
            the chunk. Defaults to the whole chunk.
    """

    statement: str
    quote: str
    transcript_chunk_id: int
    cited_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Extraction:
    """One extraction pass's fields, already split along this task's line.

    The four `Optional[str]` attributes are regeneratable per-session
    state: each, if not `None`, overwrites its own `current_state` row and
    is never written to `memories`. `None` means this pass did not produce
    that field and leaves the corresponding row untouched — a pass that
    returns a resolved question but says nothing about `blockers_now` must
    not blank out a blocker a previous pass recorded.

    `decisions` and `resolved_questions` are the only fields that ever
    reach `memories`, each admitted through `admit_decision` (task 3.3) and
    so subject to the same quote-grounding and channel-tiering rules as
    every other durable memory. Both default to empty: an extraction pass
    that found no durable claim writes nothing to `memories`.
    """

    current_task: str | None = None
    remaining_work: str | None = None
    blockers_now: str | None = None
    open_questions: str | None = None
    decisions: Sequence[GroundedClaim] = field(default_factory=tuple)
    resolved_questions: Sequence[GroundedClaim] = field(default_factory=tuple)


@dataclass(frozen=True)
class PersistResult:
    """What one `persist_extraction` call wrote, by destination table.

    Attributes:
        current_state_keys_written: Which of `EPHEMERAL_KEYS` this call
            upserted, in the order checked. A field left `None` on the
            `Extraction` contributes nothing here.
        decision_memory_ids: `memories.id` for each admitted decision, in
            `extraction.decisions` order.
        resolved_question_memory_ids: `memories.id` for each admitted
            resolved question, in `extraction.resolved_questions` order.
    """

    current_state_keys_written: tuple[str, ...]
    decision_memory_ids: tuple[int, ...]
    resolved_question_memory_ids: tuple[int, ...]


def upsert_current_state(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    session_id: int | None,
    key: str,
    value: str,
) -> None:
    """Write one `current_state` row, updating in place if it already exists.

    Looks the row up with `session_id IS ?` rather than trusting the
    table's `UNIQUE (project_id, session_id, key)` constraint to have
    deduplicated it — SQLite treats NULL as distinct from NULL in a UNIQUE
    index, so two prior inserts with `session_id IS NULL` could both have
    succeeded under that constraint alone. The explicit lookup makes this
    function correct regardless: at most one row for
    `(project_id, session_id, key)` exists after this call, whether
    `session_id` is a concrete id or `None`. `updated_at` is set to the
    current timestamp on every call, insert or update, so a row's staleness
    is always meaningful — see the module docstring.

    Args:
        conn: Open connection to a database carrying the `current_state`
            table.
        project_id: `projects.id` this state belongs to.
        session_id: `sessions.id` this state belongs to, or `None` for
            project-scoped state with no single owning session.
        key: One of `EPHEMERAL_KEYS`, or any other key a future caller
            defines — this function does not restrict `key` itself,
            `persist_extraction` is what limits it to the four ephemeral
            fields this task routes.
        value: The new value. Must be non-`None`; the schema's own
            `NOT NULL` on `current_state.value` would reject `None` anyway,
            but callers should not rely on that surfacing a useful message.
    """
    existing = conn.execute(
        "SELECT id FROM current_state WHERE project_id = ? AND session_id IS ? AND key = ?",
        (project_id, session_id, key),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO current_state(project_id, session_id, key, value, updated_at) "
            f"VALUES (?, ?, ?, ?, {_UPDATED_AT_NOW})",
            (project_id, session_id, key, value),
        )
    else:
        (row_id,) = existing
        conn.execute(
            f"UPDATE current_state SET value = ?, updated_at = {_UPDATED_AT_NOW} WHERE id = ?",
            (value, row_id),
        )


def _admit_claim(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    session_id: int | None,
    claim: GroundedClaim,
    origin: str,
) -> int:
    """Ground and write one durable claim, returning its new `memories.id`."""
    admitted = admit_decision(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement=claim.statement,
        quote=claim.quote,
        transcript_chunk_id=claim.transcript_chunk_id,
        origin=origin,
        cited_span=claim.cited_span,
    )
    return admitted.memory_id


def persist_extraction(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    session_id: int | None,
    extraction: Extraction,
    origin: str = "observer-extraction",
) -> PersistResult:
    """Route one extraction pass's fields to `current_state` or `memories`.

    Regeneratable fields (`current_task`, `remaining_work`, `blockers_now`,
    `open_questions`) are upserted into `current_state`, one row per key,
    via `upsert_current_state`. Decisions and resolved questions are each
    admitted into `memories` via `admit_decision` (task 3.3), grounded
    against their cited quote and tiered by channel. Nothing here writes an
    ephemeral field to `memories` or a durable claim to `current_state`.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 4. The caller owns committing it.
        project_id: `projects.id` this extraction belongs to.
        session_id: `sessions.id` this extraction was produced from, or
            `None` for extraction not tied to one session.
        extraction: The pass's fields, already split along this task's
            line — see `Extraction`.
        origin: Free-text origin recorded on any `memories` row this call
            writes, suffixed with `:decision` or `:resolved_question` so
            the two kinds of durable claim stay distinguishable in
            `memories.origin`.

    Returns:
        A `PersistResult` naming exactly what was written where.

    Raises:
        QuoteNotGroundedError: A decision's or resolved question's quote
            does not ground in its cited span (INV-6). Raised by
            `admit_decision`, before that claim's row is written; claims
            processed earlier in the same call remain committed-pending on
            `conn` as normal — this function does not wrap them in its own
            transaction.
    """
    keys_written = []
    for key, value in (
        (CURRENT_TASK, extraction.current_task),
        (REMAINING_WORK, extraction.remaining_work),
        (BLOCKERS_NOW, extraction.blockers_now),
        (OPEN_QUESTIONS, extraction.open_questions),
    ):
        if value is None:
            continue
        upsert_current_state(
            conn, project_id=project_id, session_id=session_id, key=key, value=value
        )
        keys_written.append(key)

    decision_ids = tuple(
        _admit_claim(
            conn,
            project_id=project_id,
            session_id=session_id,
            claim=claim,
            origin=f"{origin}:decision",
        )
        for claim in extraction.decisions
    )
    resolved_question_ids = tuple(
        _admit_claim(
            conn,
            project_id=project_id,
            session_id=session_id,
            claim=claim,
            origin=f"{origin}:resolved_question",
        )
        for claim in extraction.resolved_questions
    )

    return PersistResult(
        current_state_keys_written=tuple(keys_written),
        decision_memory_ids=decision_ids,
        resolved_question_memory_ids=resolved_question_ids,
    )
