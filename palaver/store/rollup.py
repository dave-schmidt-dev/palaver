"""Pruning regeneratable state without ever making a durable memory unresolvable.

Palaver's store splits along one line (task 3.4): `memories` is durable and
append-only, `current_state` is regeneratable and expected to be overwritten.
This module acts on that split from the other end — deleting the regeneratable
half once it has aged out, so a long-running store does not grow without bound
on rows nothing will ever read again.

**Why this is not an age query.** The obvious retention rule — delete
everything older than N days — is wrong here, and wrong in a way that shows up
much later than the change that caused it. Task 2.2 anchored evidence as a
`(chunk, start_offset, end_offset)` triple, and `resolve_evidence` re-reads the
chunk's *current* content on every call, raising `EvidenceAnchorError` when the
row is gone or has been shortened past the stored offsets. It never caches. So
deleting an old `transcript_chunks` row that some `memory_evidence` row names
does not delete a memory — it converts that memory into one that raises when
read, permanently, while `memories` still reports it as present. INV-6 says
every memory carries at least one evidence link to stored raw transcript; an
age-only prune keeps the link and destroys the transcript.

`prune_ephemeral` therefore excludes any chunk named by *any* `memory_evidence`
row, without regard to the state of the memory on the other end. In particular
it does not consult `superseded_memories`: supersession is derived, not stored
(INV-4), a superseded memory is still a memory, and `resolve_evidence` still
raises for it if its chunk is gone. "Live" in the exclusion means "some
evidence row points here," nothing narrower.

**What this module does not touch, deliberately.**

* `memories` — append-only under INV-4, and the schema's `memories_no_delete`
  trigger aborts a `DELETE` regardless of what this module intends. Nothing
  here issues one.
* `events` — `memory_evidence.CHECK` allows an anchor to be a
  `transcript_chunks` row *or* an `events` row, and `resolve_evidence` slices
  `events.payload` by the same offsets. Events are safe here only because this
  module never deletes from that table. A later change that reaches for
  `events` under "prune what is regeneratable" needs the same `NOT EXISTS`
  exclusion `transcript_chunks` gets below; it does not inherit one.
* `memory_evidence` — deleting an anchor row would silently drop a memory
  below INV-6's one-link floor, which `write_memory` enforces at write time and
  nothing re-checks afterward.

**Why this lives in `palaver/store/` and not `palaver/memory/`.** The plan
places it at `palaver/memory/rollup.py`, but task 2.1's accepted done-when
criteria include `rg -n 'DELETE|DROP' palaver/memory` returning no match, and
`tests/test_memory.py::test_no_delete_or_drop_sql_is_ever_issued_by_the_memory_module`
enforces that as an AST scan over every `execute*` call in that package. A
module whose entire job is issuing `DELETE` cannot live there without either
failing that gate or carving an exception into it, and a directory-wide static
rule with one exception is a weaker guarantee than the same rule with none.
`palaver/store/` already owns schema and migration — how rows are kept — which
is what retention is. The gate keeps its absolute form and this module keeps
its `DELETE`s; only the path moved.

**On the module name.** The plan (task 4.3) calls this "rollup and pruning" and
describes rolling up per-session `current_state` history. That table has no
history to roll up: `UNIQUE (project_id, session_id, key)` plus
`upsert_current_state`'s overwrite-in-place means one live row per
`(project, session, key)` and no prior versions anywhere. Folding an expiring
per-session row into a project-scoped (`session_id IS NULL`) row was considered
and rejected — it would mint a `current_state` row no extraction pass can
reproduce, in the one table whose entire justification is that it is
regeneratable, and the unique constraint makes it a single slot several
expiring sessions would silently overwrite in no defensible order. The task's
own done-when criteria name only pruning. The file keeps the plan's name; see
the dated amendment in the plan task for the full record.

Deletion counts and the cutoff come back in a `PruneReport` rather than being
logged and discarded, so a caller can assert on what a pass did and a scheduled
pass can report it through INV-1's progress channel instead of running silent.

This repository is public. Nothing in this module's docstrings or examples is
derived from a real observed session.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

#: How long regeneratable rows are kept, in days. A policy choice, not a
#: measurement: nothing in the plan or in any benchmark derives this number.
#: It is the default of `prune_ephemeral`'s `retention_days` parameter rather
#: than a constant read inside it, so a caller — or a test — can name a
#: different window without patching the module or moving the clock.
DEFAULT_RETENTION_DAYS = 30.0

#: Seconds-resolution half of the stored timestamp format, matching
#: `schema._CREATED_AT_DEFAULT` and `persist._UPDATED_AT_NOW`. Both write UTC
#: with a `Z` suffix and *millisecond* precision, which makes stored timestamps
#: sort lexicographically in true chronological order — so the cutoff
#: comparisons below can be plain string `<`, with no per-row parsing and no
#: dependence on SQLite's date functions agreeing with Python's. The
#: sub-second digits are appended separately because SQLite's `%f` emits three
#: of them and Python's emits six; a six-digit cutoff would sort *after* every
#: same-second stored row rather than alongside it.
_TIMESTAMP_SECONDS_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Rows older than the cutoff and named by no evidence anchor. The `NOT EXISTS`
#: is the whole safety property of this module, so it lives in one place and is
#: never inlined a second time.
_UNREFERENCED_EXPIRED_CHUNKS = """
    SELECT id FROM transcript_chunks
    WHERE created_at < ?
      AND NOT EXISTS (
          SELECT 1 FROM memory_evidence
          WHERE memory_evidence.transcript_chunk_id = transcript_chunks.id
      )
"""


@dataclass(frozen=True)
class PruneReport:
    """What one `prune_ephemeral` pass deleted, and what it deliberately kept.

    Attributes:
        cutoff: The retention boundary this pass used, as a stored-format
            timestamp. Rows at or after it were kept regardless of anything
            else.
        current_state_rows_deleted: `current_state` rows removed for age.
        transcript_chunks_deleted: `transcript_chunks` rows removed for age
            *and* for being named by no `memory_evidence` row.
        transcript_chunks_kept_for_evidence: Rows that were old enough to
            delete and were kept anyway because evidence names them. Reported
            separately because a pass that keeps everything and a pass that
            keeps nothing are both suspicious, and a single "deleted" count
            cannot tell them apart.
    """

    cutoff: str
    current_state_rows_deleted: int
    transcript_chunks_deleted: int
    transcript_chunks_kept_for_evidence: int

    @property
    def total_rows_deleted(self) -> int:
        """Every row this pass removed, across both tables."""
        return self.current_state_rows_deleted + self.transcript_chunks_deleted


def retention_cutoff(
    retention_days: float = DEFAULT_RETENTION_DAYS,
    now: datetime | None = None,
) -> str:
    """Render the retention boundary as a stored-format timestamp.

    Args:
        retention_days: Window length in days. Fractional values are allowed
            so a caller can express a window shorter than a day.
        now: The instant the window is measured back from, defaulting to the
            current UTC time. A naive datetime is read as UTC.

    Returns:
        The cutoff timestamp, directly comparable with `<` against
        `transcript_chunks.created_at` and `current_state.updated_at`.

    Raises:
        ValueError: If `retention_days` is negative. A negative window would
            put the cutoff in the future and delete rows written seconds ago,
            which is never what a retention policy means and is far too easy
            to reach by subtracting two numbers in the wrong order.
    """
    if retention_days < 0:
        raise ValueError(f"retention_days must not be negative, got {retention_days}")
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    cutoff = moment.astimezone(UTC) - timedelta(days=retention_days)
    milliseconds = cutoff.microsecond // 1000
    return f"{cutoff.strftime(_TIMESTAMP_SECONDS_FORMAT)}.{milliseconds:03d}Z"


def _count_referenced_expired_chunks(conn: sqlite3.Connection, cutoff: str) -> int:
    """Count expired chunks kept because some evidence row names them."""
    row = conn.execute(
        """
        SELECT COUNT(*) FROM transcript_chunks
        WHERE created_at < ?
          AND EXISTS (
              SELECT 1 FROM memory_evidence
              WHERE memory_evidence.transcript_chunk_id = transcript_chunks.id
          )
        """,
        (cutoff,),
    ).fetchone()
    return int(row[0])


def prune_ephemeral(
    conn: sqlite3.Connection,
    *,
    retention_days: float = DEFAULT_RETENTION_DAYS,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> PruneReport:
    """Delete regeneratable rows older than the retention window.

    Removes `current_state` rows whose `updated_at` predates the cutoff, and
    `transcript_chunks` rows whose `created_at` predates it *and* which no
    `memory_evidence` row names. Issues no statement against `memories`,
    `memory_evidence`, or `events` — see the module docstring for why each
    exclusion is deliberate rather than incidental.

    The caller owns committing the connection, matching `write_memory` and
    `persist_extraction`: a scheduled pass that also writes something else
    should be able to put both in one transaction.

    Args:
        conn: Open connection to a migrated database.
        retention_days: Window length in days; rows at or after the cutoff are
            kept.
        now: The instant the window is measured back from, defaulting to the
            current UTC time.
        on_status: INV-1 progress channel. Called before each delete with what
            is about to be attempted, and once with the outcome, so a pass over
            a large store is never a silent wait.

    Returns:
        A `PruneReport` describing what was deleted and what was kept.
    """
    cutoff = retention_cutoff(retention_days, now)
    if on_status is not None:
        on_status(f"pruning regeneratable rows older than {cutoff}")

    kept_for_evidence = _count_referenced_expired_chunks(conn, cutoff)

    state_deleted = conn.execute(
        "DELETE FROM current_state WHERE updated_at < ?", (cutoff,)
    ).rowcount
    if on_status is not None:
        on_status(f"current_state: {state_deleted} row(s) deleted")

    # Deleted by id rather than by repeating the predicate in a DELETE, so the
    # rows counted and the rows removed are the same set by construction. The
    # `_ad` trigger from migration 2 removes each id from the FTS5 index as it
    # goes; leaving those entries behind would let a pruned chunk keep matching
    # `schema.search`.
    expired = [row[0] for row in conn.execute(_UNREFERENCED_EXPIRED_CHUNKS, (cutoff,))]
    for chunk_id in expired:
        conn.execute("DELETE FROM transcript_chunks WHERE id = ?", (chunk_id,))
    if on_status is not None:
        on_status(
            f"transcript_chunks: {len(expired)} row(s) deleted, "
            f"{kept_for_evidence} kept for evidence"
        )

    return PruneReport(
        cutoff=cutoff,
        current_state_rows_deleted=state_deleted,
        transcript_chunks_deleted=len(expired),
        transcript_chunks_kept_for_evidence=kept_for_evidence,
    )
