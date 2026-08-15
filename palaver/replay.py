"""Replay harness: a recorded fixture through adapter, signals, events, and memory (task 2.5).

This is the regression net the plan names task 2.5 as: an extraction change
that alters what gets stored should surface as a replay diff, not as a
surprise in production. `replay()` is the first code in this repository that
actually drives `palaver.ingest.adapters.claude_code.ClaudeCodeAdapter` and
`palaver.observer.turn_boundary.observe_session` against a real file and
writes what they produce into the SQLite store — Phase 1 (`palaver
status`/`inspect`) never touches the database at all, and Phase 2's memory
modules (`write.py`, `evidence.py`, `scope.py`, `supersede.py`) all assume
`projects`/`sessions` rows already exist. This module is the missing wiring
between the two, exercised here as a harness rather than as the daemon
Phase 4 will eventually build around the same pieces.

**Idempotency comes from `palaver.ingest.cursors.CursorStore`, not from a
second, bespoke "have I seen this row before" check.** `cursor_root` is
derived from `db_path` (`db_path.parent / "cursors"`) rather than taken as an
independent parameter, so a `(db_path, cursor)` pair can never be mismatched
by a caller passing one without the other. A second `replay()` call against
the same `db_path` loads the cursor `replay()` itself persisted after the
first call's `tail()`; against an unmodified fixture (INV-2 guarantees this
module never writes to it), that cursor already sits at end-of-file, so
`tail()` returns zero events and every write below — `events`,
`transcript_chunks`, and the one memory this module writes — is downstream of
that same, empty record loop. There is no separate dedup path to trust.

**What gets written to `memories`, and what deliberately does not.** Task
3.4 (`palaver v1 plan`) draws a line this module respects even though the
`current_state` table it names is not yet wired to anything: regeneratable
per-session fields belong there, and only durable decisions or resolved
questions belong in the append-only `memories` table. The session's derived
`Status` is exactly the regeneratable case — `derive_status()` recomputes it
fresh every call and Phase 1 persists it nowhere — so this module does *not*
write status to `memories`. What it writes instead, and only once per
session ever (gated on this being the pass that first creates the
`sessions` row, not on every pass that happens to see new records), is a
durable ingest fact: that this session was first observed, anchored to the
earliest transcript record that renders any text (see the chunk note
below). That statement makes no channel-attribution
claim (it does not say who wrote the anchored record, or what it means) —
INV-8 exists because `type: "user"` is not reliably "the human," and this
module has no business asserting authorship a later phase's normalizer and
channel classifier are responsible for. Tier is `TIER_OBSERVER_INFERENCE`
(4): the fact is deterministic, but it is *palaver's own account of what it
did*, not literally an observed tool/command result and never a tier-1 quote
(minting tier-1 is task 3.3's quote-grounding gate, not this module's to
claim).

**Transcript content is stored normalized; the raw record lives in
`events.payload`.** This reverses task 2.5's original choice, deliberately
and in one place (task 3.3's amendment, 2026-08-14). `transcript_chunks.content`
holds `palaver.extract.normalize.normalize_records([record])` — exactly the
line(s) the normalizer emits for that record in full-transcript output, with
their trailing newline, so concatenating a session's chunks in `seq` order
reproduces the normalizer's own output byte for byte. `events.payload` still
holds `json.dumps(record, sort_keys=True)`, the entire decoded JSONL record
verbatim, so nothing is lost by the change; the raw record simply stops
being stored twice.

The reason is the evidence anchor. `memory_evidence` indexes
`transcript_chunks.content` by offset, and task 3.3's quote gate checks a
model-supplied quote against those bytes — but the model reads *normalized*
text. Against raw JSON, that check silently admits plain one-line quotes and
rejects any quote containing a newline, a double quote, or a backslash,
because `json.dumps` escapes all three: a pass rate that reads as a quality
measurement while actually measuring quote punctuation. Anchors now index
exactly the bytes the model read, `resolve_evidence` returns text a human
can read instead of a JSON slice, and the FTS5 index over this column
searches semantic text rather than JSON punctuation.

**One chunk per ingested record, including records that normalize to
nothing.** A record whose rendering is empty (an assistant turn carrying
only a `thinking` block, say) stores the empty string rather than being
dropped: `seq` is the ordering evidence anchors are read against, and a hole
in the numbering would silently repoint every anchor after it. The one
consequence is that this module's own memory anchors to the first
*non-empty* chunk — `memory_evidence`'s `CHECK (end_offset > start_offset)`
admits no zero-length span, and an empty chunk is not evidence of anything.

Records that are not `user`/`assistant` still produce no chunk (the event
loop below writes one only for `kind == "message"`), which is unchanged from
task 2.5. The normalizer does render `system` records as `SYSTEM(kind):`
lines, so for a session containing one, the chunk concatenation above
reproduces the normalizer's output minus those lines —
`tests/test_replay.py::test_system_records_render_a_line_with_no_chunk_to_anchor_it`
pins that gap as a measured fact rather than leaving it implied.

**Replay is incremental, not fresh-database-only, so a store can hold chunks
written under both content rules.** `_get_or_create_session` and the cursor
above exist precisely to resume an existing store, and this module only ever
`INSERT`s a chunk — it never `UPDATE`s or `DELETE`s one, and nothing else in
the package writes this column. A database first populated before the change
above therefore keeps its raw-JSON rows verbatim and gains normalized rows
alongside them. The FTS5 index does not go stale from this: `_fts_statements`
indexes on the `AFTER INSERT` trigger, so each row was indexed from the bytes
it actually holds, and an `integrity-check` on a deliberately mixed store
passes (pinned by
`tests/test_replay.py::test_fts_index_stays_consistent_when_a_store_mixes_content_rules`).
Nothing in the package issues an FTS `rebuild`, and after this change nothing
needs to. What a mixed store *does* carry is heterogeneous content: task
3.3's gate would check a quote against raw JSON for the pre-change rows and
so hit exactly the escaping failure described above. Rewriting those rows is
a schema-migration decision, not this module's, and is deliberately not done
here.

**Determinism in the byte-identical-dump comparison.** `memories.created_at`
and `memory_evidence.created_at` come from `write_memory` (task 2.1), whose
`INSERT` has no parameter to override the schema's wall-clock
`created_at`/`started_at` defaults — forking that statement here to add one
would fork the append-only write path this module exists to exercise, not
replace. So `dump_database` normalizes every column named in
`_TIMESTAMP_COLUMNS`, uniformly across every table (not only the two that
strictly need it, so there is one policy rather than a special case), to a
fixed sentinel before comparing. Row ids, statements, tiers, kinds, JSON
payloads, and evidence offsets are all compared for real — two full replays
into two fresh, empty databases issue an identical sequence of `INSERT`s, so
ids match without needing their own normalization.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from palaver.extract.normalize import normalize_records
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import CursorStore
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.tiers import TIER_OBSERVER_INFERENCE
from palaver.memory.write import write_memory
from palaver.observer.signals import Status, derive_status
from palaver.observer.turn_boundary import observe_session
from palaver.store.migrate import connect, migrate

#: Tables `dump_database` reports, in this fixed order.
_DUMP_TABLES = (
    "projects",
    "sessions",
    "transcript_chunks",
    "events",
    "memories",
    "memory_evidence",
    "memory_relationships",
    "current_state",
    "model_runs",
)

#: Columns holding a write-time wall-clock stamp rather than fixture-derived
#: content — see the module docstring's determinism note.
_TIMESTAMP_COLUMNS = frozenset(
    {"created_at", "started_at", "ended_at", "updated_at", "finished_at"}
)

_TIMESTAMP_SENTINEL = "<TIMESTAMP>"


@dataclass(frozen=True)
class ReplayResult:
    """One `replay()` call's outcome.

    Attributes:
        fixture_path: The fixture that was replayed.
        db_path: The database it was replayed into.
        session_key: The session's durable identity
            (`ClaudeCodeAdapter.session_key_for`).
        project_id: `projects.id` the session's rows were scoped under.
        session_id: `sessions.id` for this session.
        status: `derive_status()`'s result for the fixture's *entire*
            current content, computed fresh on every call regardless of how
            many records this particular pass tailed.
        events_written: `events` rows this pass inserted (0 on a fully
            caught-up replay).
        chunks_written: `transcript_chunks` rows this pass inserted.
        memories_written: 0 or 1 — this module writes at most one memory
            per session, ever (see the module docstring).
    """

    fixture_path: Path
    db_path: Path
    session_key: str
    project_id: int
    session_id: int
    status: Status
    events_written: int
    chunks_written: int
    memories_written: int


def replay(
    fixture_path: str | Path,
    db_path: str | Path,
    *,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> ReplayResult:
    """Replay one session fixture through adapter, signals, events, and memory.

    Opens `fixture_path` read-only, end to end (INV-2) — every read here
    goes through `ClaudeCodeAdapter.tail`/`observe_session`, both of which
    route through `palaver.ingest.adapters.base.open_source_readonly`. This
    function never opens `fixture_path` itself and never writes to it.

    Args:
        fixture_path: Path to a JSONL session store.
        db_path: SQLite database to replay into. Migrated to the latest
            schema version if not already there. Its parent directory is
            created if missing. The per-session tail cursor this call reads
            and updates lives at `db_path.parent / "cursors"` — derived,
            not a separate parameter, so a cursor can never be paired with
            the wrong database by accident.
        now: Reference time for `observe_session`'s mtime corroboration.
            Defaults to the current UTC time; tests pass a fixed value.
        on_status: Progress channel (INV-1). Defaults to doing nothing;
            callers that want stderr progress pass one explicitly (the CLI
            layer does).

    Returns:
        A `ReplayResult` describing what this pass did.

    Raises:
        OSError: `fixture_path` cannot be read (including "does not
            exist") — raised by the adapter's own read path, not wrapped
            here.
    """
    fixture_path = Path(fixture_path)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_root = db_path.parent / "cursors"

    adapter = ClaudeCodeAdapter()
    session_key = adapter.session_key_for(fixture_path)
    project_key = adapter.project_key_for(fixture_path)

    if on_status is not None:
        on_status(f"replaying {session_key} from {fixture_path}")

    migrate(db_path)
    conn = connect(db_path)
    try:
        project_id = _get_or_create_project(conn, project_key, str(fixture_path.resolve().parent))
        session_id, session_is_new = _get_or_create_session(
            conn, project_id, adapter.source, session_key
        )

        observation = observe_session(fixture_path, now=now)
        status = derive_status(observation.signals)

        cursor_store = CursorStore(cursor_root)
        cursor = cursor_store.load(session_key)
        tail_result = adapter.tail(fixture_path, cursor)

        events_written = 0
        chunks_written = 0
        anchor_chunk_id: int | None = None
        anchor_chunk_content: str | None = None
        seq = _next_seq(conn, session_id)

        for event in tail_result.events:
            payload_json = json.dumps(event.payload, sort_keys=True)
            conn.execute(
                "INSERT INTO events(session_id, kind, payload) VALUES (?, ?, ?)",
                (session_id, event.kind, payload_json),
            )
            events_written += 1

            if event.kind == "message":
                record = event.payload
                message = record.get("message")
                role = message.get("role") if isinstance(message, dict) else None
                role = role or record.get("type") or "unknown"
                # One `normalize_records` call per record, not a second
                # rendering path of this module's own: for a single record it
                # returns exactly that record's lines, each newline-terminated,
                # so `"".join(chunk contents in seq order)` is the normalizer's
                # full-transcript output for those records, byte for byte.
                content = normalize_records([record])
                chunk_cursor = conn.execute(
                    "INSERT INTO transcript_chunks(session_id, seq, role, content) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, seq, role, content),
                )
                # First *non-empty* chunk, not first chunk: a record that
                # normalizes to nothing still gets its row (seq stays
                # contiguous) but is a zero-length span, which
                # `memory_evidence`'s CHECK (end_offset > start_offset)
                # rejects and which is no evidence of anything anyway.
                if anchor_chunk_id is None and content:
                    anchor_chunk_id = chunk_cursor.lastrowid
                    anchor_chunk_content = content
                chunks_written += 1
                seq += 1

        memories_written = 0
        # Gated on session_is_new, not merely chunks_written > 0: a session
        # already known to this database has already had this fact written
        # once, and a live, growing session tailed incrementally must not
        # accrue a fresh "first observed" memory on every tick that happens
        # to bring in new records (see the module docstring).
        if session_is_new and anchor_chunk_id is not None:
            write_memory(
                conn,
                project_id=project_id,
                session_id=session_id,
                statement=(
                    f"replay: first tail of session {session_key!r} via the "
                    f"{adapter.source!r} adapter ingested its earliest transcript record"
                ),
                origin="replay-harness",
                tier=TIER_OBSERVER_INFERENCE,
                evidence=[
                    EvidenceAnchor(
                        transcript_chunk_id=anchor_chunk_id,
                        start_offset=0,
                        end_offset=len(anchor_chunk_content),
                    )
                ],
            )
            memories_written = 1

        conn.commit()
        # Cursor is persisted only after a successful commit: a crash
        # between the two would leave the cursor behind the database state,
        # so a subsequent replay re-tails and re-writes already-stored
        # records rather than silently losing any (favors a bounded,
        # detectable duplicate over silent data loss, same tradeoff
        # `read_complete_records`'s shrink recovery makes).
        cursor_store.save(session_key, tail_result.cursor)
    finally:
        conn.close()

    if on_status is not None:
        on_status(
            f"replayed {session_key}: {events_written} events, {chunks_written} chunks, "
            f"{memories_written} memories, status={status.value}"
        )

    return ReplayResult(
        fixture_path=fixture_path,
        db_path=db_path,
        session_key=session_key,
        project_id=project_id,
        session_id=session_id,
        status=status,
        events_written=events_written,
        chunks_written=chunks_written,
        memories_written=memories_written,
    )


def _get_or_create_project(conn: sqlite3.Connection, name: str, path: str) -> int:
    """Return `projects.id` for `name`, inserting a row if none exists yet."""
    row = conn.execute("SELECT id FROM projects WHERE name = ?", (name,)).fetchone()
    if row is not None:
        return row[0]
    cursor = conn.execute("INSERT INTO projects(name, path) VALUES (?, ?)", (name, path))
    return cursor.lastrowid


def _get_or_create_session(
    conn: sqlite3.Connection, project_id: int, source: str, external_id: str
) -> tuple[int, bool]:
    """Return `(sessions.id, is_new)` for `(source, external_id)`.

    `is_new` is True only when this call inserted the row — the signal
    `replay()` gates its one-time memory write on.
    """
    row = conn.execute(
        "SELECT id FROM sessions WHERE source = ? AND external_id = ?",
        (source, external_id),
    ).fetchone()
    if row is not None:
        return row[0], False
    cursor = conn.execute(
        "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
        (project_id, source, external_id),
    )
    return cursor.lastrowid, True


def _next_seq(conn: sqlite3.Connection, session_id: int) -> int:
    """Return the next `transcript_chunks.seq` value for `session_id`."""
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) FROM transcript_chunks WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row[0] + 1


def dump_database(conn: sqlite3.Connection) -> str:
    """Render every row of every table as deterministic text, for comparison.

    One line per row, `{table}: {json array of column values}`, tables in
    `_DUMP_TABLES` order and rows within a table by `id` — both fixed, so
    two databases populated by an identical sequence of writes dump
    identically. Columns in `_TIMESTAMP_COLUMNS` are replaced with a fixed
    sentinel (see the module docstring); every other column, including
    every id, is compared as written.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 5 (so every table in `_DUMP_TABLES` exists).

    Returns:
        The dump text. Empty string only if every table is empty.
    """
    lines: list[str] = []
    for table in _DUMP_TABLES:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        rows = conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY id").fetchall()
        for row in rows:
            normalized = [
                _TIMESTAMP_SENTINEL if column in _TIMESTAMP_COLUMNS and value is not None else value
                for column, value in zip(columns, row, strict=True)
            ]
            lines.append(f"{table}: {json.dumps(normalized)}")
    return "\n".join(lines)
