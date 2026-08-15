"""OpenCode adapter over the `message` and `part` tables (Task 7.2).

OpenCode's real session store is `~/.local/share/opencode/opencode.db`, a
single SQLite database (schema v4, ~2.2 GB, 2,156 sessions on this
machine — `docs/research.md` section 3). The per-turn table the original
task list named turned out to be empty in the real store; this module reads
only `message` (one row per turn, JSON in `data`) and `part` (one row per
content unit, JSON in `data`), which `docs/research.md` confirms hold every
signal this adapter needs.

**INV-3 — read-only and table-allowlisted.** `account` and `credential`
carry plaintext `access_token`/`refresh_token` and this module has no use
for either. Rather than a second, narrower allowlist, `open_store_readonly`
delegates entirely to `palaver.ingest.adapters.opencode_guard`
(`open_guarded_readonly`), which already carries both INV-3 defenses: a
`file:...?mode=ro` connection URI, plus a SQLite authorizer
(`install_table_allowlist`) that denies any `SQLITE_READ` of a table outside
`opencode_guard.ALLOWED_TABLES` before the statement compiles — a query
naming `credential` or `account` never reaches execution. This module never
opens a `sqlite3.connect()` of its own and never widens or re-derives that
allowlist; `ALLOWED_TABLES` already includes `session` and `project` for a
future identity-resolution need, but the two functions below only ever name
`message` and `part` in their SQL, as literals, never built from caller
input.

**The two-layer seam (2026-08-14 orchestrator amendment).** The committed
fixture corpus at `tests/fixtures/opencode/` is JSONL using an invented
`opencode_message`/`opencode_part` record shape (`palaver/cli/fixture_lint.py`'s
`OPENCODE_RECORD_SHAPES`) — deliberately, because `fixture-lint` scans JSONL
text and a committed `.db` would be an opaque blob on a public remote. The
real store is SQLite. To keep both halves genuinely exercised rather than
having the fixture corpus measure a code path production never runs, this
module is split in two:

- **Layer 1** (`open_store_readonly`, `fetch_messages`, `fetch_parts`) opens
  the guarded, read-only connection and turns `message`/`part` rows into
  plain dicts: `{"id", "session_id", "data": <parsed JSON>}` for a message,
  `{"id", "message_id", "data": <parsed JSON>}` for a part — the `data`
  column's JSON text is decoded here and nowhere else.
- **Layer 2** (`classify_part_channel`, `is_compaction_part`,
  `is_turn_boundary`, `events_for_message`) is pure: it takes those same
  dict shapes and does all classification, turn-boundary, and event work
  with no SQL and no knowledge of where the dicts came from. A fixture's own
  `data` field, decoded from JSON by whatever reads the fixture, is
  structurally identical to what Layer 1 produces and can be handed to
  Layer 2 directly — `tests/test_adapter_opencode.py` does exactly that
  against the committed corpus, and separately drives Layer 1 against a
  temporary SQLite database it builds, so the SQLite half is executed by at
  least one test rather than only ever exercised in production.

**INV-8 — provenance is per-part, not per-message.** `part.data.synthetic
== true` marks harness-generated content, and a synthetic continuation nudge
can be attached to the very same `role: "user"` message as genuine user
text (`tests/fixtures/opencode/compaction.jsonl` is built exactly this way)
— so `message.data.role` alone is not sufficient to decide whether a part is
something the human said. This reproduces the Claude Code `isMeta` lesson
(`palaver.ingest.adapters.claude_code.classify_channel`) in a second,
independently-shaped store: `classify_part_channel` here returns the same
`CHANNEL_HUMAN`/`CHANNEL_INJECTED` values that module defines, imported
rather than redefined, so any downstream code that already recognizes those
two tags (`palaver.extract.normalize.CHANNEL_TAG`,
`palaver.extract.quote_gate`) recognizes an OpenCode part's classification
without needing OpenCode-specific handling. Like `classify_channel`, this
function assumes the caller only invokes it for a part belonging to a
`role: "user"` message; an assistant turn's text carries no INV-8 channel
ambiguity to resolve, the same reasoning the Claude Code adapter applies.

**Turn boundary — doubly confirmed, not just `finish == "stop"`.**
`message.data.finish == "stop"` alone is corroborated by the message's
*last* part being `type == "step-finish"` with `reason == "stop"` — the
stricter of the two signals research.md measured as agreeing on every
sampled finished session. `is_turn_boundary` requires both, not the finish
value alone, so a message that merely carries `finish == "stop"` without a
matching terminal `step-finish` part does not emit a boundary.

**Compaction — exact, never sniffed.** `part.data.type == "compaction"` is
the only compaction signal this module recognizes (rare: 3 rows in 53,378 in
the real store), never a text-prefix or keyword match on a part's rendered
content.

**No `discover_sessions`/`tail` subclass of `palaver.ingest.adapters.base.Adapter`
here, deliberately.** That base class's `discover_sessions` windows every
session on `os.stat(path).st_mtime` and `session_key_for` derives
identity from a filesystem path alone — both assume one store path is one
session. OpenCode is the opposite shape: one database file holds all 2,156
sessions, so a single file mtime cannot window them individually (the floor
degenerates to all-or-nothing) and `session_key_for(path)` has no path to
read a session identity out of. Forcing that interface here would mean
inventing per-session pseudo-paths the rest of the interface was never
designed to carry. `base.py`'s own module docstring still says OpenCode will
implement the four abstract methods (written at task 1.3, before this
mismatch was concrete); that line is now stale and is flagged here rather
than edited, since `base.py` is not a file this task owns. `events_for_message`
still returns the shared `palaver.ingest.adapters.base.Event` type, so
whatever wires OpenCode into discovery later (task 7.3 or a dedicated
follow-up) has a canonical event to build on rather than a third shape.

This repository is public. No prose in this module's docstrings or examples
is derived from a real observed session (INV-9).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from palaver.ingest.adapters.base import Event
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED
from palaver.ingest.adapters.opencode_guard import open_guarded_readonly

#: This adapter's `sessions.source` value.
SOURCE = "opencode"

#: `message.data.finish` values `docs/research.md` observed. Only
#: `FINISH_STOP` is meaningful to this module's turn-boundary logic; the
#: others are named for callers that want to distinguish them.
FINISH_STOP = "stop"
FINISH_TOOL_CALLS = "tool-calls"
FINISH_UNKNOWN = "unknown"

#: `part.data.type` values this module recognizes.
PART_TYPE_TEXT = "text"
PART_TYPE_TOOL = "tool"
PART_TYPE_STEP_FINISH = "step-finish"
PART_TYPE_COMPACTION = "compaction"

#: Canonical `Event.kind` values this adapter emits.
KIND_MESSAGE = "message"
KIND_TURN_BOUNDARY = "turn_boundary"
KIND_COMPACTION = "compaction"
KIND_ERROR = "error"


# --- Layer 1: guarded SQL read, yields plain row dicts ----------------------


def open_store_readonly(path: str) -> sqlite3.Connection:
    """Open an OpenCode-shaped SQLite store with both INV-3 defenses installed.

    A thin, documented delegate to
    `palaver.ingest.adapters.opencode_guard.open_guarded_readonly` — this
    module never opens its own `sqlite3.connect()` and never re-derives the
    table allowlist, so there is exactly one place either defense could be
    accidentally dropped, and it is not here.

    Args:
        path: Filesystem path to the SQLite database.

    Returns:
        A connection opened `mode=ro` with the INV-3 table-allowlist
        authorizer already attached.
    """
    return open_guarded_readonly(path)


def fetch_messages(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Read every `message` row for one session, oldest id first.

    `message.id` is KSUID-shaped and lexicographically monotonic with
    insertion order in the real store (`docs/research.md` section 3), so
    ordering by `id` is ordering by time. The table name and column list
    are literals in the query below, never interpolated from an argument,
    so nothing this function's caller passes in can widen which table gets
    read — the INV-3 allowlist enforced on `conn` (see
    `open_store_readonly`) is the second, independent defense regardless.

    Args:
        conn: A connection from `open_store_readonly` (or any connection
            carrying the same INV-3 guard).
        session_id: `message.session_id` to filter on.

    Returns:
        One dict per row, in ascending `id` order:
        `{"id": str, "session_id": str, "data": dict}`, with `data` already
        decoded from the column's JSON text.
    """
    rows = conn.execute(
        "SELECT id, session_id, data FROM message WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    return [{"id": row[0], "session_id": row[1], "data": json.loads(row[2])} for row in rows]


def fetch_parts(conn: sqlite3.Connection, message_id: str) -> list[dict]:
    """Read every `part` row for one message, oldest id first.

    See `fetch_messages` for the ordering rationale and the allowlist note;
    the same reasoning applies here with `part.message_id` in place of
    `message.session_id`.

    Args:
        conn: A connection from `open_store_readonly` (or any connection
            carrying the same INV-3 guard).
        message_id: `part.message_id` to filter on.

    Returns:
        One dict per row, in ascending `id` order:
        `{"id": str, "message_id": str, "data": dict}`, with `data` already
        decoded from the column's JSON text.
    """
    rows = conn.execute(
        "SELECT id, message_id, data FROM part WHERE message_id = ? ORDER BY id ASC",
        (message_id,),
    ).fetchall()
    return [{"id": row[0], "message_id": row[1], "data": json.loads(row[2])} for row in rows]


# --- Layer 2: classification, turn boundary, compaction, events ------------


def classify_part_channel(part_data: dict) -> str:
    """Classify one text part of a `role: "user"` message (INV-8).

    Callers must only invoke this for a part belonging to a `role: "user"`
    message — the same convention
    `palaver.ingest.adapters.claude_code.classify_channel` uses for
    `type: "user"` records. An assistant turn's text carries no INV-8
    channel ambiguity to resolve.

    Args:
        part_data: A part's decoded `data` JSON, e.g.
            `{"type": "text", "text": "...", "synthetic": true}`.

    Returns:
        `CHANNEL_INJECTED` (imported from
        `palaver.ingest.adapters.claude_code`, not redefined) if
        `part_data["synthetic"]` is `True`; `CHANNEL_HUMAN` otherwise,
        including when the part carries no `synthetic` key at all.
    """
    if part_data.get("synthetic") is True:
        return CHANNEL_INJECTED
    return CHANNEL_HUMAN


def is_compaction_part(part_data: dict) -> bool:
    """Report whether a part is OpenCode's compaction marker.

    Exact structural match on `part_data["type"] == "compaction"` only —
    never a text-prefix or keyword match on rendered content. Compaction
    parts are rare (3 rows in 53,378 in the real store) but this field is
    exact, so no heuristic is needed.

    Args:
        part_data: A part's decoded `data` JSON.

    Returns:
        `True` if `part_data["type"]` is `"compaction"`.
    """
    return part_data.get("type") == PART_TYPE_COMPACTION


def _terminal_step_finish(parts_data: Sequence[dict]) -> dict | None:
    """Return the last part's data if it is a `step-finish` part, else `None`."""
    if not parts_data:
        return None
    last = parts_data[-1]
    return last if last.get("type") == PART_TYPE_STEP_FINISH else None


def is_turn_boundary(message_data: dict, parts_data: Sequence[dict]) -> bool:
    """Report whether one message's turn ended (doubly confirmed).

    Requires `message_data["finish"] == "stop"` **and** the message's last
    part to be `type == "step-finish"` with `reason == "stop"` — the
    stricter of the two signals, matching what `docs/research.md` measured
    as agreeing on every sampled finished session. `finish == "stop"` alone,
    without a matching terminal `step-finish` part, is not a turn boundary
    under this function.

    Args:
        message_data: A message's decoded `data` JSON.
        parts_data: That message's parts' decoded `data` JSON, in the same
            order `fetch_parts` (or a fixture) returns them — the *last*
            element is the one checked.

    Returns:
        `True` only when both signals agree.
    """
    if message_data.get("finish") != FINISH_STOP:
        return False
    step_finish = _terminal_step_finish(parts_data)
    return step_finish is not None and step_finish.get("reason") == FINISH_STOP


def _tool_part_is_error(part_data: dict) -> bool:
    """Report whether a `type: "tool"` part's state is `"error"`."""
    if part_data.get("type") != PART_TYPE_TOOL:
        return False
    state = part_data.get("state")
    return isinstance(state, dict) and state.get("status") == "error"


def events_for_message(session_key: str, message: dict, parts: Sequence[dict]) -> list[Event]:
    """Map one message and its ordered parts to the canonical events they produce.

    Pure Layer 2 logic: `message` and each element of `parts` are plain
    dicts shaped like `fetch_messages`/`fetch_parts`'s output (or a
    fixture's own `data` field, wrapped the same way) — nothing here touches
    a database connection.

    Args:
        session_key: Durable identity of the session this message belongs
            to. This module does not derive that identity itself (see the
            module docstring's note on why it does not subclass `Adapter`);
            callers supply it.
        message: `{"id", "session_id", "data"}` for one `message` row.
        parts: That message's parts, in ascending order, each
            `{"id", "message_id", "data"}`.

    Returns:
        Canonical events, in a fixed order: one `KIND_MESSAGE` event first,
        then one `KIND_COMPACTION` or `KIND_ERROR` event per part that
        earns one (in part order), then a `KIND_TURN_BOUNDARY` event if
        `is_turn_boundary` holds, then a `KIND_ERROR` event if the message
        itself carries an unresolved error (`data.error` present with
        `data.finish` absent/`None` — the turn-level error shape
        `docs/research.md` documents alongside the tool-level one).
    """
    message_data = message["data"]
    events = [Event(session_key=session_key, kind=KIND_MESSAGE, payload=message)]

    for part in parts:
        part_data = part["data"]
        if is_compaction_part(part_data):
            events.append(Event(session_key=session_key, kind=KIND_COMPACTION, payload=part))
        elif _tool_part_is_error(part_data):
            events.append(
                Event(
                    session_key=session_key,
                    kind=KIND_ERROR,
                    payload={"message": message, "part": part},
                )
            )

    if is_turn_boundary(message_data, [part["data"] for part in parts]):
        events.append(Event(session_key=session_key, kind=KIND_TURN_BOUNDARY, payload=message))

    if message_data.get("error") is not None and message_data.get("finish") is None:
        events.append(Event(session_key=session_key, kind=KIND_ERROR, payload={"message": message}))

    return events
