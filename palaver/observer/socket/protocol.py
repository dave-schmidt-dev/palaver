"""
The daemon side: the closed set of operations a request may name, and applying one to
the store.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from .core import UnsupportedOperationError, log

# ---------------------------------------------------------------------------
# The daemon side: what a request is allowed to ask for, and how it is served.
# ---------------------------------------------------------------------------

#: Every operation the write path performs, by name. A closed set, checked
#: before anything touches the database.
#:
#: The protocol carries an operation *name* and typed arguments — never SQL,
#: and never a table or column name. That is what makes "an UPDATE or DELETE
#: naming an existing memory row" unreachable from a caller rather than
#: merely discouraged: there is no request shape that expresses one. The
#: schema's own triggers (`memories_no_delete`, `memories_id_immutable`, and
#: the rest) are the second layer, and they are what would catch a future
#: operation added here carelessly.
#:
#: `query` is here despite not being a write a *user* asked for: it is
#: Palaver recording its own retrievals (task 6.4). It travels this path
#: because the MCP process is read-only and this machine has exactly one
#: writer — the alternative was a second writable connection, opened for
#: telemetry, which is the thing this module exists to make impossible.
WRITE_OPERATIONS = frozenset({"correct", "query"})

#: What `palaver_correct` records as a memory's origin, so a correction is
#: distinguishable from an extraction in any later read.
CORRECTION_ORIGIN = "user-correction"


def apply_request(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> dict:
    """Perform one write request on the daemon's connection.

    Args:
        conn: The daemon's single writable connection.
        payload: A decoded request. `op` names the operation; the remaining
            keys are its arguments.

    Returns:
        A reply dict. `{"ok": True, ...}` on success; `{"ok": False,
        "error": ..., "detail": ...}` when the request was refused, so a
        refusal reaches the caller as data rather than as a dropped
        connection they would have to guess about.

    Raises:
        Nothing. Every failure is turned into a reply. A daemon that let an
        exception escape here would drop the connection, and the MCP process
        would report "the daemon did not reply" — which is what it says when
        the daemon has *crashed*. Two very different situations must not
        produce the same message (INV-7).
    """
    op = payload.get("op")
    try:
        if op not in WRITE_OPERATIONS:
            raise UnsupportedOperationError(
                f"{op!r} is not something the write path does. It performs exactly "
                f"{sorted(WRITE_OPERATIONS)}, and takes an operation name with typed "
                "arguments — never SQL, a table name, or a column name. Memories are "
                "append-only (INV-4): a correction is a new row that supersedes the "
                "old one, and the old row is never modified or removed."
            )
        if op == "query":
            return _record_query(conn, payload)
        return _correct(conn, payload)
    except Exception as exc:  # noqa: BLE001 - the reply *is* the error channel
        log.warning("write request %r refused: %s", op, exc)
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)}


def _correct(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> dict:
    """Supersede one memory with a corrected statement at tier 1.

    The successor inherits the predecessor's evidence anchors rather than
    inventing new ones. That is the honest reading of a human correction:
    the same span of transcript is being pointed at, and what changed is the
    reading of it. Fabricating an anchor to satisfy INV-6 would put a
    citation in the store that leads somewhere the statement does not come
    from, which is worse than no citation at all.
    """
    from palaver.memory.evidence import EvidenceAnchor  # noqa: PLC0415 - cycle
    from palaver.memory.supersede import supersede_memory  # noqa: PLC0415 - cycle
    from palaver.memory.tiers import TIER_USER_INSTRUCTION  # noqa: PLC0415 - cycle

    memory_id = payload.get("memory_id")
    statement = payload.get("statement")
    if not isinstance(memory_id, int) or isinstance(memory_id, bool):
        raise ValueError(f"memory_id must be an integer, got {memory_id!r}")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")

    anchors = [
        EvidenceAnchor(
            start_offset=start,
            end_offset=end,
            transcript_chunk_id=chunk_id,
            event_id=event_id,
        )
        for start, end, chunk_id, event_id in conn.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ).fetchall()
    ]
    if not anchors:
        raise LookupError(
            f"memory {memory_id} has no evidence to inherit, so a correction of it "
            "would have none either (INV-6). Either the id names no memory, or the "
            "store is inconsistent."
        )

    successor_id = supersede_memory(
        conn,
        predecessor_id=memory_id,
        statement=statement.strip(),
        origin=CORRECTION_ORIGIN,
        tier=TIER_USER_INSTRUCTION,
        evidence=anchors,
    )
    conn.commit()
    return {"ok": True, "memory_id": successor_id, "supersedes": memory_id}


def _record_query(conn: sqlite3.Connection, payload: Mapping[str, Any]) -> dict:
    """Record one retrieval: what was asked, and which memories came back.

    Validated as strictly as a correction is, for a reason that is not
    symmetry. A query event is written on behalf of a caller who never sees
    the reply (see `palaver.mcp.query_events`), so a malformed one that were
    let through would land as a bad row nobody is watching. The strictness
    is what turns that into a WARNING in the daemon's log instead.
    """
    tool = payload.get("tool")
    scope_kind = payload.get("scope_kind")
    scope_value = payload.get("scope_value")
    result_count = payload.get("result_count")
    memory_ids = payload.get("memory_ids", [])

    if not isinstance(tool, str) or not tool.strip():
        raise ValueError(f"tool must be a non-empty string, got {tool!r}")
    if scope_kind not in {"project", "session"}:
        raise ValueError(f"scope_kind must be 'project' or 'session', got {scope_kind!r}")
    if not isinstance(scope_value, str) or not scope_value.strip():
        raise ValueError(f"scope_value must be a non-empty string, got {scope_value!r}")
    if result_count is not None and (
        not isinstance(result_count, int) or isinstance(result_count, bool) or result_count < 0
    ):
        raise ValueError(
            f"result_count must be a non-negative integer or null, got {result_count!r}"
        )
    if not isinstance(memory_ids, list) or not all(
        isinstance(memory_id, int) and not isinstance(memory_id, bool) for memory_id in memory_ids
    ):
        raise ValueError(f"memory_ids must be a list of integers, got {memory_ids!r}")

    event_id = conn.execute(
        "INSERT INTO query_events (tool, scope_kind, scope_value, result_count) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (tool.strip(), scope_kind, scope_value.strip(), result_count),
    ).fetchone()[0]
    # `dict.fromkeys` rather than `set`: a page cannot repeat a memory, but
    # if one ever did, the composite primary key would abort the whole event
    # — losing the parent row to a duplicate in the child list. Order is
    # kept because it is the order the agent was shown them in.
    conn.executemany(
        "INSERT INTO query_event_memories (query_event_id, memory_id) VALUES (?, ?)",
        [(event_id, memory_id) for memory_id in dict.fromkeys(memory_ids)],
    )
    conn.commit()
    return {"ok": True, "query_event_id": event_id, "memory_ids": len(set(memory_ids))}
