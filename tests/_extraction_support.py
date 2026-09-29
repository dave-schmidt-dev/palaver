"""Tests for `palaver.extract.quote_gate` — the memory write boundary (task 3.3).

Every chunk these tests gate against is produced by the real pipeline, not
hand-written: each test writes a JSONL fixture under `tmp_path`, replays it
through `palaver.replay.replay` (adapter -> `classify_channel` ->
normalizer -> `transcript_chunks`), and then runs the gate over the rows that
lands. A hand-written `transcript_chunks.content` string would let these
tests keep passing while the normalizer, the channel classifier, or the
replay writer drifted out from under them; the point of INV-8 is that the
channel tag on a stored line is the classifier's verdict, so the tests have
to go through the classifier to mean anything.

**Every negative assertion here is paired with a positive control on the
same input shape** — usually the same fixture, often the same chunk and the
same quote, with one property changed. A gate that returned tier-4
unconditionally would satisfy every "is not tier-1" assertion in this file
and fail every control; a gate deleted outright fails both.

INV-9: no prose in this module was copied, sampled, or paraphrased from a
real agent session, and no real `~/.claude/` session store is opened. Where
a test reproduces a failure the observer model actually produced, it
reproduces the *shape* of that failure — a real quote carrying a statement
the model wrote itself — with content invented here, and names the model run
in the test's docstring.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from palaver.replay import replay
from palaver.store.migrate import connect

#: Fixed reference time so nothing here depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

ORIGIN = "observer-extraction"


# --- fixture builders (invented content only, INV-9) ------------------------


def _user_record(text: str, *, is_meta: bool = False) -> dict:
    return {
        "type": "user",
        "sessionId": "gate-fixture",
        "isMeta": is_meta,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _replayed(tmp_path: Path, records: list[dict]) -> tuple[sqlite3.Connection, int, int]:
    """Write `records` as a session fixture, replay it, and open the store.

    Returns:
        `(connection, project_id, session_id)`. The caller closes the
        connection; every chunk in it was written by `replay()` itself.
    """
    fixture = tmp_path / "gate-project" / "gate-session.jsonl"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = replay(fixture, tmp_path / "store" / "replay.db", now=NOW)
    assert result.chunks_written == len(records), "fixture builder wrote a record with no chunk"
    return connect(result.db_path), result.project_id, result.session_id


def _chunk_id(conn: sqlite3.Connection, seq: int) -> int:
    (chunk_id,) = conn.execute("SELECT id FROM transcript_chunks WHERE seq = ?", (seq,)).fetchone()
    return chunk_id


def _content(conn: sqlite3.Connection, chunk_id: int) -> str:
    (content,) = conn.execute(
        "SELECT content FROM transcript_chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    return content


def _memory_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    (memories,) = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    (evidence,) = conn.execute("SELECT COUNT(*) FROM memory_evidence").fetchone()
    return memories, evidence
