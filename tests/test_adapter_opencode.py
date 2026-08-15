"""Tests for the OpenCode adapter (Task 7.2).

Three layers of evidence, matching the module's own two-layer seam plus the
write boundary downstream of it:

- Layer 2 (pure classification/turn-boundary/compaction logic) is exercised
  directly against the committed fixture corpus at
  `tests/fixtures/opencode/*.jsonl` (task 7.0) — `_load_fixture` parses the
  JSONL and hands each record's `data` field to `palaver.ingest.adapters.opencode`
  exactly as `fetch_messages`/`fetch_parts` would, so the same dict shape
  drives both the fixture tests and the SQLite integration test below.
- Layer 1 (guarded SQL read) is exercised against a temporary SQLite database
  this file builds itself, mirroring the real `message`/`part` schema
  (`tests/test_invariants.py`) — never against the real OpenCode store.
- The write path (`palaver.extract.quote_gate.admit_decision`) is exercised
  against a temporary, freshly-migrated store this file also builds itself.
  `tests/test_extraction.py`'s docstring rules out hand-written
  `transcript_chunks.content` for its own (Claude Code) coverage, because
  `palaver.replay.replay` is the real path there. No equivalent OpenCode
  rendering path exists yet — `palaver.replay` and `palaver.extract.normalize`
  are both hardcoded to Claude Code's JSONL shape and out of this task's file
  scope — so this file constructs `transcript_chunks.content` directly,
  tagged via the same `CHANNEL_TAG` vocabulary `normalize.py` uses. Closing
  this gap for real means an OpenCode rendering path in the normalizer; that
  is out of scope here and is called out again in the module's own docstring
  concern below.

INV-9: this repository is public and every fixture record and prose string
in this file is invented for the test, never copied from a real observed
session. The two-part-per-user-message shape (an ordinary part plus a
`synthetic: true` part on the same message) mirrors
`tests/fixtures/opencode/compaction.jsonl`, built the same way at task 7.0.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from palaver.extract.normalize import CHANNEL_TAG
from palaver.extract.quote_gate import admit_decision
from palaver.ingest.adapters import opencode
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED
from palaver.memory.tiers import TIER_OBSERVER_INFERENCE, TIER_USER_INSTRUCTION
from palaver.store.migrate import connect, migrate

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "opencode"


def _load_fixture(name: str) -> list[dict]:
    """Parse a committed fixture JSONL file into its raw records."""
    lines = (FIXTURES_DIR / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _group_by_message(records: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Group a fixture's flat record list into (message, parts) pairs.

    Mirrors what `fetch_messages`/`fetch_parts` would hand a caller: each
    message dict is `{"id", "session_id", "data"}`, each part dict is
    `{"id", "message_id", "data"}` — the fixture's own `type` discriminator
    and denormalized `session_id`-on-part field (a fixture-format
    convenience `tests/fixtures/opencode/README.md` documents; the real
    `part` table has no such column) are dropped here, not passed through.
    """
    messages: dict[str, dict] = {}
    parts_by_message: dict[str, list[dict]] = {}
    order: list[str] = []
    for record in records:
        if record["type"] == "opencode_message":
            messages[record["id"]] = {
                "id": record["id"],
                "session_id": record["session_id"],
                "data": record["data"],
            }
            order.append(record["id"])
        elif record["type"] == "opencode_part":
            parts_by_message.setdefault(record["message_id"], []).append(
                {"id": record["id"], "message_id": record["message_id"], "data": record["data"]}
            )
    return [(messages[message_id], parts_by_message.get(message_id, [])) for message_id in order]


# --- Layer 2: classification (INV-8) ----------------------------------------


def test_synthetic_part_is_injected_channel():
    part_data = {"type": "text", "text": "session continuation notice", "synthetic": True}
    assert opencode.classify_part_channel(part_data) == CHANNEL_INJECTED


def test_ordinary_part_is_human_channel():
    part_data = {"type": "text", "text": "restart the worker queue"}
    assert opencode.classify_part_channel(part_data) == CHANNEL_HUMAN


def test_synthetic_false_is_still_human_channel():
    """`synthetic` present but not True must not be treated as injected."""
    part_data = {"type": "text", "text": "restart the worker queue", "synthetic": False}
    assert opencode.classify_part_channel(part_data) == CHANNEL_HUMAN


def test_compaction_fixture_parts_classify_by_synthetic_flag():
    records = _load_fixture("compaction.jsonl")
    grouped = _group_by_message(records)
    assert len(grouped) == 1
    _message, parts = grouped[0]
    ordinary_part = next(p for p in parts if p["id"].endswith("part-1"))
    compaction_part = next(p for p in parts if p["id"].endswith("part-2"))
    synthetic_part = next(p for p in parts if p["id"].endswith("part-3"))

    assert opencode.classify_part_channel(ordinary_part["data"]) == CHANNEL_HUMAN
    assert opencode.classify_part_channel(synthetic_part["data"]) == CHANNEL_INJECTED
    # The compaction part carries no `synthetic` key at all; classification
    # must not raise or misclassify a part shape it wasn't designed to read.
    assert opencode.classify_part_channel(compaction_part["data"]) == CHANNEL_HUMAN


# --- Layer 2: compaction (exact match, never sniffed) -----------------------


def test_is_compaction_part_true_for_compaction_type():
    assert opencode.is_compaction_part({"type": "compaction", "auto": True}) is True


def test_is_compaction_part_false_for_text_type():
    assert opencode.is_compaction_part({"type": "text", "text": "compaction"}) is False


# --- Layer 2: turn boundary (doubly confirmed) -------------------------------


def test_turn_boundary_true_on_stop_finish_with_terminal_stop_step_finish():
    message_data = {"role": "assistant", "finish": "stop"}
    parts_data = [
        {"type": "text", "text": "the worker queue is restarted"},
        {"type": "step-finish", "reason": "stop"},
    ]
    assert opencode.is_turn_boundary(message_data, parts_data) is True


def test_turn_boundary_false_when_finish_is_tool_calls():
    message_data = {"role": "assistant", "finish": "tool-calls"}
    parts_data = [
        {"type": "tool", "state": {"status": "error"}},
        {"type": "step-finish", "reason": "tool-calls"},
    ]
    assert opencode.is_turn_boundary(message_data, parts_data) is False


def test_turn_boundary_false_when_finish_stop_but_terminal_step_finish_reason_differs():
    """Pins the stricter-than-the-bullet choice: `finish == "stop"` alone is
    not enough if the terminal step-finish part disagrees with it."""
    message_data = {"role": "assistant", "finish": "stop"}
    parts_data = [
        {"type": "text", "text": "the worker queue is restarted"},
        {"type": "step-finish", "reason": "tool-calls"},
    ]
    assert opencode.is_turn_boundary(message_data, parts_data) is False


def test_turn_boundary_false_when_no_parts():
    assert opencode.is_turn_boundary({"role": "assistant", "finish": "stop"}, []) is False


def test_turn_boundary_false_when_finish_is_not_stop_even_if_step_finish_reason_is_stop():
    """Discriminates the `finish` check from the step-finish check independently.

    Every other negative case in this file varies `finish` and the terminal
    part's `reason` together, so a mutant that drops the `finish == "stop"`
    check and relies on `reason` alone still passes them (confirmed by
    mutation testing: dropping that check left this file green). This is not
    a shape `docs/research.md` observed together, but it isolates the two
    signals `is_turn_boundary` is documented to require both of.
    """
    message_data = {"role": "assistant", "finish": "tool-calls"}
    parts_data = [{"type": "step-finish", "reason": "stop"}]
    assert opencode.is_turn_boundary(message_data, parts_data) is False


# --- Layer 2 fed by fixtures: events_for_message -----------------------------


def test_events_for_turn_finished_fixture_emits_turn_boundary():
    grouped = _group_by_message(_load_fixture("turn-finished.jsonl"))
    assistant_message, assistant_parts = grouped[1]
    assert assistant_message["data"]["role"] == "assistant"

    events = opencode.events_for_message(
        "sess/fixture-oc-finished", assistant_message, assistant_parts
    )
    kinds = [event.kind for event in events]
    assert opencode.KIND_TURN_BOUNDARY in kinds
    assert opencode.KIND_ERROR not in kinds


def test_events_for_tool_call_error_fixture_emits_error_not_turn_boundary():
    grouped = _group_by_message(_load_fixture("tool-call-error.jsonl"))
    assistant_message, assistant_parts = grouped[1]
    assert assistant_message["data"]["role"] == "assistant"

    events = opencode.events_for_message(
        "sess/fixture-oc-toolerr", assistant_message, assistant_parts
    )
    kinds = [event.kind for event in events]
    assert opencode.KIND_ERROR in kinds
    assert opencode.KIND_TURN_BOUNDARY not in kinds

    error_event = next(event for event in events if event.kind == opencode.KIND_ERROR)
    assert error_event.payload["part"]["data"]["state"]["status"] == "error"


def test_events_for_compaction_fixture_emits_compaction():
    grouped = _group_by_message(_load_fixture("compaction.jsonl"))
    user_message, user_parts = grouped[0]

    events = opencode.events_for_message("sess/fixture-oc-compaction", user_message, user_parts)
    kinds = [event.kind for event in events]
    assert opencode.KIND_COMPACTION in kinds
    assert opencode.KIND_TURN_BOUNDARY not in kinds

    compaction_event = next(event for event in events if event.kind == opencode.KIND_COMPACTION)
    assert compaction_event.payload["data"]["type"] == "compaction"


# --- `session_message` correction is not merely stated, it's absent ---------


def test_module_never_names_the_dead_table():
    """The original task list named a per-turn table that is empty in the
    real store (`docs/research.md`); this module must not read or mention it
    anywhere, docstring included."""
    source = Path(opencode.__file__).read_text(encoding="utf-8")
    assert "session_message" not in source


# --- Layer 1: guarded SQL read against a temporary SQLite database ----------


def _build_real_shape_store(path: Path) -> None:
    """Build a temp DB mirroring the real `message`/`part` schema.

    Never the real OpenCode store — this connection is opened directly with
    `sqlite3.connect`, write mode, purely to seed fixture rows; every read
    this test performs afterward goes through `opencode.open_store_readonly`.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT);
            CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT);
            CREATE TABLE account (id TEXT PRIMARY KEY, access_token TEXT);
            CREATE TABLE credential (id TEXT PRIMARY KEY, access_token TEXT);
            """
        )
        conn.execute(
            "INSERT INTO message VALUES (?, ?, ?)",
            (
                "msg-01",
                "sess-01",
                json.dumps({"role": "assistant", "finish": "stop"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?)",
            (
                "part-01",
                "msg-01",
                json.dumps({"type": "text", "text": "the worker queue is restarted"}),
            ),
        )
        conn.execute(
            "INSERT INTO part VALUES (?, ?, ?)",
            ("part-02", "msg-01", json.dumps({"type": "step-finish", "reason": "stop"})),
        )
        conn.execute(
            "INSERT INTO account VALUES (?, ?)",
            ("acct-01", "fixture-invented-access-token-not-real"),
        )
        conn.execute(
            "INSERT INTO credential VALUES (?, ?)",
            ("cred-01", "fixture-invented-refresh-token-not-real"),
        )
        conn.commit()
    finally:
        conn.close()


def test_layer1_fetch_from_temp_sqlite_db_feeds_layer2_directly(tmp_path):
    db_path = tmp_path / "opencode-real-shape.db"
    _build_real_shape_store(db_path)

    conn = opencode.open_store_readonly(db_path)
    try:
        messages = opencode.fetch_messages(conn, "sess-01")
        assert [m["id"] for m in messages] == ["msg-01"]
        assert messages[0]["data"] == {"role": "assistant", "finish": "stop"}

        parts = opencode.fetch_parts(conn, "msg-01")
        assert [p["id"] for p in parts] == ["part-01", "part-02"]
        assert parts[-1]["data"]["type"] == "step-finish"

        # Layer 1's own output, unmodified, is what Layer 2 consumes.
        events = opencode.events_for_message("sess-01", messages[0], parts)
        assert opencode.KIND_TURN_BOUNDARY in [event.kind for event in events]
    finally:
        conn.close()


def test_open_store_readonly_blocks_credential_and_account_before_execution(tmp_path):
    db_path = tmp_path / "opencode-real-shape.db"
    _build_real_shape_store(db_path)

    conn = opencode.open_store_readonly(db_path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            conn.execute("SELECT access_token FROM credential")
        with pytest.raises(sqlite3.DatabaseError, match="prohibited"):
            conn.execute("SELECT access_token FROM account")
        # Positive control: the allowed tables this module actually reads
        # from are unaffected by the same guard.
        assert conn.execute("SELECT id FROM message").fetchall() == [("msg-01",)]
    finally:
        conn.close()


def test_open_store_readonly_rejects_writes():
    """`mode=ro` is the other half of INV-3 — attempted writes fail too."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "opencode-real-shape.db"
        _build_real_shape_store(db_path)
        conn = opencode.open_store_readonly(db_path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO message VALUES ('msg-02', 'sess-01', '{}')")
        finally:
            conn.close()


# --- Write path: classification must be acted on, not just recognized ------


def _migrated_store(tmp_path: Path) -> sqlite3.Connection:
    db_path = tmp_path / "write-path.db"
    migrate(db_path)
    return connect(db_path)


def _insert_chunk(
    conn: sqlite3.Connection, *, project_id: int, session_id: int, seq: int, channel: str, text: str
) -> int:
    """Build one `transcript_chunks` row tagged the way `normalize.py` would.

    `channel` must come from `opencode.classify_part_channel`, never a
    literal tag string, so this exercises the real classification-to-tag
    mapping rather than assuming it.
    """
    content = f"{CHANNEL_TAG[channel]}: {text}\n"
    cursor = conn.execute(
        "INSERT INTO transcript_chunks (session_id, seq, role, content) VALUES (?, ?, ?, ?)",
        (session_id, seq, "user", content),
    )
    return cursor.lastrowid


def test_tier1_write_rejected_when_evidence_anchors_to_a_synthetic_part(tmp_path):
    conn = _migrated_store(tmp_path)
    try:
        project_id = conn.execute(
            "INSERT INTO projects (name, path) VALUES (?, ?)",
            ("opencode-write-path-test", str(tmp_path / "project")),
        ).lastrowid
        session_id = conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?)",
            (project_id, opencode.SOURCE, "fixture-oc-compaction"),
        ).lastrowid
        conn.commit()

        grouped = _group_by_message(_load_fixture("compaction.jsonl"))
        _message, parts = grouped[0]
        ordinary_part = next(p for p in parts if p["id"].endswith("part-1"))
        synthetic_part = next(p for p in parts if p["id"].endswith("part-3"))

        ordinary_text = ordinary_part["data"]["text"]
        ordinary_channel = opencode.classify_part_channel(ordinary_part["data"])
        ordinary_chunk_id = _insert_chunk(
            conn,
            project_id=project_id,
            session_id=session_id,
            seq=1,
            channel=ordinary_channel,
            text=ordinary_text,
        )

        synthetic_text = synthetic_part["data"]["text"]
        synthetic_channel = opencode.classify_part_channel(synthetic_part["data"])
        synthetic_chunk_id = _insert_chunk(
            conn,
            project_id=project_id,
            session_id=session_id,
            seq=2,
            channel=synthetic_channel,
            text=synthetic_text,
        )
        conn.commit()

        # Positive control: an ordinary human part reaches tier-1 exactly.
        control = admit_decision(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=ordinary_text,
            quote=ordinary_text,
            transcript_chunk_id=ordinary_chunk_id,
            origin="test-opencode-write-path",
        )
        assert control.grounded.tier == TIER_USER_INSTRUCTION
        assert control.grounded.is_tier_one

        # The synthetic-anchored decision must not reach tier-1, even though
        # both parts belong to the same `role: "user"` message.
        rejected = admit_decision(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement=synthetic_text,
            quote=synthetic_text,
            transcript_chunk_id=synthetic_chunk_id,
            origin="test-opencode-write-path",
        )
        assert rejected.grounded.tier == TIER_OBSERVER_INFERENCE
        assert not rejected.grounded.is_tier_one
        conn.commit()
    finally:
        conn.close()
