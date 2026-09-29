"""
Record ordinal/ordering helpers, and CodexAdapter's cursor-resumable tailing of a
rollout file.
"""

import json

from palaver.ingest.adapters.base import Event
from palaver.ingest.adapters.codex import (
    KIND_SESSION_META,
    KIND_TURN_BOUNDARY,
    CodexAdapter,
    order_records,
    record_ordinal,
)
from palaver.ingest.cursors import Cursor
from tests._adapter_codex_support import (
    _event,
    _function_call,
    _message,
    _session_meta,
    _write_rollout,
)

# --- cursor and ordinals ----------------------------------------------------


def test_record_ordinal_prefers_the_records_own_number():
    assert record_ordinal({"ordinal": 7}, 0) == 7
    assert record_ordinal({}, 3) == 3
    assert record_ordinal({"ordinal": "7"}, 3) == 3
    assert record_ordinal({"ordinal": True}, 3) == 3, "a boolean is not an ordinal"
    assert record_ordinal({"ordinal": 0}, 5) == 0


def test_records_are_ordered_by_ordinal_when_every_record_has_one():
    records = [{"ordinal": 2, "n": "c"}, {"ordinal": 0, "n": "a"}, {"ordinal": 1, "n": "b"}]
    assert [r["n"] for r in order_records(records)] == ["a", "b", "c"]


def test_records_keep_file_order_when_ordinals_are_absent_or_partial():
    """A batch with no shared coordinate system is left alone.

    Mixing real ordinals with fallback line indices would interleave the two
    arbitrarily, which is worse than the arrival order it replaced.
    """
    no_ordinals = [{"n": "a"}, {"n": "b"}, {"n": "c"}]
    assert [r["n"] for r in order_records(no_ordinals)] == ["a", "b", "c"]

    partial = [{"ordinal": 9, "n": "a"}, {"n": "b"}, {"ordinal": 1, "n": "c"}]
    assert [r["n"] for r in order_records(partial)] == ["a", "b", "c"]


def test_ordering_is_stable_for_equal_ordinals():
    records = [{"ordinal": 1, "n": "a"}, {"ordinal": 1, "n": "b"}]
    assert [r["n"] for r in order_records(records)] == ["a", "b"]


def test_tail_resumes_from_its_cursor_without_re_reading(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-20-00-fixture.jsonl",
        [_session_meta(), _message("user", "check the staging deploy status")],
    )
    adapter = CodexAdapter(root=root)
    first = adapter.tail(path, Cursor())
    assert len(first.events) == 2
    assert first.cursor.offset == path.stat().st_size

    second = adapter.tail(path, first.cursor)
    assert second.events == ()
    assert second.cursor.offset == first.cursor.offset

    with path.open("ab") as handle:
        handle.write(
            (json.dumps(_event("task_complete", last_agent_message=None)) + "\n").encode("utf-8")
        )
    third = adapter.tail(path, second.cursor)
    assert [event.kind for event in third.events] == [KIND_TURN_BOUNDARY]


def test_tail_does_not_advance_past_a_torn_write(tmp_path):
    """A record the agent is mid-way through flushing is not ingested.

    Codex appends to these files while Palaver reads them, so a partial line
    is expected, not exceptional.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-21-00-fixture.jsonl", [_session_meta()])
    complete_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"type": "response_item", "payl')

    result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META]
    assert result.cursor.offset == complete_size
    assert result.malformed_records == 0


def test_tail_counts_complete_malformed_records_without_logging_source_content(tmp_path, caplog):
    """A corrupt line must not crash the tail, and must not vanish silently."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-22-00-fixture.jsonl", [_session_meta()])
    with path.open("ab") as handle:
        handle.write(b"{not json secret-source-content}\n")
        handle.write((json.dumps(_event("task_complete", last_agent_message=None)) + "\n").encode())

    with caplog.at_level("WARNING"):
        result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META, KIND_TURN_BOUNDARY]
    assert result.malformed_records == 1
    assert any("Unparseable Codex rollout record" in record.message for record in caplog.records)
    assert "secret-source-content" not in caplog.text


def test_a_non_object_record_is_skipped(tmp_path, caplog):
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-23-00-fixture.jsonl", [_session_meta()])
    with path.open("ab") as handle:
        handle.write(b"[1, 2, 3]\n")

    with caplog.at_level("WARNING"):
        result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META]
    assert result.malformed_records == 1
    assert any("Non-object Codex rollout record" in record.message for record in caplog.records)


# --- tail payload fidelity --------------------------------------------------


def test_tail_payloads_are_the_source_records_byte_for_byte(tmp_path):
    """INV-6: an evidence anchor must index the record as written.

    A reshaped payload would make the anchor point at Palaver's rendering of
    a record rather than the record, and the quote-grounding gate would then
    be checking a quote against text the source never contained.
    """
    root = tmp_path / "sessions"
    records = [
        _session_meta(),
        _message("user", "check the staging deploy status"),
        _event("error", message="a transient fixture timeout", codex_error_info="x"),
    ]
    path = _write_rollout(root, "rollout-2026-08-14T10-24-00-fixture.jsonl", records)
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert [event.payload for event in events] == records


def test_tail_stamps_every_event_with_the_session_key(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root, "rollout-2026-08-14T10-25-00-fixture.jsonl", [_session_meta(), _function_call()]
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert {event.session_key for event in events} == {"rollout-2026-08-14T10-25-00-fixture"}
    assert all(isinstance(event, Event) for event in events)


def test_an_unrecognized_record_type_is_ingested_under_its_own_kind(tmp_path):
    """A future Codex release's records stay visible as evidence.

    Dropping them would make the transcript Palaver stores quietly
    incomplete, and INV-7's `UNKNOWN` status depends on absent signals being
    absent for a reason rather than by omission.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-26-00-fixture.jsonl",
        [{"type": "world_state", "payload": {"anything": 1}}, {"payload": {}}],
    )
    kinds = [event.kind for event in CodexAdapter(root=root).tail(path, Cursor()).events]
    assert kinds == ["world_state", "unknown"]
