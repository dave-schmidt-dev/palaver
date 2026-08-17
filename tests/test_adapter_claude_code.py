"""Tests for the Claude Code JSONL adapter.

Every fixture here is a JSONL file this module writes itself under pytest's
`tmp_path`, with prose invented for the test — no real `~/.claude/` session
store is ever opened, globbed, or read (INV-3). `ClaudeCodeAdapter` is always
constructed with an explicit `root` pointing into `tmp_path`.
"""

import json
import logging
import os
from pathlib import Path

from palaver.ingest.adapters.claude_code import (
    CHANNEL_HUMAN,
    CHANNEL_INJECTED,
    ClaudeCodeAdapter,
    classify_channel,
)
from palaver.ingest.cursors import Cursor


def _jsonl_line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_bytes(b"".join(_jsonl_line(r) for r in records))


def _project_dir(tmp_path: Path, name: str = "-Users-test-project") -> Path:
    d = tmp_path / "projects" / name
    d.mkdir(parents=True)
    return d


def _user_record(text: str, *, is_meta: bool = False, session_id: str = "session-1") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "isMeta": is_meta,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant_text_record(text: str, session_id: str = "session-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _assistant_tool_use_record(tool_name: str, session_id: str = "session-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tu-1", "name": tool_name, "input": {}},
            ],
        },
    }


def _tool_result_record(
    *, is_error: bool, content: str = "ran", session_id: str = "session-1"
) -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "isMeta": False,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-1",
                    "is_error": is_error,
                    "content": content,
                }
            ],
        },
    }


def _compact_boundary_record(session_id: str = "session-1") -> dict:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "sessionId": session_id,
        "content": "Conversation compacted",
    }


# --- session/project identity ------------------------------------------------


def test_session_key_and_project_key_derived_from_path(tmp_path):
    """Session identity combines the encoded project directory and the filename
    stem (Claude Code names the file after its own sessionId) — no file open."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "abc-123.jsonl"
    _write_records(path, [_user_record("hello")])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.project_key_for(path) == "-Users-test-project"
    assert adapter.session_key_for(path) == "-Users-test-project/abc-123"


def test_list_store_paths_finds_jsonl_without_opening_them(tmp_path, monkeypatch):
    """`list_store_paths` enumerates session files under every project directory
    and never opens any of them (the base contract discover_sessions relies on)."""
    project_a = _project_dir(tmp_path, "-Users-test-a")
    project_b = _project_dir(tmp_path, "-Users-test-b")
    path_a = project_a / "session-a.jsonl"
    path_b = project_b / "session-b.jsonl"
    _write_records(path_a, [_user_record("hi")])
    _write_records(path_b, [_user_record("hi")])

    opened = []
    real_open = os.open

    def _spy_open(p, *args, **kwargs):
        opened.append(p)
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(os, "open", _spy_open)

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    paths = list(adapter.list_store_paths())

    assert set(paths) == {path_a, path_b}
    assert opened == []


def test_list_store_paths_missing_root_returns_empty(tmp_path):
    """A root that does not exist yet (no Claude Code sessions observed) yields
    an empty list rather than raising."""
    adapter = ClaudeCodeAdapter(root=tmp_path / "does-not-exist")

    assert list(adapter.list_store_paths()) == []


def test_list_store_paths_excludes_files_nested_deeper_than_one_level(tmp_path):
    """A `.jsonl` nested under a subdirectory of a project directory (Claude
    Code writes such subdirectories for attachments/tool-result blobs) is not
    admitted as a session store — `project_key_for`/`session_key_for` assume
    exactly one level of nesting, and a deeper file would derive a bogus
    identity for it (e.g. the subdirectory name as the "project"). The
    positive control (a real session file in the same project directory) is
    still returned, proving this is a depth filter, not a wholesale exclusion
    of the project directory."""
    project_dir = _project_dir(tmp_path)
    real_session = project_dir / "session-1.jsonl"
    _write_records(real_session, [_user_record("hi")])
    nested = project_dir / "some-uuid" / "tool-results" / "inner.jsonl"
    nested.parent.mkdir(parents=True)
    _write_records(nested, [_user_record("not a session")])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    paths = list(adapter.list_store_paths())

    assert paths == [real_session]


# --- last message-bearing record / tail semantics ----------------------------


def test_last_message_bearing_record_skips_trailing_bookkeeping_records(tmp_path):
    """A fixture ending on `ai-title` (and other bookkeeping types) is not the
    last message-bearing record — the adapter must return the preceding
    assistant record instead of the last line in the file."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    last_assistant = _assistant_text_record("all done here")
    _write_records(
        path,
        [
            _user_record("please help"),
            last_assistant,
            {"type": "attachment", "sessionId": "session-1", "path": "/tmp/x.png"},
            {"type": "last-prompt", "sessionId": "session-1", "text": "please help"},
            {"type": "ai-title", "sessionId": "session-1", "title": "Help request"},
        ],
    )

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.last_message_bearing_record(path) == last_assistant


def test_last_message_bearing_record_none_when_no_message_records(tmp_path):
    """A file with only bookkeeping records (no conversational turn yet) yields
    `None`, not a spurious bookkeeping record treated as message-bearing."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(path, [{"type": "mode", "sessionId": "session-1", "mode": "default"}])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.last_message_bearing_record(path) is None


def test_has_unresolved_trailing_tool_use_true_when_last_message_is_tool_use(tmp_path):
    """A trailing assistant `tool_use` block with nothing message-bearing after it
    is unresolved — even when a non-message bookkeeping record follows it."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(
        path,
        [
            _user_record("run the build"),
            _assistant_tool_use_record("Bash"),
            {"type": "mode", "sessionId": "session-1", "mode": "default"},
        ],
    )

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.has_unresolved_trailing_tool_use(path) is True


def test_has_unresolved_trailing_tool_use_false_when_tool_result_follows(tmp_path):
    """Positive control: the same tool_use, resolved by a following tool_result,
    is not reported as unresolved — proving the check is conditioned on there
    being no message-bearing record after the tool_use, not on tool_use alone."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(
        path,
        [
            _user_record("run the build"),
            _assistant_tool_use_record("Bash"),
            _tool_result_record(is_error=False),
        ],
    )

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.has_unresolved_trailing_tool_use(path) is False


def test_has_unresolved_trailing_tool_use_false_for_plain_text_reply(tmp_path):
    """Positive control: an ordinary trailing assistant text reply (no tool_use
    block at all) is not reported as unresolved."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(path, [_user_record("hi"), _assistant_text_record("hello there")])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")

    assert adapter.has_unresolved_trailing_tool_use(path) is False


# --- channel classification (INV-8) ------------------------------------------


def test_classify_channel_injected_for_every_isMeta_record_human_for_none():
    """INV-8: every isMeta user record classifies as injected, and none of them
    classify as human. A genuinely human record (no isMeta, no known prefix) is
    the positive control proving `classify_channel` can return human at all."""
    meta_records = [
        _user_record("<system-reminder>Session started.</system-reminder>", is_meta=True),
        _user_record("Here is the output of your last command.", is_meta=True),
        _user_record("<command-name>/compact</command-name>", is_meta=True),
    ]
    human_record = _user_record("please refactor the auth module")

    meta_channels = [classify_channel(r) for r in meta_records]

    assert all(channel == CHANNEL_INJECTED for channel in meta_channels)
    assert CHANNEL_HUMAN not in meta_channels
    assert classify_channel(human_record) == CHANNEL_HUMAN


def test_classify_channel_injected_by_prefix_without_isMeta():
    """A record with `isMeta: false` but text matching the injected-prefix table
    (e.g. a rendered local-command block) still classifies as injected — isMeta
    alone is not sufficient, per INV-8's rationale."""
    prefixed_records = [
        _user_record("<local-command-stdout>build ok</local-command-stdout>", is_meta=False),
        _user_record(
            "Caveat: The messages below were generated by the user while running "
            "local commands.\n\nfoo",
            is_meta=False,
        ),
        _user_record("[Request interrupted by user]", is_meta=False),
    ]

    for record in prefixed_records:
        assert classify_channel(record) == CHANNEL_INJECTED


def test_classify_channel_human_positive_control_not_vacuous():
    """Guards against a classifier that always returns injected: an ordinary
    human message with no isMeta flag and no matching prefix must come back
    human."""
    assert classify_channel(_user_record("what's the status of the deploy?")) == CHANNEL_HUMAN


# --- tail: message / compaction / error events -------------------------------


def test_tail_emits_message_events_for_user_and_assistant_records(tmp_path):
    """Ordinary conversational records produce "message" kind events carrying
    the decoded record as payload."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    user = _user_record("hello")
    reply = _assistant_text_record("hi there")
    _write_records(path, [user, reply])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    result = adapter.tail(path, Cursor())

    assert [e.kind for e in result.events] == ["message", "message"]
    assert result.events[0].payload == user
    assert result.events[1].payload == reply
    assert all(e.session_key == adapter.session_key_for(path) for e in result.events)


def test_tail_emits_compaction_event_for_compact_boundary(tmp_path):
    """A `system`/`compact_boundary` record emits a "compaction" kind event."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    boundary = _compact_boundary_record()
    _write_records(path, [_user_record("hi"), boundary])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    result = adapter.tail(path, Cursor())

    compaction_events = [e for e in result.events if e.kind == "compaction"]
    assert len(compaction_events) == 1
    assert compaction_events[0].payload == boundary


def test_tail_other_system_subtypes_do_not_emit_compaction(tmp_path):
    """Positive control: a non-`compact_boundary` system subtype does not emit a
    "compaction" event — proving the compaction mapping is conditioned on the
    subtype, not on `type == "system"` alone."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(
        path,
        [{"type": "system", "subtype": "turn_duration", "sessionId": "session-1", "ms": 12}],
    )

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    result = adapter.tail(path, Cursor())

    assert [e.kind for e in result.events] == ["turn_duration"]
    assert not any(e.kind == "compaction" for e in result.events)


def test_tail_emits_error_event_for_is_error_tool_result(tmp_path):
    """An `is_error` tool result emits an "error" kind event in addition to the
    ordinary "message" event for the record it lives in."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    error_result = _tool_result_record(is_error=True, content="command not found")
    _write_records(path, [_assistant_tool_use_record("Bash"), error_result])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    result = adapter.tail(path, Cursor())

    error_events = [e for e in result.events if e.kind == "error"]
    assert len(error_events) == 1
    assert error_events[0].payload["record"] == error_result
    assert error_events[0].payload["tool_result"]["content"] == "command not found"
    # The record itself is still a message event too.
    assert any(e.kind == "message" and e.payload == error_result for e in result.events)


def test_tail_no_error_event_for_successful_tool_result(tmp_path):
    """Positive control: a tool_result with `is_error: false` produces no "error"
    event — proving the error mapping is conditioned on `is_error`, not emitted
    for every tool_result. Asserts the "message" events still came through, so
    a stub `tail` returning nothing at all cannot pass this."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    _write_records(path, [_assistant_tool_use_record("Bash"), _tool_result_record(is_error=False)])

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    result = adapter.tail(path, Cursor())

    assert [e.kind for e in result.events] == ["message", "message"]
    assert not any(e.kind == "error" for e in result.events)


def test_tail_counts_complete_malformed_records_without_logging_source_content(tmp_path, caplog):
    """A structurally complete but non-JSON line is logged at WARNING and
    skipped, rather than raised or silently dropped — later valid records in
    the same tail call are still returned."""
    project_dir = _project_dir(tmp_path)
    path = project_dir / "session-1.jsonl"
    good = _user_record("hello")
    malformed = b"{not valid json secret-source-content\n"
    path.write_bytes(_jsonl_line(good) + malformed + _jsonl_line(good))

    adapter = ClaudeCodeAdapter(root=tmp_path / "projects")
    with caplog.at_level(logging.WARNING, logger="palaver.ingest.adapters.claude_code"):
        result = adapter.tail(path, Cursor())

    assert [e.payload for e in result.events] == [good, good]
    assert result.malformed_records == 1
    assert result.cursor.offset == path.stat().st_size
    assert any(
        record.levelno == logging.WARNING and "Unparseable" in record.message
        for record in caplog.records
    )
    assert "secret-source-content" not in caplog.text
