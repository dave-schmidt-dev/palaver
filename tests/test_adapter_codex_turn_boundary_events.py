"""Event-kind derivation: turn boundaries, compaction pairs, and the error kind."""

import pytest

from palaver.ingest.adapters.codex import (
    KIND_COMPACTION,
    KIND_ERROR,
    KIND_MESSAGE,
    KIND_SESSION_META,
    KIND_TURN_BOUNDARY,
    CodexAdapter,
)
from palaver.ingest.cursors import Cursor
from tests._adapter_codex_support import _event, _message, _session_meta, _write_rollout

# --- turn boundary ----------------------------------------------------------


def test_task_complete_emits_a_turn_boundary_event(tmp_path):
    """Done-when: `task_complete` emits a turn-boundary event."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-00-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _message("assistant", "the staging deploy is healthy", block_type="output_text"),
            _event("task_complete", last_agent_message="the fixture worker finished"),
        ],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    kinds = [event.kind for event in events]
    assert kinds == [KIND_SESSION_META, KIND_MESSAGE, KIND_MESSAGE, KIND_TURN_BOUNDARY]
    assert events[-1].payload["payload"]["type"] == "task_complete"


def test_turn_aborted_also_emits_a_turn_boundary_event(tmp_path):
    """Codex closes a turn two ways, and neither is the sole signal."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-01-00-fixture.jsonl",
        [_session_meta(), _event("turn_aborted", reason="interrupted")],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert events[-1].kind == KIND_TURN_BOUNDARY


def test_a_non_boundary_event_is_not_a_turn_boundary(tmp_path):
    """Positive control for the boundary kind.

    Without this, `_event_msg_kind` could return `KIND_TURN_BOUNDARY` for
    every `event_msg` and both boundary tests above would still pass.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-02-00-fixture.jsonl",
        [
            _session_meta(),
            _event("agent_reasoning_delta"),
            _event("context_compacted"),
            _event("error", message="a transient fixture timeout", codex_error_info="x"),
        ],
    )
    kinds = [event.kind for event in CodexAdapter(root=root).tail(path, Cursor()).events]
    assert KIND_TURN_BOUNDARY not in kinds
    assert kinds == [KIND_SESSION_META, "agent_reasoning_delta", KIND_COMPACTION, KIND_ERROR]


# --- compaction -------------------------------------------------------------


def test_the_paired_compaction_marker_produces_two_compaction_events(tmp_path):
    """Both halves of Codex's compaction pair are recognized independently.

    Keying only on the `compacted` envelope, or only on `context_compacted`,
    would leave a release that emits one of them invisible.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-03-00-fixture.jsonl",
        [
            _session_meta(),
            {
                "type": "compacted",
                "payload": {
                    "replacement_history": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "earlier turn"}],
                        }
                    ]
                },
            },
            _event("context_compacted"),
        ],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    compactions = [event for event in events if event.kind == KIND_COMPACTION]
    assert len(compactions) == 2
    assert compactions[0].payload["type"] == "compacted"
    assert compactions[1].payload["payload"]["type"] == "context_compacted"
    # The replacement history survives byte-for-byte, so INV-6 evidence
    # anchored into it still indexes the record as written.
    assert compactions[0].payload["payload"]["replacement_history"][0]["role"] == "user"


# --- errors, at all three layers -------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        _event("exec_command_end", exit_code=1, status="completed"),
        _event("exec_command_end", exit_code=0, status="failed"),
        _event("exec_command_end", exit_code=127, status="failed"),
        _event("patch_apply_end", success=False),
        _event("error", message="a transient fixture timeout", codex_error_info="x"),
    ],
    ids=["nonzero_exit", "failed_status", "both", "patch_failed", "event_error"],
)
def test_every_error_layer_maps_to_the_error_kind(tmp_path, record):
    """Research §2 names three error layers; all three are read, independently.

    `exit_code` and `status` are checked separately rather than conjunctively
    — requiring both would go blind if a release stopped emitting one.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-04-00-fixture.jsonl", [record])
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert [event.kind for event in events] == [KIND_ERROR]


@pytest.mark.parametrize(
    "record",
    [
        _event("exec_command_end", exit_code=0, status="completed"),
        _event("patch_apply_end", success=True),
    ],
    ids=["clean_exec", "clean_patch"],
)
def test_a_successful_command_is_not_an_error(tmp_path, record):
    """Positive control: the error mapping discriminates.

    A success still produces an event — nothing is dropped — but under its
    own kind. Without this, `_event_msg_kind` could return `KIND_ERROR` for
    every `exec_command_end` and the five tests above would pass.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-05-00-fixture.jsonl", [record])
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert len(events) == 1
    assert events[0].kind != KIND_ERROR
    assert events[0].kind == record["payload"]["type"]


def test_a_non_integer_exit_code_is_not_read_as_a_failure(tmp_path):
    """`exit_code: true` is not exit code 1.

    `isinstance(True, int)` is true in Python, so a bare integer check would
    read a boolean as a non-zero exit status.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-06-00-fixture.jsonl",
        [_event("exec_command_end", exit_code=True, status="completed")],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert events[0].kind != KIND_ERROR
