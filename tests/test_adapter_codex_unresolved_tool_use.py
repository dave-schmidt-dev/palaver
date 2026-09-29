"""
Tool-call pairing: dangling calls, boundary resets, and correlation-id-free pending
calls.
"""

from palaver.ingest.adapters.codex import (
    CodexAdapter,
)
from tests._adapter_codex_support import (
    _event,
    _function_call,
    _message,
    _session_meta,
    _write_rollout,
)


def _function_call_output(call_id: str | None = "call-1") -> dict:
    payload = {"type": "function_call_output"}
    if call_id is not None:
        payload["call_id"] = call_id
    return {"type": "response_item", "payload": payload}


# --- unresolved trailing tool use (the Codex inversion) ---------------------


def test_a_turn_boundary_clears_a_dangling_tool_call(tmp_path):
    """The case that proves Claude Code's logic was not copied.

    For Codex the last line usually *is* the turn boundary, so a
    `function_call` with no matching output is resolved by the boundary that
    followed it. Reading it as unresolved would pin a finished session into
    `discover_sessions`'s always-include path for as long as the file exists.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-13-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _function_call("call-1"),
            _event("task_complete", last_agent_message=None),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False


def test_an_unanswered_tool_call_with_no_boundary_is_unresolved(tmp_path):
    """Positive control: the check can return True."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-14-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _function_call("call-1"),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_matching_output_resolves_a_tool_call(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-15-00-fixture.jsonl",
        [_session_meta(), _function_call("call-1"), _function_call_output("call-1")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False


def test_a_mismatched_output_does_not_resolve_a_tool_call(tmp_path):
    """Correlation is by `call_id`, not by arrival order."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-16-00-fixture.jsonl",
        [_session_meta(), _function_call("call-1"), _function_call_output("call-2")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_call_reopened_after_a_boundary_is_unresolved_again(tmp_path):
    """A boundary clears the calls before it, not the ones after it."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-17-00-fixture.jsonl",
        [
            _session_meta(),
            _function_call("call-1"),
            _event("task_complete", last_agent_message=None),
            _function_call("call-2"),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_call_without_a_correlation_id_still_counts_as_pending(tmp_path):
    """An unkeyed call cannot be matched, so ignoring it would under-report."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-18-00-fixture.jsonl",
        [_session_meta(), _function_call(None)],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_session_with_no_tool_calls_is_resolved(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-19-00-fixture.jsonl",
        [_session_meta(), _message("user", "check the staging deploy status")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False
