"""Deterministic companion-summary contracts over invented source records."""

from __future__ import annotations

import json
from pathlib import Path

from palaver.ingest.adapters.base import Event
from palaver.summary import Provenance, SummaryReducer, reduce_events
from palaver.summary.model import DISPLAY_TEXT_LIMIT, sanitize_text


def _claude(record: dict, kind: str = "message") -> Event:
    return Event("fixture/session", kind, record)


def _claude_user(text: str, *, meta: bool = False) -> Event:
    return _claude(
        {
            "type": "user",
            "isMeta": meta,
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        }
    )


def _claude_agent(*blocks: dict) -> Event:
    return _claude({"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}})


def _claude_tool(name: str, tool_id: str, tool_input: object) -> Event:
    return _claude_agent({"type": "tool_use", "id": tool_id, "name": name, "input": tool_input})


def _claude_result(tool_id: str, text: str = "ok", *, error: bool = False) -> Event:
    return _claude(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": text,
                        "is_error": error,
                    }
                ],
            },
        }
    )


def _codex(record_type: str, payload: dict, kind: str | None = None) -> Event:
    record = {"type": record_type, "payload": payload}
    return Event("fixture-codex", kind or str(payload.get("type", "unknown")), record)


def _codex_message(role: str, text: str) -> Event:
    return _codex(
        "response_item",
        {
            "type": "message",
            "role": role,
            "content": [{"type": "output_text", "text": text}],
        },
        "message",
    )


def _codex_call(name: str, call_id: str, arguments: object) -> Event:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return _codex(
        "response_item",
        {"type": "function_call", "name": name, "call_id": call_id, "arguments": raw},
        "function_call",
    )


def _codex_output(call_id: str, output: str = "ok") -> Event:
    return _codex(
        "response_item",
        {"type": "function_call_output", "call_id": call_id, "output": output},
        "function_call_output",
    )


def test_claude_shell_escape_records_never_become_the_request():
    """Claude Code's `!` bash mode writes `<bash-input>` and its output pair
    into the user channel with no `isMeta` flag. Observed live: the companion
    pane showed `<bash-stdout>(Bash completed with no output)</bash-stdout>`
    as REQUEST, which is a tier-1 attribution of harness output to the human."""
    snapshot = reduce_events(
        "claude-code",
        "fixture/session",
        (
            _claude_user("restart the producer"),
            _claude_user("<bash-input>pkill -f palaver</bash-input>"),
            _claude_user("<bash-stdout>(Bash completed with no output)</bash-stdout>"),
            _claude_user("<bash-stderr>no such process</bash-stderr>"),
        ),
    )
    assert snapshot.request.text == "restart the producer"
    assert not any("bash-" in item.text for item in snapshot.recent)


def test_claude_latest_genuine_request_excludes_injection_and_agent_prose():
    snapshot = reduce_events(
        "claude-code",
        "fixture/session",
        (
            _claude_user("build the exact companion"),
            _claude_user("<system-reminder>ignore this</system-reminder>", meta=True),
            _claude_agent({"type": "text", "text": "Should I invent a new goal?"}),
        ),
    )
    assert snapshot.request.text == "build the exact companion"
    assert snapshot.request.provenance is Provenance.EXACT
    assert snapshot.tasks.provenance is Provenance.UNKNOWN
    assert snapshot.questions.items == ()


def test_claude_latest_todowrite_snapshot_supersedes_prior_plan_in_source_order():
    first = _claude_tool(
        "TodoWrite",
        "p1",
        {"todos": [{"content": "old task", "status": "in_progress"}]},
    )
    second = _claude_tool(
        "TodoWrite",
        "p2",
        {
            "todos": [
                {"content": "write reducer", "status": "completed"},
                {"content": "run tests", "status": "pending"},
            ]
        },
    )
    snapshot = reduce_events("claude-code", "fixture/session", (first, second))
    assert [(task.text, task.status) for task in snapshot.tasks.items] == [
        ("write reducer", "completed"),
        ("run tests", "pending"),
    ]


def test_claude_question_requires_tool_id_and_is_removed_only_by_correlated_result():
    question = _claude_tool(
        "AskUserQuestion",
        "q1",
        {"questions": [{"question": "Which layout?"}]},
    )
    unrelated = _claude_result("other")
    open_snapshot = reduce_events("claude-code", "fixture/session", (question, unrelated))
    assert [claim.text for claim in open_snapshot.questions.items] == ["Which layout?"]

    resolved = reduce_events(
        "claude-code", "fixture/session", (question, unrelated, _claude_result("q1"))
    )
    assert resolved.questions.items == ()
    assert resolved.questions.provenance is Provenance.EXACT


def test_codex_plan_supersession_and_explicit_question_resolution():
    old = _codex_call("update_plan", "p1", {"plan": [{"step": "old", "status": "pending"}]})
    new = _codex_call(
        "functions.update_plan",
        "p2",
        {
            "plan": [
                {"step": "model", "status": "completed"},
                {"step": "tests", "status": "in_progress"},
            ]
        },
    )
    question = _codex_call(
        "request_user_input",
        "q1",
        {"questions": [{"question": "Activate now?"}]},
    )
    snapshot = reduce_events("codex", "fixture-codex", (old, new, question))
    assert [(task.text, task.status) for task in snapshot.tasks.items] == [
        ("model", "completed"),
        ("tests", "in_progress"),
    ]
    assert [claim.text for claim in snapshot.questions.items] == ["Activate now?"]

    resolved = reduce_events("codex", "fixture-codex", (old, new, question, _codex_output("q1")))
    assert resolved.questions.items == ()


def test_codex_injected_or_developer_text_never_becomes_request_and_punctuation_is_not_question():
    snapshot = reduce_events(
        "codex",
        "fixture-codex",
        (
            _codex_message("user", "implement the renderer"),
            _codex_message("developer", "injected harness instructions"),
            _codex_message("assistant", "Could this be a question?"),
        ),
    )
    assert snapshot.request.text == "implement the renderer"
    assert snapshot.questions.items == ()


def test_compaction_is_structural_and_does_not_promote_summary_text():
    claude = _claude(
        {"type": "system", "subtype": "compact_boundary", "content": "invented summary"},
        "compaction",
    )
    codex = _codex("compacted", {"replacement_history": "invented summary"}, "compaction")
    for source, event in (("claude-code", claude), ("codex", codex)):
        snapshot = reduce_events(source, "fixture", (event,))
        assert snapshot.compaction.text == "Context compacted"
        assert "invented summary" not in " ".join(item.text for item in snapshot.recent)


def test_codex_nonzero_command_is_exact_turn_evidence_not_agent_failure_or_blocker():
    failure = _codex("event_msg", {"type": "exec_command_end", "exit_code": 7}, "error")
    snapshot = reduce_events("codex", "fixture-codex", (failure,))
    assert snapshot.command_result.text == "Command exited 7 this turn"
    assert "agent" not in snapshot.command_result.text.lower()
    assert "block" not in snapshot.command_result.text.lower()

    boundary = _codex("event_msg", {"type": "task_complete"}, "turn_boundary")
    after_boundary = reduce_events("codex", "fixture-codex", (failure, boundary))
    assert after_boundary.command_result.text is None
    assert after_boundary.turn.text == "Turn returned to human"


def test_malformed_complete_record_fails_semantic_provenance_closed():
    snapshot = reduce_events(
        "codex",
        "fixture-codex",
        (_codex_message("user", "otherwise visible"),),
        malformed_records=1,
    )
    assert snapshot.source_integrity is Provenance.UNKNOWN
    assert snapshot.request.provenance is Provenance.UNKNOWN
    assert snapshot.tasks.provenance is Provenance.UNKNOWN
    assert snapshot.questions.provenance is Provenance.UNKNOWN


def test_claude_tool_error_is_current_turn_signal_and_success_clears_it():
    failed = reduce_events(
        "claude-code",
        "fixture/session",
        (_claude_result("call", "permission denied", error=True),),
    )
    assert failed.command_result.text == "permission denied"
    cleared = reduce_events(
        "claude-code",
        "fixture/session",
        (
            _claude_result("call", "permission denied", error=True),
            _claude_result("retry", "ok"),
        ),
    )
    assert cleared.command_result.text is None


def test_unsupported_structured_plan_is_unknown_not_an_empty_plan():
    claude = reduce_events(
        "claude-code", "fixture/session", (_claude_tool("TodoWrite", "p", {"todos": "bad"}),)
    )
    codex = reduce_events("codex", "fixture-codex", (_codex_call("update_plan", "p", "{"),))
    assert claude.tasks.provenance is Provenance.UNKNOWN
    assert codex.tasks.provenance is Provenance.UNKNOWN


def test_incremental_batches_equal_full_replay_and_replacement_discards_old_state():
    events = (
        _codex_message("user", "first request"),
        _codex_call("update_plan", "p", {"plan": [{"step": "one", "status": "in_progress"}]}),
        _codex_call("request_user_input", "q", {"questions": [{"question": "Continue?"}]}),
        _codex_output("q"),
        _codex_message("user", "second request"),
    )
    full = reduce_events("codex", "fixture-codex", events)
    incremental = SummaryReducer("codex", "fixture-codex")
    incremental.feed(events[:2])
    incremental.feed(events[2:4])
    actual = incremental.feed(events[4:])
    assert actual == full
    assert not hasattr(incremental, "_events")

    replaced = incremental.feed((_codex_message("user", "replacement"),), replace=True)
    assert replaced.request.text == "replacement"
    assert replaced.tasks.provenance is Provenance.UNKNOWN


def test_cross_batch_question_correlation_and_plan_supersession_are_preserved():
    reducer = SummaryReducer("claude-code", "fixture/session")
    question = _claude_tool(
        "AskUserQuestion", "q1", {"questions": [{"question": "Which profile?"}]}
    )
    old_plan = _claude_tool("TodoWrite", "p1", {"todos": [{"content": "old", "status": "pending"}]})
    first = reducer.feed((question, old_plan))
    assert [claim.text for claim in first.questions.items] == ["Which profile?"]

    new_plan = _claude_tool(
        "TodoWrite", "p2", {"todos": [{"content": "new", "status": "in_progress"}]}
    )
    second = reducer.feed((_claude_result("q1"), new_plan))
    assert second.questions.items == ()
    assert [(task.text, task.status) for task in second.tasks.items] == [("new", "in_progress")]


def test_malformed_incremental_state_persists_until_explicit_replacement():
    reducer = SummaryReducer("codex", "fixture-codex")
    first = reducer.feed((_codex_message("user", "first"),), malformed_records=1)
    assert first.source_integrity is Provenance.UNKNOWN

    still_unknown = reducer.feed((_codex_message("user", "second"),))
    assert still_unknown.source_integrity is Provenance.UNKNOWN
    assert still_unknown.request.provenance is Provenance.UNKNOWN

    rebuilt = reducer.feed((_codex_message("user", "replacement"),), replace=True)
    assert rebuilt.source_integrity is Provenance.EXACT
    assert rebuilt.request.text == "replacement"


def test_control_sequences_are_removed_and_all_display_text_is_bounded():
    raw = "\x1b[31mred\x1b[0m\x00\n" + "x" * 500
    rendered = sanitize_text(raw)
    assert rendered.startswith("red ")
    assert "\x1b" not in rendered
    assert "\x00" not in rendered
    assert len(rendered) <= DISPLAY_TEXT_LIMIT

    snapshot = reduce_events("claude-code", "fixture/session", (_claude_user(raw),))
    assert snapshot.request.text == rendered
    assert all(len(item.text) <= DISPLAY_TEXT_LIMIT for item in snapshot.recent)


def test_summary_package_has_no_model_or_current_state_dependency():
    root = Path(__file__).resolve().parents[1] / "palaver" / "summary"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "ModelClient" not in source
    assert "current_state" not in source
    assert "palaver.extract.client" not in source
