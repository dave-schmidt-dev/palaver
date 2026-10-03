"""The bounded plan and question collections both summary reducers build.

Pins the cap (`MAX_COLLECTION_ITEMS`) and the "unknown rather than guess"
reasons for Claude Code and Codex, so the two sources cannot drift apart.
"""

from __future__ import annotations

import json

import pytest

from palaver.ingest.adapters.base import Event
from palaver.summary import Provenance, SummaryReducer, reduce_events
from palaver.summary.model import MAX_COLLECTION_ITEMS


def _claude_tool(name: str, tool_input: object, tool_id: str = "t1") -> Event:
    block = {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
    record = {"type": "assistant", "message": {"role": "assistant", "content": [block]}}
    return Event("fixture/session", "message", record)


def _codex_call(name: str, arguments: object, call_id: str = "c1") -> Event:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    payload = {"type": "function_call", "name": name, "call_id": call_id, "arguments": raw}
    return Event("fixture-codex", "function_call", {"type": "response_item", "payload": payload})


def _plan(source: str, items: object):
    if source == "claude-code":
        event = _claude_tool("TodoWrite", {"todos": items})
        return reduce_events(source, "fixture/session", (event,)).tasks
    event = _codex_call("update_plan", {"plan": items})
    return reduce_events(source, "fixture-codex", (event,)).tasks


def _questions(source: str, items: object):
    if source == "claude-code":
        event = _claude_tool("AskUserQuestion", {"questions": items})
        return reduce_events(source, "fixture/session", (event,)).questions
    event = _codex_call("request_user_input", {"questions": items})
    return reduce_events(source, "fixture-codex", (event,)).questions


_PLAN_SHAPES = {
    "claude-code": ("content", "TodoWrite", "TodoWrite input is"),
    "codex": ("step", "update_plan", "update_plan arguments are"),
}


@pytest.mark.parametrize("source", ["claude-code", "codex"])
@pytest.mark.parametrize("count", [MAX_COLLECTION_ITEMS, MAX_COLLECTION_ITEMS + 1])
def test_plan_snapshot_cap_is_inclusive(source, count):
    key, label, _ = _PLAN_SHAPES[source]
    tasks = _plan(source, [{key: f"task {n}", "status": "pending"} for n in range(count)])
    if count == MAX_COLLECTION_ITEMS:
        assert tasks.provenance is Provenance.EXACT
        assert tasks.evidence_kind == label
        assert len(tasks.items) == MAX_COLLECTION_ITEMS
    else:
        assert tasks.provenance is Provenance.UNKNOWN
        assert tasks.items == ()
        assert tasks.reason == f"{label} exceeds bounded task limit"


@pytest.mark.parametrize("source", ["claude-code", "codex"])
def test_plan_snapshot_unknown_reasons_name_the_source_and_field(source):
    key, label, container = _PLAN_SHAPES[source]
    good = {key: "do it", "status": "pending"}
    assert _plan(source, "bad").reason == f"{container} unsupported"
    assert _plan(source, ["bad"]).reason == f"{label} item is unsupported"
    assert _plan(source, [{key: "x"}]).reason == f"{label} item lacks {key} or status"
    assert _plan(source, [{"status": "x"}]).reason == f"{label} item lacks {key} or status"
    ok = _plan(source, [good])
    assert [(item.text, item.status) for item in ok.items] == [("do it", "pending")]
    assert _plan(source, []).provenance is Provenance.EXACT


@pytest.mark.parametrize("source", ["claude-code", "codex"])
@pytest.mark.parametrize("count", [MAX_COLLECTION_ITEMS, MAX_COLLECTION_ITEMS + 1])
def test_question_batch_cap_is_inclusive(source, count):
    questions = _questions(source, [{"question": f"q{n}?"} for n in range(count)])
    if count == MAX_COLLECTION_ITEMS:
        assert questions.provenance is Provenance.EXACT
        assert len(questions.items) == MAX_COLLECTION_ITEMS
    else:
        assert questions.provenance is Provenance.UNKNOWN
        assert questions.items == ()


@pytest.mark.parametrize("source", ["claude-code", "codex"])
@pytest.mark.parametrize("items", ["bad", ["bad"], [{"question": ""}], [{"other": "x"}]])
def test_unsupported_question_batches_are_unknown(source, items):
    assert _questions(source, items).provenance is Provenance.UNKNOWN


@pytest.mark.parametrize("source", ["claude-code", "codex"])
def test_pending_questions_resume_from_an_initial_snapshot(source):
    """A second batch sees the first batch's open questions through `initial`."""
    first_event = (
        _claude_tool("AskUserQuestion", {"questions": [{"question": "a?"}]}, "q1")
        if source == "claude-code"
        else _codex_call("request_user_input", {"questions": [{"question": "a?"}]}, "q1")
    )
    second_event = (
        _claude_tool("AskUserQuestion", {"questions": [{"question": "b?"}]}, "q2")
        if source == "claude-code"
        else _codex_call("request_user_input", {"questions": [{"question": "b?"}]}, "q2")
    )
    key = "fixture/session" if source == "claude-code" else "fixture-codex"
    reducer = SummaryReducer(source, key)
    reducer.feed((first_event,))
    resumed = reducer.feed((second_event,))
    full = reduce_events(source, key, (first_event, second_event))
    assert [claim.text for claim in resumed.questions.items] == ["a?", "b?"]
    assert resumed.questions == full.questions
