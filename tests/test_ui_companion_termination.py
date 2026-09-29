"""
Termination classification: user close vs manager close, agent teardown, and process-end
handling.
"""

from __future__ import annotations

import asyncio

from palaver.ui import companion
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    ROLE_VARIABLE,
)
from tests._companion_support import controller, paired_app, stub_iterm


def test_user_close_disables_but_manager_close_does_not(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    pair = companion.CompanionPair("agent", "summary", tmp_path / "state.json")
    pair.state_path.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": pair}
    app.tab.sessions.remove(summary)
    asyncio.run(ctl.handle_termination(app, "summary"))
    assert agent.vars[DISABLED_VARIABLE] is True
    assert not pair.state_path.exists()

    agent.vars.clear()
    ctl._pairs = {"agent": pair}
    ctl._manager_closing.add("summary")
    asyncio.run(ctl.handle_termination(app, "summary"))
    assert DISABLED_VARIABLE not in agent.vars


def test_agent_teardown_closes_only_its_exact_registered_companion(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", tmp_path / "state.json")}
    app.tab.sessions.remove(agent)
    asyncio.run(ctl.handle_termination(app, "agent"))
    assert summary.close_calls == [{"force": True}]
    assert agent.close_calls == []


def test_agent_process_end_closes_companion_while_leaving_agent_pane_enabled(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    state_path = tmp_path / "state.json"
    state_path.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", state_path)}

    asyncio.run(ctl.handle_termination(app, "agent"))

    assert ctl.pairs == {}
    assert summary.close_calls == [{"force": True}]
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert not state_path.exists()
    assert agent.close_calls == []
    assert agent.sent_text == []


def test_agent_process_end_never_closes_an_unmarked_paired_pane(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    state_path = tmp_path / "state.json"
    state_path.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", state_path)}

    asyncio.run(ctl.handle_termination(app, "agent"))

    assert summary.close_calls == []
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert not state_path.exists()
    assert agent.close_calls == []


def test_visible_process_end_does_not_disable_companion(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    summary.job_pid = None
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", tmp_path / "state")}
    ctl._retry_after["agent"] = 10.0

    asyncio.run(ctl.handle_termination(app, "summary"))

    assert app.refresh_calls == 1
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert ctl.pairs["agent"].companion_id == "summary"


def test_operation_trace_never_closes_or_sends_text_to_agent(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    ctl, _ = controller(tmp_path)
    asyncio.run(ctl.reconcile(app))
    app.tab.sessions.remove(agent)
    asyncio.run(ctl.handle_termination(app, "agent"))
    assert agent.close_calls == []
    assert agent.sent_text == []
    assert summary.close_calls == [{"force": True}]
