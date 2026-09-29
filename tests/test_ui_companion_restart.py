"""Agent-exit grace period, pending-rollover protection, and GUID-restart rebinding."""

from __future__ import annotations

import asyncio
from pathlib import Path

from palaver.ui import companion
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    ROLE_VARIABLE,
)
from palaver.ui.pane_join import SupportedPaneProcess
from tests._companion_support import (
    FakeApp,
    FakeSession,
    FakeTab,
    controller,
    paired_app,
    stub_iterm,
)


def _ended_agent_pair(tmp_path, now):
    """A reciprocal pair whose agent pane no longer runs a supported process.

    This is the shape `SessionTerminationMonitor` never reports: `claude`
    exits, its `zsh` and `login` keep the pane's own command alive, so the
    only evidence the session ended is that the pane's process tree no longer
    holds an agent.
    """
    agent = FakeSession("agent")
    summary = FakeSession("summary", job_pid=20)
    agent.split_result = summary
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, summary])
    state = companion.opaque_state_path(tmp_path, "agent")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("state", encoding="utf-8")
    running = [False]

    def detector(variables, **_kwargs):
        if not running[0]:
            return None
        return SupportedPaneProcess(variables.pane_id, 10, "codex", Path("/tmp"))

    ctl, _ = controller(tmp_path, clock=lambda: now[0], detector=detector)
    return ctl, FakeApp(tab), agent, summary, state, running


def test_ended_agent_session_closes_its_companion_after_the_grace_period(tmp_path):
    now = [0.0]
    ctl, app, agent, summary, state, _running = _ended_agent_pair(tmp_path, now)

    asyncio.run(ctl.reconcile(app))
    now[0] += companion.AGENT_EXIT_GRACE
    result = asyncio.run(ctl.reconcile(app))

    assert result.closed == ("summary",)
    assert summary.close_calls == [{"force": True}]
    assert ctl.pairs == {}
    assert not state.exists()
    # The pane itself is the user's, and it must stay eligible: disabling it
    # here would mean the next agent started in it never got a companion.
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert agent.close_calls == []
    assert agent.sent_text == []


def test_a_momentary_missing_agent_process_does_not_close_the_companion(tmp_path):
    now = [0.0]
    ctl, app, agent, summary, state, running = _ended_agent_pair(tmp_path, now)

    asyncio.run(ctl.reconcile(app))
    now[0] += companion.AGENT_EXIT_GRACE - 1.0
    result = asyncio.run(ctl.reconcile(app))
    # An empty process table is what a failed `ps` looks like; one of those
    # inside the grace window must not end the pair.
    assert summary.close_calls == []
    assert len(result.pairs) == 1

    running[0] = True
    now[0] += companion.AGENT_EXIT_GRACE
    result = asyncio.run(ctl.reconcile(app))

    assert summary.close_calls == []
    assert len(result.pairs) == 1
    assert state.exists()
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary"


def test_a_new_agent_in_a_torn_down_pane_is_paired_again(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    now = [0.0]
    ctl, app, agent, summary, _state, running = _ended_agent_pair(tmp_path, now)

    asyncio.run(ctl.reconcile(app))
    now[0] += companion.AGENT_EXIT_GRACE
    asyncio.run(ctl.reconcile(app))
    app.tab.sessions.remove(summary)

    running[0] = True
    replacement = FakeSession("summary-2", job_pid=21)
    agent.split_result = replacement
    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary-2",)
    assert agent.vars.get(DISABLED_VARIABLE) is not True


def test_a_pending_rollover_never_protects_a_companion_whose_agent_is_gone(tmp_path):
    summary = FakeSession("summary", job_pid=20)
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([summary])
    state = companion.opaque_state_path(tmp_path, "agent")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path)
    # Left behind by a renderer restart that the agent pane's close interrupted.
    ctl._rollover_pending = {"agent": "summary"}

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert result.closed == ("summary",)
    assert summary.close_calls == [{"force": True}]
    assert not state.exists()
    assert ctl._rollover_pending == {}


def test_agent_termination_clears_a_pending_rollover(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    state_path = tmp_path / "state.json"
    state_path.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", state_path)}
    ctl._rollover_pending = {"agent": "summary-old"}

    asyncio.run(ctl.handle_termination(app, "agent"))

    assert ctl._rollover_pending == {}
    assert summary.close_calls == [{"force": True}]


def test_exited_companion_restart_rebinds_new_guid_without_layout_mutation(tmp_path):
    now = [0.0]
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    summary.job_pid = None
    replacement = FakeSession("summary-new", job_pid=30)
    replacement.vars.update(summary.vars)

    def roll_guid():
        app.tab.sessions.remove(summary)
        app.tab.sessions.append(replacement)
        replacement.tab = app.tab

    app.refresh_hook = roll_guid
    ctl, _ = controller(tmp_path, clock=lambda: now[0])

    first = asyncio.run(ctl.reconcile(app))
    assert first.created == ()
    assert summary.restart_calls == [{"only_if_exited": True}]
    assert summary.close_calls == []
    assert first.pairs[0].companion_id == "summary-new"
    assert ctl.pairs["agent"].companion_id == "summary-new"
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary-new"
    assert agent.split_calls == []
    assert agent.activate_calls == []
    assert app.refresh_calls == 1


def test_restart_failure_keeps_existing_pair_and_link_under_backoff(tmp_path):
    now = [0.0]
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    summary.job_pid = None
    ctl, _ = controller(tmp_path, clock=lambda: now[0])

    async def fail_restart(**_kwargs):
        raise RuntimeError("still exited")

    summary.async_restart = fail_restart
    first = asyncio.run(ctl.reconcile(app))
    assert first.pairs[0].companion_id == "summary"
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary"
    assert ctl._retry_after["agent"] == companion.INITIAL_RESTART_BACKOFF
    asyncio.run(ctl.reconcile(app))
    assert ctl._restart_attempts["agent"] == 1
    now[0] = companion.INITIAL_RESTART_BACKOFF
    asyncio.run(ctl.reconcile(app))
    assert ctl._restart_attempts["agent"] == 2


def test_restart_refresh_failure_never_publishes_stale_handle_or_splits(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    summary.job_pid = None

    async def fail_refresh():
        app.refresh_calls += 1
        raise RuntimeError("layout unavailable")

    app.async_refresh = fail_refresh
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.pairs == ()
    assert ctl.pairs == {}
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary"
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert agent.split_calls == []
    assert summary.close_calls == []


def test_stale_old_guid_termination_after_rollover_is_harmless(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary-new"
    summary.session_id = "summary-new"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary-new", tmp_path / "state")}

    asyncio.run(ctl.handle_termination(app, "summary-old"))

    assert ctl.pairs["agent"].companion_id == "summary-new"
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary-new"
    assert agent.vars.get(DISABLED_VARIABLE) is not True


def test_termination_rebinds_visible_new_guid_exact_marker(tmp_path):
    app, agent, summary = paired_app()
    summary.session_id = "summary-new"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary-old"
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary-old", tmp_path / "state")}

    asyncio.run(ctl.handle_termination(app, "summary-old"))

    assert ctl.pairs["agent"].companion_id == "summary-new"
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary-new"
    assert agent.vars.get(DISABLED_VARIABLE) is not True
    assert summary.close_calls == []
