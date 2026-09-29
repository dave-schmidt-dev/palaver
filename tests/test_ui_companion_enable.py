"""Explicit enable/disable and the rollover gate that blocks re-enabling mid-restart."""

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
from tests._companion_support import controller, paired_app


def test_explicit_disable_closes_companion_and_enable_clears_suppression(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", tmp_path / "state.json")}
    assert asyncio.run(ctl.set_enabled(app, "agent", enabled=False))
    assert agent.vars[DISABLED_VARIABLE] is True
    assert summary.close_calls == [{"force": True}]
    assert asyncio.run(ctl.set_enabled(app, "agent", enabled=True))
    assert agent.vars[DISABLED_VARIABLE] is False
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""


def test_disabled_companion_stays_absent_across_reconciliation(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars.update(
        {
            DISABLED_VARIABLE: True,
            COMPANION_SESSION_VARIABLE: "summary",
        }
    )
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))
    assert result.pairs == ()
    assert result.created == ()
    assert summary.close_calls == [{"force": True}]

    app.tab.sessions.remove(summary)
    later = asyncio.run(ctl.reconcile(app))
    assert later.created == ()
    assert agent.split_calls == []


def test_stuck_rollover_requires_disable_cleanup_before_enable(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary-old"
    summary.session_id = "summary-new"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    state_path = companion.opaque_state_path(tmp_path, "agent")
    state_path.write_text("state", encoding="utf-8")
    ctl, _ = controller(tmp_path, detector=lambda *_args, **_kwargs: None)
    ctl._rollover_pending["agent"] = "summary-old"

    assert not asyncio.run(ctl.set_enabled(app, "agent", enabled=True))
    assert summary.close_calls == []
    assert ctl._rollover_pending == {"agent": "summary-old"}

    assert asyncio.run(ctl.set_enabled(app, "agent", enabled=False))
    assert summary.close_calls == [{"force": True}]
    assert agent.vars[DISABLED_VARIABLE] is True
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert ctl._rollover_pending == {}
    assert not state_path.exists()
    assert agent.close_calls == []
    assert agent.sent_text == []

    app.tab.sessions.remove(summary)
    assert asyncio.run(ctl.set_enabled(app, "agent", enabled=True))
    assert agent.vars[DISABLED_VARIABLE] is False
    assert agent.split_calls == []


def test_enable_rejects_companion_and_ordinary_shell_targets(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path, detector=lambda *_args, **_kwargs: None)
    assert not asyncio.run(ctl.set_enabled(app, "summary", enabled=True))
    assert not asyncio.run(ctl.set_enabled(app, "agent", enabled=True))
    assert agent.vars == {}
    assert summary.vars == {
        ROLE_VARIABLE: COMPANION_ROLE,
        AGENT_SESSION_VARIABLE: "agent",
    }


def test_known_pair_owner_remains_a_valid_target_after_process_exit(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    ctl, _ = controller(tmp_path, detector=lambda *_args, **_kwargs: None)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", tmp_path / "state")}
    assert asyncio.run(ctl.set_enabled(app, "agent", enabled=False))
    assert summary.close_calls == [{"force": True}]
