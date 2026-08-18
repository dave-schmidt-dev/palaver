"""Companion lifecycle tests use inert iTerm-shaped objects only."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from palaver.ui import companion
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    ROLE_VARIABLE,
    CompanionController,
    SessionMetadata,
)
from palaver.ui.companion_state import JoinState, read_state
from palaver.ui.pane_join import PaneVariables, SupportedPaneProcess


class FakeSession:
    def __init__(self, session_id: str, *, height: int = 30, job_pid: int | None = 10):
        self.session_id = session_id
        self.grid_size = types.SimpleNamespace(width=100, height=height)
        self.preferred_size = None
        self.job_pid = job_pid
        self.vars: dict[str, object] = {}
        self.split_calls = []
        self.activate_calls = []
        self.close_calls = []
        self.sent_text = []
        self.split_result = None
        self.split_error = None
        self.tab = None
        self.make_split_active = True

    async def async_set_variable(self, name, value):
        self.vars[name] = value

    async def async_split_pane(self, **kwargs):
        self.split_calls.append(kwargs)
        if self.split_error:
            raise self.split_error
        assert self.split_result is not None
        if self.tab is not None and self.split_result not in self.tab.sessions:
            self.tab.sessions.append(self.split_result)
            self.split_result.tab = self.tab
            if self.make_split_active:
                self.tab.current_session = self.split_result
        return self.split_result

    async def async_activate(self, **kwargs):
        self.activate_calls.append(kwargs)

    async def async_close(self, **kwargs):
        self.close_calls.append(kwargs)

    async def async_send_text(self, text, **kwargs):
        self.sent_text.append((text, kwargs))

    async def async_restart(self, **kwargs):
        self.restart_calls = getattr(self, "restart_calls", [])
        self.restart_calls.append(kwargs)


class FakeTab:
    def __init__(self, sessions):
        self.sessions = list(sessions)
        for session in self.sessions:
            session.tab = self
        self.current_session = self.sessions[0] if self.sessions else None
        self.layout_updates = 0

    async def async_update_layout(self):
        self.layout_updates += 1


class FakeApp:
    def __init__(self, tab):
        self.tab = tab
        self.terminal_windows = [types.SimpleNamespace(tabs=[tab])]

    def get_session_by_id(self, session_id):
        return next((item for item in self.tab.sessions if item.session_id == session_id), None)


def metadata_reader(session, tab):
    async def read():
        return SessionMetadata(
            session=session,
            tab=tab,
            pane=PaneVariables(
                session.session_id,
                session.job_pid,
                "codex" if session.vars.get(ROLE_VARIABLE) != COMPANION_ROLE else "python",
                "/tmp",
            ),
            role=session.vars.get(ROLE_VARIABLE),
            agent_session=session.vars.get(AGENT_SESSION_VARIABLE) or None,
            companion_session=session.vars.get(COMPANION_SESSION_VARIABLE) or None,
            disabled=session.vars.get(DISABLED_VARIABLE) is True,
        )

    return read()


def detected(variables, **_kwargs):
    if variables.job_name != "codex":
        return None
    return SupportedPaneProcess(variables.pane_id, 10, "codex", Path("/tmp"))


def controller(tmp_path, *, clock=lambda: 0.0, detector=detected, joiner=lambda *_a, **_k: None):
    writes = []
    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        write_initial_state=lambda *args: writes.append(args),
        process_detector=detector,
        transcript_joiner=joiner,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
        clock=clock,
    )
    return ctl, writes


def paired_app(*, height=30, focus_companion=True):
    agent = FakeSession("agent", height=height)
    summary = FakeSession("summary", job_pid=20)
    agent.split_result = summary
    agent.make_split_active = focus_companion
    tab = FakeTab([agent])
    tab.current_session = agent
    return FakeApp(tab), agent, summary


def stub_iterm(monkeypatch):
    monkeypatch.setattr(
        companion,
        "import_iterm2",
        lambda: types.SimpleNamespace(Size=lambda width, height: (width, height)),
    )


def test_companion_profile_is_quiet_bounded_and_persistent(monkeypatch):
    class Profile:
        def __init__(self):
            self.calls = {}

        def __getattr__(self, name):
            if not name.startswith("set_"):
                raise AttributeError(name)

            def setter(value):
                self.calls[name] = value

            return setter

    profile = Profile()
    monkeypatch.setattr(
        companion,
        "import_iterm2",
        lambda: types.SimpleNamespace(LocalWriteOnlyProfile=lambda: profile),
    )
    assert companion.build_companion_profile("render") is profile
    assert profile.calls == {
        "set_use_custom_command": "Yes",
        "set_command": "render",
        "set_close_sessions_on_end": False,
        "set_prompt_before_closing": False,
        "set_unlimited_scrollback": False,
        "set_scrollback_lines": companion.SCROLLBACK_LINES,
        "set_silence_bell": True,
        "set_send_bell_alert": False,
        "set_flashing_bell": False,
        "set_visual_bell": False,
        "set_use_custom_window_title": True,
        "set_custom_window_title": "Palaver",
    }


def test_supported_process_gets_unjoined_companion_above_it(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    ctl, states = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert agent.split_calls == [
        {
            "vertical": False,
            "before": True,
            "profile_customizations": states and companion.companion_command(states[0][0]),
        }
    ]
    assert states[0][1].source == "codex"
    assert states[0][2] is None
    assert summary.vars[ROLE_VARIABLE] == COMPANION_ROLE
    assert summary.vars[AGENT_SESSION_VARIABLE] == "agent"
    assert agent.vars[COMPANION_SESSION_VARIABLE] == "summary"
    assert summary.preferred_size == (100, 6)
    assert "palaver.ui.companion_render" in agent.split_calls[0]["profile_customizations"]


def test_default_state_transport_marks_an_unresolved_join_unjoined(tmp_path):
    path = tmp_path / "opaque.json"
    process = SupportedPaneProcess("agent", 10, "codex", Path("/tmp/example"))
    companion.write_initial_state(path, process, None)
    state = read_state(path)
    assert state.join_state is JoinState.UNJOINED
    assert state.project == "example"
    assert state.source == "codex"


def test_marked_companion_is_never_probed_or_split(tmp_path):
    agent = FakeSession("agent")
    summary = FakeSession("summary", job_pid=20)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, summary])
    seen = []
    ctl, _ = controller(tmp_path, detector=lambda variables, **_k: seen.append(variables.pane_id))

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert len(result.pairs) == 1
    assert seen == []
    assert not agent.split_calls and not summary.split_calls


def test_valid_pair_is_reused_without_resize_or_focus(tmp_path):
    app, agent, summary = paired_app(focus_companion=False)
    app.tab.sessions.append(summary)
    summary.tab = app.tab
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ()
    assert result.pairs[0].companion_id == "summary"
    assert summary.preferred_size is None
    assert agent.activate_calls == []


def test_only_exact_marker_orphan_and_duplicate_are_closed(tmp_path):
    agent = FakeSession("agent")
    keeper = FakeSession("keep", job_pid=20)
    duplicate = FakeSession("duplicate", job_pid=21)
    ordinary = FakeSession("ordinary")
    agent.vars[COMPANION_SESSION_VARIABLE] = "keep"
    for item in (keeper, duplicate):
        item.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, keeper, duplicate, ordinary])
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert result.closed == ("duplicate",)
    assert duplicate.close_calls == [{"force": True}]
    assert ordinary.close_calls == []


def test_one_sided_pair_cleanup_leaves_no_orphan_state(tmp_path):
    agent = FakeSession("agent")
    summary = FakeSession("summary", job_pid=20)
    agent.vars[COMPANION_SESSION_VARIABLE] = "missing-peer"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, summary])
    state = companion.opaque_state_path(tmp_path, "agent")
    state.write_text("stale", encoding="utf-8")
    ctl, _ = controller(tmp_path)

    asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert summary.close_calls == [{"force": True}]
    assert not state.exists()
    assert agent.vars[DISABLED_VARIABLE] is True


def test_small_agent_is_refused_without_split(tmp_path):
    app, agent, _summary = paired_app(height=6)
    ctl, writes = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(app))
    assert result.refused == ("agent",)
    assert agent.split_calls == []
    assert writes == []


def test_focus_restores_only_if_new_companion_remains_active(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, _summary = paired_app(focus_companion=False)
    ctl, _ = controller(tmp_path)
    asyncio.run(ctl.reconcile(app))
    assert agent.activate_calls == []

    app2, agent2, _summary2 = paired_app(focus_companion=True)
    ctl2, _ = controller(tmp_path)
    asyncio.run(ctl2.reconcile(app2))
    assert agent2.activate_calls == [{"select_tab": False, "order_window_front": False}]


def test_partial_creation_failure_closes_only_new_companion(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()

    async def fail(_name, _value):
        raise RuntimeError("variable refused")

    summary.async_set_variable = fail
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(app))
    assert result.pairs == ()
    assert summary.close_calls == [{"force": True}]
    assert agent.close_calls == []


def test_split_failure_removes_the_state_seed(tmp_path):
    app, agent, _summary = paired_app()
    agent.split_error = RuntimeError("cannot split")
    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_detector=detected,
        transcript_joiner=lambda *_a, **_k: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )
    asyncio.run(ctl.reconcile(app))
    assert list(tmp_path.glob("*.json")) == []


def test_user_close_disables_but_manager_close_does_not(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    pair = companion.CompanionPair("agent", "summary", tmp_path / "state.json")
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": pair}
    asyncio.run(ctl.handle_termination(app, "summary"))
    assert agent.vars[DISABLED_VARIABLE] is True

    agent.vars.clear()
    ctl._pairs = {"agent": pair}
    ctl._manager_closing.add("summary")
    asyncio.run(ctl.handle_termination(app, "summary"))
    assert DISABLED_VARIABLE not in agent.vars


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


def test_agent_teardown_closes_only_its_exact_registered_companion(tmp_path):
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    ctl, _ = controller(tmp_path)
    ctl._pairs = {"agent": companion.CompanionPair("agent", "summary", tmp_path / "state.json")}
    asyncio.run(ctl.handle_termination(app, "agent"))
    assert summary.close_calls == [{"force": True}]
    assert agent.close_calls == []


def test_exited_companion_restarts_in_place_and_failure_uses_backoff(tmp_path):
    now = [0.0]
    app, agent, summary = paired_app()
    app.tab.sessions.append(summary)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    summary.job_pid = None
    ctl, _ = controller(tmp_path, clock=lambda: now[0])

    first = asyncio.run(ctl.reconcile(app))
    assert first.created == ()
    assert summary.restart_calls == [{"only_if_exited": True}]
    assert summary.close_calls == []

    async def fail_restart(**_kwargs):
        raise RuntimeError("still exited")

    summary.async_restart = fail_restart
    asyncio.run(ctl.reconcile(app))
    assert ctl._retry_after["agent"] == companion.INITIAL_RESTART_BACKOFF
    asyncio.run(ctl.reconcile(app))
    assert ctl._restart_attempts["agent"] == 1
    now[0] = companion.INITIAL_RESTART_BACKOFF
    asyncio.run(ctl.reconcile(app))
    assert ctl._restart_attempts["agent"] == 2


def test_one_pane_failure_does_not_prevent_another_creation(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    bad = FakeSession("bad")
    bad.split_error = RuntimeError("split failed")
    good = FakeSession("good")
    summary = FakeSession("summary", job_pid=20)
    good.split_result = summary
    tab = FakeTab([bad, good, summary])
    tab.sessions.remove(summary)
    tab.current_session = summary
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(FakeApp(tab)))
    assert result.created == ("summary",)
    assert "bad" in result.refused


def test_agent_mutations_are_bounded_to_split_link_and_conditional_focus():
    source = Path(companion.__file__).read_text(encoding="utf-8")
    assert "agent.session.async_send_text" not in source
    assert "agent.session.async_close" not in source
    assert "agent.session.async_set_profile_properties" not in source
    assert source.count("agent.session.async_activate(") == 1
    assert "select_tab=False" in source and "order_window_front=False" in source


def test_operation_trace_never_closes_or_sends_text_to_agent(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    ctl, _ = controller(tmp_path)
    asyncio.run(ctl.reconcile(app))
    asyncio.run(ctl.handle_termination(app, "agent"))
    assert agent.close_calls == []
    assert agent.sent_text == []
    assert summary.close_calls == [{"force": True}]
