"""Companion lifecycle tests use inert iTerm-shaped objects only."""

from __future__ import annotations

import asyncio
import threading
import types
from pathlib import Path

import pytest

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
        self.split_joins_tab = True
        self.window = None

    async def async_set_variable(self, name, value):
        self.vars[name] = value

    async def async_split_pane(self, **kwargs):
        self.split_calls.append(kwargs)
        if self.split_error:
            raise self.split_error
        assert self.split_result is not None
        # iTerm divides the split pane's rows evenly between it and the new one.
        half = self.grid_size.height // 2
        self.split_result.grid_size.width = self.grid_size.width
        self.split_result.grid_size.height = half
        self.grid_size.height -= half
        if not self.split_joins_tab:
            return self.split_result
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
        self.tab_id = "tab-1"
        self.layout_updates = 0
        self.on_update_layout = None

    async def async_update_layout(self):
        self.layout_updates += 1
        if self.on_update_layout is not None:
            self.on_update_layout()


class RebuiltTab:
    """What an app refresh leaves behind: a new object with the same tab id."""

    def __init__(self, tab):
        self._tab = tab
        self.tab_id = tab.tab_id

    @property
    def sessions(self):
        return self._tab.sessions

    @property
    def current_session(self):
        return self._tab.current_session

    async def async_update_layout(self):
        await self._tab.async_update_layout()


class FakeFrame:
    def __init__(self, x, y, width, height):
        self.origin = types.SimpleNamespace(x=x, y=y)
        self.size = types.SimpleNamespace(width=width, height=height)


class FakeWindow:
    def __init__(self, frame, tabs=()):
        self.frame = frame
        self.tabs = list(tabs)
        self.set_frames = []
        self.set_error = None
        self.get_error = None

    def move_to(self, frame):
        """Stand in for iTerm resizing the window behind Palaver's back."""
        self.frame = frame

    async def async_get_frame(self):
        if self.get_error is not None:
            raise self.get_error
        return self.frame

    async def async_set_frame(self, frame):
        if self.set_error is not None:
            raise self.set_error
        self.set_frames.append(frame)
        self.frame = frame


class FakeApp:
    def __init__(self, tab):
        self.tab = tab
        # Sizing a companion is refused outright without a window frame to put
        # back, so the default fake models one, as iTerm always does.
        self.window = FakeWindow(FakeFrame(0, 0, 1400, 900), [tab])
        self.terminal_windows = [self.window]
        for session in tab.sessions:
            session.window = self.window
        self.refresh_calls = 0
        self.refresh_hook = None

    def get_session_by_id(self, session_id):
        return next((item for item in self.tab.sessions if item.session_id == session_id), None)

    async def async_refresh(self):
        self.refresh_calls += 1
        if self.refresh_hook is not None:
            result = self.refresh_hook()
            if asyncio.iscoroutine(result):
                await result


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
    assert summary.preferred_size == (100, 10)
    assert "palaver.ui.companion_render" in agent.split_calls[0]["profile_customizations"]


def test_created_companion_is_sized_without_changing_tab_geometry(tmp_path, monkeypatch):
    """The one layout write redistributes rows; it never resizes the window.

    Regression guard for the horizontal resize. `Tab.async_update_layout` is a
    whole-tab write, and iterm2 caches `preferred_size` when it builds a
    `Session` and never refreshes it, so a pane Palaver first saw at a
    different window width pushed that dead width back at iTerm, which resized
    the window to match. Every preferred size is resynced from live geometry
    first, leaving the agent and its companion dividing the rows they already
    occupy and both tab totals unchanged.
    """
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    neighbour = FakeSession("neighbour", height=30)
    neighbour.vars[ROLE_VARIABLE] = "shell"
    neighbour.preferred_size = (40, 12)
    app.tab.sessions.append(neighbour)
    neighbour.tab = app.tab
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert summary.preferred_size == (100, 10)
    assert agent.preferred_size == (100, 20)
    assert neighbour.preferred_size == (100, 30)
    assert app.tab.layout_updates == 1
    requested = [agent.preferred_size, summary.preferred_size, neighbour.preferred_size]
    live = [(item.grid_size.width, item.grid_size.height) for item in (agent, summary, neighbour)]
    assert [size[0] for size in requested] == [size[0] for size in live]
    assert sum(size[1] for size in requested) == sum(size[1] for size in live)


def test_a_layout_write_that_moves_the_window_puts_the_frame_back(tmp_path, monkeypatch):
    """iTerm owns the outcome of a layout write, so the frame is a hard guard."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    window = app.window
    original = window.frame
    app.tab.on_update_layout = lambda: window.move_to(FakeFrame(0, 0, 2560, 900))
    ctl, _ = controller(tmp_path)

    asyncio.run(ctl.reconcile(app))

    assert window.set_frames == [original]
    assert window.frame is original


def test_the_frame_is_restored_even_when_the_move_is_not_visible_yet(tmp_path, monkeypatch):
    """iTerm shrinks the window after answering, so a read-back races it.

    Measured against a live iTerm: `Tab.async_update_layout` shrinks the window
    by the pane title bars and dividers its protobuf does not describe -- every
    tab in the window, not just this one -- and it does so after replying, so
    reading the frame straight back usually still reports the old one. The
    restore is therefore unconditional rather than conditional on a move this
    process can see.
    """
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    original = app.window.frame
    ctl, _ = controller(tmp_path)

    asyncio.run(ctl.reconcile(app))

    assert app.tab.layout_updates == 1
    assert app.window.set_frames == [original]


def test_an_unreadable_window_frame_leaves_iterms_own_size_alone(tmp_path, monkeypatch):
    """A write that cannot be undone is the bug; iTerm's even split is not."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    app.window.get_error = RuntimeError("no frame")
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert app.tab.layout_updates == 0
    assert summary.preferred_size is None


def test_a_session_the_app_rebuilt_is_matched_to_its_window_by_tab(tmp_path, monkeypatch):
    """The session delegate resolves by identity, which a refresh can break."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    agent.window = None

    # The refresh inside the sizing call rebuilds the tab, so the window ends up
    # holding a different object from the one the metadata read captured.
    def rebuild_the_tab():
        if agent.split_calls:
            app.window.tabs = [RebuiltTab(app.tab)]

    app.refresh_hook = rebuild_the_tab
    original = app.window.frame
    ctl, _ = controller(tmp_path)

    asyncio.run(ctl.reconcile(app))

    assert app.tab.layout_updates == 1
    assert app.window.set_frames == [original]


def test_a_window_that_refuses_a_frame_set_still_yields_a_companion(tmp_path, monkeypatch):
    """A fullscreen window rejects `async_set_frame`; the pair still stands."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    window = app.window
    window.set_error = RuntimeError("fullscreen")
    app.tab.on_update_layout = lambda: window.move_to(FakeFrame(0, 0, 2560, 900))
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert window.set_frames == []


def test_a_split_missing_from_the_tab_tree_is_left_at_iterms_own_size(tmp_path, monkeypatch):
    """Describing a tab that does not exist yet is how a window gets resized."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    agent.split_joins_tab = False
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert app.tab.layout_updates == 0
    assert summary.preferred_size is None
    assert agent.preferred_size is None


def test_a_split_that_lands_late_is_sized_on_a_later_refresh(tmp_path, monkeypatch):
    """Giving up on the first refresh leaves the companion at half the pane."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    agent.split_joins_tab = False
    ctl, _ = controller(tmp_path)
    skipped = []

    def land_the_split():
        if not agent.split_calls or summary in app.tab.sessions:
            return
        if not skipped:
            skipped.append(True)
            return
        app.tab.sessions.append(summary)
        summary.tab = app.tab

    app.refresh_hook = land_the_split

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert app.tab.layout_updates == 1
    assert summary.preferred_size == (100, companion.SUMMARY_ROWS)
    assert agent.preferred_size == (100, 20)


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


@pytest.mark.parametrize("height", [6, 10])
def test_valid_pair_is_reused_without_resize_or_focus(tmp_path, monkeypatch, height):
    """Reuse never mutates layout, whatever height the companion is at.

    Regression guard. Resizing here ran inside the handler for the very
    layout-change event a resize raises, and `preferred_size` is advisory, so
    a companion that never landed exactly on `SUMMARY_ROWS` resized the window
    without end. `SUMMARY_ROWS` is applied once, at creation, and nowhere else.
    """
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app(focus_companion=False)
    summary.grid_size.height = height
    app.tab.sessions.append(summary)
    summary.tab = app.tab
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ()
    assert result.pairs[0].companion_id == "summary"
    assert summary.preferred_size is None
    assert app.tab.layout_updates == 0
    assert agent.activate_calls == []


def test_repeated_reconciles_never_mutate_layout(tmp_path, monkeypatch):
    """The layout monitor reconciles on every layout change; reuse must be a
    fixed point, or the two feed each other forever."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app(focus_companion=False)
    summary.grid_size.height = 6
    app.tab.sessions.append(summary)
    summary.tab = app.tab
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    for _ in range(5):
        asyncio.run(ctl.reconcile(app))

    assert app.tab.layout_updates == 0
    assert summary.preferred_size is None


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
    app, agent, _summary = paired_app(height=10)
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


def test_cancellation_after_split_cleans_partial_companion_and_state(tmp_path):
    started = asyncio.Event()

    class BlockingCompanion(FakeSession):
        async def async_set_variable(self, name, value):
            started.set()
            await asyncio.Event().wait()

    agent = FakeSession("agent")
    summary = BlockingCompanion("summary", job_pid=20)
    agent.split_result = summary
    tab = FakeTab([agent])

    def seed(path, *_args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("seed", encoding="utf-8")

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        write_initial_state=seed,
        process_detector=detected,
        transcript_joiner=lambda *_args, **_kwargs: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )

    async def drive():
        task = asyncio.create_task(ctl.reconcile(FakeApp(tab)))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert summary.close_calls == [{"force": True}]
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert list(tmp_path.glob("*.json")) == []


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


def test_initial_state_failure_is_guarded_per_pane_and_next_agent_creates(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    bad = FakeSession("bad")
    good = FakeSession("good")
    summary = FakeSession("good-summary", job_pid=20)
    good.split_result = summary
    tab = FakeTab([bad, good])
    calls = []

    def seed(path, *_args):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("state directory unavailable")

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        write_initial_state=seed,
        process_detector=detected,
        transcript_joiner=lambda *_args, **_kwargs: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert result.created == ("good-summary",)
    assert "bad" in result.refused
    assert bad.split_calls == []
    assert len(calls) == 2


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


def test_targeted_reconcile_never_creates_an_unrequested_agent(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    first = FakeSession("first")
    first_summary = FakeSession("first-summary", job_pid=20)
    first.split_result = first_summary
    second = FakeSession("second")
    second.split_result = FakeSession("second-summary", job_pid=20)
    tab = FakeTab([first, second])
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(FakeApp(tab), only_agent_id="first"))
    assert result.created == ("first-summary",)
    assert second.split_calls == []


def test_inventory_only_reconcile_does_not_probe_process_table(tmp_path):
    app, _agent, _summary = paired_app()
    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_table_reader=lambda: (_ for _ in ()).throw(AssertionError("process probe")),
    )
    result = asyncio.run(ctl.reconcile(app, create=False))
    assert result.created == ()


def test_blocking_process_probe_runs_off_event_loop_with_visible_progress(tmp_path):
    app, _agent, _summary = paired_app()
    started = threading.Event()
    release = threading.Event()
    statuses = []

    def process_table():
        started.set()
        release.wait(timeout=2)
        return {}

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_detector=lambda *_args, **_kwargs: None,
        process_table_reader=process_table,
        on_status=statuses.append,
    )

    async def drive():
        task = asyncio.create_task(ctl.reconcile(app))
        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        await task

    asyncio.run(drive())
    assert "reading process table for companion reconciliation" in statuses


def test_agent_mutations_are_bounded_to_split_link_and_conditional_focus():
    source = Path(companion.__file__).read_text(encoding="utf-8")
    assert "agent.session.async_send_text" not in source
    assert "agent.session.async_close" not in source
    assert "agent.session.async_set_profile_properties" not in source
    assert source.count(".async_activate(") == 1
    assert "select_tab=False" in source and "order_window_front=False" in source


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


def test_rebuilt_tab_instance_after_split_is_located_and_sized(tmp_path, monkeypatch):
    """When an app refresh replaces the Tab object with a distinct instance, sizing locates it."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()
    original_tab = app.tab

    class DistinctRebuiltTab(FakeTab):
        def __init__(self, tab_id, sessions):
            super().__init__(sessions)
            self.tab_id = tab_id

    def rebuild_tab_distinct():
        if agent.split_calls:
            new_tab = DistinctRebuiltTab(original_tab.tab_id, [agent, summary])
            app.tab = new_tab
            app.window.tabs = [new_tab]

    app.refresh_hook = rebuild_tab_distinct
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ("summary",)
    assert app.tab.layout_updates == 1
    assert summary.preferred_size == (100, companion.SUMMARY_ROWS)
    assert agent.preferred_size == (100, 20)


def test_create_reacquires_live_session_before_splitting(tmp_path, monkeypatch):
    """A stale inventory session is never mutated after the app exposes its replacement."""
    stub_iterm(monkeypatch)
    stale_agent = FakeSession("agent", height=30)
    stale_tab = FakeTab([stale_agent])
    live_agent = FakeSession("agent", height=30)
    summary = FakeSession("summary", job_pid=20)
    live_agent.split_result = summary
    live_tab = FakeTab([live_agent])
    app = FakeApp(live_tab)
    metadata = SessionMetadata(
        session=stale_agent,
        tab=stale_tab,
        pane=PaneVariables("agent", 10, "codex", "/tmp"),
    )
    ctl, _ = controller(tmp_path)

    result = asyncio.run(
        ctl._create(app, metadata, SupportedPaneProcess("agent", 10, "codex", Path("/tmp")), None)
    )

    assert result is not None
    assert stale_agent.split_calls == []
    assert len(live_agent.split_calls) == 1
