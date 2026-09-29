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
        companion.creation,
        "import_iterm2",
        lambda: types.SimpleNamespace(Size=lambda width, height: (width, height)),
    )
