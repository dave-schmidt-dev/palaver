"""
Companion creation sizing and the window-frame restore that keeps a layout write from
resizing the user's window.
"""

from __future__ import annotations

import asyncio

from palaver.ui import companion
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    ROLE_VARIABLE,
)
from tests._companion_support import (
    FakeFrame,
    FakeSession,
    FakeTab,
    controller,
    paired_app,
    stub_iterm,
)


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
    assert summary.preferred_size == (100, 5)
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
    assert summary.preferred_size == (100, 5)
    assert agent.preferred_size == (100, 25)
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
    assert agent.preferred_size == (100, 25)


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
    assert agent.preferred_size == (100, 25)
