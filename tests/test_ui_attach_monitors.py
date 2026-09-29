"""
The new-session and termination monitors against a stubbed iterm2, including their
debounce and concurrency guarantees and the top-level autolaunch reconcile loop.
"""

from __future__ import annotations

import asyncio
import types

from palaver.ui import autolaunch
from palaver.ui.autolaunch import (
    SessionRegistry,
    watch_layout_changes,
    watch_new_sessions,
    watch_terminations,
)

# --- the monitors, against a stub ------------------------------------------


class _StubMonitor:
    """An async context manager that yields a fixed list of session ids."""

    def __init__(self, ids):
        self._ids = list(ids)
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_exc):
        return False

    async def async_get(self):
        if not self._ids:
            await asyncio.sleep(3600)  # never returns; the limit ends the loop
        return self._ids.pop(0)


def _stub_iterm2(new_ids=(), terminated_ids=()):
    """A module-shaped stub exposing only what these functions touch."""
    module = types.SimpleNamespace()
    module.NewSessionMonitor = lambda _conn: _StubMonitor(new_ids)
    module.SessionTerminationMonitor = lambda _conn: _StubMonitor(terminated_ids)
    return module


def test_new_panes_are_attached_as_they_open(monkeypatch):
    monkeypatch.setattr(autolaunch, "import_iterm2", lambda: _stub_iterm2(new_ids=["a", "b"]))
    registry = SessionRegistry()
    attached = asyncio.run(watch_new_sessions(object(), registry, limit=2))
    assert attached == 2
    assert registry.attached == frozenset({"a", "b"})


def test_a_new_pane_already_attached_at_startup_is_not_counted_twice(monkeypatch):
    monkeypatch.setattr(autolaunch, "import_iterm2", lambda: _stub_iterm2(new_ids=["a"]))
    registry = SessionRegistry(["a"])
    assert asyncio.run(watch_new_sessions(object(), registry, limit=1)) == 0


def test_closed_panes_are_forgotten(monkeypatch):
    monkeypatch.setattr(
        autolaunch, "import_iterm2", lambda: _stub_iterm2(terminated_ids=["a", "ghost"])
    )
    registry = SessionRegistry(["a", "b"])
    detached = asyncio.run(watch_terminations(object(), registry, limit=2))
    assert detached == 1, "a pane Palaver never attached to is an event, not a detach"
    assert registry.attached == frozenset({"b"})


def test_termination_callback_runs_even_when_registry_missed_the_owned_pane(monkeypatch):
    monkeypatch.setattr(
        autolaunch, "import_iterm2", lambda: _stub_iterm2(terminated_ids=["companion"])
    )
    seen = []
    asyncio.run(
        watch_terminations(
            object(), SessionRegistry(), on_detach=lambda pane_id: seen.append(pane_id), limit=1
        )
    )
    assert seen == ["companion"]


def test_layout_burst_is_trailing_edge_debounced(monkeypatch):
    monitor = _StubMonitor([1, 2, 3])
    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(LayoutChangeMonitor=lambda _conn: monitor),
    )
    monkeypatch.setattr(autolaunch, "LAYOUT_DEBOUNCE_SECONDS", 0)
    calls = []
    asyncio.run(
        watch_layout_changes(object(), on_change=lambda: calls.append("reconcile"), limit=3)
    )
    assert calls == ["reconcile"]


def test_layout_event_during_reconcile_schedules_serial_followup(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    class SplitMonitor(_StubMonitor):
        async def async_get(self):
            if self._ids[0] == 2:
                await started.wait()
            return await super().async_get()

    monitor = SplitMonitor([1, 2])
    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(LayoutChangeMonitor=lambda _conn: monitor),
    )
    monkeypatch.setattr(autolaunch, "LAYOUT_DEBOUNCE_SECONDS", 0)
    calls = []

    async def reconcile():
        calls.append("start")
        if len(calls) == 1:
            started.set()
            await release.wait()
        calls.append("end")

    async def drive():
        task = asyncio.create_task(watch_layout_changes(object(), on_change=reconcile, limit=2))
        await started.wait()
        await asyncio.sleep(0)
        release.set()
        await task

    asyncio.run(drive())
    assert calls == ["start", "end", "start", "end"]


def test_layout_watcher_cancellation_awaits_its_debounce_workers(monkeypatch):
    delivered = asyncio.Event()

    class BlockingMonitor(_StubMonitor):
        async def async_get(self):
            if self._ids:
                delivered.set()
                return self._ids.pop(0)
            await asyncio.Event().wait()

    monitor = BlockingMonitor([1])
    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(LayoutChangeMonitor=lambda _conn: monitor),
    )

    async def drive():
        task = asyncio.create_task(watch_layout_changes(object(), on_change=lambda: None))
        await delivered.wait()
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        current = asyncio.current_task()
        assert [
            item for item in asyncio.all_tasks() if item is not current and not item.done()
        ] == []

    asyncio.run(drive())


def test_the_monitors_are_entered_as_context_managers(monkeypatch):
    """Not entering them means iTerm2 is never subscribed and nothing fires."""
    monitor = _StubMonitor(["a"])
    module = types.SimpleNamespace(NewSessionMonitor=lambda _conn: monitor)
    monkeypatch.setattr(autolaunch, "import_iterm2", lambda: module)
    asyncio.run(watch_new_sessions(object(), SessionRegistry(), limit=1))
    assert monitor.entered


def test_the_two_monitors_run_concurrently_rather_than_in_turn(monkeypatch):
    """A pane can close while another opens.

    Sequenced monitors would hold every termination until the next new pane
    happened to appear. This asserts both made progress in one run, which a
    sequential implementation cannot do when the first monitor never
    exhausts.
    """
    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(
            NewSessionMonitor=lambda _conn: _StubMonitor(["new"]),
            SessionTerminationMonitor=lambda _conn: _StubMonitor(["old"]),
            async_get_app=None,
        ),
    )
    registry = SessionRegistry(["old"])

    async def drive():
        await asyncio.wait_for(
            asyncio.gather(
                watch_new_sessions(object(), registry, limit=1),
                watch_terminations(object(), registry, limit=1),
            ),
            timeout=5,
        )

    asyncio.run(drive())
    assert registry.attached == frozenset({"new"})


def test_autolaunch_reconciles_before_all_monitors_and_runs_updates(monkeypatch):
    app = types.SimpleNamespace(
        terminal_windows=(
            types.SimpleNamespace(
                tabs=(types.SimpleNamespace(sessions=(types.SimpleNamespace(session_id="old"),)),)
            ),
        )
    )

    async def async_get_app(_connection):
        return app

    events = []

    class Controller:
        pairs = {}

        async def reconcile(self, _app):
            events.append("reconcile")

        async def handle_termination(self, _app, pane_id):
            events.append(f"terminated:{pane_id}")

    class Updater:
        async def run(self, _app, _controller, *, limit):
            events.append("update")

    class Monitor(_StubMonitor):
        async def __aenter__(self):
            assert events[0] == "reconcile"
            return await super().__aenter__()

    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(
            async_get_app=async_get_app,
            NewSessionMonitor=lambda _conn: Monitor(["new"]),
            SessionTerminationMonitor=lambda _conn: Monitor(["old"]),
            LayoutChangeMonitor=lambda _conn: Monitor(["layout"]),
        ),
    )
    monkeypatch.setattr(autolaunch, "LAYOUT_DEBOUNCE_SECONDS", 0)

    registry = asyncio.run(
        autolaunch.main(
            object(),
            limit=1,
            controller=Controller(),
            updater=Updater(),
            reconcile_cadence=0,
        )
    )
    assert registry.attached == frozenset({"new"})
    assert "update" in events
    assert "terminated:old" in events
    assert events.count("reconcile") >= 2, "periodic self-heal must close startup subscription gaps"
