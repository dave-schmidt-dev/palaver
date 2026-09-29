"""
The pane registry's own bookkeeping, and attaching to panes that are already open when
the registry starts up.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from palaver.ui.autolaunch import (
    SessionRegistry,
    attach_existing,
)

# --- the registry ----------------------------------------------------------


def test_attaching_twice_reports_the_second_time_as_not_new():
    """`NewSessionMonitor` can report a pane that startup already swept."""
    registry = SessionRegistry()
    assert registry.attach("a") is True
    assert registry.attach("a") is False
    assert len(registry) == 1


def test_detaching_an_unattached_pane_is_not_an_error():
    """Termination fires for every pane, including ones Palaver never had."""
    registry = SessionRegistry(["a"])
    assert registry.detach("b") is False
    assert registry.detach("a") is True
    assert len(registry) == 0


def test_an_unnamed_pane_is_refused_rather_than_stored():
    """iTerm2 returns None for a session that vanished mid-query.

    An empty entry would never match a termination, so the registry would
    leak one slot per race, forever.
    """
    registry = SessionRegistry()
    with pytest.raises(ValueError, match="session id is required"):
        registry.attach("")


def test_the_attached_snapshot_cannot_mutate_the_registry():
    registry = SessionRegistry(["a"])
    snapshot = registry.attached
    registry.attach("b")
    assert snapshot == frozenset({"a"})


# --- attaching to what already exists --------------------------------------


def _fake_app(session_ids):
    """Build the minimum `iterm2.App` shape `attach_existing` walks."""
    sessions = [types.SimpleNamespace(session_id=sid) for sid in session_ids]
    tab = types.SimpleNamespace(sessions=sessions)
    window = types.SimpleNamespace(tabs=[tab])
    return types.SimpleNamespace(terminal_windows=[window])


def test_startup_attaches_to_panes_that_are_already_open():
    """A monitor reports only what happens next; the rest is this."""
    registry = SessionRegistry()
    hooked: list[str] = []
    count = asyncio.run(attach_existing(_fake_app(["a", "b"]), registry, on_attach=hooked.append))
    assert count == 2
    assert registry.attached == frozenset({"a", "b"})
    assert hooked == ["a", "b"]


def test_startup_does_not_re_run_the_hook_for_an_already_attached_pane():
    """Companion-pane setup may perform asynchronous work in this hook."""
    registry = SessionRegistry(["a"])
    hooked: list[str] = []
    count = asyncio.run(attach_existing(_fake_app(["a", "b"]), registry, on_attach=hooked.append))
    assert count == 1
    assert hooked == ["b"]


def test_startup_awaits_an_async_hook():
    registry = SessionRegistry()
    hooked: list[str] = []

    async def hook(session_id):
        await asyncio.sleep(0)
        hooked.append(session_id)

    asyncio.run(attach_existing(_fake_app(["a"]), registry, on_attach=hook))
    assert hooked == ["a"]


def test_a_pane_that_vanished_mid_sweep_is_skipped_not_crashed_on():
    registry = SessionRegistry()
    count = asyncio.run(attach_existing(_fake_app([None, "b"]), registry))
    assert count == 1
    assert registry.attached == frozenset({"b"})
