"""
The README's own claim that AutoLaunch is opt-in, the CLI pin/companion-toggle actions
exercised without a pin, and (behind PALAVER_RUN_LIVE_ITERM_TESTS) the real socket tests
against a running iTerm2.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import types
from pathlib import Path

import pytest

from palaver.cli import ui as cli_ui
from palaver.ui import connection
from palaver.ui.autolaunch import (
    SessionRegistry,
    attach_existing,
    watch_new_sessions,
    watch_terminations,
)
from palaver.ui.connection import (
    COOKIE_ENV,
    KEY_ENV,
    resolve_target,
)
from palaver.ui.pane_join import PIN_VARIABLE, PanePin
from palaver.ui.pane_join.pin import parse_pin

LIVE_ENV = "PALAVER_RUN_LIVE_ITERM_TESTS"
LIVE_ENABLED = (
    os.environ.get(LIVE_ENV) == "1"
    and sys.platform == "darwin"
    and Path("/Applications/iTerm.app").exists()
)
live = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason=f"set {LIVE_ENV}=1 to create disposable live iTerm tabs",
)


# --- the README setup section ---------------------------------------------


def test_the_readme_tells_the_user_to_turn_the_python_api_on():
    """It is off by default, so a fresh machine has no surface and no error."""
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "Enable Python API" in readme
    assert "AutoLaunch" in readme


class _Writes:
    """Record every variable write, and optionally refuse them."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def __call__(self, session_id, name, value):
        self.calls.append((session_id, name, value))
        if self.fail:
            raise RuntimeError("iTerm2 said no")

    def named(self, name):
        return [call for call in self.calls if call[1] == name]


def test_pin_cli_writes_named_pane_without_focus_or_selection_calls():
    """Pin operations use only the authenticated variable writer."""
    writes = _Writes()

    async def writer(session_id, name, value):
        await writes(session_id, name, value)

    encoded = asyncio.run(
        cli_ui.set_session_pin(writer, "named-pane", source="codex", session_key="rollout-1")
    )
    cleared = asyncio.run(cli_ui.set_session_pin(writer, "named-pane"))
    assert json.loads(encoded) == {"source": "codex", "session_key": "rollout-1"}
    assert cleared == ""
    assert [call[0] for call in writes.calls] == ["named-pane", "named-pane"]
    assert all(call[1] == PIN_VARIABLE for call in writes.calls)
    source = inspect.getsource(cli_ui.set_session_pin)
    assert all(token not in source for token in (".focus(", ".select(", ".activate("))


@pytest.mark.parametrize(
    ("source", "session_key"),
    [("codex", "rollout-1"), ("claude-code", "-Users-dave-proj/session-1")],
)
def test_pin_cli_output_round_trips_through_the_pane_reader(source, session_key):
    """The pin the CLI writes is exactly what the pane's strict parser accepts."""
    writes = _Writes()
    encoded = asyncio.run(
        cli_ui.set_session_pin(writes, "named-pane", source=source, session_key=session_key)
    )
    assert parse_pin(encoded) == PanePin(source=source, session_key=session_key)
    assert writes.calls == [("named-pane", PIN_VARIABLE, encoded)]


@pytest.mark.parametrize(
    ("source", "session_key"),
    [("bogus", "k"), ("codex", ""), ("codex", None), (None, "k"), ("codex", 5)],
)
def test_pin_cli_rejects_unsupported_pins_with_its_own_message(source, session_key):
    with pytest.raises(ValueError, match="pin requires a supported source and non-empty"):
        asyncio.run(
            cli_ui.set_session_pin(_Writes(), "named-pane", source=source, session_key=session_key)
        )


@pytest.mark.parametrize(
    ("flag", "enabled"), [("enable_companion", True), ("disable_companion", False)]
)
def test_companion_cli_action_works_without_a_pin(monkeypatch, flag, enabled):
    calls = []

    class Controller:
        def __init__(self, *_args, **_kwargs):
            pass

        async def reconcile(self, _app, **kwargs):
            calls.append(("reconcile", kwargs))

        async def set_enabled(self, _app, pane_id, *, enabled):
            calls.append(("enabled", pane_id, enabled))
            return True

    async def async_get_app(_connection):
        return object()

    module = types.SimpleNamespace(
        async_get_app=async_get_app,
        run_until_complete=lambda callback: asyncio.run(callback(object())),
    )
    monkeypatch.setattr(cli_ui, "preflight", lambda: None)
    monkeypatch.setattr(cli_ui, "import_iterm2", lambda: module)
    monkeypatch.setattr(cli_ui, "CompanionController", Controller)
    monkeypatch.setattr(cli_ui, "make_metadata_reader", lambda _connection: None)
    args = types.SimpleNamespace(
        session="agent",
        pin=None,
        clear_pin=False,
        enable_companion=flag == "enable_companion",
        disable_companion=flag == "disable_companion",
    )
    assert cli_ui.run(args) == 0
    assert ("enabled", "agent", enabled) in calls
    assert calls[0] == ("reconcile", {"create": False})


# --- live iTerm2 -----------------------------------------------------------


def _live_env():
    """Return an environment carrying a freshly issued cookie.

    The value is never returned to the test body, logged, or asserted on.
    """
    cookie, key = connection.request_cookie_and_key(advisory_name="palaver-test")
    env = dict(os.environ)
    env[COOKIE_ENV] = cookie
    env[KEY_ENV] = key
    return env


def _run_live(body, *, timeout=30.0):
    """Connect to the real iTerm2 and run `body(connection)` once.

    Sets the cookie into `os.environ` because the `iterm2` library reads it
    from there and offers no injection point, then removes it again.
    """
    env = _live_env()
    previous = {name: os.environ.get(name) for name in (COOKIE_ENV, KEY_ENV)}
    os.environ[COOKIE_ENV] = env[COOKIE_ENV]
    os.environ[KEY_ENV] = env[KEY_ENV]
    result = {}

    async def _main(conn):
        result["value"] = await asyncio.wait_for(body(conn), timeout=timeout)

    # Without this the second live test in a run inherits the first test's
    # `App`, still bound to a websocket that has since closed.
    connection.reset_library_state()
    try:
        iterm2 = connection.import_iterm2()
        iterm2.run_until_complete(_main)
    finally:
        connection.reset_library_state()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return result.get("value")


@live
def test_a_second_connection_in_one_process_does_not_reuse_a_dead_app():
    """The library caches `App` per process and invalidates it on disconnect.

    A clean close does not fire that callback, so without `reset_app_cache`
    the second connection here raises `ConnectionClosedError` from inside
    `App.async_refresh` — which is how the two live monitor tests below
    failed on their first run.
    """

    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        return len(app.terminal_windows)

    first = _run_live(body)
    second = _run_live(body)
    assert first >= 1 and second >= 1


@live
def test_the_real_socket_exists_and_preflight_passes_with_a_real_cookie():
    """The done-when's transport check, against the machine rather than a fixture."""
    resolved = resolve_target()
    assert resolved.exists()
    assert resolved.is_socket(), f"{resolved} exists but is not a socket"
    assert "ws://" not in str(resolved)


@live
def test_attaching_live_finds_the_panes_that_are_actually_open():
    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        registry = SessionRegistry()
        count = await attach_existing(app, registry)
        real = sum(len(tab.sessions) for w in app.terminal_windows for tab in w.tabs)
        return count, real, len(registry)

    count, real, tracked = _run_live(body)
    assert real >= 1, "the test itself is running in an iTerm2 pane"
    assert count == real
    assert tracked == real


async def _until(predicate, *, timeout=15.0, interval=0.1):
    """Poll `predicate` until it holds, or fail with a timeout.

    Not `limit=1` on the monitor. iTerm2 is a live application: another pane
    can open or close while a test runs, and a monitor bounded by one *event*
    would spend that event on somebody else's pane and then stop watching.
    That is exactly how the termination test failed on its first full run.
    Bounding on the *condition* instead makes an unrelated event harmless.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


async def _cancel(task):
    """Cancel a watcher and let its monitor unsubscribe cleanly."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@live
def test_a_pane_opened_through_the_api_is_seen_by_the_new_session_monitor():
    """Proves `NewSessionMonitor` is subscribed and iTerm2 delivers to it.

    A stub can prove the loop; only iTerm2 can prove the subscription.
    """

    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        registry = SessionRegistry()
        await attach_existing(app, registry)

        watcher = asyncio.create_task(watch_new_sessions(conn, registry))
        await asyncio.sleep(0.5)  # let the monitor subscribe before the tab opens

        window = app.current_terminal_window
        tab = await window.async_create_tab()
        opened = tab.sessions[0].session_id
        try:
            seen = await _until(lambda: opened in registry)
        finally:
            await _cancel(watcher)
            await tab.async_close(force=True)
        return opened, seen

    opened, seen = _run_live(body, timeout=40.0)
    assert seen, f"the new-session monitor never reported {opened}"


@live
def test_a_pane_closed_through_the_api_is_seen_by_the_termination_monitor():
    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        registry = SessionRegistry()

        window = app.current_terminal_window
        tab = await window.async_create_tab()
        opened = tab.sessions[0].session_id
        registry.attach(opened)

        watcher = asyncio.create_task(watch_terminations(conn, registry))
        await asyncio.sleep(0.5)
        # The control: it is still attached right up until the pane closes,
        # so a registry that had simply never recorded it cannot pass.
        assert opened in registry
        await tab.async_close(force=True)
        try:
            gone = await _until(lambda: opened not in registry)
        finally:
            await _cancel(watcher)
        return opened, gone

    opened, gone = _run_live(body, timeout=40.0)
    assert gone, f"the termination monitor never reported {opened}"
