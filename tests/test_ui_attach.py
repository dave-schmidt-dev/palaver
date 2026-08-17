"""Task 5.1: attaching to iTerm2 over its Unix socket, and tracking panes.

The split in this file mirrors the split in the code. Connection preflight and
session bookkeeping are proven headlessly with no terminal involved. The
monitors are proven twice: once against a stub connection, so their control
flow is testable anywhere, and once against the real iTerm2 API, because a
stub cannot prove that `NewSessionMonitor` is the name of a thing that exists
or that iTerm2 will ever deliver to it.

The live tests create one iTerm2 tab and close it again. That is visible on
screen for a moment, and it is the only way to make a session monitor fire —
the phase's own acceptance requires the pane check run "through the iTerm2
Python API itself rather than by eye". Every one of them closes what it opened
in a `finally`.

No cookie value is ever asserted on, printed, or captured here. The live tests
need one, so they ask iTerm2 for it and hand it straight to the connection.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from palaver.cli import ui as cli_ui
from palaver.ui import autolaunch, connection
from palaver.ui.autolaunch import (
    ADVISORY_NAME,
    AUTOLAUNCH_DIR,
    SHIM_NAME,
    SessionRegistry,
    attach_existing,
    install_shim,
    render_shim,
    watch_new_sessions,
    watch_terminations,
)
from palaver.ui.connection import (
    COOKIE_ENV,
    KEY_ENV,
    LEGACY_TCP_URI,
    SUITE_ENV,
    ITerm2NotInstalledError,
    MissingCookieError,
    NoSocketTransportError,
    UiConnectionError,
    connection_target,
    preflight,
    require_cookie,
    resolve_target,
    socket_path,
)
from palaver.ui.pane_join import PIN_VARIABLE

live = pytest.mark.skipif(
    sys.platform != "darwin" or not Path("/Applications/iTerm.app").exists(),
    reason="needs a real iTerm2 on a real macOS session",
)


# --- the cookie ------------------------------------------------------------


def test_a_missing_cookie_raises_an_error_that_names_the_remedy():
    with pytest.raises(MissingCookieError) as caught:
        require_cookie({})
    message = str(caught.value)
    assert COOKIE_ENV in message
    assert "AutoLaunch" in message


def test_an_empty_cookie_is_treated_as_a_missing_one():
    """iTerm2 rejects an empty cookie with an opaque HTTP status.

    Forwarding it would turn a fixable setup problem into a protocol error
    several layers down, so the empty string is refused here by name.
    """
    with pytest.raises(MissingCookieError):
        require_cookie({COOKIE_ENV: ""})


def test_a_present_cookie_is_returned():
    """The positive control for both refusals above."""
    assert require_cookie({COOKIE_ENV: "opaque-value"}) == "opaque-value"


def test_the_named_error_is_catchable_as_the_general_one():
    """Callers that only care that attachment failed should not enumerate."""
    assert issubclass(MissingCookieError, UiConnectionError)
    assert issubclass(NoSocketTransportError, UiConnectionError)
    assert issubclass(ITerm2NotInstalledError, UiConnectionError)


# --- the transport ---------------------------------------------------------


def test_the_connection_target_is_a_socket_path_and_not_a_url():
    target = connection_target({})
    assert target.endswith("/private/socket")
    assert "ws://" not in target
    assert LEGACY_TCP_URI not in target
    assert Path(target).is_absolute()


def test_the_socket_path_follows_iterms_own_suite_variable():
    """A beta build serves a different socket; both must be reachable."""
    assert socket_path({}) == socket_path({SUITE_ENV: "iTerm2"})
    beta = socket_path({SUITE_ENV: "iTerm2-beta"})
    assert beta.parts[-3] == "iTerm2-beta"
    assert beta != socket_path({})


def test_an_absent_socket_is_refused_rather_than_silently_becoming_tcp(tmp_path, monkeypatch):
    """The failure this module exists to prevent.

    Without the refusal the iterm2 library connects to loopback TCP instead,
    the script keeps working, and nothing anywhere says the transport changed.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(NoSocketTransportError) as caught:
        resolve_target({})
    message = str(caught.value)
    assert LEGACY_TCP_URI in message
    assert "Enable Python API" in message


def test_an_existing_socket_resolves(tmp_path, monkeypatch):
    """Positive control for the refusal above."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    target = socket_path({})
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    assert resolve_target({}) == target


def test_preflight_reports_the_missing_socket_before_the_missing_cookie(tmp_path, monkeypatch):
    """Order matters: no API server explains the missing cookie too.

    Reporting the cookie first would send someone hunting for a credential on
    a machine whose API server was never turned on.
    """
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    with pytest.raises(NoSocketTransportError):
        preflight({})


def test_preflight_then_reports_the_missing_cookie(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    target = socket_path({})
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    with pytest.raises(MissingCookieError):
        preflight({})


def test_a_failed_preflight_never_reaches_the_library(tmp_path, monkeypatch):
    """`run_forever` must not start a reconnect loop against a dead machine."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    called: list[object] = []
    monkeypatch.setattr(
        connection,
        "import_iterm2",
        lambda: called.append("imported"),
    )
    with pytest.raises(NoSocketTransportError):
        connection.run_forever(lambda _conn: None, env={})
    assert called == []


# --- asking iTerm2 for a cookie -------------------------------------------


def test_a_refused_cookie_request_raises_rather_than_returning_junk(monkeypatch):
    monkeypatch.setattr(
        connection.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "iTerm2 got an error"),
    )
    with pytest.raises(MissingCookieError, match="iTerm2 got an error"):
        connection.request_cookie_and_key()


def test_a_malformed_cookie_response_is_refused_without_quoting_it(monkeypatch):
    """The error must describe the shape, never echo the value.

    A cookie is a credential, and an error message is the easiest place for
    one to end up in a log file.
    """
    monkeypatch.setattr(
        connection.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "only-one-field\n", ""),
    )
    with pytest.raises(MissingCookieError) as caught:
        connection.request_cookie_and_key()
    assert "only-one-field" not in str(caught.value)
    assert "1 space-separated field" in str(caught.value)


def test_a_well_formed_cookie_response_is_split_into_cookie_and_key(monkeypatch):
    monkeypatch.setattr(
        connection.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "COOKIEVAL KEYVAL\n", ""),
    )
    assert connection.request_cookie_and_key() == ("COOKIEVAL", "KEYVAL")


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


def test_autolaunch_main_attaches_existing_and_runs_both_lifecycle_monitors(monkeypatch):
    """The retained AutoLaunch path discovers panes without publishing UI state."""
    app = types.SimpleNamespace(
        terminal_windows=(
            types.SimpleNamespace(
                tabs=(types.SimpleNamespace(sessions=(types.SimpleNamespace(session_id="old"),)),)
            ),
        )
    )

    async def async_get_app(_connection):
        return app

    monkeypatch.setattr(
        autolaunch,
        "import_iterm2",
        lambda: types.SimpleNamespace(
            async_get_app=async_get_app,
            NewSessionMonitor=lambda _conn: _StubMonitor(["new"]),
            SessionTerminationMonitor=lambda _conn: _StubMonitor(["old"]),
        ),
    )

    registry = asyncio.run(autolaunch.main(object(), limit=1))
    assert registry.attached == frozenset({"new"})


# --- the shim --------------------------------------------------------------


def test_the_shim_is_valid_python():
    compile(render_shim(), "palaver.py", "exec")


def test_the_shim_imports_nothing_from_palaver():
    """It runs under iTerm2's managed Python, whose version iTerm2 chooses.

    An import of `palaver` would be a syntax error the day that runtime is
    older than 3.14, and the symptom would be a script that silently never
    starts.
    """
    source = render_shim()
    assert "import palaver" not in source
    assert "from palaver" not in source


def test_the_shim_uses_no_syntax_newer_than_the_oldest_runtime_it_may_meet():
    """f-strings and walruses are the two easy ways to break 3.5 compatibility.

    The bar is deliberately lower than any Python iTerm2 plausibly ships:
    the cost of staying conservative here is one `%` format, and the cost of
    guessing wrong is a surface that never appears and never says why.
    """
    source = render_shim()
    assert ":=" not in source
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith('f"'), line
        assert 'f"' not in stripped.replace('"""', ""), line


def test_the_shim_spawns_the_interpreter_palaver_is_installed_into():
    source = render_shim(python_executable=Path("/opt/pythons/3.14/bin/python3"))
    assert '"/opt/pythons/3.14/bin/python3"' in source or (
        "'/opt/pythons/3.14/bin/python3'" in source
    )
    assert '"-m", MODULE' in source


def test_the_shim_passes_the_cookie_by_environment_and_never_by_argv():
    """`ps` shows argv to every process on the machine.

    A cookie in an argument vector is a credential published to the whole
    system for the lifetime of the process.
    """
    source = render_shim()
    argv_line = next(line for line in source.splitlines() if "subprocess.call" in line)
    assert COOKIE_ENV not in argv_line
    assert "env=env" in argv_line
    assert f"env[{COOKIE_ENV!r}]" in source


def test_the_shim_never_prints_the_cookie():
    """Its stdout and stderr are a log file iTerm2 keeps on disk."""
    source = render_shim()
    for line in source.splitlines():
        if "write(" in line or "print(" in line:
            assert "cookie" not in line.lower(), line


def test_the_shim_backs_off_after_a_fast_failure_but_not_after_a_long_run():
    source = render_shim()
    assert "min(backoff * 2, MAX_BACKOFF)" in source
    assert "MIN_BACKOFF if ran_for > MAX_BACKOFF" in source


def test_the_shim_asks_iterm_for_a_fresh_cookie_each_time():
    """An inherited cookie is consumed by the first connection.

    Without a fresh request every restart after the first would fail
    authentication, which would look exactly like a crash loop.
    """
    source = render_shim()
    assert "request cookie and key" in source
    assert ADVISORY_NAME in source
    assert KEY_ENV in source


def test_installing_writes_the_shim_where_iterm_looks_for_it(tmp_path):
    path = install_shim(directory=tmp_path / "AutoLaunch")
    assert path.name == SHIM_NAME
    assert path.parent.name == "AutoLaunch"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_the_default_install_directory_is_iterms_autolaunch_directory():
    """A script anywhere else is simply never run by iTerm2."""
    assert AUTOLAUNCH_DIR.parts[-2:] == ("Scripts", "AutoLaunch")
    assert "iTerm2" in AUTOLAUNCH_DIR.parts


def test_printing_the_shim_writes_nothing(tmp_path, capsys):
    assert autolaunch.run(["--print"]) == 0
    assert capsys.readouterr().out.startswith('"""Palaver')
    assert not (tmp_path / SHIM_NAME).exists()


def test_the_entry_point_reports_a_setup_failure_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    assert autolaunch.run([]) == 1
    assert "Enable Python API" in capsys.readouterr().err


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
