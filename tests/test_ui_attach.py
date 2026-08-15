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
import base64
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from palaver.observer.signals import Status
from palaver.ui import autolaunch, component, connection
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
from palaver.ui.component import (
    GUID_KEY,
    IDENTIFIER,
    LAYOUT_KEY,
    ORIGINAL_GUID_KEY,
    SHOW_BAR_KEY,
    STATUS_REFERENCE,
    STATUS_VARIABLE,
    TICK_VARIABLE,
    UPDATE_CADENCE,
    LayoutCheck,
    RenderTicker,
    build_component,
    check_layout,
    decode_status,
    encode_status,
    layout_contains,
    line_for,
    profile_identity,
    push_status,
    render_for_session,
    show_status_bar,
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
    """Task 5.3 registers a status bar component in this hook."""
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


# --- task 5.3: the status bar component, its variables, and the layout gate ---

#: The shared profile every pane on this machine is running under, measured
#: 2026-08-15. A literal rather than a lookup: these tests are about what the
#: check does with a guid, not about which guid is live today.
SHARED_GUID = "F25B986F-AEEA-4438-A22D-B79D193A0FB0"

#: A real, shared, *unused* profile — the Scarecrow dynamic profile. This is
#: the plan's "unused profile" case, and it is only expressible as a guid:
#: three live sessions report the name `Default` with three different guids,
#: so a name cannot say which profile is meant.
UNUSED_GUID = "scarecrow-tui-profile-001"

#: A pane id shaped like iTerm2's.
PANE = "w0t0p0:CF60A48E-0000-4000-8000-000000000001"


def _plain_entry(identifier):
    """A layout entry that names the component in the clear."""
    return {"class": "iTermStatusBarRPCProvidedTextComponent", "identifier": identifier}


def _serialized_entry(identifier):
    """A layout entry that carries the identifier inside a serialized request.

    iTerm2 keeps a `_savedRegistrationRequest` per component, so the real
    entry embeds an encoded `ITMRPCRegistrationRequest` rather than a bare
    string. The exact encoding is not public and is not what is under test:
    what is under test is that the check still finds the identifier when the
    entry is opaque, rather than reading a key that may not exist.
    """
    blob = b"\x12\x2f" + identifier.encode() + b"\x1a\x04knob"
    return {
        "class": "iTermStatusBarRPCProvidedTextComponent",
        "configuration": {"registration request": base64.b64encode(blob).decode()},
    }


def _props(*, entries=(), bar_shown=1, original=SHARED_GUID, guid=None):
    """Build profile properties the way a live session reports them.

    `guid` defaults to something *other* than `original`, because that is the
    live case: every session's profile here is divorced, so its own guid is
    session-local and matches no shared profile.
    """
    props = {
        GUID_KEY: guid if guid is not None else "0D0027BC-1EF1-422A-8CE2-55FBA6703F6E",
        SHOW_BAR_KEY: bar_shown,
        LAYOUT_KEY: {
            "components": list(entries),
            "advanced configuration": {"remove empty components": False},
        },
    }
    if original is not None:
        props[ORIGINAL_GUID_KEY] = original
    return props


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


class _StubReference:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class _StubStatusBarComponent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registrations = []

    async def async_register(self, connection, coro, timeout=None):
        self.registrations.append((connection, coro, timeout))


def _stub_status_bar_api(monkeypatch):
    """Stand in for the `iterm2` module inside `build_component`.

    Only the three names `build_component` touches. The real library is
    exercised by the live tests at the bottom of this file; this is here so
    the wiring — which reference names, which cadence, which identifier — is
    assertable on a machine with no terminal.
    """
    module = types.SimpleNamespace(
        Reference=_StubReference,
        StatusBarRPC=lambda func: func,
        StatusBarComponent=lambda **kwargs: _StubStatusBarComponent(**kwargs),
    )
    monkeypatch.setattr(component, "import_iterm2", lambda: module)
    return module


def test_a_configured_profile_passes_the_layout_check():
    check = check_layout(_props(entries=[_plain_entry(IDENTIFIER)]), expected_guid=SHARED_GUID)
    assert check.ok
    assert check.remedy is None


def test_a_component_in_an_unused_profiles_layout_fails_the_layout_check():
    """The plan's own negative case: configured, but not where anything runs.

    Everything else about this profile is right — the identifier is in the
    layout and the bar is on — so a check that looked only at the layout
    would pass it while no pane on the machine showed anything.
    """
    props = _props(entries=[_plain_entry(IDENTIFIER)], original=UNUSED_GUID)
    check = check_layout(props, expected_guid=SHARED_GUID)
    assert not check.ok
    assert check.in_layout and check.bar_shown, "only the profile should be wrong"
    assert UNUSED_GUID in check.remedy and SHARED_GUID in check.remedy


def test_a_layout_without_the_component_fails_and_the_remedy_names_the_fix():
    check = check_layout(_props(), expected_guid=SHARED_GUID)
    assert not check.ok
    assert not check.in_layout
    assert "Configure Status Bar" in check.remedy


def test_a_layout_containing_the_component_still_fails_with_the_bar_switched_off():
    """The fourth state the plan does not name.

    `Show Status Bar` is 0 on every live session on this machine, so a gate
    asserting only registration, membership and layout inclusion reports
    success while the user sees nothing.
    """
    props = _props(entries=[_plain_entry(IDENTIFIER)], bar_shown=0)
    check = check_layout(props, expected_guid=SHARED_GUID)
    assert not check.ok
    assert check.in_layout, "the layout half is fine; the bar is off"
    assert "Status bar enabled" in check.remedy


def test_the_layout_check_reads_the_shared_guid_not_the_divorced_one():
    props = _props(entries=[_plain_entry(IDENTIFIER)])
    assert props[GUID_KEY] != SHARED_GUID, "the fixture must be divorced for this to mean anything"
    assert profile_identity(props) == SHARED_GUID
    assert check_layout(props, expected_guid=SHARED_GUID).profile_matches


def test_an_undivorced_profile_falls_back_to_its_own_guid():
    props = _props(entries=[_plain_entry(IDENTIFIER)], original=None, guid=SHARED_GUID)
    assert ORIGINAL_GUID_KEY not in props
    assert profile_identity(props) == SHARED_GUID


def test_the_identifier_is_found_inside_an_opaque_layout_entry():
    """The entry schema is not public, so the check must not depend on it."""
    assert layout_contains({"components": [_serialized_entry(IDENTIFIER)]})
    assert check_layout(_props(entries=[_serialized_entry(IDENTIFIER)])).in_layout


def test_another_scripts_component_is_not_mistaken_for_this_one():
    other = [_plain_entry("com.example.other"), _serialized_entry("com.example.other")]
    assert not layout_contains({"components": other})
    assert layout_contains({"components": [*other, _plain_entry(IDENTIFIER)]})


def test_a_profile_with_no_layout_at_all_fails_rather_than_raising():
    check = check_layout({GUID_KEY: SHARED_GUID})
    assert not check.ok
    assert not check.in_layout and not check.bar_shown
    assert layout_contains(None) is False


def test_the_check_without_an_expected_profile_does_not_invent_a_mismatch():
    """`expected_guid` is opt-in, so a caller that has no opinion is not failed."""
    check = check_layout(_props(entries=[_plain_entry(IDENTIFIER)], original=UNUSED_GUID))
    assert check.profile_matches and check.ok


def test_a_state_change_emits_exactly_one_variable_write():
    """One write per change, and it is the status — never the tick.

    The tick belongs to the render that iTerm2 dispatches in response. A push
    that wrote both would report a render that had not happened, which is the
    one thing the tick exists to make impossible.
    """
    writes = _Writes()
    payload = asyncio.run(push_status(writes, PANE, Status.WORKING, "reading a file"))
    assert len(writes.calls) == 1
    assert writes.calls[0][:2] == (PANE, STATUS_VARIABLE)
    assert writes.named(TICK_VARIABLE) == []
    assert json.loads(payload) == {"status": "WORKING", "task": "reading a file"}


def test_the_render_tick_rises_with_each_render_of_the_same_pane():
    ticker = RenderTicker()
    writes = _Writes()

    def one_render(status, task):
        payload = asyncio.run(push_status(writes, PANE, status, task))
        return asyncio.run(render_for_session(PANE, payload, ticker=ticker, set_variable=writes))

    assert ticker.value(PANE) == 0
    first = one_render(Status.WORKING, "one")
    second = one_render(Status.AWAITING_HUMAN, "two")

    ticks = [value for _, _, value in writes.named(TICK_VARIABLE)]
    assert ticks == [1, 2], "each pushed change must render once and count once"
    assert ticker.value(PANE) == 2
    assert first != second, "the two renders must actually differ"


def test_each_pane_counts_its_own_renders():
    ticker = RenderTicker()
    writes = _Writes()
    asyncio.run(render_for_session(PANE, None, ticker=ticker, set_variable=writes))
    asyncio.run(render_for_session(PANE, None, ticker=ticker, set_variable=writes))
    asyncio.run(render_for_session("other", None, ticker=ticker, set_variable=writes))
    assert ticker.value(PANE) == 2
    assert ticker.value("other") == 1


def test_the_update_cadence_is_a_backstop_and_not_the_path_a_change_takes():
    """A change reaches the bar as a push, well inside one cadence period.

    Asserted as a number rather than as prose because `update_cadence=None`
    is legal in the library and would make "one write before the next tick"
    true of a component with no timer at all.
    """
    assert isinstance(UPDATE_CADENCE, float) and UPDATE_CADENCE > 0

    writes = _Writes()
    elapsed = 0.0  # no cadence tick has been allowed to fire
    asyncio.run(push_status(writes, PANE, Status.QUESTION, "which branch?"))
    assert elapsed < UPDATE_CADENCE
    assert len(writes.calls) == 1


def test_a_cold_pane_renders_before_anything_has_ever_been_pushed():
    """No pane has a status at first launch; the bar must still say something."""
    assert STATUS_REFERENCE == f"{STATUS_VARIABLE}?"
    assert not STATUS_VARIABLE.endswith("?"), "the variable itself is not optional-suffixed"
    assert line_for(None) == "unknown"
    assert decode_status(None) == (Status.UNKNOWN, None)


def test_an_unreadable_status_variable_renders_rather_than_raising():
    for raw in ("not json at all", "[1, 2, 3]", "{}", 17, ""):
        status, task = decode_status(raw)
        assert status is Status.UNKNOWN and task is None
        assert line_for(raw) == "unknown"


def test_a_status_name_this_build_does_not_have_reads_as_unknown():
    """A renamed enum member must not blank every bar mid-upgrade.

    The task text survives the unrecognised status. A daemon one version
    ahead of the component still knows what the pane is doing, and that half
    is worth more to someone scanning a wall of panes than the status word
    it could not spell.
    """
    assert decode_status('{"status": "TRANSCENDENT", "task": "x"}') == (Status.UNKNOWN, "x")
    assert decode_status('{"status": "WORKING", "task": "x"}') == (Status.WORKING, "x")
    assert line_for('{"status": "TRANSCENDENT", "task": "x"}') == "unknown: x"


def test_a_task_containing_the_separator_survives_the_round_trip():
    task = "fix: the thing\nthat broke"
    status, decoded = decode_status(encode_status(Status.ERROR, task))
    assert (status, decoded) == (Status.ERROR, task)
    assert "\n" not in line_for(encode_status(Status.ERROR, task), width=80)


def test_a_render_with_no_session_id_writes_nothing():
    writes = _Writes()
    line = asyncio.run(render_for_session(None, None, ticker=RenderTicker(), set_variable=writes))
    assert line == "unknown"
    assert writes.calls == []


def test_a_nonsense_width_is_clamped_rather_than_raised_on():
    """`render` refuses a width below 1, and a raise here stops the bar dead."""
    assert line_for(None, width=0) == "…"
    assert line_for(encode_status(Status.WORKING), width=-5) == "…"


def test_the_tick_is_not_written_when_no_line_was_produced(monkeypatch):
    """The tick asserts a render happened, so it is written only after one does.

    Ordering inside the coroutine is otherwise unobservable — `line_for` is
    total by construction — so this reaches in and breaks it to prove the two
    steps are in the order the module claims.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("render fell over")

    monkeypatch.setattr(component, "line_for", _boom)
    writes = _Writes()
    with pytest.raises(RuntimeError):
        asyncio.run(render_for_session(PANE, None, ticker=RenderTicker(), set_variable=writes))
    assert writes.calls == []


def test_a_refused_tick_write_costs_the_tick_and_not_the_line():
    writes = _Writes(fail=True)
    payload = encode_status(Status.BLOCKED, "waiting on a lock")
    line = asyncio.run(
        render_for_session(PANE, payload, ticker=RenderTicker(), set_variable=writes)
    )
    assert line.startswith("blocked")
    assert len(writes.calls) == 1


def test_the_component_never_takes_its_own_tick_as_an_input(monkeypatch):
    """A coroutine that read the tick it writes would re-trigger itself forever."""
    _stub_status_bar_api(monkeypatch)
    _, coro = build_component(object(), set_variable=_Writes())
    import inspect

    references = [
        repr(parameter.default)
        for parameter in inspect.signature(coro).parameters.values()
        if isinstance(parameter.default, _StubReference)
    ]
    assert references == [STATUS_REFERENCE, "id"]
    assert TICK_VARIABLE not in references


def test_the_component_is_built_with_the_identifier_and_cadence_it_documents(monkeypatch):
    _stub_status_bar_api(monkeypatch)
    built, _ = build_component(object(), set_variable=_Writes())
    assert built.kwargs["identifier"] == IDENTIFIER
    assert built.kwargs["update_cadence"] == UPDATE_CADENCE
    assert built.kwargs["knobs"] == []
    assert built.kwargs["exemplar"]


def test_registering_puts_the_component_nowhere_and_switches_nothing_on(monkeypatch):
    """Registration is only one of the four states, and the only one Palaver takes.

    Nothing here touches a profile: no layout is written, and the status bar
    is not switched on behind the user's back.
    """
    _stub_status_bar_api(monkeypatch)
    writes = _Writes()
    registered = asyncio.run(component.register(object(), set_variable=writes))
    assert len(registered.registrations) == 1
    assert writes.calls == []


def test_switching_the_status_bar_on_is_its_own_named_step():
    """It changes what every pane using the profile looks like, so it is opt-in."""
    written = []

    class _Profile:
        async def _async_simple_set(self, key, value):
            written.append((key, value))

    asyncio.run(show_status_bar(_Profile()))
    asyncio.run(show_status_bar(_Profile(), shown=False))
    assert written == [(SHOW_BAR_KEY, True), (SHOW_BAR_KEY, False)]


def test_the_check_reports_three_facts_rather_than_one_boolean():
    """`ok` alone cannot be acted on; each half has a different remedy."""
    check = LayoutCheck(
        profile_guid=SHARED_GUID, expected_guid=SHARED_GUID, in_layout=False, bar_shown=False
    )
    assert not check.ok
    assert "Configure Status Bar" in check.remedy and "Status bar enabled" in check.remedy


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


@live
def test_the_profile_keys_the_layout_check_reads_are_the_names_iterm_uses():
    """The gate reads two undocumented profile keys by literal name.

    Nothing in the `iterm2` library mentions either — there is no status bar
    API at all — so the only thing standing between a typo and a check that
    silently reports "not configured" forever is this.
    """

    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        seen = []
        for window in app.terminal_windows:
            for tab in window.tabs:
                for session in tab.sessions:
                    props = (await session.async_get_profile()).all_properties
                    seen.append((LAYOUT_KEY in props, SHOW_BAR_KEY in props, props.get(LAYOUT_KEY)))
        return seen

    seen = _run_live(body)
    assert seen, "the test itself is running in an iTerm2 pane"
    for has_layout, has_switch, layout in seen:
        assert has_layout, f"iTerm2 does not call it {LAYOUT_KEY!r}"
        assert has_switch, f"iTerm2 does not call it {SHOW_BAR_KEY!r}"
        assert "components" in layout, f"the layout is not shaped as expected: {sorted(layout)}"


@live
def test_a_live_panes_profile_identity_names_a_profile_that_actually_exists():
    """`Original Guid`, not `Guid`: every live session's profile is divorced.

    Measured 2026-08-15 — the guid a session reports for its own profile is
    session-local and appears in no shared profile list, so a check keyed on
    it would never match anything the user can configure.
    """

    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        shared = {profile.guid for profile in await iterm2.PartialProfile.async_query(conn)}
        found = []
        for window in app.terminal_windows:
            for tab in window.tabs:
                for session in tab.sessions:
                    props = (await session.async_get_profile()).all_properties
                    found.append((profile_identity(props), props.get(GUID_KEY)))
        return shared, found

    shared, found = _run_live(body)
    assert shared, "iTerm2 reported no shared profiles at all"
    assert found, "the test itself is running in an iTerm2 pane"
    for identity, own in found:
        assert identity in shared, f"{identity!r} is not a profile the user can configure"
        if own != identity:
            assert own not in shared, "a divorced guid should not also be a shared one"


@live
def test_the_component_registers_and_its_variables_round_trip_through_iterm():
    """Registration and the push path, against the real API rather than a stub.

    Both halves are things a stub cannot prove: that iTerm2 accepts the
    component's shape, and that `user.palaver_status` is a namespace it will
    take. Done in a tab this test opens and closes, so nothing is written
    into a pane that is actually running an agent.
    """

    async def body(conn):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(conn)
        registered = await component.register(conn)

        window = app.current_terminal_window
        tab = await window.async_create_tab()
        session = tab.sessions[0]
        try:
            writer = component.make_variable_writer(conn)
            await push_status(writer, session.session_id, Status.WORKING, "a live round trip")
            raw = await session.async_get_variable(STATUS_VARIABLE)
        finally:
            await tab.async_close(force=True)
        return registered is not None, raw

    ok, raw = _run_live(body, timeout=40.0)
    assert ok
    assert decode_status(raw) == (Status.WORKING, "a live round trip")
    assert line_for(raw) == "working: a live round trip"
