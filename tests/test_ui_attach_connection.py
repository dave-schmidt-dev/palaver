"""Task 5.1: attaching to iTerm2 over its Unix socket, and tracking panes.

The split in this file mirrors the split in the code. Connection preflight and
session bookkeeping are proven headlessly with no terminal involved. The
monitors are proven twice: once against a stub connection, so their control
flow is testable anywhere, and once against the real iTerm2 API, because a
stub cannot prove that `NewSessionMonitor` is the name of a thing that exists
or that iTerm2 will ever deliver to it.

The live tests are inert unless ``PALAVER_RUN_LIVE_ITERM_TESTS=1``. They create
one iTerm2 tab and close it again. That is visible on screen for a moment, and
it is the only way to make a session monitor fire —
the phase's own acceptance requires the pane check run "through the iTerm2
Python API itself rather than by eye". Every one of them closes what it opened
in a `finally`.

No cookie value is ever asserted on, printed, or captured here. The live tests
need one, so they ask iTerm2 for it and hand it straight to the connection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from palaver.ui import connection
from palaver.ui.connection import (
    COOKIE_ENV,
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
