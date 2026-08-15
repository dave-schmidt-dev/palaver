"""Connecting to iTerm2 over its Unix domain socket, with the cookie flow.

Task 5.1. Everything here is about refusing to connect the *wrong* way rather
than about connecting at all, because the `iterm2` library will happily do
either and never say which it did.

**The socket is not optional, even though the library treats it as optional.**
Read from the installed `iterm2` 2.20 source: `Connection._get_connect_coro`
tests whether `~/Library/Application Support/<suite>/private/socket` exists,
uses `websockets.unix_connect` if it does, and otherwise falls back to a TCP
connection to `ws://localhost:1912`. That fallback is a loopback network
listener, and Palaver has an invariant (INV-9) that observed content never
leaves the machine by socket. Loopback TCP is still a socket any other local
process can reach, and the failure is silent: the same script works either
way. `resolve_target` therefore fails closed when the Unix socket is absent
rather than letting the library choose.

Note the word "websocket" appears on both paths and means different things. On
the Unix path the library speaks the websocket *protocol* over a filesystem
socket, and the `ws://localhost/` string it passes is a protocol-level URI
that is never resolved or dialled. What must never happen is the TCP
*transport*, and that is what `connection_target` is about.

**The cookie is checked before connecting, not after.** iTerm2 injects
`ITERM2_COOKIE` and `ITERM2_KEY` into scripts it launches itself. A script
started any other way has neither, and the library's response is to ask
iTerm2 for one over AppleScript — which, depending on the user's settings,
either succeeds silently or raises a modal dialog on their screen. A daemon
must not be able to do that by accident, so `require_cookie` names the
condition and refuses.

Nothing in this module reads or transmits observed session content; it opens
the control channel only (INV-9).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: iTerm2 injects these into scripts it launches. `ITERM2_KEY` accompanies the
#: cookie and identifies the script across reconnects.
COOKIE_ENV = "ITERM2_COOKIE"
KEY_ENV = "ITERM2_KEY"

#: iTerm2 reads this to pick its Application Support directory, which is how
#: beta builds ("iTerm2-beta") keep a separate socket. Mirrored from the
#: library's own `Connection._unix_domain_socket_path`.
SUITE_ENV = "IT2_SUITE"
DEFAULT_SUITE = "iTerm2"

#: The TCP endpoint the library falls back to. Named so the refusal below can
#: say what it is refusing, and so a test can assert it never appears.
LEGACY_TCP_URI = "ws://localhost:1912"


class UiConnectionError(RuntimeError):
    """Base for every reason Palaver declines to attach to iTerm2."""


class MissingCookieError(UiConnectionError):
    """Raised when `ITERM2_COOKIE` is absent from the environment.

    Named rather than generic because the remedy is specific and not
    guessable from a stack trace: the process must be started *by* iTerm2 —
    from `AutoLaunch`, or from Scripts — or be given a cookie explicitly.
    """


class NoSocketTransportError(UiConnectionError):
    """Raised when iTerm2's Unix domain socket is absent.

    This is what would otherwise become a silent fallback to loopback TCP.
    The usual cause is that iTerm2's **Enable Python API** preference is off,
    which is its default: iTerm2 creates the socket only when the API server
    is running, so a fresh machine has no socket and no error message.
    """


class ITerm2NotInstalledError(UiConnectionError):
    """Raised when the optional `iterm2` dependency is not importable."""


def socket_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the Unix domain socket iTerm2 serves its API on.

    Args:
        env: Environment to read `IT2_SUITE` from, defaulting to `os.environ`.

    Returns:
        The socket path, whether or not it exists. Derived the same way the
        `iterm2` library derives it, so the two cannot disagree about which
        file they mean.
    """
    environ = os.environ if env is None else env
    suite = environ.get(SUITE_ENV) or DEFAULT_SUITE
    return Path.home() / "Library" / "Application Support" / suite / "private" / "socket"


def connection_target(env: Mapping[str, str] | None = None) -> str:
    """Return the transport endpoint Palaver requires, as a string.

    Returns:
        A filesystem path. Never a URL: a return value carrying a `ws://`
        scheme would mean the TCP fallback, which is the thing this module
        exists to prevent.
    """
    return str(socket_path(env))


def require_cookie(env: Mapping[str, str] | None = None) -> str:
    """Return the iTerm2 cookie, or refuse to proceed without one.

    Args:
        env: Environment to read, defaulting to `os.environ`.

    Returns:
        The cookie value.

    Raises:
        MissingCookieError: If `ITERM2_COOKIE` is absent or empty. An empty
            value is treated as absent because the library forwards it as a
            header either way and iTerm2 rejects the connection, which
            surfaces as an opaque HTTP status rather than as this.
    """
    environ = os.environ if env is None else env
    cookie = environ.get(COOKIE_ENV, "")
    if not cookie:
        raise MissingCookieError(
            f"{COOKIE_ENV} is not set. iTerm2 injects it into scripts it launches "
            "itself, so this process was started some other way. Install the "
            "AutoLaunch script with `python -m palaver.ui.autolaunch --install` and "
            "let iTerm2 start it, or set the variable from a cookie iTerm2 issued."
        )
    return cookie


def resolve_target(env: Mapping[str, str] | None = None) -> Path:
    """Return the socket to connect over, or refuse to fall back to TCP.

    Args:
        env: Environment to read, defaulting to `os.environ`.

    Returns:
        The existing socket path.

    Raises:
        NoSocketTransportError: If the socket does not exist, naming the
            preference that most often explains it.
    """
    path = socket_path(env)
    if not path.exists():
        raise NoSocketTransportError(
            f"iTerm2's API socket is not at {path}, so the iterm2 library would "
            f"fall back to {LEGACY_TCP_URI}. Turn on iTerm2 > Settings > General > "
            "Magic > Enable Python API; it is off by default and the socket exists "
            "only while the API server is running."
        )
    return path


def request_cookie_and_key(*, advisory_name: str = "palaver") -> tuple[str, str]:
    """Ask iTerm2 for a cookie and key over AppleScript.

    A process iTerm2 did not launch has no cookie, and this is the documented
    way to get one — the same request the `iterm2` library makes internally.
    Palaver calls it explicitly instead so that asking is a visible, named
    step rather than a side effect of connecting.

    **The return value is a credential.** It must never be logged, printed,
    written to a file, or passed in an argument vector, all of which would
    expose it: `ps` shows argv to every process on the machine, and Palaver's
    own log files are not credential stores. Put it in the environment of the
    process that needs it and nowhere else.

    Args:
        advisory_name: The name iTerm2 shows in its API-permission UI.

    Returns:
        A `(cookie, key)` pair.

    Raises:
        MissingCookieError: If iTerm2 is not running, refuses, or answers in
            a shape this does not recognise. Deliberately the same error the
            absent-variable path raises, because the caller's situation is
            identical: there is no cookie to connect with.
    """
    script = f'tell application "iTerm2" to request cookie and key for app named "{advisory_name}"'
    try:
        completed = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MissingCookieError(f"could not ask iTerm2 for a cookie: {exc}") from exc

    if completed.returncode != 0:
        # stderr may name the AppleScript error; it never contains the cookie,
        # because a failed request returns none.
        raise MissingCookieError(
            f"iTerm2 refused to issue a cookie: {completed.stderr.strip() or 'no reason given'}"
        )
    parts = completed.stdout.strip().split(" ")
    if len(parts) != 2 or not all(parts):
        # The response is not quoted, because it is the credential.
        raise MissingCookieError(
            "iTerm2 answered the cookie request in an unrecognised shape "
            f"({len(parts)} space-separated field(s), expected 2)"
        )
    return parts[0], parts[1]


def import_iterm2():
    """Import the optional `iterm2` dependency, or explain how to get it.

    Returns:
        The `iterm2` module.

    Raises:
        ITerm2NotInstalledError: If the `ui` extra is not installed.
    """
    try:
        import iterm2
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatching
        raise ITerm2NotInstalledError(
            "the iterm2 package is not installed; it lives in the optional `ui` "
            "extra, so install with `uv sync --extra ui` (or `pip install "
            "'palaver[ui]'`) on a machine that has iTerm2"
        ) from exc
    return iterm2


@dataclass(frozen=True)
class LibraryReset:
    """What `reset_library_state` actually cleared.

    Returned rather than discarded so a caller can tell the difference
    between "there was nothing to clear" and "the library moved and the
    clearing silently stopped working" — which otherwise look identical and
    only diverge once a stale handler fires.
    """

    app_invalidated: bool
    handler_keys_cleared: int
    #: Set when the private registry could not be reached, naming why.
    unreachable: str | None = None


def reset_library_state() -> LibraryReset:
    """Drop the `iterm2` library's two pieces of process-global state.

    Both are process-global rather than per-connection, and both outlive a
    connection that closed cleanly:

    * **The `App` singleton.** `iterm2.async_get_app` caches one per process
      and invalidates it only from a disconnect callback, so a second
      connection gets back an `App` still bound to the previous, closed
      websocket.
    * **The notification handler registry.** `iterm2.notifications` keeps one
      flat dict for the whole process, with no connection in the key. Every
      `App` ever constructed leaves its layout-change handler there, and the
      dispatcher runs *all* matching handlers for every incoming
      notification. So after a second connection, a layout change delivered
      on the live connection is also handed to the dead one's handler, which
      raises `ConnectionClosedError` from inside the library's dispatch loop
      — far from anything that names the cause.

    The registry has no public accessor, so this reaches for a private one.
    That is deliberate and bounded: the failure is reported in the return
    value rather than swallowed, and nothing here depends on the registry's
    contents, only on its emptiness.

    Returns:
        A `LibraryReset` describing what was cleared.
    """
    try:
        import iterm2.app
        import iterm2.notifications
    except ImportError as exc:  # pragma: no cover - the extra is simply absent
        return LibraryReset(False, 0, unreachable=str(exc))

    iterm2.app.invalidate_app()

    getter = getattr(iterm2.notifications, "_get_handlers", None)
    if getter is None:
        return LibraryReset(
            True, 0, unreachable="iterm2.notifications._get_handlers is gone in this version"
        )
    handlers = getter()
    cleared = len(handlers)
    handlers.clear()
    return LibraryReset(True, cleared)


def preflight(env: Mapping[str, str] | None = None) -> Path:
    """Run every check that must pass before a connection is attempted.

    Ordered deliberately: the socket first, because a missing socket is a
    setup problem the user can fix, while a missing cookie on a machine with
    no API server would send them chasing the wrong thing.

    Args:
        env: Environment to read, defaulting to `os.environ`.

    Returns:
        The socket path that will be used.

    Raises:
        UiConnectionError: The first failing check, as its specific subclass.
    """
    path = resolve_target(env)
    require_cookie(env)
    return path


def run_forever(
    coro: Callable[[Any], Coroutine[Any, Any, None]],
    *,
    env: Mapping[str, str] | None = None,
    retry: bool = True,
) -> None:
    """Attach to iTerm2 and run `coro` until the process is stopped.

    Args:
        coro: An async callable taking one `iterm2.Connection`.
        env: Environment to preflight against, defaulting to `os.environ`.
        retry: Whether the library should reconnect when iTerm2 restarts.
            True by default: a terminal emulator relaunching is routine, and
            a surface that vanished permanently on the first restart would be
            worse than no surface.

    Raises:
        UiConnectionError: If preflight fails. Nothing is attempted after
            that, so a misconfigured machine gets one clear error rather than
            a reconnect loop against a socket that will never appear.
    """
    preflight(env)
    iterm2 = import_iterm2()
    iterm2.run_forever(coro, retry=retry)
