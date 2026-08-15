"""The iTerm2 AutoLaunch entry point and the session monitors it runs.

Task 5.1. Two separable things live here, and the separation is the point.

**The shim** is what gets installed into iTerm2's `AutoLaunch` directory. It
is deliberately trivial: it obtains a fresh cookie, spawns Palaver's *own*
interpreter, and restarts it with a backoff. It never imports `palaver`.

That is not fastidiousness. iTerm2 runs a "simple" script under the Python
runtime it downloads and manages itself, whose version is iTerm2's choice and
moves when iTerm2 moves. Palaver requires 3.14. A shim that imported
`palaver` would be a syntax error on the day iTerm2 shipped an older runtime,
and the failure would appear as a script that silently never started. So the
shim is written to run under an old interpreter and does nothing that could
depend on one.

**The monitors** are what actually runs, under Palaver's interpreter, reached
by `python -m palaver.ui.autolaunch`. `iterm2.NewSessionMonitor` covers panes
opened after attach and `iterm2.SessionTerminationMonitor` covers panes that
go away; the app's existing sessions are attached once at startup, because a
monitor only reports what happens *next* and a surface that appeared solely
in new panes would look broken to anyone who had iTerm2 open already.

The bookkeeping between them is `SessionRegistry`, which is pure and has no
iTerm2 types in it at all — that is what makes attach/detach provable without
a terminal, leaving only the genuinely live parts to be proven live.

**Cookies are credentials.** Nothing here logs, prints, or writes one, and
the shim passes it to the child through the environment rather than through
an argument vector, which `ps` would expose to every process on the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from palaver.ui.connection import (
    COOKIE_ENV,
    KEY_ENV,
    UiConnectionError,
    import_iterm2,
    preflight,
)

#: Where iTerm2 looks for scripts to run at launch. Fixed by iTerm2, not by
#: Palaver: a script anywhere else is simply never started.
AUTOLAUNCH_DIR = (
    Path.home() / "Library" / "Application Support" / "iTerm2" / "Scripts" / "AutoLaunch"
)

#: The installed shim's filename.
SHIM_NAME = "palaver.py"

#: Seconds the shim waits before the first restart, doubling to
#: `SHIM_MAX_BACKOFF`. A crash loop against a stopped iTerm2 should not spin.
SHIM_MIN_BACKOFF = 2
SHIM_MAX_BACKOFF = 60

#: How the shim identifies itself when asking iTerm2 for a cookie. iTerm2
#: shows this name in its API-permission UI, so it is user-facing.
ADVISORY_NAME = "palaver"


class SessionRegistry:
    """Which panes Palaver is currently attached to.

    Holds session ids and nothing else — no `iterm2.Session` objects. A
    registry that held live handles would keep terminated sessions alive and
    would be untestable without a terminal, and neither is worth the saved
    lookup.
    """

    def __init__(self, session_ids: Iterable[str] = ()) -> None:
        self._attached: set[str] = set(session_ids)

    def __len__(self) -> int:
        return len(self._attached)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._attached

    @property
    def attached(self) -> frozenset[str]:
        """The attached session ids, as an immutable snapshot."""
        return frozenset(self._attached)

    def attach(self, session_id: str) -> bool:
        """Record a session as attached.

        Args:
            session_id: iTerm2's session id.

        Returns:
            True if this was a new attachment, False if it was already
            attached. The distinction matters: `NewSessionMonitor` can report
            a session that startup already swept, and treating that as new
            would double-count and re-register the component.

        Raises:
            ValueError: If the id is empty. iTerm2 returns None for a session
                that vanished mid-query, and an empty entry would silently
                never match a termination.
        """
        if not session_id:
            raise ValueError("a session id is required; refusing to attach to an unnamed pane")
        added = session_id not in self._attached
        self._attached.add(session_id)
        return added

    def detach(self, session_id: str) -> bool:
        """Forget a session.

        Args:
            session_id: iTerm2's session id.

        Returns:
            True if it had been attached. False for an unknown id, which is
            normal: `SessionTerminationMonitor` reports every pane closing,
            including ones Palaver never attached to.
        """
        had = session_id in self._attached
        self._attached.discard(session_id)
        return had


def _no_status(_message: str) -> None:
    """Default progress sink."""


async def attach_existing(
    app,
    registry: SessionRegistry,
    *,
    on_attach: Callable[[str], object] | None = None,
    on_status: Callable[[str], None] = _no_status,
) -> int:
    """Attach to every pane that already exists.

    A monitor reports only what happens after it starts, so without this the
    surface would appear in new panes and nowhere else.

    Args:
        app: The `iterm2.App`.
        registry: Registry to record attachments in.
        on_attach: Optional per-session hook, awaited if it returns an
            awaitable. Task 5.3 supplies the one that registers the status
            bar component.
        on_status: Progress channel (INV-1).

    Returns:
        How many sessions were newly attached.
    """
    attached = 0
    for window in app.terminal_windows:
        for tab in window.tabs:
            for session in tab.sessions:
                if await _attach_one(session.session_id, registry, on_attach):
                    attached += 1
    on_status(f"attached to {attached} existing pane(s)")
    return attached


async def _attach_one(
    session_id: str | None,
    registry: SessionRegistry,
    on_attach: Callable[[str], object] | None,
) -> bool:
    """Attach one session id, running the hook only on a genuinely new one."""
    if not session_id:
        return False
    if not registry.attach(session_id):
        return False
    if on_attach is not None:
        result = on_attach(session_id)
        if asyncio.iscoroutine(result):
            await result
    return True


async def watch_new_sessions(
    connection,
    registry: SessionRegistry,
    *,
    on_attach: Callable[[str], object] | None = None,
    on_status: Callable[[str], None] = _no_status,
    limit: int | None = None,
) -> int:
    """Attach to panes as they open, until `limit` events or forever.

    Args:
        connection: The `iterm2.Connection`.
        registry: Registry to record attachments in.
        on_attach: Optional per-session hook; see `attach_existing`.
        on_status: Progress channel (INV-1).
        limit: Stop after this many events. None means never stop, which is
            the daemon's case; a number is what makes this testable.

    Returns:
        How many sessions were newly attached.
    """
    iterm2 = import_iterm2()
    attached = 0
    seen = 0
    async with iterm2.NewSessionMonitor(connection) as monitor:
        while limit is None or seen < limit:
            session_id = await monitor.async_get()
            seen += 1
            if await _attach_one(session_id, registry, on_attach):
                attached += 1
                on_status(f"attached to new pane ({len(registry)} attached)")
    return attached


async def watch_terminations(
    connection,
    registry: SessionRegistry,
    *,
    on_status: Callable[[str], None] = _no_status,
    limit: int | None = None,
) -> int:
    """Forget panes as they close, until `limit` events or forever.

    Args:
        connection: The `iterm2.Connection`.
        registry: Registry to prune.
        on_status: Progress channel (INV-1).
        limit: Stop after this many events; None means never.

    Returns:
        How many *attached* sessions were forgotten. Terminations of panes
        Palaver never attached to are counted as events but not as detaches.
    """
    iterm2 = import_iterm2()
    detached = 0
    seen = 0
    async with iterm2.SessionTerminationMonitor(connection) as monitor:
        while limit is None or seen < limit:
            session_id = await monitor.async_get()
            seen += 1
            if session_id and registry.detach(session_id):
                detached += 1
                on_status(f"pane closed ({len(registry)} attached)")
    return detached


async def main(
    connection,
    *,
    registry: SessionRegistry | None = None,
    on_attach: Callable[[str], object] | None = None,
    on_status: Callable[[str], None] = _no_status,
    limit: int | None = None,
) -> SessionRegistry:
    """Attach to every pane and keep the registry current.

    Args:
        connection: The `iterm2.Connection`.
        registry: Registry to use, defaulting to an empty one.
        on_attach: Optional per-session hook; see `attach_existing`.
        on_status: Progress channel (INV-1).
        limit: Events per monitor before returning. None runs forever.

    Returns:
        The registry, so a bounded run can be asserted against.
    """
    iterm2 = import_iterm2()
    registry = SessionRegistry() if registry is None else registry
    app = await iterm2.async_get_app(connection)
    await attach_existing(app, registry, on_attach=on_attach, on_status=on_status)
    # Gathered rather than sequenced: a pane can close while another opens,
    # and awaiting one monitor at a time would hold the other's events until
    # the first happened to fire.
    await asyncio.gather(
        watch_new_sessions(
            connection, registry, on_attach=on_attach, on_status=on_status, limit=limit
        ),
        watch_terminations(connection, registry, on_status=on_status, limit=limit),
    )
    return registry


def render_shim(
    *,
    python_executable: Path | None = None,
    module: str = "palaver.ui.autolaunch",
    project_root: Path | None = None,
) -> str:
    """Render the AutoLaunch shim iTerm2 will run.

    Args:
        python_executable: Interpreter to spawn, defaulting to the one
            rendering this. That is the interpreter Palaver is installed
            into, which is exactly the one that must run it.
        module: Module to run with `-m`.
        project_root: Working directory for the child.

    Returns:
        Python source. Written for an old interpreter on purpose — see the
        module docstring — so it uses no syntax newer than 3.6.
    """
    executable = Path(sys.executable) if python_executable is None else python_executable
    root = Path.cwd() if project_root is None else project_root
    return f'''"""Palaver's iTerm2 AutoLaunch shim. Generated -- do not edit.

Regenerate with `python -m palaver.ui.autolaunch --install`.

This file runs under iTerm2's own managed Python, whose version iTerm2
chooses. It therefore imports nothing from Palaver and uses no modern syntax:
its whole job is to obtain a cookie and hand off to Palaver's interpreter,
which is the one that can actually run Palaver.
"""

import os
import subprocess
import sys
import time

PYTHON = {str(executable)!r}
MODULE = {module!r}
CWD = {str(root)!r}
MIN_BACKOFF = {SHIM_MIN_BACKOFF}
MAX_BACKOFF = {SHIM_MAX_BACKOFF}
ADVISORY_NAME = {ADVISORY_NAME!r}


def fresh_cookie_and_key():
    """Ask iTerm2 for a cookie over AppleScript.

    Returns a (cookie, key) pair, or (None, None). The value is never
    printed: it is a credential, and this script's stdout is a log file.
    """
    script = (
        'tell application "iTerm2" to request cookie and key for app named "%s"'
        % ADVISORY_NAME
    )
    try:
        output = subprocess.check_output(
            ["/usr/bin/osascript", "-e", script],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None, None
    parts = output.decode("utf-8").strip().split(" ")
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def main():
    backoff = MIN_BACKOFF
    while True:
        env = dict(os.environ)
        cookie, key = fresh_cookie_and_key()
        if cookie:
            # Passed by environment, never by argv: argv is world-readable
            # through ps.
            env[{COOKIE_ENV!r}] = cookie
            env[{KEY_ENV!r}] = key
        started = time.time()
        status = subprocess.call([PYTHON, "-m", MODULE], cwd=CWD, env=env)
        ran_for = time.time() - started
        sys.stderr.write(
            "palaver: attach exited with status %d after %.0fs\\n" % (status, ran_for)
        )
        # A run that lasted a while was healthy; only a fast failure escalates.
        backoff = MIN_BACKOFF if ran_for > MAX_BACKOFF else min(backoff * 2, MAX_BACKOFF)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
'''


def install_shim(
    *,
    directory: Path | None = None,
    python_executable: Path | None = None,
    project_root: Path | None = None,
) -> Path:
    """Write the shim into iTerm2's AutoLaunch directory.

    Args:
        directory: Destination, defaulting to iTerm2's AutoLaunch directory.
        python_executable: Interpreter the shim will spawn.
        project_root: Working directory the shim will use.

    Returns:
        The path written.
    """
    destination = AUTOLAUNCH_DIR if directory is None else directory
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / SHIM_NAME
    path.write_text(
        render_shim(python_executable=python_executable, project_root=project_root),
        encoding="utf-8",
    )
    return path


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the `python -m palaver.ui.autolaunch` parser."""
    parser = argparse.ArgumentParser(
        prog="python -m palaver.ui.autolaunch",
        description=(
            "Attach Palaver to every iTerm2 pane. With no arguments this is the "
            "long-running attachment iTerm2's AutoLaunch shim starts."
        ),
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="write the AutoLaunch shim into iTerm2's Scripts/AutoLaunch directory",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="render the shim to stdout and write nothing",
    )
    parser.add_argument(
        "--autolaunch-dir",
        type=Path,
        default=None,
        help=f"where to install the shim (default: {AUTOLAUNCH_DIR})",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    """Run the module entry point.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        0 on success, 1 when the machine is not set up to attach. The
        connection failure is reported by name — a missing cookie and a
        missing API socket have different remedies and must not collapse
        into one message.
    """
    args = build_parser().parse_args(argv)

    if args.print_only:
        sys.stdout.write(render_shim())
        return 0

    if args.install:
        path = install_shim(directory=args.autolaunch_dir)
        sys.stdout.write(f"installed {path}\n")
        sys.stdout.write(
            "iTerm2 runs it at next launch. Enable Python API first: "
            "iTerm2 > Settings > General > Magic > Enable Python API.\n"
        )
        return 0

    try:
        preflight()
    except UiConnectionError as exc:
        print(f"palaver ui: {exc}", file=sys.stderr)
        return 1

    iterm2 = import_iterm2()
    registry = SessionRegistry()

    async def _attached(connection):
        await main(connection, registry=registry, on_status=_stderr_status)

    # `retry` so a routine iTerm2 relaunch does not end the surface for good.
    iterm2.run_forever(_attached, retry=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(run())
