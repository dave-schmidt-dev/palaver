"""`palaver install-agent`: give the observer daemon a supervised home (task 5.0).

`palaver observe` is the single SQLite writer and the owner of the tick loop.
Every later phase assumes it is running — the MCP write path in 6.3 posts to
it, and the pane surface in Phase 5 renders what it wrote — but until now
nothing restarted it when it died. This module renders
`palaver/supervision/observe.plist.tmpl` into a launchd user agent that does.

Three decisions here are deliberate and would otherwise look arbitrary:

**Rendering and loading are separate steps.** The default writes the plist and
prints the two `launchctl` lines; `--load` is what actually bootstraps it into
the session. Loading a user agent is a change to the machine's running state,
so it is opt-in rather than a side effect of asking to see the file. `--print`
does not touch the filesystem at all.

**`bootout` is never called implicitly.** `--reload` exists and says so, but a
plain `--load` against an already-loaded label fails with launchd's own error
rather than silently tearing down a daemon that is mid-tick and holding the
write lock.

**Paths are XML-escaped.** A project checked out under a directory containing
`&` renders a plist that `plutil` rejects outright, which at least fails
loudly; the worse case is a path containing `<`, which produces a *valid*
plist with a truncated path. Escaping is not optional politeness here.

Output follows the CLI's two-stream contract: the rendered plist and the
resulting paths go to stdout, and progress goes through `on_status` to stderr
(INV-1).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from string import Template
from typing import TextIO
from xml.sax.saxutils import escape

from palaver.logging_setup import PROJECT_ROOT

NAME = "install-agent"
HELP = "render and optionally load the launchd user agent that supervises `palaver observe`"

#: Reverse-DNS under the same prefix as the machine's other user agents, so
#: `launchctl list | grep com.zerodelta` shows Palaver beside its siblings.
DEFAULT_LABEL = "com.zerodelta.palaver.observe"

#: The template rendered by `render_plist`. Kept beside the package rather
#: than inlined as a string so the XML is readable, diffable, and lintable by
#: `plutil` on its own.
TEMPLATE_PATH = PROJECT_ROOT / "palaver" / "supervision" / "observe.plist.tmpl"

#: Where a loaded user agent lives. launchd will bootstrap a plist from any
#: absolute path, but only this directory is re-read at login.
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"

#: Seconds launchd waits before restarting a job that exited. This is
#: launchd's own default, restated in the plist so the restart window below
#: is derived from a declared number rather than an assumed one.
THROTTLE_INTERVAL_SECONDS = 10

#: How long a restart may take before a caller should call it a failure. Two
#: throttle intervals plus slack: one interval can already be partly elapsed
#: when the process dies, and the machine may be busy.
RESTART_WINDOW_SECONDS = 2.0 * THROTTLE_INTERVAL_SECONDS + 5.0

#: Where launchd's captured stdout and stderr land. `.logs/` is the repo's
#: gitignored log directory, which is what the daemon's own file handler
#: already uses; putting launchd's streams anywhere else would split one
#: process's output across two conventions.
DEFAULT_LOG_DIR = PROJECT_ROOT / ".logs"

#: `launchctl print` emits one `pid = NNNN` line for a running job and omits
#: it entirely for one that is loaded but not currently running.
_PID_PATTERN = re.compile(r"^\s*pid\s*=\s*(\d+)\s*$", re.MULTILINE)

#: A label becomes a filename and a launchctl service target, so neither a
#: path separator nor whitespace can appear in one.
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InstallAgentError(RuntimeError):
    """Raised when an agent cannot be rendered, written, or loaded."""


def domain_target(uid: int | None = None) -> str:
    """Return the launchd domain a user agent belongs to.

    Args:
        uid: User id, defaulting to the current effective uid.

    Returns:
        A `gui/<uid>` domain target. The GUI domain rather than `user/<uid>`
        because the agent must reach the same session as iTerm2, which Phase
        5's surface attaches to; a job in the background user domain cannot.
    """
    return f"gui/{os.getuid() if uid is None else uid}"


def service_target(label: str, *, uid: int | None = None) -> str:
    """Return the `<domain>/<label>` target `launchctl` addresses a job by."""
    return f"{domain_target(uid)}/{label}"


def observe_program_arguments(
    *,
    executable: Path,
    db_path: Path | None = None,
    cursor_root: Path | None = None,
    interval: float | None = None,
) -> list[str]:
    """Build the argv launchd should run.

    Only explicitly supplied options are emitted. `palaver observe` already
    has defaults for each, and duplicating them here would mean two places to
    change when one of them moves.

    Args:
        executable: Absolute path to the `palaver` console script.
        db_path: Store to write into, or None to leave the daemon's default.
        cursor_root: Cursor directory, or None for the daemon's default.
        interval: Seconds between ticks, or None for the daemon's default.

    Returns:
        The argv, beginning with the executable.
    """
    argv = [str(executable), "observe"]
    if db_path is not None:
        argv += ["--db", str(db_path)]
    if cursor_root is not None:
        argv += ["--cursors", str(cursor_root)]
    if interval is not None:
        argv += ["--interval", f"{interval:g}"]
    return argv


def render_plist(
    *,
    label: str = DEFAULT_LABEL,
    program_arguments: Sequence[str],
    stdout_path: Path,
    stderr_path: Path,
    working_directory: Path = PROJECT_ROOT,
    throttle_interval: int = THROTTLE_INTERVAL_SECONDS,
    template_path: Path = TEMPLATE_PATH,
) -> str:
    """Render the observer agent's plist.

    Args:
        label: launchd label, which is also the plist's filename.
        program_arguments: argv launchd runs; must be non-empty.
        stdout_path: Where launchd writes the job's stdout.
        stderr_path: Where launchd writes the job's stderr.
        working_directory: Directory the job starts in.
        throttle_interval: Seconds launchd waits before a restart.
        template_path: Template to render, for tests that supply their own.

    Returns:
        The complete plist XML.

    Raises:
        InstallAgentError: If the label is not a usable launchd label, the
            argv is empty, or the template is missing.
    """
    if not _LABEL_PATTERN.match(label):
        raise InstallAgentError(
            f"{label!r} is not a usable launchd label: it becomes both a filename "
            "and a service target, so it must be alphanumeric with dots, dashes, "
            "or underscores"
        )
    if not program_arguments:
        raise InstallAgentError(
            "a launchd job with no ProgramArguments would load and run nothing, "
            "which launchd reports as success"
        )
    try:
        template = Template(template_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InstallAgentError(
            f"cannot read the agent template at {template_path}: {exc}"
        ) from exc

    argv_block = "\n".join(
        f"\t\t<string>{escape(argument)}</string>" for argument in program_arguments
    )
    return template.substitute(
        LABEL=escape(label),
        PROGRAM_ARGUMENTS=argv_block,
        THROTTLE_INTERVAL=int(throttle_interval),
        WORKING_DIRECTORY=escape(str(working_directory)),
        STDOUT_PATH=escape(str(stdout_path)),
        STDERR_PATH=escape(str(stderr_path)),
    )


def _launchctl(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run one `launchctl` invocation, capturing both streams."""
    return subprocess.run(
        ["launchctl", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def bootstrap(plist_path: Path, *, uid: int | None = None) -> subprocess.CompletedProcess[str]:
    """Load a plist into the GUI domain.

    Args:
        plist_path: Absolute path to a rendered plist.
        uid: User id, defaulting to the current one.

    Returns:
        The completed `launchctl bootstrap`, uninspected — callers decide
        whether a non-zero status is fatal, because "already loaded" is a
        normal outcome for a caller that is reconciling state.
    """
    return _launchctl("bootstrap", domain_target(uid), str(plist_path))


def bootout(label: str, *, uid: int | None = None) -> subprocess.CompletedProcess[str]:
    """Unload a job by label. Never called implicitly; see the module docstring."""
    return _launchctl("bootout", service_target(label, uid=uid))


def print_service(label: str, *, uid: int | None = None) -> subprocess.CompletedProcess[str]:
    """Return `launchctl print` for a job, whose exit status is the load check."""
    return _launchctl("print", service_target(label, uid=uid))


def service_pid(label: str, *, uid: int | None = None) -> int | None:
    """Return the pid of a running job, or None.

    Args:
        label: launchd label.
        uid: User id, defaulting to the current one.

    Returns:
        The pid, or None when the job is not loaded or is loaded but between
        runs. The two cases are deliberately not distinguished: a caller
        watching for a restart wants "no pid yet" either way.
    """
    result = print_service(label, uid=uid)
    if result.returncode != 0:
        return None
    match = _PID_PATTERN.search(result.stdout)
    return int(match.group(1)) if match else None


def pid_is_ours(pid: int) -> bool:
    """Report whether this process can signal `pid`.

    The distinction matters more than it looks. launchd publishes a job's pid
    the moment it forks, but at that instant the pid belongs to `xpcproxy`
    running as **uid 0** — the setuid stub that sets the job up and only then
    drops to the job's real user and execs the program. Signalling it in that
    window fails with `EPERM`, and a caller that treated "the label has a pid"
    as "the daemon is running" would be wrong for a few hundred milliseconds
    after every single start and restart.

    `os.kill(pid, 0)` is the discriminator: signal 0 performs the permission
    check and delivers nothing.

    Args:
        pid: Process id from `service_pid`.

    Returns:
        True once the pid is a process this user may signal, False while it is
        still the root stub or has already gone away.
    """
    try:
        os.kill(pid, 0)
    except PermissionError:
        return False
    except ProcessLookupError:
        return False
    return True


def wait_for_running_pid(
    label: str,
    *,
    window: float = RESTART_WINDOW_SECONDS,
    poll_interval: float = 0.25,
    uid: int | None = None,
) -> int | None:
    """Wait until a job has a pid this user can actually signal.

    Args:
        label: launchd label.
        window: Seconds to wait before giving up.
        poll_interval: Seconds between polls.
        uid: User id, defaulting to the current one.

    Returns:
        The pid, or None if the job never reached a signalable state. See
        `pid_is_ours` for why "has a pid" is the weaker condition.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        pid = service_pid(label, uid=uid)
        if pid is not None and pid_is_ours(pid):
            return pid
        time.sleep(poll_interval)
    return None


def wait_for_new_pid(
    label: str,
    previous_pid: int | None,
    *,
    window: float = RESTART_WINDOW_SECONDS,
    poll_interval: float = 0.5,
    uid: int | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int | None:
    """Wait for launchd to restart a job under a new pid.

    Args:
        label: launchd label.
        previous_pid: The pid observed before the job was killed.
        window: Seconds to wait before giving up.
        poll_interval: Seconds between `launchctl print` calls.
        uid: User id, defaulting to the current one.
        on_status: Progress channel (INV-1) — a caller waiting out a
            throttle interval should see that the wait is proceeding.

    Returns:
        The new pid, or None if none appeared inside the window.

        Two pids are rejected. One equal to `previous_pid`, because launchd
        reuses no pid that fast and seeing the old one means the kill has not
        landed yet. And one that is not yet signalable by this user, because
        that is launchd's root `xpcproxy` stub rather than the restarted
        daemon — see `pid_is_ours`.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        current = service_pid(label, uid=uid)
        if current is not None and current != previous_pid and pid_is_ours(current):
            return current
        if on_status is not None:
            remaining = deadline - time.monotonic()
            on_status(f"{label}: awaiting restart, {remaining:.0f}s of the window left")
        time.sleep(poll_interval)
    return None


def add_arguments(parser) -> None:
    """Register `install-agent`'s flags on its subparser."""
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=f"launchd label, which is also the plist filename (default: {DEFAULT_LABEL})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="store the supervised daemon writes into (default: the daemon's own default)",
    )
    parser.add_argument(
        "--cursors",
        type=Path,
        default=None,
        help="cursor directory for the supervised daemon (default: the daemon's own default)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="seconds between ticks (default: the daemon's own default)",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=None,
        help="path to the `palaver` console script (default: the running interpreter's)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"directory for launchd's captured streams (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--plist-path",
        type=Path,
        default=None,
        help=f"where to write the plist (default: {LAUNCH_AGENTS_DIR}/<label>.plist)",
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="render to stdout and write nothing",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="bootstrap the written plist into the GUI domain",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="bootout an already-loaded label first, then load (implies --load)",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def _default_executable() -> Path:
    """Locate the `palaver` console script next to the running interpreter.

    `shutil.which` is deliberately not used: under `uv run` the console
    script exists in the environment's `bin/` whether or not that directory
    is on PATH, and a PATH lookup could just as easily find a different
    checkout's script.

    The unresolved `sys.executable` is searched first and it matters. A venv's
    `bin/python3` is a symlink to the base interpreter, so resolving it lands
    in the base installation's `bin/`, where no `palaver` script exists — the
    fallback would then render a plist running `python3 observe`, which exits
    instantly, and `KeepAlive` would restart that failure forever.

    Returns:
        The console script if one was found, otherwise the interpreter — a
        caller can always override with `--executable`.
    """
    for directory in (Path(sys.executable).parent, Path(sys.executable).resolve().parent):
        candidate = directory / "palaver"
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver install-agent`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 on success, 1 when the plist cannot be rendered or written, and 1
        when a requested load fails. Rendering without loading is a success
        even though nothing is running afterwards — that is what was asked.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    executable = _default_executable() if args.executable is None else args.executable
    log_dir = DEFAULT_LOG_DIR if args.log_dir is None else args.log_dir
    plist_path = (
        LAUNCH_AGENTS_DIR / f"{args.label}.plist" if args.plist_path is None else args.plist_path
    )

    try:
        rendered = render_plist(
            label=args.label,
            program_arguments=observe_program_arguments(
                executable=executable,
                db_path=args.db,
                cursor_root=args.cursors,
                interval=args.interval,
            ),
            stdout_path=log_dir / f"{args.label}.out.log",
            stderr_path=log_dir / f"{args.label}.err.log",
        )
    except InstallAgentError as exc:
        print(f"palaver install-agent: {exc}", file=sys.stderr)
        return 1

    if args.print_only:
        out.write(rendered)
        return 0

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        plist_path.parent.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        print(f"palaver install-agent: cannot write {plist_path}: {exc}", file=sys.stderr)
        return 1

    out.write(f"wrote {plist_path}\n")

    if not (args.load or args.reload):
        out.write(f"load it with:  launchctl bootstrap {domain_target()} {plist_path}\n")
        out.write(f"unload it with:  launchctl bootout {service_target(args.label)}\n")
        return 0

    if args.reload:
        on_status(f"{args.label}: unloading any previously loaded job")
        bootout(args.label)

    on_status(f"{args.label}: bootstrapping into {domain_target()}")
    result = bootstrap(plist_path)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"status {result.returncode}"
        print(f"palaver install-agent: launchctl bootstrap failed: {detail}", file=sys.stderr)
        return 1

    pid = service_pid(args.label)
    out.write(f"loaded {service_target(args.label)}" + (f" (pid {pid})\n" if pid else "\n"))
    return 0
