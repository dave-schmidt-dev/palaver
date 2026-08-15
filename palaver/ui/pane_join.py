"""Joining an iTerm2 pane to the agent process and session store behind it.

Task 5.2. Everything here exists to answer one question — *which agent, in
which project, is this pane running?* — and to answer it with no rather than
with a guess, because a wrong join puts one session's status on another
session's pane, which is worse than a blank pane in exactly the way a
confident wrong `DONE` is worse than `UNKNOWN`.

Three measured facts on this machine (2026-08-15) shape the whole design.
Each one kills an approach that is obvious until you look.

**1. `jobPid` is not the agent — it is whatever descendant currently holds
the pane's foreground process group.** The plan warned that `pid` is the
login shell rather than the agent; the reality is further off than that. For
every Claude Code pane observed, `jobPid` resolved to a `playwright-mcp` MCP
server the agent had spawned, three hops down::

    node …/playwright-mcp   <- jobPid
      npm exec @playwright/mcp@latest
        claude              <- the agent
          -zsh              <- the login shell

So the join walks *up* the parent chain from `jobPid` looking for a known
agent, and stops at the login shell — an agent must be a descendant of the
pane's own shell, which is what makes the walk bounded and what stops it
from wandering into an unrelated ancestor.

**2. `jobName` is `node` for a Claude Code pane, not `claude`.** Keying an
agent table on `jobName` would therefore reject every Claude Code pane while
looking perfectly reasonable in review. `jobName` is used here for what it
can actually prove: it must still match the live process table's name for
`jobPid`, which detects pane variables that have gone stale relative to the
process that produced them.

**3. The file-descriptor join does not work either.** Codex holds its
rollout files open, so `lsof` on a codex pane looks like an exact
pane-to-transcript join — until you count them and find **ten** rollouts open
at once. Claude Code holds none at all. There is no per-pane transcript
discriminator on either source, which is why `PaneJoin` resolves the
*project* and reports `session_candidates` rather than picking one.

What is left is a join on **agreement between two independently-obtained
values**: the pane says `path`, the agent process's own working directory
says something, and the join happens only if they are the same directory.
That is what makes the ssh case safe without special-casing it — over an ssh
hop the foreground job is `ssh`, no agent is found in the pane's own process
tree, and the walk returns nothing.

INV-2: every probe here is read-only — `ps` and `lsof` observe, and no
process is ever signalled beyond the existence check. INV-9: nothing reads
session *content*. The candidate scan reads directory entries and mtimes,
never a byte inside a transcript.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.observer.signals import Liveness, Tri

#: Executable basenames that identify an agent, mapped to the adapter
#: `source` name that reads its session store. Matched against the agent
#: process's own name, never against `jobName` — see fact 2 in the module
#: docstring.
AGENT_SOURCES: Mapping[str, str] = {
    "claude": "claude-code",
    "codex": "codex",
    "opencode": "opencode",
}

#: Names that end the ancestry walk. An agent started in a pane is a
#: descendant of that pane's login shell, so anything at or above the shell
#: belongs to iTerm2 rather than to the pane's work, and a match up there
#: would be a coincidence rather than a join. `login` is included because
#: iTerm2 execs the shell through it.
SHELL_NAMES: frozenset[str] = frozenset(
    {"zsh", "bash", "sh", "fish", "tcsh", "csh", "dash", "ksh", "login"}
)

#: Ancestry hops to walk before giving up. Bounded rather than "walk to pid
#: 1" so a cycle in a malformed process table — or a `ppid` that fails to
#: decrease — terminates instead of hanging the status tick. Twelve is far
#: past the deepest real chain observed (four, for the Claude Code pane in
#: the module docstring).
MAX_ANCESTRY_HOPS = 12

#: How long a session store must go unwritten before `apply_liveness` will
#: call a live, turn-ended session `IDLE`. Well clear of the observer's
#: 30–60s tick, so a session cannot be reported idle merely because two
#: ticks happened to fall between two writes.
DEFAULT_IDLE_WINDOW = timedelta(minutes=10)

#: How recently a session store must have been written for the session to
#: count as a candidate for a pane. This narrows an accumulated project
#: directory — which holds every session ever run there — to the ones that
#: could plausibly be the one on screen.
DEFAULT_ACTIVITY_WINDOW = timedelta(hours=1)


@dataclass(frozen=True)
class PaneVariables:
    """The iTerm2 session variables the join reads, plus the pane's identity.

    Attributes:
        pane_id: iTerm2's own session id for the pane.
        job_pid: The `jobPid` variable — the pid of the pane's foreground
            job. `None` when iTerm2 reports no job.
        job_name: The `jobName` variable — that job's name. Corroboration
            only; see fact 2 in the module docstring.
        path: The `path` variable — the shell's reported working directory.
            `None` when shell integration is not installed, which is one of
            the two ways this value goes missing (the other being an ssh
            hop, where it reports the local side).
    """

    pane_id: str
    job_pid: int | None
    job_name: str | None
    path: str | None


@dataclass(frozen=True)
class ProcessInfo:
    """One row of the process table, as the join needs it.

    Attributes:
        pid: The process id.
        ppid: Its parent's process id.
        name: The executable's basename, from `command`.
        command: The full command line as `ps` reported it.
    """

    pid: int
    ppid: int
    name: str
    command: str


#: A process table keyed by pid, as `read_process_table` returns.
ProcessTable = Mapping[int, ProcessInfo]


@dataclass(frozen=True)
class PaneJoin:
    """A pane, resolved to an agent process and the project it is working in.

    Attributes:
        pane_id: The pane this join is for.
        pid: The *agent's* pid, found by walking up from `jobPid`. Not
            `jobPid` itself, which is usually a descendant.
        source: The adapter source name, e.g. `"claude-code"`.
        cwd: The directory both the pane and the agent process agree on.
        project_key: `cwd` encoded the way the on-disk store names its
            project directory.
        session_candidates: Session ids under `project_key` whose stores
            were written within the activity window, sorted. Possibly empty
            — a pane can be joined to a project before its agent has written
            anything.
        session_key: The single candidate, or `None` when there is not
            exactly one. `None` is the ordinary outcome for a project with
            two panes open, and it is a refusal rather than a failure: no
            evidence available on either source distinguishes two same-project
            panes (fact 3 in the module docstring), so naming one would be
            the guess this module exists to avoid.
    """

    pane_id: str
    pid: int
    source: str
    cwd: Path
    project_key: str
    session_candidates: tuple[str, ...]
    session_key: str | None


def process_name(command: str) -> str:
    """Return the executable basename from a `ps` command line.

    Args:
        command: A full command line, e.g.
            `"/Users/…/node_modules/.bin/opencode serve --pure"`.

    Returns:
        The basename of its first token, with a login shell's leading `-`
        removed (`ps` reports the login shell as `-zsh`). Empty for an empty
        command line.

    Note:
        The first token is taken by whitespace, so an executable path
        containing a space yields a wrong basename. That fails *closed*: a
        wrong basename matches no entry in `AGENT_SOURCES`, so the pane
        reports no join. `ps -o comm=` is not used instead because macOS
        truncates it to sixteen characters — measured, on the `opencode`
        binary, which lives at a path long enough that the truncation
        removes the very name being matched.
    """
    first = command.strip().split(" ", 1)[0]
    if not first:
        return ""
    return Path(first).name.lstrip("-")


def read_process_table() -> ProcessTable:
    """Snapshot the whole process table in one `ps` call.

    One call rather than one per pid, and one snapshot rather than a live
    query per hop: the ancestry walk asks about several pids and the answers
    must describe the same instant, or the walk can follow a `ppid` into a
    slot that was reused between calls. Measured at ~25 ms for the full
    table on this machine, which is why there is no caching layer here.

    Returns:
        Every visible process, keyed by pid. Empty if `ps` is unavailable or
        fails — an empty table joins nothing, which is the safe direction.
    """
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return {}
    if completed.returncode != 0:
        return {}
    return parse_process_table(completed.stdout)


def parse_process_table(output: str) -> ProcessTable:
    """Parse `ps -axo pid=,ppid=,command=` output into a `ProcessTable`.

    Split from `read_process_table` so the parsing is testable without a
    subprocess, and so a test can build a table for a process tree that does
    not exist on the machine running the test.

    Args:
        output: The raw `ps` stdout.

    Returns:
        Every parseable row, keyed by pid. Unparseable rows are skipped
        rather than raised on: `ps` output is not a stable format, and one
        odd row must not cost the join every other pane on screen.
    """
    table: dict[int, ProcessInfo] = {}
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        raw_pid, raw_ppid, command = parts
        try:
            pid, ppid = int(raw_pid), int(raw_ppid)
        except ValueError:
            continue
        table[pid] = ProcessInfo(pid=pid, ppid=ppid, name=process_name(command), command=command)
    return table


def working_directory(pid: int) -> Path | None:
    """Return a process's own working directory, or `None`.

    macOS has no `/proc`, so this shells out to `lsof` for the one file
    descriptor that carries the answer. Restricted to `-d cwd` deliberately:
    the unrestricted form lists every open file in the process, which is
    both far slower and a great deal more than this needs to know.

    Args:
        pid: The process to ask about.

    Returns:
        Its working directory, or `None` if the process is gone, is not
        ours, or `lsof` is unavailable. `None` always means "could not
        determine" and never "no directory", so a caller must not read it as
        a mismatch.
    """
    try:
        completed = subprocess.run(
            ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("n"):
            return Path(line[1:])
    return None


def process_is_alive(pid: int) -> bool:
    """Report whether `pid` names a live process this user may signal.

    Uses signal 0, which performs the permission and existence checks and
    delivers nothing (INV-2: Palaver never interrupts an observed session).

    Args:
        pid: The process to check.

    Returns:
        True only for a live process owned by this user. A process owned by
        another user raises `PermissionError` and returns False, because it
        cannot be one of this user's agents whatever else it is.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError, ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def agent_ancestor(job_pid: int, table: ProcessTable) -> ProcessInfo | None:
    """Walk up from a pane's foreground job to the agent that spawned it.

    Starts *at* `job_pid`, because a pane running the agent directly — as
    codex does — needs no hops at all, and stops at the first login shell,
    because everything at or above it belongs to iTerm2 rather than to the
    pane's work.

    Args:
        job_pid: The pane's `jobPid`.
        table: A process table snapshot, from `read_process_table`.

    Returns:
        The nearest ancestor (or `job_pid` itself) whose name is in
        `AGENT_SOURCES`, or `None` if the walk reaches a shell, leaves the
        table, or exhausts `MAX_ANCESTRY_HOPS` first.
    """
    pid = job_pid
    for _ in range(MAX_ANCESTRY_HOPS):
        info = table.get(pid)
        if info is None:
            return None
        if info.name in AGENT_SOURCES:
            return info
        if info.name in SHELL_NAMES:
            return None
        if info.ppid == pid or info.ppid <= 1:
            return None
        pid = info.ppid
    return None


def project_key_for_cwd(cwd: Path) -> str:
    """Encode a working directory the way the on-disk store names its project.

    Claude Code writes `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`,
    where the encoding replaces both `/` and `.` with `-`. Verified against
    real directories on this machine covering the plain case, a dotfile
    directory (`~/.launchd` → `-Users-dave--launchd`, note the doubled dash),
    and an underscore in a project name (preserved).

    The encoding is deliberately not invertible and no inverse is offered:
    `-Users-dave--launchd` has several pre-images, and a decoder would have
    to guess between them. Callers go in this direction only, checking that
    the encoded directory exists rather than decoding one that does.

    Args:
        cwd: An absolute working directory.

    Returns:
        The encoded project directory name.
    """
    return str(cwd).replace("/", "-").replace(".", "-")


def session_candidates(
    project_key: str,
    sessions_root: Path,
    *,
    now: datetime,
    activity_window: timedelta = DEFAULT_ACTIVITY_WINDOW,
) -> tuple[str, ...]:
    """Name the sessions in a project that could be the one on screen.

    Reads directory entries and mtimes only — never a byte of any transcript
    (INV-9).

    Args:
        project_key: The encoded project directory name.
        sessions_root: The store root, e.g. `~/.claude/projects`.
        now: Reference time the window is measured back from. Compared
            against store mtimes, which are absolute epoch seconds, so this
            is converted with `.timestamp()` and follows its rule: an aware
            datetime is exact, a naive one is read as *local* time. Passing a
            naive UTC clock therefore shifts the cutoff by the UTC offset —
            silently, since it still returns a plausible-looking list. The
            rest of the tree (`collect_status`) is aware-UTC and callers
            should stay that way; `join_pane`'s default does.
        activity_window: How recently a store must have been written.

    Returns:
        Session ids (store filename stems) written within the window,
        sorted. Empty when the project directory is absent or nothing in it
        is recent.
    """
    directory = sessions_root / project_key
    if not directory.is_dir():
        return ()
    cutoff = (now - activity_window).timestamp()
    found = []
    for store in directory.glob("*.jsonl"):
        try:
            mtime = store.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            found.append(store.stem)
    return tuple(sorted(found))


def join_pane(
    variables: PaneVariables,
    *,
    table: ProcessTable | None = None,
    cwd_reader=working_directory,
    sessions_root: Path | None = None,
    now: datetime | None = None,
    activity_window: timedelta = DEFAULT_ACTIVITY_WINDOW,
) -> PaneJoin | None:
    """Resolve a pane to its agent process and project, or refuse.

    Every refusal below returns `None`. None of them is an error condition:
    most panes on a machine are not running an agent, and a pane that is
    running one over ssh is correctly unjoinable from this side.

    The checks, in order — each is cheap before the one after it, and each
    is a thing that can be *shown* rather than assumed:

    1. `path` is present, absolute, and an existing directory here. Absent
       is the no-shell-integration case; a path that does not exist locally
       is the ssh case reported from the wrong side.
    2. `job_pid` is present and positive.
    3. `job_pid` is in the process table. A pid that vanished between
       iTerm2 publishing the variable and this read is exactly the "matches
       no agent process" case.
    4. `job_name` matches that row's name. They disagree only when the pane
       variables are stale relative to the process table, and a stale
       `jobPid` is a pid that may since have been reused.
    5. An agent is found at or above `job_pid`, below the login shell.
    6. The agent's own working directory is readable **and equal to**
       `path`. This is the check the join actually rests on, and the only
       one that corroborates a single fact from two independent sources.
    7. The encoded project directory exists under `sessions_root`.

    Args:
        variables: The pane's iTerm2 variables.
        table: A process table snapshot; read fresh when `None`. Injectable
            so a test can describe a process tree that is not running.
        cwd_reader: Callable taking a pid and returning its working
            directory or `None`. Injectable for the same reason.
        sessions_root: The store root; defaults to `~/.claude/projects`.
        now: Reference time for the candidate window; defaults to now, UTC.
        activity_window: How recently a candidate's store must have been
            written.

    Returns:
        A `PaneJoin`, or `None` if any check above fails.
    """
    if not variables.path:
        return None
    cwd = Path(variables.path)
    if not cwd.is_absolute() or not cwd.is_dir():
        return None

    if variables.job_pid is None or variables.job_pid <= 0:
        return None

    process_table = read_process_table() if table is None else table
    job = process_table.get(variables.job_pid)
    if job is None:
        return None

    if variables.job_name and job.name != variables.job_name:
        return None

    agent = agent_ancestor(variables.job_pid, process_table)
    if agent is None:
        return None

    agent_cwd = cwd_reader(agent.pid)
    if agent_cwd is None or agent_cwd != cwd:
        return None

    root = sessions_root if sessions_root is not None else Path.home() / ".claude" / "projects"
    project_key = project_key_for_cwd(cwd)
    if not (root / project_key).is_dir():
        return None

    if now is None:
        now = datetime.now(timezone.utc)
    candidates = session_candidates(project_key, root, now=now, activity_window=activity_window)
    return PaneJoin(
        pane_id=variables.pane_id,
        pid=agent.pid,
        source=AGENT_SOURCES[agent.name],
        cwd=cwd,
        project_key=project_key,
        session_candidates=candidates,
        session_key=candidates[0] if len(candidates) == 1 else None,
    )


def observe_liveness(
    pid: int | None,
    *,
    last_advance: datetime | None,
    now: datetime,
    idle_window: timedelta = DEFAULT_IDLE_WINDOW,
    alive_probe=process_is_alive,
) -> Liveness:
    """Build the `Liveness` the status layer consumes.

    Args:
        pid: The agent's pid from a `PaneJoin`, or `None` when there is no
            join — no pane, a refused join, or a headless observation.
        last_advance: When this session's store was last seen to grow, or
            `None` if Palaver has never recorded an advance for it.
        now: Reference time the idle window is measured back from.
        idle_window: How long a store must go unwritten to count as quiet.
        alive_probe: Callable taking a pid and returning whether it is live.
            Injectable so a test can describe a dead process without having
            to create and reap one.

    Returns:
        A `Liveness`. `pid=None` yields `process_alive=UNKNOWN`, never
        `FALSE`: no join was attempted, so nothing was observed to be gone.
        `last_advance=None` yields `cursor_advanced_recently=UNKNOWN` for the
        same reason — a session Palaver has not yet watched advance is not a
        session that has been quiet.
    """
    if pid is None:
        alive = Tri.UNKNOWN
    else:
        alive = Tri.TRUE if alive_probe(pid) else Tri.FALSE

    if last_advance is None:
        advanced = Tri.UNKNOWN
    else:
        advanced = Tri.TRUE if now - last_advance <= idle_window else Tri.FALSE

    return Liveness(process_alive=alive, cursor_advanced_recently=advanced)
