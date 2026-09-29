"""
Reading and parsing the OS process table, walking it to find an agent, and confirming a
pid's identity did not change between two reads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .constants import AGENT_SOURCES, MAX_ANCESTRY_HOPS, SHELL_NAMES
from .records import ProcessInfo, ProcessTable


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
    return Path(first).name.lstrip("-").lower()


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
            ["/bin/ps", "-axo", "pid=,ppid=,lstart=,command="],
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
    """Parse `ps -axo pid=,ppid=,lstart=,command=` into a `ProcessTable`.

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
        raw_pid, raw_ppid, remainder = parts
        try:
            pid, ppid = int(raw_pid), int(raw_ppid)
        except ValueError:
            continue
        # `lstart` is a fixed-width 24-character ctime value on macOS.
        # Accept the historical three-column fixture shape as well; those
        # rows simply cannot be used to prove a pid was not reused.
        start_time = None
        command = remainder
        if len(remainder) > 25 and remainder[24].isspace():
            candidate = remainder[:24]
            if candidate[3:4].isspace() and candidate[7:8].isspace():
                start_time = candidate
                command = remainder[25:].strip()
        table[pid] = ProcessInfo(
            pid=pid,
            ppid=ppid,
            name=process_name(command),
            command=command,
            start_time=start_time,
        )
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


def _is_login_shell(info: ProcessInfo) -> bool:
    """Whether `info` is a login shell rather than a task shell spawned to run a command.

    `ps` reports a login shell's `argv[0]` with a leading `-` — literally
    `-zsh`, never a path (the `-zsh` row throughout this module's fixtures).
    Confirmed against this machine's live process table (2026-08-19): both a
    `bash script.sh` invocation and the `/bin/zsh -c "..."` shell Claude Code
    and Codex spawn for every tool command report their ordinary name with no
    leading dash. That is the only signal here that distinguishes the two —
    `info.name` alone cannot, since `process_name` strips the dash before
    either row reaches this function.
    """
    first = info.command.strip().split(" ", 1)[0]
    return first.startswith("-")


def agent_ancestor(job_pid: int, table: ProcessTable) -> ProcessInfo | None:
    """Walk up from a pane's foreground job to the agent that spawned it.

    Starts *at* `job_pid`, because a pane running the agent directly — as
    codex does — needs no hops at all, and stops at the pane's login shell,
    because everything at or above it belongs to iTerm2 rather than to the
    pane's work. A *task* shell the agent spawned along the way — to run the
    very command whose pane is being joined — is walked through rather than
    treated as that boundary; see `_is_login_shell`.

    This trades one false-negative for a narrow false-positive: any pane
    whose own shell is not login-prefixed loses the boundary this walk
    depends on, not only a shell the *user* starts by hand mid-pane — an
    iTerm2 profile with a custom Command instead of "Login shell", or a
    non-login `tmux`/`screen` default-command, has the same shape. If an
    unrelated agent happens to sit above such a shell in that same process
    tree, the walk climbs through and misjoins. Accepted deliberately: on
    this machine every pane's own shell is `-zsh` under `login`, which still
    ends the walk unconditionally, so the exposure is nil in the configuration
    this module was built against, and an agent shelling out to run its own
    command is the far more common shape either way. Both callers of this
    walk share the trade-off — `join_pane` for status, and
    `detect_supported_process` for whether a companion pane is created at
    all — though the `agent_cwd == pane cwd` check both perform bounds how
    far a misjoin can reach.

    Args:
        job_pid: The pane's `jobPid`.
        table: A process table snapshot, from `read_process_table`.

    Returns:
        The nearest ancestor (or `job_pid` itself) whose name is in
        `AGENT_SOURCES`, or `None` if the walk reaches the login shell, leaves
        the table, or exhausts `MAX_ANCESTRY_HOPS` first.
    """
    pid = job_pid
    for _ in range(MAX_ANCESTRY_HOPS):
        info = table.get(pid)
        if info is None:
            return None
        if info.name.lstrip("-").lower() in AGENT_SOURCES:
            return info
        if info.name in SHELL_NAMES and (info.name == "login" or _is_login_shell(info)):
            return None
        if info.ppid == pid or info.ppid <= 1:
            return None
        pid = info.ppid
    return None


def _same_process_identity(expected: ProcessInfo, actual: ProcessInfo | None) -> bool:
    """Return whether a fresh process row still names the same process.

    PID reuse is the only way the descriptor scan can become evidence about
    a different process after the ancestry walk. `lstart` distinguishes the
    otherwise identical pid/command case; a caller without that field must
    refuse narrowing rather than pretend the weaker comparison proves it.
    """
    return (
        actual is not None
        and (
            expected.pid,
            expected.ppid,
            expected.name,
            expected.command,
            expected.start_time,
        )
        == (
            actual.pid,
            actual.ppid,
            actual.name,
            actual.command,
            actual.start_time,
        )
        and expected.start_time is not None
    )
