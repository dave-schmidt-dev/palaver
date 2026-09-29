"""The pane, process-table, and join-result records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
    pin: str | None = None


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
    start_time: str | None = None


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
            anything. Narrowed to a single id when a source-specific
            discriminator names one: Claude Code's live-process registry
            (see `registered_session`), or, for Codex, an ambiguous mtime
            scan narrowed by which rollout the agent pid still holds open
            (see `_agent_open_store_paths`). Otherwise the full recent-store
            list, exactly as documented above.
        session_key: The single candidate, or `None` when there is not
            exactly one. `None` is the ordinary outcome for a project with
            two panes open and no discriminator able to name one, and it is
            a refusal rather than a failure: guessing among genuinely
            equally-recent, equally-open candidates is the wrong answer this
            module exists to avoid.
    """

    pane_id: str
    pid: int
    source: str
    cwd: Path
    project_key: str
    session_candidates: tuple[str, ...]
    session_key: str | None
    store_path: Path | None = None


@dataclass(frozen=True)
class SupportedPaneProcess:
    """A locally-running supported agent, before any transcript is joined.

    Companion panes need only this much evidence to exist.  Keeping this
    result separate from :class:`PaneJoin` prevents an absent or ambiguous
    transcript from hiding a real Claude Code or Codex process, while still
    refusing shells, remote processes, stale pane variables, and cwd
    disagreements.
    """

    pane_id: str
    pid: int
    source: str
    cwd: Path
