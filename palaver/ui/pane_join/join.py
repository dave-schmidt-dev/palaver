"""
The two join entry points: a light process-only detector, and the full pane-to-session
join.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .candidates import project_key_for_cwd, session_candidates
from .constants import AGENT_SOURCES, CODEX_SOURCE, DEFAULT_ACTIVITY_WINDOW
from .pin import PanePin, parse_pin
from .process import (
    _same_process_identity,
    agent_ancestor,
    process_name,
    read_process_table,
    working_directory,
)
from .records import PaneJoin, PaneVariables, ProcessTable, SupportedPaneProcess
from .stores import (
    CodexCandidateProgress,
    _agent_open_store_paths,
    _codex_homes_from_open_paths,
    _codex_store_candidates,
    _narrow_codex_candidates_by_progress,
    _pinned_store_path,
    _readable_file,
    _registered_store,
    _root_for_source,
    registered_session,
)


def detect_supported_process(
    variables: PaneVariables,
    *,
    table: ProcessTable | None = None,
    cwd_reader=None,
) -> SupportedPaneProcess | None:
    """Identify a supported local agent without consulting session stores.

    The checks deliberately match the process half of :func:`join_pane`.
    This is a fail-closed detector, not a weaker join: a pane must name an
    existing local directory, its foreground process variables must agree
    with one process-table snapshot, and the supported agent's own cwd must
    equal the pane cwd.
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
    pid_is_agent = job.name.lstrip("-").lower() in AGENT_SOURCES
    if variables.job_name and job.name != process_name(variables.job_name) and not pid_is_agent:
        return None

    agent = agent_ancestor(variables.job_pid, process_table)
    if agent is None:
        return None
    read_cwd = working_directory if cwd_reader is None else cwd_reader
    agent_cwd = read_cwd(agent.pid)
    if agent_cwd is None or agent_cwd != cwd:
        return None
    return SupportedPaneProcess(
        pane_id=variables.pane_id,
        pid=agent.pid,
        source=AGENT_SOURCES[agent.name.lstrip("-").lower()],
        cwd=cwd,
    )


def join_pane(
    variables: PaneVariables,
    *,
    table: ProcessTable | None = None,
    cwd_reader=working_directory,
    sessions_root: Path | None = None,
    store_roots: Mapping[str, Path] | None = None,
    now: datetime | None = None,
    activity_window: timedelta = DEFAULT_ACTIVITY_WINDOW,
    pin: PanePin | Mapping[str, object] | str | None = None,
    candidate_cache: MutableMapping[tuple[str, str, str], tuple[Path, ...]] | None = None,
    registry_root: Path | None = None,
    open_files_reader: Callable[[int], frozenset[Path]] = _agent_open_store_paths,
    process_table_reader: Callable[[], ProcessTable] | None = None,
    candidate_progress: CodexCandidateProgress | None = None,
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
       `path`, unless a validated explicit pin is present. A pin is the
       deliberate rename/move recovery escape hatch, not an automatic guess.
    7. For Claude Code, the live-process registry names this pid's own
       session and that transcript is readable. Failing that, the source's
       own store layout supplies exactly one candidate. For Codex, the
       cwd+mtime scan supplies exactly one candidate, or — failing that —
       narrows to exactly one by which rollout the agent pid still holds
       open (`_agent_open_store_paths`).

    Args:
        variables: The pane's iTerm2 variables.
        table: A process table snapshot; read fresh when `None`. Injectable
            so a test can describe a process tree that is not running.
        cwd_reader: Callable taking a pid and returning its working
            directory or `None`. Injectable for the same reason.
        sessions_root: Legacy single-root injection. Prefer `store_roots`.
        store_roots: Explicit independent roots keyed by source. Omitting a
            source disables only that source.
        now: Reference time for the candidate window; defaults to now, UTC.
        activity_window: How recently a candidate's store must have been
            written.
        registry_root: Claude Code's live-process registry directory;
            defaults to the real one. Injectable so a test can describe a
            registry that is not running.
        open_files_reader: Callable taking a pid and returning the resolved
            paths it holds open, used to narrow an ambiguous Codex candidate
            set. Injectable for the same reason as `cwd_reader`.
        process_table_reader: Fresh process-table reader used immediately
            before descriptor narrowing. A changed pid identity leaves the
            candidate set ambiguous instead of using another process's file
            descriptors. Tests that provide a static `table` may provide the
            matching static reader too.
        candidate_progress: Optional per-pane metadata observations used to
            narrow an ambiguous Codex set after a later tick. The mapping is
            updated with current size and mtime only; transcript content is
            never read.

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

    pid_is_agent = job.name.lstrip("-").lower() in AGENT_SOURCES
    if variables.job_name and job.name != process_name(variables.job_name) and not pid_is_agent:
        return None

    agent = agent_ancestor(variables.job_pid, process_table)
    if agent is None:
        return None

    source = AGENT_SOURCES[agent.name.lstrip("-").lower()]
    raw_pin = variables.pin if pin is None else pin
    parsed_pin = raw_pin if isinstance(raw_pin, PanePin) else parse_pin(raw_pin)
    if parsed_pin is None and raw_pin not in (None, ""):
        return None
    if parsed_pin is not None and parsed_pin.source != source:
        return None

    agent_cwd = cwd_reader(agent.pid)
    if agent_cwd is None or (agent_cwd != cwd and parsed_pin is None):
        return None

    fresh_table: ProcessTable | None = None
    same_process: bool | None = None
    open_paths: frozenset[Path] | None = None

    if source == CODEX_SOURCE and store_roots is None and sessions_root is None:
        fresh_table = (
            (process_table if table is not None else read_process_table())
            if process_table_reader is None
            else process_table_reader()
        )
        same_process = _same_process_identity(agent, fresh_table.get(agent.pid))
        if not same_process:
            return None
        try:
            open_paths = open_files_reader(agent.pid)
        except OSError:
            return None
        # File descriptors are evidence only for the process that was just
        # checked. Re-read after lsof so a reused pid cannot select another
        # Codex home between the first check and path discovery.
        fresh_table = (
            process_table_reader()
            if process_table_reader is not None
            else read_process_table() if table is None else table
        )
        same_process = _same_process_identity(agent, fresh_table.get(agent.pid))
        if not same_process:
            return None
        homes = _codex_homes_from_open_paths(open_paths)
        if len(homes) > 1:
            return None
        if len(homes) == 1:
            root = (next(iter(homes)) / "sessions").resolve(strict=False)
        else:
            root = _root_for_source(source, sessions_root=None, store_roots=None)
    else:
        root = _root_for_source(source, sessions_root=sessions_root, store_roots=store_roots)
    if root is None:
        return None

    if now is None:
        now = datetime.now(timezone.utc)
    if parsed_pin is not None:
        store_path = _pinned_store_path(root, parsed_pin)
        if store_path is None:
            return None
        candidates = (parsed_pin.session_key,)
        project_key = project_key_for_cwd(cwd)
    elif source == CODEX_SOURCE:
        codex_paths = _codex_store_candidates(
            root, cwd, now=now, activity_window=activity_window, cache=candidate_cache
        )
        if len(codex_paths) > 1:
            # More than one recent rollout in this project is often several
            # sessions run here within the hour, not one -- most are from
            # processes that have since exited. Narrow to the one(s) the
            # live agent pid still holds open; a stale rollout cannot be.
            if fresh_table is None:
                fresh_table = (
                    (process_table if table is not None else read_process_table())
                    if process_table_reader is None
                    else process_table_reader()
                )
                same_process = _same_process_identity(agent, fresh_table.get(agent.pid))
            if same_process:
                if open_paths is None:
                    try:
                        open_paths = open_files_reader(agent.pid)
                    except OSError:
                        open_paths = frozenset()
                narrowed = tuple(path for path in codex_paths if path in open_paths)
                if len(narrowed) == 1:
                    codex_paths = narrowed
            if len(codex_paths) > 1 and candidate_progress is not None and same_process:
                progressed = _narrow_codex_candidates_by_progress(
                    codex_paths, candidate_progress, identity=(agent.pid, cwd)
                )
                if len(progressed) == 1:
                    codex_paths = progressed
            elif candidate_progress is not None and not same_process:
                candidate_progress.clear()
        candidates = tuple(path.stem for path in codex_paths)
        project_key = project_key_for_cwd(cwd)
        store_path = codex_paths[0] if len(codex_paths) == 1 else None
    else:
        project_key = project_key_for_cwd(cwd)
        # Claude Code names its own live session, which the mtime window
        # cannot: a project run several times within the hour offers several
        # equally recent stores and the scan below correctly refuses all of
        # them. The registry is consulted first and its answer is exact.
        named = registered_session(agent.pid, cwd, registry_root=registry_root)
        store_path = None if named is None else _registered_store(root, cwd, named)
        if store_path is not None:
            candidates = (named,)
            project_key = store_path.parent.name
        else:
            if not (root / project_key).is_dir():
                return None
            candidates = session_candidates(
                project_key, root, now=now, activity_window=activity_window
            )
            store_path = (
                (root / project_key / f"{candidates[0]}.jsonl").resolve(strict=False)
                if len(candidates) == 1
                else None
            )
            if store_path is not None and not _readable_file(store_path):
                return None
    return PaneJoin(
        pane_id=variables.pane_id,
        pid=agent.pid,
        source=source,
        cwd=cwd,
        project_key=project_key,
        session_candidates=candidates,
        session_key=candidates[0] if len(candidates) == 1 else None,
        store_path=store_path,
    )
