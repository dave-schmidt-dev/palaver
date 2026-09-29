"""
Narrowing an ambiguous Codex candidate set: the cwd+mtime scan, live-fd narrowing,
metadata-progress narrowing, pid-reuse refusal, and subagent exclusion.
"""

from __future__ import annotations

import os

from palaver.ui.pane_join import (
    CODEX_SOURCE,
    PaneVariables,
    ProcessInfo,
    join_pane,
)
from tests._pane_join_support import NOW, _codex_rollout, _table


def test_codex_join_requires_one_exact_recent_root_rollout(tmp_path):
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    store = _codex_rollout(root, cwd, "rollout-root")

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
    )
    assert joined is not None
    assert joined.source == CODEX_SOURCE
    assert joined.session_key == store.stem
    assert joined.store_path == store.resolve()

    _codex_rollout(root, cwd, "rollout-second")
    ambiguous = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
    )
    assert ambiguous is not None
    assert ambiguous.session_key is None
    assert ambiguous.store_path is None


def test_codex_ambiguity_narrows_to_the_rollout_the_live_pid_still_has_open(tmp_path):
    """The measured fix: a stale rollout cannot be what the live pid is writing.

    Reported live (2026-08-19): a project with three recent rollouts left its
    pane UNJOINED even though only one of them was still open by the pid the
    pane actually resolved to -- the other two were from codex invocations
    that had already exited, still inside the one-hour activity window. A
    candidate the live pid does not hold open is excluded from the write it
    cannot be performing, narrowing what the cwd+mtime scan alone could not.
    """
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    live = _codex_rollout(root, cwd, "rollout-live")
    _codex_rollout(root, cwd, "rollout-exited")

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
        open_files_reader=lambda _pid: frozenset({live.resolve()}),
    )
    assert joined is not None
    assert joined.session_key == live.stem
    assert joined.store_path == live.resolve()


def test_codex_ambiguity_is_not_resolved_by_an_empty_or_multi_member_open_set(tmp_path):
    """The narrowing only trusts an intersection that lands on exactly one.

    Fact 3 in the module docstring measured ten rollouts open on a single
    codex pid at once, so an fd being open proves nothing by itself -- only
    ruling every candidate but one *out* is trusted. Neither "the live pid
    holds nothing under this project open" (an untracked file, or `lsof`
    unavailable) nor "the live pid holds every candidate open" (the ten-
    rollout shape) may resolve the pane; both must still refuse exactly as
    the plain mtime scan did before this fix existed.
    """
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    first = _codex_rollout(root, cwd, "rollout-a")
    second = _codex_rollout(root, cwd, "rollout-b")

    def _join_with(open_paths):
        return join_pane(
            variables,
            table=table,
            cwd_reader=lambda _pid: cwd,
            store_roots={CODEX_SOURCE: root},
            now=NOW,
            open_files_reader=lambda _pid: open_paths,
        )

    empty = _join_with(frozenset())
    assert empty is not None
    assert empty.session_key is None

    both_open = _join_with(frozenset({first.resolve(), second.resolve()}))
    assert both_open is not None
    assert both_open.session_key is None


def test_codex_ambiguity_narrows_only_after_one_candidate_advances(tmp_path):
    """A second metadata observation can identify the active rollout."""
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    live = _codex_rollout(root, cwd, "rollout-live")
    historical = _codex_rollout(root, cwd, "rollout-historical")
    open_paths = frozenset({live.resolve(), historical.resolve()})
    progress = {}

    first = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
        open_files_reader=lambda _pid: open_paths,
        candidate_progress=progress,
    )
    assert first is not None
    assert first.session_key is None

    os.utime(live, ns=(live.stat().st_atime_ns, live.stat().st_mtime_ns + 1_000_000))
    second = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
        open_files_reader=lambda _pid: open_paths,
        candidate_progress=progress,
    )
    assert second is not None
    assert second.session_key == live.stem
    assert second.store_path == live.resolve()


def test_codex_progress_narrowing_refuses_zero_multiple_or_unstable_advances(tmp_path):
    """Progress is evidence only when exactly one stable candidate advances."""
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    first = _codex_rollout(root, cwd, "rollout-a")
    second = _codex_rollout(root, cwd, "rollout-b")
    open_paths = frozenset({first.resolve(), second.resolve()})

    def _join(progress):
        return join_pane(
            variables,
            table=table,
            cwd_reader=lambda _pid: cwd,
            store_roots={CODEX_SOURCE: root},
            now=NOW,
            open_files_reader=lambda _pid: open_paths,
            candidate_progress=progress,
        )

    zero_progress = {}
    assert _join(zero_progress).session_key is None
    assert _join(zero_progress).session_key is None

    multiple_progress = {}
    assert _join(multiple_progress).session_key is None
    for path in (first, second):
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert _join(multiple_progress).session_key is None

    unstable_progress = {}
    assert _join(unstable_progress).session_key is None
    old_mtime = first.stat().st_mtime_ns
    os.utime(first, ns=(first.stat().st_atime_ns, old_mtime - 1_000_000))
    os.utime(second, ns=(second.stat().st_atime_ns, second.stat().st_mtime_ns + 1_000_000))
    assert _join(unstable_progress).session_key is None


def test_codex_ambiguity_narrowing_asks_about_the_agent_pid_not_the_job_pid(tmp_path):
    """The two fixes in this diff interact: narrowing must ask about `agent.pid`.

    Before the walk-through-a-task-shell fix, a codex pane's `jobPid` and
    agent pid were always the same pid -- codex is its own foreground job.
    The task-shell fix makes them diverge for exactly the shape it was
    written to rescue (`bash test-gate.sh` between the pane's job and
    `codex`, as in the walk test above), and nothing before this test caught
    an `open_files_reader` call keyed on the wrong pid: every other narrowing
    test discards its `_pid` argument, so `job_pid` and `agent.pid` were
    indistinguishable there. Asking `lsof` about the task shell instead of
    the agent would silently return nothing and leave a rescuable pane
    ambiguous again -- so this asserts both the join result and the exact
    pid `open_files_reader` was called with.
    """
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(
        (
            (700, 600, "python3 scripts/hybrid_verify.py"),
            (600, 500, "/bin/bash ./app/test-gate.sh --phase native"),
            (500, 400, "codex"),
            (400, 1, "-zsh"),
        )
    )
    variables = PaneVariables("codex-pane", 700, "python3", str(cwd))
    live = _codex_rollout(root, cwd, "rollout-live")
    _codex_rollout(root, cwd, "rollout-exited")
    asked_pids: list[int] = []

    def _record_and_narrow(pid):
        asked_pids.append(pid)
        return frozenset({live.resolve()})

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
        open_files_reader=_record_and_narrow,
    )
    assert joined is not None
    assert joined.pid == 500
    assert joined.session_key == live.stem
    assert asked_pids == [500], "narrowing must ask about the agent pid, not jobPid"


def test_codex_pid_reuse_refuses_descriptor_narrowing(tmp_path):
    """A fresh process row for the same pid must not select its open files."""
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    live = _codex_rollout(root, cwd, "rollout-live")
    _codex_rollout(root, cwd, "rollout-exited")
    reused = dict(table)
    old = reused[77201]
    reused[77201] = ProcessInfo(
        pid=old.pid,
        ppid=old.ppid,
        name=old.name,
        command=old.command,
        start_time="Fri Aug 15 12:01:00 2026",
    )

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
        process_table_reader=lambda: reused,
        open_files_reader=lambda _pid: frozenset({live.resolve()}),
    )

    assert joined is not None
    assert joined.session_key is None
    assert joined.store_path is None


def test_codex_join_excludes_identity_marked_subagents(tmp_path):
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    root = tmp_path / "codex-sessions"
    _codex_rollout(root, cwd, "rollout-child", subagent=True)
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(cwd))
    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: root},
        now=NOW,
    )
    assert joined is not None
    assert joined.session_key is None
