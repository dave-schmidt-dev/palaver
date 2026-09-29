"""
The ordered pre-checks join_pane makes before resolving a session: path, job identity,
agent presence, and cwd agreement.
"""

from __future__ import annotations

from pathlib import Path

from palaver.ui.pane_join import (
    CLAUDE_SOURCE,
    CODEX_SOURCE,
    PaneVariables,
    detect_supported_process,
    join_pane,
    project_key_for_cwd,
)
from tests._pane_join_support import AGENT_PID, JOB_PID, NOW, _join, _table, _variables
from tests._pane_join_support import project as project

# --- the join: the baseline must actually join -------------------------------


def test_the_baseline_pane_joins_to_the_agent_three_hops_above_its_job(project):
    """The positive control every refusal below is measured against.

    Asserts the agent pid rather than merely that a join happened, because
    the failure this whole module exists to prevent — joining to `jobPid`
    itself — also produces a non-`None` result.
    """
    cwd, sessions_root = project

    join = _join(cwd, sessions_root)

    assert join is not None
    assert join.pid == AGENT_PID
    assert join.pid != JOB_PID, "joined to the foreground job, not to the agent"
    assert join.source == "claude-code"
    assert join.cwd == cwd
    assert join.project_key == project_key_for_cwd(cwd)
    assert join.pane_id == "pane-1"


def test_a_pane_running_the_agent_directly_joins_at_zero_hops(project):
    """Codex is its own foreground job, so the walk must accept its start pid.

    A walk that always climbed at least one hop would pass every Claude Code
    test and silently fail for codex, whose measured `jobPid` *is* the agent.
    """
    cwd, sessions_root = project
    table = _table(((77201, 62921, "codex resume 019ff309"), (62921, 1, "-zsh")))

    join = _join(cwd, sessions_root, table=table, job_pid=77201, job_name="codex")

    assert join is not None
    assert join.pid == 77201
    assert join.source == "codex"


def test_supported_process_detection_does_not_require_a_transcript(project):
    cwd, _sessions_root = project

    detected = detect_supported_process(
        _variables(cwd), table=_table(), cwd_reader=lambda _pid: cwd
    )

    assert detected is not None
    assert detected.pane_id == "pane-1"
    assert detected.pid == AGENT_PID
    assert detected.source == CLAUDE_SOURCE
    assert detected.cwd == cwd


def test_supported_process_detection_fails_closed_on_cwd_disagreement(project, tmp_path):
    cwd, _sessions_root = project
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert (
        detect_supported_process(_variables(cwd), table=_table(), cwd_reader=lambda _pid: elsewhere)
        is None
    )


def test_direct_codex_pid_can_override_a_foreground_helper_job_name(project):
    """iTerm may report a helper as jobName while jobPid is Codex itself."""
    cwd, sessions_root = project
    table = _table(((77201, 62921, "Codex"), (62921, 1, "-zsh")))

    join = _join(
        cwd,
        sessions_root,
        table=table,
        job_pid=77201,
        job_name="SkyComputerUseCl",
    )

    assert join is not None
    assert join.source == CODEX_SOURCE
    assert join.pid == 77201


# --- the join: an unreliable path --------------------------------------------


def test_a_pane_whose_path_is_absent_returns_no_join_rather_than_a_guess(project):
    """Done-when: an absent `path` yields no join, not a fallback.

    The tempting fallback is the agent process's own cwd, which is readable
    here and would produce a plausible join. It is refused: `path` absent
    means shell integration is not reporting, and a pane Palaver cannot
    corroborate is one it must not label. Both the missing and the empty
    forms are asserted, since iTerm2 returns either.
    """
    cwd, sessions_root = project

    assert _join(cwd, sessions_root, path="") is None

    # `cwd_reader` still answers, so the refusal is about `path` and not
    # about the agent's directory being unavailable.
    assert (
        join_pane(
            PaneVariables(pane_id="pane-1", job_pid=JOB_PID, job_name="node", path=None),
            table=_table(),
            cwd_reader=lambda pid: cwd,
            sessions_root=sessions_root,
            now=NOW,
        )
        is None
    )

    # Positive control: the same call with `path` present joins.
    assert _join(cwd, sessions_root) is not None


def test_a_path_the_agent_does_not_agree_with_returns_no_join(project):
    """The ssh case that survives every earlier check, and the stale-path case.

    A pane over ssh with a *local* agent process would pass the agent walk;
    what it cannot do is agree with that agent about the directory. The same
    check catches a `path` left stale by a `cd` the shell did not report.
    """
    cwd, sessions_root = project
    elsewhere = cwd.parent / "other"
    elsewhere.mkdir()

    assert _join(cwd, sessions_root, cwd_reader=lambda pid: elsewhere) is None

    # An unreadable agent cwd is a refusal too: `None` means "could not
    # determine", which is not evidence of agreement.
    assert _join(cwd, sessions_root, cwd_reader=lambda pid: None) is None

    # Positive control: agreement joins.
    assert _join(cwd, sessions_root, cwd_reader=lambda pid: cwd) is not None


def test_a_path_that_does_not_exist_here_returns_no_join(project):
    """A remote path reported from the local side names no local directory.

    The agent is made to *agree* with the bogus path here, which is the only
    version of this test that proves anything: with the usual reader the
    cwd-mismatch check refuses these panes first, so path validation could be
    deleted outright and the test would still pass. Mutation testing is how
    that was found — the mutant removing the check survived until this test
    stopped letting a later check do its work.
    """
    cwd, sessions_root = project

    def agrees_with_the_pane(pid, claimed):
        return join_pane(
            _variables(cwd, path=claimed),
            table=_table(),
            cwd_reader=lambda _pid: Path(claimed),
            sessions_root=sessions_root,
            now=NOW,
        )

    assert agrees_with_the_pane(AGENT_PID, "/srv/build/checkout") is None
    assert agrees_with_the_pane(AGENT_PID, "relative/path") is None

    # Positive control: the same call shape with a path that does exist here
    # joins, so the refusals above are about the directory and not about the
    # substituted reader.
    assert agrees_with_the_pane(AGENT_PID, str(cwd)) is not None

    # The case that reaches the check on its own merits: a store directory
    # outliving the working directory it was named for. A project deleted
    # from disk leaves its transcripts behind, and an agent started before
    # the deletion still reports the removed path as its cwd — so pane and
    # process agree, and the encoded project entry exists. Only the
    # directory check stands between that and a join to a project that is
    # gone.
    deleted = cwd.parent / "deleted-project"
    (sessions_root / project_key_for_cwd(deleted)).mkdir(parents=True)
    assert agrees_with_the_pane(AGENT_PID, str(deleted)) is None

    deleted.mkdir()
    assert agrees_with_the_pane(AGENT_PID, str(deleted)) is not None


def test_a_stale_job_name_returns_no_join(project):
    """`jobName` disagreeing with the process table means the variables are stale.

    A stale `jobPid` is a pid that may since have been reused by an
    unrelated process, so the join is refused rather than run against it.
    """
    cwd, sessions_root = project

    assert _join(cwd, sessions_root, job_name="claude") is None
    assert _join(cwd, sessions_root, job_name="node") is not None


def test_capitalized_python_job_name_joins_against_lowercase_parsed_row(project):
    """iTerm's capitalized `Python` jobName must join against a lowercase `python` row.

    iTerm2 may report `jobName` as `Python` while the process table parsed
    executable name is `python`. Both are normalized before comparison so that
    case differences do not cause the pane join to fail closed. A genuinely stale
    job name must still return no join.
    """
    cwd, sessions_root = project
    table = _table(
        (
            (63488, 63354, "python server.py"),
            (63354, 62921, "claude"),
            (62921, 1, "-zsh"),
        )
    )

    join = _join(cwd, sessions_root, table=table, job_pid=63488, job_name="Python")
    assert join is not None
    assert join.pid == 63354
    assert join.source == "claude-code"

    # A genuinely stale job name returns no join
    assert _join(cwd, sessions_root, table=table, job_pid=63488, job_name="ruby") is None


def test_a_project_with_no_store_directory_returns_no_join(project, tmp_path):
    """An agent working somewhere Palaver has no store for is not joinable."""
    cwd, sessions_root = project
    empty_root = tmp_path / "empty-store"
    empty_root.mkdir()

    assert _join(cwd, empty_root) is None
    assert _join(cwd, sessions_root) is not None
