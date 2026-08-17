"""Task 5.2: the pane-to-agent join, and the liveness layer over status.

Two halves, and they fail in opposite directions, so they are tested
differently.

The **join** half is about refusing. Almost every assertion here is that
some pane produces `None`, and a function that returned `None`
unconditionally would pass all of them — so every refusal is paired with a
positive control that differs in exactly the field under test and *does*
join. Where the difference cannot be one field (the ancestry walk), the
control is the same process tree with one row's name changed.

The **liveness** half is about not over-claiming. `IDLE` is a positive
statement that a session has nothing to do, and the way it goes wrong is by
being reachable one branch too early. So the range is proved by exhausting
the full input space rather than by example, and `IDLE`'s reachability is
asserted as an exact count over that space, not as membership: a rule that
fired on `BLOCKED` as well as `AWAITING_HUMAN` still passes a membership
check.

The live tests at the end run against this machine's real process table.
They are what caught the three findings the module docstring records — most
of all that `jobPid` is not the agent — and a mocked-only suite would have
shipped a join that never joins anything.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from palaver.extract.persist import Extraction
from palaver.observer.signals import (
    LIVE_STATUS_RANGE,
    REFINED_STATUS_RANGE,
    SIGNAL_NAMES,
    Liveness,
    Signals,
    Status,
    Tri,
    apply_liveness,
    derive_status,
    derive_status_with_liveness,
)
from palaver.ui.pane_join import (
    AGENT_SOURCES,
    CLAUDE_SOURCE,
    CODEX_SOURCE,
    DEFAULT_IDLE_WINDOW,
    MAX_ANCESTRY_HOPS,
    SHELL_NAMES,
    PaneVariables,
    ProcessInfo,
    agent_ancestor,
    encode_pin,
    join_pane,
    observe_liveness,
    parse_process_table,
    process_is_alive,
    process_name,
    project_key_for_cwd,
    read_process_table,
    session_candidates,
    working_directory,
)

NOW = datetime(2026, 8, 15, 12, 0, 0)

#: A pane running Claude Code, in the shape measured on this machine: the
#: foreground job is an MCP server the agent spawned, and the agent itself is
#: three hops up. Every join test builds from this so that "the pane's job is
#: not the agent" is the default case rather than an exotic one.
CLAUDE_TREE = (
    #  pid   ppid  command
    (63488, 63369, "node /Users/dave/.npm/_npx/abc/node_modules/.bin/playwright-mcp"),
    (63369, 63354, "npm exec @playwright/mcp@latest"),
    (63354, 62921, "claude"),
    (62921, 62920, "-zsh"),
    (62920, 83829, "/usr/bin/login -fpl dave /Applications/iTerm.app/Contents/MacOS/ShellLauncher"),
    (83829, 1, "/Users/dave/Library/Application Support/iTerm2/iTermServer-3.6.11"),
)

JOB_PID = 63488
AGENT_PID = 63354


def _table(rows=CLAUDE_TREE):
    """Build a process table from `(pid, ppid, command)` rows."""
    return {
        pid: ProcessInfo(pid=pid, ppid=ppid, name=process_name(command), command=command)
        for pid, ppid, command in rows
    }


@pytest.fixture
def project(tmp_path):
    """A working directory plus the store root whose project entry matches it.

    Returns a `(cwd, sessions_root)` pair already wired so `join_pane`
    succeeds — tests then break exactly one thing about it.
    """
    cwd = tmp_path / "Projects" / "palaver"
    cwd.mkdir(parents=True)
    sessions_root = tmp_path / "store"
    (sessions_root / project_key_for_cwd(cwd)).mkdir(parents=True)
    return cwd, sessions_root


def _variables(cwd, *, pane_id="pane-1", job_pid=JOB_PID, job_name="node", path=None):
    """Build pane variables defaulting to the measured Claude Code shape."""
    return PaneVariables(
        pane_id=pane_id,
        job_pid=job_pid,
        job_name=job_name,
        path=str(cwd) if path is None else path,
    )


def _join(cwd, sessions_root, *, table=None, cwd_reader=None, now=NOW, **kwargs):
    """Call `join_pane` with the agent's cwd reported as `cwd` by default.

    `now` is an explicit parameter rather than part of `**kwargs` so a test
    can pass `now=None` and reach `join_pane`'s own clock default. Folded into
    `**kwargs` it would be forwarded to `_variables` instead and never get
    near the code under test.
    """
    return join_pane(
        _variables(cwd, **kwargs),
        table=_table() if table is None else table,
        cwd_reader=(lambda pid: cwd) if cwd_reader is None else cwd_reader,
        sessions_root=sessions_root,
        now=now,
    )


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


def _codex_rollout(root: Path, cwd: Path, name: str, *, subagent: bool = False) -> Path:
    """Write metadata-only Codex rollout fixture data for pane identity tests."""
    path = root / "2026" / "08" / "15" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": f"{name}-child" if subagent else name,
        "session_id": name if not subagent else f"{name}-root",
        "cwd": str(cwd),
    }
    path.write_text(json.dumps({"type": "session_meta", "payload": payload}) + "\n")
    os.utime(path, (NOW.timestamp(), NOW.timestamp()))
    return path


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


def test_codex_pin_recovers_a_rollout_after_the_pane_moves(tmp_path):
    old_cwd = tmp_path / "old-codex-project"
    live_cwd = tmp_path / "renamed-codex-project"
    old_cwd.mkdir()
    live_cwd.mkdir()
    root = tmp_path / "codex-sessions"
    store = _codex_rollout(root, old_cwd, "rollout-moved")
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(live_cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: live_cwd,
        store_roots={CODEX_SOURCE: root},
        pin=encode_pin(CODEX_SOURCE, store.stem),
        now=NOW,
    )
    assert joined is not None
    assert joined.session_key == store.stem
    assert joined.store_path == store.resolve()


def test_pin_validates_source_and_supports_a_renamed_claude_cwd(tmp_path):
    old_cwd = tmp_path / "old-name"
    new_cwd = tmp_path / "renamed"
    old_cwd.mkdir()
    new_cwd.mkdir()
    root = tmp_path / "claude-projects"
    project = root / project_key_for_cwd(old_cwd)
    project.mkdir(parents=True)
    store = project / "session-1.jsonl"
    store.write_text("")
    os.utime(store, (NOW.timestamp(), NOW.timestamp()))
    variables = PaneVariables("pane-1", JOB_PID, "node", str(new_cwd))

    refused = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CODEX_SOURCE, "rollout-missing"),
        now=NOW,
    )
    assert refused is None

    missing = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CLAUDE_SOURCE, f"{project.name}/missing"),
        now=NOW,
    )
    assert missing is None

    joined = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CLAUDE_SOURCE, f"{project.name}/session-1"),
        now=NOW,
    )
    assert joined is not None
    assert joined.store_path == store.resolve()
    assert joined.session_key == f"{project.name}/session-1"


# --- the join: no agent process ----------------------------------------------


def test_a_pane_whose_job_pid_matches_no_agent_process_returns_no_join(project):
    """Done-when: `jobPid` matching no agent process yields no join.

    Three separate ways that happens, each asserted, because they fail at
    different checks and a fix for one does not fix the others: the pid is
    absent from the table entirely, the pane is a plain shell with no agent
    anywhere above the job, and the job is an `ssh` client whose agent is on
    the far side of the hop.
    """
    cwd, sessions_root = project

    # The pid is gone from the table — it exited between iTerm2 publishing
    # the variable and this read.
    assert _join(cwd, sessions_root, job_pid=999_999) is None

    # A plain shell pane: `less` under the login shell, no agent above it.
    shell_only = _table(
        (
            (500, 400, "less README.md"),
            (400, 300, "-zsh"),
            (300, 1, "/usr/bin/login -fpl dave"),
        )
    )
    assert _join(cwd, sessions_root, table=shell_only, job_pid=500, job_name="less") is None

    # An ssh hop: the agent runs on the remote host, so nothing local is one.
    over_ssh = _table(((600, 400, "ssh build-box"), (400, 1, "-zsh")))
    assert _join(cwd, sessions_root, table=over_ssh, job_pid=600, job_name="ssh") is None

    # Positive control: the same shell tree with `claude` between the job and
    # the shell does join, so the refusals above are about the agent being
    # absent rather than about the tree's shape.
    with_agent = _table(
        (
            (500, 450, "less README.md"),
            (450, 400, "claude"),
            (400, 300, "-zsh"),
            (300, 1, "/usr/bin/login -fpl dave"),
        )
    )
    joined = _join(cwd, sessions_root, table=with_agent, job_pid=500, job_name="less")
    assert joined is not None
    assert joined.pid == 450


def test_the_walk_stops_at_the_login_shell(project):
    """An agent *above* the pane's shell is a coincidence, not this pane's agent.

    Without the shell stop the walk would climb out of the pane entirely and
    join to whatever it found — which on this machine means a pane spawned
    from an agent's own shell would report that agent's project.
    """
    cwd, sessions_root = project
    above_the_shell = _table(
        (
            (500, 400, "vim"),
            (400, 350, "-zsh"),
            (350, 1, "claude"),
        )
    )

    assert _join(cwd, sessions_root, table=above_the_shell, job_pid=500, job_name="vim") is None

    # Positive control: move `claude` one hop down, below the shell, and the
    # identical walk joins.
    below_the_shell = _table(
        (
            (500, 450, "vim"),
            (450, 400, "claude"),
            (400, 1, "-zsh"),
        )
    )
    joined = _join(cwd, sessions_root, table=below_the_shell, job_pid=500, job_name="vim")
    assert joined is not None
    assert joined.pid == 450


def test_the_walk_terminates_on_a_cyclic_process_table():
    """A `ppid` cycle must end the walk rather than hang the status tick.

    `ps` should never produce one, but the walk reads a snapshot it does not
    control, and an unbounded loop here stalls every pane on screen.
    """
    cycle = _table(((10, 20, "node a"), (20, 10, "node b")))

    assert agent_ancestor(10, cycle) is None

    # A chain longer than the hop limit also terminates, and one just inside
    # it still finds its agent — so the bound is a bound, not a coincidence.
    long_chain = _table(
        [(i, i + 1, "node filler") for i in range(1, MAX_ANCESTRY_HOPS + 5)]
        + [(MAX_ANCESTRY_HOPS + 5, 1, "claude")]
    )
    assert agent_ancestor(1, long_chain) is None

    short_chain = _table(
        [(i, i + 1, "node filler") for i in range(1, MAX_ANCESTRY_HOPS - 1)]
        + [(MAX_ANCESTRY_HOPS - 1, 1, "claude")]
    )
    assert agent_ancestor(1, short_chain) is not None


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


def test_a_project_with_no_store_directory_returns_no_join(project, tmp_path):
    """An agent working somewhere Palaver has no store for is not joinable."""
    cwd, sessions_root = project
    empty_root = tmp_path / "empty-store"
    empty_root.mkdir()

    assert _join(cwd, empty_root) is None
    assert _join(cwd, sessions_root) is not None


# --- the join: session candidates --------------------------------------------


def test_one_recent_session_resolves_and_two_refuse_to_pick(project):
    """Session identity is reported only when the evidence names exactly one.

    Measured on this machine, neither source offers a per-pane transcript
    discriminator: Claude Code holds no transcript open, and codex holds ten
    rollouts open at once. So two panes on one project resolve the project
    and stop, rather than both claiming the same session.
    """
    cwd, sessions_root = project
    project_dir = sessions_root / project_key_for_cwd(cwd)
    first = project_dir / "aaaa-1111.jsonl"
    first.write_text("", encoding="utf-8")
    os.utime(first, (NOW.timestamp() - 60, NOW.timestamp() - 60))

    join = _join(cwd, sessions_root)
    assert join is not None
    assert join.session_candidates == ("aaaa-1111",)
    assert join.session_key == "aaaa-1111"

    second = project_dir / "bbbb-2222.jsonl"
    second.write_text("", encoding="utf-8")
    os.utime(second, (NOW.timestamp() - 30, NOW.timestamp() - 30))

    ambiguous = _join(cwd, sessions_root)
    assert ambiguous is not None
    assert ambiguous.session_candidates == ("aaaa-1111", "bbbb-2222")
    assert ambiguous.session_key is None, "picked one of two indistinguishable sessions"


def test_a_session_older_than_the_activity_window_is_not_a_candidate(project):
    """A project directory accumulates every session ever run there.

    Without the window the candidate list is the project's whole history, so
    `session_key` would be `None` forever and the field would be dead.
    """
    cwd, sessions_root = project
    project_dir = sessions_root / project_key_for_cwd(cwd)
    stale = project_dir / "old-session.jsonl"
    stale.write_text("", encoding="utf-8")
    old = (NOW - timedelta(days=3)).timestamp()
    os.utime(stale, (old, old))

    assert session_candidates(project_key_for_cwd(cwd), sessions_root, now=NOW) == ()

    # Positive control: the same file inside the window is a candidate, so
    # the exclusion is the window and not the scan failing to see the file.
    fresh = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(stale, (fresh, fresh))
    assert session_candidates(project_key_for_cwd(cwd), sessions_root, now=NOW) == ("old-session",)


def test_an_aware_clock_and_a_naive_local_one_pick_the_same_candidates(project):
    """The window is compared against mtimes, which are absolute epoch seconds.

    `datetime.timestamp()` reads an aware value exactly and a naive one as
    *local* time, so the same wall-clock instant expressed both ways must land
    on the same cutoff. It does today only because `NOW` is naive-local; the
    rest of the tree (`collect_status`) is aware-UTC, and 5.3 wires a live app
    to this. A caller switching to the aware clock the rest of the tree uses
    must not silently shift the window by the UTC offset.
    """
    cwd, sessions_root = project
    key = project_key_for_cwd(cwd)
    store = sessions_root / key / "aware-session.jsonl"
    store.write_text("", encoding="utf-8")
    inside = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(store, (inside, inside))

    naive_local = NOW
    aware = datetime.fromtimestamp(NOW.timestamp(), tz=timezone.utc)
    assert aware.tzinfo is not None and naive_local.tzinfo is None
    assert aware.timestamp() == naive_local.timestamp()

    assert session_candidates(key, sessions_root, now=aware) == ("aware-session",)
    assert session_candidates(key, sessions_root, now=aware) == session_candidates(
        key, sessions_root, now=naive_local
    )

    # Positive control: the agreement is not both clocks being uselessly
    # permissive. Just outside the window, both must also agree on nothing.
    outside = (NOW - timedelta(days=3)).timestamp()
    os.utime(store, (outside, outside))
    assert session_candidates(key, sessions_root, now=aware) == ()
    assert session_candidates(key, sessions_root, now=naive_local) == ()


def test_the_default_clock_is_an_absolute_instant_not_a_naive_utc_reading(project):
    """`join_pane`'s `now` default has to agree with `st_mtime`'s epoch.

    A default of `datetime.now(timezone.utc)` is correct and so is a naive
    `datetime.now()`; a naive *UTC* clock is not, because `.timestamp()` would
    then re-interpret it as local and move the cutoff by the UTC offset. West
    of UTC that pushes the cutoff into the future and a store written this
    second stops being a candidate, which is what this asserts against.
    """
    cwd, sessions_root = project
    store = sessions_root / project_key_for_cwd(cwd) / "right-now.jsonl"
    store.write_text("", encoding="utf-8")
    just_now = time.time() - 1.0
    os.utime(store, (just_now, just_now))

    join = _join(cwd=cwd, sessions_root=sessions_root, now=None)

    assert join is not None
    assert join.session_candidates == ("right-now",)
    assert join.session_key == "right-now"


# --- the pieces --------------------------------------------------------------


def test_process_name_takes_the_basename_and_strips_a_login_dash():
    """`ps` reports paths, and reports a login shell with a leading dash."""
    assert process_name("claude") == "claude"
    assert process_name("codex resume 019ff309") == "codex"
    assert process_name("/Users/dave/x/node_modules/.bin/opencode serve --pure") == "opencode"
    assert process_name("node /Users/dave/.npm/_npx/abc/.bin/playwright-mcp") == "node"
    assert process_name("-zsh") == "zsh"
    assert process_name("") == ""

    # A path containing a space yields a wrong basename, which matches no
    # agent and so fails closed. Asserted so the behaviour is a decision
    # rather than a surprise.
    assert process_name("/opt/my apps/claude") not in AGENT_SOURCES


def test_project_key_encodes_slashes_and_dots(tmp_path):
    """Verified against real store directories on this machine.

    The doubled dash in the dotfile case is the part a hand-written encoder
    gets wrong: `.` and `/` both map to `-`, so `/Users/dave/.launchd` has
    two adjacent separators.
    """
    assert project_key_for_cwd(Path("/Users/dave/Documents/Projects/palaver")) == (
        "-Users-dave-Documents-Projects-palaver"
    )
    assert project_key_for_cwd(Path("/Users/dave/.launchd")) == "-Users-dave--launchd"
    assert project_key_for_cwd(Path("/Users/dave/Projects/okx_case")) == (
        "-Users-dave-Projects-okx_case"
    )


def test_parse_process_table_keeps_commands_with_spaces_and_skips_junk():
    """`ps` output is not a stable format; one odd row must not cost the table."""
    table = parse_process_table(
        "  100   1 /usr/bin/login -fpl dave\n"
        "  200 100 -zsh\n"
        "garbage line\n"
        "  abc 100 not-a-pid\n"
        "  300 200 claude\n"
    )

    assert set(table) == {100, 200, 300}
    assert table[100].command == "/usr/bin/login -fpl dave"
    assert table[300].name == "claude"
    assert table[200].ppid == 100


def test_process_is_alive_answers_for_this_process_and_a_reaped_one():
    """Signal 0 only, so the check can never disturb an observed agent (INV-2)."""
    assert process_is_alive(os.getpid()) is True

    reaped = subprocess.Popen([sys.executable, "-c", "pass"])
    reaped.wait()
    assert process_is_alive(reaped.pid) is False

    assert process_is_alive(0) is False
    assert process_is_alive(-1) is False


# --- liveness: the two rules -------------------------------------------------


def _signals(
    *,
    source_readable=Tri.TRUE,
    signal_records_parsed=Tri.TRUE,
    unresolved_tool_error=Tri.FALSE,
    agent_turn_ended=Tri.FALSE,
):
    """Build a signal set from a clean, mid-turn (`WORKING`) baseline."""
    return Signals(
        source_readable=source_readable,
        signal_records_parsed=signal_records_parsed,
        unresolved_tool_error=unresolved_tool_error,
        agent_turn_ended=agent_turn_ended,
    )


DEAD = Liveness(process_alive=Tri.FALSE, cursor_advanced_recently=Tri.UNKNOWN)
ALIVE_QUIET = Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=Tri.FALSE)
ALIVE_BUSY = Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=Tri.TRUE)
NO_PANE = Liveness(process_alive=Tri.UNKNOWN, cursor_advanced_recently=Tri.UNKNOWN)


def test_a_dead_process_with_a_stale_working_signal_returns_unknown():
    """Done-when: a dead process demotes `WORKING` to `UNKNOWN`.

    Not to `AWAITING_HUMAN`: that would claim the human is owed something,
    and a process killed mid-turn owes them nothing it can name. The
    controls pin both halves of the rule — the same signals with a live
    process stay `WORKING`, and the same dead process leaves every other
    status alone.
    """
    working = _signals(agent_turn_ended=Tri.FALSE)
    assert derive_status(working) is Status.WORKING

    assert derive_status_with_liveness(working, DEAD) is Status.UNKNOWN

    # Control 1: liveness is what changed the answer, not the signals.
    assert derive_status_with_liveness(working, ALIVE_BUSY) is Status.WORKING
    assert derive_status_with_liveness(working, NO_PANE) is Status.WORKING

    # Control 2: the same dead process does not rewrite anything else. A
    # dead process corroborates an ended-turn status rather than
    # contradicting it, and `ERROR` still names why to look at the session.
    for status in REFINED_STATUS_RANGE - {Status.WORKING}:
        assert apply_liveness(status, DEAD) is status


def test_a_live_process_with_an_ended_turn_and_no_cursor_advance_returns_idle():
    """Done-when: alive, turn ended, quiet for the window yields `IDLE`.

    Each control removes exactly one of the three conjuncts, so `IDLE`
    cannot be reached by any two of them.
    """
    ended = _signals(agent_turn_ended=Tri.TRUE)
    assert derive_status(ended) is Status.AWAITING_HUMAN

    assert derive_status_with_liveness(ended, ALIVE_QUIET) is Status.IDLE

    # Drop "quiet": a session written to inside the window is not idle.
    assert derive_status_with_liveness(ended, ALIVE_BUSY) is Status.AWAITING_HUMAN

    # Drop "alive": an unobservable process is not evidence of anything, and
    # a dead one is certainly not idle.
    assert (
        derive_status_with_liveness(ended, Liveness(Tri.UNKNOWN, Tri.FALSE))
        is Status.AWAITING_HUMAN
    )
    assert (
        derive_status_with_liveness(ended, Liveness(Tri.FALSE, Tri.FALSE)) is Status.AWAITING_HUMAN
    )

    # Drop "turn ended": a working session sitting quiet stays `WORKING`. A
    # ten-minute gap in writes is normal for an agent thinking or running a
    # long tool call, and calling that idle is the loudest way this rule can
    # be wrong.
    assert derive_status_with_liveness(_signals(agent_turn_ended=Tri.FALSE), ALIVE_QUIET) is (
        Status.WORKING
    )


def test_liveness_never_refines_a_status_that_already_says_more():
    """`BLOCKED`, `QUESTION`, `WAITING_FOR_USER`, and `DONE` outrank `IDLE`.

    Rewriting a blocked session to `IDLE` after ten quiet minutes would hide
    the blocker behind the one status meaning "nothing to do here" — and a
    session that has been blocked for ten minutes is more worth surfacing,
    not less.
    """
    ended = _signals(agent_turn_ended=Tri.TRUE)

    for extraction, expected in (
        (Extraction(blockers_now="waiting on a review"), Status.BLOCKED),
        (Extraction(open_questions="which database?"), Status.QUESTION),
        (Extraction(remaining_work="write the tests"), Status.WAITING_FOR_USER),
        (Extraction(remaining_work=""), Status.DONE),
    ):
        assert derive_status(ended, extraction=extraction) is expected
        assert derive_status_with_liveness(ended, ALIVE_QUIET, extraction=extraction) is expected


def test_liveness_cannot_resurrect_a_status_the_rules_withdrew():
    """An alive, quiet process over an unreadable source stays `UNKNOWN`.

    This is the case both rules fall through, and the one a mutant can flip
    invisibly. It is also what keeps `apply_liveness` compatible with
    `derive_status_for_source`'s coverage gate, which may only ever weaken an
    answer: a layer running after the gate that could manufacture a
    confident status would reverse that guarantee.
    """
    unreadable = _signals(source_readable=Tri.FALSE)
    assert derive_status(unreadable) is Status.UNKNOWN

    for liveness in (DEAD, ALIVE_QUIET, ALIVE_BUSY, NO_PANE):
        assert derive_status_with_liveness(unreadable, liveness) is Status.UNKNOWN

    unparsed = _signals(signal_records_parsed=Tri.UNKNOWN)
    assert derive_status(unparsed) is Status.UNKNOWN
    assert derive_status_with_liveness(unparsed, ALIVE_QUIET) is Status.UNKNOWN

    # And rule 6's `UNKNOWN`: the source read cleanly but the turn boundary
    # was not determinable. Liveness says a process exists; it never says the
    # observation succeeded.
    indeterminate = _signals(agent_turn_ended=Tri.UNKNOWN)
    assert derive_status(indeterminate) is Status.UNKNOWN
    assert derive_status_with_liveness(indeterminate, ALIVE_QUIET) is Status.UNKNOWN


def test_liveness_rejects_a_raw_bool():
    """A `bool` where a `Tri` belongs would skip every rule silently.

    `True is Tri.TRUE` is `False`, so the status would pass through unchanged
    and look deliberate. Same guard, same reason, as `Signals.__post_init__`.
    """
    with pytest.raises(TypeError, match="process_alive must be a Tri"):
        Liveness(process_alive=True, cursor_advanced_recently=Tri.FALSE)

    with pytest.raises(TypeError, match="cursor_advanced_recently must be a Tri"):
        Liveness(process_alive=Tri.TRUE, cursor_advanced_recently=False)


# --- liveness: the range over the whole input space --------------------------


def _all_signal_combinations():
    """Every `Tri` value of every signal."""
    return [
        Signals(**dict(zip(SIGNAL_NAMES, combination, strict=True)))
        for combination in itertools.product(Tri, repeat=len(SIGNAL_NAMES))
    ]


def _all_extractions():
    """No extraction, plus every value of every refinement field."""
    return [None] + [
        Extraction(remaining_work=remaining, blockers_now=blockers, open_questions=questions)
        for remaining, blockers, questions in itertools.product((None, "", "some text"), repeat=3)
    ]


def _all_liveness():
    """Every `Tri` value of both liveness fields."""
    return [Liveness(alive, advanced) for alive, advanced in itertools.product(Tri, repeat=2)]


def test_the_live_status_range_over_the_signal_extraction_and_liveness_space():
    """With liveness in play the range is exactly `LIVE_STATUS_RANGE`.

    Established by crossing the whole signal space with the whole refinement
    space with the whole liveness space, rather than by reading the source.
    Every count is asserted before the comparison so a helper that enumerated
    nothing fails here instead of making the range check vacuous, and set
    equality is asserted in both directions because `issubset` would pass for
    an implementation that had lost a branch.
    """
    combinations = _all_signal_combinations()
    extractions = _all_extractions()
    livenesses = _all_liveness()

    assert len(combinations) == 81
    assert len(extractions) == 28
    assert len(livenesses) == 9

    returned = {
        derive_status_with_liveness(signals, liveness, extraction=extraction)
        for signals in combinations
        for extraction in extractions
        for liveness in livenesses
    }

    assert returned == set(LIVE_STATUS_RANGE)
    assert set(LIVE_STATUS_RANGE) == returned
    assert set(REFINED_STATUS_RANGE) < set(LIVE_STATUS_RANGE)

    # Every member of the enum is now reachable somewhere. Stated as a fact
    # the suite proves rather than as prose in a docstring — and it is the
    # assertion that would fail first if a later phase added a status without
    # a way to reach it.
    assert set(Status) - returned == frozenset()


def test_idle_is_reachable_only_from_an_alive_quiet_ended_turn():
    """`IDLE`'s reachability asserted as an exact count, not as membership.

    A rule that fired on `BLOCKED` too, or on any liveness where the process
    is merely not-dead, still passes a membership check over this space. The
    count does not: it is pinned to the exact conjunction, computed
    independently from `derive_status` rather than restated from
    `apply_liveness`.
    """
    cases = [
        (signals, extraction, liveness)
        for signals in _all_signal_combinations()
        for extraction in _all_extractions()
        for liveness in _all_liveness()
    ]
    assert len(cases) == 81 * 28 * 9 == 20_412

    idle = [
        (signals, extraction, liveness)
        for signals, extraction, liveness in cases
        if derive_status_with_liveness(signals, liveness, extraction=extraction) is Status.IDLE
    ]
    expected = [
        (signals, extraction, liveness)
        for signals, extraction, liveness in cases
        if liveness.process_alive is Tri.TRUE
        and liveness.cursor_advanced_recently is Tri.FALSE
        and derive_status(signals, extraction=extraction) is Status.AWAITING_HUMAN
    ]

    assert idle == expected
    assert len(idle) > 0, "IDLE is unreachable, so this test proves nothing"


def test_derive_status_still_cannot_return_idle_at_all():
    """The layering, asserted from the other side.

    `apply_liveness` is a separate step precisely so liveness is not a
    signal, and this is what would fail if someone folded it into the rule
    list to save a call — every existing caller in the tree passes no
    liveness and must keep its Phase 1 and Phase 3.6 ranges exactly.
    """
    returned = {
        derive_status(signals, extraction=extraction)
        for signals in _all_signal_combinations()
        for extraction in _all_extractions()
    }

    assert Status.IDLE not in returned
    assert returned == set(REFINED_STATUS_RANGE)


# --- liveness: building it from a pane and a clock ---------------------------


def test_observe_liveness_distinguishes_never_seen_from_quiet():
    """A session Palaver has not yet watched advance is not a quiet session.

    Collapsing `None` into "no advance" would report `IDLE` for every
    session on the first tick after a daemon restart — including ones
    actively working.
    """
    never = observe_liveness(os.getpid(), last_advance=None, now=NOW)
    assert never.cursor_advanced_recently is Tri.UNKNOWN
    assert never.process_alive is Tri.TRUE

    quiet = observe_liveness(
        os.getpid(), last_advance=NOW - DEFAULT_IDLE_WINDOW - timedelta(seconds=1), now=NOW
    )
    assert quiet.cursor_advanced_recently is Tri.FALSE

    busy = observe_liveness(os.getpid(), last_advance=NOW - timedelta(seconds=5), now=NOW)
    assert busy.cursor_advanced_recently is Tri.TRUE

    # The boundary itself counts as recent: exactly at the window edge is not
    # yet quiet.
    edge = observe_liveness(os.getpid(), last_advance=NOW - DEFAULT_IDLE_WINDOW, now=NOW)
    assert edge.cursor_advanced_recently is Tri.TRUE


def test_observe_liveness_distinguishes_no_pane_from_a_dead_process():
    """`pid=None` is `UNKNOWN`, never `FALSE`.

    `palaver observe` sees every session on the machine and only some of
    them have a pane. Reading "no join" as "dead" would demote every
    headless `WORKING` session to `UNKNOWN` — the whole status command,
    wrong, from one collapsed value.
    """
    no_pane = observe_liveness(None, last_advance=NOW, now=NOW)
    assert no_pane.process_alive is Tri.UNKNOWN

    dead = observe_liveness(999_999, last_advance=NOW, now=NOW, alive_probe=lambda pid: False)
    assert dead.process_alive is Tri.FALSE

    assert apply_liveness(Status.WORKING, no_pane) is Status.WORKING
    assert apply_liveness(Status.WORKING, dead) is Status.UNKNOWN


# --- live: against this machine's real process table -------------------------

live = pytest.mark.skipif(
    os.environ.get("PALAVER_SKIP_LIVE") == "1",
    reason="live process-table tests disabled by PALAVER_SKIP_LIVE",
)


@live
def test_the_real_process_table_contains_this_process_and_its_parent():
    """The `ps` invocation and its parsing, against the real thing.

    A format change in `ps` would leave every unit test above passing while
    the join silently stopped working on the machine it runs on.
    """
    table = read_process_table()

    assert len(table) > 10
    assert os.getpid() in table
    assert table[os.getpid()].ppid == os.getppid()
    # Case-folded: macOS runs the framework build through a `Python.app`
    # bundle, so argv[0]'s basename is `Python`, not `python3`.
    assert table[os.getpid()].name.lower().startswith("python")


@live
def test_the_real_working_directory_of_this_process_is_readable():
    """`lsof` answers for a live process and declines for a reaped one.

    The declining half matters more: `None` must mean "could not determine"
    and never leak through as a directory that happens to match.
    """
    assert working_directory(os.getpid()) == Path.cwd().resolve()

    reaped = subprocess.Popen([sys.executable, "-c", "pass"])
    reaped.wait()
    assert working_directory(reaped.pid) is None


@live
def test_a_real_agent_is_found_from_its_own_descendants_in_the_live_table():
    """The finding that shaped this module, asserted against the live machine.

    A mocked process tree proves the walk climbs; only the real table proves
    it climbs *the shape that actually occurs*. The descendant half is the
    part worth running live: on this machine the pane's foreground job is an
    MCP server two hops below the agent, and every version of this module
    that read `jobPid` directly passed its unit tests.

    Skips rather than fails when no agent is running, so the suite stays
    green on a machine that happens to be idle.
    """
    table = read_process_table()
    agents = [info for info in table.values() if info.name in AGENT_SOURCES]
    if not agents:
        pytest.skip("no known agent is running on this machine")

    agent = agents[0]
    assert process_is_alive(agent.pid)

    # Zero hops: a pane whose foreground job *is* the agent, which is the
    # measured codex shape.
    assert agent_ancestor(agent.pid, table) == agent

    # Now from a real descendant. Walking from any process below an agent,
    # with no shell in between, must reach that same agent.
    children = {info.pid: info for info in table.values()}
    for info in table.values():
        chain, cursor = [], info
        while cursor is not None and cursor.pid != agent.pid and len(chain) < MAX_ANCESTRY_HOPS:
            chain.append(cursor)
            cursor = children.get(cursor.ppid)
        reached_agent = cursor is not None and cursor.pid == agent.pid
        crosses_shell = any(link.name in SHELL_NAMES for link in chain)
        if reached_agent and chain and not crosses_shell:
            assert agent_ancestor(info.pid, table) == agent, (
                f"walking up from {info.name} (pid {info.pid}) missed its agent"
            )
            assert info.pid != agent.pid
            return

    pytest.skip(f"{agent.name} has no non-shell descendant to walk up from")
