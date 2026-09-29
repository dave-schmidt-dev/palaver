"""
Process-table mechanics: parsing ps output, the project-key encoding, the liveness
probe, and the ancestry walk with its login-shell/task-shell trade-off.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from palaver.ui.pane_join import (
    AGENT_SOURCES,
    CODEX_SOURCE,
    MAX_ANCESTRY_HOPS,
    agent_ancestor,
    parse_process_table,
    process_is_alive,
    process_name,
    project_key_for_cwd,
)
from tests._pane_join_support import _join, _table
from tests._pane_join_support import project as project

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


def test_the_walk_climbs_through_a_task_shell_the_agent_itself_spawned(project):
    """A background command's own shell is not the pane's login shell.

    Both supported agents run shell commands through `bash script.sh` or
    `zsh -c "..."` -- exactly the shape that left a live quizzler pane
    UNJOINED (2026-08-19): a backgrounded `./test-gate.sh` invocation put the
    pane's foreground job several hops below `codex`, crossing one task shell
    with no login-shell dash. Before this fix `agent_ancestor` stopped at that
    shell by name alone, the same way it correctly stops at a real login
    shell, and never found `codex` above it.
    """
    cwd, sessions_root = project
    table = _table(
        (
            (700, 600, "python3 scripts/hybrid_verify.py"),
            (600, 500, "/bin/bash ./app/test-gate.sh --phase native"),
            (500, 400, "codex"),
            (400, 1, "-zsh"),
        )
    )

    joined = _join(cwd, sessions_root, table=table, job_pid=700, job_name="python3")
    assert joined is not None
    assert joined.pid == 500
    assert joined.source == CODEX_SOURCE

    # Positive control: starting the walk at the task shell itself, one hop
    # closer to the agent, still finds it.
    joined_from_shell = _join(cwd, sessions_root, table=table, job_pid=600, job_name="bash")
    assert joined_from_shell is not None
    assert joined_from_shell.pid == 500


def test_a_hand_started_nested_shell_can_still_climb_to_a_coincidental_agent(project):
    """The trade-off `agent_ancestor` accepts in exchange for the fix above.

    Not every non-login shell is a task shell an agent spawned -- a shell the
    *user* starts by hand inside a pane (typing `bash` at their prompt) is
    also not dash-prefixed. If it happens to sit above an unrelated agent in
    the same process tree, the walk now climbs through it and reports that
    agent. Narrower than the false negative this fix closes (agents shell out
    constantly; users rarely nest an interactive shell whose ancestry passes
    through a foreign agent), and pinned down here so the trade-off is a
    decision on record, not a surprise found later.
    """
    cwd, sessions_root = project
    table = _table(
        (
            (500, 400, "vim"),
            (400, 350, "zsh"),  # hand-started inside the pane, not the login shell
            (350, 1, "claude"),
        )
    )

    joined = _join(cwd, sessions_root, table=table, job_pid=500, job_name="vim")
    assert joined is not None
    assert joined.pid == 350


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
