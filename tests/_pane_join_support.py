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

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from palaver.ui.pane_join import (
    PaneVariables,
    ProcessInfo,
    join_pane,
    process_name,
    project_key_for_cwd,
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
        pid: ProcessInfo(
            pid=pid,
            ppid=ppid,
            name=process_name(command),
            command=command,
            start_time="Fri Aug 15 12:00:00 2026",
        )
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


def _join(
    cwd, sessions_root, *, table=None, cwd_reader=None, now=NOW, registry_root=None, **kwargs
):
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
        # Hermetic by default: without this every test would consult the real
        # ~/.claude/sessions and pass only because no live pid happens to
        # claim a tmp_path cwd.
        registry_root=(sessions_root / "no-registry") if registry_root is None else registry_root,
    )


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


# --- live: against this machine's real process table -------------------------

live = pytest.mark.skipif(
    os.environ.get("PALAVER_SKIP_LIVE") == "1",
    reason="live process-table tests disabled by PALAVER_SKIP_LIVE",
)
