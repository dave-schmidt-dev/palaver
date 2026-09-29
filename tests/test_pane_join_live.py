"""
Against this machine's real process table: ps/lsof parsing and the ancestry walk over
live processes, skipped when none are running.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from palaver.ui.pane_join import (
    AGENT_SOURCES,
    MAX_ANCESTRY_HOPS,
    SHELL_NAMES,
    _agent_open_store_paths,
    agent_ancestor,
    process_is_alive,
    read_process_table,
    working_directory,
)
from tests._pane_join_support import live


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
def test_the_real_open_file_scan_finds_a_file_this_process_has_open(tmp_path):
    """The `lsof -Fn` parsing behind the codex ambiguity narrowing, for real.

    A mocked-only suite would have shipped the narrowing without ever
    confirming `lsof`'s output shape actually parses on this machine -- the
    same risk the module docstring names for `ps`. Opening a file this test
    process controls, in a directory nothing else touches, proves the scan
    finds it without depending on any codex process being alive right now.
    """
    marker = tmp_path / "open-file-marker"
    with marker.open("w") as handle:
        handle.write("held open for the duration of the scan\n")
        handle.flush()
        assert marker.resolve() in _agent_open_store_paths(os.getpid())

    reaped = subprocess.Popen([sys.executable, "-c", "pass"])
    reaped.wait()
    assert _agent_open_store_paths(reaped.pid) == frozenset()


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
