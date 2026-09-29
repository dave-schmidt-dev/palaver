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

**3. The file-descriptor join does not work alone.** Codex holds its rollout
files open, so `lsof` on a codex pane looks like an exact pane-to-transcript
join — until you count them and find **ten** rollouts open at once (session
history, not just the live thread). Claude Code holds none at all. Neither
source offers a discriminator that stands on its own, which is why `PaneJoin`
resolves the *project* and reports `session_candidates` rather than picking
one from an unambiguous scan. `lsof` still earns its keep as a narrowing
filter once the cwd+mtime scan is already ambiguous (measured 2026-08-19: a
project with three recent rollouts narrowed to the one live thread because
it, and only it, was still open by the resolved agent pid) — the ten-rollout
finding says a process's open files cannot be trusted *by themselves*, not
that they carry no information at all. See `_agent_open_store_paths` for how
that narrowing is used, and its residual pid-reuse race.

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

import json  # noqa: F401
import os  # noqa: F401
import subprocess  # noqa: F401
from collections.abc import Callable, Mapping, MutableMapping  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from datetime import datetime, timedelta, timezone  # noqa: F401
from pathlib import Path  # noqa: F401

from palaver.ingest.adapters.codex import CodexAdapter  # noqa: F401
from palaver.observer.signals import Liveness, Tri  # noqa: F401

from .candidates import project_key_for_cwd as project_key_for_cwd
from .candidates import project_keys_for_cwd as project_keys_for_cwd
from .candidates import session_candidates as session_candidates
from .constants import AGENT_SOURCES as AGENT_SOURCES
from .constants import CLAUDE_SOURCE as CLAUDE_SOURCE
from .constants import CODEX_SOURCE as CODEX_SOURCE
from .constants import DEFAULT_ACTIVITY_WINDOW as DEFAULT_ACTIVITY_WINDOW
from .constants import DEFAULT_IDLE_WINDOW as DEFAULT_IDLE_WINDOW
from .constants import MAX_ANCESTRY_HOPS as MAX_ANCESTRY_HOPS
from .constants import PANE_PIN_VARIABLE as PANE_PIN_VARIABLE
from .constants import PIN_VARIABLE as PIN_VARIABLE
from .constants import SHELL_NAMES as SHELL_NAMES
from .constants import default_registry_root as default_registry_root
from .constants import default_store_roots as default_store_roots
from .join import detect_supported_process as detect_supported_process
from .join import join_pane as join_pane
from .liveness import observe_liveness as observe_liveness
from .pin import PanePin as PanePin
from .pin import encode_pin as encode_pin
from .pin import parse_pin as parse_pin
from .process import _is_login_shell as _is_login_shell
from .process import _same_process_identity as _same_process_identity
from .process import agent_ancestor as agent_ancestor
from .process import parse_process_table as parse_process_table
from .process import process_is_alive as process_is_alive
from .process import process_name as process_name
from .process import read_process_table as read_process_table
from .process import working_directory as working_directory
from .records import PaneJoin as PaneJoin
from .records import PaneVariables as PaneVariables
from .records import ProcessInfo as ProcessInfo
from .records import ProcessTable as ProcessTable
from .records import SupportedPaneProcess as SupportedPaneProcess
from .stores import CodexCandidateProgress as CodexCandidateProgress
from .stores import _agent_open_store_paths as _agent_open_store_paths
from .stores import _codex_store_candidates as _codex_store_candidates
from .stores import _narrow_codex_candidates_by_progress as _narrow_codex_candidates_by_progress
from .stores import _pinned_store_path as _pinned_store_path
from .stores import _readable_file as _readable_file
from .stores import _registered_store as _registered_store
from .stores import _root_for_source as _root_for_source
from .stores import registered_session as registered_session

__all__ = [
    "AGENT_SOURCES",
    "CLAUDE_SOURCE",
    "CODEX_SOURCE",
    "DEFAULT_ACTIVITY_WINDOW",
    "DEFAULT_IDLE_WINDOW",
    "MAX_ANCESTRY_HOPS",
    "PANE_PIN_VARIABLE",
    "PIN_VARIABLE",
    "SHELL_NAMES",
    "CodexCandidateProgress",
    "PaneJoin",
    "PanePin",
    "PaneVariables",
    "ProcessInfo",
    "ProcessTable",
    "SupportedPaneProcess",
    "_agent_open_store_paths",
    "_codex_store_candidates",
    "_is_login_shell",
    "_narrow_codex_candidates_by_progress",
    "_pinned_store_path",
    "_readable_file",
    "_registered_store",
    "_root_for_source",
    "_same_process_identity",
    "agent_ancestor",
    "default_registry_root",
    "default_store_roots",
    "detect_supported_process",
    "encode_pin",
    "join_pane",
    "observe_liveness",
    "parse_pin",
    "parse_process_table",
    "process_is_alive",
    "process_name",
    "project_key_for_cwd",
    "project_keys_for_cwd",
    "read_process_table",
    "registered_session",
    "session_candidates",
    "working_directory",
]
