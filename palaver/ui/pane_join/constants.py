"""Shared config constants and the on-disk root helpers they parameterize."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path

#: Executable basenames that identify an agent, mapped to the adapter
#: `source` name that reads its session store. Matched against the agent
#: process's own name, never against `jobName` — see fact 2 in the module
#: docstring.
AGENT_SOURCES: Mapping[str, str] = {
    "claude": "claude-code",
    "codex": "codex",
    "opencode": "opencode",
}

#: Names that can end the ancestry walk. An agent started in a pane is a
#: descendant of that pane's login shell, so anything at or above the shell
#: belongs to iTerm2 rather than to the pane's work, and a match up there
#: would be a coincidence rather than a join. `login` is included because
#: iTerm2 execs the shell through it, and always ends the walk on sight —
#: see `agent_ancestor`. The other names end the walk only when
#: `_is_login_shell` also holds, because an agent commonly shells out
#: through the *same* binaries to run the command whose pane is being
#: joined (`bash test-gate.sh`, `/bin/zsh -c "npm test"`), and a name match
#: alone cannot tell that task shell from the pane's own login shell.
SHELL_NAMES: frozenset[str] = frozenset(
    {"zsh", "bash", "sh", "fish", "tcsh", "csh", "dash", "ksh", "login"}
)

#: Ancestry hops to walk before giving up. Bounded rather than "walk to pid
#: 1" so a cycle in a malformed process table — or a `ppid` that fails to
#: decrease — terminates instead of hanging the status tick. Twelve is far
#: past the deepest real chain observed (four, for the Claude Code pane in
#: the module docstring).
MAX_ANCESTRY_HOPS = 12

#: How long a session store must go unwritten before `apply_liveness` will
#: call a live, turn-ended session `IDLE`. Well clear of the observer's
#: 30–60s tick, so a session cannot be reported idle merely because two
#: ticks happened to fall between two writes.
DEFAULT_IDLE_WINDOW = timedelta(minutes=10)

#: How recently a session store must have been written for the session to
#: count as a candidate for a pane. This narrows an accumulated project
#: directory — which holds every session ever run there — to the ones that
#: could plausibly be the one on screen.
DEFAULT_ACTIVITY_WINDOW = timedelta(hours=1)

CLAUDE_SOURCE = "claude-code"
CODEX_SOURCE = "codex"
PIN_VARIABLE = "user.palaver_session_pin"


def default_registry_root() -> Path:
    """Return Claude Code's live-process registry directory.

    Claude Code writes one `<pid>.json` here per running process, carrying
    that process's own `sessionId` and `cwd`. It is undocumented and tied to
    the CLI version (observed on 2.1.226), so every read of it is treated as
    a hint that must agree with what the pane already proved, and its absence
    is not an error -- `join_pane` falls back to the mtime candidate scan.
    """
    return Path.home() / ".claude" / "sessions"


def default_store_roots() -> dict[str, Path]:
    """Return the independent on-disk roots used by supported file sources."""
    return {
        CLAUDE_SOURCE: Path.home() / ".claude" / "projects",
        CODEX_SOURCE: Path.home() / ".codex" / "sessions",
    }
