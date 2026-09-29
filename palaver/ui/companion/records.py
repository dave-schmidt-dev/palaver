"""
Role/session variable names, timing and layout constants, the three dataclasses, the
callable type aliases, and the original module's public-name list.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palaver.ui.pane_join import (
    PaneJoin,
    PaneVariables,
    SupportedPaneProcess,
)

ROLE_VARIABLE = "user.palaver_role"
AGENT_SESSION_VARIABLE = "user.palaver_agent_session"
COMPANION_SESSION_VARIABLE = "user.palaver_companion_session"
DISABLED_VARIABLE = "user.palaver_companion_disabled"
COMPANION_ROLE = "companion-v1"

SUMMARY_ROWS = 5
MIN_AGENT_ROWS = 1
LAYOUT_SETTLE_ATTEMPTS = 3
LAYOUT_SETTLE_DELAY = 0.05
SCROLLBACK_LINES = 100
INITIAL_RESTART_BACKOFF = 2.0
MAX_RESTART_BACKOFF = 60.0

#: How long a paired pane may show no supported agent process before its
#: companion is torn down. `SessionTerminationMonitor` reports only the end of
#: the pane's own command — `login`, then the shell — so an agent that exits
#: back to a shell prompt raises no event at all and the pair would otherwise
#: survive forever showing an unjoined surface. The grace period is what keeps
#: that teardown from firing on a momentary miss: `ps` can fail and return an
#: empty table, and `claude -c` immediately after `/exit` is a restart, not an
#: ended session.
AGENT_EXIT_GRACE = 12.0


def _no_status(_message: str) -> None:
    """Default progress sink."""


@dataclass(frozen=True)
class SessionMetadata:
    """One iTerm session plus the variables used by lifecycle decisions."""

    session: Any
    tab: Any
    pane: PaneVariables
    role: str | None = None
    agent_session: str | None = None
    companion_session: str | None = None
    disabled: bool = False

    @property
    def session_id(self) -> str:
        """Return iTerm2's stable-within-run session identifier."""
        return self.pane.pane_id

    @property
    def is_companion(self) -> bool:
        """Whether this pane bears Palaver's exact ownership marker."""
        return self.role == COMPANION_ROLE


@dataclass(frozen=True)
class CompanionPair:
    """A reciprocal live agent/companion pair."""

    agent_id: str
    companion_id: str
    state_path: Path


@dataclass(frozen=True)
class ReconcileResult:
    """Observable result of one reconciliation pass."""

    pairs: tuple[CompanionPair, ...]
    created: tuple[str, ...]
    closed: tuple[str, ...]
    refused: tuple[str, ...]


ReadMetadata = Callable[[Any, Any], Awaitable[SessionMetadata | None]]
WriteInitialState = Callable[[Path, SupportedPaneProcess, PaneJoin | None], None]
