"""
Per-agent state-file naming, the companion's launch command, and the initial-state seed
write before the renderer starts reading.
"""

from __future__ import annotations

import hashlib
import shlex
import sys
import time
from pathlib import Path

from palaver.ui.companion_state import (
    CompanionState,
    JoinState,
    atomic_write_state,
)
from palaver.ui.pane_join import (
    PaneJoin,
    SupportedPaneProcess,
)


def opaque_state_path(state_dir: Path, agent_id: str) -> Path:
    """Return a non-identifying state filename for one iTerm agent pane."""
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
    return state_dir / f"{digest}.json"


def companion_command(state_path: Path, *, executable: str | None = None) -> str:
    """Build the custom command without shell interpolation hazards."""
    python = sys.executable if executable is None else executable
    return shlex.join([python, "-m", "palaver.ui.companion_render", "--state", str(state_path)])


def write_initial_state(
    path: Path,
    detected: SupportedPaneProcess,
    joined: PaneJoin | None,
) -> None:
    """Seed the renderer transport before its process can start reading."""
    exact = joined is not None and joined.session_key is not None
    atomic_write_state(
        path,
        CompanionState(
            producer_updated_at=time.time(),
            project=detected.cwd.name,
            source=detected.source,
            status="UNKNOWN",
            join_state=JoinState.JOINED if exact else JoinState.UNJOINED,
            detail=None if exact else "Agent detected; waiting for an exact session join",
        ),
    )
