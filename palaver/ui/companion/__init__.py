"""Lifecycle ownership for Palaver's per-agent iTerm2 companion panes.

The controller owns only panes carrying Palaver's exact marker.  Agent panes
are observed and may be split once; they are never sent input or closed.  The
only layout write Palaver makes is that split and the row division between the
observed pane and its own companion, and it is written so both of the tab's
totals are unchanged: iTerm2's one pane-sizing call rewrites the entire tab, so
anything less careful resizes the user's window.  All mutating operations are
serialized so the new-session and layout monitors cannot turn one split into a
recursive chain of companions.
"""

from __future__ import annotations

import asyncio  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import shlex  # noqa: F401
import sys  # noqa: F401
import time  # noqa: F401
from collections.abc import Awaitable, Callable, Mapping  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from palaver.ui.companion_state import (
    CompanionState,  # noqa: F401
    JoinState,  # noqa: F401
    atomic_write_state,  # noqa: F401
)
from palaver.ui.connection import import_iterm2  # noqa: F401
from palaver.ui.pane_join import (
    PIN_VARIABLE,  # noqa: F401
    PaneJoin,  # noqa: F401
    PaneVariables,  # noqa: F401
    ProcessTable,  # noqa: F401
    SupportedPaneProcess,  # noqa: F401
    detect_supported_process,  # noqa: F401
    join_pane,  # noqa: F401
    read_process_table,  # noqa: F401
)

from .core import CompanionController as CompanionController
from .creation import METADATA_VARIABLES as METADATA_VARIABLES
from .creation import _decode_variable as _decode_variable
from .creation import build_companion_profile as build_companion_profile
from .creation import make_metadata_reader as make_metadata_reader
from .paths import companion_command as companion_command
from .paths import opaque_state_path as opaque_state_path
from .paths import write_initial_state as write_initial_state
from .records import AGENT_EXIT_GRACE as AGENT_EXIT_GRACE
from .records import AGENT_SESSION_VARIABLE as AGENT_SESSION_VARIABLE
from .records import COMPANION_ROLE as COMPANION_ROLE
from .records import COMPANION_SESSION_VARIABLE as COMPANION_SESSION_VARIABLE
from .records import DISABLED_VARIABLE as DISABLED_VARIABLE
from .records import INITIAL_RESTART_BACKOFF as INITIAL_RESTART_BACKOFF
from .records import LAYOUT_SETTLE_ATTEMPTS as LAYOUT_SETTLE_ATTEMPTS
from .records import LAYOUT_SETTLE_DELAY as LAYOUT_SETTLE_DELAY
from .records import MAX_RESTART_BACKOFF as MAX_RESTART_BACKOFF
from .records import MIN_AGENT_ROWS as MIN_AGENT_ROWS
from .records import ROLE_VARIABLE as ROLE_VARIABLE
from .records import SCROLLBACK_LINES as SCROLLBACK_LINES
from .records import SUMMARY_ROWS as SUMMARY_ROWS
from .records import CompanionPair as CompanionPair
from .records import ReadMetadata as ReadMetadata
from .records import ReconcileResult as ReconcileResult
from .records import SessionMetadata as SessionMetadata
from .records import WriteInitialState as WriteInitialState
from .records import _no_status as _no_status

__all__ = [
    "AGENT_EXIT_GRACE",
    "AGENT_SESSION_VARIABLE",
    "COMPANION_ROLE",
    "COMPANION_SESSION_VARIABLE",
    "DISABLED_VARIABLE",
    "INITIAL_RESTART_BACKOFF",
    "LAYOUT_SETTLE_ATTEMPTS",
    "LAYOUT_SETTLE_DELAY",
    "MAX_RESTART_BACKOFF",
    "METADATA_VARIABLES",
    "MIN_AGENT_ROWS",
    "ROLE_VARIABLE",
    "SCROLLBACK_LINES",
    "SUMMARY_ROWS",
    "CompanionController",
    "CompanionPair",
    "ReadMetadata",
    "ReconcileResult",
    "SessionMetadata",
    "WriteInitialState",
    "__all__",
    "_decode_variable",
    "_no_status",
    "build_companion_profile",
    "companion_command",
    "make_metadata_reader",
    "opaque_state_path",
    "write_initial_state",
]


__all__ = [
    "AGENT_SESSION_VARIABLE",
    "COMPANION_ROLE",
    "COMPANION_SESSION_VARIABLE",
    "DISABLED_VARIABLE",
    "ROLE_VARIABLE",
    "CompanionController",
    "CompanionPair",
    "ReconcileResult",
    "SessionMetadata",
    "build_companion_profile",
    "companion_command",
    "opaque_state_path",
]
