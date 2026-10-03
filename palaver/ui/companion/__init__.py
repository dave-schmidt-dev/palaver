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

from palaver.ui.connection import import_iterm2  # noqa: F401

from .core import CompanionController as CompanionController
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
from .records import ROLE_VARIABLE as ROLE_VARIABLE
from .records import SCROLLBACK_LINES as SCROLLBACK_LINES
from .records import SUMMARY_ROWS as SUMMARY_ROWS
from .records import CompanionPair as CompanionPair
from .records import ReadMetadata as ReadMetadata
from .records import SessionMetadata as SessionMetadata

__all__ = [
    "AGENT_SESSION_VARIABLE",
    "COMPANION_ROLE",
    "COMPANION_SESSION_VARIABLE",
    "DISABLED_VARIABLE",
    "ROLE_VARIABLE",
    "CompanionController",
    "CompanionPair",
    "SessionMetadata",
    "build_companion_profile",
    "companion_command",
    "opaque_state_path",
]
