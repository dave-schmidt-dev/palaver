"""Progress reporting shared by the CLI commands and the AutoLaunch watcher."""

from __future__ import annotations

import sys


def stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)
