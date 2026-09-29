"""
Building the Liveness signal from a resolved pid, a last-advance timestamp, and a clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from palaver.observer.signals import Liveness, Tri

from .constants import DEFAULT_IDLE_WINDOW
from .process import process_is_alive


def observe_liveness(
    pid: int | None,
    *,
    last_advance: datetime | None,
    now: datetime,
    idle_window: timedelta = DEFAULT_IDLE_WINDOW,
    alive_probe=process_is_alive,
) -> Liveness:
    """Build the `Liveness` the status layer consumes.

    Args:
        pid: The agent's pid from a `PaneJoin`, or `None` when there is no
            join — no pane, a refused join, or a headless observation.
        last_advance: When this session's store was last seen to grow, or
            `None` if Palaver has never recorded an advance for it.
        now: Reference time the idle window is measured back from.
        idle_window: How long a store must go unwritten to count as quiet.
        alive_probe: Callable taking a pid and returning whether it is live.
            Injectable so a test can describe a dead process without having
            to create and reap one.

    Returns:
        A `Liveness`. `pid=None` yields `process_alive=UNKNOWN`, never
        `FALSE`: no join was attempted, so nothing was observed to be gone.
        `last_advance=None` yields `cursor_advanced_recently=UNKNOWN` for the
        same reason — a session Palaver has not yet watched advance is not a
        session that has been quiet.
    """
    if pid is None:
        alive = Tri.UNKNOWN
    else:
        alive = Tri.TRUE if alive_probe(pid) else Tri.FALSE

    if last_advance is None:
        advanced = Tri.UNKNOWN
    else:
        advanced = Tri.TRUE if now - last_advance <= idle_window else Tri.FALSE

    return Liveness(process_alive=alive, cursor_advanced_recently=advanced)
