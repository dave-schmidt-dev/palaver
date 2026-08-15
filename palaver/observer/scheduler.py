"""What one observer tick decides to spend inference on (Task 4.1).

The daemon's economics live here, in one comparison. Each tick discovers
every session in scope, tails each one from its durable cursor, and
schedules extraction **only for the sessions whose cursor moved**. A session
nobody typed into since the last tick costs one `stat`, one `open`, and a
read of zero new bytes — and, critically, zero inference requests. That is
the difference between an observer a laptop can host and a permanent GPU
tenant: at a 30-second tick over six sessions, gating on "the file changed"
rather than "the file exists" is the difference between ~10 model requests
a day and ~17,000.

**The gate is cursor *movement*, not cursor *growth*.** A shrunken store is
new material too: `read_complete_records` recovers from truncation by
re-reading from the start (task 1.3), which returns a cursor at a *lower*
offset than the one it was given, along with every record in the rewritten
file. Gating on `>` would classify that session as idle and, worse, would
never persist the corrected cursor — leaving it permanently stuck comparing
against an offset past the end of the file it is watching. `!=` covers both
directions and is still exactly zero for a file nobody touched.

**Nothing here writes.** `plan_tick` never saves a cursor: it returns the
cursor each scheduled session *would* advance to, and `palaver.observer.
daemon` persists it only after that session's extraction succeeds. That
ordering is what makes a failed extraction retry on the next tick instead
of being silently skipped forever, and it is why this module hands back
`cursor_after` rather than committing it itself. See the daemon's module
docstring for the at-least-once argument in full.

INV-2: every read reaches the observed store through the adapter's `tail`,
which opens read-only. This module never opens a session file itself.

INV-1: `plan_tick` takes an `on_status` channel and calls it once per
session before that session is opened, because a tick over a large store is
a walk over many files and must never be a silent wait.

This repository is public. Nothing in this module's docstrings or examples
is derived from a real observed session (INV-9).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from palaver.ingest.adapters.base import DEFAULT_SINCE, Adapter, Event, SessionRef
from palaver.ingest.cursors import Cursor, CursorStore


@dataclass(frozen=True)
class SessionWork:
    """One session that changed since the last tick, and what changed.

    Attributes:
        ref: The discovered session this work belongs to.
        events: Every event the tail produced, in file order. May be empty
            even though the cursor moved — a store can grow by bytes that
            carry no adapter-recognized event (bookkeeping records, for
            instance). Such a session is still scheduled: the cursor moved,
            the file's meaning may have changed with it, and the daemon's
            extractor reads the *session*, not this tick's event slice.
        cursor_before: The cursor this tick started from.
        cursor_after: The cursor the tail reached. The daemon persists this
            only after extraction succeeds; see the module docstring.
    """

    ref: SessionRef
    events: tuple[Event, ...]
    cursor_before: Cursor
    cursor_after: Cursor


@dataclass(frozen=True)
class TickPlan:
    """Every session one tick looked at, split by whether it earns inference.

    Attributes:
        scheduled: Sessions whose cursor moved, each with its tailed events.
        skipped: Sessions discovered and tailed but unchanged. Kept rather
            than discarded so a caller — and a test — can tell "zero
            extraction requests because nothing changed" apart from "zero
            extraction requests because discovery found nothing", which are
            the same number and completely different failures.
    """

    scheduled: tuple[SessionWork, ...]
    skipped: tuple[SessionRef, ...]

    @property
    def discovered(self) -> int:
        """How many sessions this tick discovered and tailed, changed or not."""
        return len(self.scheduled) + len(self.skipped)


def plan_tick(
    adapters: Iterable[Adapter],
    cursors: CursorStore,
    *,
    since: timedelta | None = DEFAULT_SINCE,
    all: bool = False,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> TickPlan:
    """Decide which discovered sessions this tick should extract from.

    Args:
        adapters: The sources to sweep, in order. Each is asked for its own
            sessions; a `SessionRef` carries its `source`, so results from
            several adapters coexist in one plan without ambiguity.
        cursors: Durable cursor store (task 1.3). Read here, never written.
        since: Recency floor passed through to `discover_sessions`. Ignored
            when `all` is True.
        all: Skip windowing entirely and consider every session the adapter
            can enumerate.
        now: Reference time for the discovery window. Defaults to the
            current UTC time inside the adapter; tests pass a fixed value so
            discovery does not depend on when the suite runs.
        on_status: Progress channel (INV-1), called once per session before
            it is opened. Never writes to stdout.

    Returns:
        A `TickPlan` naming what changed and what did not.
    """
    scheduled: list[SessionWork] = []
    skipped: list[SessionRef] = []

    for adapter in adapters:
        refs = adapter.discover_sessions(since=since, all=all, now=now)
        total = len(refs)
        for index, ref in enumerate(refs, start=1):
            if on_status is not None:
                on_status(f"{adapter.source} {index}/{total}: tailing {ref.session_key}")
            before = cursors.load(ref.session_key)
            result = adapter.tail(ref.path, before)
            if result.cursor.offset == before.offset:
                skipped.append(ref)
                continue
            scheduled.append(
                SessionWork(
                    ref=ref,
                    events=tuple(result.events),
                    cursor_before=before,
                    cursor_after=result.cursor,
                )
            )

    return TickPlan(scheduled=tuple(scheduled), skipped=tuple(skipped))


__all__ = ["Event", "SessionWork", "TickPlan", "plan_tick"]
