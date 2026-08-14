"""Abstract adapter interface: session discovery and read-only tailing.

An adapter turns one third-party session store (Claude Code's JSONL,
Codex's JSONL, OpenCode's SQLite database, ...) into two things Palaver's
observer needs: a bounded set of sessions worth looking at
(`discover_sessions`), and, for any one of them, the canonical events
appended since a cursor (`tail`). This module defines that contract plus
the mechanics every JSONL-backed adapter needs and would otherwise
duplicate.

Two invariants shape everything here:

INV-2 (never write to an observed session). Every read this module performs
goes through `open_source_readonly`, which asks the OS for the `O_RDONLY`
access mode specifically — not merely a Python-level "r" string that happens
not to write. That is the one chokepoint through which a source file is ever
opened, so an adapter built on top of it cannot accidentally request write
access from some other code path.

A torn read is not a write, but it is a correctness hazard: these stores are
appended to by a *live* agent process while Palaver reads them, and the
writer can flush a partial JSON record before it finishes writing a whole
line. `read_complete_records` only ever advances an offset to the end of the
last newline-terminated line it saw — a trailing partial line is not
returned and does not move the offset, so it is re-read whole (and this time
complete) on the next call. Advancing past a partial line would either lose
that record or hand a downstream JSON parser a truncated string.

Session discovery windowing (`discover_sessions`). These stores are
historical archives, not a live index of what is open right now, so
`discover_sessions` defaults to a floor: only sessions active within `since`
(default 24h, by store mtime) are returned. That floor does not apply to a
session whose last record is an unresolved `tool_use` — an outstanding tool
call means the agent may still be working regardless of how old the file's
mtime looks from outside — but that always-include rule is itself bounded by
a wider `outer_window` (default 7d), and only sessions inside it are ever
opened to check. A session outside `outer_window` is excluded on its mtime
alone, via `os.stat`, and is never opened at all; that is what keeps the
always-include rule a bounded tail seek instead of a full-archive scan.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO

from palaver.ingest.cursors import Cursor

DEFAULT_SINCE = timedelta(hours=24)
DEFAULT_OUTER_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class Event:
    """One canonical event produced by an adapter's `tail`.

    Attributes:
        session_key: Durable identity of the session this event belongs to.
        kind: Adapter-defined event kind, e.g. "tool_use", "tool_result",
            "message", "compaction".
        payload: The event's adapter-native fields, decoded from the source
            record.
    """

    session_key: str
    kind: str
    payload: dict


@dataclass(frozen=True)
class SessionRef:
    """A session `discover_sessions` found, ready to be opened and tailed.

    Attributes:
        source: The adapter's source name, e.g. "claude-code".
        session_key: Durable identity of the session.
        path: Location of the session's store.
        mtime: The store's modification time (seconds since the epoch) as
            observed by `os.stat`, at discovery time.
    """

    source: str
    session_key: str
    path: Path
    mtime: float


@dataclass(frozen=True)
class TailResult:
    """The result of one `tail` call.

    Attributes:
        events: Canonical events read since the passed-in cursor, in file
            order. Empty if nothing new and complete was available.
        cursor: The cursor to persist and pass in next time. Always at or
            past the input cursor's offset, and never past the last
            complete record read.
    """

    events: tuple[Event, ...]
    cursor: Cursor


def open_source_readonly(path: str | Path) -> BinaryIO:
    """Open `path` with the OS read-only access mode, in binary mode.

    This is the one chokepoint every adapter operation in this module uses
    to open a source file. It asks the OS for `os.O_RDONLY` specifically,
    rather than a Python-level mode string that happens not to request a
    write — the distinction INV-2's gate test exists to enforce.

    Args:
        path: Path to the source file.

    Returns:
        A binary file object opened read-only.
    """
    fd = os.open(path, os.O_RDONLY)
    return os.fdopen(fd, "rb")


def read_complete_records(path: str | Path, start_offset: int) -> tuple[list[bytes], int]:
    """Read whole, newline-terminated JSONL records appended after `start_offset`.

    A record counts as complete only if it is terminated by a newline. The
    writer may have flushed a partial line at the moment this reads, so any
    trailing byte sequence with no terminating newline is dropped from the
    result and excluded from the returned offset — it is re-read whole (by
    a later call, once complete) rather than lost or handed to a JSON
    parser half-written.

    Args:
        path: Path to the JSONL source file.
        start_offset: Byte offset to start reading from (a cursor's offset).

    Returns:
        A `(records, new_offset)` pair. `records` is the list of complete
        line payloads (without their trailing newline), in file order.
        `new_offset` is where the caller's cursor should advance to; it is
        always `start_offset` plus the byte length of exactly the complete
        records returned, so it never points past the last complete record.
    """
    with open_source_readonly(path) as f:
        f.seek(start_offset)
        raw = f.read()

    lines = raw.split(b"\n")
    # split(b"\n") always yields at least one element; the last one is
    # whatever follows the final newline in `raw` — empty if `raw` ended
    # exactly on a newline, or a not-yet-complete line otherwise.
    trailing_partial = lines.pop()
    complete = [line for line in lines if line]

    consumed = len(raw) - len(trailing_partial)
    new_offset = start_offset + consumed
    return complete, new_offset


class Adapter(ABC):
    """One source's session store: discovery, unresolved-tool-use check, and tail.

    Concrete adapters (Claude Code, Codex, OpenCode) implement the four
    abstract methods below over their own store format. `discover_sessions`
    is not abstract: its floor-not-filter windowing logic is the same for
    every source and lives here once, built on top of those primitives.
    """

    source: str

    @abstractmethod
    def list_store_paths(self) -> Iterable[Path]:
        """Enumerate this source's session store paths.

        Must not open any of them — `discover_sessions` calls this before
        it has decided which sessions are even in scope.
        """

    @abstractmethod
    def session_key_for(self, path: Path) -> str:
        """Derive a session's durable identity from its store path."""

    @abstractmethod
    def has_unresolved_trailing_tool_use(self, path: Path) -> bool:
        """Report whether `path`'s last record is an unresolved `tool_use`.

        This is the one primitive `discover_sessions` calls on a session
        older than `since`, and only for sessions inside `outer_window` — the
        bound that keeps the always-include rule a check over a bounded set
        of files rather than a full-archive scan. Implementations must open
        `path` via `open_source_readonly`; reading from the end of the file
        rather than the whole of it is an optimization implementations may
        apply, not a requirement of this interface.
        """

    @abstractmethod
    def tail(self, path: Path, cursor: Cursor) -> TailResult:
        """Read every complete record appended after `cursor`, read-only.

        Implementations must open `path` via `open_source_readonly` and
        must not advance the returned cursor past a not-yet-complete
        record (see `read_complete_records`).
        """

    def discover_sessions(
        self,
        *,
        since: timedelta | None = DEFAULT_SINCE,
        all: bool = False,
        outer_window: timedelta = DEFAULT_OUTER_WINDOW,
        now: datetime | None = None,
    ) -> list[SessionRef]:
        """Return sessions worth looking at: a floor, not a filter.

        Args:
            since: Only sessions whose store mtime is within `since` of
                `now` are included on recency alone. Ignored if `all` is
                True.
            all: If True, `since`/`outer_window` filtering is skipped
                entirely and every session `list_store_paths` yields is
                returned.
            outer_window: Upper bound on how old a session may be and still
                be checked for an unresolved trailing `tool_use`. A session
                older than this is excluded on its mtime alone and is never
                opened.
            now: Reference time for age calculations. Defaults to the
                current UTC time; tests pass a fixed value.

        Returns:
            Discovered sessions, each backed by a real store path.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        refs = []
        for path in self.list_store_paths():
            if all:
                refs.append(self._ref_for(path))
                continue

            mtime = os.stat(path).st_mtime
            age = now - datetime.fromtimestamp(mtime, tz=timezone.utc)

            if since is not None and age <= since:
                refs.append(self._ref_for(path, mtime=mtime))
            elif age <= outer_window and self.has_unresolved_trailing_tool_use(path):
                refs.append(self._ref_for(path, mtime=mtime))
            # else: older than outer_window (or resolved), and never opened.

        return refs

    def _ref_for(self, path: Path, mtime: float | None = None) -> SessionRef:
        if mtime is None:
            mtime = os.stat(path).st_mtime
        return SessionRef(
            source=self.source,
            session_key=self.session_key_for(path),
            path=path,
            mtime=mtime,
        )
