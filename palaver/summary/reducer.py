"""Source-neutral deterministic summary reduction.

``reduce_events`` and ``SummaryReducer.feed`` execute the same source-specific
transition. The incremental wrapper retains only the bounded snapshot and
correlation fields embedded in it; each tail batch therefore costs O(batch),
not O(session history), and full replay is the same transition from an empty
snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable

from palaver.ingest.adapters.base import Event
from palaver.summary.model import SummarySnapshot

SUPPORTED_SOURCES = frozenset({"claude-code", "codex"})


def reduce_events(
    source: str,
    session_key: str,
    events: Iterable[Event],
    *,
    malformed_records: int = 0,
) -> SummarySnapshot:
    """Replay source-native canonical events into one immutable snapshot."""
    if source == "claude-code":
        from palaver.summary.claude_code import reduce_claude_events

        snapshot = reduce_claude_events(session_key, tuple(events))
    elif source == "codex":
        from palaver.summary.codex import reduce_codex_events

        snapshot = reduce_codex_events(session_key, tuple(events))
    else:
        return SummarySnapshot(source=source, session_key=session_key).with_unknown(
            f"unsupported summary source: {source}"
        )

    if malformed_records:
        return snapshot.with_unknown(f"{malformed_records} malformed complete record(s)")
    return snapshot


def _apply_events(
    source: str,
    session_key: str,
    events: tuple[Event, ...],
    initial: SummarySnapshot | None,
) -> SummarySnapshot:
    if source == "claude-code":
        from palaver.summary.claude_code import reduce_claude_events

        return reduce_claude_events(session_key, events, initial=initial)
    if source == "codex":
        from palaver.summary.codex import reduce_codex_events

        return reduce_codex_events(session_key, events, initial=initial)
    return SummarySnapshot(source=source, session_key=session_key).with_unknown(
        f"unsupported summary source: {source}"
    )


class SummaryReducer:
    """Incremental event collector with explicit replacement after source shrink."""

    def __init__(self, source: str, session_key: str) -> None:
        self.source = source
        self.session_key = session_key
        self._snapshot = _apply_events(source, session_key, (), None)
        self._malformed_records = 0

    @property
    def snapshot(self) -> SummarySnapshot:
        """Return the snapshot for all complete events seen so far."""
        if self._malformed_records:
            return self._snapshot.with_unknown(
                f"{self._malformed_records} malformed complete record(s)"
            )
        return self._snapshot

    def feed(
        self,
        events: Iterable[Event],
        *,
        malformed_records: int = 0,
        replace: bool = False,
    ) -> SummarySnapshot:
        """Add a tail batch, or replace prior history after truncation/replacement."""
        batch = tuple(events)
        if replace:
            self._snapshot = _apply_events(self.source, self.session_key, batch, None)
            self._malformed_records = malformed_records
        else:
            self._snapshot = _apply_events(self.source, self.session_key, batch, self._snapshot)
            self._malformed_records += malformed_records
        return self.snapshot


__all__ = ["SUPPORTED_SOURCES", "SummaryReducer", "reduce_events"]
