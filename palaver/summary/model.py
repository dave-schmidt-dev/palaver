"""Immutable values shared by deterministic session-summary reducers.

Every semantic field carries provenance.  ``UNKNOWN`` is deliberately not an
empty string or empty tuple: absence of a structured plan is not proof that a
session has no tasks, while a successfully parsed empty ``TodoWrite`` is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

DISPLAY_TEXT_LIMIT = 240
# The companion pane's activity section is the one that grows into whatever
# rows the other sections leave unused, so this matches the state schema's
# per-list ceiling rather than the single row the pane used to spare.
RECENT_ACTIVITY_LIMIT = 8
MAX_COLLECTION_ITEMS = 32
MAX_BACKGROUND_TASKS = 32
MAX_UNKNOWN_REASONS = 8

# CSI and OSC cover the terminal control sequences coding tools commonly put
# in command output. Remaining C0/C1 controls are removed below.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WHITESPACE_RE = re.compile(r"\s+")


class Provenance(Enum):
    """How strongly a displayed value is supported by source structure."""

    EXACT = "exact"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


def sanitize_text(value: object, *, limit: int = DISPLAY_TEXT_LIMIT) -> str:
    """Return bounded single-line display text with terminal controls removed."""
    if not isinstance(value, str):
        return ""
    text = _ANSI_RE.sub("", value)
    text = _CONTROL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"[:limit]
    return text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class Claim:
    """One optional string and the source evidence supporting it."""

    text: str | None
    provenance: Provenance
    evidence_kind: str | None = None
    evidence_id: str | None = None
    reason: str | None = None

    @classmethod
    def unknown(cls, reason: str) -> Claim:
        """Construct an explicitly unknown claim."""
        return cls(None, Provenance.UNKNOWN, reason=reason)

    @classmethod
    def exact(cls, text: object, kind: str, evidence_id: str | None = None) -> Claim:
        """Construct an exact source-text claim, sanitized for display."""
        rendered = sanitize_text(text)
        if not rendered:
            return cls.unknown(f"{kind} carried no displayable text")
        return cls(rendered, Provenance.EXACT, kind, evidence_id)

    @classmethod
    def structural(cls, text: str, kind: str, evidence_id: str | None = None) -> Claim:
        """Construct a fact derived only from recognized record structure."""
        return cls(sanitize_text(text), Provenance.STRUCTURAL, kind, evidence_id)


@dataclass(frozen=True)
class TaskItem:
    """One task from a source-native structured plan snapshot."""

    text: str
    status: str


T = TypeVar("T")


@dataclass(frozen=True)
class CollectionClaim(Generic[T]):
    """A collection whose emptiness is meaningful only when provenance is known."""

    items: tuple[T, ...]
    provenance: Provenance
    evidence_kind: str | None = None
    evidence_id: str | None = None
    reason: str | None = None

    @classmethod
    def unknown(cls, reason: str) -> CollectionClaim[T]:
        """Construct an explicitly unknown collection."""
        return cls((), Provenance.UNKNOWN, reason=reason)


@dataclass(frozen=True)
class RecentActivity:
    """One bounded, display-ready observation from the source event stream."""

    text: str
    provenance: Provenance
    evidence_kind: str
    evidence_id: str | None = None


@dataclass(frozen=True)
class SummarySnapshot:
    """All deterministic companion content for one source session."""

    source: str
    session_key: str
    request: Claim = field(default_factory=lambda: Claim.unknown("no genuine human request"))
    recent: tuple[RecentActivity, ...] = ()
    tasks: CollectionClaim[TaskItem] = field(
        default_factory=lambda: CollectionClaim.unknown("no structured plan snapshot")
    )
    questions: CollectionClaim[Claim] = field(
        default_factory=lambda: CollectionClaim((), Provenance.STRUCTURAL, "event_stream")
    )
    command_result: Claim = field(
        default_factory=lambda: Claim.unknown("no command failure this turn")
    )
    background_tasks: frozenset[str] = field(default_factory=frozenset)
    compaction: Claim = field(default_factory=lambda: Claim.unknown("no compaction observed"))
    turn: Claim = field(default_factory=lambda: Claim.unknown("turn boundary not observed"))
    source_integrity: Provenance = Provenance.EXACT
    unknown_reasons: tuple[str, ...] = ()

    def with_unknown(self, reason: str) -> SummarySnapshot:
        """Return a fail-closed snapshot after an incomplete source read."""
        reasons = (
            self.unknown_reasons
            if reason in self.unknown_reasons
            else (*self.unknown_reasons, reason)[-MAX_UNKNOWN_REASONS:]
        )
        return SummarySnapshot(
            source=self.source,
            session_key=self.session_key,
            request=Claim.unknown(reason),
            recent=self.recent,
            tasks=CollectionClaim.unknown(reason),
            questions=CollectionClaim.unknown(reason),
            command_result=Claim.unknown(reason),
            background_tasks=self.background_tasks,
            compaction=self.compaction,
            turn=Claim.unknown(reason),
            source_integrity=Provenance.UNKNOWN,
            unknown_reasons=reasons,
        )


def append_recent(
    current: tuple[RecentActivity, ...],
    text: object,
    provenance: Provenance,
    kind: str,
    evidence_id: str | None = None,
) -> tuple[RecentActivity, ...]:
    """Append one sanitized activity item while retaining only the display budget."""
    rendered = sanitize_text(text)
    if not rendered:
        return current
    result = (*current, RecentActivity(rendered, provenance, kind, evidence_id))
    return result[-RECENT_ACTIVITY_LIMIT:]


def append_unknown_reason(current: tuple[str, ...], reason: str) -> tuple[str, ...]:
    """Append one deduplicated diagnostic while keeping snapshot state bounded."""
    if reason in current:
        return current
    return (*current, reason)[-MAX_UNKNOWN_REASONS:]
