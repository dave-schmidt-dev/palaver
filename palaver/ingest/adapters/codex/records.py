"""
Parsing a single Codex JSONL record: the identity/measurement dataclasses, payload and
text extraction, and image-marker stripping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    CODEX_IMAGE_ATTACHMENT_MARKER,
    REQUIRED_LABELLED_RECORDS,
    TEXT_BLOCK_TYPES,
    logger,
)


class CodexTierCapError(RuntimeError):
    """Raised when a caller explicitly requests a tier the Codex cap forbids.

    `cap_codex_tier` demotes silently, which is right for a tier derived by
    the extraction pipeline. An *explicit* request for a tier above the cap
    is a different thing: silently demoting it would hide a caller that
    believes it is minting tier-1 provenance out of a heuristic channel
    classification. That caller should fail loudly instead.
    """


@dataclass(frozen=True)
class RoleClassMeasurement:
    """The committed counts from measuring `codex_role_class` against labels.

    Attributes:
        n_records: Labelled records in the classifier's measured domain.
        n_errors: Harness-labelled records the classifier called human. This
            is the error that matters: it is the one that mints a false
            tier-1, and it is the direction INV-8 exists to prevent.
        threshold_met: The measurement's own verdict, recomputed by
            `measure_role_class` rather than hand-written.
    """

    n_records: int
    n_errors: int
    threshold_met: bool

    @property
    def lifts_tier_cap(self) -> bool:
        """Whether this measurement is strong enough to lift the tier-4 cap.

        All three conditions, conjunctively. A measurement file that merely
        exists lifts nothing, and neither does one that reports a clean zero
        over too small a sample.
        """
        return (
            self.n_records >= REQUIRED_LABELLED_RECORDS
            and self.n_errors == 0
            and self.threshold_met
        )


@dataclass(frozen=True)
class CodexIdentity:
    """Identity read out of a rollout's `session_meta` record.

    Attributes:
        cwd: The session's working directory, the basis for project scoping.
        id: This thread's own stable id; matches the filename's UUID.
        session_id: The root session's id. Equal to `id` except for a
            subagent thread-spawn.
        parent_thread_id: The parent thread this file was spawned from, or
            `None` for a root session.
    """

    cwd: str | None
    id: str | None
    session_id: str | None
    parent_thread_id: str | None

    @property
    def is_subagent(self) -> bool:
        """Whether this rollout is a subagent thread rather than a root session."""
        if self.parent_thread_id is not None:
            return True
        return self.id is not None and self.session_id is not None and self.id != self.session_id


def _parse_record(raw: bytes, path: Path) -> dict | None:
    """Decode one complete JSONL line, logging and skipping on failure.

    Args:
        raw: A complete, newline-stripped line from `read_complete_records`.
        path: Source path, for the warning message only. The record's own
            bytes are truncated in the log and never the whole line (INV-9).

    Returns:
        The decoded record, or `None` if `raw` was not a parseable JSON
        object. A corrupt record is logged at WARNING rather than silently
        dropped: silent data loss is invisible, and the tail must not crash
        on one bad line either.
    """
    try:
        record = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        logger.warning("Unparseable Codex rollout record in %s", path)
        return None
    if not isinstance(record, dict):
        logger.warning("Non-object Codex rollout record in %s", path)
        return None
    return record


def _payload(record: dict) -> dict:
    """Return a record's `payload` object, or an empty dict if it has none."""
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _message_payload(record: dict) -> dict | None:
    """Return the `message` payload of a `response_item`, or `None`.

    A `response_item` can carry `function_call`, `reasoning`, and other
    payload types that are not messages and have no role to classify.
    """
    if record.get("type") != "response_item":
        return None
    payload = _payload(record)
    if payload.get("type") != "message":
        return None
    return payload


def message_role(record: dict) -> str | None:
    """Return the role of a `response_item` message record, or `None`.

    `None` means "this record carries no role at all" — a `session_meta`, an
    `event_msg`, a `compacted` envelope, or a non-message `response_item`.
    That is a different answer from an unrecognized role string, and callers
    depend on the distinction: a role-less record is never the human channel,
    but it is also outside the domain `codex_role_class` is *measured* over.

    Args:
        record: A decoded rollout record.

    Returns:
        The role string, or `None` if the record carries no message role.
    """
    payload = _message_payload(record)
    if payload is None:
        return None
    role = payload.get("role")
    return role if isinstance(role, str) else None


def message_text(record: dict) -> str:
    """Flatten a message record's content blocks to plain text.

    Args:
        record: A decoded rollout record.

    Returns:
        The concatenated text of every recognized text block, or `""` for a
        record that is not a message or carries no text.
    """
    payload = _message_payload(record)
    if payload is None:
        return ""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in TEXT_BLOCK_TYPES
    )


def strip_codex_image_attachment_markers(text: str) -> str:
    """Remove only Codex's literal image-attachment transport markers.

    The marker is embedded in a user-role message alongside any genuine prose.
    Keeping this source-specific shape here lets the summary reducer retain
    that prose without surfacing the temporary attachment path.
    """

    return CODEX_IMAGE_ATTACHMENT_MARKER.sub("", text)
