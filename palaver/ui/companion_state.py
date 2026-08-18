"""Private, atomic state transport between AutoLaunch and a companion pane.

The producer and renderer deliberately share files rather than an iTerm2
connection.  A renderer can therefore notice a dead producer and mark its
surface stale instead of leaving the last good answer looking current.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_STATE_BYTES = 64 * 1024
MAX_ITEMS = 8
MAX_TEXT = 512
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "producer_updated_at",
        "project",
        "source",
        "status",
        "join_state",
        "request",
        "command_result",
        "detail",
        "recent",
        "tasks",
        "questions",
    }
)


class CompanionStateError(ValueError):
    """A companion state file is absent, malformed, or unsupported."""


class JoinState(StrEnum):
    """How confidently the producer has paired this companion to an agent."""

    STARTING = "STARTING"
    JOINED = "JOINED"
    UNJOINED = "UNJOINED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class CompanionState:
    """One complete, immutable renderer snapshot (schema version 1)."""

    producer_updated_at: float
    project: str
    source: str
    status: str
    join_state: JoinState
    request: str | None = None
    command_result: str | None = None
    detail: str | None = None
    recent: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise CompanionStateError(f"unsupported companion state schema {self.schema_version!r}")
        if isinstance(self.producer_updated_at, bool) or not isinstance(
            self.producer_updated_at, (int, float)
        ):
            raise CompanionStateError("producer_updated_at must be a finite number")
        if not math.isfinite(float(self.producer_updated_at)):
            raise CompanionStateError("producer_updated_at must be a finite number")
        for name in ("project", "source", "status"):
            _validate_text(name, getattr(self, name), optional=False)
        if not isinstance(self.join_state, JoinState):
            raise CompanionStateError("join_state must be a JoinState")
        for name in ("request", "command_result", "detail"):
            _validate_text(name, getattr(self, name), optional=True)
        for name in ("recent", "tasks", "questions"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise CompanionStateError(f"{name} must be a tuple")
            if len(value) > MAX_ITEMS:
                raise CompanionStateError(f"{name} has more than {MAX_ITEMS} entries")
            for item in value:
                _validate_text(f"{name} item", item, optional=False)


def _validate_text(name: str, value: object, *, optional: bool) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise CompanionStateError(f"{name} must be text")
    if not optional and not value.strip():
        raise CompanionStateError(f"{name} must not be empty")
    if len(value) > MAX_TEXT:
        raise CompanionStateError(f"{name} exceeds {MAX_TEXT} characters")


def _as_payload(state: CompanionState) -> dict[str, Any]:
    return {
        "schema_version": state.schema_version,
        "producer_updated_at": float(state.producer_updated_at),
        "project": state.project,
        "source": state.source,
        "status": state.status,
        "join_state": state.join_state.value,
        "request": state.request,
        "command_result": state.command_result,
        "detail": state.detail,
        "recent": list(state.recent),
        "tasks": list(state.tasks),
        "questions": list(state.questions),
    }


def atomic_write_state(path: Path, state: CompanionState) -> None:
    """Durably replace ``path`` with one mode-0600 JSON snapshot."""

    destination = Path(path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(
        _as_payload(state), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if len(encoded) > MAX_STATE_BYTES:
        raise CompanionStateError(f"encoded state exceeds {MAX_STATE_BYTES} bytes")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_state(path: Path) -> CompanionState:
    """Read and strictly validate one schema-v1 state snapshot."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            encoded = handle.read(MAX_STATE_BYTES + 1)
        if len(encoded) > MAX_STATE_BYTES:
            raise CompanionStateError(f"state file exceeds {MAX_STATE_BYTES} bytes")
        payload = json.loads(encoded.decode("utf-8"))
    except CompanionStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompanionStateError("state file is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise CompanionStateError("state file must contain one object")
    keys = frozenset(payload)
    if keys != _REQUIRED_KEYS:
        raise CompanionStateError("state file fields do not match schema version 1")
    try:
        join_state = JoinState(payload["join_state"])
        return CompanionState(
            schema_version=payload["schema_version"],
            producer_updated_at=payload["producer_updated_at"],
            project=payload["project"],
            source=payload["source"],
            status=payload["status"],
            join_state=join_state,
            request=payload["request"],
            command_result=payload["command_result"],
            detail=payload["detail"],
            recent=_read_items("recent", payload["recent"]),
            tasks=_read_items("tasks", payload["tasks"]),
            questions=_read_items("questions", payload["questions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CompanionStateError):
            raise
        raise CompanionStateError("state file values do not match schema version 1") from exc


def _read_items(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CompanionStateError(f"{name} must be a list")
    return tuple(value)
