"""The pane-pin escape hatch: its shape, decoder, and encoder."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .constants import CLAUDE_SOURCE, CODEX_SOURCE


@dataclass(frozen=True)
class PanePin:
    """A pane-local explicit source/session override."""

    source: str
    session_key: str


def parse_pin(raw: object) -> PanePin | None:
    """Decode a strict JSON pane pin, returning ``None`` for any invalid value."""
    if isinstance(raw, Mapping):
        value = raw
    else:
        if not isinstance(raw, str) or not raw:
            return None
        try:
            value = json.loads(raw)
        except TypeError, ValueError:
            return None
    if not isinstance(value, Mapping):
        return None
    if set(value) != {"source", "session_key"}:
        return None
    source = value.get("source")
    session_key = value.get("session_key")
    if source not in {CLAUDE_SOURCE, CODEX_SOURCE}:
        return None
    if not isinstance(session_key, str) or not session_key or "\\" in session_key:
        return None
    parts = session_key.split("/")
    if source == CLAUDE_SOURCE:
        if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
            return None
    elif len(parts) != 1 or parts[0] in {".", ".."}:
        return None
    return PanePin(source=source, session_key=session_key)


def encode_pin(source: str, session_key: str) -> str:
    """Encode a pane pin in the same JSON shape the reader accepts."""
    if source not in {CLAUDE_SOURCE, CODEX_SOURCE} or not session_key:
        raise ValueError("pin source and session_key must identify a supported source")
    return json.dumps({"source": source, "session_key": session_key}, separators=(",", ":"))
