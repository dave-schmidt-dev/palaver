"""Durable per-session tail cursors.

Each adapter session is tailed incrementally: every tick reads whatever a
source has appended since the last tick and nothing more. The cursor that
tracks "since the last tick" has to survive a Palaver restart without
re-ingesting a record already stored (which would duplicate events and
evidence) or skipping one that was never read (which would silently lose
transcript). `CursorStore` is the durability half of that guarantee; the
other half — never advancing past a torn, not-yet-complete write — lives in
`palaver.ingest.adapters.base.read_complete_records`, which is what actually
produces the offsets this module persists.

That no-re-ingest guarantee has one deliberate exception, and it lives in
`read_complete_records` rather than here: if a source file shrinks below a
stored offset — truncated in place, or replaced by a shorter file under the
same path — the offset no longer refers to anything, and reading from it
would return empty forever while the session went silently dark. Recovery
re-reads from zero, re-ingesting once, because a bounded duplicate is
recoverable and a permanently invisible session is not. A cursor handed back
after that repair is *behind* the one passed in, which is the only case where
that happens.

A cursor is written to its own file, one per `(source, session)` pair, so
different adapters can never share an offset. Each write goes to a temp file
in the same directory and is atomically renamed into place with `os.replace`,
so a crash mid-write leaves either the old cursor file or the new one, never a
half-written one that would fail to parse on the next restart.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cursor:
    """A byte offset into a session store, past the last fully-ingested record.

    Attributes:
        offset: Byte offset of the first not-yet-read byte. 0 means nothing
            has been read yet.
    """

    offset: int = 0


class CursorStore:
    """Persists one `Cursor` per `(source, session_key)` as JSON on disk."""

    def __init__(self, root: str | Path) -> None:
        """Create (if absent) and use `root` as the cursor directory.

        Args:
            root: Directory the per-session cursor files live under.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, source: str, session_key: str) -> Path:
        # session_key may contain characters that are unsafe in a filename
        # (path separators, etc.), so the on-disk name is a digest of it
        # rather than the key itself. The key is still recorded inside the
        # file for debuggability.
        digest = hashlib.sha256(f"{source}\0{session_key}".encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _legacy_path_for(self, session_key: str) -> Path:
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def load(self, session_key: str, *, source: str = "claude-code") -> Cursor:
        """Return the stored cursor for `session_key`, or offset 0 if none exists.

        Args:
            session_key: Durable identity of the session, as produced by an
                adapter's `session_key_for`.
            source: Adapter source namespace. Defaults to Claude Code so
                existing callers retain the legacy behavior.

        Returns:
            The persisted `Cursor`, or a fresh `Cursor(offset=0)` the first
            time this session is ever tailed.
        """
        path = self._path_for(source, session_key)
        if not path.exists() and source == "claude-code":
            path = self._legacy_path_for(session_key)
        if not path.exists():
            return Cursor()
        data = json.loads(path.read_text(encoding="utf-8"))
        return Cursor(offset=data["offset"])

    def save(
        self,
        session_key: str,
        cursor: Cursor,
        *,
        source: str = "claude-code",
    ) -> None:
        """Persist `cursor` for `session_key`, atomically.

        Args:
            session_key: Durable identity of the session.
            cursor: The new cursor value to persist.
            source: Adapter source namespace. Defaults to Claude Code for
                compatibility with existing callers.
        """
        path = self._path_for(source, session_key)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps({"source": source, "session_key": session_key, "offset": cursor.offset}),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)


class FailureBackoffStore:
    """Persist retry timing beside cursors for the observer's one writer.

    This intentionally shares the cursor root: both describe what the
    observer has yet to finish for one source/session, and an explicit
    ``--cursors`` move must move both. The single-writer daemon is the only
    caller, so no cross-process merge protocol is needed.
    """

    _FILENAME = "failure-backoff.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self._FILENAME

    @staticmethod
    def _key(source: str, session_key: str) -> str:
        return f"{source}\0{session_key}"

    def _load(self) -> dict[str, dict[str, float | int | str]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, entries: dict[str, dict[str, float | int | str]]) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def retry_at(self, session_key: str, *, source: str) -> float | None:
        entry = self._load().get(self._key(source, session_key))
        value = None if entry is None else entry.get("retry_at")
        return float(value) if isinstance(value, int | float) else None

    def record_failure(
        self,
        session_key: str,
        *,
        source: str,
        now: float,
        initial_seconds: float,
        maximum_seconds: float,
    ) -> float:
        entries = self._load()
        key = self._key(source, session_key)
        previous = entries.get(key, {})
        attempts = int(previous.get("attempts", 0)) + 1
        delay = min(initial_seconds * (2 ** (attempts - 1)), maximum_seconds)
        retry_at = now + delay
        entries[key] = {
            "source": source,
            "session_key": session_key,
            "attempts": attempts,
            "retry_at": retry_at,
        }
        self._save(entries)
        return retry_at

    def clear(self, session_key: str, *, source: str) -> None:
        entries = self._load()
        if entries.pop(self._key(source, session_key), None) is not None:
            self._save(entries)
