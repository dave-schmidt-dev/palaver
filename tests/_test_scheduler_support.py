"""Tests for the observer tick scheduler and daemon (task 4.1).

The daemon's whole economic argument is one comparison: extract a session
only when its cursor moved. Everything here defends that claim and the two
properties that make it trustworthy rather than merely true in the happy
path.

What this module defends, test by test:

* **An idle store costs zero inference requests.** Ten ticks over a warm,
  unchanging fixture corpus record zero extraction requests — with the
  discovery count asserted non-zero in the same test, because "zero requests
  because nothing changed" and "zero requests because discovery found
  nothing" are the same number and completely different failures. The same
  test then appends one record and asserts the eleventh tick records exactly
  one request, so the counter that reported zero is proven able to count.
* **Warm cursors are what "static" means.** `Cursor()` defaults to offset 0,
  so a cold store makes *every* session look changed on the first tick. The
  idle test seeds each cursor to where a tail leaves it and asserts the
  seeded offsets are non-zero, so the zero it later reports is a real skip
  and not a discovery that never happened.
* **One request per tick per changed session**, not zero and not two — and a
  third tick with nothing appended drops back to zero, which is what
  separates "gated on change" from "gated on nothing".
* **One writer.** `sqlite3.connect` is patched with a spy that counts *live*
  connections, so `migrate()`'s own short-lived connections are counted too,
  not just the daemon's injected factory. Two ticks open none. The same test
  opens a second connection by hand and asserts the spy reports two, so the
  peak-of-one is a measurement rather than a blind spot.
* **INV-1's status channel.** One tick emits at least one status update, and
  — through the CLI, with the *default* channel rather than a recorder, so
  the assertion is not vacuous — stdout stays empty while stderr carries the
  progress.
* **At-least-once.** A failing extractor leaves the cursor where it was, so
  the next tick re-schedules the same session; a positive control then
  succeeds and shows the cursor does advance when extraction does.
* **Shrink recovery is not idleness.** A store that was truncated or swapped
  returns a *lower* cursor (task 1.3), which a `>` gate would read as idle
  and never repair. The gate is `!=`, and this pins it.

No real session store (`~/.claude/`, `~/.codex/`,
`~/.local/share/opencode/`) is opened, globbed, or read by this module —
every sample directory is built under pytest's `tmp_path`, either from
records invented in this file or copied byte-for-byte from the committed,
hand-authored corpus in `tests/fixtures/` (INV-3, INV-9). The only socket
any test here opens is to an in-process stub server bound to `127.0.0.1` on
an ephemeral port.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import CursorStore
from palaver.observer.daemon import (
    ObserverDaemon,
)

#: Fixed reference time; every fixture's mtime is set relative to it, so no
#: assertion in this module depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)


# --- helpers -----------------------------------------------------------------


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def _human(text: str = "please check the deploy", session_id: str = "session-1") -> dict:
    return {
        "type": "user",
        "sessionId": session_id,
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant(text: str = "on it", session_id: str = "session-1") -> dict:
    return {
        "type": "assistant",
        "sessionId": session_id,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _write_store(root: Path, project: str, session: str, records: list[dict]) -> Path:
    """Write one session store in the layout `ClaudeCodeAdapter` requires."""
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _set_mtime(path, timedelta(minutes=5))
    return path


class RecordingExtractor:
    """Stands in for the one inference request `ModelExtractor` issues per session.

    Counting calls here counts inference requests exactly, because the real
    extractor's `__call__` issues exactly one `ModelClient.complete` and the
    daemon calls the extractor exactly once per scheduled session. Every
    test that asserts a count of zero also, in the same test, drives this
    same class to a non-zero count — a counter that cannot count is not a
    measurement.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, work, *, conn, on_status) -> None:
        on_status(f"stub extraction for {work.ref.session_key}")
        self.calls.append(work.ref.session_key)


def _daemon(tmp_path: Path, sample_root: Path, extractor, **kwargs) -> ObserverDaemon:
    return ObserverDaemon(
        db_path=tmp_path / "store" / "palaver.db",
        adapters=(ClaudeCodeAdapter(root=sample_root),),
        cursors=CursorStore(tmp_path / "cursors"),
        extractor=extractor,
        all=True,
        **kwargs,
    )
