"""Tests for the structural turn boundary and `palaver diagnose --coverage`.

Every fixture here is a JSONL file this module writes under pytest's
`tmp_path`, with prose invented for the test. No real session store
(`~/.claude/`, `~/.codex/`, `~/.local/share/opencode/`) is opened, globbed, or
read — INV-3 forbids it, and the CLI is always given an explicit `--sample`
pointing into `tmp_path`.

The module's sharpest tests are the pair that flip a single `isMeta` byte:
a trailing harness-injected `user` record must not be read as a human turn,
and reading it as one inverts the status of exactly the sessions that use
hooks and skills most.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.observer.signals import (
    Status,
    derive_status,
)
from palaver.observer.turn_boundary import (
    observe_session,
)

#: Fixed reference time; every fixture's mtime is set relative to it, so no
#: assertion in this module depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- fixture builders --------------------------------------------------------


def _line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write(path: Path, items: list[dict | bytes]) -> Path:
    """Write a JSONL fixture. A `bytes` item is written verbatim (corrupt lines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(item if isinstance(item, bytes) else _line(item) for item in items))
    return path


def _session(tmp_path: Path, name: str, items: list[dict | bytes]) -> Path:
    """Write one session store into a Claude-Code-shaped sample directory."""
    return _write(tmp_path / "projects" / "-Users-test-project" / f"{name}.jsonl", items)


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def _human(text: str = "please run the build") -> dict:
    return {
        "type": "user",
        "sessionId": "session-1",
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _injected(text: str = "<system-reminder>A hook fired.</system-reminder>") -> dict:
    """A harness-injected `type: "user"` record — the same role, nothing said."""
    record = _human(text)
    record["isMeta"] = True
    return record


def _assistant(text: str = "the build is green") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _tool_use(name: str = "Bash") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu-1", "name": name, "input": {}}],
        },
    }


def _tool_result(*, is_error: bool = False, content: str = "ok") -> dict:
    return {
        "type": "user",
        "sessionId": "session-1",
        "isMeta": False,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-1",
                    "is_error": is_error,
                    "content": content,
                }
            ],
        },
    }


def _background_tool_result() -> dict:
    record = _tool_result()
    record["toolUseResult"] = {"backgroundTaskId": "bg-1"}
    return record


def _background_notification(status: str = "completed") -> dict:
    return {
        "type": "queue-operation",
        "operation": "enqueue",
        "content": (
            "<task-notification><task-id>bg-1</task-id>"
            f"<status>{status}</status></task-notification>"
        ),
    }


def _stop_hook(subtype: str = "stop_hook_summary") -> dict:
    return {"type": "system", "subtype": subtype, "sessionId": "session-1", "summary": "hook ran"}


def _bookkeeping() -> dict:
    return {"type": "mode", "sessionId": "session-1", "mode": "default"}


def _observe(path: Path):
    return observe_session(path, now=NOW)


def _status(path: Path) -> Status:
    return derive_status(_observe(path).signals)


# --- palaver diagnose --coverage ---------------------------------------------


def _coverage_sample(tmp_path: Path) -> Path:
    """Write a five-session sample with one deliberately undeterminable session.

    The bookkeeping-only session is what keeps the coverage assertions from
    being satisfiable by a command that prints 100% unconditionally.
    """
    _session(tmp_path, "a-final", [_human(), _assistant()])
    _session(tmp_path, "b-mid-call", [_human(), _tool_use()])
    _session(tmp_path, "c-hooked", [_human(), _assistant(), _injected()])
    _session(tmp_path, "d-error", [_human(), _tool_use(), _tool_result(is_error=True)])
    _session(tmp_path, "e-bookkeeping", [_bookkeeping()])
    return tmp_path / "projects"
