"""Tests for the Codex rollout adapter and its tier-4 cap (task 7.1).

Every rollout this module tails is a JSONL file it writes itself under
pytest's `tmp_path`, with prose invented for the test — no real
`~/.codex/sessions/` store is ever opened, globbed, or read (INV-2, INV-9).
`CodexAdapter` is always constructed with an explicit `root` pointing into
`tmp_path`.

The two committed artifacts (`tests/fixtures/labels/codex-role-labels.jsonl` and
`tests/fixtures/labels/codex-role-class-measurement.json`) are read where the
measurement itself is under test, because their *content* is the claim being
checked.

**Why the positive controls matter here more than usual.** Most of this file
asserts that something is refused: the cap holds, the channel is harness, the
tier is 4. A suite of refusals passes trivially against an implementation
that refuses everything — `cap_codex_tier` could be `return 4` and every
negative test would stay green. So each refusal has a sibling proving the
code can do the other thing under the right conditions: the cap lifts against
a synthetic passing measurement, the classifier returns the human channel for
a bare-prose record, and a non-boundary event does not produce a
turn-boundary kind.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- helpers ----------------------------------------------------------------


def _write_rollout(root: Path, name: str, records: list[dict]) -> Path:
    """Write a rollout file into a date-partitioned tree under `root`."""
    day = root / "2026" / "08" / "14"
    day.mkdir(parents=True, exist_ok=True)
    path = day / name
    path.write_bytes(b"".join((json.dumps(r) + "\n").encode("utf-8") for r in records))
    return path


def _session_meta(**overrides) -> dict:
    payload = {
        "id": "fixture-thread-1",
        "session_id": "fixture-thread-1",
        "cwd": "/tmp/fixture-codex-project",
    }
    payload.update(overrides)
    return {"type": "session_meta", "payload": payload}


def _message(role: str, text: str, block_type: str = "input_text") -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": block_type, "text": text}],
        },
    }


def _event(event_type: str, **payload) -> dict:
    return {"type": "event_msg", "payload": {"type": event_type, **payload}}


def _function_call(call_id: str | None = "call-1") -> dict:
    payload = {"type": "function_call", "name": "shell"}
    if call_id is not None:
        payload["call_id"] = call_id
    return {"type": "response_item", "payload": payload}
