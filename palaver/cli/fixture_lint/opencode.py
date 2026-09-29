"""
OpenCode's record shapes (task 7.2, this module's own fixture-format invention):
structural constants, checkers, classify_* functions, and OPENCODE_RECORD_SHAPES.
"""

from __future__ import annotations

from typing import Callable

from .checks import (
    ACCEPTED,
    Verdict,
    _check_bool,
    _check_keys,
    _check_literal,
    _check_pattern,
    _check_text,
    _reject,
)
from .constants import (
    RULE_BAD_VALUE,
    RULE_MISSING_KEY,
    RULE_NOT_AN_OBJECT,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_SUBTYPE,
    RULE_UNKNOWN_TOOL,
    SESSION_ID,
    TOOL_USE_ID,
)

# --- OpenCode ------------------------------------------------------------
#
# OpenCode's real store is SQLite rows (`message`, `part`) with a JSON `data`
# column — there is no adapter yet (task 7.2) and no natural JSONL shape to
# borrow one from. `opencode_message` / `opencode_part` below are this
# module's own fixture-format invention, documented at the top of this file:
# one JSON object per row, carrying only the columns and `data` fields a
# future adapter is expected to read, per `docs/research.md` §3.

#: `part.data.type` values this corpus classifies: a plain text turn, a tool
#: invocation, the terminal marker that doubly-confirms a turn boundary, and
#: the (rare) compaction marker.
OPENCODE_PART_TYPES = frozenset({"text", "tool", "step-finish", "compaction"})

#: `message.data.finish` and `part.data.reason` (on a `step-finish` part)
#: share one vocabulary — `docs/research.md` §3 documents them as the same
#: semantic space, confirmed paired on a finished turn (`finish="stop"` with a
#: terminal `step-finish` part `reason="stop"`).
OPENCODE_FINISH_VALUES = frozenset({"stop", "tool-calls", "unknown"})

#: `part.data.tool` on a `type: "tool"` part. Invented, structurally plausible
#: identifiers — sampling redacted string values, so this is *not* claimed to
#: be OpenCode's exact tool-name vocabulary; task 7.2 measures and owns that.
OPENCODE_TOOL_NAMES = frozenset({"bash", "read", "edit"})

#: `part.data.state.status` on a `type: "tool"` part (`docs/research.md` §3).
OPENCODE_TOOL_STATUSES = frozenset({"completed", "error"})


# --- OpenCode classifiers -------------------------------------------------


def _check_opencode_tool_state(state: object, where: str) -> Verdict | None:
    """Allowlist a `type: "tool"` part's `state` object."""
    if not isinstance(state, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    verdict = _check_keys(state, frozenset({"status"}), frozenset({"error"}), where)
    if verdict is not None:
        return verdict
    verdict = _check_literal(state["status"], OPENCODE_TOOL_STATUSES, f"{where}.status")
    if verdict is not None:
        return verdict
    if state["status"] == "error":
        if "error" not in state:
            return _reject(RULE_MISSING_KEY, f"{where} has status 'error' but no error message")
        return _check_text(state["error"], f"{where}.error")
    if "error" in state:
        return _reject(RULE_UNEXPECTED_KEY, f"{where} carries 'error' without status 'error'")
    return None


def _classify_opencode_message(record: dict) -> Verdict:
    verdict = _check_keys(
        record, frozenset({"type", "id", "session_id", "data"}), frozenset(), "record"
    )
    if verdict is not None:
        return verdict
    verdict = _check_pattern(record["id"], SESSION_ID, "record.id")
    if verdict is not None:
        return verdict
    verdict = _check_pattern(record["session_id"], SESSION_ID, "record.session_id")
    if verdict is not None:
        return verdict
    data = record["data"]
    if not isinstance(data, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.data is not an object")
    verdict = _check_keys(data, frozenset({"role"}), frozenset({"finish", "error"}), "record.data")
    if verdict is not None:
        return verdict
    verdict = _check_literal(data["role"], frozenset({"user", "assistant"}), "record.data.role")
    if verdict is not None:
        return verdict
    if "finish" in data:
        verdict = _check_literal(data["finish"], OPENCODE_FINISH_VALUES, "record.data.finish")
        if verdict is not None:
            return verdict
    if "error" in data:
        verdict = _check_text(data["error"], "record.data.error")
        if verdict is not None:
            return verdict
    return ACCEPTED


def _classify_opencode_part(record: dict) -> Verdict:
    verdict = _check_keys(
        record,
        frozenset({"type", "id", "message_id", "session_id", "data"}),
        frozenset(),
        "record",
    )
    if verdict is not None:
        return verdict
    for key in ("id", "message_id", "session_id"):
        verdict = _check_pattern(record[key], SESSION_ID, f"record.{key}")
        if verdict is not None:
            return verdict
    data = record["data"]
    if not isinstance(data, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.data is not an object")
    part_type = data.get("type")
    if part_type not in OPENCODE_PART_TYPES:
        return _reject(
            RULE_UNKNOWN_SUBTYPE,
            f"record.data.type {str(part_type)[:40]!r} is not a part type this corpus "
            f"classifies ({sorted(OPENCODE_PART_TYPES)})",
        )
    if part_type == "text":
        verdict = _check_keys(
            data, frozenset({"type", "text"}), frozenset({"synthetic"}), "record.data"
        )
        if verdict is not None:
            return verdict
        verdict = _check_text(data["text"], "record.data.text")
        if verdict is not None:
            return verdict
        if "synthetic" in data and data["synthetic"] is not True:
            return _reject(
                RULE_BAD_VALUE,
                f"record.data.synthetic must be true when present, got {data['synthetic']!r}",
            )
        return ACCEPTED
    if part_type == "tool":
        verdict = _check_keys(
            data, frozenset({"type", "tool", "callID", "state"}), frozenset(), "record.data"
        )
        if verdict is not None:
            return verdict
        if not isinstance(data["tool"], str) or data["tool"] not in OPENCODE_TOOL_NAMES:
            return _reject(
                RULE_UNKNOWN_TOOL,
                f"record.data.tool must be one of {sorted(OPENCODE_TOOL_NAMES)}, "
                f"got {data['tool']!r}",
            )
        verdict = _check_pattern(data["callID"], TOOL_USE_ID, "record.data.callID")
        if verdict is not None:
            return verdict
        verdict = _check_opencode_tool_state(data["state"], "record.data.state")
        return verdict if verdict is not None else ACCEPTED
    if part_type == "step-finish":
        verdict = _check_keys(data, frozenset({"type", "reason"}), frozenset(), "record.data")
        if verdict is not None:
            return verdict
        verdict = _check_literal(data["reason"], OPENCODE_FINISH_VALUES, "record.data.reason")
        return verdict if verdict is not None else ACCEPTED
    # compaction
    verdict = _check_keys(
        data, frozenset({"type", "auto", "tail_start_id"}), frozenset(), "record.data"
    )
    if verdict is not None:
        return verdict
    verdict = _check_bool(data["auto"], "record.data.auto")
    if verdict is not None:
        return verdict
    verdict = _check_pattern(data["tail_start_id"], SESSION_ID, "record.data.tail_start_id")
    return verdict if verdict is not None else ACCEPTED


#: OpenCode's shape allowlist, keyed by this module's own fixture-format
#: discriminator (see the module-level OpenCode note) rather than a real
#: column name.
OPENCODE_RECORD_SHAPES: dict[str, Callable[[dict], Verdict]] = {
    "opencode_message": _classify_opencode_message,
    "opencode_part": _classify_opencode_part,
}
