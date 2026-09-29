"""
Claude Code's record shapes: block/message checkers, the five classify_* functions, and
RECORD_SHAPES.
"""

from __future__ import annotations

from typing import Callable

from palaver.ingest.adapters.claude_code import SYSTEM_SUBTYPE_KINDS

from .checks import (
    ACCEPTED,
    Verdict,
    _check_bool,
    _check_input_value,
    _check_keys,
    _check_literal,
    _check_pattern,
    _check_text,
    _reject,
)
from .constants import (
    RULE_BAD_VALUE,
    RULE_NOT_AN_OBJECT,
    RULE_UNKNOWN_CONTENT_BLOCK,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_UNKNOWN_TOOL,
    SESSION_ID,
    TOOL_USE_ID,
)

#: Tool names a fixture may name. Small and explicit: a tool name is the one
#: piece of a `tool_use` block that could otherwise carry an unreviewed string.
TOOL_NAMES = frozenset({"Bash", "Read", "Edit", "AskUserQuestion"})

#: `mode` record values. Structural literals, so no phrasebook check applies.
MODE_VALUES = frozenset({"default", "plan", "acceptEdits"})


def _check_assistant_block(block: object, where: str) -> Verdict | None:
    """Allowlist one content block of an `assistant` record."""
    if not isinstance(block, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    block_type = block.get("type")
    if block_type == "text":
        verdict = _check_keys(block, frozenset({"type", "text"}), frozenset(), where)
        return verdict if verdict is not None else _check_text(block["text"], f"{where}.text")
    if block_type == "tool_use":
        verdict = _check_keys(block, frozenset({"type", "id", "name", "input"}), frozenset(), where)
        if verdict is not None:
            return verdict
        verdict = _check_pattern(block["id"], TOOL_USE_ID, f"{where}.id")
        if verdict is not None:
            return verdict
        if not isinstance(block["name"], str) or block["name"] not in TOOL_NAMES:
            return _reject(
                RULE_UNKNOWN_TOOL,
                f"{where}.name must be one of {sorted(TOOL_NAMES)}, got {block['name']!r}",
            )
        if not isinstance(block["input"], dict):
            return _reject(RULE_BAD_VALUE, f"{where}.input must be an object")
        return _check_input_value(block["input"], f"{where}.input", 0)
    return _reject(
        RULE_UNKNOWN_CONTENT_BLOCK,
        f"{where} has block type {block_type!r}, which no assistant shape declares",
    )


def _check_user_block(block: object, where: str) -> Verdict | None:
    """Allowlist one content block of a `user` record."""
    if not isinstance(block, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    block_type = block.get("type")
    if block_type == "text":
        verdict = _check_keys(block, frozenset({"type", "text"}), frozenset(), where)
        return verdict if verdict is not None else _check_text(block["text"], f"{where}.text")
    if block_type == "tool_result":
        verdict = _check_keys(
            block,
            frozenset({"type", "tool_use_id", "is_error", "content"}),
            frozenset(),
            where,
        )
        if verdict is not None:
            return verdict
        verdict = _check_pattern(block["tool_use_id"], TOOL_USE_ID, f"{where}.tool_use_id")
        if verdict is not None:
            return verdict
        verdict = _check_bool(block["is_error"], f"{where}.is_error")
        if verdict is not None:
            return verdict
        return _check_text(block["content"], f"{where}.content")
    return _reject(
        RULE_UNKNOWN_CONTENT_BLOCK,
        f"{where} has block type {block_type!r}, which no user shape declares",
    )


def _check_message(
    record: dict, role: str, block_checker: Callable[[object, str], Verdict | None]
) -> Verdict | None:
    """Allowlist a record's `message` envelope and every content block in it."""
    message = record.get("message")
    if not isinstance(message, dict):
        return _reject(RULE_NOT_AN_OBJECT, "message is not an object")
    verdict = _check_keys(message, frozenset({"role", "content"}), frozenset(), "message")
    if verdict is not None:
        return verdict
    verdict = _check_literal(message["role"], frozenset({role}), "message.role")
    if verdict is not None:
        return verdict
    content = message["content"]
    if not isinstance(content, list):
        return _reject(RULE_BAD_VALUE, "message.content must be a list of blocks")
    if not content:
        return _reject(RULE_BAD_VALUE, "message.content is empty")
    for index, block in enumerate(content):
        verdict = block_checker(block, f"message.content[{index}]")
        if verdict is not None:
            return verdict
    return None


def _classify_user(record: dict) -> Verdict:
    verdict = _check_keys(
        record,
        frozenset({"type", "sessionId", "isMeta", "message"}),
        frozenset(),
        "record",
    )
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_bool(record["isMeta"], "record.isMeta")
    if verdict is None:
        verdict = _check_message(record, "user", _check_user_block)
    return verdict if verdict is not None else ACCEPTED


def _classify_assistant(record: dict) -> Verdict:
    verdict = _check_keys(
        record, frozenset({"type", "sessionId", "message"}), frozenset(), "record"
    )
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_message(record, "assistant", _check_assistant_block)
    return verdict if verdict is not None else ACCEPTED


def _classify_system(record: dict) -> Verdict:
    verdict = _check_keys(
        record,
        frozenset({"type", "subtype", "sessionId"}),
        frozenset({"content", "summary"}),
        "record",
    )
    if verdict is not None:
        return verdict
    subtype = record["subtype"]
    if not isinstance(subtype, str) or subtype not in SYSTEM_SUBTYPE_KINDS:
        return _reject(
            RULE_UNKNOWN_SYSTEM_SUBTYPE,
            f"record.subtype {str(subtype)[:40]!r} is not a subtype the Claude Code "
            f"adapter classifies ({sorted(SYSTEM_SUBTYPE_KINDS)})",
        )
    verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is not None:
        return verdict
    for key in ("content", "summary"):
        if key in record:
            verdict = _check_text(record[key], f"record.{key}")
            if verdict is not None:
                return verdict
    return ACCEPTED


def _classify_mode(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "sessionId", "mode"}), frozenset(), "record")
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_literal(record["mode"], MODE_VALUES, "record.mode")
    return verdict if verdict is not None else ACCEPTED


def _classify_ai_title(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "sessionId", "title"}), frozenset(), "record")
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_text(record["title"], "record.title")
    return verdict if verdict is not None else ACCEPTED


#: The shape allowlist. A record type absent from this mapping is unclassified
#: and fails — including every bookkeeping type the adapter tolerates at
#: runtime (`attachment`, `last-prompt`, `bridge-session`, `summary`). The
#: adapter may safely *ignore* a record type it does not model; the corpus may
#: not safely *ship* one nobody has reviewed for content.
RECORD_SHAPES: dict[str, Callable[[dict], Verdict]] = {
    "user": _classify_user,
    "assistant": _classify_assistant,
    "system": _classify_system,
    "mode": _classify_mode,
    "ai-title": _classify_ai_title,
}
