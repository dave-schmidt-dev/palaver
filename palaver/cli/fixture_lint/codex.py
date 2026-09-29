"""
Codex's record shapes (task 7.1, authored from docs/research.md §2): structural
constants, checkers, classify_* functions, and CODEX_RECORD_SHAPES.
"""

from __future__ import annotations

import re
from typing import Callable

from .checks import (
    ACCEPTED,
    Verdict,
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
    RULE_UNKNOWN_SUBTYPE,
    SESSION_ID,
)

# --- Codex -------------------------------------------------------------------
#
# Shapes below are authored from `docs/research.md` §2 (4,884 real rollout
# files, sampled), not from an adapter — task 7.1 (the Codex adapter) has not
# landed yet. Every Codex fixture record carries exactly `{"type", "payload"}`
# at the envelope level: no `timestamp` and no `ordinal`, even though every
# real rollout record carries a `timestamp`. That omission is deliberate and
# mirrors the Claude Code corpus's missing `uuid`/`cwd`/`version` keys — a
# record pasted from `~/.codex/sessions/` is rejected on its key set before
# its prose is ever read, and nothing this corpus's consumers read depends on
# the timestamp value itself.
#
# `response_item.payload.type` values other than `"message"` (`function_call`,
# `reasoning`, …) and the top-level `type` values `turn_context`, `world_state`,
# and `inter_agent_communication_metadata` are not modelled at all — not an
# oversight, a scope decision: task 7.1 names turn boundary, compaction,
# errors, channel, and identity as the signals that matter, and none of them
# reads those shapes. `payload.item.changes` (a `FileChange` map keyed by
# absolute file path) is excluded for the same reason and one more: a map
# keyed by real paths is exactly the kind of field a structural corpus must
# not carry, since the keys themselves would be free text wearing a key's
# clothing.

#: Codex's `session_meta.payload.cwd`. A real session records the actual
#: project working directory — an identifying path — so this requires a
#: `/tmp/fixture-*` shape rather than accepting any string.
CODEX_CWD = re.compile(r"^/tmp/fixture-[a-z0-9-]{1,40}$")

#: `response_item.payload.role`, and each `compacted.payload.replacement_
#: history[]` entry's role. `developer` is Codex's one reliably-harness
#: channel (`docs/research.md` §2: 817/817 sampled were harness content).
#: `user` and `assistant` carry the same ambiguity Claude Code's `isMeta` flag
#: resolves and Codex has no equivalent for — prefix heuristic only — which is
#: why the corpus needs a `role: "user"` record wearing an injected prefix.
CODEX_ROLES = frozenset({"user", "assistant", "developer"})

#: The one `response_item.payload.content[].type` variant this corpus models.
CODEX_CONTENT_BLOCK_TYPES = frozenset({"input_text", "output_text"})

#: `event_msg.payload.type` values this corpus classifies: the two
#: turn-boundary terminals, the error shape, and the compaction pair's second
#: half. Every other observed value (`task_started`, `user_message`,
#: `token_count`, `agent_message`, `item_completed`, `sub_agent_activity`,
#: `thread_settings_applied`, …) carries no signal task 7.1 names.
CODEX_EVENT_TYPES = frozenset({"task_complete", "turn_aborted", "error", "context_compacted"})

#: `event_msg.payload.reason` on a `turn_aborted` event. Observed value only.
CODEX_TURN_ABORTED_REASONS = frozenset({"interrupted"})

#: `event_msg.payload.codex_error_info` on an `error` event. Observed value
#: only (`docs/research.md` §2).
CODEX_ERROR_CODES = frozenset({"usage_limit_exceeded"})


# --- Codex classifiers ---------------------------------------------------


def _check_codex_content_block(block: object, where: str) -> Verdict | None:
    """Allowlist one content block of a Codex `message` payload."""
    if not isinstance(block, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    block_type = block.get("type")
    if block_type not in CODEX_CONTENT_BLOCK_TYPES:
        return _reject(
            RULE_UNKNOWN_CONTENT_BLOCK,
            f"{where} has block type {block_type!r}, which no Codex content shape declares",
        )
    verdict = _check_keys(block, frozenset({"type", "text"}), frozenset(), where)
    return verdict if verdict is not None else _check_text(block["text"], f"{where}.text")


def _check_codex_message(payload: dict, where: str) -> Verdict | None:
    """Allowlist a Codex `response_item` payload of type `"message"`.

    Mirrors `_check_message` for Claude Code: role literal, then each content
    block. Codex's other `response_item.payload.type` values (`function_call`,
    `reasoning`, …) are not modelled — see the module-level Codex note.
    """
    verdict = _check_keys(payload, frozenset({"type", "role", "content"}), frozenset(), where)
    if verdict is not None:
        return verdict
    verdict = _check_literal(payload["type"], frozenset({"message"}), f"{where}.type")
    if verdict is not None:
        return verdict
    verdict = _check_literal(payload["role"], CODEX_ROLES, f"{where}.role")
    if verdict is not None:
        return verdict
    content = payload["content"]
    if not isinstance(content, list) or not content:
        return _reject(RULE_BAD_VALUE, f"{where}.content must be a non-empty list of blocks")
    for index, block in enumerate(content):
        verdict = _check_codex_content_block(block, f"{where}.content[{index}]")
        if verdict is not None:
            return verdict
    return None


def _classify_codex_session_meta(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "payload"}), frozenset(), "record")
    if verdict is not None:
        return verdict
    payload = record["payload"]
    if not isinstance(payload, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.payload is not an object")
    verdict = _check_keys(
        payload,
        frozenset({"id", "session_id", "cwd"}),
        frozenset({"parent_thread_id"}),
        "record.payload",
    )
    if verdict is not None:
        return verdict
    for key in ("id", "session_id", "parent_thread_id"):
        if key not in payload:
            continue
        verdict = _check_pattern(payload[key], SESSION_ID, f"record.payload.{key}")
        if verdict is not None:
            return verdict
    verdict = _check_pattern(payload["cwd"], CODEX_CWD, "record.payload.cwd")
    return verdict if verdict is not None else ACCEPTED


def _classify_codex_response_item(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "payload"}), frozenset(), "record")
    if verdict is not None:
        return verdict
    payload = record["payload"]
    if not isinstance(payload, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.payload is not an object")
    verdict = _check_codex_message(payload, "record.payload")
    return verdict if verdict is not None else ACCEPTED


def _classify_codex_event_msg(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "payload"}), frozenset(), "record")
    if verdict is not None:
        return verdict
    payload = record["payload"]
    if not isinstance(payload, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.payload is not an object")
    event_type = payload.get("type")
    if event_type not in CODEX_EVENT_TYPES:
        return _reject(
            RULE_UNKNOWN_SUBTYPE,
            f"record.payload.type {str(event_type)[:40]!r} is not an event type this corpus "
            f"classifies ({sorted(CODEX_EVENT_TYPES)})",
        )
    if event_type == "task_complete":
        verdict = _check_keys(
            payload, frozenset({"type", "last_agent_message"}), frozenset(), "record.payload"
        )
        if verdict is not None:
            return verdict
        if payload["last_agent_message"] is not None:
            verdict = _check_text(
                payload["last_agent_message"], "record.payload.last_agent_message"
            )
            if verdict is not None:
                return verdict
        return ACCEPTED
    if event_type == "turn_aborted":
        verdict = _check_keys(payload, frozenset({"type", "reason"}), frozenset(), "record.payload")
        if verdict is not None:
            return verdict
        verdict = _check_literal(
            payload["reason"], CODEX_TURN_ABORTED_REASONS, "record.payload.reason"
        )
        return verdict if verdict is not None else ACCEPTED
    if event_type == "error":
        verdict = _check_keys(
            payload,
            frozenset({"type", "message", "codex_error_info"}),
            frozenset(),
            "record.payload",
        )
        if verdict is not None:
            return verdict
        verdict = _check_text(payload["message"], "record.payload.message")
        if verdict is not None:
            return verdict
        verdict = _check_literal(
            payload["codex_error_info"], CODEX_ERROR_CODES, "record.payload.codex_error_info"
        )
        return verdict if verdict is not None else ACCEPTED
    # context_compacted: the second half of the compaction pair, no extra keys.
    verdict = _check_keys(payload, frozenset({"type"}), frozenset(), "record.payload")
    return verdict if verdict is not None else ACCEPTED


def _classify_codex_compacted(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "payload"}), frozenset(), "record")
    if verdict is not None:
        return verdict
    payload = record["payload"]
    if not isinstance(payload, dict):
        return _reject(RULE_NOT_AN_OBJECT, "record.payload is not an object")
    verdict = _check_keys(
        payload, frozenset({"replacement_history"}), frozenset(), "record.payload"
    )
    if verdict is not None:
        return verdict
    history = payload["replacement_history"]
    if not isinstance(history, list) or not history:
        return _reject(
            RULE_BAD_VALUE, "record.payload.replacement_history must be a non-empty list"
        )
    for index, item in enumerate(history):
        where = f"record.payload.replacement_history[{index}]"
        if not isinstance(item, dict):
            return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
        verdict = _check_keys(item, frozenset({"role", "content"}), frozenset(), where)
        if verdict is not None:
            return verdict
        verdict = _check_literal(item["role"], CODEX_ROLES, f"{where}.role")
        if verdict is not None:
            return verdict
        content = item["content"]
        if not isinstance(content, list) or not content:
            return _reject(RULE_BAD_VALUE, f"{where}.content must be a non-empty list")
        for block_index, block in enumerate(content):
            verdict = _check_codex_content_block(block, f"{where}.content[{block_index}]")
            if verdict is not None:
                return verdict
    return ACCEPTED


#: Codex's shape allowlist. `turn_context`, `world_state`, and
#: `inter_agent_communication_metadata` are absent deliberately — see the
#: module-level Codex note.
CODEX_RECORD_SHAPES: dict[str, Callable[[dict], Verdict]] = {
    "session_meta": _classify_codex_session_meta,
    "response_item": _classify_codex_response_item,
    "event_msg": _classify_codex_event_msg,
    "compacted": _classify_codex_compacted,
}
