"""`palaver fixture-lint`: the allowlist gate on the committed fixture corpus.

This is INV-9's second half — the git half — and it is the last automated gate
before a transcript-shaped file enters a public repository's history. A fixture
pushed to a public remote leaves the machine as surely as an HTTP POST does,
and unlike a POST it cannot be recalled, so the failure mode this module exists
to prevent is *silent acceptance*, not noisy rejection.

**It is an allowlist, in two layers, and both live behind `classify_record`.**

1. **Shape.** A record ships only if its `type` (and, for `system`, its
   `subtype`) names an entry in the table below, every key it carries is one
   that entry declares, and every structural value is a literal that entry
   permits. Unknown type, unknown subtype, unknown key, unknown content block,
   unknown tool name: rejected. There is no fallthrough branch and no
   "probably fine" case.

2. **Free text.** Every string that reaches a free-text position must be an
   exact member of `SYNTHESIZED_TEXT`, the corpus's phrasebook. The linter
   cannot verify authorship, so it does not try; it verifies membership in a
   set that a human wrote deliberately. Adding a sentence to the corpus
   therefore requires editing this module, which is the point — the gate is
   the edit, and the edit is visible in a diff.

Why not a denylist cross-grep against the real stores. A grep answers "does
this fixture contain a string I already know about", which requires reading the
real stores to build the query, misses everything paraphrased or truncated, and
fails *open* on anything it has not seen. Its false negatives are silent. An
allowlist fails closed: an unclassified record is a failure, and the reason is
named.

What this buys concretely: a record copied out of a real Claude Code transcript
carries `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, and `version`
keys that no shape here declares, and a `sessionId` that is a UUID rather than
the required `fixture-*`. It is rejected on the first of those, before its
prose is ever considered.

`SYSTEM_SUBTYPE_KINDS` is imported from the Claude Code adapter rather than
restated, so the set of `system` subtypes the corpus may contain is exactly the
set the adapter claims to understand. A subtype the adapter has never heard of
is, by construction, one nobody has classified.

**Three sources, one allowlist discipline.** `RECORD_SHAPES` (Claude Code),
`CODEX_RECORD_SHAPES`, and `OPENCODE_RECORD_SHAPES` are three separate tables,
one per source, because task 7.0 asked for per-source shape tables rather than
one table folding all three vocabularies together — the point being that a
reviewer can delete one source's table and watch only that source's corpus
fail (`tests/test_fixture_lint.py`'s
`test_codex_source_and_opencode_source_corpora_require_their_shape_tables`
does exactly that). `classify_record` dispatches by trying each table in turn
against the record's `type`; this is safe only because the three sources' top
level `type` vocabularies are disjoint today
(`test_source_shape_tables_are_pairwise_disjoint` pins that as an assertion,
not an assumption). Neither Codex nor OpenCode has an adapter yet (tasks 7.1
and 7.2), so unlike `SYSTEM_SUBTYPE_KINDS` there is nothing to import the
allowlisted sub-values from; `CODEX_EVENT_TYPES` and `OPENCODE_PART_TYPES`
below are authored from `docs/research.md` and are expected to become the
values those adapters import back out of this module, mirroring the existing
direction in reverse.

OpenCode has no natural JSONL representation — its real store is SQLite rows
in `message`/`part` with a JSON `data` column, not a line-delimited transcript
file. `opencode_message` / `opencode_part` (the `type` values in
`OPENCODE_RECORD_SHAPES`) are this module's own fixture-format invention: one
JSON object per row, wrapping the columns a future adapter reads. A binary
`.db` fixture would not be reviewable by a human before it reaches a public
remote, which is the exact thing INV-9's git clause exists to make possible.

Output follows the CLI's two-stream contract: the result (the report, with one
line per rejection) goes to stdout, per-file progress goes to stderr (INV-1).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from palaver.ingest.adapters.claude_code import SYSTEM_SUBTYPE_KINDS

NAME = "fixture-lint"
HELP = "check every committed fixture record against the sanitization allowlist"

# Rejection rules. Named constants rather than free-form strings because the
# tests assert *which* rule fired: a poisoned record that fails for the wrong
# reason is a test that proves nothing, and a rule name in the report is the
# difference between "the linter rejected it" and "the linter rejected it for
# the reason under test".
RULE_UNDECODABLE = "undecodable-record"
RULE_NOT_AN_OBJECT = "not-an-object"
RULE_UNKNOWN_RECORD_TYPE = "unknown-record-type"
RULE_UNKNOWN_SYSTEM_SUBTYPE = "unknown-system-subtype"
#: The Codex/OpenCode analogue of `RULE_UNKNOWN_SYSTEM_SUBTYPE`: a nested
#: discriminator (Codex's `event_msg.payload.type`, OpenCode's
#: `part.data.type`) inside an otherwise-recognized record shape, whose value
#: this corpus does not classify. One shared rule rather than one per source,
#: because it is the same dimension — "which sub-shape applies" — recurring in
#: a second and third source, not a new kind of failure.
RULE_UNKNOWN_SUBTYPE = "unknown-subtype"
RULE_MISSING_KEY = "missing-key"
RULE_UNEXPECTED_KEY = "unexpected-key"
RULE_UNKNOWN_CONTENT_BLOCK = "unknown-content-block"
RULE_UNKNOWN_TOOL = "unknown-tool"
RULE_BAD_IDENTIFIER = "bad-identifier"
RULE_BAD_VALUE = "bad-value"
RULE_UNALLOWLISTED_TEXT = "unallowlisted-text"
RULE_UNTERMINATED_FILE = "unterminated-file"

#: Every rule `classify_record` and `lint_tree` can report, for the report
#: legend and for the tests' "this rule exists" assertions.
RULE_NAMES: tuple[str, ...] = (
    RULE_UNDECODABLE,
    RULE_NOT_AN_OBJECT,
    RULE_UNKNOWN_RECORD_TYPE,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_UNKNOWN_SUBTYPE,
    RULE_MISSING_KEY,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_CONTENT_BLOCK,
    RULE_UNKNOWN_TOOL,
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNTERMINATED_FILE,
)

#: The corpus phrasebook: every free-text string any committed fixture may
#: contain, exactly. Written for the fixtures, about invented work, by a human
#: who was not looking at a real transcript while writing them. Membership is
#: checked by equality, not by pattern, because a pattern is a heuristic and a
#: heuristic that admits a real sentence fails silently.
#:
#: Adding an entry here is the deliberate act INV-9's git clause is about. It
#: should be rare, and it should be obvious in review that the new string is
#: invented.
SYNTHESIZED_TEXT = frozenset(
    {
        # Human-channel turns.
        "refactor the auth module",
        "run the test suite",
        "deploy status?",
        "widen the retry window",
        # Assistant replies.
        "the auth module is refactored",
        "the test suite is green",
        "the deploy finished",
        "the retry window is now thirty seconds",
        "should i also rename the helper?",
        # Tool results.
        "ok",
        "command not found",
        # `AskUserQuestion` input.
        "which database should the worker use?",
        "Database",
        "postgres",
        "sqlite",
        "the shared instance",
        "a file next to the worker",
        # Harness-written content.
        "earlier notes trimmed",
        "hook ran",
        "test suite run",
        "<command-name>/status</command-name>",
        # Codex: human-channel turns and assistant replies.
        "check the staging deploy status",
        "the staging deploy is healthy",
        # Codex: harness-written content on the one reliably-harness channel
        # (`role: "developer"`, `docs/research.md` §2).
        "you are operating inside a sandboxed fixture container with no network access",
        # Codex: `role: "user"` wearing an injected prefix. Codex has no
        # `isMeta` equivalent, so this is exactly the shape the prefix
        # heuristic in a future adapter (task 7.1) has to see in the corpus.
        "<environment_context>fixture sandbox: bash on linux</environment_context>",
        # Codex: a turn's final assistant message, and an error message.
        "the fixture worker finished the requested change",
        "the fixture worker is retrying after a transient timeout",
        # OpenCode: human-channel turns and assistant replies.
        "restart the worker queue",
        "the worker queue is restarted",
        # OpenCode: a tool-part error message (`state.error`).
        "fixture tool exited with a non-zero status",
        # OpenCode: a synthetic (harness-injected) text part attached to a
        # `role: "user"` message — the same channel-ambiguity lesson INV-8
        # names for Claude Code, reproduced at the *part* level.
        "session continuation: resuming after context compaction",
    }
)

#: A fixture's `sessionId`. Deliberately not "any UUID": a real Claude Code
#: session id *is* a UUID, so a pattern that admitted one would admit a record
#: pasted from a real store. Requiring a `fixture-` prefix makes provenance a
#: structural property of the value rather than a claim about it.
SESSION_ID = re.compile(r"^fixture-[a-z0-9-]{1,48}$")

#: A fixture's `tool_use_id` / `tool_use.id`. Same reasoning: real ids are
#: `toolu_…` opaque strings, and none of them match this.
TOOL_USE_ID = re.compile(r"^tu-[0-9]{1,3}$")

#: Keys permitted inside a `tool_use` block's `input` map. A plain identifier
#: carries no prose; anything else is free text wearing a key's clothing.
INPUT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")

#: Tool names a fixture may name. Small and explicit: a tool name is the one
#: piece of a `tool_use` block that could otherwise carry an unreviewed string.
TOOL_NAMES = frozenset({"Bash", "Read", "Edit", "AskUserQuestion"})

#: `mode` record values. Structural literals, so no phrasebook check applies.
MODE_VALUES = frozenset({"default", "plan", "acceptEdits"})

#: How deep a `tool_use` input map may nest before the linter stops walking.
#: `AskUserQuestion`'s real input reaches five levels (input → questions →
#: question → options → option → label), so this is that plus headroom, not an
#: arbitrary number.
MAX_INPUT_DEPTH = 8

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


@dataclass(frozen=True)
class Verdict:
    """The classifier's answer for one record.

    Attributes:
        ok: True when the record matched an allowlisted shape and every
            free-text payload it carried was in the phrasebook.
        rule: Which rule rejected it, from `RULE_NAMES`; empty when `ok`.
        detail: Human-readable specifics — which key, which value, which
            block. Never echoes more than a truncated prefix of an offending
            string, so a linter failure does not itself print a transcript.
    """

    ok: bool
    rule: str = ""
    detail: str = ""


#: The accepting verdict, as a singleton so a stub classifier in a test can
#: return exactly what the real one returns on success.
ACCEPTED = Verdict(ok=True)


def _reject(rule: str, detail: str) -> Verdict:
    return Verdict(ok=False, rule=rule, detail=detail)


def _check_keys(
    record: dict, required: frozenset[str], optional: frozenset[str], where: str
) -> Verdict | None:
    """Reject a mapping whose key set is not exactly what its shape declares."""
    keys = set(record)
    missing = required - keys
    if missing:
        return _reject(RULE_MISSING_KEY, f"{where} is missing {sorted(missing)}")
    unexpected = keys - required - optional
    if unexpected:
        return _reject(
            RULE_UNEXPECTED_KEY, f"{where} carries unexpected key(s) {sorted(unexpected)}"
        )
    return None


def _check_literal(value: object, allowed: frozenset[str], where: str) -> Verdict | None:
    """Reject a structural value that is not one of a fixed set of literals."""
    if not isinstance(value, str) or value not in allowed:
        return _reject(RULE_BAD_VALUE, f"{where} must be one of {sorted(allowed)}, got {value!r}")
    return None


def _check_pattern(value: object, pattern: re.Pattern[str], where: str) -> Verdict | None:
    """Reject an identifier that does not match its declared synthetic shape."""
    if not isinstance(value, str) or not pattern.match(value):
        return _reject(
            RULE_BAD_IDENTIFIER,
            f"{where} must match {pattern.pattern}, got {str(value)[:40]!r}",
        )
    return None


def _check_bool(value: object, where: str) -> Verdict | None:
    if not isinstance(value, bool):
        return _reject(RULE_BAD_VALUE, f"{where} must be a boolean, got {type(value).__name__}")
    return None


def _check_text(value: object, where: str) -> Verdict | None:
    """Reject any free-text payload that is not in the corpus phrasebook.

    This is the second allowlist layer. It is exact-match on purpose: a length
    bound, a character class, or a "looks synthetic" heuristic would each admit
    some real sentence, and every one of those admissions is silent.
    """
    if not isinstance(value, str):
        return _reject(RULE_BAD_VALUE, f"{where} must be a string, got {type(value).__name__}")
    if value not in SYNTHESIZED_TEXT:
        return _reject(
            RULE_UNALLOWLISTED_TEXT,
            f"{where} is not in the synthesized phrasebook: {value[:48]!r}",
        )
    return None


def _check_input_value(value: object, where: str, depth: int) -> Verdict | None:
    """Walk a `tool_use` input map, allowlisting every leaf it reaches.

    Booleans and nulls carry no prose and pass structurally. Strings go through
    the phrasebook. Numbers are *not* admitted: no fixture needs one, and every
    value type this function accepts is one somebody decided to accept.
    """
    if depth > MAX_INPUT_DEPTH:
        return _reject(RULE_BAD_VALUE, f"{where} nests deeper than {MAX_INPUT_DEPTH} levels")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return _check_text(value, where)
    if isinstance(value, list):
        for index, item in enumerate(value):
            verdict = _check_input_value(item, f"{where}[{index}]", depth + 1)
            if verdict is not None:
                return verdict
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not INPUT_KEY.match(key):
                return _reject(
                    RULE_BAD_IDENTIFIER,
                    f"{where} has a key that is not a plain identifier: {str(key)[:40]!r}",
                )
            verdict = _check_input_value(item, f"{where}.{key}", depth + 1)
            if verdict is not None:
                return verdict
        return None
    return _reject(RULE_BAD_VALUE, f"{where} has unsupported type {type(value).__name__}")


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


def classify_record(record: object) -> Verdict:
    """Classify one decoded fixture record against both allowlist layers.

    This is the single seam the whole gate rests on: shape *and* phrasebook are
    decided here, so stubbing it to accept everything must make the linter
    accept everything. A test that poisons a record and asserts a non-zero exit
    proves nothing unless stubbing this function flips that exit to zero.

    Args:
        record: A decoded JSONL record.

    Returns:
        `ACCEPTED`, or a `Verdict` naming the rule that rejected it.
    """
    if not isinstance(record, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"record is a {type(record).__name__}, not an object")
    record_type = record.get("type")
    shape = None
    if isinstance(record_type, str):
        for shapes in (RECORD_SHAPES, CODEX_RECORD_SHAPES, OPENCODE_RECORD_SHAPES):
            shape = shapes.get(record_type)
            if shape is not None:
                break
    if shape is None:
        allowed = sorted(
            set(RECORD_SHAPES) | set(CODEX_RECORD_SHAPES) | set(OPENCODE_RECORD_SHAPES)
        )
        return _reject(
            RULE_UNKNOWN_RECORD_TYPE,
            f"record type {str(record_type)[:40]!r} is not in the shape allowlist ({allowed})",
        )
    return shape(record)


@dataclass(frozen=True)
class Rejection:
    """One record the allowlist refused, located precisely enough to fix."""

    path: Path
    line: int
    rule: str
    detail: str


@dataclass(frozen=True)
class LintReport:
    """The outcome of one `fixture-lint` run.

    Attributes:
        root: Directory the corpus was read from.
        files: How many `.jsonl` fixtures were read.
        records: How many records were classified.
        rejections: Every record the allowlist refused, in file order.
    """

    root: Path
    files: int
    records: int
    rejections: tuple[Rejection, ...]


def lint_file(
    path: Path, *, classify: Callable[[object], Verdict] | None = None
) -> tuple[int, list[Rejection]]:
    """Classify every record in one fixture file.

    A file whose last byte is not a newline is itself a rejection: JSONL's
    record separator is the newline, and a trailing unterminated line is a
    record that a reader using `read_complete_records` would withhold — which
    would let it ride into git having never been classified.

    Args:
        path: The fixture file.
        classify: Classifier override; defaults to `classify_record`, resolved
            at call time so a test can substitute the module attribute.

    Returns:
        `(records_classified, rejections)`.
    """
    classify = classify_record if classify is None else classify
    raw = path.read_bytes()
    rejections: list[Rejection] = []
    if raw and not raw.endswith(b"\n"):
        rejections.append(
            Rejection(
                path=path,
                line=raw.count(b"\n") + 1,
                rule=RULE_UNTERMINATED_FILE,
                detail="file does not end in a newline, so its last record is unterminated",
            )
        )
    counted = 0
    for number, line in enumerate(raw.split(b"\n"), start=1):
        if number > raw.count(b"\n") and not line:
            continue  # the empty string after the final newline
        counted += 1
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            rejections.append(
                Rejection(path=path, line=number, rule=RULE_UNDECODABLE, detail=str(exc))
            )
            continue
        verdict = classify(record)
        if not verdict.ok:
            rejections.append(
                Rejection(path=path, line=number, rule=verdict.rule, detail=verdict.detail)
            )
    return counted, rejections


def lint_tree(
    root: Path,
    *,
    classify: Callable[[object], Verdict] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> LintReport:
    """Classify every record of every `.jsonl` fixture under `root`.

    Args:
        root: Corpus directory (or a single `.jsonl` file).
        classify: Classifier override; defaults to `classify_record`.
        on_status: Progress channel, called once per file before it is read
            (INV-1). Never writes to stdout.

    Returns:
        A `LintReport` over everything read.
    """
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    records = 0
    rejections: list[Rejection] = []
    for index, path in enumerate(paths, start=1):
        if on_status is not None:
            on_status(f"linting {index}/{len(paths)}: {path.name}")
        counted, found = lint_file(path, classify=classify)
        records += counted
        rejections.extend(found)
    return LintReport(root=root, files=len(paths), records=records, rejections=tuple(rejections))


def render_report(report: LintReport) -> str:
    """Render a `LintReport` as the command's stdout output."""
    lines = [
        "palaver fixture-lint",
        f"corpus: {report.root}",
        f"files: {report.files}",
        f"records: {report.records}",
        f"rejected: {len(report.rejections)}",
    ]
    if report.rejections:
        lines.append("")
        for rejection in report.rejections:
            lines.append(f"{rejection.path}:{rejection.line}: {rejection.rule}: {rejection.detail}")
        lines.extend(
            [
                "",
                "A rejected record is not classified, and an unclassified record does",
                "not ship: fix the fixture, or add its shape to the allowlist in",
                "palaver/cli/fixture_lint.py deliberately.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "every record matched an allowlisted shape and carried only",
                "phrasebook text.",
            ]
        )
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Register `fixture-lint`'s arguments on its subparser."""
    parser.add_argument(
        "path",
        type=Path,
        help="fixture corpus directory (or a single .jsonl fixture) to check",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver fixture-lint`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when every record classified, 1 when any record was rejected, and 2
        for a usage failure — a missing path or a corpus with no fixtures in
        it. The two non-zero codes are distinct deliberately: a test that
        asserts "the linter rejected my poisoned record" must be able to fail
        when what actually happened was that the path was wrong.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    root = Path(args.path)
    if not root.exists():
        print(f"palaver fixture-lint: no such path: {root}", file=sys.stderr)
        return 2

    report = lint_tree(root, on_status=on_status)
    if report.files == 0:
        print(f"palaver fixture-lint: no .jsonl fixtures under {root}", file=sys.stderr)
        return 2

    out.write(render_report(report))
    return 1 if report.rejections else 0
