"""JSONL session store to semantic turn stream (Task 3.1).

Raw Claude Code JSONL is dense with bytes the model never needs to see: full
tool-input JSON, multi-kilobyte tool results, `thinking` blocks, bookkeeping
records (`attachment`, `last-prompt`, `ai-title`, `mode`, `bridge-session`,
`summary`). This module reduces a session store to the turns that carry
meaning, reproduced here by `tests/test_normalize.py`'s synthesized-input test.

**Channel tags come from INV-8's classification, never from text sniffing.**
This is the one rule this module exists to enforce and the reason it is not
just a formatting pass. The spike's v1 normalizer rendered every `type:
"user"` record as `USER:`, and the model duly extracted a `frontend-design`
skill preamble as an explicit user decision — the quote was real, only the
attribution was wrong, so the fabrication check passed it. `classify_channel`
in `palaver.ingest.adapters.claude_code` is Phase 1's fix (task 1.4, gated by
task 1.8's `test_isMeta_record_classified_as_injected_channel` and its
positive control): it applies both of INV-8's signals — the structural
`isMeta` flag, then a fixed text-prefix table — and this module calls that
function rather than re-deriving any part of the decision. `CHANNEL_TAG`
below maps its two possible return values (`CHANNEL_HUMAN`, `CHANNEL_INJECTED`)
straight onto the two tags a reader (human or model) sees, so the tag on a
line is a direct rendering of the classifier's verdict, not an independent
guess that happens to agree with it most of the time.

Tool results and `system` records are never channel-classified at all —
`classify_channel` is only defined for `type: "user"` records, and a tool
result carries no channel because nothing was said, something merely
happened. Those render as `result>`/`SYSTEM(...)` lines instead, structurally
distinct from `HUMAN:`/`INJECTED:`/`AGENT:`, so nothing downstream can
mistake a tool outcome or a compaction banner for a channel-tagged turn.

Every read here goes through `read_complete_records` (in turn,
`open_source_readonly`) — this module never opens a source file itself
(INV-2), the same discipline `palaver.ingest.adapters.claude_code` documents
and `tests/test_invariants.py::test_adapters_route_every_read_through_the_chokepoint`
enforces for the adapters package.

This module has no tier concept and rejects nothing — it classifies and
formats. Task 3.3's quote-grounding gate is where an injected-channel quote
is refused as tier-1; that is a write-boundary decision this module has no
write boundary to make.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from palaver.ingest.adapters.base import read_complete_records
from palaver.ingest.adapters.claude_code import (
    CHANNEL_HUMAN,
    CHANNEL_INJECTED,
    SYSTEM_SUBTYPE_KINDS,
    classify_channel,
)
from palaver.ingest.adapters.codex import (
    CHANNEL_HUMAN as CODEX_CHANNEL_HUMAN,
)
from palaver.ingest.adapters.codex import (
    codex_role_class,
    message_text,
    order_records,
)

logger = logging.getLogger(__name__)

#: Characters kept from a single tool result before it is clipped. Tool
#: output is the single largest contributor to raw JSONL size (command logs,
#: file contents, grep dumps) and carries the least marginal meaning per
#: byte once a reader knows whether it succeeded, so it is capped hardest.
TOOL_RESULT_CHAR_CAP = 400

#: Characters kept from a single text block (a human/injected/agent turn, or
#: a system record's `content`/`summary`) before it is clipped.
TEXT_CHAR_CAP = 3000

#: Characters kept from a `Bash` tool call's `command` field. Long here-docs
#: and multi-line scripts are common and rarely need to be reproduced in
#: full for the transcript to stay legible.
BASH_COMMAND_CHAR_CAP = 160

#: Characters kept from a JSON-rendered tool-input map for tool names with no
#: dedicated summary (`_TOOL_SUMMARY_HANDLERS` below).
TOOL_INPUT_JSON_CHAR_CAP = 300

# Codex tool and error records can contain command output or serialized
# arguments. Keep those structural summaries bounded just like Claude tool
# summaries; semantic message text has the larger text cap below.
CODEX_STRUCTURAL_CHAR_CAP = 400

#: Renders a classified channel onto the tag a line actually carries. Built
#: from `classify_channel`'s own two return values rather than two
#: independently spelled string literals, so a tag on a line is traceable
#: back to the classifier's verdict.
CHANNEL_TAG: dict[str, str] = {
    CHANNEL_HUMAN: "HUMAN",
    CHANNEL_INJECTED: "INJECTED",
}

AGENT_TAG = "AGENT"

_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")

#: Tool names whose `input` gets a purpose-built one-line summary instead of
#: a generic JSON dump. Ported from the spike, which measured these as the
#: tool calls that dominate a real session's tool-use volume.
_FILE_TOOLS = frozenset({"Edit", "Write", "Read", "NotebookEdit"})
_PATTERN_TOOLS = frozenset({"Grep", "Glob"})
_STRUCTURED_TOOLS = frozenset({"TodoWrite", "Task", "Skill"})


def _clip(text: str, limit: int) -> str:
    """Collapse runs of 3+ newlines to one blank line, then cap length.

    Args:
        text: Text to clip. `None` is treated as empty.
        limit: Maximum characters kept before an elision marker is appended.

    Returns:
        `text`, unchanged if it already fits within `limit`; otherwise the
        first `limit` characters followed by `"\\n…[+N chars]"`, where `N` is
        exactly how many characters were dropped, so a reader always knows
        how much was cut rather than seeing a silently shortened line.
    """
    text = _MULTI_BLANK_LINE_RE.sub("\n\n", text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[+{len(text) - limit} chars]"


def _tool_summary(name: object, tool_input: dict) -> str:
    """One line describing a `tool_use` block — the call's shape, not its full payload."""
    name = name if isinstance(name, str) else "?"
    if name in _FILE_TOOLS:
        return f"{name}({tool_input.get('file_path', '?')})"
    if name == "Bash":
        return f"Bash: {_clip(str(tool_input.get('command', '')), BASH_COMMAND_CHAR_CAP)}"
    if name in _PATTERN_TOOLS:
        return f"{name}({tool_input.get('pattern', '?')})"
    if name in _STRUCTURED_TOOLS:
        rendered = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
        return f"{name}: {_clip(rendered, TOOL_INPUT_JSON_CHAR_CAP)}"
    rendered = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    return f"{name}({_clip(rendered, TOOL_INPUT_JSON_CHAR_CAP)})"


def _content_blocks(record: dict) -> list[dict]:
    """Return a `user`/`assistant` record's content as a list of blocks.

    Claude Code sometimes writes `message.content` as a bare string (a plain
    text turn with no blocks); that is normalized here to a single `text`
    block so callers only ever handle the list shape.
    """
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _split_user_content(record: dict) -> tuple[list[str], list[tuple[str, bool]]]:
    """Split a `user` record's content into (text blocks, tool-result blocks).

    Text blocks are what `classify_channel` classifies and what carries a
    `HUMAN`/`INJECTED` tag. Tool-result blocks carry no channel — a result is
    something that happened, not something said — and are returned
    separately as `(content, is_error)` pairs so the caller never has to
    consult `classify_channel` to render one.
    """
    texts: list[str] = []
    results: list[tuple[str, bool]] = []
    for block in _content_blocks(record):
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text", "")))
        elif block_type == "tool_result":
            content = block.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", "")) for item in content if isinstance(item, dict)
                )
            results.append((str(content or ""), bool(block.get("is_error"))))
    return texts, results


def _split_assistant_content(record: dict) -> tuple[list[str], list[tuple[object, dict]]]:
    """Split an `assistant` record's content into (text blocks, tool-use calls).

    `thinking` blocks are deliberately dropped: they are the model's private
    reasoning, never surfaced to the observer, the same choice the spike
    made and this module keeps.
    """
    texts: list[str] = []
    uses: list[tuple[object, dict]] = []
    for block in _content_blocks(record):
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_input = block.get("input")
            uses.append((block.get("name"), tool_input if isinstance(tool_input, dict) else {}))
        # "thinking" and any other block type: not surfaced.
    return texts, uses


def _ts_prefix(record: dict) -> str:
    """Render a record's `HH:MM:SS` timestamp prefix, or `""` if it has none.

    The committed fixture corpus deliberately omits `timestamp` (INV-9 —
    nothing that could distinguish a fixture from a real transcript), so a
    missing timestamp is the ordinary case for a fixture and must not crash
    or print a placeholder; a real Claude Code record always carries one.
    """
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str) and len(timestamp) >= 19:
        return f"[{timestamp[11:19]}] "
    return ""


def _render_user(record: dict, ts_prefix: str) -> list[str]:
    lines: list[str] = []
    texts, results = _split_user_content(record)
    if texts:
        tag = CHANNEL_TAG[classify_channel(record)]
        for text in texts:
            stripped = text.strip()
            if stripped:
                lines.append(f"{ts_prefix}{tag}: {_clip(stripped, TEXT_CHAR_CAP)}")
    for content, is_error in results:
        stripped = content.strip()
        if stripped:
            marker = "result[error]>" if is_error else "result>"
            lines.append(f"{ts_prefix}  {marker} {_clip(stripped, TOOL_RESULT_CHAR_CAP)}")
    return lines


def _render_assistant(record: dict, ts_prefix: str) -> list[str]:
    lines: list[str] = []
    texts, uses = _split_assistant_content(record)
    for text in texts:
        stripped = text.strip()
        if stripped:
            lines.append(f"{ts_prefix}{AGENT_TAG}: {_clip(stripped, TEXT_CHAR_CAP)}")
    for name, tool_input in uses:
        lines.append(f"{ts_prefix}  tool> {_tool_summary(name, tool_input)}")
    return lines


def _render_system(record: dict, ts_prefix: str) -> list[str]:
    """Render a `system` record by its adapter-recognized kind, never a channel tag.

    `SYSTEM_SUBTYPE_KINDS` is imported from the Claude Code adapter rather
    than restated, so the kind name a reader sees here is exactly the kind
    name the adapter's own event model assigns the same record — an
    unrecognized subtype falls back to `"system"`, matching
    `_events_for_record`'s fallback rather than inventing a second one.
    """
    kind = SYSTEM_SUBTYPE_KINDS.get(record.get("subtype"), "system")
    text = record.get("content") or record.get("summary") or ""
    stripped = str(text).strip()
    if stripped:
        return [f"{ts_prefix}SYSTEM({kind}): {_clip(stripped, TEXT_CHAR_CAP)}"]
    return [f"{ts_prefix}SYSTEM({kind})"]


def _render_record(record: dict) -> list[str]:
    """Render one decoded record to zero or more transcript lines.

    Every other top-level `type` Claude Code writes (`attachment`,
    `last-prompt`, `ai-title`, `mode`, `bridge-session`, `summary`, and any
    future bookkeeping type) renders to nothing — deliberately, so a new
    non-conversational record type introduced by a future Claude Code
    release does not need a matching update here to stay excluded, the same
    reasoning `claude_code.MESSAGE_RECORD_TYPES` documents.
    """
    record_type = record.get("type")
    ts_prefix = _ts_prefix(record)
    if record_type == "user":
        return _render_user(record, ts_prefix)
    if record_type == "assistant":
        return _render_assistant(record, ts_prefix)
    if record_type == "system":
        return _render_system(record, ts_prefix)
    return []


def _parse_record(raw: bytes, path: Path | str) -> dict | None:
    """Decode one complete JSONL line, logging and skipping on failure.

    Mirrors `claude_code._parse_record`'s contract (a corrupt record is
    logged at WARNING and skipped, never raised) without importing that
    module's private helper across a package boundary.
    """
    try:
        record = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        logger.warning("Unparseable JSONL record in %s: %r", path, raw[:200])
        return None
    if not isinstance(record, dict):
        logger.warning("Non-object JSONL record in %s: %r", path, raw[:200])
        return None
    return record


def normalize_records(records: Iterable[dict], *, source: str = "claude-code") -> str:
    """Render decoded JSONL records to a semantic turn transcript.

    Args:
        records: Decoded JSONL records, in file order. Each is independently
            classified and rendered; a record this module does not recognize
            (an unhandled `type`) contributes nothing rather than raising.
        source: Source renderer. ``claude-code`` preserves the original
            renderer; ``codex`` uses the bounded rollout renderer.

    Returns:
        The transcript, one rendered line per newline, ending in a single
        trailing newline. `""` if no record rendered any lines.
    """
    if source == "codex":
        return normalize_codex_records(records)
    if source != "claude-code":
        raise ValueError(f"unsupported normalization source: {source}")
    lines: list[str] = []
    for record in records:
        lines.extend(_render_record(record))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _codex_payload(record: dict) -> dict:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _codex_structural_summary(record: dict) -> str | None:
    """Render one recognized Codex tool/error record without its full payload."""
    payload = _codex_payload(record)
    record_type = record.get("type")
    payload_type = payload.get("type")
    if record_type == "response_item" and payload_type == "function_call":
        name = payload.get("name") or "?"
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            rendered = arguments
        elif arguments is None:
            rendered = ""
        else:
            rendered = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        return f"tool> {name}: {_clip(rendered, CODEX_STRUCTURAL_CHAR_CAP)}"
    if record_type == "response_item" and payload_type == "function_call_output":
        output = payload.get("output")
        return f"result> {_clip(str(output or ''), CODEX_STRUCTURAL_CHAR_CAP)}"
    if record_type == "event_msg" and payload.get("type") == "error":
        detail = payload.get("message") or payload.get("codex_error_info") or "error"
        return f"error> {_clip(str(detail), CODEX_STRUCTURAL_CHAR_CAP)}"
    if record_type == "event_msg" and payload.get("type") in {
        "exec_command_end",
        "patch_apply_end",
    }:
        # Only failed command/patch outcomes are surfaced. Successful tool
        # completions carry no semantic text and would make the stream noisy.
        failed = payload.get("status") == "failed" or payload.get("success") is False
        exit_code = payload.get("exit_code")
        failed = failed or (
            isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0
        )
        if failed:
            detail = payload.get("message") or payload.get("status") or "tool failed"
            return f"error> {_clip(str(detail), CODEX_STRUCTURAL_CHAR_CAP)}"
    return None


def _render_codex_record(record: dict) -> list[str]:
    """Render a bounded semantic subset of one Codex rollout record."""
    if record.get("type") == "response_item" and _codex_payload(record).get("type") == "message":
        text = message_text(record).strip()
        if not text:
            return []
        role = _codex_payload(record).get("role")
        if role == "assistant":
            tag = AGENT_TAG
        elif codex_role_class(record) == CODEX_CHANNEL_HUMAN:
            tag = "HUMAN"
        else:
            tag = "INJECTED"
        return [f"{tag}: {_clip(text, TEXT_CHAR_CAP)}"]
    summary = _codex_structural_summary(record)
    return [] if summary is None else [summary]


def normalize_codex_records(records: Iterable[dict]) -> str:
    """Render Codex rollout records to a bounded semantic transcript.

    Reasoning, bookkeeping, unknown payloads, and successful tool events are
    intentionally omitted. Channel labels use Codex's measured classifier;
    they are not inferred from message prose here.
    """
    lines: list[str] = []
    for record in order_records(list(records)):
        lines.extend(_render_codex_record(record))
    return "" if not lines else "\n".join(lines) + "\n"


def normalize_codex_path(path: str | Path) -> str:
    """Read and normalize a Codex rollout through the read-only chokepoint."""
    raw_records, _ = read_complete_records(path, 0)
    records = []
    for raw in raw_records:
        record = _parse_record(raw, path)
        if record is not None:
            records.append(record)
    return normalize_codex_records(records)


def normalize_path(path: str | Path, *, source: str = "claude-code") -> str:
    """Read one JSONL session store read-only and normalize it to a transcript.

    Reads via `read_complete_records` (in turn, `open_source_readonly`) —
    this module never opens `path` itself (INV-2).

    Args:
        path: Path to a JSONL session store.

    Returns:
        The normalized transcript; see `normalize_records`.
    """
    if source == "codex":
        return normalize_codex_path(path)
    if source != "claude-code":
        raise ValueError(f"unsupported normalization source: {source}")
    raw_records, _ = read_complete_records(path, 0)
    records = []
    for raw in raw_records:
        record = _parse_record(raw, path)
        if record is not None:
            records.append(record)
    return normalize_records(records)
