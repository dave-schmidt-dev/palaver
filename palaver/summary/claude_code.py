"""Deterministic summary reduction for Claude Code canonical events."""

from __future__ import annotations

import json
from dataclasses import replace

from palaver.ingest.adapters.base import Event
from palaver.ingest.adapters.claude_code import CHANNEL_INJECTED, classify_channel
from palaver.summary.model import (
    MAX_COLLECTION_ITEMS,
    Claim,
    CollectionClaim,
    Provenance,
    SummarySnapshot,
    TaskItem,
    append_recent,
    sanitize_text,
)

SOURCE = "claude-code"


def _blocks(record: dict) -> list[dict]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _text_blocks(record: dict) -> str:
    return "\n".join(
        str(block.get("text", "")) for block in _blocks(record) if block.get("type") == "text"
    )


def _result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return ""


def _task_snapshot(tool_input: object) -> CollectionClaim[TaskItem]:
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("todos"), list):
        return CollectionClaim.unknown("TodoWrite input is unsupported")
    if len(tool_input["todos"]) > MAX_COLLECTION_ITEMS:
        return CollectionClaim.unknown("TodoWrite exceeds bounded task limit")
    tasks: list[TaskItem] = []
    for item in tool_input["todos"]:
        if not isinstance(item, dict):
            return CollectionClaim.unknown("TodoWrite item is unsupported")
        text = sanitize_text(item.get("content"))
        status = sanitize_text(item.get("status"))
        if not text or not status:
            return CollectionClaim.unknown("TodoWrite item lacks content or status")
        tasks.append(TaskItem(text, status))
    return CollectionClaim(tuple(tasks), Provenance.EXACT, "TodoWrite")


def _question_claims(block: dict) -> tuple[str | None, tuple[Claim, ...] | None]:
    tool_id = block.get("id")
    tool_input = block.get("input")
    if not isinstance(tool_id, str) or not tool_id or not isinstance(tool_input, dict):
        return None, None
    questions = tool_input.get("questions")
    if not isinstance(questions, list):
        return tool_id, None
    if len(questions) > MAX_COLLECTION_ITEMS:
        return tool_id, None
    claims: list[Claim] = []
    for question in questions:
        if not isinstance(question, dict):
            return tool_id, None
        claim = Claim.exact(question.get("question"), "AskUserQuestion", tool_id)
        if claim.provenance is Provenance.UNKNOWN:
            return tool_id, None
        claims.append(claim)
    return tool_id, tuple(claims)


def _tool_activity(name: str, tool_input: object) -> str:
    if isinstance(tool_input, dict):
        if name == "Bash":
            detail = tool_input.get("command")
        elif name in {"Read", "Write", "Edit", "NotebookEdit"}:
            detail = tool_input.get("file_path")
        else:
            detail = json.dumps(tool_input, ensure_ascii=False, sort_keys=True)
    else:
        detail = ""
    detail_text = sanitize_text(detail)
    return f"Tool {name}" + (f": {detail_text}" if detail_text else "")


def reduce_claude_events(
    session_key: str,
    events: tuple[Event, ...],
    *,
    initial: SummarySnapshot | None = None,
) -> SummarySnapshot:
    """Reduce Claude Code events without interpreting assistant prose."""
    snapshot = initial or SummarySnapshot(source=SOURCE, session_key=session_key)
    pending: dict[str, tuple[Claim, ...]] = {}
    for claim in snapshot.questions.items:
        if claim.evidence_id is not None:
            pending.setdefault(claim.evidence_id, ())
            pending[claim.evidence_id] = (*pending[claim.evidence_id], claim)
    questions_unknown = snapshot.questions.provenance is Provenance.UNKNOWN

    for event in events:
        record = event.payload
        if not isinstance(record, dict):
            snapshot = snapshot.with_unknown("Claude event payload is unsupported")
            continue

        if event.kind == "compaction":
            compaction = Claim.structural("Context compacted", "compaction")
            recent = snapshot.recent
            if not recent or recent[-1].evidence_kind != "compaction":
                recent = append_recent(recent, compaction.text, Provenance.STRUCTURAL, "compaction")
            snapshot = replace(snapshot, compaction=compaction, recent=recent)
            continue

        record_type = record.get("type")
        if record_type not in {"user", "assistant"}:
            continue

        blocks = _blocks(record)
        if record_type == "user":
            result_blocks = [block for block in blocks if block.get("type") == "tool_result"]
            if result_blocks:
                for block in result_blocks:
                    tool_id = block.get("tool_use_id")
                    if isinstance(tool_id, str):
                        pending.pop(tool_id, None)
                    text = _result_text(block)
                    prefix = "Tool error" if block.get("is_error") else "Tool result"
                    snapshot = replace(
                        snapshot,
                        recent=append_recent(
                            snapshot.recent,
                            f"{prefix}: {text}",
                            Provenance.EXACT,
                            "tool_result",
                            tool_id if isinstance(tool_id, str) else None,
                        ),
                        turn=Claim.structural("Agent turn open", "tool_result"),
                    )
                continue

            if classify_channel(record) == CHANNEL_INJECTED:
                continue
            text = _text_blocks(record)
            request = Claim.exact(text, "human_message")
            snapshot = replace(
                snapshot,
                request=request,
                recent=append_recent(
                    snapshot.recent, f"Human: {text}", Provenance.EXACT, "human_message"
                ),
                turn=Claim.structural("Agent turn open", "human_message"),
            )
            continue

        assistant_text = _text_blocks(record)
        if assistant_text:
            snapshot = replace(
                snapshot,
                recent=append_recent(
                    snapshot.recent,
                    f"Agent: {assistant_text}",
                    Provenance.EXACT,
                    "agent_message",
                ),
            )

        tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
        for block in tool_blocks:
            name = block.get("name")
            if not isinstance(name, str) or not name:
                snapshot = snapshot.with_unknown("Claude tool call has no supported name")
                continue
            tool_id = block.get("id") if isinstance(block.get("id"), str) else None
            snapshot = replace(
                snapshot,
                recent=append_recent(
                    snapshot.recent,
                    _tool_activity(name, block.get("input")),
                    Provenance.EXACT,
                    "tool_use",
                    tool_id,
                ),
            )
            if name == "TodoWrite":
                snapshot = replace(snapshot, tasks=_task_snapshot(block.get("input")))
            elif name == "AskUserQuestion":
                question_id, claims = _question_claims(block)
                if question_id is None or claims is None:
                    questions_unknown = True
                else:
                    open_count = sum(len(value) for value in pending.values())
                    if open_count + len(claims) > MAX_COLLECTION_ITEMS:
                        questions_unknown = True
                    else:
                        pending[question_id] = claims

        if tool_blocks:
            waiting = any(block.get("name") == "AskUserQuestion" for block in tool_blocks)
            snapshot = replace(
                snapshot,
                turn=Claim.structural(
                    "Awaiting explicit answer" if waiting else "Agent turn open",
                    "tool_use",
                ),
            )
        elif assistant_text:
            snapshot = replace(
                snapshot,
                turn=Claim.structural("Turn returned to human", "assistant_final"),
            )

    if questions_unknown:
        questions = CollectionClaim.unknown("AskUserQuestion structure is unsupported")
    else:
        questions = CollectionClaim(
            tuple(claim for claims in pending.values() for claim in claims),
            Provenance.EXACT,
            "AskUserQuestion",
        )
    snapshot = replace(snapshot, questions=questions)
    if snapshot.source_integrity is Provenance.UNKNOWN:
        reason = snapshot.unknown_reasons[-1] if snapshot.unknown_reasons else "unknown source"
        return snapshot.with_unknown(reason)
    return snapshot


__all__ = ["reduce_claude_events"]
