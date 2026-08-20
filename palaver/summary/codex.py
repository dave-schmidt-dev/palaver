"""Deterministic summary reduction for Codex rollout canonical events."""

from __future__ import annotations

import json
from dataclasses import replace

from palaver.ingest.adapters.base import Event
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN
from palaver.ingest.adapters.codex import codex_role_class, message_text
from palaver.summary.model import (
    MAX_COLLECTION_ITEMS,
    Claim,
    CollectionClaim,
    Provenance,
    SummarySnapshot,
    TaskItem,
    append_recent,
    append_unknown_reason,
    fold_recent_result,
    sanitize_text,
)

SOURCE = "codex"


def _payload(record: dict) -> dict:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _arguments(payload: dict) -> dict | None:
    raw = payload.get("arguments")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _base_tool_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _task_snapshot(arguments: dict | None) -> CollectionClaim[TaskItem]:
    if arguments is None or not isinstance(arguments.get("plan"), list):
        return CollectionClaim.unknown("update_plan arguments are unsupported")
    if len(arguments["plan"]) > MAX_COLLECTION_ITEMS:
        return CollectionClaim.unknown("update_plan exceeds bounded task limit")
    tasks: list[TaskItem] = []
    for item in arguments["plan"]:
        if not isinstance(item, dict):
            return CollectionClaim.unknown("update_plan item is unsupported")
        text = sanitize_text(item.get("step"))
        status = sanitize_text(item.get("status"))
        if not text or not status:
            return CollectionClaim.unknown("update_plan item lacks step or status")
        tasks.append(TaskItem(text, status))
    return CollectionClaim(tuple(tasks), Provenance.EXACT, "update_plan")


def _question_claims(
    call_id: object, arguments: dict | None
) -> tuple[str | None, tuple[Claim, ...] | None]:
    if not isinstance(call_id, str) or not call_id or arguments is None:
        return None, None
    questions = arguments.get("questions")
    if not isinstance(questions, list):
        return call_id, None
    if len(questions) > MAX_COLLECTION_ITEMS:
        return call_id, None
    claims: list[Claim] = []
    for question in questions:
        if not isinstance(question, dict):
            return call_id, None
        claim = Claim.exact(question.get("question"), "request_user_input", call_id)
        if claim.provenance is Provenance.UNKNOWN:
            return call_id, None
        claims.append(claim)
    return call_id, tuple(claims)


def _tool_activity(name: str, payload: dict) -> str:
    raw = payload.get("arguments")
    rendered = sanitize_text(raw if isinstance(raw, str) else json.dumps(raw or {}, sort_keys=True))
    return f"Tool {name}" + (f": {rendered}" if rendered else "")


def _command_failure(payload: dict) -> Claim | None:
    if payload.get("type") != "exec_command_end":
        return None
    code = payload.get("exit_code")
    if isinstance(code, int) and not isinstance(code, bool) and code != 0:
        return Claim.structural(f"Command exited {code} this turn", "exec_command_end")
    status = payload.get("status")
    if status == "failed":
        return Claim.structural("Command status failed this turn", "exec_command_end")
    return None


def reduce_codex_events(
    session_key: str,
    events: tuple[Event, ...],
    *,
    initial: SummarySnapshot | None = None,
) -> SummarySnapshot:
    """Reduce Codex events while keeping turn failure distinct from agent failure."""
    snapshot = initial or SummarySnapshot(
        source=SOURCE,
        session_key=session_key,
        command_result=Claim(None, Provenance.STRUCTURAL, "event_stream"),
    )
    pending: dict[str, tuple[Claim, ...]] = {}
    for claim in snapshot.questions.items:
        if claim.evidence_id is not None:
            pending.setdefault(claim.evidence_id, ())
            pending[claim.evidence_id] = (*pending[claim.evidence_id], claim)
    questions_unknown = snapshot.questions.provenance is Provenance.UNKNOWN

    for event in events:
        record = event.payload
        if not isinstance(record, dict):
            snapshot = snapshot.with_unknown("Codex event payload is unsupported")
            continue
        payload = _payload(record)
        record_type = record.get("type")
        payload_type = payload.get("type")

        if event.kind == "compaction":
            compaction = Claim.structural("Context compacted", "compaction")
            recent = snapshot.recent
            if not recent or recent[-1].evidence_kind != "compaction":
                recent = append_recent(recent, compaction.text, Provenance.STRUCTURAL, "compaction")
            snapshot = replace(snapshot, compaction=compaction, recent=recent)
            continue

        if record_type == "response_item" and payload_type == "message":
            text = message_text(record)
            role = payload.get("role")
            if role == "assistant":
                snapshot = replace(
                    snapshot,
                    recent=append_recent(
                        snapshot.recent, f"Agent: {text}", Provenance.EXACT, "agent_message"
                    ),
                )
            elif codex_role_class(record) == CHANNEL_HUMAN:
                snapshot = replace(
                    snapshot,
                    request=Claim.exact(text, "human_message"),
                    recent=append_recent(
                        snapshot.recent, f"Human: {text}", Provenance.EXACT, "human_message"
                    ),
                    turn=Claim.structural("Agent turn open", "human_message"),
                )
            continue

        if record_type == "response_item" and payload_type == "function_call":
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                snapshot = snapshot.with_unknown("Codex tool call has no supported name")
                continue
            call_id = payload.get("call_id") if isinstance(payload.get("call_id"), str) else None
            arguments = _arguments(payload)
            snapshot = replace(
                snapshot,
                recent=append_recent(
                    snapshot.recent,
                    _tool_activity(name, payload),
                    Provenance.EXACT,
                    "function_call",
                    call_id,
                ),
                turn=Claim.structural("Agent turn open", "function_call"),
            )
            base_name = _base_tool_name(name)
            if base_name == "update_plan":
                snapshot = replace(snapshot, tasks=_task_snapshot(arguments))
            elif base_name == "request_user_input":
                question_id, claims = _question_claims(call_id, arguments)
                if question_id is None or claims is None:
                    questions_unknown = True
                else:
                    open_count = sum(len(value) for value in pending.values())
                    if open_count + len(claims) > MAX_COLLECTION_ITEMS:
                        questions_unknown = True
                    else:
                        pending[question_id] = claims
                        snapshot = replace(
                            snapshot,
                            turn=Claim.structural("Awaiting explicit answer", "request_user_input"),
                        )
            continue

        if record_type == "response_item" and payload_type == "function_call_output":
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                pending.pop(call_id, None)
            output = payload.get("output")
            snapshot = replace(
                snapshot,
                recent=fold_recent_result(
                    snapshot.recent,
                    output or "",
                    "function_call_output",
                    call_id if isinstance(call_id, str) else None,
                ),
                turn=Claim.structural("Agent turn open", "function_call_output"),
            )
            continue

        if event.kind == "turn_boundary":
            boundary = (
                "Turn aborted" if payload_type == "turn_aborted" else "Turn returned to human"
            )
            pending.clear()
            questions_unknown = False
            snapshot = replace(
                snapshot,
                recent=append_recent(
                    snapshot.recent, boundary, Provenance.STRUCTURAL, "turn_boundary"
                ),
                command_result=Claim(None, Provenance.STRUCTURAL, "turn_boundary"),
                turn=Claim.structural(boundary, "turn_boundary"),
            )
            continue

        if event.kind == "error":
            failure = _command_failure(payload)
            if (
                failure is None
                and payload_type == "patch_apply_end"
                and payload.get("success") is False
            ):
                failure = Claim.structural("Patch failed this turn", "patch_apply_end")
            if failure is None:
                detail = payload.get("message") or payload.get("codex_error_info")
                failure = (
                    Claim.exact(detail, "error")
                    if detail
                    else Claim.structural("Source error this turn", "error")
                )
            snapshot = replace(
                snapshot,
                command_result=failure,
                recent=append_recent(
                    snapshot.recent,
                    failure.text,
                    failure.provenance,
                    failure.evidence_kind or "error",
                ),
            )
            continue

        # Known bookkeeping is intentionally silent. An unrecognized event is
        # retained as an unknown structural observation, never interpreted.
        if event.kind == "unknown":
            snapshot = replace(
                snapshot,
                recent=append_recent(
                    snapshot.recent,
                    "Unsupported source event",
                    Provenance.UNKNOWN,
                    "unknown",
                ),
                unknown_reasons=append_unknown_reason(
                    snapshot.unknown_reasons, "unsupported Codex event"
                ),
            )

    if questions_unknown:
        questions = CollectionClaim.unknown("request_user_input structure is unsupported")
    else:
        questions = CollectionClaim(
            tuple(claim for claims in pending.values() for claim in claims),
            Provenance.EXACT,
            "request_user_input",
        )
    snapshot = replace(snapshot, questions=questions)
    if snapshot.source_integrity is Provenance.UNKNOWN:
        reason = snapshot.unknown_reasons[-1] if snapshot.unknown_reasons else "unknown source"
        return snapshot.with_unknown(reason)
    return snapshot


__all__ = ["reduce_codex_events"]
