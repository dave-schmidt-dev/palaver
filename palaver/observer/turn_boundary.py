"""Structural derivation of `agent_turn_ended`, and the Phase 1 signal reading.

This is the producer side of `palaver.observer.signals`: it computes the
`Signals` value `derive_status()` consumes, from one session store and
nothing else. No model, no network, no writes (INV-2 — every read here goes
through `read_complete_records`, which opens the source `O_RDONLY`).

**The boundary is structural, not hook-derived.** Claude Code emits stop-hook
records only when the user has a Stop hook configured: 34 of 200 sampled
transcripts (17%). A boundary resting on them is undefined for five sessions
in six, so those records are demoted to corroboration (`_corroborate`) and can
never establish a boundary. The primary derivation is the role of the last
conversational record plus `tool_use`/`tool_result` pairing, which was
computable on 200 of 200 sampled transcripts.

**Role is read through INV-8's channel classification, never raw.** A
`type: "user"` record is not evidence a human typed anything: hook output,
skill preambles, command expansions, and system reminders all arrive with that
role, tagged by `isMeta` or by a known text prefix (`classify_channel`). A
harness-injected record is therefore *transparent* here — the walk looks
through it to the last record that carries a real conversational turn. Reading
it raw is what inverts the status of exactly the sessions that use hooks and
skills most: an assistant's final message followed by a hook injection would
read as "the human just spoke, the agent owes a reply" (`WORKING`) when in
fact the session is waiting on the human. Reporting `WORKING` for a session
that needs its human is the most costly error this system can make.

Transparency is one-directional and deliberately so. It can move an answer
from `WORKING` to `AWAITING_HUMAN` (say "look at this session" about one that
is actually busy) but never the reverse, so its failure mode costs attention
rather than silence.

The five cases the backwards walk resolves, in the order it tests them:

1. A `user` record carrying `tool_result` blocks — a tool outcome, not a
   human turn. Checked *structurally, before* classification: a tool result
   carries no text, so it would otherwise fall through the prefix table and
   classify as the human channel by accident.
2. A human-channel `user` record — the human spoke and the agent owes a
   reply, so the agent holds the turn.
3. An `assistant` record with an unresolved `tool_use` block naming a
   *human-blocking* tool (`HUMAN_BLOCKING_TOOL_NAMES`, e.g.
   `AskUserQuestion`) — the call never resolved, but the tool itself only
   resolves by a human answering it. An agent that stopped to ask is not
   busy; it put a prompt in front of its human and control is already back
   with them, so `ended = TRUE` despite the open call. Checked by *name*,
   read from `message.content[…].name` — not by whether a `tool_use` block
   merely exists, which is what let this case fall into the next one before.
4. An `assistant` record with an unresolved `tool_use` block naming any
   other tool — nothing conversational follows it, so the call is
   unresolved and the agent is mid-turn.
5. An `assistant` record with no `tool_use` block — every call resolved,
   nothing after it: control is back with the human.

Anything else (no conversational record at all, or a record that did not
decode) is `Tri.UNKNOWN`. `UNKNOWN` is a first-class answer here, not a
fallthrough default.

**The signal window.** Every signal this module reports is a claim about the
*current turn*, so the window it reads is the records from the last
human-channel `user` record to the end of the file. That bound is what keeps
one corrupt line early in a long session from pinning that session to
`UNKNOWN` for the rest of its life, and what keeps a tool error the human has
already answered from being reported as the session's current state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.ingest.adapters.base import Event, read_complete_records
from palaver.ingest.adapters.claude_code import (
    CHANNEL_INJECTED,
    MESSAGE_RECORD_TYPES,
    classify_channel,
)
from palaver.observer.signals import Signals, Tri

logger = logging.getLogger(__name__)

#: How recently the store must have been written for file mtime to corroborate
#: "the agent still holds the turn". Corroboration only — mtime never moves the
#: boundary itself (see `_corroborate`).
DEFAULT_ACTIVE_WITHIN = timedelta(minutes=5)

#: Tool names whose call can only ever be resolved by the human — the agent
#: has stopped and put a prompt in front of them, not started work it will
#: report back on. An unresolved call naming one of these means the turn
#: ended (`Tri.TRUE`) even though the `tool_use` block itself never got a
#: `tool_result`. Read from `message.content[…].name`, already present on
#: every `tool_use` block; no new evidence is needed to check it. Phase 1
#: reports this as `AWAITING_HUMAN` (there is no reachable `QUESTION` yet —
#: see `palaver.observer.signals`); Phase 3.6 may refine it further.
HUMAN_BLOCKING_TOOL_NAMES = frozenset({"AskUserQuestion"})

# Why the boundary came out the way it did. Reported rather than inferred by a
# caller, so `palaver diagnose --coverage` can show *which* structure carried
# each session and a regression shows up as a shift in that distribution.
BASIS_ASSISTANT_FINAL = "assistant_final"
BASIS_UNRESOLVED_TOOL_USE = "unresolved_tool_use"
BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE = "unresolved_human_blocking_tool_use"
BASIS_TOOL_RESULT_PENDING = "tool_result_pending"
BASIS_HUMAN_MESSAGE_PENDING = "human_message_pending"
BASIS_NO_CONVERSATIONAL_RECORD = "no_conversational_record"
BASIS_UNDECODABLE_RECORD = "undecodable_record"
BASIS_SOURCE_UNREADABLE = "source_unreadable"

# The two bases only `derive_signals_from_events` can report (task 7.3).
# Named apart from the record-derived bases above rather than folded into
# them, because they are a weaker reading: an adapter told us a turn closed,
# where the bases above say which record structure proved it.
BASIS_EVENT_TURN_BOUNDARY = "event_turn_boundary"
BASIS_EVENT_MESSAGE_PENDING = "event_message_pending"

#: Every basis `derive_turn_boundary` or `derive_signals_from_events` can
#: report. The coverage command reports the distribution over this tuple.
BASIS_NAMES: tuple[str, ...] = (
    BASIS_ASSISTANT_FINAL,
    BASIS_UNRESOLVED_TOOL_USE,
    BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE,
    BASIS_TOOL_RESULT_PENDING,
    BASIS_HUMAN_MESSAGE_PENDING,
    BASIS_NO_CONVERSATIONAL_RECORD,
    BASIS_UNDECODABLE_RECORD,
    BASIS_SOURCE_UNREADABLE,
    BASIS_EVENT_TURN_BOUNDARY,
    BASIS_EVENT_MESSAGE_PENDING,
)

# The shared event-kind vocabulary the non-Claude-Code adapters emit, and the
# only thing `derive_signals_from_events` knows about a source. Declared here
# rather than imported from either adapter so this module does not depend on
# `palaver.ingest.adapters.codex` or `.opencode`; `tests/test_turn_boundary.py`
# asserts all three spellings agree, so a rename in one adapter is a test
# failure rather than a silently unreachable branch.
KIND_MESSAGE = "message"
KIND_TURN_BOUNDARY = "turn_boundary"
KIND_ERROR = "error"


@dataclass(frozen=True)
class TurnBoundary:
    """One session's turn boundary, with the structure that established it.

    Attributes:
        ended: `TRUE` when the agent handed control back to the human,
            `FALSE` when it still holds the turn (including mid-`tool_use`),
            `UNKNOWN` when no record in the window could settle it. Feeds
            `Signals.agent_turn_ended` unchanged.
        basis: Which of `BASIS_NAMES` produced `ended`.
        corroboration: `TRUE` when an independent signal agrees with `ended`,
            `FALSE` when one disagrees, `UNKNOWN` when none was available.
            Never an input to `ended` — see `_corroborate`.
    """

    ended: Tri
    basis: str
    corroboration: Tri


@dataclass(frozen=True)
class SessionObservation:
    """The full Phase 1 reading of one session store.

    Attributes:
        signals: The deterministic signal set, ready for `derive_status()`.
        boundary: The turn boundary behind `signals.agent_turn_ended`, with
            its basis and corroboration retained for diagnostics.
    """

    signals: Signals
    boundary: TurnBoundary


def _content_blocks(record: dict) -> list:
    """Return a record's message content blocks, or an empty list."""
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    return content if isinstance(content, list) else []


def _blocks_of_type(record: dict, block_type: str) -> list[dict]:
    """Return every content block of `block_type` in `record`."""
    return [
        block
        for block in _content_blocks(record)
        if isinstance(block, dict) and block.get("type") == block_type
    ]


def _is_message_bearing(record: dict) -> bool:
    return record.get("type") in MESSAGE_RECORD_TYPES


def _is_human_turn(record: dict) -> bool:
    """Report whether `record` is a human-typed `user` record.

    The `tool_result` check comes first and is structural. A tool result is a
    `type: "user"` record with no text at all, so `classify_channel` — which
    matches `isMeta` and then a text-prefix table — would classify it as the
    human channel by accident rather than by evidence.
    """
    if record.get("type") != "user":
        return False
    if _blocks_of_type(record, "tool_result"):
        return False
    return classify_channel(record) != CHANNEL_INJECTED


def derive_turn_boundary(
    records: Sequence[dict | None],
    *,
    store_mtime: float | None = None,
    now: datetime | None = None,
    active_within: timedelta = DEFAULT_ACTIVE_WITHIN,
) -> TurnBoundary:
    """Derive the turn boundary from a session's records, newest evidence first.

    Args:
        records: Every complete record in the store, in file order. `None`
            marks a line that did not decode — an opaque hole, never skipped
            silently: a hole reached before the boundary settles yields
            `UNKNOWN`, because the record it replaced could have been the
            decisive one.
        store_mtime: The store's mtime (epoch seconds), for corroboration
            only. `None` when unavailable.
        now: Reference time for the mtime comparison. Defaults to the current
            UTC time; tests pass a fixed value.
        active_within: How recently the store must have been written for
            mtime to corroborate "the agent still holds the turn".

    Returns:
        A `TurnBoundary`. `ended` is derived only from record structure and
        channel classification; corroboration is reported alongside it and
        never folded into it.
    """
    ended = Tri.UNKNOWN
    basis = BASIS_NO_CONVERSATIONAL_RECORD
    boundary_index: int | None = None

    for index in range(len(records) - 1, -1, -1):
        record = records[index]

        if record is None:
            ended, basis, boundary_index = Tri.UNKNOWN, BASIS_UNDECODABLE_RECORD, index
            break

        if not _is_message_bearing(record):
            continue

        if record.get("type") == "user":
            if _blocks_of_type(record, "tool_result"):
                # A tool outcome came back; the agent consumes it and continues.
                ended, basis, boundary_index = Tri.FALSE, BASIS_TOOL_RESULT_PENDING, index
                break
            if _is_human_turn(record):
                # The human spoke last and the agent owes a reply.
                ended, basis, boundary_index = Tri.FALSE, BASIS_HUMAN_MESSAGE_PENDING, index
                break
            # Harness-injected: not a conversational turn. Look through it.
            continue

        tool_use_blocks = _blocks_of_type(record, "tool_use")
        if tool_use_blocks:
            if any(block.get("name") in HUMAN_BLOCKING_TOOL_NAMES for block in tool_use_blocks):
                # The call never resolved, but the tool itself only resolves
                # by the human answering it: the agent stopped and put a
                # prompt in front of them, so control is already back with
                # the human despite the open call.
                ended, basis, boundary_index = (
                    Tri.TRUE,
                    BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE,
                    index,
                )
            else:
                # Nothing conversational follows this call, so it is unresolved.
                ended, basis, boundary_index = Tri.FALSE, BASIS_UNRESOLVED_TOOL_USE, index
            break

        ended, basis, boundary_index = Tri.TRUE, BASIS_ASSISTANT_FINAL, index
        break

    corroboration = _corroborate(
        ended,
        records,
        boundary_index,
        store_mtime=store_mtime,
        now=now,
        active_within=active_within,
    )
    return TurnBoundary(ended=ended, basis=basis, corroboration=corroboration)


def _corroborate(
    ended: Tri,
    records: Sequence[dict | None],
    boundary_index: int | None,
    *,
    store_mtime: float | None,
    now: datetime | None,
    active_within: timedelta,
) -> Tri:
    """Weigh independent signals against a boundary that is already decided.

    This is the corroboration branch, and it is the only place in this module
    that reads either of the two things that cannot be depended on:

    * **Stop-hook records.** `system` records with subtype
      `stop_hook_summary` or `turn_duration` exist only when the observed user
      has a Stop hook configured — 34 of 200 sampled transcripts. A stop-hook
      record positioned after the boundary record says the turn ended; one
      before it says nothing. Either way it only ever *agrees or disagrees*
      with a boundary derived above; a disagreement is reported, never
      applied. A design that let it apply would be undefined for 83% of real
      sessions.
    * **Store mtime.** It can only ever agree. mtime lies upward — a `git
      pull`, a restore, or an editor touch bumps it with no session activity —
      and it lies downward in the one case that matters most: an agent blocked
      on a long tool call writes nothing for minutes and is indistinguishable
      by mtime from an idle session. So a quiet file never contradicts "the
      agent holds the turn"; that is exactly what the unresolved-`tool_use`
      signal is for.

    Args:
        ended: The structurally derived boundary.
        records: The same record sequence `derive_turn_boundary` walked.
        boundary_index: Index of the record that established `ended`, or
            `None` when nothing did.
        store_mtime: Store mtime in epoch seconds, or `None`.
        now: Reference time for the mtime comparison, or `None` to use the
            current UTC time.
        active_within: Recency threshold for the mtime signal.

    Returns:
        `TRUE` if at least one independent signal agrees and none disagree,
        `FALSE` if one disagrees, `UNKNOWN` if none was available.
    """
    corroborating_subtypes = frozenset({"stop_hook_summary", "turn_duration"})
    agrees = False

    if ended is not Tri.UNKNOWN and boundary_index is not None:
        hook_index = None
        for index, record in enumerate(records):
            if (
                isinstance(record, dict)
                and record.get("type") == "system"
                and record.get("subtype") in corroborating_subtypes
            ):
                hook_index = index
        if hook_index is not None and hook_index > boundary_index:
            # Only a hook record positioned after the boundary makes a claim
            # about *this* turn. One before it belongs to an earlier turn and
            # is not evidence in either direction — counting its absence as a
            # denial would report a disagreement every time a session ran a
            # second turn after the last hook fired.
            if ended is Tri.TRUE:
                agrees = True
            else:
                return Tri.FALSE

        if store_mtime is not None:
            reference = now if now is not None else datetime.now(timezone.utc)
            age = reference - datetime.fromtimestamp(store_mtime, tz=timezone.utc)
            if ended is Tri.FALSE and age <= active_within:
                agrees = True
            elif ended is Tri.TRUE and age > active_within:
                agrees = True

    return Tri.TRUE if agrees else Tri.UNKNOWN


def _window_start(records: Sequence[dict | None]) -> int:
    """Return the index where the current turn begins.

    The window is the records from the last human-channel `user` record to the
    end of the file: every signal this module reports is a claim about the
    current turn. A hole stops the scan, because the line it replaced could
    have been the human record that opened this turn.
    """
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        if record is None:
            return index
        if _is_human_turn(record):
            return index
    return 0


def _unresolved_tool_error(records: Sequence[dict | None], window_start: int) -> Tri:
    """Report whether the most recent tool outcome in the window is an error.

    Scans backwards to the first record carrying `tool_result` blocks: a later
    successful outcome supersedes an earlier failure, which is what makes this
    a claim about the session's latest outcome rather than about whether an
    error ever occurred. Reaching the start of the window with no tool outcome
    at all is `FALSE` (observed absence), not `UNKNOWN` — the window was read
    in full. A hole reached first is `UNKNOWN`, because it could have been a
    later outcome.
    """
    for index in range(len(records) - 1, window_start - 1, -1):
        record = records[index]
        if record is None:
            return Tri.UNKNOWN
        results = _blocks_of_type(record, "tool_result")
        if results:
            return Tri.TRUE if any(block.get("is_error") for block in results) else Tri.FALSE
    return Tri.FALSE


def derive_signals(
    records: Sequence[dict | None],
    *,
    store_mtime: float | None = None,
    now: datetime | None = None,
    active_within: timedelta = DEFAULT_ACTIVE_WITHIN,
) -> SessionObservation:
    """Compute the full Phase 1 signal set from an already-read record sequence.

    Args:
        records: Every complete record in the store, in file order, with
            `None` for a line that did not decode.
        store_mtime: Store mtime in epoch seconds, for corroboration only.
        now: Reference time for the mtime comparison.
        active_within: Recency threshold for the mtime corroboration signal.

    Returns:
        A `SessionObservation` whose `signals.source_readable` is `TRUE` — the
        records were read to get here. `observe_session` owns the `FALSE` case.
    """
    window_start = _window_start(records)
    parsed = Tri.TRUE
    if any(record is None for record in records[window_start:]):
        parsed = Tri.FALSE

    boundary = derive_turn_boundary(
        records,
        store_mtime=store_mtime,
        now=now,
        active_within=active_within,
    )
    signals = Signals(
        source_readable=Tri.TRUE,
        signal_records_parsed=parsed,
        unresolved_tool_error=_unresolved_tool_error(records, window_start),
        agent_turn_ended=boundary.ended,
    )
    return SessionObservation(signals=signals, boundary=boundary)


def derive_signals_from_events(
    events: Sequence[Event],
    *,
    parsed: Tri = Tri.UNKNOWN,
) -> SessionObservation:
    """Compute the signal set from a canonical `Event` stream (task 7.3).

    The derivation for every source that is not Claude Code. Claude Code
    keeps `derive_signals`, which reads record structure directly and is the
    measured path this project's coverage numbers were established against;
    this one reads only `Event.kind`, from the vocabulary Codex and OpenCode
    both emit. Two derivations rather than one is the deliberate answer to a
    real asymmetry: those adapters resolve the turn boundary themselves and
    publish it as a `turn_boundary` event, while Claude Code has no such
    record and its boundary must be derived from message structure. A single
    derivation over kinds would score Claude Code at 0% boundary coverage and
    gate its statuses to `UNKNOWN` — the best-covered source silenced by a
    uniformity that buys nothing.

    What the kinds support, and nothing beyond it:

    * **`agent_turn_ended`** is `TRUE` when the last message-bearing event is
      a `turn_boundary`, `FALSE` when it is a `message`, `UNKNOWN` when the
      stream holds neither. `FALSE` covers both "the human spoke last" and
      "the agent is mid-turn": neither adapter's boundary event has fired, so
      control has not been handed back, and this module does not read either
      source's role field to say which of the two it is.
    * **`unresolved_tool_error`** is `TRUE` when an `error` event follows the
      last `turn_boundary`, and `FALSE` when a boundary came after every
      error — a closed turn resolves the errors inside it, the same reading
      `CodexAdapter.has_unresolved_trailing_tool_use` applies to pending
      calls. An error-free stream is `FALSE` (observed absence, matching
      `_unresolved_tool_error`); an empty one is `UNKNOWN`.

    Args:
        events: One session's canonical events, in source order.
        parsed: What the caller knows about decode completeness, since an
            `Event` stream cannot carry that itself — both adapters drop an
            undecodable record rather than marking it, so a hole is invisible
            downstream. The default is `Tri.UNKNOWN` ("nobody checked"),
            which is the fail-closed value: `derive_status()`'s rule 2 turns
            it into `UNKNOWN` rather than into a status derived from a view
            that may be missing its decisive record. A caller that decoded
            the source itself and counted the holes passes `TRUE`/`FALSE`.

    Returns:
        A `SessionObservation` whose `signals.source_readable` is `TRUE` —
        events were obtained, so the source was read. Corroboration is always
        `UNKNOWN`: the shared vocabulary carries no second, independent
        signal to weigh a boundary against, and inventing agreement out of
        the same event that produced the boundary would be corroboration in
        name only.
    """
    kinds = [event.kind for event in events]

    ended, basis = Tri.UNKNOWN, BASIS_NO_CONVERSATIONAL_RECORD
    for kind in reversed(kinds):
        if kind == KIND_TURN_BOUNDARY:
            ended, basis = Tri.TRUE, BASIS_EVENT_TURN_BOUNDARY
            break
        if kind == KIND_MESSAGE:
            ended, basis = Tri.FALSE, BASIS_EVENT_MESSAGE_PENDING
            break

    error = Tri.UNKNOWN if not kinds else Tri.FALSE
    for kind in reversed(kinds):
        if kind == KIND_TURN_BOUNDARY:
            error = Tri.FALSE
            break
        if kind == KIND_ERROR:
            error = Tri.TRUE
            break

    return SessionObservation(
        signals=Signals(
            source_readable=Tri.TRUE,
            signal_records_parsed=parsed,
            unresolved_tool_error=error,
            agent_turn_ended=ended,
        ),
        boundary=TurnBoundary(ended=ended, basis=basis, corroboration=Tri.UNKNOWN),
    )


def _decode(raw: bytes, path: Path) -> dict | None:
    """Decode one complete JSONL line, logging and reporting a hole on failure."""
    try:
        record = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        logger.warning("Unparseable record in %s: %r", path, raw[:200])
        return None
    if not isinstance(record, dict):
        logger.warning("Non-object record in %s: %r", path, raw[:200])
        return None
    return record


def observe_session(
    path: str | Path,
    *,
    now: datetime | None = None,
    active_within: timedelta = DEFAULT_ACTIVE_WITHIN,
) -> SessionObservation:
    """Read one session store and compute its Phase 1 signals.

    The store is read whole and read-only, through
    `palaver.ingest.adapters.base.read_complete_records` (INV-2). A trailing
    partial line from an in-flight write is withheld by that function and is
    not a decode failure here.

    Args:
        path: Path to the session store.
        now: Reference time for mtime corroboration. Defaults to the current
            UTC time; tests pass a fixed value.
        active_within: Recency threshold for the mtime corroboration signal.

    Returns:
        A `SessionObservation`. If the store could not be read at all,
        `source_readable` is `FALSE` and every other signal is `UNKNOWN` — a
        component that could not read a session reports nothing about it.
    """
    path = Path(path)
    try:
        store_mtime = path.stat().st_mtime
        raw_records, _ = read_complete_records(path, 0)
    except OSError as exc:
        logger.warning("Unreadable session store %s: %s", path, exc)
        return SessionObservation(
            signals=Signals(
                source_readable=Tri.FALSE,
                signal_records_parsed=Tri.UNKNOWN,
                unresolved_tool_error=Tri.UNKNOWN,
                agent_turn_ended=Tri.UNKNOWN,
            ),
            boundary=TurnBoundary(
                ended=Tri.UNKNOWN,
                basis=BASIS_SOURCE_UNREADABLE,
                corroboration=Tri.UNKNOWN,
            ),
        )

    records = [_decode(raw, path) for raw in raw_records]
    return derive_signals(records, store_mtime=store_mtime, now=now, active_within=active_within)
