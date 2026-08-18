"""Incremental deterministic state producer for companion panes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.adapters.codex import CodexAdapter
from palaver.ingest.cursors import Cursor
from palaver.observer.signals import Liveness, Signals, Tri, apply_liveness, derive_status
from palaver.summary import Provenance, SummaryReducer, SummarySnapshot
from palaver.ui.companion import CompanionPair, ReadMetadata
from palaver.ui.companion_state import MAX_ITEMS, CompanionState, JoinState, atomic_write_state
from palaver.ui.pane_join import join_pane, observe_liveness, read_process_table

REFRESH_SECONDS = 1.0
JOIN_SECONDS = 5.0
HEARTBEAT_SECONDS = 3.0


def _no_status(_message: str) -> None:
    pass


@dataclass
class _Runtime:
    source: str
    session_key: str
    store_path: Path
    pid: int
    reducer: SummaryReducer
    adapter: Any
    cursor: Cursor
    malformed_records: int = 0
    last_advance: datetime | None = None
    next_join: float = 0.0
    last_semantic: tuple[object, ...] | None = None
    last_write: float = 0.0
    liveness: Liveness = Liveness(Tri.UNKNOWN, Tri.UNKNOWN)


def _adapter(source: str, store_path: Path):
    if source == "claude-code":
        return ClaudeCodeAdapter(store_path.parent.parent)
    if source == "codex":
        return CodexAdapter(store_path.parents[3])
    raise ValueError(f"unsupported companion source: {source}")


def _semantic(state: CompanionState) -> tuple[object, ...]:
    return tuple(
        getattr(state, field.name) for field in fields(state) if field.name != "producer_updated_at"
    )


def _state(snapshot: SummarySnapshot, *, project: str, status: str, now: float) -> CompanionState:
    detail = snapshot.unknown_reasons[-1] if snapshot.unknown_reasons else None
    return CompanionState(
        producer_updated_at=now,
        project=project,
        source=snapshot.source,
        status=status,
        join_state=JoinState.JOINED,
        request=snapshot.request.text,
        command_result=snapshot.command_result.text,
        detail=detail,
        recent=tuple(item.text for item in snapshot.recent),
        tasks=tuple(f"{item.status}: {item.text}" for item in snapshot.tasks.items)[-MAX_ITEMS:],
        questions=tuple(item.text for item in snapshot.questions.items if item.text)[-MAX_ITEMS:],
    )


def signals_from_snapshot(snapshot: SummarySnapshot) -> Signals:
    """Translate source-neutral reducer structure into canonical status signals."""
    if snapshot.source_integrity is Provenance.UNKNOWN:
        parsed = Tri.FALSE
    else:
        parsed = Tri.TRUE
    turn = snapshot.turn.text
    if snapshot.turn.provenance is Provenance.UNKNOWN or turn is None:
        ended = Tri.UNKNOWN
    elif turn in {"Turn returned to human", "Turn aborted", "Awaiting explicit answer"}:
        ended = Tri.TRUE
    elif turn == "Agent turn open":
        ended = Tri.FALSE
    else:
        ended = Tri.UNKNOWN
    if snapshot.command_result.provenance is Provenance.UNKNOWN:
        error = Tri.UNKNOWN
    else:
        error = Tri.TRUE if snapshot.command_result.text else Tri.FALSE
    return Signals(
        source_readable=Tri.TRUE,
        signal_records_parsed=parsed,
        unresolved_tool_error=error,
        agent_turn_ended=ended,
    )


class CompanionUpdater:
    """Tail paired transcripts incrementally and publish bounded snapshots."""

    def __init__(
        self,
        *,
        read_metadata: ReadMetadata,
        clock=time.monotonic,
        wall_clock=time.time,
        joiner=join_pane,
        table_reader=read_process_table,
        state_writer=atomic_write_state,
        on_status=_no_status,
    ) -> None:
        self._read_metadata = read_metadata
        self._clock = clock
        self._wall_clock = wall_clock
        self._join = joiner
        self._read_table = table_reader
        self._write = state_writer
        self._on_status = on_status
        self._runtimes: dict[str, _Runtime] = {}
        self._next_join: dict[str, float] = {}
        self._unjoined: dict[str, tuple[CompanionState, float]] = {}

    async def refresh_once(self, app: Any, pairs: dict[str, CompanionPair]) -> int:
        """Refresh every pair once; one failed pane never stops the others."""
        active = set(pairs)
        for stale in set(self._runtimes) - active:
            self._runtimes.pop(stale, None)
            self._next_join.pop(stale, None)
            self._unjoined.pop(stale, None)
        table = None
        written = 0
        for agent_id, pair in pairs.items():
            try:
                agent = app.get_session_by_id(agent_id)
                if agent is None:
                    continue
                tab = next(
                    (
                        tab
                        for window in app.terminal_windows
                        for tab in window.tabs
                        if agent in tab.sessions
                    ),
                    None,
                )
                if tab is None:
                    continue
                metadata = await self._read_metadata(agent, tab)
                if metadata is None:
                    continue
                runtime = self._runtimes.get(agent_id)
                now_mono = self._clock()
                if runtime is None and now_mono < self._next_join.get(agent_id, 0.0):
                    cached = self._unjoined.get(agent_id)
                    if cached is not None and now_mono - cached[1] >= HEARTBEAT_SECONDS:
                        previous, _ = cached
                        state = CompanionState(
                            **{
                                **previous.__dict__,
                                "producer_updated_at": self._wall_clock(),
                            }
                        )
                        await asyncio.to_thread(self._write, pair.state_path, state)
                        self._unjoined[agent_id] = (state, now_mono)
                        written += 1
                    continue
                if runtime is None or now_mono >= runtime.next_join:
                    self._on_status("refreshing companion joins and process liveness")
                    table = await asyncio.to_thread(self._read_table) if table is None else table
                    joined = await asyncio.to_thread(self._join, metadata.pane, table=table)
                    if joined is None or joined.session_key is None or joined.store_path is None:
                        self._runtimes.pop(agent_id, None)
                        state = CompanionState(
                            producer_updated_at=self._wall_clock(),
                            project=Path(metadata.pane.path or "unknown").name or "unknown",
                            source=joined.source if joined is not None else "unknown",
                            status="UNKNOWN",
                            join_state=JoinState.UNJOINED,
                            detail="Waiting for an exact session join",
                        )
                        await asyncio.to_thread(self._write, pair.state_path, state)
                        self._next_join[agent_id] = now_mono + JOIN_SECONDS
                        self._unjoined[agent_id] = (state, now_mono)
                        written += 1
                        continue
                    identity = (joined.source, joined.session_key, joined.store_path)
                    if (
                        runtime is None
                        or (runtime.source, runtime.session_key, runtime.store_path) != identity
                    ):
                        runtime = _Runtime(
                            joined.source,
                            joined.session_key,
                            joined.store_path,
                            joined.pid,
                            SummaryReducer(joined.source, joined.session_key),
                            _adapter(joined.source, joined.store_path),
                            Cursor(offset=0),
                        )
                        self._runtimes[agent_id] = runtime
                        self._next_join.pop(agent_id, None)
                        self._unjoined.pop(agent_id, None)
                    runtime.pid = joined.pid
                    runtime.next_join = now_mono + JOIN_SECONDS
                    runtime.liveness = await asyncio.to_thread(
                        observe_liveness,
                        runtime.pid,
                        last_advance=runtime.last_advance,
                        now=datetime.now(timezone.utc),
                    )

                prior_offset = runtime.cursor.offset
                self._on_status(f"tailing companion transcript for {agent_id}")
                tail = await asyncio.to_thread(
                    runtime.adapter.tail, runtime.store_path, runtime.cursor
                )
                replace = tail.cursor.offset < prior_offset
                if replace:
                    runtime.malformed_records = 0
                runtime.cursor = tail.cursor
                runtime.malformed_records += tail.malformed_records
                snapshot = await asyncio.to_thread(
                    runtime.reducer.feed,
                    tail.events,
                    malformed_records=tail.malformed_records,
                    replace=replace,
                )
                if tail.cursor.offset != prior_offset:
                    runtime.last_advance = datetime.now(timezone.utc)
                    # The process half stays on the slower join cadence, but
                    # bytes observed in this tick are immediate positive
                    # evidence that the cursor advanced. Keeping a cached
                    # FALSE here would transiently refine a newly-ended turn
                    # to IDLE until the next process probe.
                    runtime.liveness = Liveness(
                        process_alive=runtime.liveness.process_alive,
                        cursor_advanced_recently=Tri.TRUE,
                    )
                signals = signals_from_snapshot(snapshot)
                status = derive_status(signals)
                status = apply_liveness(status, runtime.liveness)
                if snapshot.source_integrity is Provenance.UNKNOWN:
                    status = type(status).UNKNOWN
                state = _state(
                    snapshot,
                    project=Path(metadata.pane.path or "unknown").name or "unknown",
                    status=status.name,
                    now=self._wall_clock(),
                )
                semantic = _semantic(state)
                if (
                    semantic != runtime.last_semantic
                    or now_mono - runtime.last_write >= HEARTBEAT_SECONDS
                ):
                    await asyncio.to_thread(self._write, pair.state_path, state)
                    runtime.last_semantic = semantic
                    runtime.last_write = now_mono
                    written += 1
            except Exception as exc:
                self._on_status(f"companion update failed for {agent_id}: {type(exc).__name__}")
        return written

    async def run(self, app: Any, controller: Any, *, limit: int | None = None) -> int:
        """Refresh on a one-second cadence, reporting each wait through ``on_status``."""
        ticks = 0
        while limit is None or ticks < limit:
            await self.refresh_once(app, dict(controller.pairs))
            ticks += 1
            if limit is None or ticks < limit:
                self._on_status(f"companion refresh sleeping {REFRESH_SECONDS:.0f}s")
                await asyncio.sleep(REFRESH_SECONDS)
        return ticks


__all__ = ["CompanionUpdater"]
