"""Headless integration tests for incremental companion state updates."""

from __future__ import annotations

import asyncio
import threading
import types
from pathlib import Path

from palaver.ingest.adapters.base import Event, TailResult
from palaver.ingest.cursors import Cursor
from palaver.observer.signals import Liveness, Status, Tri
from palaver.summary import Claim, Provenance, SummarySnapshot
from palaver.ui import companion_update
from palaver.ui.companion import CompanionPair, SessionMetadata
from palaver.ui.companion_state import JoinState
from palaver.ui.companion_update import CompanionUpdater, signals_from_snapshot
from palaver.ui.pane_join import PaneJoin, PaneVariables


class Adapter:
    def __init__(self, batches):
        self.batches = list(batches)
        self.offsets = []

    def tail(self, _path, cursor):
        self.offsets.append(cursor.offset)
        return self.batches.pop(0)


def _event(text="work"):
    return Event(
        "session",
        "message",
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    )


def _ended_event():
    return Event(
        "session",
        "turn_boundary",
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    )


def _fixture(tmp_path, adapter, *, malformed=0):
    agent = types.SimpleNamespace(session_id="agent")
    tab = types.SimpleNamespace(sessions=[agent])
    app = types.SimpleNamespace(
        terminal_windows=[types.SimpleNamespace(tabs=[tab])],
        get_session_by_id=lambda session_id: agent if session_id == "agent" else None,
    )
    metadata = SessionMetadata(
        session=agent,
        tab=tab,
        pane=PaneVariables("agent", 10, "codex", str(tmp_path)),
    )

    async def read_metadata(_session, _tab):
        return metadata

    store = tmp_path / "rollout.jsonl"
    store.write_text("{}\n", encoding="utf-8")
    joined = PaneJoin("agent", 10, "codex", tmp_path, "project", ("session",), "session", store)
    writes = []
    clock = [0.0]
    updater = CompanionUpdater(
        read_metadata=read_metadata,
        clock=lambda: clock[0],
        wall_clock=lambda: 100.0 + clock[0],
        joiner=lambda *_args, **_kwargs: joined,
        table_reader=lambda: {},
        state_writer=lambda path, state: writes.append((path, state)),
    )
    return (
        updater,
        app,
        {"agent": CompanionPair("agent", "summary", tmp_path / "state")},
        writes,
        clock,
    )


def test_tail_is_incremental_and_semantically_unchanged_tick_does_not_rewrite(
    tmp_path, monkeypatch
):
    adapter = Adapter(
        [
            TailResult((_event(),), Cursor(10)),
            TailResult((), Cursor(10)),
        ]
    )
    monkeypatch.setattr(companion_update, "_adapter", lambda *_args: adapter)
    updater, app, pairs, writes, clock = _fixture(tmp_path, adapter)
    assert asyncio.run(updater.refresh_once(app, pairs)) == 1
    clock[0] = 1.0
    assert asyncio.run(updater.refresh_once(app, pairs)) == 0
    assert adapter.offsets == [0, 10]
    assert len(writes) == 1


def test_malformed_complete_record_forces_unknown_state(tmp_path, monkeypatch):
    adapter = Adapter([TailResult((_event(),), Cursor(10), malformed_records=1)])
    monkeypatch.setattr(companion_update, "_adapter", lambda *_args: adapter)
    updater, app, pairs, writes, _clock = _fixture(tmp_path, adapter)
    asyncio.run(updater.refresh_once(app, pairs))
    state = writes[-1][1]
    assert state.status == "UNKNOWN"
    assert state.join_state is JoinState.JOINED
    assert "malformed" in state.detail


def test_current_tail_advance_clears_cached_quiet_refinement(tmp_path, monkeypatch):
    adapter = Adapter([TailResult((_ended_event(),), Cursor(10))])
    monkeypatch.setattr(companion_update, "_adapter", lambda *_args: adapter)
    monkeypatch.setattr(
        companion_update,
        "observe_liveness",
        lambda *_args, **_kwargs: Liveness(Tri.TRUE, Tri.FALSE),
    )
    updater, app, pairs, writes, _clock = _fixture(tmp_path, adapter)
    asyncio.run(updater.refresh_once(app, pairs))
    assert writes[-1][1].status == "AWAITING_HUMAN"


def test_snapshot_turn_contract_maps_to_canonical_signals():
    base = SummarySnapshot(
        source="claude-code",
        session_key="session",
        command_result=Claim(None, Provenance.STRUCTURAL, "event_stream"),
    )
    waiting = SummarySnapshot(
        **{**base.__dict__, "turn": Claim.structural("Awaiting explicit answer", "tool_use")}
    )
    status = companion_update.derive_status(signals_from_snapshot(waiting))
    assert status is Status.AWAITING_HUMAN


def test_unjoined_pane_retries_join_on_slower_cadence(tmp_path):
    calls = []
    clock = [0.0]
    agent = types.SimpleNamespace(session_id="agent")
    tab = types.SimpleNamespace(sessions=[agent])
    app = types.SimpleNamespace(
        terminal_windows=[types.SimpleNamespace(tabs=[tab])],
        get_session_by_id=lambda _session_id: agent,
    )

    async def metadata(_session, _tab):
        return SessionMetadata(
            session=agent,
            tab=tab,
            pane=PaneVariables("agent", 10, "codex", str(tmp_path)),
        )

    updater = CompanionUpdater(
        read_metadata=metadata,
        clock=lambda: clock[0],
        wall_clock=lambda: clock[0],
        joiner=lambda *_args, **_kwargs: calls.append("join"),
        table_reader=lambda: {},
        state_writer=lambda *_args: None,
    )
    pairs = {"agent": CompanionPair("agent", "summary", Path(tmp_path / "state"))}
    asyncio.run(updater.refresh_once(app, pairs))
    clock[0] = 1.0
    asyncio.run(updater.refresh_once(app, pairs))
    assert calls == ["join"]


def test_blocking_initial_tail_does_not_stall_event_loop_and_reports_progress(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()

    class BlockingAdapter:
        def tail(self, _path, _cursor):
            started.set()
            release.wait(timeout=2)
            return TailResult((_event(),), Cursor(10))

    monkeypatch.setattr(companion_update, "_adapter", lambda *_args: BlockingAdapter())
    statuses = []
    updater, app, pairs, _writes, _clock = _fixture(tmp_path, BlockingAdapter())
    updater._on_status = statuses.append

    async def drive():
        task = asyncio.create_task(updater.refresh_once(app, pairs))
        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        await task

    asyncio.run(drive())
    assert any("tailing companion transcript" in message for message in statuses)
