"""
Companion creation failure paths: refusal, cancellation, partial-write cleanup, and
live-session reacquisition.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from palaver.ui.companion import (
    COMPANION_SESSION_VARIABLE,
    CompanionController,
    SessionMetadata,
)
from palaver.ui.pane_join import PaneVariables, SupportedPaneProcess
from tests._companion_support import (
    FakeApp,
    FakeSession,
    FakeTab,
    controller,
    detected,
    metadata_reader,
    paired_app,
    stub_iterm,
)


def test_small_agent_is_refused_without_split(tmp_path):
    app, agent, _summary = paired_app(height=5)
    ctl, writes = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(app))
    assert result.refused == ("agent",)
    assert agent.split_calls == []
    assert writes == []


def test_focus_restores_only_if_new_companion_remains_active(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, _summary = paired_app(focus_companion=False)
    ctl, _ = controller(tmp_path)
    asyncio.run(ctl.reconcile(app))
    assert agent.activate_calls == []

    app2, agent2, _summary2 = paired_app(focus_companion=True)
    ctl2, _ = controller(tmp_path)
    asyncio.run(ctl2.reconcile(app2))
    assert agent2.activate_calls == [{"select_tab": False, "order_window_front": False}]


def test_partial_creation_failure_closes_only_new_companion(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app()

    async def fail(_name, _value):
        raise RuntimeError("variable refused")

    summary.async_set_variable = fail
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(app))
    assert result.pairs == ()
    assert summary.close_calls == [{"force": True}]
    assert agent.close_calls == []


def test_cancellation_after_split_cleans_partial_companion_and_state(tmp_path):
    started = asyncio.Event()

    class BlockingCompanion(FakeSession):
        async def async_set_variable(self, name, value):
            started.set()
            await asyncio.Event().wait()

    agent = FakeSession("agent")
    summary = BlockingCompanion("summary", job_pid=20)
    agent.split_result = summary
    tab = FakeTab([agent])

    def seed(path, *_args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("seed", encoding="utf-8")

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        write_initial_state=seed,
        process_detector=detected,
        transcript_joiner=lambda *_args, **_kwargs: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )

    async def drive():
        task = asyncio.create_task(ctl.reconcile(FakeApp(tab)))
        await started.wait()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())
    assert summary.close_calls == [{"force": True}]
    assert agent.vars[COMPANION_SESSION_VARIABLE] == ""
    assert list(tmp_path.glob("*.json")) == []


def test_split_failure_removes_the_state_seed(tmp_path):
    app, agent, _summary = paired_app()
    agent.split_error = RuntimeError("cannot split")
    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_detector=detected,
        transcript_joiner=lambda *_a, **_k: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )
    asyncio.run(ctl.reconcile(app))
    assert list(tmp_path.glob("*.json")) == []


def test_initial_state_failure_is_guarded_per_pane_and_next_agent_creates(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    bad = FakeSession("bad")
    good = FakeSession("good")
    summary = FakeSession("good-summary", job_pid=20)
    good.split_result = summary
    tab = FakeTab([bad, good])
    calls = []

    def seed(path, *_args):
        calls.append(path)
        if len(calls) == 1:
            raise OSError("state directory unavailable")

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        write_initial_state=seed,
        process_detector=detected,
        transcript_joiner=lambda *_args, **_kwargs: None,
        process_table_reader=lambda: {},
        profile_builder=lambda command: command,
    )

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert result.created == ("good-summary",)
    assert "bad" in result.refused
    assert bad.split_calls == []
    assert len(calls) == 2


def test_create_reacquires_live_session_before_splitting(tmp_path, monkeypatch):
    """A stale inventory session is never mutated after the app exposes its replacement."""
    stub_iterm(monkeypatch)
    stale_agent = FakeSession("agent", height=30)
    stale_tab = FakeTab([stale_agent])
    live_agent = FakeSession("agent", height=30)
    summary = FakeSession("summary", job_pid=20)
    live_agent.split_result = summary
    live_tab = FakeTab([live_agent])
    app = FakeApp(live_tab)
    metadata = SessionMetadata(
        session=stale_agent,
        tab=stale_tab,
        pane=PaneVariables("agent", 10, "codex", "/tmp"),
    )
    ctl, _ = controller(tmp_path)

    result = asyncio.run(
        ctl._create(app, metadata, SupportedPaneProcess("agent", 10, "codex", Path("/tmp")), None)
    )

    assert result is not None
    assert stale_agent.split_calls == []
    assert len(live_agent.split_calls) == 1
