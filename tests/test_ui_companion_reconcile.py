"""
Reconciliation fixed points: reuse without mutation, targeted/inventory-only passes, and
marker/orphan cleanup.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from palaver.ui import companion
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    ROLE_VARIABLE,
    CompanionController,
)
from tests._companion_support import (
    FakeApp,
    FakeSession,
    FakeTab,
    controller,
    metadata_reader,
    paired_app,
    stub_iterm,
)


def test_marked_companion_is_never_probed_or_split(tmp_path):
    agent = FakeSession("agent")
    summary = FakeSession("summary", job_pid=20)
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, summary])
    seen = []
    ctl, _ = controller(tmp_path, detector=lambda variables, **_k: seen.append(variables.pane_id))

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert len(result.pairs) == 1
    # The agent pane is probed on every pass — that is how an agent that
    # exited to a shell prompt is noticed at all. The companion never is.
    assert "summary" not in seen
    assert not agent.split_calls and not summary.split_calls


@pytest.mark.parametrize("height", [6, 10])
def test_valid_pair_is_reused_without_resize_or_focus(tmp_path, monkeypatch, height):
    """Reuse never mutates layout, whatever height the companion is at.

    Regression guard. Resizing here ran inside the handler for the very
    layout-change event a resize raises, and `preferred_size` is advisory, so
    a companion that never landed exactly on `SUMMARY_ROWS` resized the window
    without end. `SUMMARY_ROWS` is applied once, at creation, and nowhere else.
    """
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app(focus_companion=False)
    summary.grid_size.height = height
    app.tab.sessions.append(summary)
    summary.tab = app.tab
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(app))

    assert result.created == ()
    assert result.pairs[0].companion_id == "summary"
    assert summary.preferred_size is None
    assert app.tab.layout_updates == 0
    assert agent.activate_calls == []


def test_repeated_reconciles_never_mutate_layout(tmp_path, monkeypatch):
    """The layout monitor reconciles on every layout change; reuse must be a
    fixed point, or the two feed each other forever."""
    stub_iterm(monkeypatch)
    app, agent, summary = paired_app(focus_companion=False)
    summary.grid_size.height = 6
    app.tab.sessions.append(summary)
    summary.tab = app.tab
    agent.vars[COMPANION_SESSION_VARIABLE] = "summary"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    ctl, _ = controller(tmp_path)

    for _ in range(5):
        asyncio.run(ctl.reconcile(app))

    assert app.tab.layout_updates == 0
    assert summary.preferred_size is None


def test_only_exact_marker_orphan_and_duplicate_are_closed(tmp_path):
    agent = FakeSession("agent")
    keeper = FakeSession("keep", job_pid=20)
    duplicate = FakeSession("duplicate", job_pid=21)
    ordinary = FakeSession("ordinary")
    agent.vars[COMPANION_SESSION_VARIABLE] = "keep"
    for item in (keeper, duplicate):
        item.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, keeper, duplicate, ordinary])
    ctl, _ = controller(tmp_path)

    result = asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert result.closed == ("duplicate",)
    assert duplicate.close_calls == [{"force": True}]
    assert ordinary.close_calls == []


def test_one_sided_pair_cleanup_leaves_no_orphan_state(tmp_path):
    agent = FakeSession("agent")
    summary = FakeSession("summary", job_pid=20)
    agent.vars[COMPANION_SESSION_VARIABLE] = "missing-peer"
    summary.vars.update({ROLE_VARIABLE: COMPANION_ROLE, AGENT_SESSION_VARIABLE: "agent"})
    tab = FakeTab([agent, summary])
    state = companion.opaque_state_path(tmp_path, "agent")
    state.write_text("stale", encoding="utf-8")
    ctl, _ = controller(tmp_path)

    asyncio.run(ctl.reconcile(FakeApp(tab)))

    assert summary.close_calls == [{"force": True}]
    assert not state.exists()
    assert agent.vars[DISABLED_VARIABLE] is True


def test_one_pane_failure_does_not_prevent_another_creation(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    bad = FakeSession("bad")
    bad.split_error = RuntimeError("split failed")
    good = FakeSession("good")
    summary = FakeSession("summary", job_pid=20)
    good.split_result = summary
    tab = FakeTab([bad, good, summary])
    tab.sessions.remove(summary)
    tab.current_session = summary
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(FakeApp(tab)))
    assert result.created == ("summary",)
    assert "bad" in result.refused


def test_targeted_reconcile_never_creates_an_unrequested_agent(tmp_path, monkeypatch):
    stub_iterm(monkeypatch)
    first = FakeSession("first")
    first_summary = FakeSession("first-summary", job_pid=20)
    first.split_result = first_summary
    second = FakeSession("second")
    second.split_result = FakeSession("second-summary", job_pid=20)
    tab = FakeTab([first, second])
    ctl, _ = controller(tmp_path)
    result = asyncio.run(ctl.reconcile(FakeApp(tab), only_agent_id="first"))
    assert result.created == ("first-summary",)
    assert second.split_calls == []


def test_inventory_only_reconcile_does_not_probe_process_table(tmp_path):
    app, _agent, _summary = paired_app()
    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_table_reader=lambda: (_ for _ in ()).throw(AssertionError("process probe")),
    )
    result = asyncio.run(ctl.reconcile(app, create=False))
    assert result.created == ()


def test_blocking_process_probe_runs_off_event_loop_with_visible_progress(tmp_path):
    app, _agent, _summary = paired_app()
    started = threading.Event()
    release = threading.Event()
    statuses = []

    def process_table():
        started.set()
        release.wait(timeout=2)
        return {}

    ctl = CompanionController(
        tmp_path,
        read_metadata=metadata_reader,
        process_detector=lambda *_args, **_kwargs: None,
        process_table_reader=process_table,
        on_status=statuses.append,
    )

    async def drive():
        task = asyncio.create_task(ctl.reconcile(app))
        while not started.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        await task

    asyncio.run(drive())
    assert "reading process table for companion reconciliation" in statuses
