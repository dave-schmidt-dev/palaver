"""Opt-in, disposable live acceptance for the companion-pane surface.

This module is inert unless ``PALAVER_RUN_LIVE_COMPANION_TEST=1``.  Its one
test creates a new iTerm window and limits every inventory, lookup, mutation,
and assertion to IDs created in that window.  Existing working panes are
never enumerated by the controller or used as fixtures.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from palaver.ui import connection
from palaver.ui.companion import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    ROLE_VARIABLE,
    SUMMARY_ROWS,
    CompanionController,
    make_metadata_reader,
)
from palaver.ui.companion_render import STALE_AFTER_SECONDS
from palaver.ui.companion_state import CompanionState, JoinState, atomic_write_state
from palaver.ui.connection import COOKIE_ENV, KEY_ENV
from palaver.ui.pane_join import PaneJoin, PaneVariables, SupportedPaneProcess, read_process_table

LIVE_ENV = "PALAVER_RUN_LIVE_COMPANION_TEST"
LIVE_ENABLED = (
    os.environ.get(LIVE_ENV) == "1"
    and sys.platform == "darwin"
    and Path("/Applications/iTerm.app").exists()
    and connection.socket_path().is_socket()
)
live_only = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason=f"set {LIVE_ENV}=1 to create disposable live iTerm panes",
)


class _OwnedApp:
    """Expose only the test-created window to the lifecycle controller."""

    def __init__(self, real_app, window, owned_ids: set[str]) -> None:
        self._real_app = real_app
        self._window_id = window.window_id
        self.terminal_windows = [window]
        self.owned_ids = owned_ids

    def get_session_by_id(self, session_id: str):
        if session_id not in self.owned_ids:
            return None
        return self._real_app.get_session_by_id(session_id)

    async def async_refresh(self) -> None:
        """Refresh the real model, retaining only the disposable test window."""
        await self._real_app.async_refresh()
        window = self._real_app.get_window_by_id(self._window_id)
        self.terminal_windows = [] if window is None else [window]
        if window is not None:
            self.owned_ids.update(
                session.session_id for tab in window.tabs for session in tab.sessions
            )


async def _eventually(check, *, timeout: float = 12.0, interval: float = 0.1):
    deadline = asyncio.get_running_loop().time() + timeout
    last = None
    while asyncio.get_running_loop().time() < deadline:
        last = await check()
        if last:
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"live condition did not become true; last value was {last!r}")


async def _screen_text(session) -> str:
    contents = await session.async_get_screen_contents()
    return "\n".join(contents.line(index).string for index in range(contents.number_of_lines))


async def _screen_contains(session, text: str) -> bool:
    return text in await _screen_text(session)


async def _frame_tuple(window) -> tuple[float, float, float, float]:
    """Reduce the window frame to a value that survives an iTerm refresh."""

    frame = await window.async_get_frame()
    return (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height)


async def _has_at_least_rows(session, rows: int) -> bool:
    await asyncio.sleep(0)
    return session.grid_size.height >= rows


async def _grid_is_narrower(session, width: int) -> bool:
    await asyncio.sleep(0)
    return session.grid_size.width < width


async def _job_pid(session) -> int | None:
    value = await session.async_get_variable("jobPid")
    try:
        parsed = int(value)
    except TypeError, ValueError:
        return None
    return parsed if parsed > 0 else None


async def _job_has_exited(session) -> bool:
    return await _job_pid(session) is None


async def _renderer_pid(state_path: Path) -> int | None:
    """Locate the unique test-owned renderer rather than iTerm's shell wrapper."""

    table = await asyncio.to_thread(read_process_table)
    marker = str(state_path)
    candidates = [
        info
        for info in table.values()
        if "palaver.ui.companion_render" in info.command and marker in info.command
    ]
    parent_ids = {info.ppid for info in candidates}
    leaves = [info.pid for info in candidates if info.pid not in parent_ids]
    return leaves[0] if len(leaves) == 1 else None


async def _job_status(session) -> str:
    """Return a bounded status label without exposing a process identifier."""

    if session is None:
        return "missing"
    try:
        return "running" if await _job_pid(session) is not None else "exited"
    except Exception:  # The diagnostic must survive a racing session teardown.
        return "unavailable"


async def _collect_focus(monitor, events: list[object]) -> None:
    while True:
        events.append(await monitor.async_get_next_update())


async def _collect_terminations(
    monitor,
    owned_ids: set[str],
    events: list[str],
) -> None:
    while True:
        session_id = await monitor.async_get()
        if session_id in owned_ids:
            events.append(session_id)


def _session_ids_from_node(node) -> set[str]:
    session_ids: set[str] = set()
    for link in node.links:
        if link.HasField("session"):
            session_ids.add(link.session.unique_identifier)
        elif link.HasField("node"):
            session_ids.update(_session_ids_from_node(link.node))
    return session_ids


async def _raw_session_ids(iterm2, connection_value) -> set[str] | None:
    """Read iTerm's uncached session inventory, if the diagnostic RPC succeeds."""

    try:
        result = await iterm2.rpc.async_list_sessions(connection_value)
        response = result.list_sessions_response
        session_ids: set[str] = set()
        for window in response.windows:
            for tab in window.tabs:
                session_ids.update(_session_ids_from_node(tab.root))
                session_ids.update(session.unique_identifier for session in tab.minimized_sessions)
        session_ids.update(session.unique_identifier for session in response.buried_sessions)
        return session_ids
    except Exception:  # Keep the primary failure report useful across API variants.
        return None


async def _assert_pairs_still_live(
    app,
    iterm2,
    connection_value,
    pair_ids: set[str],
    terminated_owned: list[str],
) -> None:
    """Refresh the cache and report bounded evidence for a vanished owned pane."""

    await app.async_refresh()
    raw_ids = await _raw_session_ids(iterm2, connection_value)
    cached_sessions = {
        session_id: app.get_session_by_id(session_id) for session_id in sorted(pair_ids)
    }
    cached_membership = {
        session_id: session is not None for session_id, session in cached_sessions.items()
    }
    raw_membership = (
        None
        if raw_ids is None
        else {session_id: session_id in raw_ids for session_id in sorted(pair_ids)}
    )
    missing = [
        session_id
        for session_id in sorted(pair_ids)
        if not cached_membership[session_id]
        or (raw_membership is not None and not raw_membership[session_id])
    ]
    if not missing:
        return

    job_status = {
        session_id: await _job_status(cached_sessions[session_id])
        for session_id in sorted(pair_ids)
        if raw_membership is None or raw_membership[session_id]
    }
    pytest.fail(
        "owned pair vanished after reconciliation: "
        f"missing={missing!r}; "
        f"terminated_owned={sorted(set(terminated_owned) & pair_ids)!r}; "
        f"raw_membership={raw_membership!r}; "
        f"cached_membership={cached_membership!r}; "
        f"job_status={job_status!r}"
    )


async def _focus_events_during(iterm2, connection_value, operation) -> list[object]:
    events: list[object] = []
    async with iterm2.FocusMonitor(connection_value) as monitor:
        collector = asyncio.create_task(_collect_focus(monitor, events))
        try:
            # A subscription may report current focus as its initial state.
            # The assertion concerns changes caused by the operation.
            await asyncio.sleep(0.1)
            events.clear()
            await operation()
            await asyncio.sleep(0.4)
        finally:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
    return events


def _detector(agent_ids: set[str], pane: PaneVariables, **_kwargs):
    if pane.pane_id not in agent_ids:
        return None
    return SupportedPaneProcess(pane.pane_id, 1, "codex", Path.cwd())


def _joiner(agent_ids: set[str], pane: PaneVariables, **_kwargs):
    if pane.pane_id not in agent_ids:
        return None
    key = f"live-{pane.pane_id}"
    return PaneJoin(pane.pane_id, 1, "codex", Path.cwd(), "live", (key,), key)


def _snapshot(index: int, *, updated: float | None = None, request: str | None = None):
    return CompanionState(
        producer_updated_at=time.time() if updated is None else updated,
        project=f"LIVE-{index}",
        source="codex",
        status="WORKING",
        join_state=JoinState.JOINED,
        request=request or f"request-{index}",
        command_result=f"command-{index}",
        detail=f"connection-{index}",
        recent=(f"activity-{index}",),
        tasks=(f"task-{index}",),
        questions=(f"question-{index}",),
    )


def _run_live(body, *, timeout: float = 90.0):
    """Run once with a fresh cookie, without printing or retaining it."""

    cookie, key = connection.request_cookie_and_key(advisory_name="palaver-live-test")
    previous = {name: os.environ.get(name) for name in (COOKIE_ENV, KEY_ENV)}
    os.environ[COOKIE_ENV] = cookie
    os.environ[KEY_ENV] = key
    result = {}

    async def connected(connection_value):
        result["value"] = await asyncio.wait_for(body(connection_value), timeout=timeout)

    connection.reset_library_state()
    try:
        connection.import_iterm2().run_until_complete(connected)
    finally:
        connection.reset_library_state()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return result.get("value")


@live_only
def test_three_test_owned_agents_have_isolated_resilient_companions(tmp_path):
    async def body(connection_value):
        iterm2 = connection.import_iterm2()
        app = await iterm2.async_get_app(connection_value)
        window = None
        window_id = None
        state_dir = tmp_path / ".state" / "companions"
        owned_ids: set[str] = set()
        status: list[str] = []
        agent_ids: set[str] = set()
        terminated_owned: list[str] = []
        termination_monitor = None
        termination_collector = None
        try:
            window = await iterm2.Window.async_create(connection_value, command="/bin/sleep 300")
            assert window is not None
            window_id = window.window_id
            assert window.current_tab is not None
            tabs = [window.current_tab]
            for _ in range(2):
                tab = await window.async_create_tab(command="/bin/sleep 300")
                assert tab is not None
                tabs.append(tab)
            agents = [tab.current_session for tab in tabs]
            assert all(agent is not None for agent in agents)
            agent_ids.update(agent.session_id for agent in agents)
            owned_ids.update(agent_ids)
            owned_app = _OwnedApp(app, window, owned_ids)
            controller = CompanionController(
                state_dir,
                read_metadata=make_metadata_reader(connection_value),
                process_detector=lambda pane, **kwargs: _detector(agent_ids, pane, **kwargs),
                transcript_joiner=lambda pane, **kwargs: _joiner(agent_ids, pane, **kwargs),
                process_table_reader=lambda: {},
                on_status=status.append,
            )
            termination_monitor = iterm2.SessionTerminationMonitor(connection_value)
            await termination_monitor.__aenter__()
            termination_collector = asyncio.create_task(
                _collect_terminations(termination_monitor, owned_ids, terminated_owned)
            )

            selected_tab = window.current_tab.tab_id
            # A window tall enough that a SUMMARY_ROWS companion fits beside a
            # working agent. In a cramped window iTerm clamps the request, which
            # is safe but would not exercise the sizing write.
            created_frame = await window.async_get_frame()
            await window.async_set_frame(iterm2.Frame(created_frame.origin, iterm2.Size(900, 760)))
            await _eventually(lambda: _has_at_least_rows(agents[0], SUMMARY_ROWS * 3))
            # INV-2: the one sizing write must not move or resize the window.
            frame_before = await _frame_tuple(window)
            rows_before = {agent.session_id: agent.grid_size.height for agent in agents}

            async def create_companions():
                result = await controller.reconcile(owned_app)
                assert len(result.created) == 3
                owned_ids.update(result.created)

            creation_focus = await _focus_events_during(iterm2, connection_value, create_companions)
            assert window.current_tab.tab_id == selected_tab
            assert not any(event.selected_tab_changed for event in creation_focus)
            assert not any(event.window_changed for event in creation_focus)

            last_seen: dict[str, object] = {}

            async def _companions_are_summary_height():
                await owned_app.async_refresh()
                heights = {
                    pair.companion_id: (
                        None
                        if (pane := owned_app.get_session_by_id(pair.companion_id)) is None
                        else pane.grid_size.height
                    )
                    for pair in controller.pairs.values()
                }
                last_seen["heights"] = heights
                last_seen["frame"] = await _frame_tuple(window)
                return heights if all(row == SUMMARY_ROWS for row in heights.values()) else False

            # Sizing a companion writes the whole tab, which shrinks the window
            # and every tab in it unless the frame is put back. INV-2.
            try:
                await _eventually(_companions_are_summary_height)
            except AssertionError as error:
                raise AssertionError(
                    f"companions never reached {SUMMARY_ROWS} rows. "
                    f"seen={last_seen} wanted frame={frame_before} notes={status}"
                ) from error
            assert await _frame_tuple(window) == frame_before
            assert not [note for note in status if "window frame" in note]
            for agent_id in controller.pairs:
                resized = owned_app.get_session_by_id(agent_id)
                # The agent keeps every row the companion did not take, less the
                # few iTerm spends on the pane title bars a split introduces.
                assert resized.grid_size.height <= rows_before[agent_id] - SUMMARY_ROWS
                assert resized.grid_size.height >= rows_before[agent_id] - SUMMARY_ROWS - 6

            pairs = controller.pairs
            assert set(pairs) == agent_ids
            assert len({pair.companion_id for pair in pairs.values()}) == 3
            assert not ({pair.companion_id for pair in pairs.values()} & agent_ids)
            second = await controller.reconcile(owned_app)
            assert second.created == ()
            assert len(second.pairs) == 3
            assert sum(len(tab.sessions) for tab in window.tabs) == 6
            pair_ids = agent_ids | {pair.companion_id for pair in pairs.values()}
            await _assert_pairs_still_live(
                app,
                iterm2,
                connection_value,
                pair_ids,
                terminated_owned,
            )

            for agent_id, pair in pairs.items():
                companion = owned_app.get_session_by_id(pair.companion_id)
                agent = owned_app.get_session_by_id(agent_id)
                assert companion is not None and agent is not None
                assert await companion.async_get_variable(ROLE_VARIABLE) == COMPANION_ROLE
                assert await companion.async_get_variable(AGENT_SESSION_VARIABLE) == agent_id
                assert (
                    await agent.async_get_variable(COMPANION_SESSION_VARIABLE) == pair.companion_id
                )
                assert await companion.async_get_variable(COMPANION_SESSION_VARIABLE) in {None, ""}

            ordered = [pairs[agent.session_id] for agent in agents]
            for index, pair in enumerate(ordered, start=1):
                atomic_write_state(pair.state_path, _snapshot(index))
            for index, pair in enumerate(ordered, start=1):
                companion = owned_app.get_session_by_id(pair.companion_id)
                await _eventually(
                    lambda c=companion, token=f"LIVE-{index}": _screen_contains(c, token)
                )

            atomic_write_state(ordered[0].state_path, _snapshot(1, request="UPDATED-ONLY-ONE"))
            first_companion = owned_app.get_session_by_id(ordered[0].companion_id)
            second_companion = owned_app.get_session_by_id(ordered[1].companion_id)
            await _eventually(lambda: _screen_contains(first_companion, "UPDATED-ONLY-ONE"))
            assert "UPDATED-ONLY-ONE" not in await _screen_text(second_companion)

            # The renderer must adapt to actual dimensions, not assume ten rows.
            long_request = "NARROW-" + "x" * 200
            atomic_write_state(ordered[0].state_path, _snapshot(1, request=long_request))
            first_agent = agents[0]
            original_frame = await window.async_get_frame()
            original_width = first_companion.grid_size.width
            narrow_pixels = max(360, original_frame.size.width // 2)
            await window.async_set_frame(
                iterm2.Frame(
                    original_frame.origin,
                    iterm2.Size(narrow_pixels, original_frame.size.height),
                )
            )
            first_companion.preferred_size = iterm2.Size(first_companion.grid_size.width, 2)
            first_agent.preferred_size = iterm2.Size(
                first_agent.grid_size.width,
                max(1, first_agent.grid_size.height + SUMMARY_ROWS - 2),
            )
            await tabs[0].async_update_layout()
            await _eventually(lambda: _grid_is_narrower(first_companion, original_width))
            await _eventually(lambda: _screen_contains(first_companion, "NARROW-"))
            assert long_request not in await _screen_text(first_companion)
            assert first_companion.grid_size.height <= 3

            # No producer heartbeat means the renderer independently becomes stale.
            atomic_write_state(
                ordered[1].state_path,
                _snapshot(2, updated=time.time() - STALE_AFTER_SECONDS - 1),
            )
            await _eventually(lambda: _screen_contains(second_companion, "PALAVER  STALE"))
            atomic_write_state(ordered[1].state_path, _snapshot(2))

            # Accidental typing is consumed by the companion and never reaches its agent.
            marker = "INPUT-MUST-STAY-IN-COMPANION"
            await first_companion.async_send_text(marker, suppress_broadcast=True)
            await asyncio.sleep(0.4)
            assert marker not in await _screen_text(first_companion)
            assert marker not in await _screen_text(first_agent)

            # Terminate only the test-owned renderer. iTerm keeps the same
            # pane/layout but assigns a new API session GUID on restart.
            restart_pair = ordered[2]
            restart_companion = owned_app.get_session_by_id(restart_pair.companion_id)
            renderer_pid = await _eventually(lambda: _renderer_pid(restart_pair.state_path))
            os.kill(renderer_pid, signal.SIGTERM)
            await _eventually(lambda: _job_has_exited(restart_companion), timeout=3.0)
            await controller.reconcile(owned_app)
            restarted_pair = controller.pairs[restart_pair.agent_id]
            assert restarted_pair.companion_id != restart_pair.companion_id
            restarted_companion = owned_app.get_session_by_id(restarted_pair.companion_id)
            assert restarted_companion is not None
            assert sum(len(tab.sessions) for tab in owned_app.terminal_windows[0].tabs) == 6
            await _eventually(lambda: _screen_contains(restarted_companion, "LIVE-3"))
            ordered[2] = restarted_pair

            async def steady_refresh():
                atomic_write_state(ordered[0].state_path, _snapshot(1, request="STEADY-REFRESH"))
                await _eventually(lambda: _screen_contains(first_companion, "STEADY-REFRESH"))

            steady_focus = await _focus_events_during(iterm2, connection_value, steady_refresh)
            assert steady_focus == []
            assert window.current_tab.tab_id == selected_tab

            # Closing one test agent tears down exactly its reciprocal companion.
            removed_agent = agents[1]
            removed_pair = ordered[1]
            surviving_ids = {
                agents[0].session_id,
                agents[2].session_id,
                ordered[0].companion_id,
                ordered[2].companion_id,
            }
            await removed_agent.async_close(force=True)
            agent_ids.discard(removed_agent.session_id)
            await controller.handle_termination(owned_app, removed_agent.session_id)

            async def removed() -> bool:
                return (
                    app.get_session_by_id(removed_agent.session_id) is None
                    and app.get_session_by_id(removed_pair.companion_id) is None
                )

            await _eventually(removed)
            assert all(
                app.get_session_by_id(session_id) is not None for session_id in surviving_ids
            )
        finally:
            if termination_collector is not None:
                termination_collector.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await termination_collector
            if termination_monitor is not None:
                await termination_monitor.__aexit__(None, None, None)
            if window_id is not None:
                owned_window = app.get_window_by_id(window_id)
                if owned_window is not None:
                    await owned_window.async_close(force=True)
            if state_dir.exists():
                for path in state_dir.glob("*.json"):
                    path.unlink(missing_ok=True)

    _run_live(body)
