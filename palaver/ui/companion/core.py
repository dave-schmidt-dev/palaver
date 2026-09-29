"""CompanionController: reconciliation lifecycle, composed from its mixins."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from palaver.ui.pane_join import (
    ProcessTable,
    detect_supported_process,
    join_pane,
    read_process_table,
)

from .creation import CreationMixin, build_companion_profile
from .paths import opaque_state_path, write_initial_state
from .records import (
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    CompanionPair,
    ReadMetadata,
    ReconcileResult,
    SessionMetadata,
    WriteInitialState,
    _no_status,
)
from .restart import RestartMixin


class CompanionController(CreationMixin, RestartMixin):
    """Reconcile supported agent panes with exactly one owned companion."""

    def __init__(
        self,
        state_dir: Path,
        *,
        read_metadata: ReadMetadata,
        write_initial_state: WriteInitialState = write_initial_state,
        process_detector=detect_supported_process,
        transcript_joiner=join_pane,
        process_table_reader=read_process_table,
        cwd_reader=None,
        profile_builder=build_companion_profile,
        clock: Callable[[], float] | None = None,
        on_status: Callable[[str], None] = _no_status,
    ) -> None:
        self.state_dir = state_dir
        self._read_metadata = read_metadata
        self._write_initial_state = write_initial_state
        self._detect = process_detector
        self._join = transcript_joiner
        self._read_process_table = process_table_reader
        self._cwd_reader = cwd_reader
        self._profile_builder = profile_builder
        self._clock = time.monotonic if clock is None else clock
        self._on_status = on_status
        self._lock = asyncio.Lock()
        self._pending: set[str] = set()
        self._pairs: dict[str, CompanionPair] = {}
        self._manager_closing: set[str] = set()
        self._restart_attempts: dict[str, int] = {}
        self._retry_after: dict[str, float] = {}
        # iTerm assigns a new API session GUID when an exited session is
        # restarted. Keep the old identity only long enough to reacquire the
        # exact marked pane after App refreshes its layout model.
        self._rollover_pending: dict[str, str] = {}
        # When a paired pane first showed no supported agent process. Cleared
        # by any later detection, so only a sustained absence tears the pair
        # down. See `AGENT_EXIT_GRACE`.
        self._agent_missing_since: dict[str, float] = {}

    @property
    def pairs(self) -> Mapping[str, CompanionPair]:
        """Return a copy of the current agent-keyed pair map."""
        return dict(self._pairs)

    async def reconcile(
        self, app: Any, *, create: bool = True, only_agent_id: str | None = None
    ) -> ReconcileResult:
        """Make current iTerm state agree with the companion ownership rules."""
        async with self._lock:
            return await self._reconcile_locked(app, create=create, only_agent_id=only_agent_id)

    async def _inventory(self, app: Any) -> dict[str, SessionMetadata]:
        found: dict[str, SessionMetadata] = {}
        for window in app.terminal_windows:
            for tab in window.tabs:
                for session in tab.sessions:
                    try:
                        metadata = await self._read_metadata(session, tab)
                    except Exception:
                        self._on_status(f"could not inspect pane {session.session_id}")
                        continue
                    if metadata is not None:
                        found[metadata.session_id] = metadata
        return found

    async def _reconcile_locked(
        self,
        app: Any,
        *,
        create: bool,
        only_agent_id: str | None,
        skip_restart: frozenset[str] = frozenset(),
    ) -> ReconcileResult:
        inventory = await self._inventory(app)
        companions = {
            pane_id: metadata
            for pane_id, metadata in inventory.items()
            if metadata.is_companion or pane_id in self._pending
        }
        agents = {pane_id: item for pane_id, item in inventory.items() if pane_id not in companions}
        closed: list[str] = []
        created: list[str] = []
        refused: list[str] = []
        pairs: dict[str, CompanionPair] = {}
        restarting: set[str] = set()
        rollover_protected: set[str] = set()
        # Agents whose companion this pass tore down. The creation loop below
        # reads the inventory snapshot taken above, in which the agent still
        # carries its now-cleared companion marker; without this set it would
        # read that stale marker as a user-closed pane and disable the pane
        # permanently, so the next agent started there would never be paired.
        torn_down: set[str] = set()
        table: ProcessTable | None = None

        async def process_table() -> ProcessTable:
            nonlocal table
            if table is None:
                self._on_status("reading process table for companion reconciliation")
                table = await asyncio.to_thread(self._read_process_table)
            return table

        # A reciprocal pair is reused without changing its size or focus.
        # `SUMMARY_ROWS` is applied once, at creation. Resizing here would
        # mutate layout from inside the handler for the layout-change event
        # that mutation raises, and iTerm treats `preferred_size` as advisory,
        # so a height that never lands exactly on the target would resize the
        # window without end.
        candidates: dict[str, list[SessionMetadata]] = {}
        for companion in companions.values():
            if companion.agent_session:
                candidates.setdefault(companion.agent_session, []).append(companion)
        for agent_id, owned in candidates.items():
            agent = agents.get(agent_id)
            if agent is not None and agent.disabled:
                for item in owned:
                    if await self._close_owned(item):
                        closed.append(item.session_id)
                await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                try:
                    opaque_state_path(self.state_dir, agent_id).unlink(missing_ok=True)
                except OSError:
                    self._on_status(f"could not remove disabled companion state for {agent_id}")
                self._forget_agent(agent_id)
                continue
            if agent is not None and await self._agent_has_exited(agent, await process_table()):
                # The agent process is gone but its pane is still a shell
                # prompt the user owns, so the marker is cleared before the
                # close and the pane is never disabled: starting another agent
                # here must pair again without an explicit re-enable.
                await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                for item in owned:
                    if await self._close_owned(item):
                        closed.append(item.session_id)
                try:
                    opaque_state_path(self.state_dir, agent_id).unlink(missing_ok=True)
                except OSError:
                    self._on_status(f"could not remove ended-agent companion state for {agent_id}")
                self._forget_agent(agent_id)
                torn_down.add(agent_id)
                self._on_status(f"closed companion for ended agent session {agent_id}")
                continue
            reciprocal = [
                item
                for item in owned
                if agent is not None and agent.companion_session == item.session_id
            ]
            pending_old_id = self._rollover_pending.get(agent_id)
            if (
                agent is not None
                and pending_old_id is not None
                and len(owned) == 1
                and owned[0].session_id != pending_old_id
            ):
                # Restart preserves session user variables but changes iTerm's
                # API GUID. The exact owner marker is the stable identity.
                item = owned[0]
                await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, item.session_id)
                self._rollover_pending.pop(agent_id, None)
                reciprocal = [item]
            elif pending_old_id is not None and agent is None:
                # The agent pane closed mid-restart. Nothing clears a pending
                # rollover once its agent is gone, so protecting the companion
                # here would exempt it from every later sweep and leave the
                # pane on screen for good. Drop the pending entry and let the
                # unpaired cleanup below close it.
                self._rollover_pending.pop(agent_id, None)
            elif pending_old_id is not None:
                # Never publish the pre-restart handle to the updater. A later
                # refresh/reconcile will either observe the new marked GUID or
                # keep this agent fail-closed without creating a duplicate.
                restarting.add(agent_id)
                rollover_protected.update(item.session_id for item in owned)
                continue
            exited = [item for item in reciprocal if item.pane.job_pid is None]
            for item in exited:
                if item.session_id in skip_restart:
                    continue
                if self._clock() < self._retry_after.get(agent_id, 0.0):
                    restarting.add(agent_id)
                    continue
                try:
                    await item.session.async_restart(only_if_exited=True)
                except Exception:
                    self._record_restart_failure(agent_id)
                    restarting.add(agent_id)
                else:
                    self._rollover_pending[agent_id] = item.session_id
                    new_id = await self._rebind_restarted_companion(app, agent_id)
                    if new_id is None:
                        self._record_restart_failure(agent_id)
                        return await self._reconcile_locked(
                            app,
                            create=create,
                            only_agent_id=only_agent_id,
                            skip_restart=skip_restart,
                        )
                    self._restart_attempts.pop(agent_id, None)
                    self._retry_after.pop(agent_id, None)
                    return await self._reconcile_locked(
                        app,
                        create=create,
                        only_agent_id=only_agent_id,
                        skip_restart=skip_restart | {new_id},
                    )
            keeper = sorted(reciprocal, key=lambda item: item.session_id)[:1]
            if keeper:
                item = keeper[0]
                pairs[agent_id] = CompanionPair(
                    agent_id, item.session_id, opaque_state_path(self.state_dir, agent_id)
                )
            for item in owned:
                if not keeper or item.session_id != keeper[0].session_id:
                    if await self._close_owned(item):
                        closed.append(item.session_id)
            if not keeper and owned:
                try:
                    opaque_state_path(self.state_dir, agent_id).unlink(missing_ok=True)
                except OSError:
                    self._on_status(f"could not remove unpaired state for {agent_id}")

        # Marker-only panes with no owner are Palaver-owned orphans.
        paired_companions = {pair.companion_id for pair in pairs.values()}
        for companion_id, companion in companions.items():
            if (
                companion_id in paired_companions
                or companion_id in rollover_protected
                or companion_id in closed
            ):
                continue
            if not companion.agent_session or companion.agent_session not in agents:
                if await self._close_owned(companion):
                    closed.append(companion_id)
                    if companion.agent_session:
                        try:
                            opaque_state_path(self.state_dir, companion.agent_session).unlink(
                                missing_ok=True
                            )
                        except OSError:
                            self._on_status(
                                f"could not remove orphan state for {companion.agent_session}"
                            )

        for agent_id, agent in sorted(agents.items()):
            if agent_id in pairs or agent.disabled or agent_id in torn_down:
                continue
            if not create or (only_agent_id is not None and agent_id != only_agent_id):
                continue
            if agent_id in self._rollover_pending:
                refused.append(agent_id)
                continue
            if agent.companion_session and agent_id not in restarting:
                # A missing reciprocal peer represents a user-closed pane.
                await agent.session.async_set_variable(DISABLED_VARIABLE, True)
                await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                refused.append(agent_id)
                continue
            if self._clock() < self._retry_after.get(agent_id, 0.0):
                refused.append(agent_id)
                continue
            kwargs: dict[str, Any] = {"table": await process_table()}
            if self._cwd_reader is not None:
                kwargs["cwd_reader"] = self._cwd_reader
            self._on_status(f"probing supported process for pane {agent_id}")
            detected = await asyncio.to_thread(self._detect, agent.pane, **kwargs)
            if detected is None:
                refused.append(agent_id)
                continue
            self._on_status(f"joining transcript for pane {agent_id}")
            joined = await asyncio.to_thread(self._join, agent.pane, **kwargs)
            pair = await self._create(app, agent, detected, joined)
            if pair is None:
                refused.append(agent_id)
            else:
                pairs[agent_id] = pair
                created.append(pair.companion_id)

        self._pairs = pairs
        result = ReconcileResult(
            pairs=tuple(pairs[key] for key in sorted(pairs)),
            created=tuple(created),
            closed=tuple(closed),
            refused=tuple(refused),
        )
        self._on_status(
            f"companions: {len(result.pairs)} paired, {len(created)} created, {len(closed)} cleaned"
        )
        return result

    async def handle_termination(self, app: Any, session_id: str) -> None:
        """Classify a PTY process end after refreshing iTerm's visible layout."""
        async with self._lock:
            manager_close = session_id in self._manager_closing
            self._manager_closing.discard(session_id)
            pair_by_companion = next(
                (pair for pair in self._pairs.values() if pair.companion_id == session_id), None
            )
            self._on_status(f"refreshing layout after session termination {session_id}")
            try:
                await app.async_refresh()
            except Exception:
                self._on_status(f"could not classify session termination {session_id}")
                return
            inventory = await self._inventory(app)

            if pair_by_companion is not None:
                agent_id = pair_by_companion.agent_id
                agent = inventory.get(agent_id)
                if manager_close:
                    self._pairs.pop(agent_id, None)
                    if agent is not None:
                        await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                    return

                visible = sorted(
                    (
                        item
                        for item in inventory.values()
                        if item.is_companion and item.agent_session == agent_id
                    ),
                    key=lambda item: item.session_id,
                )
                if len(visible) == 1 and agent is not None:
                    current = visible[0]
                    if current.session_id != pair_by_companion.companion_id:
                        await agent.session.async_set_variable(
                            COMPANION_SESSION_VARIABLE, current.session_id
                        )
                        self._pairs[agent_id] = CompanionPair(
                            agent_id, current.session_id, pair_by_companion.state_path
                        )
                        self._rollover_pending.pop(agent_id, None)
                        self._on_status(
                            f"rebound companion identity for {agent_id} to {current.session_id}"
                        )
                        return
                    if current.pane.job_pid is None:
                        await self._reconcile_locked(app, create=True, only_agent_id=agent_id)
                    return

                # Only absence from the refreshed visible layout is a close.
                if not visible:
                    self._pairs.pop(agent_id, None)
                    self._rollover_pending.pop(agent_id, None)
                    if agent is not None:
                        await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                        await agent.session.async_set_variable(DISABLED_VARIABLE, True)
                    try:
                        pair_by_companion.state_path.unlink(missing_ok=True)
                    except OSError:
                        self._on_status(f"could not remove companion state for {agent_id}")
                return

            pair = self._pairs.get(session_id)
            if pair is None:
                # A delayed notification for the pre-restart GUID is harmless.
                return
            # Every per-agent entry goes, not just the pair: a rollover left
            # pending here is never cleared again, and it would protect this
            # companion from the orphan sweep for the rest of the run.
            self._forget_agent(session_id)
            agent = inventory.get(session_id)
            if agent is not None:
                # A supported agent process can end while its iTerm pane stays
                # open at a shell prompt. Detach Palaver without disabling or
                # otherwise controlling that user-owned pane.
                await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
            companion = inventory.get(pair.companion_id)
            if companion is not None:
                # Pair membership selects the companion and the exact role
                # marker is still required before any close. Agent session ids
                # are never passed to async_close.
                await self._close_owned(companion)
            try:
                pair.state_path.unlink(missing_ok=True)
            except OSError:
                self._on_status(f"could not remove companion state for {session_id}")

    async def set_enabled(self, app: Any, agent_id: str, *, enabled: bool) -> bool:
        """Persist an explicit enable/disable choice on an agent pane."""
        async with self._lock:
            agent = app.get_session_by_id(agent_id)
            if agent is None:
                return False
            inventory = await self._inventory(app)
            metadata = inventory.get(agent_id)
            if metadata is None or metadata.is_companion:
                return False
            if (
                agent_id not in self._pairs
                and agent_id not in self._rollover_pending
                and not metadata.disabled
            ):
                self._on_status(f"validating companion target {agent_id}")
                table = await asyncio.to_thread(self._read_process_table)
                kwargs: dict[str, Any] = {"table": table}
                if self._cwd_reader is not None:
                    kwargs["cwd_reader"] = self._cwd_reader
                detected = await asyncio.to_thread(self._detect, metadata.pane, **kwargs)
                if detected is None:
                    return False
            if enabled:
                owned = [
                    item
                    for item in inventory.values()
                    if item.is_companion and item.agent_session == agent_id
                ]
                if agent_id in self._rollover_pending and owned:
                    self._on_status(
                        f"disable companion for {agent_id} before re-enabling rollover recovery"
                    )
                    return False
                await agent.async_set_variable(DISABLED_VARIABLE, False)
                await agent.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                self._rollover_pending.pop(agent_id, None)
                self._restart_attempts.pop(agent_id, None)
                self._retry_after.pop(agent_id, None)
            else:
                await agent.async_set_variable(DISABLED_VARIABLE, True)
                pair = self._pairs.pop(agent_id, None)
                owned = [
                    item
                    for item in inventory.values()
                    if item.is_companion and item.agent_session == agent_id
                ]
                closed_ids: set[str] = set()
                for item in owned:
                    if await self._close_owned(item):
                        closed_ids.add(item.session_id)
                if pair is not None and pair.companion_id not in closed_ids:
                    companion = app.get_session_by_id(pair.companion_id)
                    if companion is not None:
                        self._manager_closing.add(pair.companion_id)
                        await companion.async_close(force=True)
                await agent.async_set_variable(COMPANION_SESSION_VARIABLE, "")
                self._rollover_pending.pop(agent_id, None)
                self._restart_attempts.pop(agent_id, None)
                self._retry_after.pop(agent_id, None)
                state_path = (
                    pair.state_path
                    if pair is not None
                    else opaque_state_path(self.state_dir, agent_id)
                )
                try:
                    state_path.unlink(missing_ok=True)
                except OSError:
                    self._on_status(f"could not remove companion state for {agent_id}")
            return True
