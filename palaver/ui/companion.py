"""Lifecycle ownership for Palaver's per-agent iTerm2 companion panes.

The controller owns only panes carrying Palaver's exact marker.  Agent panes
are observed and may be split once; they are never sent input or closed.  The
only layout write Palaver makes is that split and the row division between the
observed pane and its own companion, and it is written so both of the tab's
totals are unchanged: iTerm2's one pane-sizing call rewrites the entire tab, so
anything less careful resizes the user's window.  All mutating operations are
serialized so the new-session and layout monitors cannot turn one split into a
recursive chain of companions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palaver.ui.companion_state import (
    CompanionState,
    JoinState,
    atomic_write_state,
)
from palaver.ui.connection import import_iterm2
from palaver.ui.pane_join import (
    PIN_VARIABLE,
    PaneJoin,
    PaneVariables,
    ProcessTable,
    SupportedPaneProcess,
    detect_supported_process,
    join_pane,
    read_process_table,
)

ROLE_VARIABLE = "user.palaver_role"
AGENT_SESSION_VARIABLE = "user.palaver_agent_session"
COMPANION_SESSION_VARIABLE = "user.palaver_companion_session"
DISABLED_VARIABLE = "user.palaver_companion_disabled"
COMPANION_ROLE = "companion-v1"

SUMMARY_ROWS = 10
MIN_AGENT_ROWS = 1
LAYOUT_SETTLE_ATTEMPTS = 3
LAYOUT_SETTLE_DELAY = 0.05
SCROLLBACK_LINES = 100
INITIAL_RESTART_BACKOFF = 2.0
MAX_RESTART_BACKOFF = 60.0


def _no_status(_message: str) -> None:
    """Default progress sink."""


@dataclass(frozen=True)
class SessionMetadata:
    """One iTerm session plus the variables used by lifecycle decisions."""

    session: Any
    tab: Any
    pane: PaneVariables
    role: str | None = None
    agent_session: str | None = None
    companion_session: str | None = None
    disabled: bool = False

    @property
    def session_id(self) -> str:
        """Return iTerm2's stable-within-run session identifier."""
        return self.pane.pane_id

    @property
    def is_companion(self) -> bool:
        """Whether this pane bears Palaver's exact ownership marker."""
        return self.role == COMPANION_ROLE


@dataclass(frozen=True)
class CompanionPair:
    """A reciprocal live agent/companion pair."""

    agent_id: str
    companion_id: str
    state_path: Path


@dataclass(frozen=True)
class ReconcileResult:
    """Observable result of one reconciliation pass."""

    pairs: tuple[CompanionPair, ...]
    created: tuple[str, ...]
    closed: tuple[str, ...]
    refused: tuple[str, ...]


ReadMetadata = Callable[[Any, Any], Awaitable[SessionMetadata | None]]
WriteInitialState = Callable[[Path, SupportedPaneProcess, PaneJoin | None], None]


def opaque_state_path(state_dir: Path, agent_id: str) -> Path:
    """Return a non-identifying state filename for one iTerm agent pane."""
    digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()
    return state_dir / f"{digest}.json"


def companion_command(state_path: Path, *, executable: str | None = None) -> str:
    """Build the custom command without shell interpolation hazards."""
    python = sys.executable if executable is None else executable
    return shlex.join([python, "-m", "palaver.ui.companion_render", "--state", str(state_path)])


METADATA_VARIABLES = (
    "jobPid",
    "jobName",
    "path",
    PIN_VARIABLE,
    ROLE_VARIABLE,
    AGENT_SESSION_VARIABLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
)


def make_metadata_reader(connection: Any) -> ReadMetadata:
    """Read all process and ownership variables in one iTerm RPC."""
    import_iterm2()
    import iterm2.api_pb2  # noqa: PLC0415 - optional UI dependency
    import iterm2.rpc  # noqa: PLC0415 - optional UI dependency

    ok = iterm2.api_pb2.VariableResponse.Status.Value("OK")

    async def read_metadata(session: Any, tab: Any) -> SessionMetadata | None:
        result = await iterm2.rpc.async_variable(
            connection, session.session_id, [], list(METADATA_VARIABLES)
        )
        response = result.variable_response
        if response.status != ok or len(response.values) < 3:
            return None
        decoded = [_decode_variable(item) for item in response.values]
        decoded.extend([None] * (len(METADATA_VARIABLES) - len(decoded)))
        job_pid, job_name, path, pin, role, agent_id, companion_id, disabled = decoded
        try:
            parsed_pid = int(job_pid) if int(job_pid) > 0 else None
        except TypeError, ValueError:
            parsed_pid = None
        pane = PaneVariables(
            pane_id=session.session_id,
            job_pid=parsed_pid,
            job_name=job_name if isinstance(job_name, str) else None,
            path=path if isinstance(path, str) else None,
            pin=pin if isinstance(pin, str) else None,
        )
        return SessionMetadata(
            session=session,
            tab=tab,
            pane=pane,
            role=role if isinstance(role, str) else None,
            agent_session=agent_id if isinstance(agent_id, str) and agent_id else None,
            companion_session=(
                companion_id if isinstance(companion_id, str) and companion_id else None
            ),
            disabled=disabled is True,
        )

    return read_metadata


def _decode_variable(raw: str) -> object:
    try:
        return json.loads(raw)
    except TypeError, ValueError:
        return None


def build_companion_profile(command: str):
    """Build session-local iTerm profile overrides for a quiet summary pane."""
    iterm2 = import_iterm2()
    profile = iterm2.LocalWriteOnlyProfile()
    profile.set_use_custom_command("Yes")
    profile.set_command(command)
    profile.set_close_sessions_on_end(False)
    profile.set_prompt_before_closing(False)
    profile.set_unlimited_scrollback(False)
    profile.set_scrollback_lines(SCROLLBACK_LINES)
    profile.set_silence_bell(True)
    profile.set_send_bell_alert(False)
    profile.set_flashing_bell(False)
    profile.set_visual_bell(False)
    profile.set_use_custom_window_title(True)
    profile.set_custom_window_title("Palaver")
    return profile


def write_initial_state(
    path: Path,
    detected: SupportedPaneProcess,
    joined: PaneJoin | None,
) -> None:
    """Seed the renderer transport before its process can start reading."""
    exact = joined is not None and joined.session_key is not None
    atomic_write_state(
        path,
        CompanionState(
            producer_updated_at=time.time(),
            project=detected.cwd.name,
            source=detected.source,
            status="UNKNOWN",
            join_state=JoinState.JOINED if exact else JoinState.UNJOINED,
            detail=None if exact else "Agent detected; waiting for an exact session join",
        ),
    )


class CompanionController:
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

        table: ProcessTable | None = None
        for agent_id, agent in sorted(agents.items()):
            if agent_id in pairs or agent.disabled:
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
            if table is None:
                self._on_status("reading process table for companion reconciliation")
                table = await asyncio.to_thread(self._read_process_table)
            kwargs: dict[str, Any] = {"table": table}
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

    async def _rebind_restarted_companion(self, app: Any, agent_id: str) -> str | None:
        """Refresh iTerm and bind an exact marked pane after GUID rollover."""
        self._on_status(f"refreshing companion identity after restart for {agent_id}")
        try:
            await app.async_refresh()
        except Exception:
            self._on_status(f"could not refresh companion identity for {agent_id}")
            return None
        inventory = await self._inventory(app)
        agent = inventory.get(agent_id)
        owned = sorted(
            (
                item
                for item in inventory.values()
                if item.is_companion and item.agent_session == agent_id
            ),
            key=lambda item: item.session_id,
        )
        if agent is None or len(owned) != 1:
            self._on_status(f"could not uniquely reacquire restarted companion for {agent_id}")
            return None
        new_id = owned[0].session_id
        await agent.session.async_set_variable(COMPANION_SESSION_VARIABLE, new_id)
        old_id = self._rollover_pending.pop(agent_id, None)
        self._on_status(f"companion identity changed for {agent_id}: {old_id} -> {new_id}")
        return new_id

    @staticmethod
    def _tab_for(app: Any, tab: Any, session_id: str | None = None) -> Any | None:
        """Locate the live tab matching `tab`, surviving app refreshes."""
        tab_id = getattr(tab, "tab_id", None)
        for candidate in getattr(app, "terminal_windows", None) or ():
            for item in getattr(candidate, "tabs", None) or ():
                if tab_id is not None and getattr(item, "tab_id", None) == tab_id:
                    return item
                if session_id is not None and any(
                    getattr(s, "session_id", None) == session_id
                    for s in getattr(item, "sessions", ()) or ()
                ):
                    return item
        return tab

    @staticmethod
    def _window_for(app: Any, tab: Any, session: Any) -> Any | None:
        """Locate the window owning `tab`, preferring the session's own link."""
        window = getattr(session, "window", None)
        if window is not None:
            return window
        # The session delegate resolves by object identity, so a session read
        # from a tab the app has since rebuilt reports no window. Match on the
        # tab id instead, which survives the rebuild.
        tab_id = getattr(tab, "tab_id", None)
        for candidate in getattr(app, "terminal_windows", None) or ():
            for item in getattr(candidate, "tabs", None) or ():
                if tab_id is not None and getattr(item, "tab_id", None) == tab_id:
                    return candidate
        return None

    async def _read_frame(self, window: Any) -> Any | None:
        """Return the window's current frame, or None if it cannot be read."""
        if window is None:
            return None
        try:
            return await window.async_get_frame()
        except Exception:
            return None

    async def _size_companion(self, app: Any, agent: SessionMetadata, companion: Any) -> None:
        """Give a freshly split companion `SUMMARY_ROWS` without moving the window.

        `Tab.async_update_layout` is the only pane-sizing call iTerm2's Python
        API offers, and it is a whole-tab write: it serializes every session in
        the tab and sends each one's `preferred_size`. Two things measured
        against a live iTerm shape this method.

        First, the library caches `preferred_size` once, when it constructs the
        `Session`, and never refreshes it -- `Session.update_from` copies
        `grid_size` and leaves `preferred_size` alone -- so a tab Palaver has
        watched across a window resize pushes long-dead sizes back at iTerm.
        Every cached size is therefore resynced from live geometry first, and
        the only change requested is how the agent and its companion divide the
        rows they already occupy.

        Second, that is not enough on its own: the write shrinks the window
        regardless, even when it requests exactly the sizes already on screen,
        because the layout protobuf does not describe the pane title bars and
        dividers iTerm draws around them. The shrink lands on every tab in the
        window, not just this one, and repeats per companion. So the frame is
        captured beforehand and put back afterwards, unconditionally -- which
        also returns the rows the shrink took, leaving the companion on exactly
        `SUMMARY_ROWS`. INV-2 forbids the alternative: only the marked companion
        is Palaver's to resize, never the user's window.

        Args:
            app: The `iterm2.App`, used to locate the window and to refresh.
            agent: The observed pane the companion was split from.
            companion: The session `async_split_pane` returned.
        """
        iterm2 = import_iterm2()
        tab = self._tab_for(app, agent.tab, agent.session_id)
        panes: dict[str, Any] = {}
        summary = observed = None
        for attempt in range(LAYOUT_SETTLE_ATTEMPTS):
            if attempt:
                await asyncio.sleep(LAYOUT_SETTLE_DELAY)
            try:
                await app.async_refresh()
            except Exception:
                self._on_status(f"could not refresh layout before sizing {agent.session_id}")
            tab = self._tab_for(app, agent.tab, agent.session_id)
            panes = {item.session_id: item for item in getattr(tab, "sessions", None) or ()}
            summary = panes.get(companion.session_id)
            observed = panes.get(agent.session_id)
            if summary is not None and observed is not None:
                break
        else:
            # The split never reached the tab tree. iTerm's own even division
            # stands: a layout write from here would describe a tab that does
            # not exist, which is precisely how a window gets resized. The
            # companion stays at half the pane rather than SUMMARY_ROWS.
            self._on_status(f"companion for {agent.session_id} never joined the layout")
            return
        rows = observed.grid_size.height + summary.grid_size.height - SUMMARY_ROWS
        if rows < MIN_AGENT_ROWS:
            return

        window = self._window_for(app, tab, observed)
        before = await self._read_frame(window)
        if before is None:
            # Measured: every whole-tab write shrinks the window, including one
            # requesting exactly the sizes already on screen, because the layout
            # protobuf does not describe the pane title bars and dividers iTerm
            # draws around them. With no frame to put back, the only choice that
            # honours INV-2 is to leave iTerm's own even split alone.
            self._on_status(f"no window frame to restore for {agent.session_id}; left unsized")
            return

        for item in panes.values():
            item.preferred_size = iterm2.Size(item.grid_size.width, item.grid_size.height)
        summary.preferred_size = iterm2.Size(summary.grid_size.width, SUMMARY_ROWS)
        observed.preferred_size = iterm2.Size(observed.grid_size.width, rows)
        await tab.async_update_layout()

        # Unconditionally, never on a detected move: iTerm applies the shrink
        # after it answers the layout call, so reading the frame back races it
        # and usually reports the old one. Putting the captured frame back also
        # returns the rows the shrink took, from this tab and from every other
        # tab in the window, while iTerm keeps the division just requested --
        # which is how the companion ends up on SUMMARY_ROWS exactly.
        try:
            await window.async_set_frame(before)
        except Exception:
            # Fullscreen windows refuse a frame set. Nothing further is safe.
            self._on_status(f"could not restore the window frame for {agent.session_id}")

    async def _create(
        self,
        app: Any,
        agent: SessionMetadata,
        detected: SupportedPaneProcess,
        joined: PaneJoin | None,
    ) -> CompanionPair | None:
        # A ten-row summary plus at least one agent row is the only local
        # precondition. iTerm owns all other layout constraints and may still
        # refuse the split, which is handled as a per-pane failure below.
        session_getter = getattr(app, "get_session_by_id", None)
        live_session = (
            session_getter(agent.session_id) if session_getter else None
        ) or agent.session
        if live_session.grid_size.height < SUMMARY_ROWS + MIN_AGENT_ROWS:
            return None
        state_path = opaque_state_path(self.state_dir, agent.session_id)
        companion = None
        try:
            self._on_status(f"writing initial companion state for {agent.session_id}")
            await asyncio.to_thread(self._write_initial_state, state_path, detected, joined)
            command = companion_command(state_path)
            profile = self._profile_builder(command)
            companion = await live_session.async_split_pane(
                vertical=False, before=True, profile_customizations=profile
            )
            self._pending.add(companion.session_id)
            await companion.async_set_variable(ROLE_VARIABLE, COMPANION_ROLE)
            await companion.async_set_variable(AGENT_SESSION_VARIABLE, agent.session_id)
            await live_session.async_set_variable(COMPANION_SESSION_VARIABLE, companion.session_id)
            await live_session.async_set_variable(DISABLED_VARIABLE, False)
            await self._size_companion(app, agent, companion)

            # Splitting activates the new pane. Restore only when it is still
            # active in this same tab; never steal focus after the user moved.
            active_tab = self._tab_for(app, agent.tab, agent.session_id)
            if (
                active_tab.current_session is not None
                and active_tab.current_session.session_id == companion.session_id
            ):
                await live_session.async_activate(select_tab=False, order_window_front=False)
            self._restart_attempts.pop(agent.session_id, None)
            self._retry_after.pop(agent.session_id, None)
            return CompanionPair(agent.session_id, companion.session_id, state_path)
        except asyncio.CancelledError:
            if companion is not None:
                self._manager_closing.add(companion.session_id)
                try:
                    await companion.async_close(force=True)
                except Exception:
                    pass
            try:
                await live_session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
            except Exception:
                pass
            try:
                state_path.unlink(missing_ok=True)
            except OSError:
                self._on_status(f"could not remove partial companion state for {agent.session_id}")
            raise
        except Exception:
            self._record_restart_failure(agent.session_id)
            if companion is not None:
                self._manager_closing.add(companion.session_id)
                try:
                    await companion.async_close(force=True)
                except Exception:
                    pass
            try:
                await live_session.async_set_variable(COMPANION_SESSION_VARIABLE, "")
            except Exception:
                pass
            try:
                state_path.unlink(missing_ok=True)
            except OSError:
                self._on_status(f"could not remove partial companion state for {agent.session_id}")
            self._on_status(f"failed to initialize companion for {agent.session_id}")
            return None
        finally:
            if companion is not None:
                self._pending.discard(companion.session_id)

    def _record_restart_failure(self, agent_id: str) -> None:
        attempts = self._restart_attempts.get(agent_id, 0) + 1
        self._restart_attempts[agent_id] = attempts
        delay = min(INITIAL_RESTART_BACKOFF * (2 ** (attempts - 1)), MAX_RESTART_BACKOFF)
        self._retry_after[agent_id] = self._clock() + delay

    async def _close_owned(self, companion: SessionMetadata) -> bool:
        """Close only an exact-marker companion, never an observed pane."""
        if not companion.is_companion:
            return False
        self._manager_closing.add(companion.session_id)
        try:
            await companion.session.async_close(force=True)
        except Exception:
            self._manager_closing.discard(companion.session_id)
            return False
        return True

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
            if session_id in inventory:
                # The PTY ended but its pane remains visible; it was not torn down.
                return
            self._pairs.pop(session_id, None)
            companion = inventory.get(pair.companion_id)
            if companion is not None and companion.is_companion:
                # Pair membership proves this is the exact companion. Agent
                # session ids are never passed to async_close.
                self._manager_closing.add(pair.companion_id)
                await companion.session.async_close(force=True)
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


__all__ = [
    "AGENT_SESSION_VARIABLE",
    "COMPANION_ROLE",
    "COMPANION_SESSION_VARIABLE",
    "DISABLED_VARIABLE",
    "ROLE_VARIABLE",
    "CompanionController",
    "CompanionPair",
    "ReconcileResult",
    "SessionMetadata",
    "build_companion_profile",
    "companion_command",
    "opaque_state_path",
]
