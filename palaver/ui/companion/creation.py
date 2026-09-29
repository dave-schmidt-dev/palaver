"""
Every call into the live iTerm2 API for one companion: pane-variable metadata reads,
profile construction, and pane creation/sizing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from palaver.ui.connection import import_iterm2
from palaver.ui.pane_join import (
    PIN_VARIABLE,
    PaneJoin,
    PaneVariables,
    SupportedPaneProcess,
)

from .paths import companion_command, opaque_state_path
from .records import (
    AGENT_SESSION_VARIABLE,
    COMPANION_ROLE,
    COMPANION_SESSION_VARIABLE,
    DISABLED_VARIABLE,
    LAYOUT_SETTLE_ATTEMPTS,
    LAYOUT_SETTLE_DELAY,
    MIN_AGENT_ROWS,
    ROLE_VARIABLE,
    SCROLLBACK_LINES,
    SUMMARY_ROWS,
    CompanionPair,
    ReadMetadata,
    SessionMetadata,
)

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


class CreationMixin:
    """
    Every call into the live iTerm2 API for one companion: pane-variable metadata reads,
    profile construction, and pane creation/sizing.
    """

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
        # A five-row summary plus at least one agent row is the only local
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
