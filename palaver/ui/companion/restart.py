"""
Restart backoff, liveness grace, and close bookkeeping for one agent/companion pair.
"""

from __future__ import annotations

import asyncio
from typing import Any

from palaver.ui.pane_join import (
    ProcessTable,
)

from .records import (
    AGENT_EXIT_GRACE,
    COMPANION_SESSION_VARIABLE,
    INITIAL_RESTART_BACKOFF,
    MAX_RESTART_BACKOFF,
    SessionMetadata,
)


class RestartMixin:
    """
    Restart backoff, liveness grace, and close bookkeeping for one agent/companion pair.
    """

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

    def _record_restart_failure(self, agent_id: str) -> None:
        attempts = self._restart_attempts.get(agent_id, 0) + 1
        self._restart_attempts[agent_id] = attempts
        delay = min(INITIAL_RESTART_BACKOFF * (2 ** (attempts - 1)), MAX_RESTART_BACKOFF)
        self._retry_after[agent_id] = self._clock() + delay

    def _forget_agent(self, agent_id: str) -> None:
        """Drop every per-agent bookkeeping entry for a pair that is over."""
        self._pairs.pop(agent_id, None)
        self._rollover_pending.pop(agent_id, None)
        self._restart_attempts.pop(agent_id, None)
        self._retry_after.pop(agent_id, None)
        self._agent_missing_since.pop(agent_id, None)

    async def _agent_has_exited(self, agent: SessionMetadata, table: ProcessTable) -> bool:
        """Whether a paired pane has had no supported agent for the grace period.

        A single miss proves nothing: `read_process_table` returns an empty
        table when `ps` fails, and an agent restarted straight after exiting
        is one session ending, not the pair ending. Only a first miss that is
        still a miss `AGENT_EXIT_GRACE` seconds later is treated as an exit.
        """
        kwargs: dict[str, Any] = {"table": table}
        if self._cwd_reader is not None:
            kwargs["cwd_reader"] = self._cwd_reader
        detected = await asyncio.to_thread(self._detect, agent.pane, **kwargs)
        if detected is not None:
            self._agent_missing_since.pop(agent.session_id, None)
            return False
        now = self._clock()
        first_miss = self._agent_missing_since.setdefault(agent.session_id, now)
        return now - first_miss >= AGENT_EXIT_GRACE

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
