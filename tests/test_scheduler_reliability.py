"""
One writer connection is ever live across ticks, and a failed extraction retries durably
instead of losing the session.
"""

import sqlite3
from datetime import timedelta

import pytest

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import CursorStore
from palaver.observer.daemon import (
    DaemonNotStartedError,
)
from tests._test_scheduler_support import NOW, RecordingExtractor, _daemon, _human, _write_store


class FailingExtractor:
    """An extractor that always raises, standing in for an unreachable server."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.exc = RuntimeError("model server unreachable") if exc is None else exc

    def __call__(self, work, *, conn, on_status) -> None:
        self.calls.append(work.ref.session_key)
        raise self.exc


class _ConnectionSpy:
    """Wraps `sqlite3.connect` and reports how many connections are still live.

    Liveness is probed rather than tracked: a closed `sqlite3.Connection`
    raises `ProgrammingError` on any statement, which is a fact about the
    connection itself and cannot drift from what the code under test
    actually did with it.
    """

    def __init__(self, real) -> None:
        self._real = real
        self.opened: list[sqlite3.Connection] = []

    def __call__(self, *args, **kwargs) -> sqlite3.Connection:
        conn = self._real(*args, **kwargs)
        self.opened.append(conn)
        return conn

    def live(self) -> int:
        count = 0
        for conn in self.opened:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue
            count += 1
        return count


# --- one writer --------------------------------------------------------------


def test_two_ticks_never_open_two_write_connections(tmp_path, monkeypatch):
    """Across two ticks exactly one connection is live, and the ticks open none.

    The spy patches `sqlite3.connect` itself rather than the daemon's
    injected factory, so `migrate()`'s own connections are inside the
    measurement. The hand-opened connection near the end is the positive
    control: without it, `live() == 1` could equally mean "the spy sees
    nothing".
    """
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    spy = _ConnectionSpy(sqlite3.connect)
    monkeypatch.setattr(sqlite3, "connect", spy)

    daemon = _daemon(tmp_path, sample_root, RecordingExtractor())
    daemon.start()
    opened_after_start = len(spy.opened)
    assert spy.live() == 1, "migration left a connection open beside the daemon's writer"

    daemon.start()  # idempotent: a second start must not open a second writer
    assert len(spy.opened) == opened_after_start

    daemon.tick(now=NOW)
    assert spy.live() == 1
    daemon.tick(now=NOW)
    assert spy.live() == 1
    assert len(spy.opened) == opened_after_start, "a tick opened its own connection"

    # Positive control: the spy can see a second live connection.
    extra = sqlite3.connect(str(daemon.db_path))
    assert spy.live() == 2
    extra.close()
    assert spy.live() == 1

    daemon.close()
    assert spy.live() == 0


def test_tick_before_start_raises_rather_than_opening_a_connection(tmp_path):
    """No implicit connect: a tick without `start()` is a named error."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    daemon = _daemon(tmp_path, sample_root, RecordingExtractor())
    with pytest.raises(DaemonNotStartedError):
        daemon.tick(now=NOW)


# --- at-least-once: a failed extraction is retried ---------------------------


def test_failed_extraction_leaves_the_cursor_and_uses_durable_backoff(tmp_path):
    """A failure retries later, and restart does not erase that backoff.

    The swap to a succeeding extractor at the end is the positive control:
    without it, "the cursor did not advance" would also pass on a daemon
    that never advances cursors at all.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)

    failing = FailingExtractor()
    daemon = _daemon(tmp_path, sample_root, failing)
    with daemon:
        first = daemon.tick(now=NOW)
        assert first.extracted == ()
        assert first.failed == ((key, "RuntimeError: model server unreachable"),)
        assert cursors.load(key).offset == 0

        second = daemon.tick(now=NOW)
        assert len(second.plan.scheduled) == 1
        assert second.deferred == (key,)
        assert failing.calls == [key]

        # A new daemon shares the durable cursor/backoff root, so a restart
        # cannot turn a down model into a hot loop.
        daemon.close()
        daemon = _daemon(tmp_path, sample_root, failing)
        daemon.start()
        deferred_after_restart = daemon.tick(now=NOW)
        assert deferred_after_restart.deferred == (key,)
        assert failing.calls == [key]

        # Positive control: the cursor does advance once extraction succeeds.
        succeeding = RecordingExtractor()
        daemon.extractor = succeeding
        third = daemon.tick(now=NOW + timedelta(seconds=30))
        assert third.extracted == (key,)
        assert cursors.load(key).offset > 0

        fourth = daemon.tick(now=NOW)
        daemon.close()

    assert fourth.plan.scheduled == ()
    assert succeeding.calls == [key]
