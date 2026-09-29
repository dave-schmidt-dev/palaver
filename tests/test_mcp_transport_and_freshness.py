"""
The app's transport type, per-call connection lifecycle, the read-only connection
string, and freshness/writer-liveness reporting (task 6.3).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from palaver.mcp import pagination
from palaver.mcp import server as mcp_server
from tests._mcp_support import _call, _conn
from tests._mcp_support import short_store as short_store
from tests._mcp_support import store as store

# =============================================================================
# The transport, and the connection underneath it
# =============================================================================


def test_the_server_transport_is_the_streamable_http_session_manager(store):
    """Done-when: the transport is Streamable HTTP, not stdio.

    stdio would be subprocess-per-client, which is several processes writing
    one SQLite file — the opposite of the single-writer property the memory
    layer rests on.
    """
    db_path, _ = store
    server, app = mcp_server.build_app(db_path)
    assert type(server.session_manager) is StreamableHTTPSessionManager
    assert type(app).__name__ == "Starlette"


def test_the_session_manager_does_not_exist_until_the_app_is_built(store):
    """Pins the SDK's lazy-init contract rather than depending on it by luck."""
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    with pytest.raises(RuntimeError):
        _ = server.session_manager


def test_the_server_binds_loopback_only(store):
    """INV-9 permits one local MCP listener, not a remotely reachable one."""
    assert mcp_server.DEFAULT_HOST == "127.0.0.1"


def test_a_tool_connection_refuses_writes(store):
    """Read-only at the SQLite layer, not merely by how the tools are written."""
    db_path, _ = store
    conn = _conn(db_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM memories")
    conn.close()


def test_a_missing_database_is_named_rather_than_reported_as_empty(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        mcp_server.open_readonly(tmp_path / "absent.db")
    assert "palaver observe" in str(excinfo.value)


class _ClosingConnection(sqlite3.Connection):
    """A connection that records whether `close()` was actually called.

    Probing a leaked connection by calling `execute()` on it does not work
    here and is worse than useless: the SDK runs a synchronous tool in a
    worker thread, so a connection opened inside the call belongs to that
    thread, and `execute()` from the test's thread raises
    `sqlite3.ProgrammingError` whether or not it was ever closed. That
    assertion passes identically against an implementation that closes and
    one that leaks — it reads as a test of lifecycle and is a test of thread
    affinity. Recording the call is the only thing that separates them.
    """

    closed = False

    def close(self) -> None:
        self.closed = True
        super().close()


def test_each_tool_call_opens_and_closes_its_own_connection(store):
    """No handle outlives a call, so a restarted daemon is never read stale."""
    db_path, _ = store
    opened: list[_ClosingConnection] = []

    def _tracking(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, factory=_ClosingConnection)
        opened.append(conn)
        return conn

    server = mcp_server.build_server(db_path, connect=_tracking)
    _call(server, "palaver_recall", {"scope": {"project": "demo"}})
    _call(server, "palaver_recall", {"scope": {"project": "demo"}})

    assert len(opened) == 2
    assert [conn.closed for conn in opened] == [True, True]


def test_a_connection_is_closed_even_when_the_tool_raises(store):
    """A refused scope must not leak the handle it opened to find out."""
    db_path, _ = store
    opened: list[_ClosingConnection] = []

    def _tracking(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, factory=_ClosingConnection)
        opened.append(conn)
        return conn

    server = mcp_server.build_server(db_path, connect=_tracking)
    with pytest.raises(Exception):
        _call(server, "palaver_recall", {"scope": {}})

    assert [conn.closed for conn in opened] == [True]


def test_each_registered_tool_answers_as_itself(store):
    """Guards the closure: a late-bound handler makes every tool the last one."""
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    recalled = _call(server, "palaver_recall", {"scope": {"project": "demo"}})
    listed = _call(server, "palaver_sessions", {"scope": {"project": "demo"}})
    assert "memories" in recalled.content[0].text
    assert "session_key" in listed.content[0].text


# =============================================================================
# Task 6.3: the MCP process reads, the daemon writes, and every read says when
# =============================================================================


def test_the_mcp_processes_connection_string_opens_the_store_read_only(store, monkeypatch):
    """Asserted on the URI the production factory actually builds.

    Not on a connection a test constructed: that would prove the test knows
    how to spell `mode=ro`. This captures what `open_readonly` passes to
    sqlite3, which is the string the server runs with.
    """
    db_path, _ = store
    seen = []
    real = sqlite3.connect

    def capture(target, *args, **kwargs):
        seen.append(target)
        return real(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", capture)
    mcp_server.open_readonly(db_path).close()
    assert seen and all("mode=ro" in target for target in seen), seen


def test_the_read_only_connection_really_refuses_a_write(store):
    """The positive control for the string check above.

    `mode=ro` appearing in a URI proves the spelling, not the behaviour --
    a typo'd parameter name is silently ignored by SQLite, leaving a fully
    writable connection whose URI still contains the text being asserted on.
    """
    db_path, seeded = store
    conn = mcp_server.open_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE memories SET statement = 'x' WHERE id = ?", (seeded["memory_id"],))
    finally:
        conn.close()


@pytest.mark.parametrize("tool", ["palaver_recall", "palaver_sessions"])
def test_every_read_tool_reports_when_the_store_was_last_written_and_who_is_watching(
    short_store, tool
):
    """A crashed daemon and a quiet one are otherwise identical from here.

    Same memories, same timestamps, same shape. A reader who cannot tell
    them apart will read a two-day-old store as current, which is INV-7's
    failure -- and the likelier reading, since a store that answers at all
    looks healthy.

    `short_store`, not `store`: on a path too deep for a socket the honest
    answer is `None`, and a test that accepted either would not be checking
    that a *probeable* absent daemon reads as absent.
    """
    db_path, _ = short_store
    server = mcp_server.build_server(db_path)
    result = _call(server, tool, {"scope": {"project": "demo"}})
    payload = json.loads(result.content[0].text)

    assert "observed_at" in payload, "no freshness stamp, so a stale answer looks current"
    assert "daemon_running" in payload
    assert payload["daemon_running"] is False, "nothing is serving this test store"


@pytest.mark.parametrize("tool", ["palaver_recall", "palaver_sessions"])
def test_a_deep_store_with_no_writer_reports_stopped_not_unknown(store, tool):
    """A path-length-independent lock probe distinguishes no writer from one."""
    db_path, _ = store
    payload = json.loads(
        _call(mcp_server.build_server(db_path), tool, {"scope": {"project": "demo"}})
        .content[0]
        .text
    )
    assert payload["daemon_running"] is False


def test_the_freshness_stamp_is_the_newest_memory_not_the_time_of_the_call(store):
    """`observed_at` answers "what has this store seen", not "what time is it".

    A wall-clock stamp would advance on every call and would therefore
    describe a dead store as freshly observed -- the precise confusion the
    field exists to prevent.
    """
    db_path, seeded = store
    conn = _conn(db_path)
    try:
        expected = conn.execute(
            "SELECT created_at FROM memories WHERE id = ?", (seeded["memory_id"],)
        ).fetchone()[0]
    finally:
        conn.close()

    server = mcp_server.build_server(db_path)
    payload = json.loads(
        _call(server, "palaver_recall", {"scope": {"project": "demo"}}).content[0].text
    )
    assert payload["observed_at"] == expected


def test_the_freshness_keys_are_counted_inside_the_byte_budget(store):
    """Keys merged after `paginate` returns are outside the bound it asserted.

    The budget has 25% headroom, so two extra keys would never actually
    overflow -- which is exactly why this needs asserting rather than
    trusting. The claim `paginate` makes is that the response it returns is
    the one it measured, and a caller bolting fields on afterwards quietly
    makes that false.
    """
    scope = {"project": "demo"}
    extras = {"observed_at": "2026-08-15T00:00:00Z", "daemon_running": False}
    # A small explicit budget and small rows, so the ~60 bytes the extras cost
    # is several rows rather than a rounding error. At the production budget
    # the same extras hide inside one 4KB row's slack, and the count would be
    # identical whether they were charged for or not.
    budget = 2000
    rows = [(index, {"s": "x" * 20}) for index in range(1, 200)]

    without = pagination.paginate(rows, scope=scope, items_key="memories", budget=budget)
    with_extras = pagination.paginate(
        rows, scope=scope, items_key="memories", extra=extras, budget=budget
    )

    assert pagination.wire_size(with_extras) <= budget
    assert with_extras["observed_at"] == extras["observed_at"]
    # The extras cost real bytes, so the page they leave room for is smaller.
    # Equal counts would mean the overhead was never charged for.
    assert len(with_extras["memories"]) < len(without["memories"])
