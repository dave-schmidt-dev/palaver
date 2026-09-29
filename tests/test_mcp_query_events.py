"""
Task 6.4: every read records a query event naming what it returned, including the two
ways recording one could silently go wrong (a hung-up sender, a malformed page).
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import select
import sqlite3

import pytest

from palaver.mcp import query_events, tools_read
from palaver.mcp import server as mcp_server
from palaver.observer import socket as writer_socket
from palaver.observer.socket import serve_request, single_writer
from palaver.store.migrate import connect
from tests._mcp_support import _call
from tests._mcp_support import short_store as short_store

# =============================================================================
# Task 6.4: query events, and the two ways recording one could go wrong
# =============================================================================
#
# Both failures here are invisible from the caller's side, which is why they
# get tests rather than a code comment. A read that quietly stopped recording
# leaves a table that looks like "no agent ever retrieved anything" — the same
# shape as a store nobody has queried. And a read that *failed* because the
# recording channel was down would be a bookkeeping outage presented as a
# missing answer, which is the worse of the two.


@contextlib.contextmanager
def _writer(db_path):
    """A listening daemon socket and its writable connection, drained by hand.

    No thread and no subprocess. A query event is fire-and-forget: `post`
    returns as soon as the bytes are handed to the kernel, so the connection
    sits in the listen backlog until something accepts it. Draining it
    afterwards on this thread is deterministic, and is itself a check that
    posting to a daemon busy inside a tick loses nothing.
    """
    conn = connect(db_path)
    try:
        with single_writer(db_path) as server:
            yield server, conn
    finally:
        conn.close()


def _drain(server, conn, *, limit=64, timeout=5.0):
    """Serve everything already queued, and never block on an empty queue.

    `serve_request` blocks in `accept()`, so a drain that simply looped a
    fixed number of times would hang rather than fail whenever the count was
    wrong. A hang reports nothing; `select` first turns the same bug into an
    assertion about how many events arrived.
    """
    served = 0
    while served < limit:
        ready, _, _ = select.select([server], [], [], timeout if served == 0 else 0.5)
        if not ready:
            return served
        if serve_request(server, conn):
            served += 1
    return served


def _query_events(db_path):
    """Every recorded event with the memory ids it links, oldest first."""
    conn = connect(db_path)
    try:
        return [
            {
                "id": event_id,
                "tool": tool,
                "scope_kind": scope_kind,
                "scope_value": scope_value,
                "result_count": result_count,
                "memory_ids": [
                    linked[0]
                    for linked in conn.execute(
                        "SELECT memory_id FROM query_event_memories "
                        "WHERE query_event_id = ? ORDER BY memory_id",
                        (event_id,),
                    )
                ],
            }
            for event_id, tool, scope_kind, scope_value, result_count in conn.execute(
                "SELECT id, tool, scope_kind, scope_value, result_count "
                "FROM query_events ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_a_read_records_a_query_event_naming_the_memories_it_returned(short_store):
    """The whole point of the task: what did the agent actually get back."""
    db_path, _ = short_store
    with _writer(db_path) as (server, conn):
        payload = json.loads(
            _call(
                mcp_server.build_server(db_path),
                "palaver_recall",
                {"scope": {"project": "demo"}},
            )
            .content[0]
            .text
        )
        assert _drain(server, conn) == 1, "the event never reached the daemon"

    returned = [memory["id"] for memory in payload["memories"]]
    assert returned, "the fixture returned no memories, so linking them proves nothing"

    events = _query_events(db_path)
    assert len(events) == 1, events
    assert events[0]["tool"] == "palaver_recall"
    assert events[0]["scope_kind"] == "project"
    assert events[0]["scope_value"] == "demo"
    assert events[0]["result_count"] == len(returned)
    assert events[0]["memory_ids"] == sorted(returned)


def test_the_query_event_count_equals_the_number_of_read_calls(short_store):
    """Not "at least one": a path that recorded once and then silently stopped
    would pass a single-call test and lose every retrieval after the first."""
    db_path, _ = short_store
    reads = 5
    with _writer(db_path) as (server, conn):
        server_under_test = mcp_server.build_server(db_path)
        for _ in range(reads):
            _call(server_under_test, "palaver_recall", {"scope": {"project": "demo"}})
        assert _drain(server, conn) == reads, "some events were dropped or coalesced"

    assert len(_query_events(db_path)) == reads


def test_a_read_still_answers_and_warns_when_the_query_event_cannot_be_posted(short_store, caplog):
    """The asymmetry with `palaver_correct`, asserted rather than assumed.

    A correction fails loudly when the daemon is down, because a human
    believing they fixed something they did not is the expensive outcome.
    A query event is Palaver's telemetry about itself: dropping one costs a
    row, and failing the read over it costs the agent its answer.
    """
    db_path, _ = short_store
    with caplog.at_level(logging.WARNING, logger="palaver.mcp.query_events"):
        payload = json.loads(
            _call(
                mcp_server.build_server(db_path),
                "palaver_recall",
                {"scope": {"project": "demo"}},
            )
            .content[0]
            .text
        )

    assert payload["memories"], "the read failed because a telemetry channel was down"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings, "the event vanished with nothing in the log to show for it"
    assert any("dropped" in record.getMessage() for record in warnings), [
        record.getMessage() for record in warnings
    ]
    assert _query_events(db_path) == [], "an event was recorded with no daemon to record it"


def test_recording_a_query_event_opens_no_second_write_connection(short_store, monkeypatch):
    """INV: this machine has exactly one writer, and telemetry does not get an
    exemption. Asserted behaviourally rather than by reading the module,
    because "it does not open one today" is what a future edit changes.
    """
    db_path, _ = short_store
    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    with _writer(db_path) as (server, conn):
        # Patched only around the read: the daemon's own writable connection
        # above is legitimate and is exactly the one this test says the MCP
        # side must borrow instead of duplicating.
        monkeypatch.setattr(sqlite3, "connect", spy)
        _call(
            mcp_server.build_server(db_path),
            "palaver_recall",
            {"scope": {"project": "demo"}},
        )
        monkeypatch.undo()
        assert _drain(server, conn) == 1, "the event never reached the daemon"

    # The positive control. Without it a spy on a name nothing calls would
    # record nothing and satisfy the assertion below vacuously.
    assert len(opened) == 1, f"expected exactly the read connection, got {opened}"
    assert "mode=ro" in opened[0], opened


def test_a_query_event_for_sessions_links_no_memory_ids(short_store):
    """`palaver_sessions` returns session ids. Recording those in a column
    every reader will take for a memory id is the confident-wrong-answer
    failure, and the foreign key would not catch it -- session and memory
    rowids overlap."""
    db_path, _ = short_store
    with _writer(db_path) as (server, conn):
        payload = json.loads(
            _call(
                mcp_server.build_server(db_path),
                "palaver_sessions",
                {"scope": {"project": "demo"}},
            )
            .content[0]
            .text
        )
        assert _drain(server, conn) == 1

    assert payload["sessions"], "no sessions came back, so this proves nothing"
    events = _query_events(db_path)
    assert len(events) == 1
    assert events[0]["tool"] == "palaver_sessions"
    assert events[0]["result_count"] == len(payload["sessions"])
    assert events[0]["memory_ids"] == []


def test_every_read_tool_puts_its_query_event_page_under_a_known_item_key(short_store):
    """`describe` records `result_count: None` for a page key it does not
    know -- honest, and useless. This is what turns that null into a failure
    here, when the tool is added, rather than a gap noticed in the data
    months later."""
    db_path, _ = short_store
    built = mcp_server.build_server(db_path)
    for tool in tools_read.READ_TOOLS:
        result = json.loads(_call(built, tool, {"scope": {"project": "demo"}}).content[0].text)
        described = query_events.describe(tool, result)
        assert described is not None, tool
        assert described["result_count"] is not None, (
            f"{tool} returns its page under none of {query_events.ITEM_KEYS}: {sorted(result)}"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool", ""),
        ("tool", None),
        ("scope_kind", "everything"),
        ("scope_value", ""),
        ("result_count", -1),
        ("result_count", True),
        ("memory_ids", "3"),
        ("memory_ids", [True]),
    ],
)
def test_a_malformed_query_event_is_refused_rather_than_recorded(short_store, field, value):
    """Validated as strictly as a correction, for a different reason.

    Nobody is watching this reply. A query event is posted fire-and-forget,
    so a bad one that got through would land as a wrong row with no caller
    to notice -- and telemetry nobody reads until months later is exactly
    where a wrong row survives. `True` appears twice above because `bool` is
    a subclass of `int`: a check written as `isinstance(x, int)` accepts it
    and records `result_count = 1` for a caller that sent `True`.
    """
    db_path, _ = short_store
    conn = connect(db_path)
    try:
        payload = {
            "op": "query",
            "tool": "palaver_recall",
            "scope_kind": "project",
            "scope_value": "demo",
            "result_count": 1,
            "memory_ids": [],
            field: value,
        }
        reply = writer_socket.apply_request(conn, payload)
    finally:
        conn.close()

    assert reply["ok"] is False, reply
    assert field in reply["detail"], reply
    assert _query_events(db_path) == []


def test_a_query_event_naming_a_memory_that_does_not_exist_is_refused(short_store):
    """The reason the foreign key is there at all. Memories are never deleted
    (INV-4), so it guards nothing against removal -- it catches an id
    recorded that names nothing, which is the only way this table can lie."""
    db_path, _ = short_store
    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn,
            {
                "op": "query",
                "tool": "palaver_recall",
                "scope_kind": "project",
                "scope_value": "demo",
                "result_count": 1,
                "memory_ids": [10_000],
            },
        )
    finally:
        conn.close()

    assert reply["ok"] is False, reply
    assert reply["error"] == "IntegrityError", reply
    # The parent row must not survive its children. A recorded event whose
    # memory list silently came back empty would read as "the agent was
    # shown nothing", which is a different and confident claim.
    assert _query_events(db_path) == []


class _HungUpClient:
    """A client that delivers its request and is gone before the reply."""

    def __init__(self, body):
        self._body = body

    def settimeout(self, _):
        pass

    def recv(self, _):
        body, self._body = self._body, b""
        return body

    def sendall(self, _):
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    def close(self):
        pass


class _HungUpServer:
    def __init__(self, body):
        self._body = body

    def accept(self):
        return _HungUpClient(self._body), None


def test_a_query_event_whose_sender_hung_up_is_still_recorded_and_still_counted(short_store):
    """Fire-and-forget means the sender is usually gone before the
    acknowledgement is written. The request was applied and committed before
    `sendall` was ever reached, so reporting it as unserved would make the
    daemon's own count of what it did disagree with what is in the database.
    """
    db_path, _ = short_store
    memory_id = json.loads(
        _call(mcp_server.build_server(db_path), "palaver_recall", {"scope": {"project": "demo"}})
        .content[0]
        .text
    )["memories"][0]["id"]

    request = (
        json.dumps(
            {
                "op": "query",
                "tool": "palaver_recall",
                "scope_kind": "project",
                "scope_value": "demo",
                "result_count": 1,
                "memory_ids": [memory_id],
            }
        ).encode()
        + b"\n"
    )

    conn = connect(db_path)
    try:
        served = serve_request(_HungUpServer(request), conn)
    finally:
        conn.close()

    assert served is True, "an applied request was reported as not served"
    events = _query_events(db_path)
    assert len(events) == 1, events
    assert events[0]["memory_ids"] == [memory_id]


def test_a_query_event_for_an_unknown_page_shape_counts_nothing_rather_than_zero(short_store):
    """`None` and `0` are different claims, and only one of them is true here.

    A read tool added later that returns its page under a key `ITEM_KEYS`
    does not list has *not* returned nothing -- nobody counted. Recording
    zero would put a confident wrong number in a table whose whole purpose
    is answering "was this ever retrieved" (INV-7).
    """
    described = query_events.describe(
        "palaver_future", {"scope": {"project": "demo"}, "widgets": [1, 2, 3]}
    )
    assert described is not None
    assert described["result_count"] is None, "an uncounted page was recorded as an empty one"
    assert described["memory_ids"] == []


def test_a_burst_of_query_events_larger_than_a_handful_all_reach_the_daemon(short_store):
    """The daemon accepts only between ticks, so every event posted during a
    30-second extraction waits in the listen backlog. A backlog sized for a
    handful of requests silently drops the rest -- and silently, from the
    poster's side, is the operative word: a refused connect is logged as a
    dropped event and the read carries on.
    """
    db_path, _ = short_store
    burst = 40
    with _writer(db_path) as (server, conn):
        built = mcp_server.build_server(db_path)
        for _ in range(burst):
            # Nothing accepts during this loop, by construction: the whole
            # point is that all 40 are still queued when it ends.
            _call(built, "palaver_recall", {"scope": {"project": "demo"}})
        assert _drain(server, conn, limit=burst * 2) == burst

    assert len(_query_events(db_path)) == burst
