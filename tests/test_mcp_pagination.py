"""
Task 6.2: the recall/palaver_sessions pagination bound is measured in wire bytes,
cursors are scope-bound and version-checked, and exhaustive paging returns every memory
exactly once -- including over the real transport.
"""

from __future__ import annotations

import asyncio
import base64
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from palaver.cli import mcp as mcp_cli
from palaver.mcp import pagination, tools_read
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.scope import read_memories
from palaver.memory.write import write_memory
from palaver.store.migrate import connect
from tests._mcp_support import _SERVE_SUBPROCESS, _conn, _endpoint_line, _seed, _with_timeout
from tests._mcp_support import store as store

# =============================================================================
# Task 6.2: the bound is bytes on the wire, and the cut is keyset
#
# The plan built this task on `mcp`'s 4 MiB `max_request_body_size`. Measured
# against a live client, that constant guards **incoming POST bodies** and
# never sees a tool result. What truncates a recall is `httpx2`'s
# `DEFAULT_MAX_EVENT_SIZE_BYTES` — 1 MiB, client-side, per SSE event — and
# over it the caller gets `MCPError: SSE stream ended without a response`,
# which names neither size nor remedy. Hence a bound asserted here, before
# the response leaves the tool. `palaver/mcp/pagination.py` records the
# measurements.
# =============================================================================


def _fill_memories(
    db_path: Path, seeded: dict, count: int, size: int, *, session_index: int = 0
) -> list[str]:
    """Write `count` memories of roughly `size` bytes each, and return them.

    The prose is generated, never copied: INV-9 treats a committed fixture
    carrying real session text as an export that cannot be recalled, and a
    multi-megabyte one would be the largest such export in the repo.

    Numbering continues from what is already stored, so a caller that fills
    in several rounds — as the exhaustion test does, writing between pages —
    gets distinct statements instead of a second `memory 0`.
    """
    conn = connect(db_path)
    chunk_id = conn.execute("SELECT id FROM transcript_chunks ORDER BY id LIMIT 1").fetchone()[0]
    start = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    written = []
    try:
        for index in range(start, start + count):
            # Quotes and backslashes are what escaping doubles, so prose
            # without them would understate the wire size it produces.
            statement = f'memory {index}: the caller said "keep it" \\ ' + "detail " * (
                max(size // 7, 1)
            )
            write_memory(
                conn,
                project_id=seeded["project_id"],
                session_id=seeded["session_ids"][session_index],
                statement=statement,
                origin="observer",
                tier=4,
                evidence=[
                    EvidenceAnchor(start_offset=0, end_offset=8, transcript_chunk_id=chunk_id)
                ],
            )
            written.append(statement)
        conn.commit()
    finally:
        conn.close()
    return written


def test_recall_over_long_session_is_bounded(store):
    """A recall whose full result is over 4 MiB still fits one response.

    4 MiB is the figure the plan named. It is also comfortably over the
    1 MiB the client actually enforces, so a fixture built to the planned
    number exercises the real limit several times over.
    """
    db_path, seeded = store
    _fill_memories(db_path, seeded, count=600, size=8000)

    conn = _conn(db_path)
    try:
        unbounded = read_memories(conn, session=seeded["session_ids"][0])
        full_bytes = pagination.wire_size({"scope": {}, "memories": unbounded})
        assert full_bytes > 4 * 1024 * 1024, f"fixture is only {full_bytes} bytes"

        page = tools_read.recall(conn, {"session": seeded["session_key"]})
    finally:
        conn.close()

    assert pagination.wire_size(page) <= pagination.RESPONSE_BUDGET
    assert pagination.wire_size(page) < pagination.MAX_SSE_EVENT_BYTES
    assert page["next_cursor"] is not None, "a truncated page must say how to continue"
    assert len(page["memories"]) < len(unbounded)


def test_the_paginate_bound_is_measured_on_the_serialized_payload_not_the_row_count(store):
    """Row count is not a proxy for bytes, and the code must not treat it as one.

    Two scopes with the *same* number of memories, differing only in how
    large each statement is, must produce different page sizes. If the cut
    were an item count, both would return the same number of rows and this
    would fail.
    """
    db_path, seeded = store
    pages = {}
    for label, size in (("small", 200), ("large", 20000)):
        # Separate databases, not two rounds against one: appending the large
        # memories after the small ones would leave page one entirely small
        # either way, and the test would compare a store against itself.
        other = db_path.parent / f"{label}.db"
        other_seeded = _seed(other)
        _fill_memories(other, other_seeded, count=400, size=size)
        conn = _conn(other)
        try:
            pages[label] = tools_read.recall(conn, {"session": other_seeded["session_key"]})
        finally:
            conn.close()

    assert len(pages["small"]["memories"]) > len(pages["large"]["memories"])
    for page in pages.values():
        assert pagination.wire_size(page) <= pagination.RESPONSE_BUDGET


def test_paginate_wire_size_counts_the_second_escaping_a_row_count_cannot_see():
    """The payload is escaped twice, and quotes are what makes that expensive.

    The tool's dict becomes JSON, and that JSON is embedded as a *string* in
    `content[0].text`, so one `"` costs four bytes on the wire. A budget
    calibrated on plain prose would be well over on quote-heavy evidence.
    """
    plain = {"scope": {}, "memories": [{"statement": "x" * 4000}]}
    quoted = {"scope": {}, "memories": [{"statement": '"' * 4000}]}
    assert pagination.wire_size(quoted) > pagination.wire_size(plain) * 1.5


def test_a_paginate_cursor_from_one_scope_is_refused_against_another(store):
    """A cursor is bound to the question it answered.

    Honoured across scopes it would return the wrong scope's rows in a
    response that looks entirely normal — the one failure the scope rules in
    `tools_read` exist to prevent, reintroduced through the back door.
    """
    db_path, seeded = store
    _fill_memories(db_path, seeded, count=400, size=8000)

    conn = _conn(db_path)
    try:
        session_page = tools_read.recall(conn, {"session": seeded["session_key"]})
        assert session_page["next_cursor"] is not None

        with pytest.raises(pagination.CursorError) as excinfo:
            tools_read.recall(conn, {"project": "demo"}, session_page["next_cursor"])
        assert "different scope" in str(excinfo.value)

        # Positive control: the same cursor against its own scope works, so
        # the refusal above is about the scope and not about the cursor
        # being unusable in general.
        again = tools_read.recall(
            conn, {"session": seeded["session_key"]}, session_page["next_cursor"]
        )
        assert again["memories"], "the cursor must still work for its own scope"
    finally:
        conn.close()


def test_a_garbled_paginate_cursor_is_refused_rather_than_silently_restarted(store):
    """Ignoring a bad cursor restarts at page one without saying so.

    The caller would re-read rows it already has, believing it advanced.
    """
    db_path, seeded = store
    conn = _conn(db_path)
    try:
        for bad in ("not-a-cursor", "", "!!!!", base64.urlsafe_b64encode(b"{}").decode()):
            with pytest.raises(pagination.CursorError):
                tools_read.recall(conn, {"session": seeded["session_key"]}, bad)
    finally:
        conn.close()


def test_paginating_to_exhaustion_returns_every_memory_exactly_once(store):
    """The property that makes the cursor keyset rather than offset.

    `palaver observe` writes to this database while an agent pages through
    it. Under `LIMIT/OFFSET` an insert between two pages shifts every later
    offset and a row is dropped with no trace — so a row is inserted between
    every pair of pages here. A test that paged a quiescent store would pass
    for an offset cursor too, and prove nothing about the one that matters.
    """
    db_path, seeded = store
    conn = _conn(db_path)
    try:
        before = len(read_memories(conn, session=seeded["session_ids"][0]))
    finally:
        conn.close()
    _fill_memories(db_path, seeded, count=500, size=6000)

    seen: list[str] = []
    cursor = None
    pages = 0
    inserted = 0
    while True:
        conn = _conn(db_path)
        try:
            page = tools_read.recall(conn, {"session": seeded["session_key"]}, cursor)
        finally:
            conn.close()
        seen.extend(memory["statement"] for memory in page["memories"])
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages < 100, "paging is not converging"
        _fill_memories(db_path, seeded, count=1, size=6000)  # observe, mid-read
        inserted += 1

    assert pages > 1, "the fixture must be large enough to actually paginate"
    conn = _conn(db_path)
    try:
        expected = [
            row["statement"] for row in read_memories(conn, session=seeded["session_ids"][0])
        ]
    finally:
        conn.close()
    assert len(seen) == len(set(seen)), "a row came back twice"
    assert seen == expected, "the pages did not reconstruct the store in order"
    # `expected` is read *after* the inserts, so `seen == expected` already
    # fails if a mid-read write was skipped. This states the count outright
    # anyway: the pair of lists could in principle agree while both being
    # short, and "the concurrent writes were visible to a later page" is the
    # property the whole test exists for. Naming it means a future change
    # that quietly stops exercising concurrency fails here rather than
    # passing as a quiescent-store test wearing this one's name.
    assert inserted > 0, "no write landed mid-read, so nothing about concurrency was tested"
    assert len(seen) == before + 500 + inserted


def test_a_single_memory_over_the_budget_is_refused_rather_than_paged_forever(store):
    """Paging cannot split a row, so an oversized one must raise.

    Returning an empty page plus a cursor would loop a caller forever on a
    row that can never be delivered.
    """
    db_path, seeded = store
    # A session of its own, so the oversized memory is the *first* row of its
    # scope. Behind a small row it would simply not fit on page one and be
    # deferred forever instead — a different bug, and not this one.
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?)",
            (seeded["project_id"], "claude-code", "session-oversized"),
        )
        conn.commit()
        seeded = {
            **seeded,
            "session_ids": [
                *seeded["session_ids"],
                conn.execute(
                    "SELECT id FROM sessions WHERE external_id = ?", ("session-oversized",)
                ).fetchone()[0],
            ],
        }
    finally:
        conn.close()

    _fill_memories(db_path, seeded, count=1, size=pagination.RESPONSE_BUDGET, session_index=1)
    conn = _conn(db_path)
    try:
        with pytest.raises(pagination.RowTooLargeError) as excinfo:
            tools_read.recall(conn, {"session": "demo/session-oversized"})
        assert "shortened at the source" in str(excinfo.value)

        # Positive control: the same scope with a normal-sized memory
        # returns it, so the refusal above is about the size and not about
        # this session being unreadable.
        ok = tools_read.recall(conn, {"session": seeded["session_key"]})
        assert ok["memories"]
    finally:
        conn.close()


def test_palaver_sessions_paginates_without_ever_returning_a_rowid(store):
    """The session list pages too, and still refuses to hand out a rowid.

    `resolve_session_id` refuses a rowid as a session identifier so a caller
    never holds one. Paginating by `sessions.id` puts that value back within
    reach, so the emitted records are checked for it explicitly.
    """
    db_path, seeded = store
    conn = connect(db_path)
    try:
        for index in range(300):
            conn.execute(
                "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?)",
                (seeded["project_id"], "claude-code", f"paginate-session-{index:04d}"),
            )
        conn.commit()
    finally:
        conn.close()

    conn = _conn(db_path)
    try:
        page = tools_read.sessions(conn, {"project": "demo"})
    finally:
        conn.close()

    assert page["sessions"]
    assert pagination.wire_size(page) <= pagination.RESPONSE_BUDGET
    for record in page["sessions"]:
        assert set(record) == {"session_key", "source", "started_at", "ended_at"}


def test_a_paginated_recall_survives_the_real_transport_a_client_speaks(tmp_path):
    """The check that would have caught this task's premise being wrong.

    Everything above measures Palaver's own arithmetic. This drives a real
    `streamable_http_client` against a real server over a loopback socket
    and follows the cursors, because the limit being budgeted for is the
    *client's*, and no in-process assertion can observe it.
    """
    db_path = tmp_path / "wire.db"
    seeded = _seed(db_path)
    _fill_memories(db_path, seeded, count=400, size=8000)
    port = mcp_cli._free_port()

    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVE_SUBPROCESS, str(db_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        url = _endpoint_line(proc)
        assert url.startswith("http://")
        pages = asyncio.run(_with_timeout(_follow_cursors(url, seeded["session_key"]), 60.0))
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)
    except BaseException:
        proc.kill()
        raise

    statements = [memory["statement"] for page in pages for memory in page["memories"]]
    assert len(pages) > 1, "the fixture must be large enough to actually paginate"
    assert len(statements) == len(set(statements))
    assert len(statements) == 401  # 400 written here, plus the one `_seed` writes


async def _follow_cursors(url: str, session_key: str) -> list[dict]:
    """Page a real client through to exhaustion, returning every page."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    pages: list[dict] = []
    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            cursor = None
            while True:
                arguments: dict = {"scope": {"session": session_key}}
                if cursor is not None:
                    arguments["cursor"] = cursor
                result = await session.call_tool("palaver_recall", arguments)
                assert not result.is_error, result.content[0].text
                page = json.loads(result.content[0].text)
                pages.append(page)
                cursor = page["next_cursor"]
                if cursor is None or len(pages) > 100:
                    return pages


def test_paginate_wire_size_includes_the_framing_the_budget_is_measured_against():
    """The budget is per *SSE event*, not per tool payload.

    `httpx2` counts the `event:`/`data:` lines and everything the JSON-RPC
    envelope adds around the result. A `wire_size` that returned only the
    tool's own JSON would report a page as fitting when the event it becomes
    does not — and with 25% headroom in the budget, no size-based test would
    notice. Mutation testing found exactly that hole.
    """
    payload = {"scope": {"session": "demo/x"}, "memories": [{"statement": "y" * 1000}]}
    inner = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    framed = pagination.wire_size(payload)
    # `event: message\r\ndata: ` + `\r\n\r\n` is 26 bytes before the envelope
    # keys, so anything at or below the bare payload is not counting them.
    assert framed >= inner + 26

    # The request id belongs to the transport, so a tool cannot know it and
    # must model it at its widest. A one-digit placeholder would make this
    # function under-report late in a long session, which is the one
    # direction an estimate of a ceiling must never be wrong in.
    actual = len(
        (
            "event: message\r\ndata: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4_294_967_295,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    payload, ensure_ascii=False, separators=(",", ":")
                                ),
                            }
                        ],
                        "isError": False,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\r\n\r\n"
        ).encode()
    )
    assert framed >= actual, "wire_size under-reports once request ids grow past one digit"


def test_a_paginate_cursor_from_an_older_encoding_is_refused_by_version(store):
    """A version bump has to be the reason, not the scope check downstream.

    Every malformed cursor in the test above also fails the scope
    fingerprint, so the version check could be deleted and nothing would
    fail. This builds a cursor whose fingerprint is *correct* and whose
    version is not, which only the version check can catch. A future
    encoding change would otherwise be read as if it were the current one.
    """
    _, seeded = store
    echo = {"session": seeded["session_key"]}
    forged = (
        base64.urlsafe_b64encode(
            json.dumps(
                {
                    "v": pagination._CURSOR_VERSION + 1,
                    "s": pagination._scope_fingerprint(echo),
                    "a": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    with pytest.raises(pagination.CursorError) as excinfo:
        pagination.decode_cursor(forged, echo)
    assert "version" in str(excinfo.value)


def test_paginating_a_project_scope_to_exhaustion_returns_every_memory_once(store):
    """The project branch has its own query, and its own chance to lose the keyset.

    `read_memories` builds two separate SQL statements. The session one is
    covered above; a keyset dropped from the project one would re-read from
    the start every page and loop, or duplicate rows, with nothing else
    noticing. Mutation testing found this branch uncovered.
    """
    db_path, seeded = store
    _fill_memories(db_path, seeded, count=500, size=6000)

    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        conn = _conn(db_path)
        try:
            page = tools_read.recall(conn, {"project": "demo"}, cursor)
        finally:
            conn.close()
        seen.extend(memory["id"] for memory in page["memories"])
        pages += 1
        cursor = page["next_cursor"]
        if cursor is None:
            break
        assert pages < 100, "project-scoped paging is not converging"

    assert pages > 1, "the fixture must be large enough to actually paginate"
    conn = _conn(db_path)
    try:
        expected = [row["id"] for row in read_memories(conn, project="demo")]
    finally:
        conn.close()
    assert seen == expected
    assert len(seen) == len(set(seen))
