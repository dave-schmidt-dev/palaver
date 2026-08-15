"""Task 6.1: the MCP read surface, and the two ways it could quietly lie.

The tests here are organised around failures that are invisible from the
caller's side, because those are the ones an API-shaped test misses:

* **A defaulted scope.** A tool that answers a project-wide question when a
  session was meant returns something that reads exactly as authoritative as
  the right answer. Nothing downstream can tell. So the refusal is asserted
  at both layers — the parser, and a real tool call through the server.
* **A silently-resolved identifier.** `read_memories(session=...)` takes a
  rowid; an MCP caller holds a `session_key`. If the tool accepted either,
  an integer-looking key would resolve against an unrelated row and answer
  confidently. The rowid path is asserted *refused*, not merely unused.

The transport tests exist because the concurrency question Phase 6 is
accepted on is a transport question, and a test that called the tool
functions in-process would prove none of it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import errno
import io
import signal
import sqlite3
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import palaver
from palaver.cli import SUBCOMMANDS
from palaver.cli import mcp as mcp_cli
from palaver.mcp import server as mcp_server
from palaver.mcp import tools_read
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate

# =============================================================================
# Fixtures
# =============================================================================


def _seed(db_path: Path, *, sources=("claude-code",), external_id="session-aaa") -> dict:
    """Build a small database and return the ids a test needs to assert on.

    `sources` is a tuple so a test can create the same `external_id` under
    two sources — the collision `sessions`' own `UNIQUE (source, external_id)`
    permits and a key-only lookup therefore cannot resolve.
    """
    migrate(db_path)
    conn = connect(db_path)
    project_id = conn.execute(
        "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
        ("demo", str(db_path.parent)),
    ).fetchone()[0]

    session_ids = []
    for source in sources:
        session_id = conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
            (project_id, source, external_id),
        ).fetchone()[0]
        session_ids.append(session_id)

    chunk_id = conn.execute(
        "INSERT INTO transcript_chunks (session_id, seq, role, content) VALUES (?, ?, ?, ?) "
        "RETURNING id",
        (session_ids[0], 0, "assistant", "the evidence text"),
    ).fetchone()[0]
    memory_id = write_memory(
        conn,
        project_id=project_id,
        session_id=session_ids[0],
        statement="the first session decided to use SQLite",
        origin="observer",
        tier=4,
        evidence=[EvidenceAnchor(start_offset=0, end_offset=8, transcript_chunk_id=chunk_id)],
    )
    conn.commit()
    conn.close()
    return {
        "project_id": project_id,
        "session_ids": session_ids,
        "memory_id": memory_id,
        "external_id": external_id,
        "session_key": f"demo/{external_id}",
    }


@pytest.fixture
def store(tmp_path):
    """A migrated database with one project, one session, and one memory."""
    db_path = tmp_path / "store.db"
    seeded = _seed(db_path)
    return db_path, seeded


def _conn(db_path: Path) -> sqlite3.Connection:
    return mcp_server.open_readonly(db_path)


def _call(server, name, arguments):
    """Call a tool the way a client does, through the server's dispatcher."""
    return asyncio.run(server.call_tool(name, arguments))


# =============================================================================
# Scope is required, and never defaulted
# =============================================================================


def test_a_read_tool_called_with_no_scope_raises_rather_than_answering():
    """The whole point: no scope is not a request for everything."""
    with pytest.raises(tools_read.ScopeError) as excinfo:
        tools_read.parse_scope({})
    assert "exactly one" in str(excinfo.value)


def test_a_read_tool_called_with_both_scope_keys_raises():
    with pytest.raises(tools_read.ScopeError):
        tools_read.parse_scope({"project": "demo", "session": "demo/session-aaa"})


def test_an_unknown_scope_key_raises_rather_than_being_ignored():
    """A typo'd key must not silently leave the scope empty and default."""
    with pytest.raises(tools_read.ScopeError) as excinfo:
        tools_read.parse_scope({"proejct": "demo"})
    assert "proejct" in str(excinfo.value)


def test_an_unknown_key_alongside_a_good_one_is_still_refused():
    """The case a dropped unknown-key check would sail straight through.

    With only the previous test, a parser that ignored unknown keys still
    raises — on "neither key present" — and its message still quotes the
    typo. Pairing the typo with a valid key removes that cover: the scope
    now resolves, so the refusal has to come from the unknown-key check
    itself or not at all. A caller who wrote `{"session": ..., "porject":
    ...}` believes they asked a narrower question than they did.
    """
    with pytest.raises(tools_read.ScopeError) as excinfo:
        tools_read.parse_scope({"project": "demo", "sesion": "demo/x"})
    assert "sesion" in str(excinfo.value)


@pytest.mark.parametrize("value", [None, 3, "", "   ", ["demo"]])
def test_a_scope_value_that_is_not_a_non_empty_string_raises(value):
    with pytest.raises(tools_read.ScopeError):
        tools_read.parse_scope({"project": value})


def test_a_scope_that_is_not_a_mapping_says_so(store):
    """Asserted on the message, not merely on the exception type.

    A bare string is iterable, so a parser that skipped the type check
    still raises — its `set(scope) - set(SCOPE_KEYS)` finds unknown
    *characters*. Same exception, useless message. A caller who passed
    `"demo"` needs to be told the shape is wrong, not handed a list of
    letters.
    """
    with pytest.raises(tools_read.ScopeError) as excinfo:
        tools_read.parse_scope("demo")
    assert "must be a mapping" in str(excinfo.value)


def test_a_valid_scope_parses_to_exactly_one_side():
    assert tools_read.parse_scope({"project": " demo "}) == tools_read.Scope(project="demo")
    assert tools_read.parse_scope({"session": "demo/x"}) == tools_read.Scope(session="demo/x")


def test_the_scope_refusal_holds_through_a_real_tool_call(store):
    """Asserted at the tool boundary too, not only in the parser.

    The parser could be perfect and the server could still register a tool
    that never calls it. This drives the same refusal through the server's
    own dispatcher, which is the path a client takes.
    """
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    for arguments in ({"scope": {}}, {"scope": {"project": "demo", "session": "demo/x"}}):
        with pytest.raises(Exception) as excinfo:
            _call(server, "palaver_recall", arguments)
        assert "exactly one" in str(excinfo.value)


def test_every_registered_read_tool_takes_a_scope(store):
    """A tool added later without a scope argument fails here, not in review."""
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == set(tools_read.READ_TOOLS)
    for tool in tools:
        assert "scope" in tool.input_schema["required"], tool.name


# =============================================================================
# The identifier a caller actually holds (TASKS.md Task 7)
# =============================================================================


def test_a_session_key_resolves_to_that_sessions_memories(store):
    """The identifier `palaver status` prints is the one the tool accepts."""
    db_path, seeded = store
    conn = _conn(db_path)
    result = tools_read.recall(conn, {"session": seeded["session_key"]})
    conn.close()
    assert [row["id"] for row in result["memories"]] == [seeded["memory_id"]]


def test_a_bare_session_id_resolves_the_same_way(store):
    db_path, seeded = store
    conn = _conn(db_path)
    result = tools_read.recall(conn, {"session": seeded["external_id"]})
    conn.close()
    assert [row["id"] for row in result["memories"]] == [seeded["memory_id"]]


def test_an_internal_rowid_is_refused_rather_than_silently_answered(store):
    """The trap this resolver exists to avoid.

    The seeded session's `sessions.id` is a perfectly valid rowid. Passing
    it must fail, because a caller who holds a rowid holds it by accident,
    and a resolver that fell back to rowids would answer a session-id-shaped
    string against a completely unrelated row.
    """
    db_path, seeded = store
    rowid = seeded["session_ids"][0]
    conn = _conn(db_path)
    with pytest.raises(tools_read.SessionLookupError) as excinfo:
        tools_read.resolve_session_id(conn, str(rowid))
    conn.close()
    assert "rowid" in str(excinfo.value)


def test_a_rowid_that_is_valid_does_not_answer_through_the_tool_either(store):
    """Positive control on the refusal: the rowid really does exist."""
    db_path, seeded = store
    rowid = seeded["session_ids"][0]
    conn = _conn(db_path)
    assert conn.execute("SELECT 1 FROM sessions WHERE id = ?", (rowid,)).fetchone() == (1,)
    with pytest.raises(tools_read.SessionLookupError):
        tools_read.recall(conn, {"session": str(rowid)})
    conn.close()


def test_a_session_id_shared_by_two_sources_is_refused_not_guessed(tmp_path):
    """`sessions` is UNIQUE (source, external_id), so a key alone can be ambiguous."""
    db_path = tmp_path / "two.db"
    seeded = _seed(db_path, sources=("claude-code", "codex"))
    conn = _conn(db_path)
    with pytest.raises(tools_read.SessionLookupError) as excinfo:
        tools_read.resolve_session_id(conn, seeded["external_id"])
    conn.close()
    message = str(excinfo.value)
    assert "more than one" in message
    assert "claude-code" in message and "codex" in message


def test_the_project_half_of_a_session_key_selects_between_projects(tmp_path):
    """The full key means something; it is not decoration on the bare id.

    Two projects hold a session under the same external id (legal — the
    uniqueness constraint is on `(source, external_id)`). The bare id is
    therefore ambiguous, and only the project half can settle it. A resolver
    that ignored the project half would answer with whichever row sorted
    first.
    """
    db_path = tmp_path / "two-projects.db"
    migrate(db_path)
    conn = connect(db_path)
    wanted = None
    for index, (project, source) in enumerate((("alpha", "claude-code"), ("beta", "codex"))):
        project_id = conn.execute(
            "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
            (project, f"{tmp_path}/{project}"),
        ).fetchone()[0]
        session_id = conn.execute(
            "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
            (project_id, source, "shared-id"),
        ).fetchone()[0]
        if index == 1:
            wanted = session_id
    conn.commit()
    conn.close()

    reader = _conn(db_path)
    assert tools_read.resolve_session_id(reader, "beta/shared-id") == wanted
    with pytest.raises(tools_read.SessionLookupError):
        tools_read.resolve_session_id(reader, "shared-id")
    reader.close()


def test_a_session_key_with_no_session_id_is_refused_by_name(store):
    """`demo/` is a caller who built the key by concatenation and lost the id."""
    db_path, _ = store
    conn = _conn(db_path)
    with pytest.raises(tools_read.SessionLookupError) as excinfo:
        tools_read.resolve_session_id(conn, "demo/")
    conn.close()
    assert "names no session id" in str(excinfo.value)


def test_an_unknown_session_raises_rather_than_returning_nothing(store):
    """Empty and absent are different answers; collapsing them hides a typo."""
    db_path, _ = store
    conn = _conn(db_path)
    with pytest.raises(tools_read.SessionLookupError):
        tools_read.recall(conn, {"session": "demo/no-such-session"})
    conn.close()


def test_an_unknown_project_raises_rather_than_returning_nothing(store):
    db_path, _ = store
    conn = _conn(db_path)
    with pytest.raises(LookupError):
        tools_read.recall(conn, {"project": "no-such-project"})
    conn.close()


def test_a_project_scope_and_a_session_scope_are_not_the_same_answer(tmp_path):
    """The confusion the scope rule exists to prevent, made concrete."""
    db_path = tmp_path / "two-sessions.db"
    seeded = _seed(db_path)
    conn = connect(db_path)
    other = conn.execute(
        "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
        (seeded["project_id"], "claude-code", "session-bbb"),
    ).fetchone()[0]
    chunk = conn.execute(
        "INSERT INTO transcript_chunks (session_id, seq, role, content) VALUES (?, ?, ?, ?) "
        "RETURNING id",
        (other, 0, "assistant", "another session's text"),
    ).fetchone()[0]
    write_memory(
        conn,
        project_id=seeded["project_id"],
        session_id=other,
        statement="the second session decided something else",
        origin="observer",
        tier=4,
        evidence=[EvidenceAnchor(start_offset=0, end_offset=7, transcript_chunk_id=chunk)],
    )
    conn.commit()
    conn.close()

    reader = _conn(db_path)
    by_project = tools_read.recall(reader, {"project": "demo"})
    by_session = tools_read.recall(reader, {"session": seeded["session_key"]})
    reader.close()

    assert len(by_project["memories"]) == 2
    assert len(by_session["memories"]) == 1


# =============================================================================
# Provenance travels with the answer
# =============================================================================


def test_every_returned_memory_record_carries_a_tier(store):
    db_path, _ = store
    conn = _conn(db_path)
    result = tools_read.recall(conn, {"project": "demo"})
    conn.close()
    assert result["memories"]
    for row in result["memories"]:
        assert "tier" in row
        assert isinstance(row["tier"], int)


def test_every_returned_memory_record_names_its_tier(store):
    """The number alone tells a caller nothing about which way it ranks."""
    db_path, _ = store
    conn = _conn(db_path)
    result = tools_read.recall(conn, {"project": "demo"})
    conn.close()
    assert [row["tier_name"] for row in result["memories"]] == ["observer_inference"]


def test_the_scope_is_echoed_back_with_the_answer(store):
    """So a logged response says which question it answered."""
    db_path, seeded = store
    conn = _conn(db_path)
    assert tools_read.recall(conn, {"project": "demo"})["scope"] == {"project": "demo"}
    assert tools_read.recall(conn, {"session": seeded["session_key"]})["scope"] == {
        "session": seeded["session_key"]
    }
    conn.close()


@pytest.fixture
def busy_store(tmp_path):
    """Two projects, three sessions — so a scope that did nothing is visible.

    A store with one project and one session cannot distinguish "listed the
    scope" from "listed everything": the two answers are the same list.
    """
    db_path = tmp_path / "busy.db"
    seeded = _seed(db_path)
    conn = connect(db_path)
    conn.execute(
        "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?)",
        (seeded["project_id"], "claude-code", "session-bbb"),
    )
    other_project = conn.execute(
        "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
        ("elsewhere", str(db_path.parent / "elsewhere")),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?)",
        (other_project, "claude-code", "session-ccc"),
    )
    conn.commit()
    conn.close()
    return db_path, seeded


def test_palaver_sessions_lists_the_keys_a_session_scope_needs(busy_store):
    """A caller with only a project name must be able to obtain a session key."""
    db_path, seeded = busy_store
    conn = _conn(db_path)
    result = tools_read.sessions(conn, {"project": "demo"})
    conn.close()
    keys = [row["session_key"] for row in result["sessions"]]
    assert keys == [seeded["session_key"], "demo/session-bbb"]
    # The other project's session exists and is deliberately absent.
    assert "elsewhere/session-ccc" not in keys


def test_palaver_sessions_confirms_one_key_under_a_session_scope(busy_store):
    """One session out of three, not three."""
    db_path, seeded = busy_store
    conn = _conn(db_path)
    result = tools_read.sessions(conn, {"session": seeded["external_id"]})
    conn.close()
    assert [row["session_key"] for row in result["sessions"]] == [seeded["session_key"]]


def test_palaver_sessions_carries_the_source_each_session_came_from(busy_store):
    db_path, _ = busy_store
    conn = _conn(db_path)
    result = tools_read.sessions(conn, {"project": "demo"})
    conn.close()
    assert {row["source"] for row in result["sessions"]} == {"claude-code"}


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
# The subcommand and its selftest
# =============================================================================


def test_the_mcp_subcommand_is_registered():
    """The Phase 6 gate invokes `palaver mcp`; no other task registers it."""
    assert mcp_cli in SUBCOMMANDS
    assert mcp_cli.NAME == "mcp"


def _selftest_args(**overrides) -> Namespace:
    defaults = {
        "selftest": True,
        "clients": 2,
        "db": None,
        "host": mcp_server.DEFAULT_HOST,
        "port": mcp_server.DEFAULT_PORT,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_the_selftest_drives_real_concurrent_clients_over_the_transport():
    """The Phase 6 acceptance check, at a smaller client count for the suite.

    Two rather than six here because the gate command runs six and this file
    runs on every commit; the code path is identical and the count is a
    parameter.
    """
    out = io.StringIO()
    status: list[str] = []
    exit_code = mcp_cli.run(_selftest_args(), out=out, on_status=status.append)
    report = out.getvalue()
    assert exit_code == 0, report
    assert "2 concurrent client(s) each got the seeded answer" in report
    assert "2 ok, 0 failed" in report
    assert any("opening 2 concurrent client(s)" in message for message in status)


def test_the_selftest_reports_the_endpoint_it_served(tmp_path):
    out = io.StringIO()
    mcp_cli.run(_selftest_args(clients=1), out=out, on_status=lambda _: None)
    assert "serving Streamable HTTP at http://127.0.0.1:" in out.getvalue()


def test_the_selftest_seed_is_what_the_selftest_asserts(tmp_path):
    """The seed and the assertion read the same constants, not two copies."""
    db_path = tmp_path / "seeded.db"
    session_key = mcp_cli.seed_selftest_db(db_path)
    assert session_key == f"{mcp_cli.SELFTEST_PROJECT}/{mcp_cli.SELFTEST_SESSION_ID}"
    conn = _conn(db_path)
    result = tools_read.recall(conn, {"session": session_key})
    conn.close()
    assert [row["statement"] for row in result["memories"]] == [mcp_cli.SELFTEST_STATEMENT]
    assert [row["tier"] for row in result["memories"]] == [mcp_cli.SELFTEST_TIER]


def _good_payload(session_key: str) -> dict:
    """Exactly what a healthy client returns, for a test to then spoil."""
    return {
        "sessions": {"sessions": [{"session_key": session_key, "source": "claude-code"}]},
        "recall": {
            "memories": [{"statement": mcp_cli.SELFTEST_STATEMENT, "tier": mcp_cli.SELFTEST_TIER}]
        },
    }


def _selftest_with_clients(monkeypatch, fake, *, clients=2) -> tuple[int, str]:
    """Run the selftest with `_one_client` replaced, and return (code, output)."""
    monkeypatch.setattr(mcp_cli, "_one_client", fake)
    out = io.StringIO()
    code = mcp_cli.run(_selftest_args(clients=clients), out=out, on_status=lambda _: None)
    return code, out.getvalue()


def test_the_selftest_is_not_merely_counting_connections(monkeypatch):
    """Negative control. Without it, every check below could be vacuous.

    The selftest's own passing run cannot tell whether it compared answers
    or only counted sockets — both report success. This hands it six
    perfectly healthy connections returning the wrong session and requires
    it to fail.
    """

    async def _wrong_session(url, session_key):
        payload = _good_payload(session_key)
        payload["sessions"]["sessions"] = [{"session_key": "someone-else/xyz"}]
        return payload

    code, report = _selftest_with_clients(monkeypatch, _wrong_session)
    assert code == 1
    assert "listed ['someone-else/xyz']" in report
    assert "0 ok, 2 failed" in report


def test_the_selftest_checks_the_statement_it_recalled(monkeypatch):
    async def _wrong_statement(url, session_key):
        payload = _good_payload(session_key)
        payload["recall"]["memories"] = []
        return payload

    code, report = _selftest_with_clients(monkeypatch, _wrong_statement)
    assert code == 1
    assert "unexpected memory" in report


def test_the_selftest_checks_the_tier_it_recalled(monkeypatch):
    """A response that lost its provenance is not a correct response."""

    async def _wrong_tier(url, session_key):
        payload = _good_payload(session_key)
        payload["recall"]["memories"][0]["tier"] = mcp_cli.SELFTEST_TIER + 1
        return payload

    code, report = _selftest_with_clients(monkeypatch, _wrong_tier)
    assert code == 1
    assert "tier [4]" in report


def test_a_client_that_raised_is_reported_as_a_failure(monkeypatch):
    """A dropped connection is the exact thing Phase 6 is accepted on."""

    async def _dropped(url, session_key):
        raise ConnectionResetError("peer went away")

    code, report = _selftest_with_clients(monkeypatch, _dropped)
    assert code == 1
    assert "ConnectionResetError" in report
    assert "peer went away" in report


def test_the_selftest_opens_exactly_the_number_of_clients_asked_for(monkeypatch):
    """`--clients 6` must open six, not one and a reassuring message."""
    seen: list[str] = []

    async def _counting(url, session_key):
        seen.append(url)
        return _good_payload(session_key)

    code, report = _selftest_with_clients(monkeypatch, _counting, clients=5)
    assert code == 0, report
    assert len(seen) == 5


def test_a_client_count_below_one_is_a_usage_error():
    out = io.StringIO()
    assert mcp_cli.run(_selftest_args(clients=0), out=out, on_status=lambda _: None) == 2
    assert "--clients must be at least 1" in out.getvalue()


def test_serving_a_missing_database_exits_one_and_says_so(tmp_path):
    out = io.StringIO()
    args = _selftest_args(selftest=False, db=tmp_path / "absent.db")
    assert mcp_cli.run(args, out=out, on_status=lambda _: None) == 1
    assert "no Palaver database" in out.getvalue()


# =============================================================================
# Serving for real, which the selftest does not exercise
#
# `--selftest` picks a free ephemeral port, drives clients, and tears down
# inside one `asyncio.run`. The invocation a person actually types after
# `claude mcp add` does none of those things: it binds the *fixed* default
# port and runs until a signal stops it. Both differences produced a defect
# that every test in the sections above passed straight over.
# =============================================================================


async def _with_timeout(coro, seconds: float = 10.0):
    """Fail rather than hang; a serve call that never returns is the defect."""
    async with asyncio.timeout(seconds):
        return await coro


def test_a_port_already_in_use_is_refused_before_a_url_is_printed(tmp_path):
    """The failure mode is the printed URL, not the failed bind.

    stdout is this command's result. A caller that reads a URL from it has
    no way to know the process died a moment later, so it holds a confident
    endpoint for a server that is not there — INV-7's failure, arrived at
    through a socket instead of a status.
    """
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    held = mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    try:
        port = held.getsockname()[1]
        out = io.StringIO()
        status: list[str] = []
        args = _selftest_args(selftest=False, db=db_path, port=port)
        assert mcp_cli.run(args, out=out, on_status=status.append) == 1
        assert "http://" not in out.getvalue()
        assert "cannot bind" in out.getvalue()
        assert "already serving it" in out.getvalue()
    finally:
        held.close()


def test_a_bound_listener_refuses_a_second_one(tmp_path):
    """The positive control for the test above, and for `SO_REUSEADDR`.

    `bind_listener` sets `SO_REUSEADDR` so a restart is not blocked by the
    previous process's sockets in `TIME_WAIT`. On a platform where that
    option also permitted a second live listener, the refusal above would
    never fire and the test would pass by never reaching its own subject.
    """
    first = mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    try:
        with pytest.raises(OSError) as excinfo:
            mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, first.getsockname()[1]).close()
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        first.close()


def test_the_stop_signals_are_claimed_before_asyncio_can_take_them():
    """`asyncio.run` only installs its handler if the slot is still default.

    That check is the whole mechanism `own_stop_signals` relies on, so this
    asserts the precondition rather than trusting it: SIGINT starts as
    `default_int_handler`, and afterwards it does not.
    """
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        stop = mcp_cli.own_stop_signals()
        for sig in (signal.SIGINT, signal.SIGTERM):
            installed = signal.getsignal(sig)
            assert installed is not signal.default_int_handler
            assert installed.__self__ is stop
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def test_a_stop_that_arrives_before_the_server_exists_is_not_dropped(tmp_path):
    """The window between claiming the signal and uvicorn claiming it back.

    uvicorn installs its handlers inside `serve()`, so everything before
    that — ASGI startup included — is covered only by the handler
    `own_stop_signals` left in the slot. A stop recorded there and then
    forgotten leaves a server running that was told to stop, which is the
    worse failure of the two: at least a traceback exits.
    """
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    _, app = mcp_server.build_app(db_path)
    stop = mcp_cli._StopRequest()
    stop.record(signal.SIGTERM, None)
    announced: list[str] = []

    # Returns rather than serving: with no timeout in sight, a test that
    # hangs here is the defect, not a slow test.
    asyncio.run(
        _with_timeout(
            mcp_cli._serve_forever(
                app,
                mcp_server.DEFAULT_HOST,
                mcp_cli._free_port(),
                stop=stop,
                announce=lambda: announced.append("up"),
            )
        )
    )
    assert announced == [], "announced an endpoint it never served"


def test_a_stop_that_arrives_after_the_server_exists_reaches_it(tmp_path):
    """The other half: once attached, a recorded stop is applied to uvicorn."""
    db_path = tmp_path / "serve.db"
    mcp_cli.seed_selftest_db(db_path)
    _, app = mcp_server.build_app(db_path)
    stop = mcp_cli._StopRequest()
    server = mcp_cli._build_server(app, mcp_server.DEFAULT_HOST, mcp_cli._free_port())
    stop.attach(server)
    assert server.should_exit is False
    stop.record(signal.SIGINT, None)
    assert server.should_exit is True


#: Runs `palaver mcp` the way a person does, in a process of its own, so a
#: real signal can be delivered to it. In-process this is untestable: the
#: defect was `asyncio.run` raising `KeyboardInterrupt` out of the serve
#: call, and pytest's own process cannot be interrupted to find out.
_SERVE_SUBPROCESS = """
import sys
from palaver.cli import main
db, port = sys.argv[1], sys.argv[2]
sys.argv = ["palaver", "mcp", "--db", db, "--port", port]
sys.exit(main())
"""


def _endpoint_line(proc, deadline: float = 30.0) -> str:
    """Read the endpoint from stdout, or fail — never wait forever.

    A bare `readline()` on a live child has no deadline, so a server that
    binds nothing and prints nothing turns a *failing* test into a *hanging*
    one. That is not hypothetical: a mutation battery over `bind_listener`
    lost its entire budget to the one mutant that hangs, and the run was
    killed mid-flight with a mutant still applied to the tree. A test that
    can hang is a test that cannot be run in bulk.
    """
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        line = pool.submit(proc.stdout.readline).result(timeout=deadline)
    except concurrent.futures.TimeoutError:
        proc.kill()  # Unblocks the reader thread by closing the pipe.
        raise AssertionError(f"no endpoint on stdout within {deadline}s") from None
    finally:
        pool.shutdown(wait=False)
    return line.strip()


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_a_stop_signal_exits_cleanly_rather_than_as_a_traceback(tmp_path, stop_signal):
    """Ctrl-C is the normal way to stop a foreground server, not a crash.

    It printed a twenty-five line `KeyboardInterrupt` traceback, because
    `sse_starlette` chains uvicorn's signal handler to whatever was
    installed when the app started, and under `asyncio.run` that is the
    handler which cancels the main task. SIGTERM is here for the same
    reason at one remove: it is what launchd sends, and a service that
    exits non-zero on a deliberate stop gets treated as a crash and
    throttled.
    """
    db_path = tmp_path / "serve.db"
    session_key = mcp_cli.seed_selftest_db(db_path)
    port = mcp_cli._free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVE_SUBPROCESS, str(db_path), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for the endpoint on stdout rather than sleeping: the URL is
        # written after the bind, so its arrival means the port is held.
        url = _endpoint_line(proc)
        assert url == f"http://{mcp_server.DEFAULT_HOST}:{port}{mcp_server.DEFAULT_PATH}"
        # The URL means "serving", so a client acting on it the instant it
        # appears must be answered. Also the only check anywhere that the
        # real serve path — not the selftest's ephemeral one — answers a
        # client at all.
        seen = asyncio.run(_with_timeout(mcp_cli._one_client(url, session_key)))
        assert [m["statement"] for m in seen["recall"]["memories"]] == [mcp_cli.SELFTEST_STATEMENT]
        proc.send_signal(stop_signal)
        stdout, stderr = proc.communicate(timeout=30)
    except BaseException:
        proc.kill()
        raise
    assert proc.returncode == 0, f"exited {proc.returncode}\n{stderr}"
    assert "Traceback" not in stderr, stderr
    assert "KeyboardInterrupt" not in stderr, stderr
    # Reached the end of `run()` rather than merely failing to crash.
    assert "stopped" in stderr, stderr


def test_the_server_advertises_a_version_in_the_handshake(store):
    """An empty version tells a user comparing two machines nothing."""
    server = mcp_server.build_server(Path("unused.db"), connect=lambda _: _conn(store))
    assert server.version == palaver.__version__
    assert server.version
