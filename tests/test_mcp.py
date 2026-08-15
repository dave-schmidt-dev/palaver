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
import base64
import concurrent.futures
import errno
import io
import json
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

import palaver
from palaver.cli import SUBCOMMANDS
from palaver.cli import mcp as mcp_cli
from palaver.mcp import pagination, tools_read, tools_write
from palaver.mcp import server as mcp_server
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.scope import read_memories
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
def short_store():
    """A seeded store on a path short enough for the daemon socket.

    pytest's `tmp_path` is ~90 bytes before a test name is appended, and
    `sun_path` holds 103 -- so any test that must reach the *real* socket
    path needs a shorter home than the default fixture provides.
    """
    directory = Path(tempfile.mkdtemp(prefix="plv", dir="/tmp"))  # noqa: S108
    try:
        db_path = directory / "palaver.db"
        yield db_path, _seed(db_path)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


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
    """A tool added later without a scope argument fails here, not in review.

    The registered set is pinned to `READ_TOOLS | WRITE_TOOLS` rather than
    just iterated: a tool registered from neither mapping -- a debug helper
    left in, say -- would otherwise be exempt from every check below simply
    by not being in a list this test reads.
    """
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == set(tools_read.READ_TOOLS) | set(tools_write.WRITE_TOOLS)

    for name in tools_read.READ_TOOLS:
        assert "scope" in tools[name].input_schema["required"], name


def test_the_write_tool_takes_a_memory_id_and_never_a_scope(store):
    """`palaver_correct` names one row, so a scope would be the wrong shape.

    A scoped correction would be a bulk edit -- exactly the operation an
    append-only store must not offer -- and `memory_id` comes from a prior
    recall, so the caller has already chosen a scope to get it.
    """
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    schema = tools["palaver_correct"].input_schema
    assert set(schema["required"]) == {"memory_id", "statement"}
    assert "scope" not in schema["properties"]
    # `ctx` is the server's, not the caller's: a client that could pass one
    # would be choosing which session the elicitation goes to.
    assert "ctx" not in schema["properties"]


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
        # A subprocess, not an in-process `run()`: if the guard ever stops
        # working, `run()` does not return an error, it serves forever, and
        # an in-process call would hang the suite rather than fail it.
        # Measured — mutating the bind away cost a mutation battery its whole
        # budget twice before this test was moved out of process.
        proc = subprocess.Popen(
            [sys.executable, "-c", _SERVE_SUBPROCESS, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError(
                "the collision was never refused: the process is still serving a "
                "port another socket holds"
            ) from None
        assert proc.returncode == 1, f"exited {proc.returncode}\n{stderr}"
        assert "http://" not in stdout, stdout
        assert "cannot bind" in stdout, stdout
        assert "already serving it" in stdout, stdout
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
def test_every_read_tool_reports_when_the_store_was_last_written_and_who_is_watching(store, tool):
    """A crashed daemon and a quiet one are otherwise identical from here.

    Same memories, same timestamps, same shape. A reader who cannot tell
    them apart will read a two-day-old store as current, which is INV-7's
    failure -- and the likelier reading, since a store that answers at all
    looks healthy.
    """
    db_path, _ = store
    server = mcp_server.build_server(db_path)
    result = _call(server, tool, {"scope": {"project": "demo"}})
    payload = json.loads(result.content[0].text)

    assert "observed_at" in payload, "no freshness stamp, so a stale answer looks current"
    assert "daemon_running" in payload
    assert payload["daemon_running"] is False, "nothing is serving this test store"


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


# =============================================================================
# Sign-off: the write happens only after a human says so
# =============================================================================


class _StubContext:
    """A request context that answers an elicitation however a test needs.

    Stands in for the client's side of the round trip. The transport itself
    is exercised separately in `tests/test_supervision.py`; what is under
    test here is what `correct` does with each of the three answers, which
    would be tedious and slow to drive through a real client three times.
    """

    def __init__(self, action="accept", approved=True, note=""):
        self.action = action
        self.approved = approved
        self.note = note
        self.message = None

    async def elicit(self, message, schema):
        from mcp.server.elicitation import (
            AcceptedElicitation,
            CancelledElicitation,
            DeclinedElicitation,
        )

        self.message = message
        if self.action == "decline":
            return DeclinedElicitation()
        if self.action == "cancel":
            return CancelledElicitation()
        return AcceptedElicitation(data=schema(approved=self.approved, note=self.note))


def _correct(db_path, ctx, memory_id, statement):
    conn = mcp_server.open_readonly(db_path)
    try:
        return asyncio.run(tools_write.correct(conn, db_path, ctx, memory_id, statement))
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("action", "approved"),
    [("decline", True), ("cancel", True), ("accept", False)],
)
def test_a_correction_without_a_yes_writes_nothing(store, action, approved):
    """Three ways to say no, and none of them may write.

    "accept" with `approved=false` is the one worth naming: the elicitation
    itself succeeded, so a check that only looked at `result.action` would
    treat a refusal as consent.
    """
    db_path, seeded = store
    ctx = _StubContext(action=action, approved=approved)

    with pytest.raises(tools_write.WriteRefused):
        _correct(db_path, ctx, seeded["memory_id"], "should never land")

    conn = _conn(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_the_sign_off_prompt_quotes_both_statements_in_full(store):
    """Approving text you cannot see is a keystroke, not sign-off."""
    db_path, seeded = store
    ctx = _StubContext(action="decline")
    with pytest.raises(tools_write.WriteRefused):
        _correct(db_path, ctx, seeded["memory_id"], "the corrected reading")

    conn = _conn(db_path)
    try:
        original = conn.execute(
            "SELECT statement FROM memories WHERE id = ?", (seeded["memory_id"],)
        ).fetchone()[0]
    finally:
        conn.close()
    assert original in ctx.message
    assert "the corrected reading" in ctx.message


def test_an_approved_correction_with_no_daemon_still_writes_nothing(short_store):
    """Consent is not the last gate; the single writer is.

    A human saying yes does not create a writer. This is the case where the
    tool is most tempted to "just do it" -- permission has been granted and
    only the plumbing is missing -- and it is exactly where a second writer
    would be opened.

    Uses `short_store` because pytest's `tmp_path` overruns `sun_path`, and
    the resulting `SocketPathTooLongError` would let this pass for the wrong
    reason: no write happens either way, but the refusal under test is the
    missing daemon.
    """
    db_path, seeded = short_store
    with pytest.raises(Exception) as excinfo:
        _correct(db_path, _StubContext(), seeded["memory_id"], "approved but undeliverable")
    assert "no palaver observe daemon" in str(excinfo.value)
    assert "second" in str(excinfo.value)

    conn = _conn(db_path)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_correcting_a_memory_that_does_not_exist_asks_nobody_anything(store):
    """The lookup precedes the prompt, deliberately.

    A sign-off dialog quoting a memory that is not there invites approval of
    a change that cannot happen, and the error that follows arrives after
    the human has already said yes.
    """
    db_path, _ = store
    ctx = _StubContext()
    with pytest.raises(LookupError):
        _correct(db_path, ctx, 999_999, "no such memory")
    assert ctx.message is None, "the human was asked about a memory that does not exist"


def test_correcting_an_already_superseded_memory_is_refused_before_the_prompt(store):
    """A memory has at most one successor (INV-4), so this could never land."""
    db_path, seeded = store
    conn = connect(db_path)
    try:
        chunk_id = conn.execute("SELECT id FROM transcript_chunks LIMIT 1").fetchone()[0]
        write_memory(
            conn,
            project_id=seeded["project_id"],
            session_id=seeded["session_ids"][0],
            statement="the first correction",
            origin="user-correction",
            tier=1,
            evidence=[EvidenceAnchor(start_offset=0, end_offset=8, transcript_chunk_id=chunk_id)],
            supersedes=seeded["memory_id"],
        )
        conn.commit()
    finally:
        conn.close()

    ctx = _StubContext()
    with pytest.raises(LookupError, match="already superseded"):
        _correct(db_path, ctx, seeded["memory_id"], "a second correction")
    assert ctx.message is None
