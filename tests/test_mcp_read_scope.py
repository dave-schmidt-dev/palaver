"""
Read-tool scope validation, session/project key resolution (TASKS.md Task 7), and the
tier/source provenance carried on every returned record.
"""

from __future__ import annotations

import asyncio

import pytest

from palaver.mcp import server as mcp_server
from palaver.mcp import tools_read, tools_write
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from tests._mcp_support import _call, _conn, _seed
from tests._mcp_support import store as store

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
