"""
The correction tool's sign-off gate: a write happens only after a human says yes,
mediated by a stubbed daemon context.
"""

from __future__ import annotations

import asyncio

import pytest

from palaver.mcp import server as mcp_server
from palaver.mcp import tools_write
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.write import write_memory
from palaver.store.migrate import connect
from tests._mcp_support import _conn
from tests._mcp_support import short_store as short_store
from tests._mcp_support import store as store

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


def test_a_refusal_from_the_daemon_is_not_reported_as_a_successful_correction(store, monkeypatch):
    """The branch a live daemon almost never takes, and must not get wrong.

    Consent given and the daemon reachable is the path where everything
    looks fine; if the daemon then refuses -- the memory was superseded by
    someone else in between, the store is inconsistent -- reporting its
    `{"ok": false}` verbatim as the tool's result reads as success, and the
    caller believes a correction landed that did not.

    The refusal is injected because provoking it against a real daemon means
    winning a race deliberately, and a test that depended on winning one
    would be the flaky kind.
    """
    db_path, seeded = store
    monkeypatch.setattr(
        tools_write,
        "request",
        lambda *_args, **_kwargs: {"ok": False, "error": "IntegrityError", "detail": "refused"},
    )

    with pytest.raises(tools_write.WriteRefused) as excinfo:
        _correct(db_path, _StubContext(), seeded["memory_id"], "the daemon will refuse this")
    assert "IntegrityError" in str(excinfo.value)
    assert "refused" in str(excinfo.value)


def test_an_accepted_reply_from_the_daemon_is_returned_to_the_caller(store, monkeypatch):
    """The positive control for the refusal above.

    Without it, a `correct` that raised on *every* reply would pass the test
    above and never write anything at all.
    """
    db_path, seeded = store
    monkeypatch.setattr(
        tools_write,
        "request",
        lambda *_args, **_kwargs: {"ok": True, "memory_id": 99, "supersedes": seeded["memory_id"]},
    )

    result = _correct(db_path, _StubContext(note="looks right"), seeded["memory_id"], "corrected")
    assert result["ok"] is True
    assert result["memory_id"] == 99
    assert result["note"] == "looks right"
    # The caller sees what they replaced, so a correction is auditable from
    # the reply alone without a second round trip to read the old row.
    assert result["superseded_statement"] == "the first session decided to use SQLite"
