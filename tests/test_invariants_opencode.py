"""
INV-3: OpenCode's account/credential tables are unreachable through the guard, and the
allowlist -- not mode=ro -- is what blocks them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from palaver.ingest.adapters import opencode_guard

# Invented, obviously-fake token values for the fixture db. Never real.
FIXTURE_ACCESS_TOKEN = "invented-access-token-not-real-c0ffee"
FIXTURE_REFRESH_TOKEN = "invented-refresh-token-not-real-decaf0"


# =============================================================================
# INV-3 — OpenCode `account`/`credential` unreachable through the guard
# =============================================================================


def _build_fixture_opencode_db(path: Path) -> None:
    """Create a fixture SQLite db mirroring OpenCode's real schema.

    Table shapes follow `docs/research.md` section 3 (verified against the
    real store): `session` + `project` for identity, `message` + `part` for
    turn content, `account` + `credential` for OAuth material. Every value
    inserted here is invented for this test; none of it is real.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT);
        CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT);
        CREATE TABLE account (id TEXT PRIMARY KEY, access_token TEXT, refresh_token TEXT);
        CREATE TABLE credential (id TEXT PRIMARY KEY, access_token TEXT, refresh_token TEXT);
        """
    )
    conn.execute(
        "INSERT INTO session VALUES (?, ?)",
        ("fixture-session-1", "/tmp/fixture-project"),
    )
    conn.execute(
        "INSERT INTO account VALUES (?, ?, ?)",
        ("fixture-account-1", FIXTURE_ACCESS_TOKEN, FIXTURE_REFRESH_TOKEN),
    )
    conn.execute(
        "INSERT INTO credential VALUES (?, ?, ?)",
        ("fixture-credential-1", FIXTURE_ACCESS_TOKEN, FIXTURE_REFRESH_TOKEN),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_opencode_db(tmp_path: Path) -> Path:
    path = tmp_path / "opencode-fixture.db"
    _build_fixture_opencode_db(path)
    return path


@pytest.mark.inv3
def test_allowed_tables_excludes_account_and_credential():
    """Catches `account`/`credential` being added to the allowlist by mistake."""
    assert "account" not in opencode_guard.ALLOWED_TABLES
    assert "credential" not in opencode_guard.ALLOWED_TABLES
    assert opencode_guard.ALLOWED_TABLES == {"session", "project", "message", "part"}


@pytest.mark.inv3
def test_opencode_credential_tables_unreachable(fixture_opencode_db):
    """A query naming `credential` (or `account`) raises before any SQL executes,
    and the allowlist — not the read-only flag — is what raises it.

    The bullet's own positive control comes first: a raw, unguarded
    read-only `sqlite3` connection against the identical fixture *succeeds*
    in reaching the credential table, proving the table is genuinely
    reachable and the fixture is not accidentally empty or malformed. Only
    then does the guarded connection's failure mean anything.

    "Before any SQL executes" is pinned to the authorizer specifically, not
    just to "the call raised": a guard that instead ran the SELECT, fetched
    rows, and raised only after inspecting the table name would make an
    unqualified `pytest.raises(sqlite3.DatabaseError)` pass identically. The
    message SQLite's own authorizer produces on denial —
    `"access to <table>.<column> is prohibited"` — only exists on the
    compile-time rejection path, so asserting it is present rules out a
    post-hoc check that happened to also raise `DatabaseError`.

    LAYER PROOF follows on the same connection object: `mode=ro` never
    changes; only the authorizer is stripped (`set_authorizer(None)`), and
    the identical query that just raised now succeeds. If `mode=ro` were
    what had blocked it, stripping the allowlist could not have changed the
    outcome. A write attempt on that same de-allowlisted connection still
    fails, proving `mode=ro` was independently in force the whole time
    rather than one layer silently subsuming the other.
    """
    raw_conn = sqlite3.connect(opencode_guard.readonly_uri(fixture_opencode_db), uri=True)
    try:
        rows = raw_conn.execute("SELECT access_token FROM credential").fetchall()
        assert rows == [(FIXTURE_ACCESS_TOKEN,)]
    finally:
        raw_conn.close()

    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited") as excinfo:
            conn.execute("SELECT access_token FROM credential")
        assert "credential" in str(excinfo.value)

        with pytest.raises(sqlite3.DatabaseError, match="prohibited") as excinfo:
            conn.execute("SELECT access_token, refresh_token FROM account")
        assert "account" in str(excinfo.value)

        # LAYER PROOF: strip only the allowlist; mode=ro is untouched.
        conn.set_authorizer(None)
        rows = conn.execute("SELECT access_token FROM credential").fetchall()
        assert rows == [(FIXTURE_ACCESS_TOKEN,)]

        # mode=ro is still independently in force on this same connection.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO session VALUES ('s2', '/tmp/y')")
    finally:
        conn.close()


@pytest.mark.inv3
def test_opencode_allowlist_permits_allowed_tables(fixture_opencode_db):
    """Positive control: the allowlist blocks `credential`/`account` specifically.

    An authorizer that denied every query would also "block" credential —
    for the wrong reason. Reading an allowlisted table through the same
    guarded connection must still work.
    """
    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        rows = conn.execute("SELECT directory FROM session").fetchall()
        assert rows == [("/tmp/fixture-project",)]
    finally:
        conn.close()


@pytest.mark.inv3
def test_opencode_allowlist_blocks_credential_regardless_of_query_shape(fixture_opencode_db):
    """The allowlist is a structural SQLite check, not a text match on the SQL string.

    Varying case and wrapping the reference in a subquery would both slip
    past a naive `"credential" in sql.lower()` substring check; the real
    guard uses SQLite's own authorizer, which resolves the referenced table
    regardless of surface syntax.
    """
    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT * FROM CREDENTIAL")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT * FROM (SELECT access_token FROM credential)")
    finally:
        conn.close()


@pytest.mark.inv3
def test_open_guarded_readonly_uses_mode_ro_uri(fixture_opencode_db, monkeypatch):
    """Asserts the connection string `open_guarded_readonly` actually issues carries `mode=ro`.

    Spies on `sqlite3.connect` itself rather than inspecting a helper in
    isolation, so this proves what is actually passed to the database
    driver, not just that some string somewhere contains the substring.
    """
    captured = {}
    real_connect = sqlite3.connect

    def _spy_connect(database, *args, **kwargs):
        captured["database"] = database
        captured["kwargs"] = kwargs
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _spy_connect)

    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    conn.close()

    assert "mode=ro" in captured["database"]
    assert captured["kwargs"].get("uri") is True
