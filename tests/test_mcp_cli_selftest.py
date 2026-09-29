"""
The `mcp` CLI subcommand and its --selftest: concurrent clients over the real transport,
failure reporting, usage errors.
"""

from __future__ import annotations

import io
from argparse import Namespace

from palaver.cli import SUBCOMMANDS
from palaver.cli import mcp as mcp_cli
from palaver.mcp import server as mcp_server
from palaver.mcp import tools_read
from tests._mcp_support import _conn

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
