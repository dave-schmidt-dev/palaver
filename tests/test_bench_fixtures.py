"""
Fixture synthesis and store bookkeeping: sessions are reused not duplicated, and zero of
either is rejected.
"""

from __future__ import annotations

import pytest

from palaver.bench import (
    SessionTiming,
    run_bench,
    synthesize_sessions,
    synthetic_prompt,
)
from palaver.store.migrate import connect, migrate
from tests._bench_support import SESSIONS, TEST_PROMPT_WORDS, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# Fixture synthesis and store bookkeeping
# =============================================================================


def test_synthesized_sessions_are_reused_rather_than_duplicated(tmp_path):
    db_path = tmp_path / "bench.db"
    migrate(db_path)
    conn = connect(db_path)
    try:
        first = synthesize_sessions(conn, SESSIONS)
        second = synthesize_sessions(conn, SESSIONS)
        conn.commit()
        assert first == second
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == SESSIONS
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
    finally:
        conn.close()


def test_synthesizing_zero_sessions_is_rejected(tmp_path):
    db_path = tmp_path / "bench.db"
    migrate(db_path)
    conn = connect(db_path)
    try:
        with pytest.raises(ValueError, match="must be positive"):
            synthesize_sessions(conn, 0)
        # Positive control: one session is accepted on the same connection.
        assert len(synthesize_sessions(conn, 1)) == 1
    finally:
        conn.close()


def test_every_session_records_a_model_runs_row(stub_server, tmp_path):
    db_path = tmp_path / "bench.db"
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, db_path)
    assert report.ok

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT session_id, status FROM model_runs WHERE purpose = 'bench-extraction'"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == SESSIONS
    assert {status for _, status in rows} == {"done"}
    assert {session_id for session_id, _ in rows} == {
        timing.session_id for timing in report.timings
    }


def test_a_zero_word_prompt_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        synthetic_prompt(0)
    # Positive control: a small prompt is built, and carries its label so two
    # concurrent requests never present the server with identical input.
    prompt = synthetic_prompt(TEST_PROMPT_WORDS, label="bench-session-3")
    assert prompt.startswith("bench-session-3")
    assert "invented benchmark transcript line" in prompt


def test_zero_sessions_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        run_bench(db_path=tmp_path / "bench.db", sessions=0)


def test_a_timing_reports_its_own_success(tmp_path):
    assert SessionTiming(label="a", session_id=1, latency_ms=5).ok
    assert not SessionTiming(
        label="a", session_id=1, latency_ms=5, error="boom", error_kind="response"
    ).ok
