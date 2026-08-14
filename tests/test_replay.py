"""Tests for the replay harness (task 2.5): fixture -> adapter -> signals -> events -> memory.

Every negative assertion here is paired with the positive control that makes
it meaningful, per the task's own requirement: a harness that silently does
nothing would satisfy "the second pass writes zero new rows" and "two
replays dump identically" just as well as a real one, so each such test also
asserts the first pass wrote something nonzero, or that the dump actually
contains fixture-derived content.

`DEFAULT_DB_PATH` (a fixed path under the OS temp directory, `palaver/cli/replay.py`)
is deliberately never used here: every test passes its own `tmp_path`-scoped
`--db`/`db_path` so tests never share mutable state with each other, with a
previous run of this same suite, or with a manual `palaver replay` invocation
on the same machine. The one place the default matters — the plan's literal
done-when command, `uv run palaver replay tests/fixtures/waiting-for-user-reply.jsonl`
with no `--db` flag — is exercised by hand for the report this task requires,
not by a test that would otherwise accumulate state across CI runs.
"""

from __future__ import annotations

import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from palaver.cli import replay as replay_cli
from palaver.memory.evidence import resolve_evidence
from palaver.memory.tiers import TIER_OBSERVER_INFERENCE
from palaver.observer.signals import Status
from palaver.replay import dump_database, replay
from palaver.store.migrate import connect

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
FIXTURE = FIXTURES_DIR / "waiting-for-user-reply.jsonl"
BOOKKEEPING_FIXTURE = FIXTURES_DIR / "bookkeeping-only.jsonl"

#: Fixed reference time so nothing here depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

_TABLES = (
    "projects",
    "sessions",
    "transcript_chunks",
    "events",
    "memories",
    "memory_evidence",
    "memory_relationships",
    "current_state",
    "model_runs",
)


def _row_counts(conn) -> dict[str, int]:
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in _TABLES}


# --- adapter, signals, events, memory: one pass -----------------------------


def test_replay_writes_projects_sessions_chunks_events_and_a_memory(tmp_path):
    """First pass over the fixture: every layer the plan names actually wrote something.

    `waiting-for-user-reply.jsonl` (see `tests/fixtures/README.md`) has four
    message-bearing records and no bookkeeping tail, so this asserts exact
    counts rather than just "more than zero" -- the precise numbers are what
    make `test_replay_second_pass_writes_zero_new_rows`'s "zero new rows"
    meaningful rather than vacuous: a harness that silently does nothing
    would also report zero new rows on its second pass.
    """
    db_path = tmp_path / "replay.db"

    result = replay(FIXTURE, db_path, now=NOW)

    assert result.status is Status.AWAITING_HUMAN  # tests/fixtures/README.md ground truth
    assert result.events_written == 4
    assert result.chunks_written == 4
    assert result.memories_written == 1

    conn = connect(db_path)
    try:
        counts = _row_counts(conn)
    finally:
        conn.close()
    assert counts == {
        "projects": 1,
        "sessions": 1,
        "transcript_chunks": 4,
        "events": 4,
        "memories": 1,
        "memory_evidence": 1,
        "memory_relationships": 0,
        "current_state": 0,
        "model_runs": 0,
    }


def test_replay_second_pass_writes_zero_new_rows(tmp_path):
    """Idempotency: replaying the identical, unmodified fixture a second time
    against the same database adds nothing.

    Paired with the first pass's nonzero counts above -- both asserted in
    this same test -- so this is not vacuous. Mechanism: idempotency comes
    from `palaver.ingest.cursors.CursorStore`, not a second, independent
    dedup check. The persisted cursor for this session already sits at
    end-of-file after the first `replay()` call, so `ClaudeCodeAdapter.tail()`'s
    second call reads zero new complete records from the (unmodified)
    fixture, and every write in `replay()` -- events, transcript_chunks, and
    the one memory -- is downstream of that same record loop.
    """
    db_path = tmp_path / "replay.db"

    first = replay(FIXTURE, db_path, now=NOW)
    assert first.events_written == 4  # positive control, see test above
    assert first.memories_written == 1

    conn = connect(db_path)
    try:
        before = _row_counts(conn)
    finally:
        conn.close()
    assert sum(before.values()) > 0

    second = replay(FIXTURE, db_path, now=NOW)

    assert second.events_written == 0
    assert second.chunks_written == 0
    assert second.memories_written == 0

    conn = connect(db_path)
    try:
        after = _row_counts(conn)
    finally:
        conn.close()
    assert after == before


def test_replay_of_a_bookkeeping_only_fixture_writes_no_memory(tmp_path):
    """A fixture with no message-bearing records (only `mode`/`ai-title`
    bookkeeping) writes events but no chunk and no memory.

    This is what proves the memory gate is a real condition (`session_is_new
    and first_chunk_id is not None`) rather than decoration that happens to
    coincide with every other fixture in the corpus: `bookkeeping-only.jsonl`
    is a *new* session on its first replay, but there is no transcript chunk
    to anchor evidence to, so no memory can legitimately be written.
    """
    db_path = tmp_path / "replay.db"

    result = replay(BOOKKEEPING_FIXTURE, db_path, now=NOW)

    assert result.status is Status.UNKNOWN  # tests/fixtures/README.md ground truth
    assert result.events_written == 2  # "mode" and "ai-title" records
    assert result.chunks_written == 0
    assert result.memories_written == 0

    conn = connect(db_path)
    try:
        counts = _row_counts(conn)
    finally:
        conn.close()
    assert counts["events"] == 2
    assert counts["transcript_chunks"] == 0
    assert counts["memories"] == 0


# --- byte-identical dumps ----------------------------------------------------


def test_replay_two_independent_runs_produce_identical_database_dumps(tmp_path):
    """Two full replays into two separate, fresh databases dump identically.

    Determinism note: `memories.created_at`/`memory_evidence.created_at` are
    wall-clock defaults from `palaver/store/schema.py` -- `write_memory`
    (task 2.1) has no parameter to override them, and this harness does not
    fork that INSERT to add one. `dump_database` normalizes every known
    timestamp column to a fixed sentinel instead, uniformly across every
    table (not a special case for just these two), so the same policy
    applies everywhere. Nothing else is excluded: ids, statements, tiers,
    payloads, and evidence offsets are compared for real, so this is not a
    dump of nothing.
    """
    first_db = tmp_path / "first" / "replay.db"
    second_db = tmp_path / "second" / "replay.db"

    replay(FIXTURE, first_db, now=NOW)
    replay(FIXTURE, second_db, now=NOW)

    conn_a = connect(first_db)
    conn_b = connect(second_db)
    try:
        dump_a = dump_database(conn_a)
        dump_b = dump_database(conn_b)
    finally:
        conn_a.close()
        conn_b.close()

    assert dump_a == dump_b
    assert dump_a != ""
    assert dump_a.count("transcript_chunks: ") == 4
    assert dump_a.count("events: ") == 4
    assert "memories: " in dump_a
    assert "waiting-for-user-reply" in dump_a  # the memory's statement, not just a count
    assert "<TIMESTAMP>" in dump_a  # normalization actually fired, not a no-op


# --- INV-2: the fixture is opened read-only and never mutated ---------------


def test_replay_never_mutates_the_fixture_file(tmp_path):
    """The fixture's bytes and mtime are unchanged after replay (INV-2).

    `ClaudeCodeAdapter.tail`/`observe_session` already route every read
    through `open_source_readonly` (O_RDONLY) at the modules they belong to;
    this checks the observable outcome at this module's own boundary, since
    task 2.5 is the first code path that drives both of them against a real
    file from application code (rather than another test).
    """
    before_bytes = FIXTURE.read_bytes()
    before_mtime = FIXTURE.stat().st_mtime

    replay(FIXTURE, tmp_path / "replay.db", now=NOW)

    assert FIXTURE.read_bytes() == before_bytes
    assert FIXTURE.stat().st_mtime == before_mtime


# --- memory layer: tier and evidence resolve through the real stack --------


def test_replay_memory_is_tier_observer_inference_with_resolvable_evidence(tmp_path):
    """The memory replay writes is tier-4 (observer inference, not a model
    claim or a literal quote) and its evidence anchor resolves back to the
    exact transcript_chunks content it was written against, live, through
    `palaver.memory.evidence.resolve_evidence` -- exercising task 2.1's
    writer, task 2.2's resolver, and this module's anchor together, not just
    row counts.
    """
    db_path = tmp_path / "replay.db"
    replay(FIXTURE, db_path, now=NOW)

    conn = connect(db_path)
    try:
        memory_id, tier, statement = conn.execute(
            "SELECT id, tier, statement FROM memories"
        ).fetchone()
        assert tier == TIER_OBSERVER_INFERENCE
        assert "waiting-for-user-reply" in statement

        (evidence_id,) = conn.execute(
            "SELECT id FROM memory_evidence WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        resolved = resolve_evidence(conn, evidence_id)
        assert resolved != ""

        # The anchor points at the *first* transcript_chunks row (this
        # replay's earliest ingested record), not an arbitrary span.
        (first_chunk_content,) = conn.execute(
            "SELECT content FROM transcript_chunks ORDER BY id ASC LIMIT 1"
        ).fetchone()
        assert resolved == first_chunk_content
    finally:
        conn.close()


# --- CLI: palaver replay -----------------------------------------------------


def test_default_db_path_is_outside_the_repository():
    """`DEFAULT_DB_PATH` must never risk a stray file landing inside version
    control -- a pure path assertion, no I/O, independent of any other test.
    """
    assert not str(replay_cli.DEFAULT_DB_PATH.resolve()).startswith(str(REPO_ROOT.resolve()))


def test_cli_replay_run_reports_status_events_chunks_and_memories(tmp_path):
    args = SimpleNamespace(fixture=FIXTURE, db=tmp_path / "replay.db")
    out = io.StringIO()

    exit_code = replay_cli.run(args, out=out, now=NOW)

    assert exit_code == 0
    text = out.getvalue()
    assert "status: AWAITING_HUMAN" in text
    assert "events written: 4" in text
    assert "chunks written: 4" in text
    assert "memories written: 1" in text


def test_cli_replay_unreadable_fixture_is_a_diagnosable_error_not_a_crash(tmp_path, capsys):
    args = SimpleNamespace(fixture=tmp_path / "does-not-exist.jsonl", db=tmp_path / "replay.db")

    exit_code = replay_cli.run(args, now=NOW)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does-not-exist.jsonl" in captured.err


def test_console_script_runs_replay_against_the_fixture_named_in_the_plan(tmp_path):
    """The literal acceptance command from the plan's done-when, end to end
    through the installed console script, with progress on stderr and the
    result on stdout (INV-1).

    Uses `--db tmp_path/...`, not `DEFAULT_DB_PATH`: this test's job is to
    exercise the console script wiring end to end, and row counts / status
    for a guaranteed-fresh database are already covered directly against
    `replay()` and `replay_cli.run()` above. The literal no-`--db` command
    is additionally run by hand for this task's report.
    """
    script = Path(sys.executable).parent / "palaver"
    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run(
        [str(script), "replay", str(FIXTURE), "--db", str(tmp_path / "replay.db")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "session: fixtures/waiting-for-user-reply" in result.stdout
    assert "status: AWAITING_HUMAN" in result.stdout
    assert "replayed" in result.stderr
    assert "replayed" not in result.stdout


def test_console_script_help_lists_replay(tmp_path):
    script = Path(sys.executable).parent / "palaver"
    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0
    assert "replay" in result.stdout
