"""
Store growth: samples, projection, and the per-table report, all read out of a synthetic
growth store.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import palaver.bench as palaver_bench
from palaver.bench import (
    GROWTH_TABLES,
    BenchReport,
    GrowthSample,
    growth_samples,
    measure_slot_files,
    measure_tables,
    project_growth,
    store_bytes,
)
from palaver.cli import bench as bench_cli
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.tiers import TIER_OBSERVED_RESULT
from palaver.memory.write import write_memory
from palaver.store.migrate import connect, migrate
from tests._bench_support import SESSIONS, TEST_PROMPT_WORDS, _cli, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# Store growth: measured byte deltas, never a row count times a constant
# =============================================================================


def _samples(*points: tuple[str, int, int]) -> list[GrowthSample]:
    """Build cumulative samples from `(day, cumulative_rows, cumulative_bytes)`."""
    return [
        GrowthSample(day=day, cumulative_rows=rows, cumulative_payload_bytes=payload)
        for day, rows, payload in points
    ]


def _growth_store(tmp_path: Path):
    db_path = tmp_path / "growth.db"
    migrate(db_path)
    conn = connect(db_path)
    project_id = conn.execute(
        "INSERT INTO projects(name, path) VALUES ('growth', '/tmp/fixture/growth')"
    ).lastrowid
    session_id = conn.execute(
        "INSERT INTO sessions(project_id, source, external_id) VALUES (?, 'bench', 'growth-1')",
        (project_id,),
    ).lastrowid
    return db_path, conn, project_id, session_id


def test_growth_projection_reads_bytes_and_ignores_row_counts():
    """Bullet 3: linear in the measured byte delta, blind to row counts.

    Two halves, because either alone is satisfiable by the wrong function. The
    first proves the projection *moves* with bytes. The second proves it does
    not move with rows — a `rows x constant` implementation passes the first
    and fails the second.
    """
    lean = project_growth(
        _samples(("2026-08-01", 100, 1_000), ("2026-08-11", 200, 3_000)),
        current_bytes=3_000,
    )
    fat = project_growth(
        _samples(("2026-08-01", 100, 1_000), ("2026-08-11", 200, 9_000)),
        current_bytes=9_000,
    )
    assert fat.bytes_per_day > lean.bytes_per_day

    # Same bytes, wildly different row counts, over the same span.
    few_rows = project_growth(
        _samples(("2026-08-01", 2, 1_000), ("2026-08-11", 4, 3_000)), current_bytes=3_000
    )
    many_rows = project_growth(
        _samples(("2026-08-01", 5_000, 1_000), ("2026-08-11", 90_000, 3_000)),
        current_bytes=3_000,
    )
    assert few_rows.bytes_per_day == many_rows.bytes_per_day
    assert few_rows.horizons == many_rows.horizons


def test_growth_projection_is_linear_in_the_horizon():
    projection = project_growth(
        _samples(("2026-08-01", 10, 1_000), ("2026-08-11", 20, 3_000)), current_bytes=3_000
    )

    current = projection.current_bytes
    thirty = projection.at(30) - current
    ninety = projection.at(90) - current

    assert ninety == pytest.approx(thirty * 3, rel=1e-6)
    assert thirty == pytest.approx(projection.bytes_per_day * 30, rel=1e-6)


def test_growth_projection_refuses_fewer_than_two_measurement_points():
    with pytest.raises(ValueError, match="at least two measurement points"):
        project_growth([], current_bytes=1_000)
    with pytest.raises(ValueError, match="at least two measurement points"):
        project_growth(_samples(("2026-08-01", 10, 1_000)), current_bytes=1_000)
    # Positive control: two points on different days project.
    assert (
        project_growth(
            _samples(("2026-08-01", 10, 1_000), ("2026-08-02", 20, 2_000)), current_bytes=2_000
        ).samples_used
        == 2
    )


def test_growth_projection_refuses_a_span_shorter_than_a_day():
    """Two readings from the same day are two readings, not a trend."""
    with pytest.raises(ValueError, match="spanning at least one day"):
        project_growth(
            _samples(("2026-08-01", 10, 1_000), ("2026-08-01", 20, 5_000)), current_bytes=5_000
        )


def test_growth_samples_accumulate_across_days_and_tables(tmp_path):
    db_path, conn, project_id, session_id = _growth_store(tmp_path)
    try:
        for index, (day, text) in enumerate(
            [
                ("2026-08-01", "aaaa"),
                ("2026-08-01", "bb"),
                ("2026-08-03", "cccccc"),
            ]
        ):
            conn.execute(
                "INSERT INTO transcript_chunks(session_id, seq, role, content, created_at) "
                "VALUES (?, ?, 'user', ?, ?)",
                (session_id, index, text, f"{day}T09:00:00.000Z"),
            )
        conn.execute(
            "INSERT INTO events(session_id, kind, payload, created_at) VALUES (?, 'k', ?, ?)",
            (session_id, "dd", "2026-08-03T10:00:00.000Z"),
        )
        conn.commit()

        samples = growth_samples(conn)
    finally:
        conn.close()

    assert [sample.day for sample in samples] == ["2026-08-01", "2026-08-03"]
    assert [sample.cumulative_rows for sample in samples] == [2, 4]
    # 4 + 2 = 6 on the first day; + 6 + 2 = 14 cumulative on the third.
    assert [sample.cumulative_payload_bytes for sample in samples] == [6, 14]


def test_growth_is_measured_over_exactly_the_append_only_tables():
    assert [table for table, _ in GROWTH_TABLES] == [
        "transcript_chunks",
        "events",
        "memories",
        "memory_evidence",
    ]
    # memory_evidence stores offsets, not a copy of the quote (migration 4),
    # so it has no text column to sum.
    assert dict(GROWTH_TABLES)["memory_evidence"] is None


def test_growth_of_memory_evidence_is_counted_in_rows_not_text(tmp_path):
    db_path, conn, project_id, session_id = _growth_store(tmp_path)
    try:
        content = "fixture: invented line supporting an invented claim"
        chunk_id = conn.execute(
            "INSERT INTO transcript_chunks(session_id, seq, role, content) "
            "VALUES (?, 1, 'user', ?)",
            (session_id, content),
        ).lastrowid
        write_memory(
            conn,
            project_id=project_id,
            session_id=session_id,
            statement="fixture: invented claim",
            origin="observer",
            tier=TIER_OBSERVED_RESULT,
            evidence=[
                EvidenceAnchor(
                    transcript_chunk_id=chunk_id, start_offset=0, end_offset=len(content)
                )
            ],
        )
        conn.commit()
        usage = {entry.table: entry for entry in measure_tables(conn)}
        total = store_bytes(conn)
    finally:
        conn.close()

    assert usage["memory_evidence"].rows == 1
    assert usage["memory_evidence"].payload_bytes == 0
    # Positive control: a table that does store text reports it, so the zero
    # above is about this table's shape and not about a broken measurement.
    assert usage["transcript_chunks"].payload_bytes == len(content)
    assert total > 0


def test_growth_page_bytes_are_measured_when_dbstat_is_available(tmp_path):
    db_path, conn, project_id, session_id = _growth_store(tmp_path)
    try:
        conn.execute(
            "INSERT INTO transcript_chunks(session_id, seq, role, content) "
            "VALUES (?, 1, 'user', ?)",
            (session_id, "fixture: invented line"),
        )
        conn.commit()
        usage = {entry.table: entry for entry in measure_tables(conn)}
    finally:
        conn.close()

    pages = usage["transcript_chunks"].page_bytes
    # Either a real measurement, or the explicit sentinel — never a bare 0
    # that would read as "this table occupies nothing".
    assert pages > 0 or pages == -1
    if pages > 0:
        assert pages >= usage["transcript_chunks"].payload_bytes


def test_growth_report_carries_per_table_bytes_and_both_horizons(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")
    projected = report.__class__(
        **{
            **report.__dict__,
            "projection": project_growth(
                _samples(("2026-08-01", 10, 1_000), ("2026-08-11", 20, 3_000)),
                current_bytes=report.store_total_bytes or 4096,
            ),
        }
    )

    rendered = bench_cli.render_report(projected, host=handle.host, port=handle.port, detailed=True)

    for table, _ in GROWTH_TABLES:
        assert re.search(rf"{table}: \d+ row\(s\), [0-9.]+ KiB of text", rendered), rendered
    assert re.search(r"30-day projection: [0-9.]+ MiB", rendered)
    assert re.search(r"90-day projection: [0-9.]+ MiB", rendered)


def test_growth_horizons_say_why_when_the_store_is_too_young(stub_server, tmp_path):
    """A fresh store gets the labelled lines and a reason, never a number."""
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    rendered = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=False)

    assert report.projection is None
    assert report.projection_detail
    for days in (30, 90):
        assert f"{days}-day projection: unavailable — " in rendered
    assert not re.search(r"\d+-day projection: [0-9.]+ MiB", rendered)


def test_growth_reports_page_bytes_as_unavailable_when_dbstat_is_absent(monkeypatch, tmp_path):
    """`dbstat` is a compile-time option, and its absence is not a zero.

    Simulated rather than waited for: this SQLite build has `dbstat`, so the
    branch that handles a build without it would otherwise never run, and a
    sentinel of 0 would be indistinguishable from a measured 0 in the report.
    """
    db_path, conn, project_id, session_id = _growth_store(tmp_path)
    try:
        conn.execute(
            "INSERT INTO transcript_chunks(session_id, seq, role, content) "
            "VALUES (?, 1, 'user', 'fixture: invented line')",
            (session_id,),
        )
        conn.commit()
        monkeypatch.setattr(palaver_bench.growth, "_page_bytes_by_table", lambda _conn: {})
        usage = measure_tables(conn)
    finally:
        conn.close()

    assert all(entry.page_bytes < 0 for entry in usage), usage

    report = BenchReport(
        sessions=1,
        tick_interval_s=30.0,
        tick_wall_s=0.1,
        peak_in_flight=1,
        timings=(),
        rss_before_bytes=1,
        rss_after_bytes=1,
        slot_files=measure_slot_files(None),
        unreachable=False,
        tables=usage,
        store_total_bytes=4096,
        projection_detail="fixture: invented reason",
    )
    rendered = bench_cli.render_report(report, host="127.0.0.1", port=1, detailed=False)

    assert "pages unavailable (no dbstat in this SQLite build)" in rendered
    assert "0.0 KiB of pages" not in rendered


def test_growth_db_flag_measures_the_named_store_not_the_benchmark_one(
    stub_server, tmp_path, capsys
):
    """`--growth-db` is wired end to end, not merely accepted by the parser.

    The failure this excludes is a name mismatch between `args.growth_db` and
    `run_bench(growth_db_path=...)`, which would silently measure the throwaway
    benchmark store while the report claimed otherwise — and which every other
    growth test passes through, because none of them set the flag.
    """
    other = tmp_path / "other.db"
    migrate(other)
    conn = connect(other)
    try:
        project_id = conn.execute(
            "INSERT INTO projects(name, path) VALUES ('other', '/tmp/fixture/other')"
        ).lastrowid
        session_id = conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (?,'bench','other-1')",
            (project_id,),
        ).lastrowid
        for seq in range(200):
            conn.execute(
                "INSERT INTO transcript_chunks(session_id, seq, role, content) "
                "VALUES (?, ?, 'user', ?)",
                (session_id, seq, "fixture: invented line " * 20),
            )
        conn.commit()
    finally:
        conn.close()

    handle = stub_server()
    argv = [
        "bench",
        "--sessions",
        "1",
        "--report",
        "--host",
        handle.host,
        "--port",
        str(handle.port),
        "--prompt-words",
        str(TEST_PROMPT_WORDS),
        "--db",
        str(tmp_path / "bench.db"),
    ]

    assert _cli([*argv, "--growth-db", str(other)], sys.stdout) == 0
    pointed = capsys.readouterr().out
    assert _cli(argv, sys.stdout) == 0
    default = capsys.readouterr().out

    assert "transcript_chunks: 200 row(s)" in pointed
    # The control: without the flag the same command measures the benchmark
    # store, which holds no transcript chunks at all.
    assert "transcript_chunks: 0 row(s)" in default
