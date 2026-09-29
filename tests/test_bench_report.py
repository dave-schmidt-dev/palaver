"""
The report: RSS and wall time are numbers, RSS is bytes not the platform unit, and the
per-session table is opt-in.
"""

from __future__ import annotations

import re
import sys

import palaver.bench as palaver_bench
from palaver.bench import (
    peak_rss_bytes,
)
from palaver.cli import bench as bench_cli
from tests._bench_support import SESSIONS, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# The report: numbers a human reads, and units that are not silently wrong
# =============================================================================


def test_the_report_emits_peak_rss_and_per_tick_wall_time_as_numbers(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    rendered = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=True)

    wall = re.search(r"tick wall time: ([0-9]+\.[0-9]+) s", rendered)
    rss = re.search(r"peak RSS: ([0-9]+\.[0-9]+) MiB", rendered)
    assert wall and float(wall.group(1)) >= 0.0
    assert rss and float(rss.group(1)) > 0.0


def test_peak_rss_is_normalized_to_bytes_not_the_platform_unit(monkeypatch):
    """A 1024x-wrong memory number reads as a measurement, not as a bug.

    Both branches are exercised regardless of which platform the suite runs
    on. Comparing a live reading against a live reading would agree with
    whichever unit the module happened to pick, which is the whole bug.
    """
    monkeypatch.setattr(palaver_bench.resource_usage, "_RSS_IN_BYTES", True)
    assert palaver_bench.normalize_rss(4096) == 4096
    monkeypatch.setattr(palaver_bench.resource_usage, "_RSS_IN_BYTES", False)
    assert palaver_bench.normalize_rss(4096) == 4096 * 1024

    monkeypatch.undo()
    # And the flag itself is right for *this* platform, checked empirically
    # rather than by restating the source line. A pytest process occupies
    # somewhere between a few MiB and a few GiB; getting the unit backwards
    # puts the normalized figure a factor of 1024 outside that band in
    # whichever direction the mistake was made, on either platform.
    live = peak_rss_bytes()
    assert 4 * 1024 * 1024 < live < 4 * 1024 * 1024 * 1024, f"implausible peak RSS: {live} bytes"
    assert sys.platform in {"darwin", "linux"}


def test_the_rss_delta_is_the_attributable_number(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    assert report.rss_after_bytes >= report.rss_before_bytes
    assert report.rss_delta_bytes == report.rss_after_bytes - report.rss_before_bytes
    # `ru_maxrss` is a never-reset high-water mark, so the endpoint is much
    # larger than the growth. Asserting only on the endpoint would let a report
    # attribute the whole interpreter to the benchmark.
    assert report.rss_before_bytes > report.rss_delta_bytes


def test_the_report_flag_adds_the_per_session_table(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    detailed = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=True)
    summary = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=False)

    assert "bench-session-1" in detailed
    assert "bench-session-1" not in summary
    # Both still carry the measurement itself: the flag widens the output, it
    # does not gate the run.
    assert "peak in flight: 6" in summary
