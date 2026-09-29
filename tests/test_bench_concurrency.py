"""
Sessions are driven concurrently, not serially, and a straggler is reported as a failure
rather than hung on.
"""

from __future__ import annotations

from palaver.bench import (
    BenchReport,
)
from palaver.cli import bench as bench_cli
from tests._bench_support import BARRIER_TIMEOUT, SESSIONS, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# Concurrency: the property a serial loop cannot fake
# =============================================================================


def test_six_sessions_are_driven_concurrently_not_serially(stub_server, tmp_path):
    """All six requests are outstanding at one instant, proven from both sides.

    The barrier proves the requests genuinely reached the server together; the
    harness's own peak-in-flight gauge proves it issued them together. Either
    alone is weaker: a gauge could count threads that never got a socket, and
    a barrier says nothing about what the harness reports.
    """
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, tmp_path / "bench.db")

    assert report.peak_in_flight == SESSIONS
    assert not handle.barrier_broke
    assert handle.request_count == SESSIONS
    assert report.ok
    assert len(report.successful_latencies_ms) == SESSIONS


def test_a_concurrent_round_that_arrives_late_is_reported_as_a_failure(stub_server, tmp_path):
    """The barrier's failure path is live, not merely configured.

    Five parties against six requests can never complete, so the barrier times
    out. Without this control, `not handle.barrier_broke` above would pass
    against a barrier that was never actually engaged.
    """
    handle = stub_server(barrier_parties=SESSIONS + 1)

    report = _run(handle, tmp_path / "bench.db", timeout=BARRIER_TIMEOUT + 20)

    assert handle.barrier_broke
    assert not report.ok
    assert not report.unreachable


def test_a_concurrent_round_fits_inside_a_configured_tick_interval(stub_server, tmp_path):
    """Bullet 4, with the paired overrun that makes it non-vacuous.

    Judged against a tick interval passed in, not against `DEFAULT_TICK_INTERVAL`
    — six instant requests fit inside 30 s no matter how the harness is
    written, so asserting against the default would prove nothing. The same
    stub and the same round are then judged against a budget they cannot meet,
    which proves the harness can report an overrun at all. A benchmark that can
    only ever say "fits" is not a measurement.
    """
    handle = stub_server(barrier_parties=SESSIONS, delay=0.2)

    report = _run(handle, tmp_path / "bench.db", tick_interval=10.0)

    assert report.fits_tick_interval
    assert report.tick_wall_s <= report.tick_interval_s
    # Concurrent, so the round costs about one delay, not six.
    assert report.tick_wall_s < 0.2 * SESSIONS

    overrun = BenchReport(
        sessions=report.sessions,
        tick_interval_s=0.01,
        tick_wall_s=report.tick_wall_s,
        peak_in_flight=report.peak_in_flight,
        timings=report.timings,
        rss_before_bytes=report.rss_before_bytes,
        rss_after_bytes=report.rss_after_bytes,
        slot_files=report.slot_files,
        unreachable=False,
    )
    assert not overrun.fits_tick_interval
    rendered = bench_cli.render_report(overrun, host="127.0.0.1", port=1, detailed=False)
    assert "OVER" in rendered
