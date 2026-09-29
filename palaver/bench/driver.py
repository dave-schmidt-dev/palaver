"""
Driving synthesized sessions concurrently on their own threads and assembling the
round's report.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from palaver.extract.client import (
    ModelClient,
    ModelClientError,
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from palaver.observer.daemon import DEFAULT_INTERVAL, DEFAULT_MODEL, extraction_schema
from palaver.store.migrate import connect, migrate

from .growth import _measure_growth
from .prompt_sizing import DEFAULT_PROMPT_FRACTION, _derive_prompt_words, synthetic_prompt
from .records import BenchError, BenchReport, SessionTiming
from .resource_usage import measure_slot_files, peak_rss_bytes

#: Sessions driven concurrently by default. The plan's command line is
#: `palaver bench --sessions 6`; six is the number Phase 4 asks about because
#: it exceeds the four KV slots the observed server was measured to have.
DEFAULT_SESSIONS = 6

#: The tick budget a run is measured against, taken from the daemon's own
#: default so the benchmark and the scheduler cannot drift apart.
DEFAULT_TICK_INTERVAL = DEFAULT_INTERVAL

#: Recorded in `model_runs.purpose`, distinct from the daemon's
#: `observer-extraction`, so benchmark traffic is separable from real
#: extraction traffic in the same table.
BENCH_PURPOSE = "bench-extraction"

#: Seconds each worker's sqlite connection waits for a write lock before
#: raising. Six threads writing `model_runs` rows to one WAL database contend
#: only briefly, but the default of 5 s is a silent failure under load.
_WRITE_LOCK_TIMEOUT = 30.0


class _InFlightGauge:
    """Thread-safe counter of concurrently outstanding requests.

    Measured on the client side on purpose. The stub server in
    `tests/test_bench.py` proves the requests genuinely arrive together via a
    barrier; this gauge proves the *harness* issued them together, which is the
    property a serial loop violates and the one the benchmark reports.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.peak = 0

    @contextmanager
    def track(self) -> Iterator[None]:
        """Count one request as in flight for the duration of the block."""
        with self._lock:
            self._current += 1
            self.peak = max(self.peak, self._current)
        try:
            yield
        finally:
            with self._lock:
                self._current -= 1


def _error_kind(exc: ModelClientError) -> str:
    """Classify a client failure for the report."""
    if isinstance(exc, ModelConnectionError):
        return "connection"
    if isinstance(exc, ModelTimeoutError):
        return "timeout"
    if isinstance(exc, ModelResponseError):
        return "response"
    return "client"


def _worker_connection(db_path: Path) -> sqlite3.Connection:
    """Open this thread's own connection, with room to wait for the write lock.

    Deliberately not `store.migrate.connect`: that helper takes the sqlite3
    default five-second busy timeout, which six threads writing `model_runs`
    rows to one file can exhaust under a slow round. The pragmas are otherwise
    identical to it.
    """
    conn = sqlite3.connect(str(db_path), timeout=_WRITE_LOCK_TIMEOUT)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def synthesize_sessions(conn: sqlite3.Connection, count: int) -> tuple[tuple[int, str], ...]:
    """Write `count` invented sessions and return their `(id, label)` pairs.

    These stand in for observed sessions so the benchmark never needs six real
    open panes, and so nothing it sends the model came from a real transcript.

    Args:
        conn: Open connection to a migrated store. The caller owns committing.
        count: How many sessions to create. Must be positive.

    Returns:
        `(session_id, label)` in creation order.

    Raises:
        ValueError: If `count` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    project_id = conn.execute(
        "INSERT INTO projects(name, path) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET path = excluded.path RETURNING id",
        ("palaver-bench", "/tmp/palaver-bench"),
    ).fetchone()[0]
    created = []
    for index in range(1, count + 1):
        label = f"bench-session-{index}"
        session_id = conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?) "
            "ON CONFLICT(source, external_id) DO UPDATE SET project_id = excluded.project_id "
            "RETURNING id",
            (project_id, "bench", label),
        ).fetchone()[0]
        created.append((session_id, label))
    return tuple(created)


def _drive_one(
    *,
    db_path: Path,
    session_id: int,
    label: str,
    host: str,
    port: int,
    timeout: float,
    model: str,
    prompt: str,
    gauge: _InFlightGauge,
    on_status: Callable[[str], None] | None,
) -> SessionTiming:
    """Send one session's request on this thread and time it."""
    conn = _worker_connection(db_path)
    started = time.monotonic()
    try:
        with gauge.track():
            client = ModelClient(conn, host=host, port=port, timeout=timeout)
            client.complete(
                model=model,
                purpose=BENCH_PURPOSE,
                prompt=prompt,
                schema=extraction_schema(),
                session_id=session_id,
                on_status=on_status,
            )
    except ModelClientError as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        if on_status is not None:
            on_status(f"{label}: failed after {latency_ms} ms — {exc}")
        return SessionTiming(
            label=label,
            session_id=session_id,
            latency_ms=latency_ms,
            error=str(exc),
            error_kind=_error_kind(exc),
        )
    else:
        latency_ms = int((time.monotonic() - started) * 1000)
        if on_status is not None:
            on_status(f"{label}: returned in {latency_ms} ms")
        return SessionTiming(label=label, session_id=session_id, latency_ms=latency_ms)
    finally:
        conn.commit()
        conn.close()


def run_bench(
    *,
    db_path: Path | str,
    sessions: int = DEFAULT_SESSIONS,
    host: str = "127.0.0.1",
    port: int = 8090,
    timeout: float = 60.0,
    tick_interval: float = DEFAULT_TICK_INTERVAL,
    model: str = DEFAULT_MODEL,
    prompt_words: int | None = None,
    prompt_fraction: float = DEFAULT_PROMPT_FRACTION,
    growth_db_path: Path | str | None = None,
    slot_save_path: Path | str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> BenchReport:
    """Drive `sessions` synthesized sessions concurrently and measure the round.

    Every session is dispatched on its own thread before any of them is
    awaited, which is the whole point: the peak in-flight count on the returned
    report is 1 for a serial implementation and `sessions` for this one.

    Args:
        db_path: Store to synthesize sessions into and record `model_runs`
            rows against. Migrated if it does not already exist.
        sessions: How many sessions to drive at once. Must be positive.
        host: Model server host; `127.0.0.1` in any real deployment (INV-9).
        port: Model server port.
        timeout: Seconds each request is allowed before it is a timeout.
        tick_interval: The scheduler budget `tick_wall_s` is judged against.
        model: Recorded in `model_runs.model`.
        prompt_words: Approximate size of the synthesized prompt. `None`, the
            default, derives it from the server's own reported context budget
            — see `resolve_prompt_words` for why a fixed size cannot be right.
        prompt_fraction: Share of one slot's context a derived prompt may fill.
            Ignored when `prompt_words` is given.
        growth_db_path: Store to measure growth against, defaulting to
            `db_path`. A throwaway benchmark store written entirely today has
            no growth curve to read, so an operator sizing a disk points this
            at the real one.
        slot_save_path: The server's `--slot-save-path`, if the caller knows
            it. Unknowable over HTTP; see `measure_slot_files`.
        on_status: INV-1 progress channel, called as sessions are dispatched
            and as each settles, so a multi-minute round is never silent.

    Returns:
        A `BenchReport`. A run against an unreachable server returns a report
        with `ok` False and `unreachable` True rather than raising — the
        caller decides what a failed measurement is worth, and the per-session
        errors are more informative than one exception.

    Raises:
        ValueError: If `sessions` is not positive.
        BenchError: If the store could not be prepared.
    """
    if sessions <= 0:
        raise ValueError(f"sessions must be positive, got {sessions}")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if on_status is not None:
        on_status(f"preparing benchmark store at {db_path}")
    try:
        migrate(db_path)
        setup = connect(db_path)
        try:
            created = synthesize_sessions(setup, sessions)
            setup.commit()
        finally:
            setup.close()
    except sqlite3.Error as exc:
        raise BenchError(f"could not prepare the benchmark store at {db_path}: {exc}") from exc

    if prompt_words is None:
        prompt_words = _derive_prompt_words(
            host=host, port=port, timeout=timeout, fraction=prompt_fraction, on_status=on_status
        )

    gauge = _InFlightGauge()
    rss_before = peak_rss_bytes()
    if on_status is not None:
        on_status(
            f"dispatching {sessions} concurrent request(s) of ~{prompt_words} words "
            f"to {host}:{port}"
        )

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=sessions) as pool:
        futures = [
            pool.submit(
                _drive_one,
                db_path=db_path,
                session_id=session_id,
                label=label,
                host=host,
                port=port,
                timeout=timeout,
                model=model,
                prompt=synthetic_prompt(prompt_words, label=label),
                gauge=gauge,
                on_status=on_status,
            )
            for session_id, label in created
        ]
        timings = tuple(future.result() for future in futures)
    tick_wall_s = time.monotonic() - started

    rss_after = peak_rss_bytes()
    tables, total_bytes, projection, projection_detail = _measure_growth(
        db_path if growth_db_path is None else Path(growth_db_path), on_status=on_status
    )
    unreachable = bool(timings) and all(timing.error_kind == "connection" for timing in timings)
    if on_status is not None:
        on_status(f"round finished in {tick_wall_s:.3f} s, peak in flight {gauge.peak}")

    return BenchReport(
        sessions=sessions,
        tick_interval_s=tick_interval,
        tick_wall_s=tick_wall_s,
        peak_in_flight=gauge.peak,
        timings=timings,
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_after,
        slot_files=measure_slot_files(slot_save_path),
        unreachable=unreachable,
        tables=tables,
        store_total_bytes=total_bytes,
        projection=projection,
        projection_detail=projection_detail,
    )
