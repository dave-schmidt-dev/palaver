"""Measuring what six observed sessions cost at once, not what one costs six times.

Phase 4 asks one question about inference: does the observer hold up when
several sessions tick at the same time? A serial loop answers a different
question and reports the same totals — six requests, some total wall time, an
average latency — while never once putting two requests on the server
simultaneously. Every contention effect this phase exists to find (slot
queueing on a server with fewer slots than sessions, memory growth under
parallel decode, tick overrun) is invisible to it. So the harness below
dispatches every session on its own thread and records the peak number of
requests it had in flight at one instant; a serial implementation reports a
peak of 1 and fails the test that asserts on it.

**Threads, and one sqlite connection each.** `ModelClient` writes a
`model_runs` row from the calling thread, and `store.migrate.connect` does not
pass `check_same_thread=False`, so sharing one connection across six worker
threads raises `sqlite3.ProgrammingError`. Each worker therefore opens its own
connection to the same file. WAL journaling is already on, which is what makes
six concurrent writers workable; `_WRITE_LOCK_TIMEOUT` gives each one room to
wait out the others rather than failing on `database is locked`.

**What "peak RSS" means here.** `resource.getrusage` reports a high-water mark
for the whole process that is never reset, so a single reading taken at the end
of a run includes every allocation the interpreter made before the benchmark
started. This module samples it twice and reports both endpoints plus the
delta, and the delta is the number attributable to the run. The unit differs by
platform — bytes on macOS, kilobytes on Linux — and `peak_rss_bytes` normalizes
that explicitly rather than letting a report be silently wrong by a factor of
1024, which in a benchmark reads as a passing measurement rather than as a bug.

**Slot-file disk usage needs a path nobody will tell us.** Task 4.2 established
that `/props` carries no top-level `argv`, `cmdline`, or `params` key, so a
running server will not reveal its `--slot-save-path`. Re-checked live on
2026-08-15: the one nested `params` object,
`default_generation_settings.params`, holds sampling settings — `temperature`,
`top_k`, `seed`, `samplers` — and nothing about the invocation. Rather
than guess a location or shell out to `ps`, `measure_slot_files` reports the
measurement as unavailable, with the reason, unless a caller supplies the
path. An unavailable measurement that says so is useful; a zero that means
"we did not look" is not.

**Prompt size is asked for, never assumed.** A fixed prompt length is wrong
on any server but the one it was tuned against: the first live run of this
benchmark sent ~19,700 tokens to a four-slot server reporting `n_ctx: 32768`
and every request came back `500 Context size has been exceeded`, because that
figure is the whole server's context divided among its slots. `run_bench`
therefore reads `/props` through task 4.2's `SlotClient` and sizes the prompt
to a fraction of one slot. See `resolve_prompt_words`.

**Failure is loud.** A benchmark that cannot reach the model server must not
report zeros. Every per-session failure is carried on the report, `ok` is False
when any session failed, and `unreachable` is True when *every* session failed
to connect — which is what `palaver bench` turns into a non-zero exit naming
the server it could not reach.

The six sessions are synthesized fixtures written into a throwaway store, never
six real observed sessions. Nothing in this module reads a real transcript, and
the prompt it sends is generated from an invented line (INV-9).

This repository is public. Nothing here is derived from a real observed session.
"""

from __future__ import annotations

import resource
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from palaver.extract.client import (
    ModelClient,
    ModelClientError,
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from palaver.extract.slots import SlotClient
from palaver.observer.daemon import DEFAULT_INTERVAL, DEFAULT_MODEL, extraction_schema
from palaver.store.migrate import connect, migrate

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

#: Tokens per word for the synthesized prompt, measured against the observed
#: server on 2026-08-15: 1,000 words produced 1,572 prompt tokens and 4,000
#: produced 6,567 (`usage.prompt_tokens`, both requests). The higher of the two
#: ratios is used so the estimate errs toward a shorter prompt.
TOKENS_PER_WORD = 1.65

#: Share of one slot's context the synthesized prompt may fill, leaving the
#: rest for the response. Not a measurement — a margin.
DEFAULT_PROMPT_FRACTION = 0.6

#: Used only when the server's context budget could not be read, which in
#: practice means the server is unreachable and every request is about to fail
#: anyway. Small enough to fit any plausible slot.
FALLBACK_PROMPT_WORDS = 1000

#: The one invented line the synthesized prompt is built from. No observed
#: session content ever reaches the model from this module (INV-9).
_PROMPT_LINE = "fixture: invented benchmark transcript line carrying no observed content"

#: `ru_maxrss` is in bytes on macOS and kilobytes on Linux. Checked once here
#: rather than at each call site so the two cannot disagree.
_RSS_IN_BYTES = sys.platform == "darwin"

#: Printed when slot-file disk usage was not measured because no path was
#: given. A module constant because `tests/test_bench.py` asserts the report
#: carries it — a benchmark reporting 0 bytes when it never looked is the
#: quiet-zero failure this task's last criterion exists to prevent.
SLOT_PATH_UNKNOWN_NOTE = (
    "not measured: llama-server exposes no --slot-save-path over HTTP "
    "(/props carries no argv or command line), so pass --slot-save-path "
    "to measure slot-file disk usage"
)


class BenchError(Exception):
    """Raised when a benchmark run could not be set up or performed at all."""


@dataclass(frozen=True)
class SessionTiming:
    """One synthesized session's result from a benchmark round.

    Attributes:
        label: Human-readable session name, e.g. `bench-session-1`.
        session_id: `sessions.id` this round drove.
        latency_ms: Wall time of the request, including a failed one — a
            request that took 30 s to time out cost 30 s.
        prompt_tokens: What the server reported for this request, when it
            reported anything.
        error: The failure message, empty on success.
        error_kind: `""`, `"connection"`, `"timeout"`, or `"response"`. The
            distinction matters: six connection failures mean the server is
            not there, while six response failures mean it is there and
            answering wrongly.
    """

    label: str
    session_id: int
    latency_ms: int
    prompt_tokens: int | None = None
    error: str = ""
    error_kind: str = ""

    @property
    def ok(self) -> bool:
        """Whether this session's request returned a usable object."""
        return not self.error


@dataclass(frozen=True)
class SlotFileUsage:
    """Slot-file disk usage, or an explicit statement that it was not measured.

    Attributes:
        path: The directory measured, empty when none was supplied.
        available: Whether a measurement was actually taken. False means
            `file_count` and `total_bytes` are both 0 because nothing was
            looked at, not because the directory was empty.
        file_count: Files found under `path`.
        total_bytes: Their summed `st_size`.
        detail: Why a measurement is unavailable, empty when it is available.
    """

    path: str
    available: bool
    file_count: int
    total_bytes: int
    detail: str


@dataclass(frozen=True)
class BenchReport:
    """Everything one concurrent benchmark round measured.

    Attributes:
        sessions: How many sessions were driven.
        tick_interval_s: The budget `tick_wall_s` is judged against.
        tick_wall_s: Wall time from dispatching the first request to the last
            one settling — the number a scheduler has to fit inside its tick.
        peak_in_flight: The most requests this harness had outstanding at one
            instant. A serial implementation reports 1.
        timings: One entry per session, in dispatch order.
        rss_before_bytes: Process peak RSS before the round, normalized to
            bytes on every platform.
        rss_after_bytes: The same high-water mark after the round.
        slot_files: Slot-file disk usage, or why it was not measured.
        unreachable: True when *every* session failed to connect. Distinguished
            from `ok` because "the server is not running" and "the server
            answered badly" call for different responses from the operator.
    """

    sessions: int
    tick_interval_s: float
    tick_wall_s: float
    peak_in_flight: int
    timings: tuple[SessionTiming, ...]
    rss_before_bytes: int
    rss_after_bytes: int
    slot_files: SlotFileUsage
    unreachable: bool

    @property
    def ok(self) -> bool:
        """Whether every session's request succeeded."""
        return all(timing.ok for timing in self.timings)

    @property
    def failures(self) -> tuple[SessionTiming, ...]:
        """Every session that did not return a usable object."""
        return tuple(timing for timing in self.timings if not timing.ok)

    @property
    def rss_delta_bytes(self) -> int:
        """Growth in the process's peak RSS across the round.

        The attributable number: `rss_after_bytes` on its own includes every
        allocation made before the benchmark started, because `ru_maxrss` is a
        high-water mark that is never reset.
        """
        return self.rss_after_bytes - self.rss_before_bytes

    @property
    def fits_tick_interval(self) -> bool:
        """Whether the concurrent round finished inside its tick budget."""
        return self.tick_wall_s <= self.tick_interval_s

    @property
    def successful_latencies_ms(self) -> tuple[int, ...]:
        """Per-session latency for the sessions that succeeded, in dispatch order."""
        return tuple(timing.latency_ms for timing in self.timings if timing.ok)


def normalize_rss(raw: int) -> int:
    """Convert a raw `ru_maxrss` reading to bytes.

    Split out from `peak_rss_bytes` so the platform mapping is testable on
    both branches without moving the clock or the machine: a test that only
    compares a live reading against itself agrees with whichever unit the
    module happens to have chosen.
    """
    return int(raw) if _RSS_IN_BYTES else int(raw) * 1024


def peak_rss_bytes() -> int:
    """Return this process's peak RSS in bytes, normalized across platforms.

    `resource.getrusage(RUSAGE_SELF).ru_maxrss` is bytes on macOS and
    kilobytes on Linux. Returning the raw value would make a benchmark report
    wrong by 1024x on one of them, in a direction no reader could detect.
    """
    return normalize_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def resolve_prompt_words(
    n_ctx: int | None, total_slots: int, *, fraction: float = DEFAULT_PROMPT_FRACTION
) -> int:
    """Size the synthesized prompt to fit one of the server's KV slots.

    **`n_ctx` is the whole server's context, shared across its slots, not one
    slot's.** Measured on 2026-08-15: a server reporting `total_slots: 4` and
    `default_generation_settings.n_ctx: 32768` rejected a ~19,700-token prompt
    with `{"error":{"code":500,"message":"Context size has been
    exceeded."}}` — which cannot happen against a 32,768-token slot, and is
    exactly what a 8,192-token slot does. A benchmark whose every request 500s
    measures nothing, so the size is derived here rather than guessed.

    Args:
        n_ctx: The server's reported context length, or `None` when it did not
            report one.
        total_slots: The server's slot count; values below 1 are treated as 1.
        fraction: Share of the per-slot budget the prompt may occupy.

    Returns:
        A positive word count. `FALLBACK_PROMPT_WORDS` when `n_ctx` is unknown
        or not positive.
    """
    if not n_ctx or n_ctx <= 0:
        return FALLBACK_PROMPT_WORDS
    per_slot_tokens = n_ctx // max(1, total_slots)
    prompt_tokens = per_slot_tokens * fraction
    return max(1, int(prompt_tokens / TOKENS_PER_WORD))


def synthetic_prompt(words: int, *, label: str = "") -> str:
    """Build an invented prompt of roughly `words` words.

    Deterministic, so two runs of the benchmark send the same number of tokens
    and their latencies are comparable. `label` is prefixed so each concurrent
    request differs by at least one token, which keeps a server-side prompt
    cache from making five of the six requests trivially fast and the
    measurement meaningless.

    Args:
        words: Approximate word count. Must be positive.
        label: Optional per-session prefix.

    Returns:
        The prompt text.

    Raises:
        ValueError: If `words` is not positive — a zero-word prompt would
            measure the server's overhead, not its inference.
    """
    if words <= 0:
        raise ValueError(f"words must be positive, got {words}")
    line_words = len(_PROMPT_LINE.split())
    # `+ 1` for the index appended to each line. Counting only `_PROMPT_LINE`'s
    # own words overshot the requested size by about 10%, which eats into the
    # margin `resolve_prompt_words` leaves for the response.
    repeats = max(1, words // (line_words + 1))
    body = "\n".join(f"{_PROMPT_LINE} {index}" for index in range(repeats))
    return f"{label}\n{body}" if label else body


def measure_slot_files(path: Path | str | None) -> SlotFileUsage:
    """Sum the on-disk size of a server's KV slot files.

    Args:
        path: The server's `--slot-save-path`, or `None` when it is unknown.
            It is unknown by default: task 4.2 established that `/props`
            carries no invocation, so nothing here can discover it.

    Returns:
        A `SlotFileUsage`. `available` is False both when no path was given
        and when the given path is not a directory, with `detail` saying
        which — never a bare 0.
    """
    if path is None:
        return SlotFileUsage("", False, 0, 0, SLOT_PATH_UNKNOWN_NOTE)
    directory = Path(path)
    if not directory.is_dir():
        return SlotFileUsage(
            str(directory), False, 0, 0, f"not measured: no directory at {directory}"
        )
    files = [entry for entry in directory.rglob("*") if entry.is_file()]
    return SlotFileUsage(
        str(directory), True, len(files), sum(entry.stat().st_size for entry in files), ""
    )


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


def _derive_prompt_words(
    *,
    host: str,
    port: int,
    timeout: float,
    fraction: float,
    on_status: Callable[[str], None] | None,
) -> int:
    """Ask the server for its context budget and size the prompt to one slot.

    Reuses task 4.2's `SlotClient` rather than opening its own socket, so the
    benchmark and `palaver doctor` read the same `/props` through the same
    parser. A failure here is not fatal: an unreachable server means every
    request is about to fail with a connection error, and that report is more
    useful than an exception raised before the round even started.
    """
    try:
        properties = SlotClient(host=host, port=port, timeout=timeout).properties(
            on_status=on_status
        )
    except ModelClientError as exc:
        if on_status is not None:
            on_status(f"could not read the server's context budget ({exc}); using a small prompt")
        return FALLBACK_PROMPT_WORDS
    words = resolve_prompt_words(properties.n_ctx, properties.total_slots, fraction=fraction)
    if on_status is not None:
        on_status(
            f"server reports n_ctx {properties.n_ctx} across {properties.total_slots} slot(s); "
            f"synthesizing a ~{words}-word prompt"
        )
    return words


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
    )
