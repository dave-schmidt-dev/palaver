"""
The result shapes a benchmark round produces, and the error raised when a round could
not be set up.
"""

from __future__ import annotations

from dataclasses import dataclass


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
class TableUsage:
    """One append-only table's current size, by rows and by bytes.

    Attributes:
        table: Table name.
        rows: Row count.
        payload_bytes: Summed byte length of the table's one text column —
            what the store was asked to keep. Always 0 for `memory_evidence`,
            which stores offsets into its source rather than a copy of it.
        page_bytes: What SQLite actually spends on that table's pages, from
            `dbstat`. `-1` when `dbstat` is not compiled into this build,
            which is reported rather than silently replaced by a zero.
    """

    table: str
    rows: int
    payload_bytes: int
    page_bytes: int


@dataclass(frozen=True)
class GrowthSample:
    """Cumulative store size as of the end of one calendar day.

    Attributes:
        day: `YYYY-MM-DD`, from the rows' own `created_at` values.
        cumulative_rows: Rows written on or before `day`, across
            `GROWTH_TABLES`.
        cumulative_payload_bytes: Their summed text length.
    """

    day: str
    cumulative_rows: int
    cumulative_payload_bytes: int


@dataclass(frozen=True)
class GrowthProjection:
    """A linear extrapolation of measured byte growth.

    Attributes:
        bytes_per_day: Measured slope, in on-disk bytes.
        span_days: Days between the first and last sample.
        samples_used: How many samples the slope was measured from.
        current_bytes: On-disk total at the last sample.
        horizons: `(days, projected_total_bytes)`, ascending.
    """

    bytes_per_day: float
    span_days: float
    samples_used: int
    current_bytes: int
    horizons: tuple[tuple[int, int], ...]

    def at(self, days: int) -> int:
        """Return the projected total for a horizon this projection carries.

        Raises:
            KeyError: If `days` is not one of the projected horizons — better
                than interpolating a horizon nobody asked for.
        """
        for horizon, projected in self.horizons:
            if horizon == days:
                return projected
        raise KeyError(f"no projection for {days} days; have {[h for h, _ in self.horizons]}")


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
        tables: Per-table size of the append-only store, measured after the
            round so the round's own rows are counted (task 4.5).
        store_total_bytes: The whole store's on-disk size.
        projection: Linear growth extrapolation, or `None` when the store does
            not yet hold two days of measurement points.
        projection_detail: Why `projection` is `None`, empty when it is not.
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
    tables: tuple[TableUsage, ...] = ()
    store_total_bytes: int = 0
    projection: GrowthProjection | None = None
    projection_detail: str = ""

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
