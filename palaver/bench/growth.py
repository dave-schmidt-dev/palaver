"""
Measuring the append-only store's current size and projecting its growth from measured
byte deltas.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from palaver.store.migrate import connect

from .records import GrowthProjection, GrowthSample, TableUsage

#: The append-only tables growth is measured over, paired with the one text
#: column each stores its payload in. INV-4 guarantees these only ever grow,
#: which is what makes a row's `created_at` also the moment the store grew.
#:
#: `memory_evidence` is paired with `None`, not a column name: migration 4
#: replaced its `quote` column with `(start_offset, end_offset)`, because task
#: 2.2 resolves evidence live against the source rather than storing a second
#: copy of it. So evidence rows add rows and index pages but no text at all,
#: which is precisely why `project_growth` scales its slope by the store's
#: measured page-to-text ratio instead of projecting text alone.
GROWTH_TABLES: tuple[tuple[str, str | None], ...] = (
    ("transcript_chunks", "content"),
    ("events", "payload"),
    ("memories", "statement"),
    ("memory_evidence", None),
)

#: Horizons `palaver bench --report` projects, in days. The plan names both.
PROJECTION_HORIZONS_DAYS: tuple[int, ...] = (30, 90)

#: `TableUsage.page_bytes` when this SQLite build has no `dbstat`. Negative so
#: it can never be mistaken for a measured zero.
_PAGE_BYTES_UNAVAILABLE = -1


def measure_tables(conn: sqlite3.Connection) -> tuple[TableUsage, ...]:
    """Measure the current size of every append-only table.

    Args:
        conn: Open connection to a migrated store.

    Returns:
        One `TableUsage` per entry in `GROWTH_TABLES`, in that order.
    """
    page_bytes = _page_bytes_by_table(conn)
    usage = []
    for table, column in GROWTH_TABLES:
        measure = "0" if column is None else f"COALESCE(SUM(LENGTH(CAST({column} AS BLOB))), 0)"
        rows, payload = conn.execute(f"SELECT COUNT(*), {measure} FROM {table}").fetchone()
        usage.append(
            TableUsage(
                table=table,
                rows=int(rows),
                payload_bytes=int(payload),
                page_bytes=page_bytes.get(table, _PAGE_BYTES_UNAVAILABLE),
            )
        )
    return tuple(usage)


def _page_bytes_by_table(conn: sqlite3.Connection) -> dict[str, int]:
    """Read per-table page usage from `dbstat`, or return nothing if it is absent.

    `dbstat` is a compile-time option (`SQLITE_ENABLE_DBSTAT_VTAB`). A build
    without it raises `OperationalError` on the first query, which is reported
    as an unavailable measurement rather than as a zero — the same rule
    `measure_slot_files` follows.
    """
    try:
        rows = conn.execute("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {name: int(size) for name, size in rows}


def store_bytes(conn: sqlite3.Connection) -> int:
    """Return the whole store's on-disk size, from SQLite's own page accounting."""
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    return int(page_count) * int(page_size)


def growth_samples(conn: sqlite3.Connection) -> tuple[GrowthSample, ...]:
    """Reconstruct the store's growth curve from the rows' own timestamps.

    Every table in `GROWTH_TABLES` is append-only under INV-4, so a row's
    `created_at` is also the moment the store grew by that row's bytes. That
    makes the history recoverable from the store itself, with no separate
    measurement log to keep, seed, or lose.

    Args:
        conn: Open connection to a migrated store.

    Returns:
        One cumulative sample per calendar day that has at least one row,
        oldest first. A store written entirely within one day yields one
        sample, which `project_growth` refuses to extrapolate from.
    """
    per_day: dict[str, list[int]] = {}
    for table, column in GROWTH_TABLES:
        measure = "0" if column is None else f"COALESCE(SUM(LENGTH(CAST({column} AS BLOB))), 0)"
        rows = conn.execute(
            f"SELECT substr(created_at, 1, 10) AS day, COUNT(*), {measure} "
            f"FROM {table} GROUP BY day"
        ).fetchall()
        for day, count, payload in rows:
            bucket = per_day.setdefault(day, [0, 0])
            bucket[0] += int(count)
            bucket[1] += int(payload)

    samples = []
    running_rows = 0
    running_bytes = 0
    for day in sorted(per_day):
        rows_today, bytes_today = per_day[day]
        running_rows += rows_today
        running_bytes += bytes_today
        samples.append(
            GrowthSample(
                day=day, cumulative_rows=running_rows, cumulative_payload_bytes=running_bytes
            )
        )
    return tuple(samples)


def project_growth(
    samples: Sequence[GrowthSample],
    *,
    current_bytes: int,
    horizons: Sequence[int] = PROJECTION_HORIZONS_DAYS,
) -> GrowthProjection:
    """Extrapolate on-disk growth linearly from measured byte deltas.

    The slope comes from the *bytes* between the first and last sample, never
    from a row count multiplied by an assumed per-record size: rows in these
    tables vary from a one-line memory statement to a multi-kilobyte transcript
    chunk, so a per-record constant would be an estimate dressed as a
    measurement. Row counts are carried on the samples for reporting and are
    not read here.

    Payload bytes are scaled to on-disk bytes by the ratio SQLite is currently
    exhibiting on this very store (`current_bytes` over the last sample's
    payload total), which is itself measured — pages, indexes, and FTS shadow
    tables included — rather than assumed.

    Args:
        samples: Cumulative samples, oldest first, from `growth_samples`.
        current_bytes: The store's present on-disk size, from `store_bytes`.
        horizons: Horizons to project, in days.

    Returns:
        A `GrowthProjection` whose `horizons` are ascending.

    Raises:
        ValueError: If fewer than two samples were given, or if the two ends
            of the span fall on the same day. One measurement point is a
            reading, not a trend, and extrapolating from it would produce a
            confident number backed by nothing.
    """
    if len(samples) < 2:
        raise ValueError(
            f"a projection needs at least two measurement points, got {len(samples)}; "
            "a single reading is not a trend"
        )
    first, last = samples[0], samples[-1]
    span_days = (_parse_day(last.day) - _parse_day(first.day)).days
    if span_days <= 0:
        raise ValueError(
            f"a projection needs measurement points spanning at least one day, "
            f"got {first.day} to {last.day}"
        )

    payload_delta = last.cumulative_payload_bytes - first.cumulative_payload_bytes
    overhead = (
        current_bytes / last.cumulative_payload_bytes if last.cumulative_payload_bytes > 0 else 1.0
    )
    bytes_per_day = (payload_delta / span_days) * overhead
    return GrowthProjection(
        bytes_per_day=bytes_per_day,
        span_days=float(span_days),
        samples_used=len(samples),
        current_bytes=current_bytes,
        horizons=tuple(
            (days, int(current_bytes + bytes_per_day * days)) for days in sorted(horizons)
        ),
    )


def _parse_day(day: str) -> date:
    """Parse a `YYYY-MM-DD` bucket key."""
    return date.fromisoformat(day)


def _measure_growth(
    db_path: Path, *, on_status: Callable[[str], None] | None
) -> tuple[tuple[TableUsage, ...], int, GrowthProjection | None, str]:
    """Measure the store and project its growth, reporting why if it cannot.

    A store that cannot be projected from is not an error: it is a store
    younger than one day, which is the normal state of a fresh benchmark
    store. The reason is carried out so the report can print it in place of a
    number instead of printing a number that means nothing.
    """
    conn = connect(db_path)
    try:
        tables = measure_tables(conn)
        total_bytes = store_bytes(conn)
        samples = growth_samples(conn)
    finally:
        conn.close()
    try:
        projection = project_growth(samples, current_bytes=total_bytes)
    except ValueError as exc:
        if on_status is not None:
            on_status(f"growth projection unavailable: {exc}")
        return tables, total_bytes, None, str(exc)
    if on_status is not None:
        on_status(
            f"growth measured over {projection.span_days:.0f} day(s): "
            f"{projection.bytes_per_day:.0f} bytes/day"
        )
    return tables, total_bytes, projection, ""
