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

**Growth is read out of the store, not logged into it.** Every table task 4.5
measures is append-only under INV-4, so a row's `created_at` is also the moment
the store grew by that row's bytes — which makes the whole growth curve
recoverable from the store itself, with no separate measurement log to seed,
keep, or lose. `project_growth` extrapolates from the measured byte delta
between the oldest and newest sample and refuses to extrapolate from one
sample at all, because a single reading is a reading and not a trend.

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

import resource  # noqa: F401
import sqlite3  # noqa: F401
import sys  # noqa: F401
import threading  # noqa: F401
import time  # noqa: F401
from collections.abc import Callable, Iterator, Sequence  # noqa: F401
from concurrent.futures import ThreadPoolExecutor  # noqa: F401
from contextlib import contextmanager  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from datetime import date  # noqa: F401
from pathlib import Path  # noqa: F401

from palaver.extract.client import (
    ModelClient,  # noqa: F401
    ModelClientError,  # noqa: F401
    ModelConnectionError,  # noqa: F401
    ModelResponseError,  # noqa: F401
    ModelTimeoutError,  # noqa: F401
)
from palaver.extract.slots import SlotClient  # noqa: F401
from palaver.observer.daemon import DEFAULT_INTERVAL, DEFAULT_MODEL, extraction_schema  # noqa: F401
from palaver.store.migrate import connect, migrate  # noqa: F401

from .driver import _WRITE_LOCK_TIMEOUT as _WRITE_LOCK_TIMEOUT
from .driver import BENCH_PURPOSE as BENCH_PURPOSE
from .driver import DEFAULT_SESSIONS as DEFAULT_SESSIONS
from .driver import DEFAULT_TICK_INTERVAL as DEFAULT_TICK_INTERVAL
from .driver import _drive_one as _drive_one
from .driver import _error_kind as _error_kind
from .driver import _InFlightGauge as _InFlightGauge
from .driver import _worker_connection as _worker_connection
from .driver import run_bench as run_bench
from .driver import synthesize_sessions as synthesize_sessions
from .growth import _PAGE_BYTES_UNAVAILABLE as _PAGE_BYTES_UNAVAILABLE
from .growth import GROWTH_TABLES as GROWTH_TABLES
from .growth import PROJECTION_HORIZONS_DAYS as PROJECTION_HORIZONS_DAYS
from .growth import _measure_growth as _measure_growth
from .growth import _page_bytes_by_table as _page_bytes_by_table
from .growth import _parse_day as _parse_day
from .growth import growth_samples as growth_samples
from .growth import measure_tables as measure_tables
from .growth import project_growth as project_growth
from .growth import store_bytes as store_bytes
from .prompt_sizing import _PROMPT_LINE as _PROMPT_LINE
from .prompt_sizing import DEFAULT_PROMPT_FRACTION as DEFAULT_PROMPT_FRACTION
from .prompt_sizing import FALLBACK_PROMPT_WORDS as FALLBACK_PROMPT_WORDS
from .prompt_sizing import TOKENS_PER_WORD as TOKENS_PER_WORD
from .prompt_sizing import _derive_prompt_words as _derive_prompt_words
from .prompt_sizing import resolve_prompt_words as resolve_prompt_words
from .prompt_sizing import synthetic_prompt as synthetic_prompt
from .records import BenchError as BenchError
from .records import BenchReport as BenchReport
from .records import GrowthProjection as GrowthProjection
from .records import GrowthSample as GrowthSample
from .records import SessionTiming as SessionTiming
from .records import SlotFileUsage as SlotFileUsage
from .records import TableUsage as TableUsage
from .resource_usage import _RSS_IN_BYTES as _RSS_IN_BYTES
from .resource_usage import SLOT_PATH_UNKNOWN_NOTE as SLOT_PATH_UNKNOWN_NOTE
from .resource_usage import measure_slot_files as measure_slot_files
from .resource_usage import normalize_rss as normalize_rss
from .resource_usage import peak_rss_bytes as peak_rss_bytes

__all__ = [
    "BENCH_PURPOSE",
    "DEFAULT_PROMPT_FRACTION",
    "DEFAULT_SESSIONS",
    "DEFAULT_TICK_INTERVAL",
    "FALLBACK_PROMPT_WORDS",
    "GROWTH_TABLES",
    "PROJECTION_HORIZONS_DAYS",
    "SLOT_PATH_UNKNOWN_NOTE",
    "TOKENS_PER_WORD",
    "_PAGE_BYTES_UNAVAILABLE",
    "_PROMPT_LINE",
    "_RSS_IN_BYTES",
    "_WRITE_LOCK_TIMEOUT",
    "BenchError",
    "BenchReport",
    "GrowthProjection",
    "GrowthSample",
    "SessionTiming",
    "SlotFileUsage",
    "TableUsage",
    "_InFlightGauge",
    "_derive_prompt_words",
    "_drive_one",
    "_error_kind",
    "_measure_growth",
    "_page_bytes_by_table",
    "_parse_day",
    "_worker_connection",
    "growth_samples",
    "measure_slot_files",
    "measure_tables",
    "normalize_rss",
    "peak_rss_bytes",
    "project_growth",
    "resolve_prompt_words",
    "run_bench",
    "store_bytes",
    "synthesize_sessions",
    "synthetic_prompt",
]
