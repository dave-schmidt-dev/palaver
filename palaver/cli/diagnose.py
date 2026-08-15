"""`palaver diagnose --coverage`: per-signal coverage over a transcript sample.

Coverage is the fraction of sampled sessions for which a signal was
*determinable* — anything other than `Tri.UNKNOWN`. It ships as a standing
command rather than a number captured once during development, because the
regression it guards against is a future Claude Code release changing JSONL
shape, and a one-time measurement cannot catch that. Run it against a fresh
sample whenever the observed harness changes.

**Coverage is not accuracy.** This command counts the sessions a signal could
be computed for, never the sessions it was right about. A uniformly wrong
classifier scores 100% here. Accuracy is the independently-labelled fixture
corpus's job (task 1.7) and is asserted against ground truth, not against
these numbers. The report says so in its own footer so the distinction
survives being pasted into an issue without its context.

**Task 7.3: three sources, and the gate the numbers feed.** The report now
covers Claude Code, Codex and OpenCode, and each source's coverage is not
merely printed — it is the input to `derive_status`'s per-source gate. A
source whose coverage for a consulted signal falls below the threshold has
its statuses withdrawn to `UNKNOWN`, and the report shows both figures: the
status distribution after the gate, and what the gate took away. That is the
plan's adapter-interface rollback point made observable, rather than a
threshold buried in a library where nobody sees it fire.

Two things about how the three sources are read here:

* **The derivations differ, and deliberately.** Claude Code goes through
  `observe_session`, which reads record structure directly and is the path
  every coverage number in this project was measured against. Codex and
  OpenCode go through `derive_signals_from_events`, which reads only the
  canonical `Event` kinds both adapters emit. `palaver.observer.turn_boundary`
  documents why one derivation for all three would silence the best-covered
  source.
* **Naming a sample scopes the run to that source.** With no sample flag at
  all, every source is measured at its production default. With one or more
  given, only the named sources are measured — so a test that points
  `--sample` at a `tmp_path` never reads the real Codex rollouts or the real
  OpenCode database, which is INV-2 and INV-3's requirement, not a
  convenience.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from palaver.ingest.adapters import opencode
from palaver.ingest.adapters.base import Cursor, Event, read_complete_records
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.adapters.codex import CodexAdapter
from palaver.observer.signals import (
    DEFAULT_COVERAGE_THRESHOLD,
    SIGNAL_NAMES,
    Signals,
    Status,
    StatusDerivation,
    Tri,
    derive_status_with_provenance,
    under_covered,
)
from palaver.observer.turn_boundary import (
    BASIS_NAMES,
    BASIS_SOURCE_UNREADABLE,
    SessionObservation,
    TurnBoundary,
    derive_signals_from_events,
    observe_session,
)

NAME = "diagnose"
HELP = "measure per-signal coverage over a sample of session transcripts"

#: Every source `--coverage` can measure, in report order.
SOURCES: tuple[str, ...] = (ClaudeCodeAdapter.source, CodexAdapter.source, opencode.SOURCE)

#: OpenCode's production store. Named here rather than in the adapter
#: because the adapter deliberately holds no path — it takes a connection.
DEFAULT_OPENCODE_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


@dataclass(frozen=True)
class CoverageReport:
    """Counts behind one source's `--coverage` run.

    Attributes:
        sample_root: Directory or database the sample was read from.
        source: Adapter source name the sample was read with.
        sessions: How many sessions were observed.
        determinable: Per signal name, how many sessions produced a value
            other than `Tri.UNKNOWN`.
        statuses: How many sessions produced each `Status`, **after** this
            source's own coverage gate. This is the distribution Palaver
            would actually report for this source.
        ungated_statuses: The same distribution before the gate. Kept so the
            report can show what the gate withdrew rather than only its
            result — a gate whose effect is invisible is a gate nobody can
            tell is misconfigured.
        bases: How many sessions each turn-boundary basis accounted for.
        corroboration: How many boundaries an independent signal agreed with
            (`Tri.TRUE`), disagreed with (`Tri.FALSE`), or was unavailable
            for (`Tri.UNKNOWN`).
        threshold: The coverage percentage the gate was applied at.
    """

    sample_root: Path
    source: str
    sessions: int
    determinable: dict[str, int]
    statuses: Counter
    ungated_statuses: Counter
    bases: Counter
    corroboration: Counter
    threshold: float = DEFAULT_COVERAGE_THRESHOLD

    def percentage(self, signal_name: str) -> float:
        """Return `signal_name`'s coverage as a percentage of sampled sessions.

        Args:
            signal_name: A name from `SIGNAL_NAMES`.

        Returns:
            The percentage of sessions the signal was determinable for, or
            `0.0` when the sample is empty (never 100% by vacuity).
        """
        if self.sessions == 0:
            return 0.0
        return 100.0 * self.determinable[signal_name] / self.sessions

    def as_coverage(self) -> dict[str, float]:
        """Return this source's coverage in the mapping `derive_status` takes.

        Returns:
            `{signal_name: percentage}` over every entry in `SIGNAL_NAMES`.
        """
        return {name: self.percentage(name) for name in SIGNAL_NAMES}

    def under_covered_signals(self) -> tuple[str, ...]:
        """Name the signals this source does not cover to `threshold`."""
        return under_covered(SIGNAL_NAMES, self.as_coverage(), threshold=self.threshold)


def add_arguments(parser) -> None:
    """Register `diagnose`'s flags on its subparser."""
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="report per-signal coverage over the sample",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help=(
            "directory of Claude Code session stores to measure, laid out as "
            "<root>/<project>/<session>.jsonl. Naming any sample flag limits "
            "the run to the sources named; naming none measures all three at "
            "their production defaults"
        ),
    )
    parser.add_argument(
        "--codex-sample",
        type=Path,
        default=None,
        help="directory of Codex rollout files to measure (default: ~/.codex/sessions)",
    )
    parser.add_argument(
        "--opencode-db",
        type=Path,
        default=None,
        help="OpenCode SQLite store to measure (default: ~/.local/share/opencode/opencode.db)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="observe at most this many sessions from each source's sample",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "coverage percentage a signal must reach before a status derived "
            f"from it is reported for that source (default: {DEFAULT_COVERAGE_THRESHOLD})"
        ),
    )


def _unreadable() -> SessionObservation:
    """The observation for a session that could not be read at all.

    Matches `observe_session`'s own unreadable branch: a component that could
    not read a session reports nothing about it.
    """
    return SessionObservation(
        signals=Signals(
            source_readable=Tri.FALSE,
            signal_records_parsed=Tri.UNKNOWN,
            unresolved_tool_error=Tri.UNKNOWN,
            agent_turn_ended=Tri.UNKNOWN,
        ),
        boundary=TurnBoundary(
            ended=Tri.UNKNOWN,
            basis=BASIS_SOURCE_UNREADABLE,
            corroboration=Tri.UNKNOWN,
        ),
    )


def _jsonl_parse_completeness(path: Path) -> Tri:
    """Report whether every complete line in a JSONL store decodes to an object.

    The `parsed` input `derive_signals_from_events` cannot infer for itself:
    the Codex adapter drops an undecodable record rather than marking it, so
    a hole is invisible by the time the events exist.

    Coarser than `derive_signals`'s window-scoped rule, and knowingly so.
    This is file-wide, because `order_records` sorts a rollout by its own
    sequence numbers and a dropped line has no position left to compare
    against a window. `TRUE` therefore means strictly more than it needs to
    ("every record decoded", not just the ones in the current turn) and is
    exactly true when it is returned; `FALSE` is conservative — one corrupt
    line anywhere pins the session to `UNKNOWN` — and is the fail-closed
    direction.

    Args:
        path: The store to read.

    Returns:
        `TRUE`, `FALSE`, or `UNKNOWN` when the store could not be read.
    """
    try:
        raw_records, _ = read_complete_records(path, 0)
    except OSError:
        return Tri.UNKNOWN
    for raw in raw_records:
        try:
            record = json.loads(raw)
        except json.JSONDecodeError, UnicodeDecodeError:
            return Tri.FALSE
        if not isinstance(record, dict):
            return Tri.FALSE
    return Tri.TRUE


def _tally(
    observations: Sequence[SessionObservation],
    *,
    sample_root: Path,
    source: str,
    threshold: float,
) -> CoverageReport:
    """Turn one source's per-session observations into its report.

    The gate is applied here rather than during the sweep, because it cannot
    be applied earlier: a session's status depends on the coverage of the
    whole sample, which is not known until the last session has been read.
    So each session's status is derived once, with the signals it consulted
    recorded, and the threshold is applied to those records afterwards.

    Args:
        observations: One per session, in sweep order.
        sample_root: Where the sample was read from, for the report header.
        source: The adapter source name.
        threshold: Coverage percentage the gate is applied at.

    Returns:
        The finished `CoverageReport`.
    """
    determinable = dict.fromkeys(SIGNAL_NAMES, 0)
    bases: Counter = Counter()
    corroboration: Counter = Counter()
    derivations: list[StatusDerivation] = []

    for observation in observations:
        for name in SIGNAL_NAMES:
            if getattr(observation.signals, name) is not Tri.UNKNOWN:
                determinable[name] += 1
        derivations.append(derive_status_with_provenance(observation.signals))
        bases[observation.boundary.basis] += 1
        corroboration[observation.boundary.corroboration] += 1

    report = CoverageReport(
        sample_root=sample_root,
        source=source,
        sessions=len(observations),
        determinable=determinable,
        statuses=Counter(),
        ungated_statuses=Counter(derivation.status for derivation in derivations),
        bases=bases,
        corroboration=corroboration,
        threshold=threshold,
    )
    coverage = report.as_coverage()
    gated = Counter(
        Status.UNKNOWN
        if under_covered(derivation.consulted, coverage, threshold=threshold)
        else derivation.status
        for derivation in derivations
    )
    return CoverageReport(
        sample_root=report.sample_root,
        source=report.source,
        sessions=report.sessions,
        determinable=report.determinable,
        statuses=gated,
        ungated_statuses=report.ungated_statuses,
        bases=report.bases,
        corroboration=report.corroboration,
        threshold=threshold,
    )


def _announce(on_status: Callable[[str], None] | None, index: int, total: int, key: str) -> None:
    """Emit one per-session progress line (INV-1). Never writes to stdout."""
    if on_status is not None:
        on_status(f"observing {index}/{total}: {key}")


def collect_coverage(
    sample_root: Path | None,
    *,
    limit: int | None = None,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageReport:
    """Observe every Claude Code session in a sample and count what was determinable.

    Args:
        sample_root: Sample directory, or `None` for the adapter's default.
        limit: Maximum number of sessions to observe, or `None` for all.
        now: Reference time passed through to each observation.
        on_status: Progress channel, called once per session before it is
            read (INV-1). Never writes to stdout.
        threshold: Coverage percentage the per-source gate is applied at.

    Returns:
        A `CoverageReport` over the sessions actually observed.
    """
    adapter = ClaudeCodeAdapter(root=sample_root)
    refs = adapter.discover_sessions(all=True)
    if limit is not None:
        refs = refs[:limit]

    observations = []
    for index, ref in enumerate(refs, start=1):
        _announce(on_status, index, len(refs), ref.session_key)
        observations.append(observe_session(ref.path, now=now))

    return _tally(
        observations,
        sample_root=Path(adapter.root),
        source=adapter.source,
        threshold=threshold,
    )


def collect_codex_coverage(
    sample_root: Path | None,
    *,
    limit: int | None = None,
    on_status: Callable[[str], None] | None = None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageReport:
    """Measure the Codex adapter over a directory of rollout files.

    Args:
        sample_root: Codex sessions root, or `None` for the adapter's default.
        limit: Maximum number of sessions to observe, or `None` for all.
        on_status: Progress channel (INV-1).
        threshold: Coverage percentage the per-source gate is applied at.

    Returns:
        A `CoverageReport` over the rollouts actually read.
    """
    adapter = CodexAdapter(root=sample_root)
    refs = adapter.discover_sessions(all=True)
    if limit is not None:
        refs = refs[:limit]

    observations = []
    for index, ref in enumerate(refs, start=1):
        _announce(on_status, index, len(refs), ref.session_key)
        parsed = _jsonl_parse_completeness(ref.path)
        try:
            events: Iterable[Event] = adapter.tail(ref.path, Cursor(offset=0)).events
        except OSError:
            observations.append(_unreadable())
            continue
        observations.append(derive_signals_from_events(tuple(events), parsed=parsed))

    return _tally(
        observations,
        sample_root=Path(adapter.root),
        source=adapter.source,
        threshold=threshold,
    )


def _opencode_events(conn: sqlite3.Connection, session_id: str) -> list[Event]:
    """Read one OpenCode session's canonical events, oldest message first."""
    events: list[Event] = []
    for message in opencode.fetch_messages(conn, session_id):
        parts = opencode.fetch_parts(conn, message["id"])
        events.extend(opencode.events_for_message(session_id, message, parts))
    return events


def collect_opencode_coverage(
    db_path: Path | None,
    *,
    limit: int | None = None,
    on_status: Callable[[str], None] | None = None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageReport:
    """Measure the OpenCode adapter over a SQLite store.

    The store is opened through `opencode.open_store_readonly`, so both INV-3
    defenses (read-only URI and the table allowlist) are installed — this
    command reads `session`, `message` and `part` and structurally cannot
    reach `account` or `credential`.

    A store that does not exist is not an error: it reports zero sessions,
    the same as an empty sample. A machine with two of the three harnesses
    installed should still get a report about the two.

    Args:
        db_path: The store, or `None` for `DEFAULT_OPENCODE_DB`.
        limit: Maximum number of sessions to observe, or `None` for all.
        on_status: Progress channel (INV-1).
        threshold: Coverage percentage the per-source gate is applied at.

    Returns:
        A `CoverageReport` over the sessions actually read.
    """
    path = DEFAULT_OPENCODE_DB if db_path is None else Path(db_path)
    if not path.exists():
        return _tally([], sample_root=path, source=opencode.SOURCE, threshold=threshold)

    conn = opencode.open_store_readonly(str(path))
    observations = []
    try:
        session_ids = opencode.fetch_sessions(conn, limit=limit)
        for index, session_id in enumerate(session_ids, start=1):
            _announce(on_status, index, len(session_ids), session_id)
            try:
                events = _opencode_events(conn, session_id)
            except sqlite3.Error, json.JSONDecodeError:
                observations.append(_unreadable())
                continue
            # Every row that reached here decoded: `fetch_messages` and
            # `fetch_parts` raise on a malformed `data` column rather than
            # dropping it, and that raise is caught above as unreadable.
            observations.append(derive_signals_from_events(tuple(events), parsed=Tri.TRUE))
    finally:
        conn.close()

    return _tally(observations, sample_root=path, source=opencode.SOURCE, threshold=threshold)


def collect_all_coverage(
    *,
    sample_root: Path | None = None,
    codex_root: Path | None = None,
    opencode_db: Path | None = None,
    limit: int | None = None,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> list[CoverageReport]:
    """Measure every requested source, in `SOURCES` order.

    Args:
        sample_root: Claude Code sample, or `None`.
        codex_root: Codex sample, or `None`.
        opencode_db: OpenCode store, or `None`.
        limit: Per-source session cap.
        now: Reference time for the Claude Code observations.
        on_status: Progress channel (INV-1).
        threshold: Coverage percentage the per-source gate is applied at.

    Returns:
        One report per measured source. Naming any sample restricts the run
        to the named sources; naming none measures all three at their
        production defaults — see the module docstring for why a test that
        names one sample must never reach the other two sources' real
        stores.
    """
    requested = [
        source
        for source, root in zip(SOURCES, (sample_root, codex_root, opencode_db), strict=True)
        if root is not None
    ]
    if not requested:
        requested = list(SOURCES)

    reports = []
    for source in SOURCES:
        if source not in requested:
            continue
        if source == ClaudeCodeAdapter.source:
            reports.append(
                collect_coverage(
                    sample_root,
                    limit=limit,
                    now=now,
                    on_status=on_status,
                    threshold=threshold,
                )
            )
        elif source == CodexAdapter.source:
            reports.append(
                collect_codex_coverage(
                    codex_root, limit=limit, on_status=on_status, threshold=threshold
                )
            )
        else:
            reports.append(
                collect_opencode_coverage(
                    opencode_db, limit=limit, on_status=on_status, threshold=threshold
                )
            )
    return reports


def render_coverage(report: CoverageReport) -> str:
    """Render one source's `CoverageReport` as a block of the command's output.

    One row per entry in `SIGNAL_NAMES`, so a signal added to `Signals`
    appears here without editing this function and a signal dropped from the
    report is a test failure rather than an omission nobody notices.

    Args:
        report: The counts to render.

    Returns:
        The block's text, newline-terminated.
    """
    lines = [
        f"sample: {report.sample_root}",
        f"source: {report.source}",
        f"sessions: {report.sessions}",
        "",
        f"{'signal':<24} {'determinable':>12} {'coverage':>9}",
    ]
    for name in SIGNAL_NAMES:
        counted = f"{report.determinable[name]}/{report.sessions}"
        lines.append(f"{name:<24} {counted:>12} {report.percentage(name):>8.1f}%")

    statuses = (
        ", ".join(
            f"{status.value} {report.statuses[status]}"
            for status in Status
            if report.statuses[status]
        )
        or "none"
    )
    bases = (
        ", ".join(f"{basis} {report.bases[basis]}" for basis in BASIS_NAMES if report.bases[basis])
        or "none"
    )

    under = report.under_covered_signals()
    if under:
        withdrawn = report.statuses[Status.UNKNOWN] - report.ungated_statuses[Status.UNKNOWN]
        gate = (
            f"gate: {', '.join(under)} below {report.threshold:.1f}% — "
            f"{withdrawn} status(es) withdrawn to UNKNOWN for this source"
        )
    else:
        gate = f"gate: every signal at or above {report.threshold:.1f}%, nothing withdrawn"

    lines.extend(
        [
            "",
            f"status: {statuses}",
            f"boundary basis: {bases}",
            (
                "corroboration: "
                f"agrees {report.corroboration[Tri.TRUE]}, "
                f"disagrees {report.corroboration[Tri.FALSE]}, "
                f"unavailable {report.corroboration[Tri.UNKNOWN]}"
            ),
            gate,
        ]
    )
    return "\n".join(lines) + "\n"


def render_reports(reports: Sequence[CoverageReport]) -> str:
    """Render every measured source, then the footer, once.

    Args:
        reports: One report per source, in `SOURCES` order.

    Returns:
        The command's stdout output, newline-terminated.
    """
    blocks = ["palaver diagnose --coverage\n"]
    blocks.extend(render_coverage(report) for report in reports)
    blocks.append(
        "note: coverage counts sessions a signal was determinable for, not\n"
        "sessions it was right about. Accuracy is the labelled fixture\n"
        "corpus's job (task 1.7), not this command's.\n"
    )
    return "\n".join(blocks)


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver diagnose`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 on a completed measurement, 1 when no measured source held a
        single session (a coverage report over an empty sample is not a
        measurement), 2 when no mode flag was given. One source being empty
        is not a failure when another was measured — a machine need not run
        all three harnesses.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    if not args.coverage:
        print("palaver diagnose: nothing to do; pass --coverage", file=sys.stderr)
        return 2

    reports = collect_all_coverage(
        sample_root=args.sample,
        codex_root=args.codex_sample,
        opencode_db=args.opencode_db,
        limit=args.limit,
        on_status=on_status,
        threshold=args.threshold,
    )
    if not any(report.sessions for report in reports):
        roots = ", ".join(str(report.sample_root) for report in reports)
        print(f"palaver diagnose: no sessions found under {roots}", file=sys.stderr)
        return 1

    out.write(render_reports(reports))
    return 0
