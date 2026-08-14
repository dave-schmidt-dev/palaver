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
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import SIGNAL_NAMES, Status, Tri, derive_status
from palaver.observer.turn_boundary import BASIS_NAMES, observe_session

NAME = "diagnose"
HELP = "measure per-signal coverage over a sample of session transcripts"


@dataclass(frozen=True)
class CoverageReport:
    """Counts behind one `--coverage` run.

    Attributes:
        sample_root: Directory the sample was read from.
        source: Adapter source name the sample was read with.
        sessions: How many sessions were observed.
        determinable: Per signal name, how many sessions produced a value
            other than `Tri.UNKNOWN`.
        statuses: How many sessions produced each `Status`.
        bases: How many sessions each turn-boundary basis accounted for.
        corroboration: How many boundaries an independent signal agreed with
            (`Tri.TRUE`), disagreed with (`Tri.FALSE`), or was unavailable
            for (`Tri.UNKNOWN`).
    """

    sample_root: Path
    source: str
    sessions: int
    determinable: dict[str, int]
    statuses: Counter
    bases: Counter
    corroboration: Counter

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
            "directory of session stores to measure, laid out as "
            "<root>/<project>/<session>.jsonl (default: Claude Code's own "
            "projects directory)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="observe at most this many sessions from the sample",
    )


def collect_coverage(
    sample_root: Path | None,
    *,
    limit: int | None = None,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> CoverageReport:
    """Observe every session in a sample and count what was determinable.

    Args:
        sample_root: Sample directory, or `None` for the adapter's default.
        limit: Maximum number of sessions to observe, or `None` for all.
        now: Reference time passed through to each observation.
        on_status: Progress channel, called once per session before it is
            read (INV-1). Never writes to stdout.

    Returns:
        A `CoverageReport` over the sessions actually observed.
    """
    adapter = ClaudeCodeAdapter(root=sample_root)
    refs = adapter.discover_sessions(all=True)
    if limit is not None:
        refs = refs[:limit]

    determinable = dict.fromkeys(SIGNAL_NAMES, 0)
    statuses: Counter = Counter()
    bases: Counter = Counter()
    corroboration: Counter = Counter()

    total = len(refs)
    for index, ref in enumerate(refs, start=1):
        if on_status is not None:
            on_status(f"observing {index}/{total}: {ref.session_key}")
        observation = observe_session(ref.path, now=now)
        for name in SIGNAL_NAMES:
            if getattr(observation.signals, name) is not Tri.UNKNOWN:
                determinable[name] += 1
        statuses[derive_status(observation.signals)] += 1
        bases[observation.boundary.basis] += 1
        corroboration[observation.boundary.corroboration] += 1

    return CoverageReport(
        sample_root=Path(adapter.root),
        source=adapter.source,
        sessions=total,
        determinable=determinable,
        statuses=statuses,
        bases=bases,
        corroboration=corroboration,
    )


def render_coverage(report: CoverageReport) -> str:
    """Render a `CoverageReport` as the command's stdout output.

    One row per entry in `SIGNAL_NAMES`, so a signal added to `Signals`
    appears here without editing this function and a signal dropped from the
    report is a test failure rather than an omission nobody notices.

    Args:
        report: The counts to render.

    Returns:
        The report text, newline-terminated.
    """
    lines = [
        "palaver diagnose --coverage",
        f"sample: {report.sample_root}",
        f"source: {report.source}",
        f"sessions: {report.sessions}",
        "",
        f"{'signal':<24} {'determinable':>12} {'coverage':>9}",
    ]
    for name in SIGNAL_NAMES:
        counted = f"{report.determinable[name]}/{report.sessions}"
        lines.append(f"{name:<24} {counted:>12} {report.percentage(name):>8.1f}%")

    statuses = ", ".join(
        f"{status.value} {report.statuses[status]}" for status in Status if report.statuses[status]
    )
    bases = ", ".join(
        f"{basis} {report.bases[basis]}" for basis in BASIS_NAMES if report.bases[basis]
    )
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
            "",
            "note: coverage counts sessions a signal was determinable for, not",
            "sessions it was right about. Accuracy is the labelled fixture",
            "corpus's job (task 1.7), not this command's.",
        ]
    )
    return "\n".join(lines) + "\n"


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
        0 on a completed measurement, 1 when the sample held no sessions (a
        coverage report over an empty sample is not a measurement), 2 when no
        mode flag was given.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    if not args.coverage:
        print("palaver diagnose: nothing to do; pass --coverage", file=sys.stderr)
        return 2

    report = collect_coverage(args.sample, limit=args.limit, on_status=on_status)
    if report.sessions == 0:
        print(
            f"palaver diagnose: no sessions found under {report.sample_root}",
            file=sys.stderr,
        )
        return 1

    out.write(render_coverage(report))
    return 0
