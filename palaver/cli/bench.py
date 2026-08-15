"""`palaver bench`: what six sessions ticking at once actually costs.

The measurement itself lives in `palaver.bench`; this module is the argument
surface and the report renderer. Two choices here are worth stating because a
reader could reasonably expect the opposite:

**A failed round exits non-zero.** A benchmark whose model server is not
running would otherwise print a table of zeros, which reads exactly like a
passing measurement. When every session failed to connect, the first line of
stderr names the server that refused, because "connection refused" six times
over does not tell an operator which port to look at.

**`--report` widens the output, it does not enable the measurement.** The run
is identical either way; without the flag the command prints a summary, with it
the per-session table as well. The plan names the flag, and a flag that
silently did nothing would be worse than one that does something small.

Output follows the CLI's two-stream contract: the report goes to stdout and
per-request progress goes through `on_status` to stderr (INV-1), so a
multi-minute round against a real server is a wait with visible reasons.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from palaver.bench import (
    DEFAULT_SESSIONS,
    DEFAULT_TICK_INTERVAL,
    BenchError,
    BenchReport,
    run_bench,
)

NAME = "bench"
HELP = "drive N synthesized sessions concurrently and report what the round cost"

#: Written under the system temp directory for the same reason as
#: `palaver.cli.observe.DEFAULT_DB_PATH`: a benchmark store is throwaway
#: measurement scaffolding, not the user's real memory store, and a default
#: that wrote into the real one would mix benchmark `model_runs` rows with
#: observed ones.
DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "palaver-bench" / "bench.db"


def _megabytes(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _render_slot_files(report: BenchReport) -> list[str]:
    usage = report.slot_files
    if not usage.available:
        return ["slot files: unavailable", f"  {usage.detail}"]
    return [
        f"slot files: {usage.file_count} file(s), {_megabytes(usage.total_bytes)}",
        f"  {usage.path}",
    ]


def _render_sessions(report: BenchReport) -> list[str]:
    lines = ["", "per-session:"]
    for timing in report.timings:
        if timing.ok:
            lines.append(f"  {timing.label}: {timing.latency_ms} ms")
        else:
            lines.append(f"  {timing.label}: FAILED after {timing.latency_ms} ms ({timing.error})")
    return lines


def render_report(report: BenchReport, *, host: str, port: int, detailed: bool) -> str:
    """Render a finished round as the command's stdout output.

    Args:
        report: The finished round.
        host: Server host the round was driven against.
        port: Server port.
        detailed: Whether to include the per-session table (`--report`).

    Returns:
        The full report, newline-terminated.
    """
    verdict = "within" if report.fits_tick_interval else "OVER"
    lines = [
        "palaver bench",
        f"server: {host}:{port}",
        f"sessions: {report.sessions}",
        f"peak in flight: {report.peak_in_flight}",
        f"tick wall time: {report.tick_wall_s:.3f} s "
        f"({verdict} the {report.tick_interval_s:.1f} s tick interval)",
        f"peak RSS: {_megabytes(report.rss_after_bytes)} "
        f"(delta over the round {_megabytes(report.rss_delta_bytes)})",
        *_render_slot_files(report),
    ]
    if detailed:
        lines.extend(_render_sessions(report))
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Register `bench`'s arguments on its subparser."""
    parser.add_argument(
        "--sessions",
        type=int,
        default=DEFAULT_SESSIONS,
        help=f"sessions to drive concurrently (default: {DEFAULT_SESSIONS})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="also print the per-session latency table",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="llama-server host (default: 127.0.0.1; INV-9 permits no other)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="llama-server port (default: 8090)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds allowed per request (default: 60)",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=DEFAULT_TICK_INTERVAL,
        help=f"tick budget the round is judged against (default: {DEFAULT_TICK_INTERVAL})",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=None,
        help=(
            "approximate synthesized prompt size (default: derived from the "
            "server's own reported context budget, divided by its slot count)"
        ),
    )
    parser.add_argument(
        "--slot-save-path",
        type=Path,
        default=None,
        help="the server's --slot-save-path, which it will not report over HTTP",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"throwaway benchmark store (default: {DEFAULT_DB_PATH})",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver bench`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when every session's request succeeded, and 1 when any failed or the
        store could not be prepared. A round that overran its tick interval is
        still a successful *measurement* and exits 0 — the overrun is the
        finding, reported on stdout, not an error in the harness.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status
    db_path = DEFAULT_DB_PATH if args.db is None else args.db

    try:
        report = run_bench(
            db_path=db_path,
            sessions=args.sessions,
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            tick_interval=args.tick_interval,
            prompt_words=args.prompt_words,
            slot_save_path=args.slot_save_path,
            on_status=on_status,
        )
    except (BenchError, ValueError) as exc:
        print(f"palaver bench: {exc}", file=sys.stderr)
        return 1

    out.write(render_report(report, host=args.host, port=args.port, detailed=args.report))

    if report.unreachable:
        print(
            f"palaver bench: llama-server at {args.host}:{args.port} is unreachable — "
            f"all {report.sessions} session(s) failed to connect",
            file=sys.stderr,
        )
        return 1
    if not report.ok:
        for timing in report.failures:
            print(f"palaver bench: {timing.label} failed: {timing.error}", file=sys.stderr)
        return 1
    return 0
