"""`palaver observe`: run the observer daemon, or one tick of it (task 4.1).

This is the long-running process the rest of the system assumes exists — the
single SQLite writer and the owner of the tick loop. Its default mode is
continuous, which is what makes it a daemon rather than a cron job:
`--once` runs exactly one tick and exits, which is what makes it testable
and scriptable.

`--dry-run` answers "what *would* this tick spend inference on" without
spending any. It calls the scheduler directly, so it opens no database,
issues no model request, and — the part that matters — **saves no cursor**.
A dry run that advanced cursors would make the next real run believe every
session was already extracted, which is the exact failure a dry run exists
to avoid.

Output follows the CLI's two-stream rule (INV-1): one result line per tick
on stdout, and per-session progress on stderr, so `palaver observe --once |
...` stays a clean pipe while a sweep over a large store is never a silent
wait.

This repository is public. Nothing in this module's docstrings or examples
is derived from a real observed session (INV-9).
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import CursorStore
from palaver.observer.daemon import DEFAULT_INTERVAL, ObserverDaemon, TickResult
from palaver.observer.scheduler import TickPlan, plan_tick
from palaver.observer.socket import SingleWriterError, serve_until, single_writer

NAME = "observe"

HELP = "watch every discovered session and extract the ones that changed"

#: Default store location. Outside the repository for the same reason
#: `palaver.cli.replay.DEFAULT_DB_PATH` is: a default that writes inside a
#: checkout turns a stray invocation into a dirty tree. Task 5.0's launchd
#: agent passes an explicit `--db`, which is where the real store gets a
#: permanent home; nothing here should be treated as that home.
DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "palaver-observe" / "observe.db"

#: Default cursor root, kept beside the store so the two stay consistent
#: when either is thrown away.
DEFAULT_CURSOR_ROOT = Path(tempfile.gettempdir()) / "palaver-observe" / "cursors"


def add_arguments(parser) -> None:
    """Register `observe`'s flags on its subparser."""
    parser.add_argument(
        "--once",
        action="store_true",
        help="run exactly one tick and exit, instead of looping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report what one tick would extract, without opening the store, "
            "issuing any inference request, or saving any cursor"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"seconds between ticks (default: {DEFAULT_INTERVAL:g})",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="stop after this many ticks (default: run until interrupted)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"database file to write into (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--cursors",
        type=Path,
        default=None,
        help=f"directory holding durable cursors (default: {DEFAULT_CURSOR_ROOT})",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help=(
            "directory of session stores to observe, laid out as "
            "<root>/<project>/<session>.jsonl (default: Claude Code's own "
            "projects directory)"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="observe every discoverable session, skipping the recency window",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def render_tick(result: TickResult) -> str:
    """Render one tick as its single stdout line.

    Args:
        result: The tick to render.

    Returns:
        One line, newline-terminated, naming every count that distinguishes
        a healthy tick from a broken one — including `failed`, so a daemon
        whose model server is down does not read as a quiet daemon.
    """
    return (
        f"tick {result.tick}: discovered={result.plan.discovered} "
        f"changed={len(result.plan.scheduled)} "
        f"extracted={len(result.extracted)} "
        f"failed={len(result.failed)}\n"
    )


def render_plan(plan: TickPlan) -> str:
    """Render a dry run's plan: the counts, then one line per changed session."""
    lines = [
        f"dry-run: discovered={plan.discovered} "
        f"changed={len(plan.scheduled)} unchanged={len(plan.skipped)}"
    ]
    for work in plan.scheduled:
        advance = work.cursor_after.offset - work.cursor_before.offset
        lines.append(
            f"  {work.ref.source} {work.ref.session_key} "
            f"{advance:+d} bytes {len(work.events)} events"
        )
    return "\n".join(lines) + "\n"


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> int:
    """Run `palaver observe`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).
        now: Reference time for the discovery window. Not a CLI flag — tests
            that need determinism call `run()` directly with a fixed value,
            the same way `status.run` is tested (task 1.9).

    Returns:
        0. A tick whose extractions failed is still a tick that ran; the
        failure count is on the tick's stdout line, and a run that cannot
        even open its store raises rather than returning a status.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    adapters = (ClaudeCodeAdapter(root=args.sample),)
    cursor_root = DEFAULT_CURSOR_ROOT if args.cursors is None else args.cursors
    cursors = CursorStore(cursor_root)

    if args.dry_run:
        plan = plan_tick(
            adapters,
            cursors,
            all=args.all,
            now=now,
            on_status=on_status,
        )
        out.write(render_plan(plan))
        return 0

    db_path = DEFAULT_DB_PATH if args.db is None else args.db
    max_ticks = 1 if args.once else args.max_ticks

    def emit(result: TickResult) -> None:
        out.write(render_tick(result))
        out.flush()

    # `single_writer` wraps the daemon rather than the other way round: it
    # is what establishes the right to write at all, so nothing that writes
    # may be constructed before it is held. A daemon opened first would have
    # migrated the store and taken a connection before discovering another
    # daemon owns it (task 6.3).
    try:
        with single_writer(db_path, on_status=on_status) as request_socket:
            with ObserverDaemon(
                db_path=db_path,
                adapters=adapters,
                cursors=cursors,
                on_status=on_status,
                all=args.all,
            ) as daemon:
                # Requests are served *in place of* the sleep between ticks,
                # on this same thread. A second thread accepting them would
                # put two threads on one SQLite connection — the two-writer
                # problem moved inside the process, where the lock cannot
                # see it. See `serve_until`.
                def idle(seconds: float) -> None:
                    served = serve_until(request_socket, daemon.conn, seconds)
                    if served:
                        on_status(f"served {served} write request(s) while idle")

                try:
                    daemon.run(
                        interval=args.interval,
                        max_ticks=max_ticks,
                        on_tick=emit,
                        sleep=idle,
                        now=now,
                    )
                except KeyboardInterrupt:
                    # The unbounded mode's normal ending. Exiting 0 here —
                    # after the context manager closes the writer — is what
                    # makes a launchd stop (task 5.0) a clean shutdown
                    # rather than a crash report.
                    on_status("interrupted; closing the writer")
    except SingleWriterError as exc:
        # Not a traceback: a second daemon on one machine is an ordinary
        # thing to attempt (launchd retry, a stale terminal), and the person
        # who did it needs the sentence, not the stack.
        out.write(f"palaver observe: {exc}\n")
        return 2
    return 0
