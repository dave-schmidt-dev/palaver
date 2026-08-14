"""`palaver replay <fixture>`: replay a recorded session fixture end to end (task 2.5).

Thin argument-parsing and rendering shell around `palaver.replay.replay` —
same split as every other subcommand in this package (`status.py`,
`inspect.py`): this module owns argparse wiring and stdout/stderr framing
(INV-1: stdout is the result, stderr is progress), and none of the actual
adapter/signals/events/memory logic, which lives in `palaver.replay` so it
can be called directly from tests without going through argument parsing.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from palaver.replay import ReplayResult, replay

NAME = "replay"
HELP = "replay a recorded session fixture through adapter, signals, events, and memory"

#: Default database location when `--db` is not given: a fixed path under the
#: OS temp directory, deliberately outside this repository. Palaver has no
#: project-local data directory of its own yet — the plan's Phase 6.3 is
#: where `palaver.db` and `palaver.lock` get a real home beside each other —
#: so a bare `palaver replay <fixture>` (the plan's task 2.5 done-when
#: criterion) must not risk writing a stray, untracked file into the repo
#: tree. Fixed rather than a fresh temp directory per call, so two manual
#: runs against the same fixture actually show the idempotent second pass
#: the done-when criteria describe, not just prove it in the test suite.
DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "palaver-replay" / "replay.db"


def add_arguments(parser) -> None:
    """Register `replay`'s positional argument and flags on its subparser."""
    parser.add_argument(
        "fixture",
        type=Path,
        help="path to a JSONL session store, opened read-only (INV-2)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"database file to replay into (default: {DEFAULT_DB_PATH})",
    )


def render_replay(result: ReplayResult) -> str:
    """Render a `ReplayResult` as `palaver replay`'s stdout output."""
    lines = (
        f"session: {result.session_key}",
        f"status: {result.status.value}",
        f"events written: {result.events_written}",
        f"chunks written: {result.chunks_written}",
        f"memories written: {result.memories_written}",
        f"db: {result.db_path}",
    )
    return "".join(f"{line}\n" for line in lines)


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> int:
    """Run `palaver replay`.

    Args:
        args: Parsed arguments from this subcommand's parser (`fixture`,
            `db`).
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).
        now: Reference time, forwarded to `replay()`. Not a CLI flag — tests
            that need determinism call `run()` directly with a fixed value,
            same convention as `status.run`/`inspect.run`.

    Returns:
        0 having replayed the fixture and printed its result. 1 if the
        fixture could not be read — a diagnosable error, not a crash.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status
    db_path = DEFAULT_DB_PATH if args.db is None else args.db

    try:
        result = replay(args.fixture, db_path, now=now, on_status=on_status)
    except OSError as exc:
        print(f"palaver replay: {exc}", file=sys.stderr)
        return 1

    out.write(render_replay(result))
    return 0
