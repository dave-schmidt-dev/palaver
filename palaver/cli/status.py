"""`palaver status --once`: one line per discovered session, right now.

This is the headline command the whole phase exists to ship (plan §1, "Phase 1
ships a working `palaver status` with no model in the loop at all"). It reuses
exactly the pieces earlier tasks built and adds no new derivation of its own:
`ClaudeCodeAdapter.discover_sessions()` (task 1.3) for the *floor, not a
filter* activity window, `observe_session()` (task 1.6) for the deterministic
signal set, and `derive_status()` (task 1.5) for the status itself. Neither
this module nor `inspect.py` constructs a model client — there is nothing
here for INV-9's gate test to catch, and that absence is the point: Phase 1's
status is a pure function of structure.

**`--once` is deliberate, not a placeholder flag.** Phase 1 has no scheduler
(`palaver observe`, task 4.1) and no continuous mode; `--once` names that
honestly instead of silently defaulting to a single pass that a future
watch-mode `palaver status` (no flag) would then have to change the meaning
of. Running the command with no flag is a usage error today, not a silent
alias for `--once`.

**The "age" column is store mtime, not a record timestamp.** Phase 1's
fixtures deliberately carry no `timestamp` field (`tests/fixtures/README.md`)
and nothing this module reads parses one from a real transcript either — the
signal set INV-7 defined is structural, and mtime is the one piece of
"when did this session last do anything" information `discover_sessions`
already collects on every `SessionRef` for its own windowing. Reusing it here
means `palaver status`'s age column and the window that selected the row it's
printed on are answering the same question from the same evidence.

Two output rules, same as every subcommand: stdout is the result (one line
per session), stderr is progress (`on_status`, INV-1) — a scan across many
stores is never a silent wait, and stdout stays clean for a pipe.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, TextIO

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import Status, derive_status
from palaver.observer.turn_boundary import observe_session

NAME = "status"
HELP = "show current status for discovered sessions"


@dataclass(frozen=True)
class SessionStatusRow:
    """One `palaver status` line's worth of data.

    Attributes:
        project: The project key `discover_sessions` found the session
            under (Claude Code's encoded project-directory name).
        session_id: The session's own id (its store's filename stem).
        status: The derived `Status` for this session, right now.
        age: How long ago the session's store was last modified.
    """

    project: str
    session_id: str
    status: Status
    age: timedelta


def _format_age(age: timedelta) -> str:
    """Render an age as the coarsest whole unit that fits, e.g. `45s`, `12m`, `3h`, `5d`.

    A pure function of the `timedelta` — no wall-clock read here — so it
    contributes nothing to the byte-identical-output test's determinism
    concerns; that rests entirely on the `now` passed into `collect_status`.
    A negative age (a store mtime after `now`, e.g. a clock skew) floors to
    `0s` rather than printing a sign nothing downstream expects.

    Args:
        age: Elapsed time since the session's store was last modified.

    Returns:
        The formatted age string.
    """
    seconds = max(0, int(age.total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def collect_status(
    sample_root: Path | None,
    *,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[SessionStatusRow, ...]:
    """Discover sessions within the default activity window and derive each one's status.

    Args:
        sample_root: Root directory laid out as
            `<root>/<project>/<session>.jsonl`, or `None` for the adapter's
            default (Claude Code's own projects directory).
        now: Reference time for both the discovery window and every
            observation's mtime corroboration. Defaults to the current UTC
            time; tests pass a fixed value so two calls are byte-identical.
        on_status: Progress channel, called once per session before it is
            read (INV-1). Never writes to stdout.

    Returns:
        One `SessionStatusRow` per session `discover_sessions` returns, in
        its order (the adapter's `list_store_paths` is itself sorted, so
        this order is deterministic for a fixed store).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    adapter = ClaudeCodeAdapter(root=sample_root)
    refs = adapter.discover_sessions(now=now)

    rows = []
    total = len(refs)
    for index, ref in enumerate(refs, start=1):
        if on_status is not None:
            on_status(f"observing {index}/{total}: {ref.session_key}")
        observation = observe_session(ref.path, now=now)
        rows.append(
            SessionStatusRow(
                project=adapter.project_key_for(ref.path),
                session_id=ref.path.stem,
                status=derive_status(observation.signals),
                age=now - datetime.fromtimestamp(ref.mtime, tz=timezone.utc),
            )
        )
    return tuple(rows)


def render_status(rows: tuple[SessionStatusRow, ...]) -> str:
    """Render `collect_status`'s rows as `palaver status`'s stdout output.

    One line per row, in the order given, with no header and no summary: the
    command's contract is "one line per discovered session" exactly, so a
    piped consumer sees only session lines.

    Args:
        rows: Rows from `collect_status`.

    Returns:
        The report text. Empty string when `rows` is empty — no sessions
        discovered is a valid, quiet answer, not a failure.
    """
    lines = [
        f"{row.project} {row.session_id} {row.status.value} {_format_age(row.age)}" for row in rows
    ]
    return "".join(f"{line}\n" for line in lines)


def add_arguments(parser) -> None:
    """Register `status`'s flags on its subparser."""
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "run a single pass over discovered sessions and exit (the only "
            "mode Phase 1 implements; a continuous mode is task 4.1's)"
        ),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help=(
            "directory of session stores to scan, laid out as "
            "<root>/<project>/<session>.jsonl (default: Claude Code's own "
            "projects directory)"
        ),
    )


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
    """Run `palaver status`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).
        now: Reference time, forwarded to `collect_status`. Not a CLI flag —
            tests that need determinism call `run()` directly with a fixed
            value instead of going through argument parsing, the same way
            `diagnose.collect_coverage` is tested (task 1.6).

    Returns:
        0 having printed a status line for every discovered session (zero
        lines is a valid outcome), 2 when `--once` was not given — Phase 1
        has no other mode to fall back to.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    if not args.once:
        print(
            "palaver status: nothing to do; pass --once "
            "(continuous mode is not implemented until task 4.1)",
            file=sys.stderr,
        )
        return 2

    rows = collect_status(args.sample, now=now, on_status=on_status)
    out.write(render_status(rows))
    return 0
