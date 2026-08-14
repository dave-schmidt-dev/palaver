"""`palaver inspect <session>`: the full signal set for one session, three-valued.

`palaver status` (task 1.9, same file list) collapses a session down to one
`Status` word. That collapse is deliberately lossy — `derive_status()`'s rule
list (`palaver/observer/signals.py`) folds `FALSE` and `UNKNOWN` together in
both directions depending on the rule, so two sessions that print the same
status line can have gotten there for different structural reasons. This
command exists so a wrong status is diagnosable *without a debugger*: it
prints every signal `Signals` defines, each with its own three-valued answer,
plus the turn-boundary basis and corroboration behind `agent_turn_ended`
specifically — the one signal `derive_turn_boundary` computes structurally
rather than reading directly off a record.

**Every signal name comes from `SIGNAL_NAMES`, never a hardcoded list.**
`SIGNAL_NAMES` is itself derived from `dataclasses.fields(Signals)`
(`palaver/observer/signals.py`), so a signal added to `Signals` in a later
phase appears here automatically, including — and this is the point — one
whose value comes back `Tri.UNKNOWN` for the session being inspected.
Absence is exactly the case this command must show, not hide: hardcoding a
list of names presumed non-empty would silently pass every field it was
written before, whether that field's value is present or not.

Like `status.py`, no model client, results on stdout, progress on stderr
(INV-1). Unlike `status.py`, discovery here ignores the default activity
window (`discover_sessions(all=True)`): naming one session by id is an
explicit request, and Phase 1's activity floor exists to bound an unscoped
scan, not to hide a session someone asked for by name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from palaver.ingest.adapters.base import SessionRef
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import SIGNAL_NAMES, Signals, derive_status
from palaver.observer.turn_boundary import TurnBoundary, observe_session

NAME = "inspect"
HELP = "show the full signal set for one session"


class SessionLookupError(Exception):
    """Raised when `resolve_session` cannot resolve exactly one session.

    Args:
        message: A description naming what was searched for and why the
            lookup failed (zero matches, or which matches were ambiguous) —
            `run()` writes this straight to stderr, so it is written for
            that audience, not just for a traceback.
    """


def resolve_session(sample_root: Path | None, session: str) -> SessionRef:
    """Find the one session `session` names, searching every discovered session.

    Accepts either a fully-qualified `session_key` (`"<project>/<session-id>"`,
    as printed by `palaver status`) or a bare session id (a store's filename
    stem). Search is unwindowed (`discover_sessions(all=True)`): `inspect`
    names one session explicitly, so the activity floor that bounds
    `palaver status`'s unscoped scan does not apply here.

    Args:
        sample_root: Root directory laid out as
            `<root>/<project>/<session>.jsonl`, or `None` for the adapter's
            default (Claude Code's own projects directory).
        session: A `session_key` or a bare session id to look up.

    Returns:
        The one matching `SessionRef`.

    Raises:
        SessionLookupError: If no discovered session matches `session`, or
            more than one does (a bare id is not guaranteed unique across
            projects; a full `session_key` is, but is still checked the same
            way rather than special-cased).
    """
    adapter = ClaudeCodeAdapter(root=sample_root)
    matches = [
        ref
        for ref in adapter.discover_sessions(all=True)
        if ref.session_key == session or ref.path.stem == session
    ]

    if not matches:
        raise SessionLookupError(f"no discovered session matches {session!r}")
    if len(matches) > 1:
        keys = ", ".join(sorted(ref.session_key for ref in matches))
        raise SessionLookupError(
            f"{session!r} matches more than one session ({keys}); "
            "pass the full session_key (<project>/<session-id>) to disambiguate"
        )
    return matches[0]


@dataclass(frozen=True)
class Inspection:
    """The full Phase 1 reading of one session, ready to render.

    Attributes:
        session_key: The session's durable identity (`<project>/<session-id>`).
        path: Path to the session's store, for display only.
        status: The derived `Status`, same value `palaver status` would print.
        signals: The full three-valued `Signals` behind `status`.
        boundary: The turn boundary behind `signals.agent_turn_ended`, with
            its basis and corroboration.
    """

    session_key: str
    path: Path
    status: object
    signals: Signals
    boundary: TurnBoundary


def collect_inspection(
    sample_root: Path | None,
    session: str,
    *,
    now: datetime | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Inspection:
    """Resolve `session` and read its full Phase 1 signal set.

    Args:
        sample_root: Root directory laid out as
            `<root>/<project>/<session>.jsonl`, or `None` for the adapter's
            default (Claude Code's own projects directory).
        session: A `session_key` or bare session id, per `resolve_session`.
        now: Reference time for mtime corroboration. Defaults to the current
            UTC time; tests pass a fixed value.
        on_status: Progress channel, called once before the session is read
            (INV-1). Never writes to stdout.

    Returns:
        The `Inspection` for the resolved session.

    Raises:
        SessionLookupError: Propagated from `resolve_session`.
    """
    ref = resolve_session(sample_root, session)
    if on_status is not None:
        on_status(f"observing {ref.session_key}")
    observation = observe_session(ref.path, now=now)
    return Inspection(
        session_key=ref.session_key,
        path=ref.path,
        status=derive_status(observation.signals),
        signals=observation.signals,
        boundary=observation.boundary,
    )


def render_inspection(inspection: Inspection) -> str:
    """Render an `Inspection` as `palaver inspect`'s stdout output.

    Every line under "signals:" comes from iterating `SIGNAL_NAMES`
    (`palaver/observer/signals.py`) rather than naming fields directly, so a
    signal added to `Signals` in a later phase appears here without this
    function changing — including, deliberately, one whose value on this
    session is `Tri.UNKNOWN`: absence is itself the answer this command
    exists to show, not a line to omit.

    Args:
        inspection: The inspection to render.

    Returns:
        The report text, terminated by a single trailing newline.
    """
    lines = [
        f"session: {inspection.session_key}",
        f"path: {inspection.path}",
        f"status: {inspection.status.value}",
        "signals:",
    ]
    lines.extend(f"  {name}: {getattr(inspection.signals, name).value}" for name in SIGNAL_NAMES)
    lines.append("turn_boundary:")
    lines.append(f"  ended: {inspection.boundary.ended.value}")
    lines.append(f"  basis: {inspection.boundary.basis}")
    lines.append(f"  corroboration: {inspection.boundary.corroboration.value}")
    return "".join(f"{line}\n" for line in lines)


def add_arguments(parser) -> None:
    """Register `inspect`'s positional argument and flags on its subparser."""
    parser.add_argument(
        "session",
        help="a session_key (<project>/<session-id>) or bare session id",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help=(
            "directory of session stores to search, laid out as "
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
    """Run `palaver inspect`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).
        now: Reference time, forwarded to `collect_inspection`. Not a CLI
            flag; tests call `run()` directly with a fixed value, the same
            pattern `status.py` uses.

    Returns:
        0 having printed the resolved session's full signal set, 1 if
        `session` did not resolve to exactly one session (message on
        stderr, nothing on stdout).
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    try:
        inspection = collect_inspection(args.sample, args.session, now=now, on_status=on_status)
    except SessionLookupError as exc:
        print(f"palaver inspect: {exc}", file=sys.stderr)
        return 1

    out.write(render_inspection(inspection))
    return 0
