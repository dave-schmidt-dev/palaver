"""
Encoding a cwd into its on-disk project key, and listing the sessions recently written
under it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from .constants import DEFAULT_ACTIVITY_WINDOW


def project_key_for_cwd(cwd: Path) -> str:
    """Encode a working directory the way the on-disk store names its project.

    Claude Code writes `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`,
    where the encoding replaces both `/` and `.` with `-`. Verified against
    real directories on this machine covering the plain case, a dotfile
    directory (`~/.launchd` → `-Users-dave--launchd`, note the doubled dash),
    and an underscore in a project name (preserved).

    The encoding is deliberately not invertible and no inverse is offered:
    `-Users-dave--launchd` has several pre-images, and a decoder would have
    to guess between them. Callers go in this direction only, checking that
    the encoded directory exists rather than decoding one that does.

    Args:
        cwd: An absolute working directory.

    Returns:
        The encoded project directory name.
    """
    return str(cwd).replace("/", "-").replace(".", "-")


def project_keys_for_cwd(cwd: Path) -> tuple[str, ...]:
    """Name every encoding under which this cwd's project directory may exist.

    Measured on this machine (2026-08-18): `~/.claude/projects` holds both
    `-Users-dave-Documents-Projects-CipherBlade-okx_case`, which preserved an
    underscore, and `-Users-dave-Documents-Projects-fairaday-labs`, which did
    not -- Claude Code changed the encoding between the two, and both
    directories are still on disk. A single rule therefore cannot be right,
    so both are offered and the one that exists wins.

    Args:
        cwd: An absolute working directory.

    Returns:
        Candidate directory names, most historically faithful first.
    """
    preserved = project_key_for_cwd(cwd)
    converted = preserved.replace("_", "-")
    return (preserved,) if converted == preserved else (preserved, converted)


def session_candidates(
    project_key: str,
    sessions_root: Path,
    *,
    now: datetime,
    activity_window: timedelta = DEFAULT_ACTIVITY_WINDOW,
) -> tuple[str, ...]:
    """Name the sessions in a project that could be the one on screen.

    Reads directory entries and mtimes only — never a byte of any transcript
    (INV-9).

    Args:
        project_key: The encoded project directory name.
        sessions_root: The store root, e.g. `~/.claude/projects`.
        now: Reference time the window is measured back from. Compared
            against store mtimes, which are absolute epoch seconds, so this
            is converted with `.timestamp()` and follows its rule: an aware
            datetime is exact, a naive one is read as *local* time. Passing a
            naive UTC clock therefore shifts the cutoff by the UTC offset —
            silently, since it still returns a plausible-looking list. The
            rest of the tree (`collect_status`) is aware-UTC and callers
            should stay that way; `join_pane`'s default does.
        activity_window: How recently a store must have been written.

    Returns:
        Session ids (store filename stems) written within the window,
        sorted. Empty when the project directory is absent or nothing in it
        is recent.
    """
    directory = sessions_root / project_key
    if not directory.is_dir():
        return ()
    cutoff = (now - activity_window).timestamp()
    found = []
    for store in directory.glob("*.jsonl"):
        try:
            mtime = store.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            found.append(store.stem)
    return tuple(sorted(found))
