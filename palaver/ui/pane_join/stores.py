"""
Resolving a project's session store: the Claude registry, Codex candidate narrowing, and
a validated pin.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping, MutableMapping
from datetime import datetime, timedelta
from pathlib import Path

from palaver.ingest.adapters.codex import CodexAdapter

from .candidates import project_keys_for_cwd
from .constants import CLAUDE_SOURCE, CODEX_SOURCE, default_registry_root, default_store_roots
from .pin import PanePin


def registered_session(pid: int, cwd: Path, *, registry_root: Path | None = None) -> str | None:
    """Read the session id Claude Code itself published for `pid`.

    This is the only exact pane-to-transcript discriminator found on this
    machine. The alternative -- narrowing a project directory by mtime --
    cannot separate the live session from the several others run in the same
    project within the hour, which is what leaves a pane UNJOINED.

    Every field is checked rather than trusted: the record must claim the pid
    that was asked for, must claim the same cwd the pane and the agent
    process already agreed on, and must carry a session id that is a plain
    filename. A registry naming some other directory is a stale file from a
    reused pid, not a join.

    Args:
        pid: The agent process resolved from the pane.
        cwd: The directory the pane and the agent process agree on.
        registry_root: The registry directory; defaults to the real one.

    Returns:
        The session id, or `None` if the registry cannot prove one.
    """
    root = default_registry_root() if registry_root is None else registry_root
    try:
        raw = (root / f"{pid}.json").read_text(encoding="utf-8")
    except OSError, ValueError:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(record, dict) or record.get("pid") != pid:
        return None
    registered_cwd = record.get("cwd")
    if not isinstance(registered_cwd, str) or Path(registered_cwd) != cwd:
        return None
    session_id = record.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        return None
    if session_id in {".", ".."} or "/" in session_id or "\\" in session_id:
        return None
    return session_id


def _registered_store(root: Path, cwd: Path, session_id: str) -> Path | None:
    """Locate one named session's transcript under either project encoding."""
    for key in project_keys_for_cwd(cwd):
        store = root / key / f"{session_id}.jsonl"
        if _readable_file(store):
            return store.resolve(strict=False)
    # Neither encoding exists, so the directory name is one this code does
    # not know how to spell. The session id is unique, so a single match
    # anywhere under the root identifies it without guessing at spelling.
    matches = sorted(root.glob(f"*/{session_id}.jsonl"))
    if len(matches) == 1 and _readable_file(matches[0]):
        return matches[0].resolve(strict=False)
    return None


def _root_for_source(
    source: str,
    *,
    sessions_root: Path | None,
    store_roots: Mapping[str, Path] | None,
) -> Path | None:
    """Resolve one source root without making a missing source disable others."""
    if store_roots is not None:
        raw = store_roots.get(source)
        return None if raw is None else Path(raw).expanduser()
    if sessions_root is not None:
        # Backward-compatible single-root injection used by existing callers.
        return Path(sessions_root)
    return default_store_roots().get(source)


def _codex_home_from_lock_path(path: Path) -> Path | None:
    """Extract a Codex home directory from an exact lockfile path.

    Live Codex processes hold `<home>/tmp/arg0/codex-arg0XXXX/.lock`.
    Validates the exact ancestor chain and refuses non-absolute paths,
    malformed ancestors, or root/system directories.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute() or path.name != ".lock":
        return None
    parents = path.parents
    if len(parents) < 4:
        return None
    if not parents[0].name.startswith("codex-arg0"):
        return None
    if parents[1].name != "arg0":
        return None
    if parents[2].name != "tmp":
        return None
    home = parents[3].resolve(strict=False)
    if (
        home == home.parent
        or not home.name
        or str(home)
        in (
            "/",
            "/private",
            "/var",
            "/tmp",
            "/etc",
            "/usr",
            "/System",
            "/Library",
            "/bin",
            "/sbin",
            "/dev",
        )
    ):
        return None
    return home


def _codex_home_from_rollout_path(path: Path) -> Path | None:
    """Extract a Codex home directory from a canonical rollout file path.

    Accepts `<home>/sessions/YYYY/MM/DD/rollout-*.jsonl`.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute():
        return None
    if not (path.name.startswith("rollout-") and path.name.endswith(".jsonl")):
        return None
    parents = path.parents
    if len(parents) < 5:
        return None
    if len(parents[0].name) != 2 or not parents[0].name.isdigit():
        return None
    if len(parents[1].name) != 2 or not parents[1].name.isdigit():
        return None
    if len(parents[2].name) != 4 or not parents[2].name.isdigit():
        return None
    if parents[3].name != "sessions":
        return None
    home = parents[4].resolve(strict=False)
    if (
        home == home.parent
        or not home.name
        or str(home)
        in (
            "/",
            "/private",
            "/var",
            "/tmp",
            "/etc",
            "/usr",
            "/System",
            "/Library",
            "/bin",
            "/sbin",
            "/dev",
        )
    ):
        return None
    return home


def _codex_homes_from_open_paths(paths: Iterable[Path]) -> set[Path]:
    """Extract unique validated Codex home directories from open file paths."""
    homes: set[Path] = set()
    for path in paths:
        home = _codex_home_from_lock_path(path)
        if home is None:
            home = _codex_home_from_rollout_path(path)
        if home is not None:
            homes.add(home)
    return homes


def _readable_file(path: Path) -> bool:
    """Return whether ``path`` is a regular readable file."""
    try:
        return path.is_file() and os.access(path, os.R_OK)
    except OSError:
        return False


def _codex_store_candidates(
    root: Path,
    cwd: Path,
    *,
    now: datetime,
    activity_window: timedelta,
    cache: MutableMapping[tuple[str, str, str], tuple[Path, ...]] | None = None,
) -> tuple[Path, ...]:
    """Find recent root Codex rollouts whose metadata names exactly ``cwd``."""
    cache_key = (CODEX_SOURCE, str(root.resolve(strict=False)), str(cwd.resolve(strict=False)))
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    cutoff = (now - activity_window).timestamp()
    adapter = CodexAdapter(root)
    candidates: list[Path] = []
    for path in adapter.list_store_paths():
        try:
            if path.stat().st_mtime < cutoff:
                continue
            identity = adapter.read_identity(path)
        except OSError, ValueError, TypeError:
            continue
        # An identity may be temporarily unavailable while Codex flushes its
        # first record. Do not drop it and let descriptor narrowing select an
        # older candidate: the only safe answer is to refuse this tick.
        if identity is None:
            result: tuple[Path, ...] = ()
            if cache is not None:
                cache[cache_key] = result
            return result
        if identity.is_subagent or identity.cwd is None:
            continue
        if Path(identity.cwd).resolve(strict=False) != cwd.resolve(strict=False):
            continue
        if _readable_file(path):
            candidates.append(path.resolve(strict=False))
    result = tuple(sorted(candidates))
    if cache is not None:
        cache[cache_key] = result
    return result


def _agent_open_store_paths(pid: int) -> frozenset[Path]:
    """Return every absolute path `pid` currently holds a file descriptor open for.

    A candidate rollout this pid does not hold open cannot be the transcript
    it is currently writing — a structural fact about file descriptors, not a
    guess — so `join_pane` uses this to narrow an already-ambiguous
    cwd-and-mtime candidate set rather than to pick among it blind. Deliberately
    not used alone: fact 3 in the module docstring measured **ten** rollouts
    open on a single codex pid at once, spanning old sessions as well as the
    live one, so an open fd proves nothing on its own — only an intersection
    that narrows the existing candidate set to exactly one is trusted by the
    caller.

    `join_pane` verifies the process identity again immediately before it
    calls this function. A reused pid is therefore left ambiguous instead of
    being allowed to narrow candidates using an unrelated process's files.

    Args:
        pid: The agent's own pid, resolved by `agent_ancestor`.

    Returns:
        Resolved absolute paths of every open regular file, or an empty set
        if `lsof` is unavailable, the pid is gone, or it holds nothing open —
        any of which leaves the caller's candidate set exactly as ambiguous
        as before, never more so. Pipes, sockets, and other non-path
        descriptors (`lsof` reports these as e.g. `n->0x...`, with no leading
        `/`) are excluded rather than resolved against the caller's own cwd.
    """
    try:
        completed = subprocess.run(
            ["/usr/sbin/lsof", "-a", "-p", str(pid), "-Fn"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError, subprocess.SubprocessError, UnicodeDecodeError:
        return frozenset()
    if completed.returncode != 0:
        return frozenset()
    return frozenset(
        Path(line[1:]).resolve(strict=False)
        for line in completed.stdout.splitlines()
        if line.startswith("n/")
    )


CodexCandidateProgress = MutableMapping[str, tuple[int, int]]


def _narrow_codex_candidates_by_progress(
    paths: tuple[Path, ...],
    progress: CodexCandidateProgress,
    *,
    identity: tuple[int, Path],
) -> tuple[Path, ...]:
    """Narrow an ambiguous Codex set to one rollout that advanced between ticks.

    The observation contains only file metadata: size and nanosecond mtime.
    A candidate set must be identical across two observations, and metadata
    must never move backwards. Any unreadable, changing, or multiply
    advancing set remains ambiguous.
    """
    identity_prefix = f"{identity[0]}\0{identity[1]}\0"
    current: dict[str, tuple[int, int]] = {}
    try:
        for path in paths:
            stat = path.stat()
            current[f"{identity_prefix}{path}"] = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        progress.clear()
        return paths

    previous = dict(progress)
    progress.clear()
    progress.update(current)
    if set(previous) != set(current):
        return paths

    advancing: list[Path] = []
    for path in paths:
        old_size, old_mtime = previous[f"{identity_prefix}{path}"]
        size, mtime = current[f"{identity_prefix}{path}"]
        if size < old_size or mtime < old_mtime:
            return paths
        if size > old_size or mtime > old_mtime:
            advancing.append(path)
    return tuple(advancing) if len(advancing) == 1 else paths


def _pinned_store_path(root: Path, pin: PanePin) -> Path | None:
    """Validate a pin's source store, allowing an intentional cwd mismatch."""
    if pin.source == CLAUDE_SOURCE:
        project_key, session_id = pin.session_key.split("/")
        candidate = root / project_key / f"{session_id}.jsonl"
        matches = [candidate] if _readable_file(candidate) else []
    else:
        matches = (
            [path for path in root.rglob(f"{pin.session_key}.jsonl") if _readable_file(path)]
            if root.is_dir()
            else []
        )
        adapter = CodexAdapter(root)
        matches = [
            path
            for path in matches
            if (identity := adapter.read_identity(path)) is not None and not identity.is_subagent
        ]
    if len(matches) != 1:
        return None
    return matches[0].resolve(strict=False)
