"""
Resolving session identity within a project: the activity-window scan, the Claude Code
live-process registry, and an explicit pane pin.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from palaver.ui.pane_join import (
    CLAUDE_SOURCE,
    CODEX_SOURCE,
    PaneVariables,
    encode_pin,
    join_pane,
    project_key_for_cwd,
    session_candidates,
)
from tests._pane_join_support import AGENT_PID, JOB_PID, NOW, _codex_rollout, _join, _table
from tests._pane_join_support import project as project


def test_codex_pin_recovers_a_rollout_after_the_pane_moves(tmp_path):
    old_cwd = tmp_path / "old-codex-project"
    live_cwd = tmp_path / "renamed-codex-project"
    old_cwd.mkdir()
    live_cwd.mkdir()
    root = tmp_path / "codex-sessions"
    store = _codex_rollout(root, old_cwd, "rollout-moved")
    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("codex-pane", 77201, "codex", str(live_cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: live_cwd,
        store_roots={CODEX_SOURCE: root},
        pin=encode_pin(CODEX_SOURCE, store.stem),
        now=NOW,
    )
    assert joined is not None
    assert joined.session_key == store.stem
    assert joined.store_path == store.resolve()


def test_pin_validates_source_and_supports_a_renamed_claude_cwd(tmp_path):
    old_cwd = tmp_path / "old-name"
    new_cwd = tmp_path / "renamed"
    old_cwd.mkdir()
    new_cwd.mkdir()
    root = tmp_path / "claude-projects"
    project = root / project_key_for_cwd(old_cwd)
    project.mkdir(parents=True)
    store = project / "session-1.jsonl"
    store.write_text("")
    os.utime(store, (NOW.timestamp(), NOW.timestamp()))
    variables = PaneVariables("pane-1", JOB_PID, "node", str(new_cwd))

    refused = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CODEX_SOURCE, "rollout-missing"),
        now=NOW,
    )
    assert refused is None

    missing = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CLAUDE_SOURCE, f"{project.name}/missing"),
        now=NOW,
    )
    assert missing is None

    joined = join_pane(
        variables,
        table=_table(),
        cwd_reader=lambda _pid: new_cwd,
        store_roots={CLAUDE_SOURCE: root},
        pin=encode_pin(CLAUDE_SOURCE, f"{project.name}/session-1"),
        now=NOW,
    )
    assert joined is not None
    assert joined.store_path == store.resolve()
    assert joined.session_key == f"{project.name}/session-1"


# --- the join: session candidates --------------------------------------------


def test_one_recent_session_resolves_and_two_refuse_to_pick(project):
    """Session identity is reported only when the evidence names exactly one.

    Measured on this machine, neither source offers a per-pane transcript
    discriminator: Claude Code holds no transcript open, and codex holds ten
    rollouts open at once. So two panes on one project resolve the project
    and stop, rather than both claiming the same session.
    """
    cwd, sessions_root = project
    project_dir = sessions_root / project_key_for_cwd(cwd)
    first = project_dir / "aaaa-1111.jsonl"
    first.write_text("", encoding="utf-8")
    os.utime(first, (NOW.timestamp() - 60, NOW.timestamp() - 60))

    join = _join(cwd, sessions_root)
    assert join is not None
    assert join.session_candidates == ("aaaa-1111",)
    assert join.session_key == "aaaa-1111"

    second = project_dir / "bbbb-2222.jsonl"
    second.write_text("", encoding="utf-8")
    os.utime(second, (NOW.timestamp() - 30, NOW.timestamp() - 30))

    ambiguous = _join(cwd, sessions_root)
    assert ambiguous is not None
    assert ambiguous.session_candidates == ("aaaa-1111", "bbbb-2222")
    assert ambiguous.session_key is None, "picked one of two indistinguishable sessions"


def test_a_session_older_than_the_activity_window_is_not_a_candidate(project):
    """A project directory accumulates every session ever run there.

    Without the window the candidate list is the project's whole history, so
    `session_key` would be `None` forever and the field would be dead.
    """
    cwd, sessions_root = project
    project_dir = sessions_root / project_key_for_cwd(cwd)
    stale = project_dir / "old-session.jsonl"
    stale.write_text("", encoding="utf-8")
    old = (NOW - timedelta(days=3)).timestamp()
    os.utime(stale, (old, old))

    assert session_candidates(project_key_for_cwd(cwd), sessions_root, now=NOW) == ()

    # Positive control: the same file inside the window is a candidate, so
    # the exclusion is the window and not the scan failing to see the file.
    fresh = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(stale, (fresh, fresh))
    assert session_candidates(project_key_for_cwd(cwd), sessions_root, now=NOW) == ("old-session",)


def test_an_aware_clock_and_a_naive_local_one_pick_the_same_candidates(project):
    """The window is compared against mtimes, which are absolute epoch seconds.

    `datetime.timestamp()` reads an aware value exactly and a naive one as
    *local* time, so the same wall-clock instant expressed both ways must land
    on the same cutoff. It does today only because `NOW` is naive-local; the
    rest of the tree (`collect_status`) is aware-UTC, and 5.3 wires a live app
    to this. A caller switching to the aware clock the rest of the tree uses
    must not silently shift the window by the UTC offset.
    """
    cwd, sessions_root = project
    key = project_key_for_cwd(cwd)
    store = sessions_root / key / "aware-session.jsonl"
    store.write_text("", encoding="utf-8")
    inside = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(store, (inside, inside))

    naive_local = NOW
    aware = datetime.fromtimestamp(NOW.timestamp(), tz=timezone.utc)
    assert aware.tzinfo is not None and naive_local.tzinfo is None
    assert aware.timestamp() == naive_local.timestamp()

    assert session_candidates(key, sessions_root, now=aware) == ("aware-session",)
    assert session_candidates(key, sessions_root, now=aware) == session_candidates(
        key, sessions_root, now=naive_local
    )

    # Positive control: the agreement is not both clocks being uselessly
    # permissive. Just outside the window, both must also agree on nothing.
    outside = (NOW - timedelta(days=3)).timestamp()
    os.utime(store, (outside, outside))
    assert session_candidates(key, sessions_root, now=aware) == ()
    assert session_candidates(key, sessions_root, now=naive_local) == ()


def test_the_default_clock_is_an_absolute_instant_not_a_naive_utc_reading(project):
    """`join_pane`'s `now` default has to agree with `st_mtime`'s epoch.

    A default of `datetime.now(timezone.utc)` is correct and so is a naive
    `datetime.now()`; a naive *UTC* clock is not, because `.timestamp()` would
    then re-interpret it as local and move the cutoff by the UTC offset. West
    of UTC that pushes the cutoff into the future and a store written this
    second stops being a candidate, which is what this asserts against.
    """
    cwd, sessions_root = project
    store = sessions_root / project_key_for_cwd(cwd) / "right-now.jsonl"
    store.write_text("", encoding="utf-8")
    just_now = time.time() - 1.0
    os.utime(store, (just_now, just_now))

    join = _join(cwd=cwd, sessions_root=sessions_root, now=None)

    assert join is not None
    assert join.session_candidates == ("right-now",)
    assert join.session_key == "right-now"


# --- the registry: naming the live session the mtime window cannot ----------


def _registry(root: Path, pid: int, cwd: Path, session_id: str, overrides=None) -> Path:
    """Write one Claude Code live-process record, in the shape measured."""
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "pid": pid,
        "sessionId": session_id,
        "cwd": str(cwd),
        "startedAt": 1787091441961,
        "version": "2.1.226",
        "kind": "interactive",
    }
    record.update(overrides or {})
    path = root / f"{pid}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _two_recent_sessions(sessions_root: Path, cwd: Path) -> Path:
    """Leave a project directory in the state that refuses every candidate."""
    project_dir = sessions_root / project_key_for_cwd(cwd)
    project_dir.mkdir(parents=True, exist_ok=True)
    for name, age in (("aaaa-1111", 60), ("bbbb-2222", 30)):
        store = project_dir / f"{name}.jsonl"
        store.write_text("{}\n", encoding="utf-8")
        os.utime(store, (NOW.timestamp() - age, NOW.timestamp() - age))
    return project_dir


def test_the_registry_names_the_live_session_among_equally_recent_ones(project, tmp_path):
    """The measured failure: one project, several stores written this hour.

    Every one of them is inside the activity window, so the scan is right to
    refuse and the pane is left UNJOINED. The registry is the only thing on
    this machine that says which pid owns which session.
    """
    cwd, sessions_root = project
    _two_recent_sessions(sessions_root, cwd)
    registry = tmp_path / "registry"
    _registry(registry, AGENT_PID, cwd, "bbbb-2222")

    join = _join(cwd, sessions_root, registry_root=registry)

    assert join is not None
    assert join.session_key == "bbbb-2222", "did not use the pid's own session"
    assert join.session_candidates == ("bbbb-2222",)
    assert join.store_path == (
        sessions_root / project_key_for_cwd(cwd) / "bbbb-2222.jsonl"
    ).resolve(strict=False)


def test_without_a_registry_equally_recent_sessions_are_still_refused(project, tmp_path):
    """The fallback is unchanged, so an older Claude Code still fails closed."""
    cwd, sessions_root = project
    _two_recent_sessions(sessions_root, cwd)

    join = _join(cwd, sessions_root, registry_root=tmp_path / "absent")

    assert join is not None
    assert join.session_key is None
    assert join.session_candidates == ("aaaa-1111", "bbbb-2222")


def test_the_registry_finds_a_project_directory_spelled_with_dashes(tmp_path):
    """Claude Code changed how it encodes `_`, and both spellings are on disk.

    Measured on this machine: `...-CipherBlade-okx_case` kept its underscore
    and `...-fairaday-labs` did not. The encoder can only produce one of the
    two, which is why this pane never joined at all -- its project directory
    existed under a name the encoder never generates.
    """
    cwd = tmp_path / "Projects" / "fairaday_labs"
    cwd.mkdir(parents=True)
    sessions_root = tmp_path / "store"
    dashed = sessions_root / project_key_for_cwd(cwd).replace("_", "-")
    dashed.mkdir(parents=True)
    (dashed / "cccc-3333.jsonl").write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry"
    _registry(registry, AGENT_PID, cwd, "cccc-3333")

    join = _join(cwd, sessions_root, registry_root=registry)

    assert join is not None, "refused a project directory it could not spell"
    assert join.session_key == "cccc-3333"
    assert join.store_path == (dashed / "cccc-3333.jsonl").resolve(strict=False)
    assert join.project_key == dashed.name


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"cwd": "/somewhere/else"}, "a record for a different directory"),
        ({"pid": AGENT_PID + 1}, "a record claiming another pid"),
        ({"sessionId": "../bbbb-2222"}, "a session id that escapes its directory"),
        ({"sessionId": ""}, "an empty session id"),
        ({"sessionId": 17}, "a session id that is not a string"),
    ],
)
def test_a_registry_that_cannot_prove_itself_falls_back(project, tmp_path, overrides, reason):
    """A stale registry file from a reused pid must not become a wrong join.

    Falling back is the whole safety property: the pane returns to the
    ambiguous-but-honest state rather than adopting an unverified session.
    """
    cwd, sessions_root = project
    _two_recent_sessions(sessions_root, cwd)
    registry = tmp_path / "registry"
    _registry(registry, AGENT_PID, cwd, "bbbb-2222", overrides)

    join = _join(cwd, sessions_root, registry_root=registry)

    assert join is not None
    assert join.session_key is None, f"trusted {reason}"


def test_a_registry_naming_a_transcript_that_is_not_there_falls_back(project, tmp_path):
    """The record is only a name; the transcript it names must exist."""
    cwd, sessions_root = project
    _two_recent_sessions(sessions_root, cwd)
    registry = tmp_path / "registry"
    _registry(registry, AGENT_PID, cwd, "dddd-4444")

    join = _join(cwd, sessions_root, registry_root=registry)

    assert join is not None
    assert join.session_key is None


def test_an_unreadable_registry_record_falls_back(project, tmp_path):
    """Malformed JSON is a version change, not a reason to fail the pane."""
    cwd, sessions_root = project
    _two_recent_sessions(sessions_root, cwd)
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / f"{AGENT_PID}.json").write_text("not json at all", encoding="utf-8")

    join = _join(cwd, sessions_root, registry_root=registry)

    assert join is not None
    assert join.session_key is None
