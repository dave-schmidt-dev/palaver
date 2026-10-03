"""
Regression tests for Codex alternate CODEX_HOME discovery in companion joining.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from palaver.ui.pane_join import (
    CODEX_SOURCE,
    PanePin,
    PaneVariables,
    ProcessInfo,
    join_pane,
)
from tests._pane_join_support import NOW, _codex_rollout, _table


def test_codex_alternate_lock_home_discovers_sessions_root_and_pins(tmp_path):
    """A live process holding an alternate CODEX_HOME lock discovers that session root."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    alt_home = tmp_path / "codex-secondary"
    alt_sessions = alt_home / "sessions"
    store = _codex_rollout(alt_sessions, cwd, "rollout-alt")
    lock = alt_home / "tmp" / "arg0" / "codex-arg0123" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert joined is not None
    assert joined.source == CODEX_SOURCE
    assert joined.session_key == "rollout-alt"
    assert joined.store_path == store.resolve()

    pinned = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        pin=PanePin(CODEX_SOURCE, "rollout-alt"),
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert pinned is not None
    assert pinned.session_key == "rollout-alt"
    assert pinned.store_path == store.resolve()


def test_codex_canonical_rollout_open_path_discovers_home(tmp_path):
    """An open canonical rollout path can also identify an alternate Codex home."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    alt_home = tmp_path / "codex-rollout-home"
    alt_sessions = alt_home / "sessions"
    store = _codex_rollout(alt_sessions, cwd, "rollout-canonical")

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset({store.resolve()}),
    )
    assert joined is not None
    assert joined.session_key == "rollout-canonical"
    assert joined.store_path == store.resolve()


def test_codex_simultaneous_primary_and_secondary_cwd_does_not_cross_join(tmp_path):
    """Two agents running concurrently in the same cwd join only their own home's rollout."""
    cwd = tmp_path / "shared-project"
    cwd.mkdir()
    primary_home = tmp_path / "codex-primary"
    secondary_home = tmp_path / "codex-secondary"

    primary_store = _codex_rollout(primary_home / "sessions", cwd, "rollout-primary")
    secondary_store = _codex_rollout(secondary_home / "sessions", cwd, "rollout-secondary")

    primary_lock = primary_home / "tmp" / "arg0" / "codex-arg0111" / ".lock"
    secondary_lock = secondary_home / "tmp" / "arg0" / "codex-arg0222" / ".lock"

    table = _table(((101, 1, "codex"), (202, 1, "codex")))
    open_map = {101: frozenset({primary_lock}), 202: frozenset({secondary_lock})}

    pane_primary = PaneVariables("pane-primary", 101, "codex", str(cwd))
    pane_secondary = PaneVariables("pane-secondary", 202, "codex", str(cwd))

    joined_p = join_pane(
        pane_primary,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda pid: open_map[pid],
    )
    assert joined_p is not None
    assert joined_p.session_key == "rollout-primary"
    assert joined_p.store_path == primary_store.resolve()

    joined_s = join_pane(
        pane_secondary,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda pid: open_map[pid],
    )
    assert joined_s is not None
    assert joined_s.session_key == "rollout-secondary"
    assert joined_s.store_path == secondary_store.resolve()


def test_codex_conflicting_homes_refuse_joining(tmp_path):
    """Open paths pointing to multiple distinct homes must refuse joining."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    home_a = tmp_path / "codex-a"
    home_b = tmp_path / "codex-b"

    _codex_rollout(home_a / "sessions", cwd, "rollout-a")
    _codex_rollout(home_b / "sessions", cwd, "rollout-b")

    lock_a = home_a / "tmp" / "arg0" / "codex-arg0111" / ".lock"
    lock_b = home_b / "tmp" / "arg0" / "codex-arg0222" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset({lock_a, lock_b}),
    )
    assert joined is None


def test_codex_stale_or_reused_pid_refuses_joining(tmp_path):
    """A reused or stale process identity refuses descriptor-derived home discovery."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    alt_home = tmp_path / "codex-secondary"
    _codex_rollout(alt_home / "sessions", cwd, "rollout-alt")
    lock = alt_home / "tmp" / "arg0" / "codex-arg0123" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    reused = dict(table)
    old = reused[77201]
    reused[77201] = ProcessInfo(
        pid=old.pid,
        ppid=old.ppid,
        name=old.name,
        command=old.command,
        start_time="Fri Aug 15 12:05:00 2026",
    )

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        process_table_reader=lambda: reused,
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert joined is None


def test_codex_default_home_refuses_fallback_when_open_files_lookup_fails(tmp_path, monkeypatch):
    """An unreadable process descriptor scan must not fall back across Codex homes."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    primary_sessions = tmp_path / "codex-primary" / "sessions"
    _codex_rollout(primary_sessions, cwd, "rollout-primary")
    monkeypatch.setattr(
        "palaver.ui.pane_join.stores.default_store_roots",
        lambda: {CODEX_SOURCE: primary_sessions},
    )

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    def failed_open_files(_pid: int) -> frozenset[Path]:
        raise OSError("descriptor scan failed")

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=failed_open_files,
    )
    assert joined is None


@pytest.mark.parametrize("post_read_state", ["reused", "absent"])
def test_codex_home_discovery_revalidates_pid_after_open_files_read(tmp_path, post_read_state):
    """A pid that changes or exits during lsof cannot supply a Codex home."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    alternate_home = tmp_path / "codex-secondary"
    _codex_rollout(alternate_home / "sessions", cwd, "rollout-alt")
    lock = alternate_home / "tmp" / "arg0" / "codex-arg0123" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    if post_read_state == "reused":
        post_read_table = dict(table)
        old = post_read_table[77201]
        post_read_table[77201] = ProcessInfo(
            pid=old.pid,
            ppid=old.ppid,
            name=old.name,
            command=old.command,
            start_time="Fri Aug 15 12:05:00 2026",
        )
    else:
        post_read_table = {pid: row for pid, row in table.items() if pid != 77201}

    snapshots = iter((table, post_read_table))
    events: list[str] = []

    def read_process_table():
        events.append("process_table")
        return next(snapshots)

    def read_open_files(_pid: int) -> frozenset[Path]:
        events.append("open_files")
        return frozenset({lock})

    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))
    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        process_table_reader=read_process_table,
        open_files_reader=read_open_files,
    )

    assert joined is None
    assert events == ["process_table", "open_files", "process_table"]


def test_codex_explicit_roots_precedence_and_omission(tmp_path):
    """Explicit store_roots or sessions_root override descriptor-derived home discovery."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    explicit_home = tmp_path / "codex-explicit"
    alt_home = tmp_path / "codex-alt"

    explicit_store = _codex_rollout(explicit_home / "sessions", cwd, "rollout-explicit")
    _codex_rollout(alt_home / "sessions", cwd, "rollout-alt")
    lock = alt_home / "tmp" / "arg0" / "codex-arg0123" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={CODEX_SOURCE: explicit_home / "sessions"},
        now=NOW,
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert joined is not None
    assert joined.session_key == "rollout-explicit"
    assert joined.store_path == explicit_store.resolve()

    joined_legacy = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        sessions_root=explicit_home / "sessions",
        now=NOW,
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert joined_legacy is not None
    assert joined_legacy.session_key == "rollout-explicit"
    assert joined_legacy.store_path == explicit_store.resolve()

    omitted = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        store_roots={"claude-code": tmp_path / "claude"},
        now=NOW,
        open_files_reader=lambda _pid: frozenset({lock}),
    )
    assert omitted is None


def test_codex_absence_of_lsof_evidence_preserves_fallback(tmp_path, monkeypatch):
    """When no open paths provide home evidence, the default fallback root is used."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    fallback_sessions = tmp_path / "fallback-codex-sessions"
    monkeypatch.setattr(
        "palaver.ui.pane_join.stores.default_store_roots",
        lambda: {CODEX_SOURCE: fallback_sessions},
    )
    store = _codex_rollout(fallback_sessions, cwd, "rollout-fallback")

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset(),
    )
    assert joined is not None
    assert joined.session_key == "rollout-fallback"
    assert joined.store_path == store.resolve()

    unrelated = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset({Path("/usr/lib/libSystem.B.dylib")}),
    )
    assert unrelated is not None
    assert unrelated.session_key == "rollout-fallback"
    assert unrelated.store_path == store.resolve()


def test_codex_malformed_lock_paths_excluded(tmp_path, monkeypatch):
    """Malformed lock paths do not infer homes and preserve fallback behavior."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    fallback_sessions = tmp_path / "fallback-codex-sessions"
    monkeypatch.setattr(
        "palaver.ui.pane_join.stores.default_store_roots",
        lambda: {CODEX_SOURCE: fallback_sessions},
    )
    store = _codex_rollout(fallback_sessions, cwd, "rollout-fallback")

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    malformed_paths = (
        Path("/tmp/arg0/codex-arg0123/.lock"),
        Path("/Users/user/alt/tmp/arg0/wrong-prefix/.lock"),
        Path("/Users/user/alt/tmp/arg1/codex-arg0123/.lock"),
        Path("/Users/user/alt/cache/arg0/codex-arg0123/.lock"),
        Path("/Users/user/alt/tmp/arg0/codex-arg0123/other.txt"),
        Path("relative/tmp/arg0/codex-arg0123/.lock"),
    )

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=lambda _pid: frozenset(malformed_paths),
    )
    assert joined is not None
    assert joined.session_key == "rollout-fallback"
    assert joined.store_path == store.resolve()


def test_codex_open_files_reader_called_once_per_join(tmp_path):
    """Open-path reuse prevents duplicate lsof invocations in a single join."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    alt_home = tmp_path / "codex-secondary"
    alt_sessions = alt_home / "sessions"

    live = _codex_rollout(alt_sessions, cwd, "rollout-live")
    _codex_rollout(alt_sessions, cwd, "rollout-exited")
    lock = alt_home / "tmp" / "arg0" / "codex-arg0123" / ".lock"

    table = _table(((77201, 62921, "codex"), (62921, 1, "-zsh")))
    variables = PaneVariables("pane-1", 77201, "codex", str(cwd))

    calls: list[int] = []

    def counted_reader(pid: int) -> frozenset[Path]:
        calls.append(pid)
        return frozenset({lock, live.resolve()})

    joined = join_pane(
        variables,
        table=table,
        cwd_reader=lambda _pid: cwd,
        now=NOW,
        open_files_reader=counted_reader,
    )
    assert joined is not None
    assert joined.session_key == "rollout-live"
    assert len(calls) == 1, f"open_files_reader should be called once, was called {len(calls)}"
