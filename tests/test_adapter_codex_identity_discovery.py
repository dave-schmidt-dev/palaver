"""
CodexIdentity from session_meta, subagent detection, session keys, and store discovery.
"""

from pathlib import Path

from palaver.ingest.adapters.codex import (
    CodexAdapter,
)
from tests._adapter_codex_support import _message, _session_meta, _write_rollout

# --- identity ---------------------------------------------------------------


def test_identity_is_read_from_session_meta(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-07-00-fixture.jsonl",
        [
            _session_meta(
                id="fixture-thread-child",
                session_id="fixture-thread-root",
                parent_thread_id="fixture-thread-root",
            ),
            _message("user", "check the staging deploy status"),
        ],
    )
    adapter = CodexAdapter(root=root)
    identity = adapter.read_identity(path)
    assert identity is not None
    assert identity.cwd == "/tmp/fixture-codex-project"
    assert identity.id == "fixture-thread-child"
    assert identity.session_id == "fixture-thread-root"
    assert identity.parent_thread_id == "fixture-thread-root"
    assert identity.is_subagent is True
    assert adapter.project_key_for(path) == "/tmp/fixture-codex-project"


def test_a_root_session_is_not_a_subagent(tmp_path):
    """Positive control for `is_subagent`: it can be False."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-08-00-fixture.jsonl", [_session_meta()])
    identity = CodexAdapter(root=root).read_identity(path)
    assert identity is not None
    assert identity.is_subagent is False


def test_a_differing_session_id_alone_marks_a_subagent(tmp_path):
    """`parent_thread_id` is not the only linkage; `.id != .session_id` is too."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-09-00-fixture.jsonl",
        [_session_meta(id="fixture-thread-child", session_id="fixture-thread-root")],
    )
    identity = CodexAdapter(root=root).read_identity(path)
    assert identity is not None
    assert identity.parent_thread_id is None
    assert identity.is_subagent is True


def test_identity_is_none_when_the_file_has_no_session_meta(tmp_path):
    """A truncated or not-yet-flushed rollout is a legitimate state, not a crash."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-10-00-fixture.jsonl",
        [_message("user", "check the staging deploy status")],
    )
    adapter = CodexAdapter(root=root)
    assert adapter.read_identity(path) is None
    assert adapter.project_key_for(path) is None


def test_session_key_is_derived_from_the_path_without_opening_the_file(tmp_path):
    """`discover_sessions` calls this for paths it will never open."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-11-00-fixture.jsonl", [_session_meta()])
    adapter = CodexAdapter(root=root)
    assert adapter.session_key_for(path) == "rollout-2026-08-14T10-11-00-fixture"
    assert adapter.session_key_for(Path("/nonexistent/rollout-x.jsonl")) == "rollout-x"


# --- discovery --------------------------------------------------------------


def test_store_discovery_is_recursive_and_prefix_scoped(tmp_path):
    """Codex partitions by `YYYY/MM/DD`, so the glob has to be recursive.

    The `rollout-` prefix keeps any other `.jsonl` in that tree — a cache, a
    sidecar a future release drops in — from being mistaken for a session
    store and handed a bogus session key.
    """
    root = tmp_path / "sessions"
    _write_rollout(root, "rollout-2026-08-14T10-12-00-a.jsonl", [_session_meta()])
    deep = root / "2026" / "08" / "15"
    deep.mkdir(parents=True)
    (deep / "rollout-2026-08-15T09-00-00-b.jsonl").write_text("", encoding="utf-8")
    (deep / "not-a-rollout.jsonl").write_text("", encoding="utf-8")

    paths = list(CodexAdapter(root=root).list_store_paths())
    names = [path.name for path in paths]
    assert names == [
        "rollout-2026-08-14T10-12-00-a.jsonl",
        "rollout-2026-08-15T09-00-00-b.jsonl",
    ]


def test_discovery_of_a_missing_root_is_empty_not_an_error(tmp_path):
    assert list(CodexAdapter(root=tmp_path / "absent").list_store_paths()) == []
