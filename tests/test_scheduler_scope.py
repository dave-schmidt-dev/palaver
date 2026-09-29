"""ensure_scope is idempotent across ticks and isolates equal external ids by source."""

from palaver.ingest.adapters.base import SessionRef
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.daemon import (
    ensure_scope,
)
from tests._test_scheduler_support import RecordingExtractor, _daemon, _human, _write_store

# --- scoping -----------------------------------------------------------------


def test_ensure_scope_is_idempotent_across_ticks(tmp_path):
    """Two ticks over one session produce one project row and one session row."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    adapter = ClaudeCodeAdapter(root=sample_root)
    (ref,) = adapter.discover_sessions(all=True)

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        first = ensure_scope(daemon.conn, ref)
        second = ensure_scope(daemon.conn, ref)
        projects = daemon.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        sessions = daemon.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    assert first == second
    assert projects == 1
    assert sessions == 1
    assert path.exists()


def test_ensure_scope_isolates_equal_external_ids_by_source(tmp_path):
    """Equal session keys from two adapters create distinct source rows."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "same-session", [_human()])
    (claude_ref,) = ClaudeCodeAdapter(root=sample_root).discover_sessions(all=True)
    codex_ref = SessionRef(
        source="codex",
        session_key=claude_ref.session_key,
        path=path,
        mtime=claude_ref.mtime,
        project=claude_ref.project,
    )

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        claude_scope = ensure_scope(daemon.conn, claude_ref)
        codex_scope = ensure_scope(daemon.conn, codex_ref)
        rows = daemon.conn.execute(
            "SELECT source, external_id FROM sessions ORDER BY source"
        ).fetchall()

    assert claude_scope[0] == codex_scope[0]
    assert claude_scope[1] != codex_scope[1]
    assert rows == [
        ("claude-code", claude_ref.session_key),
        ("codex", claude_ref.session_key),
    ]
