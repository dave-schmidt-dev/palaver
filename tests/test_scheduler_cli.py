"""
The observe CLI: default paths, adapter selection, INV-1's status channel, dry-run, and
the tick loop.
"""

import io
from pathlib import Path
from types import SimpleNamespace

from palaver.cli import observe as observe_cli
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import Cursor, CursorStore
from tests._test_scheduler_support import NOW, RecordingExtractor, _daemon, _human, _write_store


def _cli_args(tmp_path: Path, sample_root: Path, **overrides) -> SimpleNamespace:
    args = {
        "once": True,
        "dry_run": False,
        "interval": 0.0,
        "max_ticks": None,
        "db": tmp_path / "store" / "palaver.db",
        "cursors": tmp_path / "cursors",
        "sample": sample_root,
        "all": True,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_observe_defaults_use_the_project_local_durable_state_directory():
    """The daemon's accumulated memory must not disappear with `/tmp`."""
    expected = Path.cwd() / ".state" / "observer"
    assert observe_cli.DEFAULT_DB_PATH == expected / "observe.db"
    assert observe_cli.DEFAULT_CURSOR_ROOT == expected / "cursors"


def test_observe_default_and_explicit_roots_select_only_fixture_sources(monkeypatch, tmp_path):
    """Default construction names both sources; one fixture root scopes to one."""
    constructed = []

    class FakeAdapter:
        def __init__(self, *, root=None):
            constructed.append((type(self).source, root))

    class FakeClaude(FakeAdapter):
        source = "claude-code"

    class FakeCodex(FakeAdapter):
        source = "codex"

    monkeypatch.setattr(observe_cli, "ClaudeCodeAdapter", FakeClaude)
    monkeypatch.setattr(observe_cli, "CodexAdapter", FakeCodex)

    assert [adapter.source for adapter in observe_cli._configured_adapters(SimpleNamespace())] == [
        "claude-code",
        "codex",
    ]
    assert constructed == [("claude-code", None), ("codex", None)]

    constructed.clear()
    claude_root = tmp_path / "claude-fixture"
    codex_root = tmp_path / "codex-fixture"
    assert [
        adapter.source
        for adapter in observe_cli._configured_adapters(
            SimpleNamespace(sample=claude_root, codex_root=None)
        )
    ] == ["claude-code"]
    assert [
        adapter.source
        for adapter in observe_cli._configured_adapters(
            SimpleNamespace(sample=None, codex_root=codex_root)
        )
    ] == ["codex"]
    assert constructed == [("claude-code", claude_root), ("codex", codex_root)]


# --- INV-1: the status channel -----------------------------------------------


def test_observer_tick_emits_status(tmp_path, capsys):
    """One tick emits progress, and the default channel keeps stdout clean.

    Split deliberately. The first half injects a recorder, which proves the
    channel is *called* — but makes "nothing on stdout" trivially true,
    since the real channel never ran. The second half goes through the CLI
    with the default channel and asserts stderr carries the progress while
    stdout carries only the result line. That pairing is INV-1's gate; the
    recorder alone is not.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    messages: list[str] = []
    with _daemon(tmp_path, sample_root, RecordingExtractor(), on_status=messages.append) as daemon:
        daemon.tick(now=NOW)
    assert messages, "a tick emitted no status update at all"
    # Both assertions name a *production* emitter. "session-1 appears
    # somewhere" would also be satisfied by `RecordingExtractor`'s own
    # status line, which is this test's fixture, not the daemon's behavior.
    assert any("tailing" in message and "session-1" in message for message in messages), (
        "the scheduler emitted no per-session progress"
    )
    assert any(message.startswith("tick 1:") for message in messages), (
        "the daemon emitted no tick-level progress"
    )

    # The default channel, through the CLI. Cursors are pre-warmed so the
    # tick schedules nothing: this test must never reach for a model server.
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cli-cursors")
    cursors.save(adapter.session_key_for(path), adapter.tail(path, Cursor()).cursor)

    capsys.readouterr()  # discard anything the daemon half emitted
    out = io.StringIO()
    args = _cli_args(tmp_path, sample_root, cursors=tmp_path / "cli-cursors")
    exit_code = observe_cli.run(args, out=out, now=NOW)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err != "", "the default status channel emitted nothing"
    assert captured.out == "", "the status channel wrote to stdout"
    assert out.getvalue().startswith("tick 1:")
    assert "changed=0 extracted=0 failed=0 deferred=0" in out.getvalue()


# --- the CLI -----------------------------------------------------------------


def test_dry_run_reports_the_plan_without_saving_a_cursor(tmp_path, capsys):
    """`--dry-run` answers "what would this tick extract" and changes nothing."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    cursors = CursorStore(tmp_path / "cursors")

    out = io.StringIO()
    exit_code = observe_cli.run(_cli_args(tmp_path, sample_root, dry_run=True), out=out, now=NOW)
    capsys.readouterr()

    assert exit_code == 0
    assert "dry-run: discovered=1 changed=1 unchanged=0" in out.getvalue()
    assert key in out.getvalue()
    assert cursors.load(key).offset == 0
    assert not (tmp_path / "store" / "palaver.db").exists()


def test_once_runs_exactly_one_tick(tmp_path, capsys):
    """`--once` is one tick, not a loop with a short interval."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    out = io.StringIO()
    exit_code = observe_cli.run(
        _cli_args(tmp_path, sample_root, cursors=tmp_path / "warm"),
        out=out,
        now=NOW,
    )
    capsys.readouterr()

    assert exit_code == 0
    assert out.getvalue().count("tick ") == 1


def test_run_sleeps_between_ticks_and_never_after_the_last(tmp_path):
    """A bounded run returns as soon as its work is done, not one interval later."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    naps: list[float] = []
    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        results = daemon.run(interval=7.0, max_ticks=3, sleep=naps.append, now=NOW)

    assert len(results) == 3
    assert naps == [7.0, 7.0]


def test_run_stops_when_asked(tmp_path):
    """`stop()` ends the loop without needing `max_ticks`."""
    sample_root = tmp_path / "projects"
    _write_store(sample_root, "proj", "session-1", [_human()])
    ticks: list[int] = []

    def stop() -> bool:
        return len(ticks) >= 2

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        results = daemon.run(
            interval=0.0,
            stop=stop,
            sleep=lambda _: ticks.append(1),
            now=NOW,
        )

    assert len(results) == 2
