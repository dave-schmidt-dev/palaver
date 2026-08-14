"""Tests for `palaver status` and `palaver inspect` (task 1.9).

`palaver status` collapses each discovered session to one line: project,
session id, status, age. `palaver inspect <session>` refuses that collapse
for one session at a time, printing every signal `Signals` defines,
three-valued, plus the turn-boundary basis and corroboration behind
`agent_turn_ended` — the point being that a wrong status line from `status`
is diagnosable from `inspect`'s output alone, without a debugger.

What this module defends, test by test:

* The status line's four fields are in the documented order and are real
  (not constants) — proven by a fixture with two sessions in two different
  states printing two different status words.
* `--once` is Phase 1's only mode; omitting it is a usage error, not a
  silent no-op and not a silent single pass.
* `discover_sessions`'s floor-not-filter window (task 1.3) is exercised
  through the command, not just the adapter directly: an old-but-unresolved
  session surfaces, an old-and-resolved one does not.
* The byte-identical-output acceptance criterion runs against a copy of the
  real `tests/fixtures/` corpus laid out in `tmp_path` in the shape
  `ClaudeCodeAdapter` requires — never the live `~/.claude/` store, and
  nothing derived from it is written back to the repository.
* `palaver inspect` enumerates `SIGNAL_NAMES` and `BASIS_NAMES` programmatically
  (imported from `palaver.observer.signals`/`turn_boundary`, never
  hardcoded), including the case where a signal's value is `Tri.UNKNOWN` —
  absence is exactly what this command must show, not hide.
* Session lookup: an unresolvable id is a diagnosable error (stderr, exit 1),
  and an ambiguous bare id must fail rather than silently pick one — proven
  against a positive control where the fully-qualified `session_key`
  resolves the identical lookup unambiguously.
* The installed console script wires both commands end to end, with progress
  on stderr and results on stdout (INV-1), and `--help` lists both.

No real session store (`~/.claude/`, `~/.codex/`, `~/.local/share/opencode/`)
is opened, globbed, or read by this module — every sample directory here is
built under pytest's `tmp_path`, either from fixtures invented in this file
or copied byte-for-byte from the committed, hand-authored corpus in
`tests/fixtures/` (INV-3).
"""

import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from palaver.cli import inspect as inspect_cli
from palaver.cli import status as status_cli
from palaver.cli.inspect import SessionLookupError
from palaver.observer.signals import SIGNAL_NAMES
from palaver.observer.turn_boundary import BASIS_NAMES

#: Fixed reference time; every fixture's mtime is set relative to it, so no
#: assertion in this module depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


# --- fixture builders --------------------------------------------------------


def _line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write(path: Path, items: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_line(item) for item in items))
    return path


def _session(tmp_path: Path, project: str, name: str, items: list[dict]) -> Path:
    """Write one session store into a Claude-Code-shaped sample directory."""
    return _write(tmp_path / "projects" / project / f"{name}.jsonl", items)


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def _human(text: str = "please check the deploy") -> dict:
    return {
        "type": "user",
        "sessionId": "session-1",
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant(text: str = "the deploy looks clean") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _tool_use(name: str = "Bash") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu-1", "name": name, "input": {}}],
        },
    }


def _bookkeeping() -> dict:
    return {"type": "mode", "sessionId": "session-1", "mode": "default"}


def _frozen_sample(tmp_path: Path) -> Path:
    """Copy the committed `tests/fixtures/` corpus into an adapter-shaped sample root.

    `ClaudeCodeAdapter.list_store_paths()` requires exactly one project
    directory level (`<root>/<project>/<session>.jsonl`); the committed
    fixtures are flat (`tests/fixtures/*.jsonl`, the same shape
    `test_fixture_lint.py` reads them in directly). This copies their bytes
    verbatim into that shape under `tmp_path` — reading only from the
    repository's frozen corpus, writing only under `tmp_path` — so the
    byte-identical-output test below runs against a fixed, committed sample
    and never against the real `~/.claude/` store. Every copy's mtime is
    pinned a few minutes behind `NOW`, so discovery does not depend on when
    the suite actually runs.
    """
    project_dir = tmp_path / "projects" / "fixture-corpus"
    project_dir.mkdir(parents=True)
    for fixture in sorted(FIXTURES_DIR.glob("*.jsonl")):
        copy = project_dir / fixture.name
        copy.write_bytes(fixture.read_bytes())
        _set_mtime(copy, timedelta(minutes=5))
    return tmp_path / "projects"


# --- palaver status: line format ---------------------------------------------


def test_status_line_reports_project_session_status_and_age(tmp_path):
    """One line per session, in the documented field order.

    Two sessions in different states must print two different status words —
    the positive control that proves the status column is a real derivation
    and not a constant a stub could satisfy.
    """
    working = _session(tmp_path, "proj-a", "session-a-working", [_human(), _tool_use()])
    _set_mtime(working, timedelta(seconds=30))
    waiting = _session(tmp_path, "proj-a", "session-b-waiting", [_human(), _assistant()])
    _set_mtime(waiting, timedelta(minutes=10))

    args = SimpleNamespace(once=True, sample=tmp_path / "projects")
    out = io.StringIO()
    exit_code = status_cli.run(args, out=out, now=NOW)

    assert exit_code == 0
    assert out.getvalue() == (
        "proj-a session-a-working WORKING 30s\nproj-a session-b-waiting AWAITING_HUMAN 10m\n"
    )


def test_status_without_once_is_a_usage_error_not_a_silent_default(tmp_path, capsys):
    """`--once` is Phase 1's only mode (there is no watch mode until task
    4.1). Omitting it must be a visible usage error, not a silent no-op and
    not a silent single pass that would later collide with a real flag."""
    args = SimpleNamespace(once=False, sample=tmp_path / "projects")

    exit_code = status_cli.run(args, now=NOW)

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "nothing to do" in captured.err


def test_status_respects_the_discovery_floor_not_filter_window(tmp_path):
    """Same age, different last record: an old session with an unresolved
    trailing `tool_use` still surfaces (the always-include rule), while an
    equally old, resolved session ages out. Proves `status` exercises the
    real `discover_sessions` window rather than a recency cutoff of its own.
    """
    stale_unresolved = _session(tmp_path, "proj-a", "stale-unresolved", [_human(), _tool_use()])
    _set_mtime(stale_unresolved, timedelta(days=3))
    stale_resolved = _session(tmp_path, "proj-a", "stale-resolved", [_human(), _assistant()])
    _set_mtime(stale_resolved, timedelta(days=3))

    args = SimpleNamespace(once=True, sample=tmp_path / "projects")
    out = io.StringIO()
    status_cli.run(args, out=out, now=NOW)

    session_ids = [line.split()[1] for line in out.getvalue().splitlines()]
    assert "stale-unresolved" in session_ids
    assert "stale-resolved" not in session_ids


def test_status_age_reflects_store_mtime_not_wall_clock(tmp_path):
    """Age is `now - store mtime`, computed from the fixed `now` passed to
    `run()`. Two sessions with different mtimes must print different ages
    under the same fixed reference time — if age silently read the real
    clock instead, both would print the same (wrong) value every run."""
    recent = _session(tmp_path, "proj-a", "recent", [_human(), _assistant()])
    _set_mtime(recent, timedelta(seconds=5))
    old = _session(tmp_path, "proj-a", "old", [_human(), _assistant()])
    _set_mtime(old, timedelta(hours=2))

    args = SimpleNamespace(once=True, sample=tmp_path / "projects")
    out = io.StringIO()
    status_cli.run(args, out=out, now=NOW)

    ages = {line.split()[1]: line.split()[3] for line in out.getvalue().splitlines()}
    assert ages["recent"] == "5s"
    assert ages["old"] == "2h"


# --- palaver status: determinism against the frozen fixture corpus ----------


def test_status_against_the_frozen_fixture_store_is_byte_identical_across_runs(tmp_path):
    """The acceptance criterion: run twice against a fixed sample and a
    fixed `now`, byte-identical stdout both times. Runs against a copy of
    the committed `tests/fixtures/` corpus, never the live store, or this
    would be flaky by construction. The non-empty and line-count assertions
    rule out two empty strings comparing equal for the wrong reason."""
    sample = _frozen_sample(tmp_path)
    args = SimpleNamespace(once=True, sample=sample)
    fixture_count = len(list(FIXTURES_DIR.glob("*.jsonl")))

    first = io.StringIO()
    second = io.StringIO()
    status_cli.run(args, out=first, now=NOW)
    status_cli.run(args, out=second, now=NOW)

    assert first.getvalue() == second.getvalue()
    assert first.getvalue() != ""
    assert len(first.getvalue().splitlines()) == fixture_count == 11


# --- palaver inspect: signal enumeration and diagnostics ---------------------


def test_inspect_prints_every_signal_name_including_absent_ones(tmp_path):
    """Enumerated from the real `SIGNAL_NAMES` (imported, never hardcoded),
    so a signal added to `Signals` in a later phase appears here without
    this test changing. The bookkeeping-only fixture guarantees at least one
    signal comes back `Tri.UNKNOWN` — the absent case this command exists to
    surface, not omit."""
    _session(tmp_path, "proj-a", "bookkeeping-only", [_bookkeeping()])
    args = SimpleNamespace(session="bookkeeping-only", sample=tmp_path / "projects")
    out = io.StringIO()

    exit_code = inspect_cli.run(args, out=out, now=NOW)
    text = out.getvalue()

    assert exit_code == 0
    # Non-vacuity: there is a real, non-trivial set of names to enumerate.
    assert len(SIGNAL_NAMES) >= 4
    for name in SIGNAL_NAMES:
        assert f"{name}: " in text
    assert "agent_turn_ended: unknown" in text


def test_inspect_basis_is_always_a_member_of_basis_names(tmp_path):
    """The reported turn-boundary basis is checked against the real
    `BASIS_NAMES` tuple (imported, never hardcoded), across three sessions
    that reach it by three different structural routes — a basis added
    later still has to pass here without this test changing."""
    # Non-vacuity: there is a real, non-trivial set of bases to check against.
    assert len(BASIS_NAMES) >= 4

    cases = {
        "final": [_human(), _assistant()],
        "midcall": [_human(), _tool_use()],
        "question": [_human(), _tool_use("AskUserQuestion")],
    }
    for name, items in cases.items():
        _session(tmp_path, "proj-a", name, items)

    seen_bases = set()
    for name in cases:
        args = SimpleNamespace(session=name, sample=tmp_path / "projects")
        out = io.StringIO()
        inspect_cli.run(args, out=out, now=NOW)
        basis_line = next(
            line for line in out.getvalue().splitlines() if line.strip().startswith("basis:")
        )
        basis = basis_line.split("basis:", 1)[1].strip()
        assert basis in BASIS_NAMES
        seen_bases.add(basis)

    # The three cases really do reach different bases, not one basis three times.
    assert len(seen_bases) == 3


def test_inspect_unknown_session_is_a_diagnosable_error_not_a_crash(tmp_path, capsys):
    """A session id that resolves to nothing exits 1 with the id named on
    stderr, and nothing on stdout — diagnosable from the CLI's own output,
    no traceback required."""
    _session(tmp_path, "proj-a", "real-session", [_human(), _assistant()])
    args = SimpleNamespace(session="does-not-exist", sample=tmp_path / "projects")

    exit_code = inspect_cli.run(args, now=NOW)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "does-not-exist" in captured.err


def test_inspect_ambiguous_bare_id_requires_the_qualified_session_key(tmp_path):
    """Two projects each name a session `shared`: the bare id is ambiguous
    and must fail rather than silently resolving to whichever one sorts
    first. The fully-qualified `session_key` resolves the identical lookup
    unambiguously — the positive control proving the failure is about the
    bare id, not about the fixtures being unresolvable in general."""
    _session(tmp_path, "proj-a", "shared", [_human(), _assistant()])
    _session(tmp_path, "proj-b", "shared", [_human(), _tool_use()])
    sample = tmp_path / "projects"

    with pytest.raises(SessionLookupError):
        inspect_cli.resolve_session(sample, "shared")

    ref = inspect_cli.resolve_session(sample, "proj-a/shared")
    assert ref.session_key == "proj-a/shared"


# --- console script: end to end, INV-1's two streams -------------------------


def test_console_script_runs_status_with_progress_on_stderr(tmp_path):
    """End to end through the installed `palaver` console script: exits 0,
    one status line per session on stdout, per-session progress on stderr
    (INV-1) — a scan is never a silent wait, and stdout stays pipeable."""
    sample = _frozen_sample(tmp_path)
    script = Path(sys.executable).parent / "palaver"
    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run(
        [str(script), "status", "--once", "--sample", str(sample)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 11
    assert "observing 11/11" in result.stderr
    assert "observing" not in result.stdout


def test_console_script_runs_inspect_with_progress_on_stderr(tmp_path):
    """Same end-to-end shape for `inspect`: exits 0, the full signal set on
    stdout, progress on stderr."""
    _session(tmp_path, "proj-a", "session-x", [_human(), _assistant()])
    script = Path(sys.executable).parent / "palaver"
    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run(
        [str(script), "inspect", "proj-a/session-x", "--sample", str(tmp_path / "projects")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "session: proj-a/session-x" in result.stdout
    assert "observing proj-a/session-x" in result.stderr
    assert "observing" not in result.stdout


def test_console_script_help_lists_status_and_inspect(tmp_path):
    """`palaver --help` exits 0 and both subcommands task 1.9 adds are
    listed in the subcommand table."""
    script = Path(sys.executable).parent / "palaver"
    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0
    assert "status" in result.stdout
    assert "inspect" in result.stdout
