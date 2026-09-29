"""
`palaver diagnose --coverage`: the single-source percentage report and console-script
wiring.
"""

import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path

from palaver.cli import main
from palaver.cli.diagnose import (
    collect_coverage,
)
from palaver.observer.signals import (
    SIGNAL_NAMES,
    Tri,
)
from palaver.observer.turn_boundary import (
    observe_session,
)
from tests._turn_boundary_support import NOW, REPO_ROOT, _coverage_sample


def _coverage_rows(stdout: str) -> dict[str, tuple[str, str]]:
    """Parse the report's per-signal rows into {name: (counted, percentage)}."""
    rows = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in SIGNAL_NAMES:
            rows[parts[0]] = (parts[1], parts[2])
    return rows


def test_coverage_reports_a_percentage_for_every_signal(tmp_path, capsys):
    """The report emits one row per entry in `SIGNAL_NAMES` with the measured
    percentage, and the percentages are the real counts: four of the five
    sample sessions support a turn boundary, so `agent_turn_ended` is 80.0% —
    not the 100% a stub would print."""
    sample = _coverage_sample(tmp_path)

    exit_code = main(["diagnose", "--coverage", "--sample", str(sample)])
    stdout = capsys.readouterr().out
    rows = _coverage_rows(stdout)

    assert exit_code == 0
    assert set(rows) == set(SIGNAL_NAMES)
    assert len(rows) == len(SIGNAL_NAMES) == 4
    assert rows["source_readable"] == ("5/5", "100.0%")
    assert rows["signal_records_parsed"] == ("5/5", "100.0%")
    assert rows["unresolved_tool_error"] == ("5/5", "100.0%")
    assert rows["agent_turn_ended"] == ("4/5", "80.0%")
    assert "status: WORKING 1, AWAITING_HUMAN 2, ERROR 1, UNKNOWN 1" in stdout
    assert "coverage counts sessions a signal was determinable for" in stdout


def test_coverage_counts_match_an_independent_reading_of_the_same_sample(tmp_path):
    """The reported counts are the same ones `observe_session` produces
    session by session — computed here independently of the command, so a
    report that hardcoded its numbers fails."""
    sample = _coverage_sample(tmp_path)

    report = collect_coverage(sample, now=NOW)
    expected = dict.fromkeys(SIGNAL_NAMES, 0)
    paths = sorted(sample.glob("*/*.jsonl"))
    for path in paths:
        signals = observe_session(path, now=NOW).signals
        for name in SIGNAL_NAMES:
            if getattr(signals, name) is not Tri.UNKNOWN:
                expected[name] += 1

    assert len(paths) == 5
    assert report.sessions == 5
    assert report.determinable == expected
    assert report.percentage("agent_turn_ended") == 80.0


def test_coverage_over_an_empty_sample_is_not_reported_as_success(tmp_path, capsys):
    """A sample with no sessions exits non-zero instead of printing a vacuous
    100%. The populated control exits 0, so this is an empty-sample rule and
    not a command that always fails."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main(["diagnose", "--coverage", "--sample", str(empty)]) == 1
    assert "no sessions found" in capsys.readouterr().err

    assert main(["diagnose", "--coverage", "--sample", str(_coverage_sample(tmp_path))]) == 0


def test_console_script_runs_the_coverage_report_with_progress_on_stderr(tmp_path):
    """End-to-end through the installed `palaver` console script: it exits 0,
    the report goes to stdout, and per-session progress goes to stderr (INV-1
    — a scan over many stores is never a silent wait, and stdout stays the
    result channel so the report can be piped)."""
    sample = _coverage_sample(tmp_path)
    script = Path(sys.executable).parent / "palaver"

    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run(
        [str(script), "diagnose", "--coverage", "--sample", str(sample)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert _coverage_rows(result.stdout)["agent_turn_ended"] == ("4/5", "80.0%")
    assert "observing 5/5" in result.stderr
    assert "observing" not in result.stdout


def test_console_script_help_exits_zero_and_lists_diagnose(tmp_path):
    """`palaver --help` works, and the subcommand table is the CLI's extension
    point task 1.9 adds `status` and `inspect` to."""
    script = Path(sys.executable).parent / "palaver"

    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0
    assert "diagnose" in result.stdout


def test_pyproject_declares_the_console_script_at_an_importable_target():
    """The `palaver` script is declared in `pyproject.toml` and its target
    really resolves — a declaration pointing at a missing callable fails at
    install time for the user, not here."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["palaver"]
    module_name, _, attribute = target.partition(":")

    assert target == "palaver.cli:main"
    assert callable(getattr(import_module(module_name), attribute))
