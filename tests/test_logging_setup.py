"""Tests for logging setup.

`setup_logging()` used to write `.logs/palaver.log` relative to the current
working directory: fine from the repo root, but it litters a `.logs/`
directory into whatever directory a caller happens to be in, and breaks
outright under launchd, whose default working directory is `/`. It now
anchors to `palaver.logging_setup.PROJECT_ROOT`, resolved from this
module's own file location rather than the cwd.

Most tests here monkeypatch `PROJECT_ROOT` to an isolated `tmp_path` rather
than exercising the real repository's `.logs/` — that keeps the suite from
littering actual project state and lets each test assert the anchored
destination without caring where pytest itself was invoked from. The cwd is
separately pointed at a *different* tmp directory in every test, standing in
for "launched from anywhere, including `/`".
"""

import logging
import logging.handlers
from pathlib import Path

import pytest

import palaver.logging_setup as logging_setup
from palaver.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def anchored_project(tmp_path, monkeypatch):
    """Anchor `PROJECT_ROOT` to an isolated directory, cwd to a different one.

    Returns the anchored project directory. The cwd is moved to a sibling
    tmp directory that has no relation to it, so any test using this fixture
    is proof against "it only works because cwd happens to equal the
    anchor."
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.setattr(logging_setup, "PROJECT_ROOT", project_dir)
    monkeypatch.chdir(elsewhere)
    return project_dir


def test_logging_creates_logs_directory(anchored_project):
    """setup_logging() creates .logs/ under the anchor, not under the cwd."""
    logs_dir = anchored_project / ".logs"
    assert not logs_dir.exists(), ".logs should not exist before setup"

    setup_logging()

    assert logs_dir.is_dir(), ".logs should be created under the anchored project dir"
    assert not (Path.cwd() / ".logs").exists(), "no .logs should appear in the cwd"


def test_logging_writes_warning_record(anchored_project):
    """A WARNING record reaches the anchored log file with its message intact."""
    logger = setup_logging()
    message = "canary-warning-record"

    logger.warning(message)

    log_file = anchored_project / ".logs" / "palaver.log"
    assert log_file.exists(), "log file should exist after a record is emitted"
    assert message in log_file.read_text(), "log file should contain the WARNING text"


def test_debug_flag_writes_debug_records(anchored_project):
    """With debug=True a DEBUG record is written, not filtered out."""
    logger = setup_logging(debug=True)
    message = "canary-debug-record"

    logger.debug(message)

    log_file = anchored_project / ".logs" / "palaver.log"
    assert message in log_file.read_text(), "DEBUG record should reach the log file"


def test_default_level_drops_debug_records(anchored_project):
    """At the default level a DEBUG record is filtered out, not written.

    The mirror of the test above: asserting the level attribute alone would
    pass even if the handler let everything through.
    """
    logger = setup_logging()
    logger.debug("should-not-appear")

    log_file = anchored_project / ".logs" / "palaver.log"
    assert "should-not-appear" not in log_file.read_text()


@pytest.mark.parametrize(
    ("debug", "expected"),
    [(True, logging.DEBUG), (False, logging.WARNING)],
)
def test_logger_level_follows_debug_flag(anchored_project, debug, expected):
    """The debug flag maps to the expected logger level."""
    assert setup_logging(debug=debug).level == expected


def test_logging_uses_rotating_file_handler(anchored_project):
    """The handler is a RotatingFileHandler with a bounded size and backups."""
    logger = setup_logging()

    handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert handlers, "logger should have a RotatingFileHandler"
    assert handlers[0].maxBytes > 0, "rotation requires a non-zero maxBytes"
    assert handlers[0].backupCount > 0, "rotation requires at least one backup"


def test_repeated_setup_does_not_duplicate_handlers(anchored_project):
    """Calling setup twice leaves one handler, so records are not written twice."""
    setup_logging()
    logger = setup_logging()

    assert len(logger.handlers) == 1, "repeated setup should not stack handlers"


def test_log_destination_independent_of_cwd(tmp_path, monkeypatch):
    """The resolved log path is identical no matter which directory started it.

    Regression guard for the CWD-relative bug: runs `setup_logging()` twice
    under the same anchored `PROJECT_ROOT` but from two unrelated cwds (one
    standing in for "launched from `/`") and asserts both produce the exact
    same log file path with no `.logs/` left behind in either cwd.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    cwd_a = tmp_path / "cwd_a"
    cwd_a.mkdir()
    cwd_b = tmp_path / "root_stand_in"
    cwd_b.mkdir()

    monkeypatch.setattr(logging_setup, "PROJECT_ROOT", project_dir)

    monkeypatch.chdir(cwd_a)
    logger_a = setup_logging()
    path_a = Path(logger_a.handlers[0].baseFilename)

    monkeypatch.chdir(cwd_b)
    logger_b = setup_logging()
    path_b = Path(logger_b.handlers[0].baseFilename)

    assert path_a == path_b == project_dir / ".logs" / "palaver.log"
    assert not (cwd_a / ".logs").exists(), "no .logs should be left in the first cwd"
    assert not (cwd_b / ".logs").exists(), "no .logs should be left in the second cwd"


def test_cli_run_from_temp_cwd_creates_no_logs_there(anchored_project):
    """Invoking the CLI from an unrelated cwd writes no `.logs/` there.

    Exercises the real wiring (`palaver.cli.main` calling `setup_logging`),
    not just the function in isolation. A positive control -- the same run
    asserted to land a record at the anchored path -- proves this would
    actually catch a regression: without it, the "no .logs here" half would
    pass just as well if logging were silently broken everywhere.
    """
    from palaver.cli import main

    assert main([]) == 2, "no subcommand should print help and exit 2"

    cwd_logs = Path.cwd() / ".logs"
    assert not cwd_logs.exists(), "the CLI must not create .logs/ in the invoking cwd"

    message = "canary-cli-record"
    logging.getLogger("palaver").warning(message)

    log_file = anchored_project / ".logs" / "palaver.log"
    assert log_file.exists(), "positive control: the anchored log file should exist"
    assert message in log_file.read_text(), "positive control: the record should land there"


def test_gitignore_contains_logs_entry():
    """The repository ignores .logs/ so log files are never committed.

    Resolved from this file's location rather than the cwd: the assertion is
    about the repository, not about wherever pytest happened to be started.
    """
    gitignore = REPO_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore should exist at the repository root"

    entries = {line.strip() for line in gitignore.read_text().splitlines()}
    assert ".logs/" in entries, ".gitignore should contain a .logs/ entry"


def test_project_root_matches_repository_root():
    """PROJECT_ROOT resolves to the real repo root, not a symlink or copy.

    Guards the "keep the existing gitignored .logs/ location when running
    inside the repo" requirement: the anchor must be the actual checkout,
    not merely *an* anchored directory.
    """
    assert logging_setup.PROJECT_ROOT == REPO_ROOT
