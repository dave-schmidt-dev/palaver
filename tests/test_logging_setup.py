"""Tests for logging setup.

Every test that calls `setup_logging()` runs inside `tmp_path`, because the
function writes `.logs/palaver.log` relative to the current working directory.
Without the chdir these tests would litter the repository root and would pass
or fail depending on where pytest was invoked from.
"""

import logging
import logging.handlers
from pathlib import Path

import pytest

from palaver.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run a test in an empty directory, restoring the cwd afterwards."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_logging_creates_logs_directory(in_tmp_cwd):
    """setup_logging() creates the .logs directory when it is absent."""
    logs_dir = in_tmp_cwd / ".logs"
    assert not logs_dir.exists(), ".logs should not exist before setup"

    setup_logging()

    assert logs_dir.is_dir(), ".logs should be created as a directory"


def test_logging_writes_warning_record(in_tmp_cwd):
    """A WARNING record reaches the log file with its message intact."""
    logger = setup_logging()
    message = "canary-warning-record"

    logger.warning(message)

    log_file = in_tmp_cwd / ".logs" / "palaver.log"
    assert log_file.exists(), "log file should exist after a record is emitted"
    assert message in log_file.read_text(), "log file should contain the WARNING text"


def test_debug_flag_writes_debug_records(in_tmp_cwd):
    """With debug=True a DEBUG record is written, not filtered out."""
    logger = setup_logging(debug=True)
    message = "canary-debug-record"

    logger.debug(message)

    log_file = in_tmp_cwd / ".logs" / "palaver.log"
    assert message in log_file.read_text(), "DEBUG record should reach the log file"


def test_default_level_drops_debug_records(in_tmp_cwd):
    """At the default level a DEBUG record is filtered out, not written.

    The mirror of the test above: asserting the level attribute alone would
    pass even if the handler let everything through.
    """
    logger = setup_logging()
    logger.debug("should-not-appear")

    log_file = in_tmp_cwd / ".logs" / "palaver.log"
    assert "should-not-appear" not in log_file.read_text()


@pytest.mark.parametrize(
    ("debug", "expected"),
    [(True, logging.DEBUG), (False, logging.WARNING)],
)
def test_logger_level_follows_debug_flag(in_tmp_cwd, debug, expected):
    """The debug flag maps to the expected logger level."""
    assert setup_logging(debug=debug).level == expected


def test_logging_uses_rotating_file_handler(in_tmp_cwd):
    """The handler is a RotatingFileHandler with a bounded size and backups."""
    logger = setup_logging()

    handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert handlers, "logger should have a RotatingFileHandler"
    assert handlers[0].maxBytes > 0, "rotation requires a non-zero maxBytes"
    assert handlers[0].backupCount > 0, "rotation requires at least one backup"


def test_repeated_setup_does_not_duplicate_handlers(in_tmp_cwd):
    """Calling setup twice leaves one handler, so records are not written twice."""
    setup_logging()
    logger = setup_logging()

    assert len(logger.handlers) == 1, "repeated setup should not stack handlers"


def test_gitignore_contains_logs_entry():
    """The repository ignores .logs/ so log files are never committed.

    Resolved from this file's location rather than the cwd: the assertion is
    about the repository, not about wherever pytest happened to be started.
    """
    gitignore = REPO_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore should exist at the repository root"

    entries = {line.strip() for line in gitignore.read_text().splitlines()}
    assert ".logs/" in entries, ".gitignore should contain a .logs/ entry"
