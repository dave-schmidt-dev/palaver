"""Logging configuration for Palaver."""

import logging
import logging.handlers
from pathlib import Path

#: The project directory, resolved from this module's own file location
#: rather than the current working directory. `logging_setup.py` lives at
#: `<project root>/palaver/logging_setup.py`, so two `.parent` hops land on
#: the project root regardless of where the process was launched from --
#: including launchd, whose default working directory is `/`. Anchoring
#: here (instead of an XDG/`~/.local` state dir) keeps everything inside
#: the project per repo convention, and keeps `.logs/` in its existing
#: gitignored spot when running from within the repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with RotatingFileHandler.

    Creates <project root>/.logs/ if absent. Writes to
    <project root>/.logs/palaver.log at WARNING level by default, dropping
    to DEBUG when debug=True. The destination is anchored to `PROJECT_ROOT`,
    not the current working directory, so it is identical no matter where
    the process was started from.

    Args:
        debug: If True, set logging level to DEBUG; otherwise WARNING.

    Returns:
        The configured logger for the palaver package.
    """
    logs_dir = PROJECT_ROOT / ".logs"
    logs_dir.mkdir(exist_ok=True)

    log_file = logs_dir / "palaver.log"
    level = logging.DEBUG if debug else logging.WARNING

    logger = logging.getLogger("palaver")
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
