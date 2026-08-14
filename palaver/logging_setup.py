"""Logging configuration for Palaver."""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with RotatingFileHandler.

    Creates .logs/ directory if absent. Writes to .logs/palaver.log at WARNING
    level by default, dropping to DEBUG when debug=True.

    Args:
        debug: If True, set logging level to DEBUG; otherwise WARNING.

    Returns:
        The configured logger for the palaver package.
    """
    logs_dir = Path(".logs")
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

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
