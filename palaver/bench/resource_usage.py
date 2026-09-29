"""
Peak RSS normalized to bytes, and slot-file disk usage reported honestly as unmeasured
rather than as a zero.
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

from .records import SlotFileUsage

#: `ru_maxrss` is in bytes on macOS and kilobytes on Linux. Checked once here
#: rather than at each call site so the two cannot disagree.
_RSS_IN_BYTES = sys.platform == "darwin"

#: Printed when slot-file disk usage was not measured because no path was
#: given. A module constant because `tests/test_bench.py` asserts the report
#: carries it — a benchmark reporting 0 bytes when it never looked is the
#: quiet-zero failure this task's last criterion exists to prevent.
SLOT_PATH_UNKNOWN_NOTE = (
    "not measured: llama-server exposes no --slot-save-path over HTTP "
    "(/props carries no argv or command line), so pass --slot-save-path "
    "to measure slot-file disk usage"
)


def normalize_rss(raw: int) -> int:
    """Convert a raw `ru_maxrss` reading to bytes.

    Split out from `peak_rss_bytes` so the platform mapping is testable on
    both branches without moving the clock or the machine: a test that only
    compares a live reading against itself agrees with whichever unit the
    module happens to have chosen.
    """
    return int(raw) if _RSS_IN_BYTES else int(raw) * 1024


def peak_rss_bytes() -> int:
    """Return this process's peak RSS in bytes, normalized across platforms.

    `resource.getrusage(RUSAGE_SELF).ru_maxrss` is bytes on macOS and
    kilobytes on Linux. Returning the raw value would make a benchmark report
    wrong by 1024x on one of them, in a direction no reader could detect.
    """
    return normalize_rss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def measure_slot_files(path: Path | str | None) -> SlotFileUsage:
    """Sum the on-disk size of a server's KV slot files.

    Args:
        path: The server's `--slot-save-path`, or `None` when it is unknown.
            It is unknown by default: task 4.2 established that `/props`
            carries no invocation, so nothing here can discover it.

    Returns:
        A `SlotFileUsage`. `available` is False both when no path was given
        and when the given path is not a directory, with `detail` saying
        which — never a bare 0.
    """
    if path is None:
        return SlotFileUsage("", False, 0, 0, SLOT_PATH_UNKNOWN_NOTE)
    directory = Path(path)
    if not directory.is_dir():
        return SlotFileUsage(
            str(directory), False, 0, 0, f"not measured: no directory at {directory}"
        )
    files = [entry for entry in directory.rglob("*") if entry.is_file()]
    return SlotFileUsage(
        str(directory), True, len(files), sum(entry.stat().st_size for entry in files), ""
    )
