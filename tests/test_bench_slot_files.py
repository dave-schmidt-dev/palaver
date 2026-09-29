"""
Slot-file disk usage: unavailable when unsupplied, measured when given a path, and
honest about a missing path.
"""

from __future__ import annotations

from palaver.bench import (
    SLOT_PATH_UNKNOWN_NOTE,
    SlotFileUsage,
    measure_slot_files,
)
from palaver.cli import bench as bench_cli
from tests._bench_support import SESSIONS, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# Slot files: an honest "not measured" instead of a zero
# =============================================================================


def test_slot_file_usage_is_unavailable_when_no_path_is_supplied():
    usage = measure_slot_files(None)

    assert usage == SlotFileUsage("", False, 0, 0, SLOT_PATH_UNKNOWN_NOTE)
    assert "--slot-save-path" in usage.detail


def test_slot_file_usage_is_measured_when_a_path_is_supplied(tmp_path):
    slot_dir = tmp_path / "slots"
    slot_dir.mkdir()
    (slot_dir / "slot-0.bin").write_bytes(b"x" * 2048)
    (slot_dir / "slot-1.bin").write_bytes(b"y" * 1024)

    usage = measure_slot_files(slot_dir)

    assert usage.available
    assert usage.file_count == 2
    assert usage.total_bytes == 3072


def test_slot_file_usage_says_so_when_the_path_does_not_exist(tmp_path):
    usage = measure_slot_files(tmp_path / "absent")

    assert not usage.available
    assert usage.total_bytes == 0
    assert "no directory" in usage.detail


def test_the_rendered_report_never_shows_an_unmeasured_zero(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)
    report = _run(handle, tmp_path / "bench.db")

    rendered = bench_cli.render_report(report, host=handle.host, port=handle.port, detailed=False)

    assert "slot files: unavailable" in rendered
    assert "slot files: 0 file(s), 0.0 MiB" not in rendered
