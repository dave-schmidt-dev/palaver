"""The shared progress writer keeps stdout the result channel (INV-1)."""

from __future__ import annotations

from palaver.progress import stderr_status


def test_stderr_status_writes_one_line_to_stderr_and_nothing_to_stdout(capsys):
    stderr_status("working")
    captured = capsys.readouterr()
    assert captured.err == "working\n"
    assert captured.out == ""
