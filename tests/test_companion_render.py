from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
import signal
import stat
import struct
import subprocess
import sys
import termios
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from palaver.ui import companion_render
from palaver.ui.companion_render import (
    _LABEL_WIDTH,
    ENTER_SCREEN,
    LEAVE_SCREEN,
    TerminalSession,
    cell_width,
    clip_cells,
    render_frame,
    run_renderer,
    sanitize,
)
from palaver.ui.companion_state import (
    MAX_ITEMS,
    MAX_STATE_BYTES,
    CompanionState,
    CompanionStateError,
    JoinState,
    atomic_write_state,
    read_state,
)


def _state(*, updated: float = 100.0, **changes) -> CompanionState:
    values = {
        "producer_updated_at": updated,
        "project": "Alpha Project",
        "source": "codex",
        "status": "working",
        "join_state": JoinState.JOINED,
        "request": "ship companion panes",
        "command_result": "tests are passing",
        "detail": "exact pane joined",
        "recent": ("parsed tool result", "updated plan"),
        "tasks": ("render", "verify"),
        "questions": ("deploy now?",),
    }
    values.update(changes)
    return CompanionState(**values)


def _frame_text(state: CompanionState, width: int, height: int) -> str:
    return _strip_ansi(render_frame(state, width, height, now=101.0).decode())


def _strip_ansi(value: str | bytes) -> str | bytes:
    pattern = r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])"
    if isinstance(value, bytes):
        return re.sub(pattern.encode(), b"", value)
    return re.sub(pattern, "", value)


def test_golden_frame_20_by_2_prioritizes_request():
    assert _frame_text(_state(), 20, 2) == ("PALAVER  WORKING  A…\r\nREQUEST  ship compa…")


def test_golden_frame_40_by_4_uses_latest_activity_and_question():
    assert _frame_text(_state(), 40, 4) == (
        "PALAVER  WORKING  Alpha Project · codex \r\n"
        "REQUEST  ship companion panes           \r\n"
        "NOW      updated plan                   \r\n"
        "ASK      deploy now?                    "
    )


def test_golden_frame_80_by_6_shows_every_section_once():
    assert _frame_text(_state(), 80, 6) == (
        "PALAVER  WORKING  Alpha Project · codex                                         \r\n"
        "REQUEST  ship companion panes                                                   \r\n"
        "NOW      updated plan                                                           \r\n"
        "TASKS    render                                                                 \r\n"
        "ASK      deploy now?                                                            \r\n"
        "COMMAND  tests are passing                                                      "
    )


def test_golden_frame_80_by_10_grows_lists_under_one_label():
    assert _frame_text(_state(), 80, 10) == (
        "PALAVER  WORKING  Alpha Project · codex                                         \r\n"
        "REQUEST  ship companion panes                                                   \r\n"
        "NOW      updated plan                                                           \r\n"
        "         parsed tool result                                                     \r\n"
        "TASKS    render                                                                 \r\n"
        "         verify                                                                 \r\n"
        "ASK      deploy now?                                                            \r\n"
        "COMMAND  tests are passing                                                      \r\n"
        "DETAIL   exact pane joined                                                      \r\n"
        "                                                                                "
    )


def test_spare_rows_go_to_recent_activity_newest_first():
    busy = _state(recent=tuple(f"step {index}" for index in range(MAX_ITEMS)))
    rows = _frame_text(busy, 40, 10).split("\r\n")
    assert rows[2].startswith("NOW      ")
    assert [row[_LABEL_WIDTH:].strip() for row in rows[2:5]] == ["step 7", "step 6", "step 5"]
    assert [row.split()[0] for row in rows[5:]] == ["TASKS", "verify", "ASK", "COMMAND", "DETAIL"]
    taller = _frame_text(busy, 40, 12).split("\r\n")
    assert [row[_LABEL_WIDTH:].strip() for row in taller[2:7]] == [
        f"step {n}" for n in (7, 6, 5, 4, 3)
    ]


def test_a_question_and_a_command_result_no_longer_compete_for_one_row():
    frame = _frame_text(_state(), 80, 10)
    assert "ASK      deploy now?" in frame
    assert "COMMAND  tests are passing" in frame


def test_the_detail_row_appears_only_when_the_producer_supplied_one():
    joined = _frame_text(_state(detail=None), 80, 10)
    assert "DETAIL" not in joined
    unjoined = _state(detail="Waiting for an exact session join", join_state=JoinState.UNJOINED)
    assert "DETAIL   Waiting for an exact session join" in _frame_text(unjoined, 80, 10)


def test_join_and_stale_states_override_model_status():
    unjoined = _state(join_state=JoinState.UNJOINED, status="done")
    assert "PALAVER  UNJOINED" in _frame_text(unjoined, 40, 2)
    assert "PALAVER  STALE" in _strip_ansi(render_frame(unjoined, 40, 2, now=200).decode())
    assert "PALAVER  STALE" in _strip_ansi(render_frame(unjoined, 40, 2, now=0).decode())
    failed = _state(join_state=JoinState.ERROR, status="working")
    assert "PALAVER  ERROR" in _frame_text(failed, 40, 2)


@pytest.mark.parametrize(
    ("status", "color"),
    [
        ("WORKING", companion_render.GREEN),
        ("DONE", companion_render.GREEN),
        ("AWAITING_HUMAN", companion_render.AMBER),
        ("WAITING_FOR_USER", companion_render.AMBER),
        ("QUESTION", companion_render.AMBER),
        ("IDLE", companion_render.AMBER),
        ("BLOCKED", companion_render.RED),
        ("UNKNOWN", companion_render.RED),
    ],
)
def test_every_live_status_has_a_semantic_color(status, color):
    frame = render_frame(_state(status=status), 40, 2, now=101).decode()
    assert f"{color}{status}{companion_render.RESET}" in frame


@pytest.mark.parametrize(
    ("join_state", "color"),
    [
        (JoinState.STARTING, companion_render.CYAN),
        (JoinState.UNJOINED, companion_render.AMBER),
        (JoinState.ERROR, companion_render.RED),
    ],
)
def test_every_nonjoined_state_has_a_semantic_color(join_state, color):
    frame = render_frame(_state(join_state=join_state), 40, 2, now=101).decode()
    assert f"{color}{join_state.value}{companion_render.RESET}" in frame


def test_control_and_ansi_sequences_are_removed_not_rendered():
    hostile = "safe\x1b[31m red\x1b[0m\nnext\x00end"
    assert sanitize(hostile) == "safe red next end"
    frame = render_frame(_state(request=hostile), 80, 2, now=101)
    assert b"\x1b[31m" not in frame
    assert b"\x1b[36m" in frame
    assert b"[31m" not in frame


def test_payload_clips_to_cells_and_only_owned_tokens_are_colored():
    state = _state(request="first 界 second third", status="done", project="user text")
    frame = render_frame(state, 16, 5, now=101).decode()
    plain = _strip_ansi(frame)
    rows = plain.split("\r\n")
    assert rows[1] == "REQUEST  first …"
    assert all(cell_width(row) == 16 for row in rows)
    assert b"\x1b[36mPALAVER" in frame.encode()
    assert b"\x1b[32mDONE" in frame.encode()
    assert b"\x1b[36mREQUEST" in frame.encode()
    assert b"\x1b[36mfirst" not in frame.encode()


def test_cell_clipping_accounts_for_wide_and_combining_characters():
    value = "A界e\u0301Z"
    assert cell_width(value) == 5
    assert clip_cells(value, 4) == "A界…"
    assert cell_width(clip_cells(value, 4)) == 4


def test_state_is_immutable_bounded_and_round_trips_mode_0600(tmp_path):
    path = tmp_path / ".state" / "companion.json"
    state_value = _state()
    atomic_write_state(path, state_value)
    assert read_state(path) == state_value
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FrozenInstanceError):
        state_value.status = "done"  # type: ignore[misc]
    with pytest.raises(CompanionStateError, match="more than"):
        _state(tasks=tuple(str(index) for index in range(MAX_ITEMS + 1)))


def test_atomic_replace_overwrites_a_collision_and_repairs_permissions(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o644)
    atomic_write_state(path, _state(status="done"))
    assert read_state(path).status == "done"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.tmp"))


def test_directory_collision_fails_without_leaving_a_temporary_file(tmp_path):
    path = tmp_path / "state.json"
    path.mkdir()
    with pytest.raises(OSError):
        atomic_write_state(path, _state())
    assert path.is_dir()
    assert not list(tmp_path.glob(".*.tmp"))


def test_reader_refuses_unknown_fields_versions_and_oversized_files(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_state(path, _state())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompanionStateError, match="fields"):
        read_state(path)
    payload.pop("unknown")
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CompanionStateError, match="unsupported"):
        read_state(path)
    path.write_bytes(b"x" * (MAX_STATE_BYTES + 1))
    with pytest.raises(CompanionStateError, match="exceeds"):
        read_state(path)


def _set_size(fd: int, width: int, height: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))


def _restorable_attributes(attributes: list) -> list:
    """Mask macOS's read-only PENDIN report after canonical mode changes."""

    normalized = list(attributes)
    normalized[3] &= ~getattr(termios, "PENDIN", 0)
    return normalized


def _read_available(fd: int, timeout: float = 0.2) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], min(0.03, deadline - time.monotonic()))
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _read_until(fd: int, needle: bytes, timeout: float = 3.0) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while (
        needle not in output
        and needle not in _strip_ansi(bytes(output))
        and time.monotonic() < deadline
    ):
        output.extend(_read_available(fd, 0.08))
    assert needle in output or needle in _strip_ansi(bytes(output)), output
    return bytes(output)


def _start_renderer(path: Path, *, width: int = 80, height: int = 10):
    master, slave = pty.openpty()
    _set_size(slave, width, height)
    process = subprocess.Popen(
        [sys.executable, "-m", "palaver.ui.companion_render", "--state", str(path)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        close_fds=True,
    )
    return process, master, slave


def _stop_renderer(process: subprocess.Popen, master: int, slave: int) -> bytes:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    process.wait(timeout=3)
    output = _read_available(master, 0.3)
    os.close(master)
    os.close(slave)
    return output


def test_pty_initializes_once_discards_input_and_cleans_up_on_sigterm(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_state(path, _state(updated=time.time()))
    process, master, slave = _start_renderer(path)
    output = _read_until(master, b"PALAVER")
    assert output.count(ENTER_SCREEN) == 1
    marker = b"ACCIDENTAL_INPUT_MUST_NOT_ECHO"
    os.write(master, marker)
    assert marker not in _read_available(master, 0.3)
    trailing = _stop_renderer(process, master, slave)
    assert LEAVE_SCREEN in trailing
    assert process.returncode == 0


def test_pty_redraws_at_new_width_after_sigwinch(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_state(path, _state(updated=time.time(), project="A very long project name"))
    process, master, slave = _start_renderer(path, width=80)
    _read_until(master, b"A very long project name")
    _set_size(slave, 20, 4)
    process.send_signal(signal.SIGWINCH)
    resized = _read_until(master, "PALAVER  WORKING  A…".encode())
    assert b"A very long project name" not in resized
    _stop_renderer(process, master, slave)


def test_pty_marks_a_quiet_producer_stale_without_another_write(tmp_path):
    path = tmp_path / "state.json"
    process, master, slave = _start_renderer(path)
    _read_until(master, b"PALAVER  ERROR")
    crossing = time.time() - companion_render.STALE_AFTER_SECONDS + 1.0
    atomic_write_state(path, _state(updated=crossing))
    initial = _read_until(master, b"PALAVER  WORKING")
    assert b"STALE" not in initial
    written_stamp = path.stat().st_mtime_ns
    transitioned = _read_until(master, b"PALAVER  STALE", timeout=3)
    assert b"PALAVER  STALE" in _strip_ansi(transitioned)
    assert path.stat().st_mtime_ns == written_stamp
    _stop_renderer(process, master, slave)


def test_pty_recovers_from_missing_then_malformed_then_valid_state(tmp_path):
    path = tmp_path / "state.json"
    process, master, slave = _start_renderer(path)
    _read_until(master, b"PALAVER  ERROR")
    path.write_text("{bad", encoding="utf-8")
    time.sleep(0.15)
    assert process.poll() is None
    atomic_write_state(path, _state(updated=time.time(), status="done"))
    recovered = _read_until(master, b"PALAVER  DONE")
    assert b"PALAVER  DONE" in _strip_ansi(recovered)
    _stop_renderer(process, master, slave)


def test_normal_return_restores_terminal_flags_and_screen(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_state(path, _state(updated=time.time()))
    master, slave = pty.openpty()
    _set_size(slave, 40, 4)
    original_attributes = termios.tcgetattr(slave)
    original_flags = fcntl.fcntl(slave, fcntl.F_GETFL)
    polls = 0

    def should_stop() -> bool:
        nonlocal polls
        polls += 1
        return polls > 1

    assert run_renderer(path, input_fd=slave, output_fd=slave, should_stop=should_stop) == 0
    output = _read_available(master)
    assert ENTER_SCREEN in output
    assert LEAVE_SCREEN in output
    assert _restorable_attributes(termios.tcgetattr(slave)) == original_attributes
    assert fcntl.fcntl(slave, fcntl.F_GETFL) & os.O_NONBLOCK == original_flags & os.O_NONBLOCK
    os.close(master)
    os.close(slave)


def test_terminal_session_keeps_shared_output_blocking_while_discarding_input():
    master, slave = pty.openpty()
    output = os.dup(slave)
    original_attributes = termios.tcgetattr(slave)
    original_nonblocking = fcntl.fcntl(output, fcntl.F_GETFL) & os.O_NONBLOCK
    assert original_nonblocking == 0

    with TerminalSession(slave, output) as terminal:
        assert fcntl.fcntl(slave, fcntl.F_GETFL) & os.O_NONBLOCK == original_nonblocking
        assert fcntl.fcntl(output, fcntl.F_GETFL) & os.O_NONBLOCK == original_nonblocking
        _read_available(master)
        marker = b"DISCARD_WITHOUT_NONBLOCK"
        os.write(master, marker)
        terminal.discard_input()
        assert marker not in _read_available(master)

    assert _restorable_attributes(termios.tcgetattr(slave)) == original_attributes
    assert fcntl.fcntl(output, fcntl.F_GETFL) & os.O_NONBLOCK == original_nonblocking
    assert LEAVE_SCREEN in _read_available(master)
    os.close(output)
    os.close(master)
    os.close(slave)


def test_enter_failure_restores_terminal_modes(monkeypatch):
    master, slave = pty.openpty()
    original_attributes = termios.tcgetattr(slave)
    original_flags = fcntl.fcntl(slave, fcntl.F_GETFL)

    def fail_write(_fd, _data):
        raise OSError("fixture write failure")

    monkeypatch.setattr(companion_render.os, "write", fail_write)
    with pytest.raises(OSError, match="fixture write failure"):
        TerminalSession(slave, slave).__enter__()
    assert _restorable_attributes(termios.tcgetattr(slave)) == original_attributes
    assert fcntl.fcntl(slave, fcntl.F_GETFL) == original_flags
    os.close(master)
    os.close(slave)


def test_leave_screen_write_failure_does_not_mask_primary_exception(monkeypatch):
    master, slave = pty.openpty()
    original_write = companion_render.os.write

    def fail_leave(fd, data):
        if data == LEAVE_SCREEN:
            raise OSError("leave screen failed")
        return original_write(fd, data)

    monkeypatch.setattr(companion_render.os, "write", fail_leave)
    with pytest.raises(ValueError, match="primary failure"):
        with TerminalSession(slave, slave):
            raise ValueError("primary failure")
    os.close(master)
    os.close(slave)


@pytest.mark.parametrize("arguments", [[], ["--bogus"], ["--state"]])
def test_cli_rejects_missing_and_bad_arguments(arguments):
    result = subprocess.run(
        [sys.executable, "-m", "palaver.ui.companion_render", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_one_frame_is_one_output_write(monkeypatch):
    writes = []

    def record_write(fd, data):
        writes.append((fd, data))
        return len(data)

    monkeypatch.setattr(companion_render.os, "write", record_write)
    TerminalSession(4, 7).draw(render_frame(_state(), 20, 2, now=101))
    assert len(writes) == 1
    assert writes[0][0] == 7
    assert writes[0][1].startswith(b"\x1b[H")
