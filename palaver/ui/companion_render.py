"""Standalone terminal renderer for one Palaver companion pane."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import signal
import sys
import termios
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from palaver.ui.companion_state import CompanionState, CompanionStateError, JoinState, read_state

POLL_INTERVAL_SECONDS = 0.1
STALE_AFTER_SECONDS = 10.0
ENTER_SCREEN = b"\x1b[?1049h\x1b[?25l"
LEAVE_SCREEN = b"\x1b[0m\x1b[?25h\x1b[?1049l"
HOME = "\x1b[H"
RESET = "\x1b[0m"
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])")


def sanitize(value: str) -> str:
    """Return printable, single-line text with terminal controls removed."""

    value = _ANSI_ESCAPE.sub("", value)
    cleaned = []
    for character in value:
        category = unicodedata.category(character)
        if character.isspace():
            cleaned.append(" ")
        elif character == "\x1b" or category.startswith("C"):
            cleaned.append(" ")
        else:
            cleaned.append(character)
    return " ".join("".join(cleaned).split())


def cell_width(value: str) -> int:
    """Return terminal-cell width without external dependencies."""

    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def clip_cells(value: str, width: int) -> str:
    """Sanitize and clip text to ``width`` terminal cells."""

    if width <= 0:
        return ""
    text = sanitize(value)
    if cell_width(text) <= width:
        return text
    if width == 1:
        return "…"
    target = width - 1
    used = 0
    output = []
    for character in text:
        character_width = cell_width(character)
        if used + character_width > target:
            break
        output.append(character)
        used += character_width
    return "".join(output) + "…"


def _line(label: str, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return f"{label} {value}"


def _joined(values: Sequence[str]) -> str | None:
    return " · ".join(values) if values else None


def _content_lines(state: CompanionState) -> list[str]:
    latest = state.recent[-1] if state.recent else None
    question_or_command = (
        _line("ASK", _joined(state.questions))
        if state.questions
        else _line("COMMAND", state.command_result)
    )
    connection = state.detail or f"{state.source} · {state.join_state.value.lower()}"
    candidates = [
        _line("REQUEST", state.request),
        _line("NOW", latest),
        question_or_command,
        _line("TASKS", _joined(state.tasks)),
        _line("DETAIL", connection),
    ]
    return [line for line in candidates if line is not None]


def render_frame(
    state: CompanionState,
    width: int,
    height: int,
    *,
    now: float | None = None,
    stale_after: float = STALE_AFTER_SECONDS,
) -> bytes:
    """Render one complete screen frame, including cursor-home and erase."""

    now = time.time() if now is None else now
    stale = _is_stale(state, now, stale_after)
    if state.join_state is JoinState.ERROR:
        label = "ERROR"
    elif stale:
        label = "STALE"
    elif state.join_state is not JoinState.JOINED:
        label = state.join_state.value
    else:
        label = state.status.upper()
    header = f"PALAVER  {label}  {state.project}"
    lines = [header, *_content_lines(state)]
    return _encode_frame(lines, width, height)


def render_error_frame(detail: str, width: int, height: int) -> bytes:
    """Render a recoverable transport error without exposing its path."""

    return _encode_frame(["PALAVER  ERROR", f"DETAIL {detail}"], width, height)


def _encode_frame(lines: Sequence[str], width: int, height: int) -> bytes:
    width = max(1, width)
    height = max(1, height)
    visible = list(lines[:height])
    while len(visible) < height:
        visible.append("")
    padded = []
    for line in visible:
        clipped = clip_cells(line, width)
        padded.append(clipped + " " * max(0, width - cell_width(clipped)))
    body = "\r\n".join(padded)
    return f"{HOME}{body}{RESET}".encode("utf-8")


class TerminalSession:
    """Own and always restore the renderer's TTY modes."""

    def __init__(self, input_fd: int, output_fd: int) -> None:
        self.input_fd = input_fd
        self.output_fd = output_fd
        self._attributes: list | None = None
        self._flags: int | None = None

    def __enter__(self) -> TerminalSession:
        if not os.isatty(self.input_fd) or not os.isatty(self.output_fd):
            raise RuntimeError("companion renderer requires a terminal")
        self._attributes = termios.tcgetattr(self.input_fd)
        changed = termios.tcgetattr(self.input_fd)
        changed[3] &= ~(termios.ECHO | termios.ICANON)
        changed[6][termios.VMIN] = 0
        changed[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(self.input_fd, termios.TCSANOW, changed)
            self._flags = fcntl.fcntl(self.input_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.input_fd, fcntl.F_SETFL, self._flags | os.O_NONBLOCK)
            os.write(self.output_fd, ENTER_SCREEN)
        except BaseException:
            try:
                termios.tcflush(self.input_fd, termios.TCIFLUSH)
            except OSError:
                pass
            try:
                termios.tcsetattr(self.input_fd, termios.TCSANOW, self._attributes)
            finally:
                if self._flags is not None:
                    fcntl.fcntl(self.input_fd, fcntl.F_SETFL, self._flags)
            raise
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            try:
                termios.tcflush(self.input_fd, termios.TCIFLUSH)
            except OSError:
                pass
            if self._attributes is not None:
                termios.tcsetattr(self.input_fd, termios.TCSANOW, self._attributes)
        finally:
            try:
                if self._flags is not None:
                    fcntl.fcntl(self.input_fd, fcntl.F_SETFL, self._flags)
            finally:
                os.write(self.output_fd, LEAVE_SCREEN)

    def size(self) -> tuple[int, int]:
        size = os.get_terminal_size(self.output_fd)
        return max(1, size.columns), max(1, size.lines)

    def discard_input(self) -> None:
        while True:
            ready, _, _ = select.select([self.input_fd], [], [], 0)
            if not ready:
                return
            try:
                if not os.read(self.input_fd, 4096):
                    return
            except BlockingIOError:
                return

    def draw(self, frame: bytes) -> None:
        written = os.write(self.output_fd, frame)
        if written != len(frame):
            raise OSError(f"incomplete terminal frame ({written}/{len(frame)} bytes)")


def _state_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _is_stale(state: CompanionState, now: float, stale_after: float) -> bool:
    """Bound trust in both old timestamps and implausible future timestamps."""

    return abs(now - state.producer_updated_at) > stale_after


def run_renderer(
    state_path: Path,
    *,
    input_fd: int = 0,
    output_fd: int = 1,
    clock: Callable[[], float] = time.time,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    stale_after: float = STALE_AFTER_SECONDS,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Render until signalled, recovering when the state file becomes valid."""

    stop_requested = False
    resize_requested = True

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    def request_resize(_signum, _frame) -> None:
        nonlocal resize_requested
        resize_requested = True

    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGWINCH)
    }
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGWINCH, request_resize)
    stop = (lambda: False) if should_stop is None else should_stop
    try:
        with TerminalSession(input_fd, output_fd) as terminal:
            state: CompanionState | None = None
            error_detail = "waiting for state"
            stamp: tuple[int, int] | None | object = object()
            previous_frame: bytes | None = None
            previous_stale = False
            next_poll = monotonic()
            while not stop_requested and not stop():
                terminal.discard_input()
                current_time = monotonic()
                if current_time >= next_poll:
                    new_stamp = _state_stamp(state_path)
                    if new_stamp != stamp:
                        stamp = new_stamp
                        try:
                            state = read_state(state_path)
                            error_detail = ""
                        except CompanionStateError:
                            state = None
                            error_detail = "state unavailable or invalid"
                    next_poll = current_time + poll_interval

                width, height = terminal.size()
                stale = state is not None and _is_stale(state, clock(), stale_after)
                if resize_requested or stale != previous_stale:
                    previous_frame = None
                if state is None:
                    frame = render_error_frame(error_detail, width, height)
                else:
                    frame = render_frame(state, width, height, now=clock(), stale_after=stale_after)
                if frame != previous_frame:
                    terminal.draw(frame)
                    previous_frame = frame
                previous_stale = stale
                resize_requested = False
                sleep(min(poll_interval, max(0.0, next_poll - monotonic())))
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m palaver.ui.companion_render")
    parser.add_argument("--state", required=True, type=Path, help="private companion state file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_renderer(args.state)
    except RuntimeError as exc:
        print(f"palaver companion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised as a PTY process
    raise SystemExit(main())
