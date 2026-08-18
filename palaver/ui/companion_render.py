"""Standalone terminal renderer for one Palaver companion pane."""

from __future__ import annotations

import argparse
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
CYAN = "\x1b[36m"
GREEN = "\x1b[32m"
AMBER = "\x1b[33m"
MUTED = "\x1b[2m"
RED = "\x1b[31m"
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

    value = _ANSI_ESCAPE.sub("", value)
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


def _layout_text(value: str) -> str:
    """Remove terminal controls while retaining intentional layout spaces."""

    value = _ANSI_ESCAPE.sub("", value)
    return "".join(
        " " if character.isspace() or unicodedata.category(character).startswith("C") else character
        for character in value
    )


def _clip_layout_cells(value: str, width: int) -> str:
    """Sanitize and clip a frame row without collapsing indentation."""

    if width <= 0:
        return ""
    text = _layout_text(value)
    if cell_width(text) <= width:
        return text
    if width == 1:
        return "…"
    prefix, _ = _take_cells(text, width - 1)
    return prefix + "…"


def _take_cells(value: str, width: int) -> tuple[str, str]:
    """Take a cell-bounded prefix, keeping the unconsumed text."""

    if width <= 0:
        return "", value
    used = 0
    for index, character in enumerate(value):
        character_width = cell_width(character)
        if used == 0 and character_width > width:
            return "…", value[index + 1 :]
        if used + character_width > width:
            return value[:index], value[index:]
        used += character_width
    return value, ""


def _wrap_words(value: str, width: int) -> list[str]:
    """Wrap normalized text to terminal cells, splitting overlong words."""

    if width <= 0:
        return []
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        chunks: list[str] = []
        remainder = word
        while remainder:
            chunk, remainder = _take_cells(remainder, width)
            if not chunk:
                chunk, remainder = remainder[:1], remainder[1:]
            chunks.append(chunk)
        if len(chunks) > 1:
            if current:
                lines.append(current)
                current = ""
            lines.extend(chunks[:-1])
            current = chunks[-1]
            continue
        candidate = word if not current else f"{current} {word}"
        if not current or cell_width(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _line(label: str, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return f"{label} {value}"


def _joined(values: Sequence[str]) -> str | None:
    return " · ".join(values) if values else None


def _content_lines(state: CompanionState, width: int | None = None) -> list[str]:
    latest = state.recent[-1] if state.recent else None
    connection = state.detail or f"{state.source} · {state.join_state.value.lower()}"
    candidates = [
        ("REQUEST", state.request),
        ("NOW", latest),
        ("ASK", _joined(state.questions)) if state.questions else ("COMMAND", state.command_result),
        ("TASKS", _joined(state.tasks)),
        ("DETAIL", connection),
    ]
    if width is None:
        return [line for label, value in candidates if (line := _line(label, value)) is not None]
    wrapped: list[str] = []
    width = max(1, width)
    for label, value in candidates:
        if value is None or not value.strip():
            continue
        clean_value = sanitize(value)
        prefix = f"{label} "
        prefix_width = cell_width(prefix)
        first_width = width - prefix_width
        if first_width > 0:
            value_lines = _wrap_words(clean_value, first_width)
            first_value = value_lines.pop(0) if value_lines else ""
            wrapped.append(_take_cells(prefix, width)[0] + first_value)
            continuation_width = max(1, width - prefix_width)
            for continuation in value_lines:
                continuation, _ = _take_cells(continuation, continuation_width)
                wrapped.append(" " * prefix_width + continuation)
        else:
            wrapped.append(_take_cells(prefix, width)[0])
            wrapped.extend(_wrap_words(clean_value, width))
    return wrapped


_STATUS_COLORS = {
    "WORKING": GREEN,
    "DONE": GREEN,
    "AWAITING_HUMAN": AMBER,
    "WAITING_FOR_USER": AMBER,
    "QUESTION": AMBER,
    "IDLE": AMBER,
    "BLOCKED": RED,
    "STALE": MUTED,
    "STARTING": CYAN,
    "UNJOINED": AMBER,
    "ERROR": RED,
    "UNKNOWN": RED,
}
_CONTENT_LABELS = {"REQUEST", "NOW", "ASK", "COMMAND", "TASKS", "DETAIL"}


def _style_line(line: str) -> str:
    """Color only Palaver-owned header and semantic labels."""

    if line.startswith("PALAVER"):
        styled = f"{CYAN}PALAVER{RESET}"
        remainder = line[len("PALAVER") :]
        status_match = re.match(r"\s+(\S+)", remainder)
        status = status_match.group(1) if status_match else ""
        if status in _STATUS_COLORS:
            offset = status_match.start(1) if status_match else 0
            styled += remainder[:offset] + f"{_STATUS_COLORS[status]}{status}{RESET}"
            styled += remainder[offset + len(status) :]
            return styled
        return styled + remainder
    label, separator, value = line.partition(" ")
    if separator and label in _CONTENT_LABELS:
        return f"{CYAN}{label}{RESET}{separator}{value}"
    return line


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
    header = f"PALAVER {label} {sanitize(state.project)}"
    lines = [header, *_content_lines(state, width)]
    return _encode_frame(lines, width, height)


def render_error_frame(detail: str, width: int, height: int) -> bytes:
    """Render a recoverable transport error without exposing its path."""

    return _encode_frame(["PALAVER ERROR", f"DETAIL {sanitize(detail)}"], width, height)


def _encode_frame(lines: Sequence[str], width: int, height: int) -> bytes:
    width = max(1, width)
    height = max(1, height)
    visible = list(lines[:height])
    while len(visible) < height:
        visible.append("")
    padded = []
    for line in visible:
        clipped = _clip_layout_cells(line, width)
        padded.append(_style_line(clipped) + " " * max(0, width - cell_width(clipped)))
    body = "\r\n".join(padded)
    return f"{HOME}{body}{RESET}".encode("utf-8")


class TerminalSession:
    """Own and always restore the renderer's TTY modes."""

    def __init__(self, input_fd: int, output_fd: int) -> None:
        self.input_fd = input_fd
        self.output_fd = output_fd
        self._attributes: list | None = None

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
            os.write(self.output_fd, ENTER_SCREEN)
        except BaseException:
            try:
                termios.tcflush(self.input_fd, termios.TCIFLUSH)
            except OSError:
                pass
            termios.tcsetattr(self.input_fd, termios.TCSANOW, self._attributes)
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
                os.write(self.output_fd, LEAVE_SCREEN)
            except OSError:
                # Preserve the exception that caused context teardown. A dead
                # output fd must not replace the renderer's primary failure.
                if _exc_type is None:
                    raise

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
