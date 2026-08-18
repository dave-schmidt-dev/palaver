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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from palaver.ui.companion_state import (
    MAX_ITEMS,
    CompanionState,
    CompanionStateError,
    JoinState,
    read_state,
)

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


def _nonempty(*values: str | None) -> tuple[str, ...]:
    """Keep only the values that would put something on screen."""

    return tuple(value for value in values if value and value.strip())


def _section_items(state: CompanionState) -> dict[str, tuple[str, ...]]:
    """Return each labeled section's items, in the order they should be read.

    ``recent`` is stored oldest-first and is reversed here so NOW's first row
    is always the newest activity. That keeps the row honest in a pane only
    tall enough for one of them, and stops a resize from changing which end
    of the history the label refers to.
    """

    return {
        "REQUEST": _nonempty(state.request),
        "NOW": _nonempty(*reversed(state.recent)),
        "TASKS": _nonempty(*state.tasks),
        "ASK": _nonempty(*state.questions),
        "COMMAND": _nonempty(state.command_result),
        "DETAIL": _nonempty(state.detail),
    }


# Every section with content earns its first row in this order, so a two-line
# pane still shows the request and a four-line one still reaches the question.
_ROW_ORDER = ("REQUEST", "NOW", "ASK", "TASKS", "COMMAND", "DETAIL")
# Spare rows then go to the sections that hold lists, smallest appetite first,
# and NOW absorbs whatever is left because it is the only open-ended one.
_GROWTH_ORDER = ("ASK", "TASKS", "NOW")
_GROWTH_CAPS = {"ASK": 2, "TASKS": 3, "NOW": MAX_ITEMS}
# Reading order down the pane, which is deliberately not the order above.
_DISPLAY_ORDER = ("REQUEST", "NOW", "TASKS", "ASK", "COMMAND", "DETAIL")
# The widest label plus the gutter that lines every section's items up.
_LABEL_WIDTH = 9


def _allocate_rows(items: Mapping[str, Sequence[str]], rows: int) -> dict[str, int]:
    """Divide the available content rows among the sections that have any."""

    counts = dict.fromkeys(items, 0)
    remaining = max(0, rows)
    for label in _ROW_ORDER:
        if remaining <= 0:
            break
        if items[label]:
            counts[label] = 1
            remaining -= 1
    for label in _GROWTH_ORDER:
        if remaining <= 0:
            break
        if not counts[label]:
            continue
        allowed = min(counts[label] + remaining, len(items[label]), _GROWTH_CAPS[label])
        remaining -= allowed - counts[label]
        counts[label] = allowed
    return counts


def _content_lines(state: CompanionState, width: int, rows: int) -> list[str]:
    """Lay the sections out as one clipped row per item under a single label."""

    items = _section_items(state)
    counts = _allocate_rows(items, rows)
    gutter = min(_LABEL_WIDTH, max(1, width))
    value_width = width - gutter
    lines: list[str] = []
    for label in _DISPLAY_ORDER:
        for index, value in enumerate(items[label][: counts[label]]):
            prefix = label.ljust(gutter) if index == 0 else " " * gutter
            if value_width <= 0:
                lines.append(prefix)
            else:
                lines.append(prefix + clip_cells(value, value_width))
    return lines


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
    header = f"PALAVER  {label}  {sanitize(state.project)} · {sanitize(state.source)}"
    lines = [header, *_content_lines(state, width, height - 1)]
    return _encode_frame(lines, width, height)


def render_error_frame(detail: str, width: int, height: int) -> bytes:
    """Render a recoverable transport error without exposing its path."""

    body = "DETAIL".ljust(_LABEL_WIDTH) + clip_cells(detail, max(0, width - _LABEL_WIDTH))
    return _encode_frame(["PALAVER  ERROR", body], width, height)


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
