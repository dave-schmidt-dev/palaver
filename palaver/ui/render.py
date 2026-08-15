"""Turn a status, a task string, and a width into one line of text.

This is the only part of the status bar that can be tested without iTerm2
running, so it is deliberately the part that holds all the decisions. The
component in task 5.3 does connection and registration; it does not decide
what the text says. Anything that has to be argued about belongs here, where
a headless test can pin it.

**Width is a character count, not a promise about pixels.** iTerm2 measures
status bar components in points and the maximum width is a user-configurable
knob, so no character budget exists to target and the plan asserts none. What
`width` buys is determinism: the same inputs produce the same string, and a
400-character task from a real session cannot push anything else off the bar.
Callers pass whatever budget they have; the function degrades rather than
assuming it was obeyed.

**No per-status glyph.** Every compact symbol considered for the job --
`▶` `◆` `■` `\xb7` `…` -- reports `east_asian_width == "A"`,
ambiguous: one cell in a normally configured terminal and two where ambiguous
characters are set to double width. The status words are ASCII and always one
cell, they need no legend, and leaving the glyph slot empty means task 5.5's
ladybug can mean exactly one thing when it appears -- Palaver itself is
failing -- instead of competing with eight decorative neighbours.
`test_no_rendered_character_is_double_width` is what keeps that true: the
ladybug is `east_asian_width == "W"` and would fail the moment it was added
to `STATUS_LABELS` by mistake.

**The label is never half-shown when it can be shown whole.** At a width that
fits the status word but not the task, the word wins and the task is dropped.
`"question"` tells a human scanning a wall of panes more than `"questio…"`
does, and the truncation marker on a status word reads as a rendering bug
rather than as information.
"""

from __future__ import annotations

import unicodedata

from palaver.observer.signals import Status

#: Appended wherever content was cut, so a clipped line is distinguishable
#: from one that happened to end there. One character, because at the narrow
#: end of the range every character spent on punctuation is one not spent on
#: the status word.
ELLIPSIS = "…"

#: Between the status word and the task text. ASCII, so it cannot widen.
SEPARATOR = ": "

#: The word shown for each status. Lower case because the bar sits in a row
#: of proportional-font components and shouting reads as an error state.
#:
#: `AWAITING_HUMAN` is the union of `DONE`, `WAITING_FOR_USER` and `QUESTION`
#: -- the turn ended and nothing is known about why -- so it gets the vaguest
#: word here. `WAITING_FOR_USER` is the narrower, more urgent case: the turn
#: ended with work still outstanding, so it says so.
STATUS_LABELS: dict[Status, str] = {
    Status.WORKING: "working",
    Status.AWAITING_HUMAN: "your turn",
    Status.ERROR: "error",
    Status.UNKNOWN: "unknown",
    Status.DONE: "done",
    Status.WAITING_FOR_USER: "needs you",
    Status.QUESTION: "question",
    Status.BLOCKED: "blocked",
    Status.IDLE: "idle",
}

# A missing entry would be a `KeyError` at render time, in a coroutine iTerm2
# calls on its own schedule, where the traceback goes to a log nobody is
# reading. Adding a `Status` member is exactly the change that causes it, and
# task 5.2 adding `IDLE` is the proof that members do get added late. Fail at
# import instead, where the test suite cannot avoid seeing it.
_MISSING = set(Status) - set(STATUS_LABELS)
if _MISSING:
    raise RuntimeError(
        "STATUS_LABELS is missing a label for: "
        + ", ".join(sorted(status.name for status in _MISSING))
    )
del _MISSING


def one_line(text: str | None) -> str:
    """Flatten arbitrary session text into something a single row can hold.

    Task strings originate in observed session content, which is prose
    written by and for other programs. It contains newlines, tab-aligned
    output, and -- since agents echo terminal output back into their own
    transcripts -- ANSI escape sequences. A raw `\\x1b` reaching a status bar
    is not a cosmetic problem; it is an escape sequence being handed to a
    terminal by a component whose entire job is to be inert.

    Every C0/C1 control character becomes a space, which disarms the escape
    byte itself, and runs of whitespace then collapse to one. What survives
    of a colour code is the harmless `[31m` tail.

    Args:
        text: The task string, or None when extraction had no opinion.

    Returns:
        A single-line string with no control characters and no leading,
        trailing, or repeated whitespace. Empty when there was nothing.
    """
    if not text:
        return ""
    disarmed = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    return " ".join(disarmed.split())


def render(status: Status, task: str | None = None, width: int = 40) -> str:
    """Compose the status bar line for one session.

    Args:
        status: The status to report, any member of `Status`.
        task: What the session is doing, or None when nothing is known.
        width: Maximum characters. See the module docstring on why this is
            a character count and not a pixel budget.

    Returns:
        A non-empty string of at most `width` characters. When the content
        had to be cut, the length is exactly `width` -- overflow always
        fills the space it was given -- and the last character is
        `ELLIPSIS`. When it fits, the string is returned unpadded.

    Raises:
        ValueError: If `width` is less than one. A zero or negative width
            has no correct rendering, and returning `""` would make an
            arithmetic bug upstream look like a session with nothing to say.
    """
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")

    label = STATUS_LABELS[status]
    text = one_line(task)
    body = f"{label}{SEPARATOR}{text}" if text else label

    if len(body) <= width:
        return body

    if width < len(label):
        # Not even the status word fits. Cut into it, because a partial word
        # is still better than nothing and the caller asked for this width.
        return ELLIPSIS if width == 1 else label[: width - 1] + ELLIPSIS

    if width == len(label):
        # The word fits exactly. Spending its last character on an ellipsis
        # would trade the whole meaning for the news that there was more.
        return label

    return body[: width - 1] + ELLIPSIS
