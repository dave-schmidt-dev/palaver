"""Tests for the status bar's pure render function.

Two things are being defended here and they pull in opposite directions.

The first is that the function never raises, because it runs inside a
coroutine iTerm2 schedules and an exception there is invisible: the bar
simply stops updating. So the tests push hostile inputs at it -- a
400-character task, a width of one, ANSI escapes, a task that is nothing but
whitespace -- and assert a string comes back.

The second is that "never raises" is not satisfied by returning `""`. A
function that swallows everything passes the first set of tests perfectly and
is useless. So every input above also has an assertion about what the string
actually *says*, and the width arithmetic is checked exhaustively rather than
at a couple of sampled points.
"""

from __future__ import annotations

import unicodedata

import pytest

from palaver.observer.signals import Status
from palaver.ui.render import (
    ELLIPSIS,
    SEPARATOR,
    STATUS_LABELS,
    one_line,
    render,
)

#: Longer than any width under test, so it always forces the overflow path.
LONG_TASK = "wire the pane join into the status bar " * 12

#: The width the plan names, and the interesting one: it fits some status
#: words whole, cuts into others, and leaves room for no task at all.
NARROW = 8


# --- the done-when clauses ---------------------------------------------------


def test_every_status_in_the_enum_renders_a_non_empty_string():
    """The bar must have something to say about every status that exists.

    Asserted over `Status` itself rather than over a list written here, so
    that adding a member to the enum -- which is exactly what task 5.2 did
    with `IDLE` -- fails this test instead of quietly rendering nothing.
    """
    for status in Status:
        rendered = render(status, "a task", width=40)

        assert rendered, f"{status.name} rendered empty"
        assert rendered.strip() == rendered, f"{status.name} rendered with loose padding"

    # Positive control: the strings are distinct, so "non-empty for every
    # member" is not being satisfied by one constant returned nine times.
    distinct = {render(status, "a task", width=40) for status in Status}
    assert len(distinct) == len(list(Status))


def test_a_four_hundred_character_task_fills_exactly_the_requested_width():
    """Real task strings are unbounded; the width is the only thing that is.

    A session's task text comes from extraction over session prose and has no
    length limit. This is the case where an unguarded f-string hands iTerm2 a
    400-character line.
    """
    task = "x" * 400
    assert len(task) == 400

    rendered = render(Status.WORKING, task, width=40)

    assert len(rendered) == 40
    assert rendered.endswith(ELLIPSIS)
    assert rendered.startswith(STATUS_LABELS[Status.WORKING])


def test_a_narrow_width_truncates_rather_than_raising():
    """Width 8 is narrower than several status words and every full line."""
    rendered = render(Status.WORKING, LONG_TASK, width=NARROW)

    assert len(rendered) == NARROW
    assert rendered == "working" + ELLIPSIS

    # Positive control: the same call at a generous width is not truncated,
    # so the clipping above is the width doing it and not the function
    # clipping unconditionally.
    roomy = render(Status.WORKING, "short", width=NARROW * 10)
    assert not roomy.endswith(ELLIPSIS)
    assert roomy == "working" + SEPARATOR + "short"


def test_a_narrow_width_below_the_status_word_still_returns_something():
    """`needs you` is nine characters and does not fit the narrow case."""
    label = STATUS_LABELS[Status.WAITING_FOR_USER]
    assert len(label) > NARROW

    rendered = render(Status.WAITING_FOR_USER, LONG_TASK, width=NARROW)

    assert len(rendered) == NARROW
    assert rendered == label[: NARROW - 1] + ELLIPSIS


# --- the width arithmetic ----------------------------------------------------


def test_overflow_always_fills_the_width_and_fitting_content_is_never_padded():
    """The whole contract, checked over every status at every small width.

    Sampling two or three widths misses the boundaries that actually break --
    `width == len(label)`, `width == len(label) + 1`, and `width == 1` -- and
    those boundaries move whenever a label is reworded.
    """
    for status in Status:
        label = STATUS_LABELS[status]
        for width in range(1, 80):
            overflowing = render(status, LONG_TASK, width=width)
            assert len(overflowing) == width, (
                f"{status.name} at width {width} returned {len(overflowing)} "
                f"characters: {overflowing!r}"
            )

            fitting = render(status, None, width=width)
            assert len(fitting) <= width, f"{status.name} at width {width} overran"
            if width >= len(label):
                assert fitting == label, (
                    f"{status.name} at width {width} had room for its label "
                    f"but rendered {fitting!r}"
                )


def test_content_that_exactly_fills_the_width_is_left_alone():
    """The boundary between "fits" and "must be cut" is a `<=`, not a `<`.

    Every other test here renders content that is either comfortably short or
    hugely too long, and both pass whichever comparison is written. Only the
    exact fit tells them apart, and getting it wrong spends a character on an
    ellipsis announcing that nothing was omitted.
    """
    for status in Status:
        label = STATUS_LABELS[status]
        task = "abcdefghij"
        body = label + SEPARATOR + task
        exact = len(body)

        assert render(status, task, width=exact) == body
        assert ELLIPSIS not in render(status, task, width=exact)

        # Controls on either side: one character more changes nothing, one
        # character less must clip. Together they place the boundary.
        assert render(status, task, width=exact + 1) == body
        clipped = render(status, task, width=exact - 1)
        assert clipped != body
        assert clipped.endswith(ELLIPSIS)
        assert len(clipped) == exact - 1


def test_the_status_word_survives_whole_at_exactly_its_own_width():
    """The one place an ellipsis would cost more than it reports.

    At `width == len(label)` there is room for the word or for the word minus
    a character plus a marker. The word wins.
    """
    for status in Status:
        label = STATUS_LABELS[status]

        rendered = render(status, LONG_TASK, width=len(label))

        assert rendered == label
        assert ELLIPSIS not in rendered

        # Positive control: one character more and the task starts to appear,
        # so the label is not simply always returned regardless of width.
        wider = render(status, LONG_TASK, width=len(label) + 1)
        assert wider == label + ELLIPSIS


def test_a_width_below_one_raises_rather_than_returning_empty():
    """A zero width is an arithmetic bug upstream, not a rendering request.

    Returning `""` would make it look like a session with nothing to say and
    the real fault would be hunted in the observer instead.
    """
    for width in (0, -1, -100):
        with pytest.raises(ValueError, match="at least 1"):
            render(Status.WORKING, "a task", width=width)

    # Positive control: one is a legal width and renders.
    assert render(Status.WORKING, "a task", width=1) == ELLIPSIS


# --- hostile task text -------------------------------------------------------


def test_control_characters_never_reach_the_bar():
    """An escape sequence in a task string is a terminal being handed input.

    Agents echo terminal output into their own transcripts, so colour codes
    do appear in the text extraction reads. INV-2 says Palaver never controls
    an observed session; a component that forwards `\\x1b[` sequences to the
    terminal drawing it is not honouring that in spirit.
    """
    hostile = "red \x1b[31mtext\x1b[0m and a bell \x07 and a null \x00"

    rendered = render(Status.WORKING, hostile, width=200)

    assert "\x1b" not in rendered
    assert "\x07" not in rendered
    assert "\x00" not in rendered
    assert not any(unicodedata.category(ch) == "Cc" for ch in rendered)

    # Positive control: the readable text survives, so the sanitizer is
    # removing control characters rather than gutting the string.
    assert "red" in rendered
    assert "text" in rendered


def test_a_multiline_task_becomes_one_line():
    """The bar is one row; a newline in the text is not a rendering option."""
    rendered = render(Status.WORKING, "first line\nsecond line\n\n  third  ", width=200)

    assert "\n" not in rendered
    assert rendered == "working" + SEPARATOR + "first line second line third"


def test_a_task_of_only_whitespace_reads_as_no_task_at_all():
    """Whitespace-only text must not leave a dangling separator on the bar."""
    for blank in ("", "   ", "\n\n", "\t \r\n", None):
        rendered = render(Status.BLOCKED, blank, width=40)

        assert rendered == STATUS_LABELS[Status.BLOCKED]
        assert SEPARATOR.strip() not in rendered

    # Positive control: actual text does produce the separator.
    assert SEPARATOR in render(Status.BLOCKED, "on a decision", width=40)


def test_one_line_collapses_without_touching_ordinary_text():
    """The sanitizer is used on every task string, so it gets its own test."""
    assert one_line(None) == ""
    assert one_line("  spaced   out  ") == "spaced out"
    assert one_line("a\tb\nc") == "a b c"

    # Positive control: text that is already clean comes back untouched, so
    # the collapse is not rewriting every string it sees.
    assert one_line("already clean text") == "already clean text"


# --- the label table ---------------------------------------------------------


def test_every_status_has_a_distinct_label_and_no_label_is_orphaned():
    """The table is written by hand, so its shape is asserted rather than
    assumed. The import-time guard in the module catches a missing label; this
    catches a duplicated one, which the guard cannot see."""
    assert set(STATUS_LABELS) == set(Status)

    labels = list(STATUS_LABELS.values())
    assert len(set(labels)) == len(labels), f"duplicate status label in {labels}"
    assert all(label and label.strip() == label for label in labels)


def test_no_rendered_character_is_double_width():
    """The bar's own characters must occupy one cell, whatever the terminal.

    Task 5.5 adds a ladybug glyph, and `🐞` is `east_asian_width == "W"`.
    This is the test that fails if it is ever dropped into `STATUS_LABELS`
    rather than kept as the separate failure indicator it is meant to be.
    Task text is excluded because a session's own prose may legitimately be
    CJK; what is asserted is the chrome this module chooses.
    """
    chrome = "".join(STATUS_LABELS.values()) + SEPARATOR + ELLIPSIS

    wide = [ch for ch in chrome if unicodedata.east_asian_width(ch) in ("W", "F")]
    assert wide == [], f"double-width characters in the bar's own chrome: {wide}"

    # Positive control: the check can see a wide character when there is one.
    assert unicodedata.east_asian_width("🐞") == "W"
