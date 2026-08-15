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

Task 5.5 adds two more sections, per the plan's own file list: the ladybug
glyph and the freshness horizon, and then `palaver ui --selftest`, which is
the Phase 5 gate's second half and is therefore tested here beside the thing
it gates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import types
import unicodedata

import pytest

from palaver.cli import ui
from palaver.observer.signals import Status
from palaver.ui import component, publisher
from palaver.ui.component import (
    LADYBUG,
    PUSHED_AT_KEY,
    STALE_AFTER,
    UPDATE_CADENCE,
    RenderTicker,
    build_component,
    decode_status,
    encode_status,
    is_fresh,
    line_for,
    push_status,
    render_or_ladybug,
)
from palaver.ui.connection import NoSocketTransportError
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


# --- task 5.5: the glyph, and a status that stops being believed ---
#
# Two failures, and they are not the same failure.
#
# The first is loud: something raised where iTerm2 could see it. That is what
# the glyph is for, and the tests below drive it by breaking `line_for` under
# the coroutine rather than by asserting the string constant exists.
#
# The second is silent, and it is the one worth the trouble. Kill the daemon
# and nothing raises anywhere — the variable simply keeps its last value and
# the bar goes on reporting `working: reading a file` at a pane where nothing
# has happened for an hour. No exception handler reaches that, because
# nothing failed. Only a freshness stamp does, so most of what follows is
# about the stamp.

#: A fixed epoch second, so every age below is arithmetic and no test sleeps.
NOW = 1_800_000_000.0

PANE = "w0t0p0:CF60A48E-0000-4000-8000-000000000002"


class _Recorder:
    """A `SetVariable` that records, and optionally refuses."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    async def __call__(self, session_id, name, value):
        self.calls.append((session_id, name, value))
        if self.fail:
            raise RuntimeError("iTerm2 refused the write")


def _boom(*args, **kwargs):
    raise RuntimeError("render fell over")


def _guarded(raw, *, now=NOW, fail_tick=False):
    """Run one render through the boundary iTerm2 calls."""
    writes = _Recorder(fail=fail_tick)
    line = asyncio.run(
        render_or_ladybug(PANE, raw, ticker=RenderTicker(), set_variable=writes, now=now)
    )
    return line, writes


def _stub_iterm2(monkeypatch):
    """The three names `build_component` reaches for, and nothing else."""
    module = types.SimpleNamespace(
        Reference=lambda name: None,
        StatusBarRPC=lambda func: func,
        StatusBarComponent=lambda **kwargs: types.SimpleNamespace(kwargs=kwargs),
    )
    monkeypatch.setattr(component, "import_iterm2", lambda: module)
    return module


def test_a_render_that_raises_shows_the_ladybug_glyph(monkeypatch):
    monkeypatch.setattr(component, "line_for", _boom)
    line, writes = _guarded(encode_status(Status.WORKING, "reading a file", now=NOW))
    assert line == LADYBUG
    assert writes.calls == [], "no render tick is claimed for a render that did not happen"


def test_the_glyph_is_absent_when_the_render_works():
    """Positive control for the test above: the fault is what produces it."""
    line, writes = _guarded(encode_status(Status.WORKING, "reading a file", now=NOW))
    assert line == "working: reading a file"
    assert LADYBUG not in line
    assert len(writes.calls) == 1


def test_no_line_is_cached_behind_the_glyph(monkeypatch):
    """A failed render returns the glyph and not the last good line.

    Falling back on a remembered line would be the stale-value failure
    arriving by a second route, and it would be worse than the first: the
    line would be a real status, so nothing on screen would look wrong.
    """
    ticker, writes = RenderTicker(), _Recorder()
    payload = encode_status(Status.WORKING, "reading a file", now=NOW)

    def once():
        return asyncio.run(
            render_or_ladybug(PANE, payload, ticker=ticker, set_variable=writes, now=NOW)
        )

    assert once() == "working: reading a file"
    monkeypatch.setattr(component, "line_for", _boom)
    assert once() == LADYBUG
    monkeypatch.undo()
    assert once() == "working: reading a file", "and it recovers rather than latching"


def test_the_glyph_is_never_a_status_label():
    """It means "this component is broken", which no pane state can mean."""
    assert LADYBUG not in STATUS_LABELS.values()
    for status in Status:
        assert LADYBUG not in render(status, None, 40)
        assert LADYBUG not in render(status, "a task", 40)


def test_a_refused_tick_write_does_not_show_the_glyph():
    """The tick is observability; losing it is not a failure worth showing."""
    line, writes = _guarded(
        encode_status(Status.BLOCKED, "waiting on a lock", now=NOW), fail_tick=True
    )
    assert line == "blocked: waiting on a lock"
    assert len(writes.calls) == 1


def test_the_glyph_is_logged_with_a_traceback_before_it_is_shown(monkeypatch, caplog):
    """The glyph is all the user gets, so the cause has to be somewhere."""
    monkeypatch.setattr(component, "line_for", _boom)
    with caplog.at_level(logging.ERROR, logger="palaver.ui.component"):
        line, _ = _guarded(None)
    assert line == LADYBUG
    assert "render fell over" in caplog.text
    assert "Traceback" in caplog.text


def test_the_component_coroutine_is_the_one_that_falls_back(monkeypatch):
    """`build_component` must wire the guarded path, not the raising one.

    Both are async, both take the same arguments, and swapping them back
    changes nothing any other test in this file would notice.
    """
    _stub_iterm2(monkeypatch)
    writes = _Recorder()
    _, coro = build_component(object(), set_variable=writes)
    payload = encode_status(Status.QUESTION, "which branch?")

    assert asyncio.run(coro(None, payload, PANE)) == "question: which branch?"
    monkeypatch.setattr(component, "line_for", _boom)
    assert asyncio.run(coro(None, payload, PANE)) == LADYBUG


def test_a_status_nobody_refreshed_stops_being_shown():
    payload = encode_status(Status.WORKING, "reading a file", now=NOW)
    assert decode_status(payload, now=NOW + 1.0) == (Status.WORKING, "reading a file")
    assert decode_status(payload, now=NOW + STALE_AFTER + 1.0) == (Status.UNKNOWN, None)


def test_a_stale_status_takes_its_task_text_with_it():
    """`unknown: reading a file` is still a confident claim about the pane."""
    payload = encode_status(Status.WORKING, "reading a file", now=NOW)
    line = line_for(payload, now=NOW + STALE_AFTER + 1.0)
    assert line == "unknown"
    assert "reading" not in line


def test_a_dead_daemons_last_status_degrades_through_the_render_path():
    """End to end: the push that was never followed by another one."""
    writes = _Recorder()
    payload = asyncio.run(push_status(writes, PANE, Status.WORKING, "reading a file", now=NOW))
    assert _guarded(payload, now=NOW + 1.0)[0] == "working: reading a file"
    assert _guarded(payload, now=NOW + STALE_AFTER + 1.0)[0] == "unknown"


def test_a_stale_pane_is_noticed_without_anyone_pushing_anything():
    """The cadence re-render is what withdraws the claim, not a new push.

    Nothing is written in this test at all — the same bytes are re-read on
    each cadence tick, and the answer changes because the clock moved. That
    is the whole reason `UPDATE_CADENCE` is a number rather than `None`.
    """
    assert UPDATE_CADENCE > 0
    payload = encode_status(Status.WORKING, "reading a file", now=NOW)
    ticks = int(STALE_AFTER // UPDATE_CADENCE) + 1
    lines = [line_for(payload, now=NOW + UPDATE_CADENCE * n) for n in range(1, ticks + 1)]
    assert lines[0] == "working: reading a file", "one missed push is not death"
    assert lines[-1] == "unknown"


def test_the_freshness_horizon_is_several_cadence_ticks():
    """Short enough to be honest, long enough not to flicker on a busy Mac.

    Both bounds matter and only one of them is obvious. Without the upper
    bound, a horizon of a day satisfies every other test in this file --
    they all express staleness as `STALE_AFTER + 1`, so they scale with the
    constant and can never notice it growing.
    """
    assert STALE_AFTER >= 2 * UPDATE_CADENCE
    assert STALE_AFTER <= 5 * UPDATE_CADENCE, "a horizon this long is not a horizon"
    assert is_fresh(NOW, now=NOW + UPDATE_CADENCE)


def test_the_horizon_edge_is_exact():
    assert is_fresh(NOW, now=NOW + STALE_AFTER)
    assert not is_fresh(NOW, now=NOW + STALE_AFTER + 0.001)


def test_a_payload_from_a_build_that_did_not_stamp_is_not_believed():
    """An unstamped payload has an unknowable age, which is not "fresh".

    The cost is one blank render right after upgrading a running daemon; the
    alternative is that a pre-upgrade value never expires at all.
    """
    old = json.dumps({"status": "WORKING", "task": "reading a file"})
    assert decode_status(old, now=NOW) == (Status.UNKNOWN, None)
    assert line_for(old, now=NOW) == "unknown"
    assert PUSHED_AT_KEY not in json.loads(old)


def test_a_stamp_that_is_not_a_number_is_not_believed():
    """`True` is in here on purpose: `isinstance(True, int)` is `True`.

    The second assertion is the only clock at which the bool check is
    observable -- at any plausible `now`, `True` is a stamp from 1970 and
    fails the horizon anyway. It is asserted at epoch 1 not because a machine
    could have that clock, but because otherwise nothing distinguishes a
    type check from arithmetic that happens to agree with it.
    """
    for stamp in (None, True, False, "soon", [NOW], {}, ""):
        assert not is_fresh(stamp, now=NOW), f"{stamp!r} was accepted as a timestamp"
    assert not is_fresh(True, now=1.0), "a bool is not a timestamp at any clock"
    assert is_fresh(1.0, now=1.0), "the positive control for the line above"


def test_a_stamp_from_the_future_is_not_believed():
    """A clock jump must not pin a pane to a status that can never expire."""
    assert not is_fresh(NOW + STALE_AFTER + 60.0, now=NOW)
    assert is_fresh(NOW + 1.0, now=NOW), "a second of skew is not a failure"


# --- task 5.5: `palaver ui --selftest`, the Phase 5 gate's second half ---
#
# The command's whole job is to be honest about a machine it cannot control,
# so the interesting cases are the ones where it must *not* fail: no cookie,
# no socket, a profile nobody has configured. Each of those is a skip with a
# reason, and a skip must never be counted as a pass -- which is what most of
# what follows checks.
#
# `run_checks` is driven against a fake iTerm2 here rather than being left to
# the live test alone, because it is the function the phase gate's exit code
# comes from and a live-only test cannot assert the failure paths at all.

PROPS_CONFIGURED = {
    ui.component.LAYOUT_KEY: {"components": [{"identifier": ui.component.IDENTIFIER}]},
    ui.component.SHOW_BAR_KEY: 1,
    ui.component.ORIGINAL_GUID_KEY: "F25B986F-AEEA-4438-A22D-B79D193A0FB0",
}
PROPS_BARE = {ui.component.SHOW_BAR_KEY: 0, ui.component.ORIGINAL_GUID_KEY: "some-profile"}


class _FakeSession:
    """A pane whose variables live in a dict, as iTerm2's do on its side."""

    def __init__(self, session_id, store, props):
        self.session_id = session_id
        self.store = store
        self.props = props

    async def async_get_variable(self, name):
        return self.store.get(name)

    async def async_get_profile(self):
        return types.SimpleNamespace(all_properties=self.props)


def _fake_iterm2(monkeypatch, session, *, dropped=False, drop_tick=False, stall_tick_after=False):
    """Stub every iTerm2 entry point `run_checks` and `register` reach for."""

    async def async_get_app(connection):
        return types.SimpleNamespace(
            current_terminal_window=types.SimpleNamespace(
                current_tab=types.SimpleNamespace(current_session=session)
            ),
            get_session_by_id=lambda sid: session if sid == session.session_id else None,
        )

    class _Component:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def async_register(self, connection, coro, timeout=None):
            self.registered = (coro, timeout)

    module = types.SimpleNamespace(
        async_get_app=async_get_app,
        Reference=lambda name: None,
        StatusBarRPC=lambda func: func,
        StatusBarComponent=lambda **kwargs: _Component(**kwargs),
    )

    async def writer(session_id, name, value):
        # `drop_tick` refuses the tick writes that come *from the render
        # path*, while letting the selftest's own baseline through. That is
        # not a hypothetical split: `render_for_session` catches and logs a
        # failed tick write by design (task 5.3, so a lost tick never costs
        # the user the line), which means a render can complete normally with
        # the variable never updated. The value is what tells the two apart.
        if dropped or (drop_tick and name == component.TICK_VARIABLE and value != ui.TICK_BASELINE):
            return
        if stall_tick_after and name == component.TICK_VARIABLE and value == ui.RENDERS:
            return
        session.store[name] = value

    async def read_variables(_pane_id):
        # Task 5.6's publish check runs over the same fake connection. A pane
        # iTerm2 will not describe is the ordinary case for a stub, and the
        # publisher answers it `UNKNOWN` -- which is what this check is about
        # anyway: that something wrote, not what it decided.
        return None

    monkeypatch.setattr(ui, "import_iterm2", lambda: module)
    monkeypatch.setattr(component, "import_iterm2", lambda: module)
    monkeypatch.setattr(component, "make_variable_writer", lambda connection: writer)
    monkeypatch.setattr(publisher, "make_variables_reader", lambda connection: read_variables)
    return module


def _checks(
    monkeypatch,
    *,
    props=PROPS_BARE,
    store=None,
    dropped=False,
    drop_tick=False,
    stall_tick_after=False,
):
    session = _FakeSession(PANE, {} if store is None else store, props)
    _fake_iterm2(
        monkeypatch,
        session,
        dropped=dropped,
        drop_tick=drop_tick,
        stall_tick_after=stall_tick_after,
    )
    return asyncio.run(ui.run_checks(object())), session


def _named(checks, name):
    return next(check for check in checks if check.name == name)


def test_the_ui_subcommand_is_registered():
    from palaver.cli import SUBCOMMANDS, build_parser

    assert ui in SUBCOMMANDS
    args = build_parser().parse_args(["ui", "--selftest"])
    assert args.handler is ui.run and args.selftest


def test_ui_with_no_flags_does_nothing_and_says_so(capsys):
    args = argparse.Namespace(selftest=False, enable_status_bar=False, disable_status_bar=False)
    assert ui.run(args, on_status=lambda _m: None) == 2
    assert "nothing to do" in capsys.readouterr().out


def test_an_unconfigured_profile_is_reported_and_not_failed(monkeypatch):
    """The default state of a fresh machine must not fail the phase gate."""
    checks, _ = _checks(monkeypatch, props=PROPS_BARE)
    layout = _named(checks, "layout")
    assert layout.passed is None
    assert "Configure Status Bar" in layout.detail
    assert ui.exit_code(checks) == 0


def test_a_configured_profile_reports_that_iterm_dispatches_it(monkeypatch):
    """Positive control for the test above: the skip is about the profile."""
    checks, _ = _checks(monkeypatch, props=PROPS_CONFIGURED)
    assert _named(checks, "layout").passed is True
    assert _named(checks, "dispatched by iTerm2").passed is True
    assert ui.exit_code(checks) == 0


def test_the_selftest_proves_the_variables_round_trip(monkeypatch):
    checks, _ = _checks(monkeypatch)
    assert _named(checks, "register").passed is True
    assert _named(checks, "variables").passed is True
    assert _named(checks, "render tick").passed is True
    assert _named(checks, "render").passed is True


def test_a_variable_that_does_not_round_trip_fails_the_selftest(monkeypatch):
    """The negative control: a writer iTerm2 silently ignores."""
    checks, _ = _checks(monkeypatch, dropped=True)
    assert _named(checks, "variables").passed is False
    assert ui.exit_code(checks) == 1


def test_a_tick_left_by_an_earlier_run_does_not_fail_the_selftest(monkeypatch):
    """The bug the live gate caught on its second run, in one line.

    `RenderTicker` counts within one process and starts at zero, so the
    second `palaver ui --selftest` of the day found the 2 the first one left
    behind, counted 1 and 2 again, and reported that the tick had not
    advanced -- against a component that was working perfectly.
    """
    store = {component.TICK_VARIABLE: 99}
    checks, session = _checks(monkeypatch, store=store)
    assert _named(checks, "render tick").passed is True
    assert session.store[component.TICK_VARIABLE] == ui.RENDERS
    assert ui.exit_code(checks) == 0
    assert ui.RENDERS >= 2, "one render cannot show that anything was counted"


def test_a_leftover_tick_equal_to_the_render_count_is_not_mistaken_for_a_pass(monkeypatch):
    """Why the baseline is written rather than relying on the count alone.

    The leftover value on this machine is exactly 2, which is exactly the
    number of renders the selftest performs. Without a baseline, a run whose
    tick writes were all silently dropped would read back that 2 and call it
    a pass -- the one coincidence that makes the check meaningless.
    """
    store = {component.TICK_VARIABLE: ui.RENDERS}
    checks, _ = _checks(monkeypatch, store=store, drop_tick=True)
    assert _named(checks, "render tick").passed is False
    assert ui.exit_code(checks) == 1


def test_a_tick_that_stalls_partway_fails_the_selftest(monkeypatch):
    """A bar that updated once and then stopped is the realistic failure.

    `render_for_session` catches a failed tick write on purpose, so a render
    can complete and return its line while the variable stays behind. The
    check therefore has to compare the exact count, not merely notice that
    the value is no longer the baseline.
    """
    checks, session = _checks(monkeypatch, stall_tick_after=True)
    assert session.store[component.TICK_VARIABLE] == ui.RENDERS - 1
    assert _named(checks, "render tick").passed is False
    assert ui.exit_code(checks) == 1


def test_a_tick_that_does_not_advance_fails_the_selftest(monkeypatch):
    """The tick is the only evidence the render path ran end to end.

    A component that never renders is exactly the failure this whole command
    exists to catch, and it looks like success from every other angle: the
    status variable round-trips, registration succeeded, nothing raised.
    """
    checks, _ = _checks(monkeypatch, drop_tick=True)
    assert _named(checks, "variables").passed is True, "only the tick should be wrong"
    assert _named(checks, "render tick").passed is False
    assert component.TICK_VARIABLE in _named(checks, "render tick").detail
    assert ui.exit_code(checks) == 1


def test_a_render_that_falls_back_to_the_glyph_fails_the_selftest(monkeypatch):
    """This is what ties the glyph to the gate: a broken bar is not a pass."""
    monkeypatch.setattr(component, "line_for", _boom)
    checks, _ = _checks(monkeypatch)
    render_check = _named(checks, "render")
    assert render_check.passed is False
    assert LADYBUG in render_check.detail
    assert ui.exit_code(checks) == 1


def test_the_selftest_restores_the_panes_previous_status(monkeypatch):
    """Leaving the probe value behind would create the stale status 5.5 detects."""
    previous = encode_status(Status.WORKING, "what the daemon last said")
    store = {component.STATUS_VARIABLE: previous}
    checks, session = _checks(monkeypatch, store=store)
    assert ui.exit_code(checks) == 0
    assert session.store[component.STATUS_VARIABLE] == previous


def test_the_probe_value_is_restored_even_when_a_check_fails(monkeypatch):
    monkeypatch.setattr(component, "line_for", _boom)
    previous = encode_status(Status.WORKING, "what the daemon last said")
    _, session = _checks(monkeypatch, store={component.STATUS_VARIABLE: previous})
    assert session.store[component.STATUS_VARIABLE] == previous


def test_the_selftest_never_switches_the_status_bar_on(monkeypatch):
    """It reports the bar is off. Turning it on is `--enable-status-bar`."""

    async def _refuse(*args, **kwargs):
        raise AssertionError("the selftest must not change how panes look")

    monkeypatch.setattr(component, "show_status_bar", _refuse)
    checks, _ = _checks(monkeypatch, props=PROPS_BARE)
    assert "Status bar enabled" in _named(checks, "layout").detail


def test_the_selftest_proves_something_writes_to_the_pane(monkeypatch):
    """Task 5.6's check, and the one whose absence let the gap ship.

    Every other check here proves a pane *can* be written to. This one runs
    the production publisher over the same connection, which is the only
    check that fails if nothing writes.
    """
    checks, _ = _checks(monkeypatch)
    publish = _named(checks, "publish")
    assert publish.passed is True
    assert publish.mark == "ok"


def test_a_publisher_that_writes_nothing_fails_the_selftest(monkeypatch):
    """The negative control for the check above.

    A `PanePush` with no payload is exactly what a refused write produces,
    and it is indistinguishable from a successful one by status alone.
    """

    async def wrote_nothing(pane_ids, **kwargs):
        return tuple(
            publisher.PanePush(pane_id=pane_id, status=Status.WORKING, task=None, payload=None)
            for pane_id in pane_ids
        )

    monkeypatch.setattr(publisher, "publish_once", wrote_nothing)
    checks, _ = _checks(monkeypatch)
    assert _named(checks, "publish").passed is False
    assert ui.exit_code(checks) == 1


def test_the_status_bar_can_be_switched_back_off(monkeypatch):
    """The reversal for `--enable-status-bar`.

    A change that only Palaver can make and only iTerm2's preferences can
    undo is a change a user cannot try, so both directions go through one
    function and this asserts the flag reaches it.
    """
    asked = []

    async def record(profile, *, shown):
        asked.append(shown)

    class _Profile:
        guid = "profile-1"

        async def async_get_full_profile(self):
            return self

    module = types.SimpleNamespace(
        PartialProfile=types.SimpleNamespace(async_query=lambda connection: _resolved([_Profile()]))
    )
    monkeypatch.setattr(ui, "import_iterm2", lambda: module)
    monkeypatch.setattr(component, "show_status_bar", record)

    off = asyncio.run(ui.set_status_bar(object(), shown=False, on_status=lambda _m: None))
    on = asyncio.run(ui.set_status_bar(object(), shown=True, on_status=lambda _m: None))

    assert asked == [False, True]
    assert "switched off" in off[0].detail
    assert "switched on" in on[0].detail


def _resolved(value):
    """Wrap a value in an awaitable, for a stub whose real form is async."""

    async def wait():
        return value

    return wait()


def test_the_two_status_bar_flags_cannot_be_given_together(capsys):
    """Opposite instructions in one invocation are refused, not silently ordered."""
    args = argparse.Namespace(
        selftest=False,
        enable_status_bar=True,
        disable_status_bar=True,
        session=None,
        width=40,
        no_ask_cookie=True,
    )
    assert ui.run(args) == 2
    assert "contradict" in capsys.readouterr().out


def test_a_skipped_check_is_never_counted_as_a_pass():
    """Asserted twice: in the summary, and on the line a human reads first.

    The summary is computed from `passed` and the line prefix from `mark`, so
    a change to either alone leaves the other still telling the truth. Only
    checking the summary would let every line print `ok` under a correct
    total.
    """
    checks = [ui.Check("a", None, "could not run"), ui.Check("b", None, "nor this")]
    report = ui.format_report(checks)
    assert ui.exit_code(checks) == 0
    assert "0 ok, 2 skipped" in report
    assert "could not run" in report and "nor this" in report
    assert [line.split()[0] for line in report.splitlines()[:2]] == ["skipped", "skipped"]
    assert ui.Check("c", True, "-").mark == "ok", "the positive control for the marks"


def test_a_check_that_ran_and_failed_fails_the_command():
    checks = [ui.Check("a", True, "fine"), ui.Check("b", False, "broken"), ui.Check("c", None, "-")]
    assert ui.exit_code(checks) == 1
    assert "1 ok, 1 failed, 1 skipped" in ui.format_report(checks)


def test_a_machine_with_no_socket_reports_the_reason_and_exits_zero(monkeypatch, capsys):
    """An unattachable machine is a skip: there is nothing to have failed."""

    def _no_socket(*args, **kwargs):
        raise NoSocketTransportError("iTerm2's API socket is not at /nowhere")

    monkeypatch.setattr(ui, "preflight", _no_socket)
    monkeypatch.setattr(ui, "ensure_cookie", lambda **kwargs: ui.Check("cookie", True, "set"))
    args = argparse.Namespace(
        selftest=True,
        enable_status_bar=False,
        disable_status_bar=False,
        session=None,
        width=40,
        no_ask_cookie=True,
    )
    assert ui.run(args, on_status=lambda _m: None) == 0
    out = capsys.readouterr().out
    assert "skipped" in out and "API socket" in out


def test_without_a_cookie_nothing_live_is_tried(monkeypatch, capsys):
    monkeypatch.delenv("ITERM2_COOKIE", raising=False)
    monkeypatch.setattr(ui, "preflight", lambda *a, **k: pytest.fail("must not attach"))
    args = argparse.Namespace(
        selftest=True,
        enable_status_bar=False,
        disable_status_bar=False,
        session=None,
        width=40,
        no_ask_cookie=True,
    )
    assert ui.run(args, on_status=lambda _m: None) == 0
    assert "no cookie" in capsys.readouterr().out


def test_the_cookie_is_never_printed(monkeypatch, capsys):
    """It is a credential. It goes in the environment and nowhere else."""
    secret = "a-cookie-that-must-not-appear"
    monkeypatch.setattr(ui, "request_cookie_and_key", lambda **kwargs: (secret, "key-material"))
    env = {}
    check = ui.ensure_cookie(ask=True, env=env)

    assert check.passed is True
    assert env["ITERM2_COOKIE"] == secret
    assert secret not in check.detail
    assert secret not in ui.format_report([check])
    assert secret not in capsys.readouterr().out
