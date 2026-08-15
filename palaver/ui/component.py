"""Register Palaver's status bar component, and check that it can be seen.

Task 5.3. The plan names three states -- registration, profile membership,
layout inclusion -- and says only the third puts pixels on screen. Measured
against the running app on 2026-08-15, there are **four**: `Show Status Bar`
is `0` on every live session here, so a layout that contains this component's
identifier still shows nothing at all. `check_layout` asserts both, because
the plan's own done-when is satisfiable while the bar is invisible.

**Palaver registers and checks; it does not write the layout entry.** The
`components` list in `Status Bar Layout` is empty in every profile on this
machine, so no example of an entry can be read back, and iTerm2's binary
shows the component holds a `_savedRegistrationRequest` -- the entry embeds a
serialized `ITMRPCRegistrationRequest`, not merely an identifier string.
Writing a profile key stores whatever it is given verbatim, so a guessed
entry reads back exactly as written whether or not iTerm2 can load it: the
readback cannot validate the guess, and only rendering can, which needs the
bar turned on in David's own windows. A gate that reports "not configured,
here is the fix" is honest; one that writes a hand-encoded protobuf into a
preferences file is not checkable. `LayoutCheck.remedy` carries the one-time
manual step.

**The profile identity is `Original Guid`, not `Guid` and not the name.**
Every live session's profile is divorced: its `Guid` is session-local and
absent from the shared profile list, while `Original Guid` names the shared
profile it came from. Three sessions here report the name `Default` with
three different guids, so the name is not an identity either. A write to the
shared profile was measured to propagate to already-divorced sessions, so the
shape is: configure the shared profile once, read back from the session.

**The render tick is counted in Python, never read back through a
`Reference`.** The coroutine is re-invoked whenever one of its referenced
variables changes. A coroutine that read `user.palaver_render_tick` as an
input and wrote it as its last act would re-trigger itself forever, at
whatever rate iTerm2 is willing to dispatch. `RenderTicker` keeps the counter
on this side of the connection, so the tick is write-only from iTerm2's point
of view and observable by anything that reads the variable.

**A status that cannot be refreshed must stop being shown (task 5.5).** The
variable outlives the process that wrote it: kill the daemon and
`user.palaver_status` keeps its last value in iTerm2 for as long as the pane
lives, so the bar goes on confidently reporting `working: reading a file`
hours after anything was reading anything. That is precisely the failure
INV-7 exists to prevent, and no amount of exception handling reaches it,
because nothing has failed -- there is simply nobody left to push. So every
payload carries the epoch second it was pushed, and a payload older than
`STALE_AFTER` decodes to `UNKNOWN` with **no task text**: `unknown: reading a
file` would still be a confident claim about what the pane is doing.
`UPDATE_CADENCE` is what makes this visible with no push at all -- the
component re-renders on its own timer, re-reads the same stale value, and
degrades it.

**The ladybug is the other half, for the failures that are failures.** iTerm2
shows 🐞 for a component that is not registered or whose RPC errored, and its
own troubleshooting page documents it as clickable for detail. Palaver
catches its render exceptions and returns the glyph itself, which trades that
popover away for a traceback in `.logs/palaver.log` and a return value a
headless test can assert. What Palaver does not catch -- a failed
registration, a dropped connection -- still gets iTerm2's own ladybug, so the
indicator means the same thing on both paths: this component is broken, not
this pane is idle.

Nothing here reads observed session content. It writes two `user.` variables
into panes Palaver already tracks, and reads profile properties (INV-9).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from palaver.observer.signals import Status
from palaver.ui.connection import import_iterm2
from palaver.ui.render import render

log = logging.getLogger(__name__)

#: Reverse-DNS, and stable for the life of the component. iTerm2 keys the
#: layout entry off this, so changing it orphans every profile already
#: configured with the old one.
IDENTIFIER = "com.zerodelta.palaver.status"

SHORT_DESCRIPTION = "Palaver status"
DETAILED_DESCRIPTION = "What the agent in this pane is doing, and whether it needs you."
#: Shown in iTerm2's component picker. A real line from `render`, so what the
#: user previews is what they get.
EXEMPLAR = "your turn: waiting on a decision"

#: The per-session variables. `user.` prefix is required by iTerm2 for
#: script-defined variables; it rejects any other namespace.
STATUS_VARIABLE = "user.palaver_status"
TICK_VARIABLE = "user.palaver_render_tick"

#: The `?` suffix marks the reference optional: iTerm2 passes null instead of
#: refusing to evaluate when the variable has never been set. That is the
#: cold-start case -- at first launch no pane has a status yet -- and without
#: it the component would fail to render at all until the first push.
STATUS_REFERENCE = f"{STATUS_VARIABLE}?"

#: A backstop, not the primary path: state changes are pushed with
#: `push_status`, which lands within the same tick they happen. A real number
#: rather than `None` so that a pane whose push was lost to a dropped
#: connection recovers on its own, and so "one write before the next cadence
#: tick" is a claim about something rather than vacuously true.
UPDATE_CADENCE = 30.0

#: How long a pushed status stays believable without being refreshed. Three
#: cadence ticks: one missed render is a busy machine, three in a row is
#: nobody home. Derived from `UPDATE_CADENCE` rather than chosen separately,
#: so raising the cadence cannot silently make the horizon too tight to ever
#: be met.
STALE_AFTER = 3.0 * UPDATE_CADENCE

#: What the bar shows when Palaver itself is broken, matching the glyph
#: iTerm2 shows for a component it cannot reach. Never a `Status` label: it
#: means "this component failed", and a status word would make it compete
#: with the nine things the pane could legitimately be doing.
LADYBUG = "🐞"

#: Characters, matching `render`'s contract. iTerm2 sizes components in
#: points and the maximum width is a user knob, so this is a determinism
#: budget rather than a fit.
DEFAULT_WIDTH = 40

#: Profile keys, spelled exactly as iTerm2 spells them.
LAYOUT_KEY = "Status Bar Layout"
SHOW_BAR_KEY = "Show Status Bar"
ORIGINAL_GUID_KEY = "Original Guid"
GUID_KEY = "Guid"

#: What to tell a user whose profile is not configured. One-time and manual,
#: for the reason in the module docstring.
LAYOUT_REMEDY = (
    "add it in iTerm2 > Settings > Profiles > Session > Configure Status Bar, "
    f"where it is listed as {SHORT_DESCRIPTION!r}"
)
SHOW_BAR_REMEDY = (
    "turn the status bar on in iTerm2 > Settings > Profiles > Session > Status bar enabled"
)

#: The payload's freshness stamp: absolute epoch seconds from `time.time()`,
#: never a monotonic clock and never a formatted timestamp. The value crosses
#: a process boundary -- one process writes it, another reads it, possibly
#: after a restart -- and a monotonic clock is meaningless outside the process
#: that read it.
PUSHED_AT_KEY = "pushed_at"

#: An async `(session_id, name, value) -> None`. Injected rather than reached
#: for, so every path through this module is testable without iTerm2 running.
SetVariable = Callable[[str, str, Any], Awaitable[None]]


def encode_status(status: Status, task: str | None = None, *, now: float | None = None) -> str:
    """Serialize what a pane should show into the value of one variable.

    Args:
        status: The pane's derived status.
        task: What it is doing, if anything is known.
        now: Epoch seconds to stamp the payload with, defaulting to the
            current time. A parameter so tests can age a payload without
            sleeping, not so callers can choose a clock.

    Returns:
        A JSON object string. JSON rather than a bare word because the task
        text is arbitrary session-derived prose and a delimiter-joined pair
        would break on the first colon or newline in it -- and because the
        freshness stamp needs somewhere to live.
    """
    return json.dumps(
        {
            "status": status.name,
            "task": task,
            PUSHED_AT_KEY: time.time() if now is None else float(now),
        },
        ensure_ascii=False,
    )


def is_fresh(stamp: Any, *, now: float | None = None, horizon: float = STALE_AFTER) -> bool:
    """Report whether a payload's stamp is recent enough to still believe.

    An absent or unreadable stamp is **not** fresh. Nothing but `push_status`
    writes this variable and it always stamps, so the only way to see one
    without a stamp is a payload left behind by an older build -- exactly the
    case where its age is unknowable and could be days. One blank render
    immediately after upgrading a running daemon is the cost, and the next
    push clears it.

    A stamp from the future fails too, by the same absolute bound. Clock
    skew large enough to matter would otherwise pin a pane to a status that
    can never expire.

    Args:
        stamp: The payload's `pushed_at` value, or anything else.
        now: Epoch seconds to compare against, defaulting to the current time.
        horizon: How many seconds of age is still believable.

    Returns:
        True if the stamp is a real number within `horizon` seconds of `now`.
    """
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return False
    return abs((time.time() if now is None else now) - float(stamp)) <= horizon


def decode_status(raw: Any, *, now: float | None = None) -> tuple[Status, str | None]:
    """Read back what `encode_status` wrote, tolerating anything else.

    Total by construction. Every caller is inside a status bar coroutine, and
    a raise there goes to a log nobody is reading while the bar silently
    stops updating, so an unreadable value has to degrade to a renderable one
    instead.

    A stale payload degrades the same way an unreadable one does, and loses
    its task text with it. See the module docstring: `unknown` is the honest
    reading of a value nobody is refreshing, and `unknown: reading a file`
    would keep making the claim this is meant to withdraw.

    Args:
        raw: The variable's value: the JSON string, `None` before anything
            has ever been pushed, or whatever a stale build left behind.
        now: Epoch seconds to judge freshness against, defaulting to the
            current time.

    Returns:
        A `(status, task)` pair, falling back to `(Status.UNKNOWN, None)`.
    """
    if not raw:
        return Status.UNKNOWN, None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except TypeError, ValueError:
            return Status.UNKNOWN, None
    if not isinstance(raw, Mapping):
        return Status.UNKNOWN, None
    if not is_fresh(raw.get(PUSHED_AT_KEY), now=now):
        return Status.UNKNOWN, None

    name = raw.get("status")
    try:
        status = Status[name] if isinstance(name, str) else Status.UNKNOWN
    except KeyError:
        # A status this build does not have. Renaming a member is exactly the
        # kind of change that would otherwise blank every bar mid-upgrade.
        status = Status.UNKNOWN

    task = raw.get("task")
    return status, task if isinstance(task, str) else None


def line_for(raw: Any, width: int = DEFAULT_WIDTH, *, now: float | None = None) -> str:
    """Render the line a pane should show, from the raw variable value.

    Args:
        raw: As `decode_status`.
        width: Character budget. Clamped rather than validated, for the same
            reason `decode_status` is total.
        now: Epoch seconds to judge freshness against.

    Returns:
        One line of text, always.
    """
    status, task = decode_status(raw, now=now)
    return render(status, task, max(1, width))


async def push_status(
    set_variable: SetVariable,
    session_id: str,
    status: Status,
    task: str | None = None,
    *,
    now: float | None = None,
) -> str:
    """Push a pane's new state, as exactly one variable write.

    One write, not two: the tick is written by the render coroutine iTerm2
    dispatches in response to this one. Writing both here would report a
    render that had not happened.

    Args:
        set_variable: The variable writer.
        session_id: The pane to write to.
        status: Its new status.
        task: What it is doing, if known.
        now: Epoch seconds to stamp the payload with.

    Returns:
        The encoded payload, so a caller can suppress an unchanged push
        without re-deriving it. Note that two pushes of the same state are
        **not** byte-equal, because the stamp moves: comparing payloads to
        suppress a redundant write would suppress nothing. Compare the
        `(status, task)` pair.
    """
    payload = encode_status(status, task, now=now)
    await set_variable(session_id, STATUS_VARIABLE, payload)
    return payload


class RenderTicker:
    """A per-session counter of renders, monotonic within one process.

    Deliberately not persisted and not read back from iTerm2. Its only claim
    is that a later render has a higher number than an earlier one for the
    same pane, which is what makes "the bar is still updating" observable
    from outside; an absolute count would imply a durability nothing here
    provides.
    """

    def __init__(self) -> None:
        self._ticks: dict[str, int] = {}

    def advance(self, session_id: str) -> int:
        """Return this pane's next tick, counting from 1."""
        tick = self._ticks.get(session_id, 0) + 1
        self._ticks[session_id] = tick
        return tick

    def value(self, session_id: str) -> int:
        """Return this pane's current tick, 0 if it has never rendered."""
        return self._ticks.get(session_id, 0)


async def render_for_session(
    session_id: str | None,
    raw_status: Any,
    *,
    ticker: RenderTicker,
    set_variable: SetVariable,
    width: int = DEFAULT_WIDTH,
    now: float | None = None,
) -> str:
    """The body of the status bar coroutine, with its dependencies passed in.

    Separated from the decorated coroutine so that the whole render path is
    reachable headlessly. The coroutine itself is then a signature iTerm2
    reflects on and nothing else.

    The tick is written **after** the line is computed, so a tick that
    advanced is evidence a line was produced rather than evidence the
    coroutine was entered.

    Args:
        session_id: The pane being rendered, or `None` if iTerm2 did not say.
        raw_status: The pane's `user.palaver_status` value.
        ticker: Where render counts are kept.
        set_variable: The variable writer.
        width: Character budget.
        now: Epoch seconds to judge the status's freshness against.

    Returns:
        The line to display.

    Raises:
        Exception: Whatever the render or the tick write raised. This is the
            *inner* body and it is allowed to fail; `render_or_ladybug` is
            the boundary that turns a failure into something displayable.
            Keeping the raise here is what makes "the tick is not written
            when no line was produced" an assertable claim.
    """
    line = line_for(raw_status, width, now=now)
    if session_id:
        try:
            await set_variable(session_id, TICK_VARIABLE, ticker.advance(session_id))
        except Exception:  # pragma: no cover - defended, not provoked
            # The tick is observability. Failing to write it must not cost
            # the user the line itself, which is the actual product.
            log.warning("could not write %s for %s", TICK_VARIABLE, session_id, exc_info=True)
    return line


async def render_or_ladybug(
    session_id: str | None,
    raw_status: Any,
    *,
    ticker: RenderTicker,
    set_variable: SetVariable,
    width: int = DEFAULT_WIDTH,
    now: float | None = None,
) -> str:
    """Render a pane's line, or say so visibly when that is not possible.

    The boundary iTerm2 actually calls. `generic_handle_rpc` in the library
    catches whatever escapes and reports an error back to iTerm2, which draws
    its own ladybug -- so an uncaught raise is not silent, but the traceback
    ends up in iTerm2's dialog rather than in Palaver's log, and nothing
    headless can assert on it.

    Catching here costs the user iTerm2's clickable error popover for this
    class of failure. It buys a traceback in `.logs/palaver.log`, which is
    where Palaver's other failures already are, and a return value tests can
    assert. Registration failures and dropped connections never reach this
    function and still get iTerm2's own ladybug, so the glyph means one thing
    either way.

    There is no cache in this path, deliberately. A failed render returns the
    glyph and nothing else; the last good line is not held anywhere to fall
    back on, because falling back on it would be the stale-value failure
    arriving by a different route.

    Args:
        session_id: The pane being rendered, or `None` if iTerm2 did not say.
        raw_status: The pane's `user.palaver_status` value.
        ticker: Where render counts are kept.
        set_variable: The variable writer.
        width: Character budget.
        now: Epoch seconds to judge the status's freshness against.

    Returns:
        The line to display, or `LADYBUG` if producing one raised.
    """
    try:
        return await render_for_session(
            session_id, raw_status, ticker=ticker, set_variable=set_variable, width=width, now=now
        )
    except Exception:
        # Logged before the glyph is chosen: the glyph is all the user gets,
        # and without the traceback there would be nothing to act on.
        log.exception("status bar render failed for %s", session_id)
        return LADYBUG


def make_variable_writer(connection: Any) -> SetVariable:
    """Return a `SetVariable` that writes over an iTerm2 connection.

    Goes through `iterm2.rpc.async_variable` rather than
    `Session.async_set_variable`, because the coroutine is handed a session
    *id* and the library's `Session.__init__` says not to construct one
    directly. Values are JSON-encoded here, matching what the library's own
    setter does.

    Args:
        connection: A live `iterm2.Connection`.

    Returns:
        An async `(session_id, name, value)` writer that raises on refusal.
    """
    # Called for its error, not its return: it names the missing `ui` extra,
    # where a bare `import iterm2.rpc` would raise a plain ImportError.
    import_iterm2()
    import iterm2.api_pb2  # noqa: PLC0415 - optional extra, checked immediately above
    import iterm2.rpc  # noqa: PLC0415

    async def set_variable(session_id: str, name: str, value: Any) -> None:
        result = await iterm2.rpc.async_variable(
            connection, session_id, [(name, json.dumps(value))], []
        )
        status = result.variable_response.status
        ok = iterm2.api_pb2.VariableResponse.Status.Value("OK")
        if status != ok:
            raise iterm2.rpc.RPCException(iterm2.api_pb2.VariableResponse.Status.Name(status))

    return set_variable


def build_component(
    connection: Any,
    *,
    ticker: RenderTicker | None = None,
    set_variable: SetVariable | None = None,
    width: int = DEFAULT_WIDTH,
) -> tuple[Any, Any]:
    """Build the component and the coroutine that feeds it.

    Args:
        connection: A live `iterm2.Connection`, used for the default writer.
        ticker: Where render counts are kept; a fresh one by default.
        set_variable: Override the variable writer.
        width: Character budget passed to `render`.

    Returns:
        A `(component, coroutine)` pair. Not registered: registration is a
        separate, awaitable step so a caller can decide when to take the
        name.
    """
    iterm2 = import_iterm2()
    ticks = RenderTicker() if ticker is None else ticker
    writer = make_variable_writer(connection) if set_variable is None else set_variable

    @iterm2.StatusBarRPC
    async def palaver_status_line(
        knobs,
        status=iterm2.Reference(STATUS_REFERENCE),
        session_id=iterm2.Reference("id"),
    ):
        # `knobs` is required by the decorator even with no knobs declared:
        # it reflects on this signature to build the RPC.
        del knobs
        return await render_or_ladybug(
            session_id, status, ticker=ticks, set_variable=writer, width=width
        )

    component = iterm2.StatusBarComponent(
        short_description=SHORT_DESCRIPTION,
        detailed_description=DETAILED_DESCRIPTION,
        knobs=[],
        exemplar=EXEMPLAR,
        update_cadence=UPDATE_CADENCE,
        identifier=IDENTIFIER,
    )
    return component, palaver_status_line


async def register(
    connection: Any,
    *,
    ticker: RenderTicker | None = None,
    set_variable: SetVariable | None = None,
    width: int = DEFAULT_WIDTH,
    timeout: float | None = None,
) -> Any:
    """Register the component with iTerm2 and return it.

    Registration makes the component *offerable*. It does not put it in any
    profile and does not put it on screen; `check_layout` is what reports
    whether either happened.

    Args:
        connection: A live `iterm2.Connection`.
        ticker: Where render counts are kept.
        set_variable: Override the variable writer.
        width: Character budget passed to `render`.
        timeout: Passed through to the library's registration call.

    Returns:
        The registered `iterm2.StatusBarComponent`.
    """
    component, coro = build_component(
        connection, ticker=ticker, set_variable=set_variable, width=width
    )
    await component.async_register(connection, coro, timeout=timeout)
    return component


def _decoded_base64(text: str) -> bytes:
    """Return `text` decoded as base64, or empty bytes if it is not base64."""
    if len(text) < 8:
        return b""
    try:
        return base64.b64decode(text, validate=True)
    except binascii.Error, ValueError:
        return b""


def _mentions(value: Any, needle: str) -> bool:
    """Report whether `needle` appears anywhere inside a decoded-plist value.

    A recursive scan rather than a field read, because the entry schema is
    not public: iTerm2 keeps a `_savedRegistrationRequest` in each component,
    so an entry is a nested structure that may carry the identifier inside a
    serialized request rather than at a known key. A scan stays true when
    that shape changes; a field read would quietly start reporting False.
    Base64 strings are decoded once, since that is how a serialized request
    survives a JSON round trip.
    """
    if isinstance(value, str):
        return needle in value or needle.encode() in _decoded_base64(value)
    if isinstance(value, (bytes, bytearray)):
        return needle.encode() in bytes(value)
    if isinstance(value, Mapping):
        return any(_mentions(key, needle) or _mentions(item, needle) for key, item in value.items())
    if isinstance(value, Sequence):
        return any(_mentions(item, needle) for item in value)
    return False


def layout_contains(layout: Any, identifier: str = IDENTIFIER) -> bool:
    """Report whether a `Status Bar Layout` value includes this component.

    Args:
        layout: The profile's layout value, or `None` if it has no layout.
        identifier: The component identifier to look for.

    Returns:
        True if the identifier appears among the layout's components.
    """
    if not isinstance(layout, Mapping):
        return False
    return _mentions(layout.get("components"), identifier)


def profile_identity(props: Mapping[str, Any]) -> str | None:
    """Return the guid of the *shared* profile a session's profile came from.

    `Original Guid` rather than `Guid`: iTerm2 divorces a session's profile
    the moment anything writes a per-session property, after which `Guid` is
    session-local and matches no shared profile. Falls back to `Guid` for an
    undivorced profile, which has no `Original Guid` at all.
    """
    return props.get(ORIGINAL_GUID_KEY) or props.get(GUID_KEY)


@dataclass(frozen=True)
class LayoutCheck:
    """Whether a profile will actually show Palaver's component.

    Three independent facts rather than one boolean, because the remedies
    differ and "it is not showing" is not an actionable thing to be told.
    """

    #: The shared profile this was read from, per `profile_identity`.
    profile_guid: str | None
    #: The profile the caller expected, or `None` if it did not care.
    expected_guid: str | None
    #: Whether the layout's components include the identifier.
    in_layout: bool
    #: Whether the profile shows a status bar at all.
    bar_shown: bool

    @property
    def profile_matches(self) -> bool:
        """Whether this is the profile the caller meant to check."""
        return self.expected_guid is None or self.profile_guid == self.expected_guid

    @property
    def ok(self) -> bool:
        """Whether the component is on screen for this profile."""
        return self.profile_matches and self.in_layout and self.bar_shown

    @property
    def remedy(self) -> str | None:
        """What to do about it, or `None` if there is nothing to do."""
        if self.ok:
            return None
        reasons = []
        if not self.profile_matches:
            reasons.append(
                f"this is profile {self.profile_guid!r}, not the expected "
                f"{self.expected_guid!r}; check the profile the observed panes use"
            )
        if not self.in_layout:
            reasons.append(f"the component is not in the status bar layout: {LAYOUT_REMEDY}")
        if not self.bar_shown:
            reasons.append(f"the status bar is switched off: {SHOW_BAR_REMEDY}")
        return "; ".join(reasons)


def check_layout(
    props: Mapping[str, Any],
    *,
    identifier: str = IDENTIFIER,
    expected_guid: str | None = None,
) -> LayoutCheck:
    """Report whether these profile properties put the component on screen.

    Pure over the properties dict, so the whole gate is testable without
    iTerm2 running. Callers get the dict from
    `(await session.async_get_profile()).all_properties`.

    Args:
        props: A profile's properties.
        identifier: The component identifier to look for.
        expected_guid: The shared profile the observed panes should be using.
            Pass it to catch the case the plan calls out -- a component
            configured into a profile nothing is running under, which
            otherwise reads as a pass.

    Returns:
        A `LayoutCheck`.
    """
    return LayoutCheck(
        profile_guid=profile_identity(props),
        expected_guid=expected_guid,
        in_layout=layout_contains(props.get(LAYOUT_KEY), identifier),
        bar_shown=bool(props.get(SHOW_BAR_KEY)),
    )


async def show_status_bar(profile: Any, *, shown: bool = True) -> None:
    """Turn a profile's status bar on. Never called as a side effect.

    Switching the status bar on changes what every pane using the profile
    looks like, so it is its own named step that a human asks for -- from
    `palaver ui --enable-status-bar` or by hand -- rather than something
    attaching to iTerm2 does on the way past. `palaver ui --selftest`
    deliberately does not call this: it reports that the bar is off and
    leaves the decision alone. Palaver observes; it does not redecorate.

    The library has no accessor for this key, so this goes through the
    generic setter it uses for every property it *does* name.

    Args:
        profile: An `iterm2.Profile` or `PartialProfile`. Pass the **shared**
            profile: a write there was measured to reach sessions whose
            profile is already divorced from it.
        shown: False to turn it back off, which is what a selftest owes the
            user afterwards if they did not have it on to begin with.
    """
    await profile._async_simple_set(SHOW_BAR_KEY, bool(shown))  # noqa: SLF001 - no public setter
