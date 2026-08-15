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

Nothing here reads observed session content. It writes two `user.` variables
into panes Palaver already tracks, and reads profile properties (INV-9).
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
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

#: An async `(session_id, name, value) -> None`. Injected rather than reached
#: for, so every path through this module is testable without iTerm2 running.
SetVariable = Callable[[str, str, Any], Awaitable[None]]


def encode_status(status: Status, task: str | None = None) -> str:
    """Serialize what a pane should show into the value of one variable.

    Args:
        status: The pane's derived status.
        task: What it is doing, if anything is known.

    Returns:
        A JSON object string. JSON rather than a bare word because the task
        text is arbitrary session-derived prose and a delimiter-joined pair
        would break on the first colon or newline in it.
    """
    return json.dumps({"status": status.name, "task": task}, ensure_ascii=False)


def decode_status(raw: Any) -> tuple[Status, str | None]:
    """Read back what `encode_status` wrote, tolerating anything else.

    Total by construction. Every caller is inside a status bar coroutine, and
    a raise there goes to a log nobody is reading while the bar silently
    stops updating, so an unreadable value has to degrade to a renderable one
    instead.

    Args:
        raw: The variable's value: the JSON string, `None` before anything
            has ever been pushed, or whatever a stale build left behind.

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

    name = raw.get("status")
    try:
        status = Status[name] if isinstance(name, str) else Status.UNKNOWN
    except KeyError:
        # A status this build does not have. Renaming a member is exactly the
        # kind of change that would otherwise blank every bar mid-upgrade.
        status = Status.UNKNOWN

    task = raw.get("task")
    return status, task if isinstance(task, str) else None


def line_for(raw: Any, width: int = DEFAULT_WIDTH) -> str:
    """Render the line a pane should show, from the raw variable value.

    Args:
        raw: As `decode_status`.
        width: Character budget. Clamped rather than validated, for the same
            reason `decode_status` is total.

    Returns:
        One line of text, always.
    """
    status, task = decode_status(raw)
    return render(status, task, max(1, width))


async def push_status(
    set_variable: SetVariable,
    session_id: str,
    status: Status,
    task: str | None = None,
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

    Returns:
        The encoded payload, so a caller can suppress an unchanged push
        without re-deriving it.
    """
    payload = encode_status(status, task)
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

    Returns:
        The line to display.
    """
    line = line_for(raw_status, width)
    if session_id:
        try:
            await set_variable(session_id, TICK_VARIABLE, ticker.advance(session_id))
        except Exception:  # pragma: no cover - defended, not provoked
            # The tick is observability. Failing to write it must not cost
            # the user the line itself, which is the actual product.
            log.warning("could not write %s for %s", TICK_VARIABLE, session_id, exc_info=True)
    return line


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
        return await render_for_session(
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
    `palaver ui --selftest` or by hand -- rather than something attaching to
    iTerm2 does on the way past. Palaver observes; it does not redecorate.

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
