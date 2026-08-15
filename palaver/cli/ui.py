"""`palaver ui`: check the iTerm2 pane surface, and say what is not working.

Task 5.5, and the Phase 5 gate's second half. Everything in Phase 5 that can
be proved headlessly is proved in `palaver.ui.render` and
`palaver.ui.component`; this is what checks the half that cannot be, against
a running iTerm2 on the machine the user is actually looking at.

**A profile that is not configured is a report, not a failure.** The status
bar is off in every profile on this machine and the component is in no
layout, which is the honest default state: registration makes the component
*offerable*, and a human chooses it once in iTerm2's own settings. If that
counted as a failed selftest, the phase gate could never pass on a fresh
machine and the exit code would be measuring iTerm2's preferences rather
than Palaver's code. So `--selftest` exits 0 with a remedy, and exits 1 only
when something Palaver is responsible for actually broke: registration
refused, a variable that would not round-trip, a render tick that did not
advance.

**A check that cannot run is reported as skipped, with the reason.** The
phase Acceptance is explicit that a missing read path falls back to
asserting the RPC was invoked and "never to asserting nothing", so a skip
prints what it was, why it could not run, and what would make it runnable.
An all-skipped selftest still exits 0 -- there is nothing to fail -- and the
output makes that impossible to mistake for a pass.

**Turning the status bar on is a separate flag.** `--enable-status-bar`
changes what every pane using the profile looks like, so it is never a side
effect of asking whether things work. `--selftest` will tell you the bar is
off; it will not switch it on for you.

Two-stream contract: the report goes to stdout, progress goes through
`on_status` to stderr (INV-1). Nothing here reads observed session content;
it writes Palaver's own two `user.` variables and reads profile properties
(INV-9), and it never sends text to a session (INV-2).
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from palaver.observer.signals import Status
from palaver.ui import component, publisher
from palaver.ui.connection import (
    COOKIE_ENV,
    KEY_ENV,
    UiConnectionError,
    import_iterm2,
    preflight,
    request_cookie_and_key,
)

NAME = "ui"
HELP = "check the iTerm2 pane surface, or turn its status bar on"

#: Seconds to wait for registration. Generous: the answer comes from another
#: application that may be mid-relaunch, and a spurious timeout here reads as
#: a broken component.
REGISTER_TIMEOUT = 15.0

#: The status pushed during the round-trip check. `UNKNOWN` on purpose --
#: whatever this test leaves behind for the fraction of a second before the
#: previous value is restored should not be a claim about the pane.
PROBE_STATUS = Status.UNKNOWN
PROBE_TASK = "palaver ui --selftest"

#: Written to the tick variable before the render path is invoked, so the
#: check compares against a value this command set rather than against
#: whatever the last process to touch the pane happened to leave. Negative
#: because `RenderTicker` counts from 1 and never produces it.
TICK_BASELINE = -1

#: How many times the render path is invoked. More than one, so "the tick
#: advanced" is a claim about counting rather than about a single write.
RENDERS = 2


@dataclass(frozen=True)
class Check:
    """One thing the selftest tried, and how it went.

    `passed is None` means the check could not run at all, which is neither a
    pass nor a failure. Kept as a third state rather than folded into either,
    because folding a skip into a pass is how a gate ends up green on a
    machine where nothing was tested.
    """

    name: str
    passed: bool | None
    detail: str

    @property
    def mark(self) -> str:
        return {True: "ok", False: "FAILED", None: "skipped"}[self.passed]


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def exit_code(checks: list[Check]) -> int:
    """Return 0 unless a check actually ran and failed.

    Args:
        checks: Everything the selftest tried.

    Returns:
        1 if any check failed, else 0. Skips do not fail: see the module
        docstring on why an unconfigured profile is a report.
    """
    return 1 if any(check.passed is False for check in checks) else 0


def format_report(checks: list[Check]) -> str:
    """Render the checks as the command's stdout.

    Args:
        checks: Everything the selftest tried.

    Returns:
        One line per check plus a summary line. The summary counts skips
        separately from passes, so "3 ok" and "3 ok, 4 skipped" cannot be
        confused for each other at a glance.
    """
    if not checks:
        return "palaver ui: nothing was checked\n"
    width = max(len(check.mark) for check in checks)
    lines = [f"{check.mark:<{width}}  {check.name}: {check.detail}" for check in checks]
    passed = sum(check.passed is True for check in checks)
    failed = sum(check.passed is False for check in checks)
    skipped = sum(check.passed is None for check in checks)
    summary = f"{passed} ok"
    if failed:
        summary += f", {failed} failed"
    if skipped:
        summary += f", {skipped} skipped"
    lines.append(summary)
    return "\n".join(lines) + "\n"


def ensure_cookie(*, ask: bool, env: dict[str, str] | None = None) -> Check:
    """Make sure this process can attach, asking iTerm2 for a cookie if told to.

    iTerm2 injects `ITERM2_COOKIE` only into scripts it launches itself, so a
    selftest run from a shell has none and the connection would either fail
    or -- worse -- have the library raise a modal dialog on the user's screen
    behind their back. Asking here is explicit, announced, and refusable.

    The cookie and key are credentials. They go into this process's own
    environment and nowhere else: never printed, never logged, never passed
    as an argument (`ps` shows argv to every process on the machine).

    Args:
        ask: Whether to request a cookie when the environment has none.
        env: Environment to read and write, defaulting to `os.environ`.

    Returns:
        A `Check` describing what happened. `passed is None` when there is no
        cookie and asking was declined or refused, which makes every live
        check below a skip rather than a failure.
    """
    environ = os.environ if env is None else env
    if environ.get(COOKIE_ENV):
        return Check("cookie", True, f"{COOKIE_ENV} is already set")
    if not ask:
        return Check("cookie", None, f"{COOKIE_ENV} is unset and --no-ask-cookie was given")
    try:
        cookie, key = request_cookie_and_key()
    except UiConnectionError as exc:
        return Check("cookie", None, f"iTerm2 would not issue one: {exc}")
    environ[COOKIE_ENV] = cookie
    environ[KEY_ENV] = key
    return Check("cookie", True, "iTerm2 issued one for this process")


async def _probe_session(app: Any, session_id: str | None) -> Any:
    """Return the session to test against: the named one, or the current one."""
    if session_id:
        return app.get_session_by_id(session_id)
    window = app.current_terminal_window
    if window is None or window.current_tab is None:
        return None
    return window.current_tab.current_session


async def run_checks(
    connection: Any,
    *,
    session_id: str | None = None,
    width: int = component.DEFAULT_WIDTH,
    on_status: Callable[[str], None] = lambda _message: None,
) -> list[Check]:
    """Run every live check against a connected iTerm2, in dependency order.

    The order is not cosmetic: registration has to succeed before a variable
    write means anything, and the write has to land before a tick read can be
    interpreted. A check whose precondition failed is skipped naming that,
    rather than run and reported as a second failure.

    **The render tick is driven from here, not by iTerm2.** iTerm2 dispatches
    the component's coroutine only for a component that is in a shown status
    bar, which is the state this very command exists to report as absent. So
    the fallback the phase Acceptance names is taken: the same coroutine body
    is invoked over the same live connection, writing the real tick variable
    into the real pane, and the tick is read back through
    `async_get_variable`. That proves the writer, the render, and the
    round-trip against iTerm2 itself; what it does not prove is that iTerm2
    would have called it, which is reported separately and honestly.

    Args:
        connection: A live `iterm2.Connection`.
        session_id: Pane to test against; the current pane by default.
        width: Character budget passed to the render.
        on_status: Progress channel (INV-1).

    Returns:
        The checks, in the order they were tried.
    """
    iterm2 = import_iterm2()
    checks: list[Check] = []

    on_status("registering the status bar component")
    ticker = component.RenderTicker()
    writer = component.make_variable_writer(connection)
    try:
        await component.register(
            connection, ticker=ticker, set_variable=writer, width=width, timeout=REGISTER_TIMEOUT
        )
    except Exception as exc:
        checks.append(Check("register", False, f"iTerm2 refused the component: {exc!r}"))
        return checks
    checks.append(Check("register", True, f"iTerm2 accepted {component.IDENTIFIER}"))

    app = await iterm2.async_get_app(connection)
    session = await _probe_session(app, session_id)
    if session is None:
        checks.append(Check("pane", None, "no pane to test against; open a window and retry"))
        return checks
    pane = session.session_id
    checks.append(Check("pane", True, f"testing against {pane}"))

    previous = await session.async_get_variable(component.STATUS_VARIABLE)

    try:
        on_status(f"pushing a status to {pane}")
        pushed = await component.push_status(writer, pane, PROBE_STATUS, PROBE_TASK)
        read_back = await session.async_get_variable(component.STATUS_VARIABLE)
        decoded = component.decode_status(read_back)
        if decoded == (PROBE_STATUS, PROBE_TASK):
            checks.append(
                Check(
                    "variables", True, f"{component.STATUS_VARIABLE} round-tripped through iTerm2"
                )
            )
        else:
            checks.append(
                Check(
                    "variables",
                    False,
                    f"pushed {pushed!r} and read back {read_back!r}, which decodes to {decoded!r}",
                )
            )

        on_status("invoking the render and reading the tick back")
        # Baseline first. `RenderTicker` counts within one process and starts
        # at zero, so a value left in the pane by a previous run of this very
        # command -- or by the daemon -- reads as "the tick did not advance"
        # and fails a working component. What is asserted below is that the
        # render path wrote back exactly the count it kept, which the
        # baseline makes non-vacuous rather than something the baseline
        # itself could satisfy.
        await writer(pane, component.TICK_VARIABLE, TICK_BASELINE)
        lines = [
            await component.render_or_ladybug(
                pane, read_back, ticker=ticker, set_variable=writer, width=width
            )
            for _ in range(RENDERS)
        ]
        after = await session.async_get_variable(component.TICK_VARIABLE)
        expected = ticker.value(pane)
        if after == expected == RENDERS:
            checks.append(
                Check(
                    "render tick",
                    True,
                    f"{component.TICK_VARIABLE} advanced from {TICK_BASELINE} to {after} "
                    f"across {RENDERS} renders",
                )
            )
        else:
            checks.append(
                Check(
                    "render tick",
                    False,
                    f"{component.TICK_VARIABLE} reads {after!r} after {RENDERS} renders "
                    f"from a baseline of {TICK_BASELINE}; the render path counted {expected}",
                )
            )
        if component.LADYBUG in lines:
            checks.append(Check("render", False, f"the render returned {component.LADYBUG}"))
        else:
            checks.append(Check("render", True, f"the pane renders as {lines[-1]!r}"))

        # The production path, over the same connection: everything above
        # this line proves a pane can be written to, and none of it proves
        # anything writes to one. Reads this machine's real pane variables,
        # process table, and session stores, so a status here is derived the
        # way the AutoLaunch process derives it rather than from a probe.
        on_status(f"publishing one tick to {pane} the way the daemon would")
        pushes = await publisher.publish_once(
            [pane],
            read_variables=publisher.make_variables_reader(connection),
            set_variable=writer,
            on_status=on_status,
        )
        if pushes and pushes[0].published:
            checks.append(
                Check(
                    "publish",
                    True,
                    f"the publisher derived {pushes[0].status.name.lower()} for this pane "
                    f"and iTerm2 accepted it",
                )
            )
        else:
            checks.append(Check("publish", False, f"the publisher wrote nothing to {pane}"))
    finally:
        # Whatever the daemon had put there is the truth about this pane; the
        # probe value is not, and leaving it would be this command creating
        # the exact stale status it is meant to detect.
        await writer(pane, component.STATUS_VARIABLE, previous)

    props = (await session.async_get_profile()).all_properties
    layout = component.check_layout(props)
    if layout.ok:
        checks.append(
            Check("layout", True, f"the component is shown in profile {layout.profile_guid}")
        )
        checks.append(
            Check("dispatched by iTerm2", True, "iTerm2 renders this component on its own cadence")
        )
    else:
        checks.append(Check("layout", None, layout.remedy))
        checks.append(
            Check(
                "dispatched by iTerm2",
                None,
                "iTerm2 only dispatches a component in a shown status bar, so the tick "
                "above was driven by this command over the same connection",
            )
        )
    return checks


async def set_status_bar(
    connection: Any, *, shown: bool, on_status: Callable[[str], None]
) -> list[Check]:
    """Turn the status bar on or off for every shared profile, by explicit request.

    Every shared profile rather than the current pane's, because a session's
    profile is divorced -- writing there would change one pane until it
    closed, which looks like it worked and then silently is not there
    tomorrow. The shared profile is the durable one, and a write to it was
    measured to reach already-divorced sessions.

    The off direction exists because this is the one thing `palaver ui` does
    that changes what every pane on the machine looks like, and a change that
    can only be made from Palaver and only undone from iTerm2's preferences
    is not a change a user can try.

    Args:
        connection: A live `iterm2.Connection`.
        shown: Whether the bar should be shown.
        on_status: Progress channel (INV-1).

    Returns:
        One check naming how many profiles were written.
    """
    iterm2 = import_iterm2()
    profiles = await iterm2.PartialProfile.async_query(connection)
    changed = []
    for profile in profiles:
        full = await profile.async_get_full_profile()
        await component.show_status_bar(full, shown=shown)
        changed.append(profile.guid)
        on_status(f"status bar {'on' if shown else 'off'} for profile {profile.guid}")
    if not shown:
        return [Check("status bar", True, f"switched off for {len(changed)} shared profile(s)")]
    return [
        Check(
            "status bar",
            True,
            f"switched on for {len(changed)} shared profile(s); the component still "
            f"has to be added to the layout by hand ({component.LAYOUT_REMEDY})",
        )
    ]


def add_arguments(parser) -> None:
    """Register `ui`'s flags on its subparser."""
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check the pane surface against a running iTerm2",
    )
    parser.add_argument(
        "--enable-status-bar",
        action="store_true",
        help="turn the status bar on in every shared profile; changes how every pane looks",
    )
    parser.add_argument(
        "--disable-status-bar",
        action="store_true",
        help="turn the status bar back off in every shared profile",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="pane to test against (default: the current pane)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=component.DEFAULT_WIDTH,
        help=f"character budget for the rendered line (default: {component.DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--no-ask-cookie",
        action="store_true",
        help=(
            "do not ask iTerm2 for a connection cookie; live checks are skipped "
            "instead, which is what an unattended run wants"
        ),
    )


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver ui`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when nothing that ran failed, including the ordinary case of an
        unconfigured profile. 1 when a check ran and failed, and 2 when the
        command was given nothing to do.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    if args.enable_status_bar and args.disable_status_bar:
        print(
            "palaver ui: --enable-status-bar and --disable-status-bar contradict each other",
            file=out,
        )
        return 2

    if not (args.selftest or args.enable_status_bar or args.disable_status_bar):
        print(
            "palaver ui: nothing to do; pass --selftest to check the pane surface "
            "or --enable-status-bar to turn it on",
            file=out,
        )
        return 2

    checks: list[Check] = []
    if args.selftest:
        checks.append(ensure_cookie(ask=not args.no_ask_cookie))
        if checks[-1].passed is not True:
            checks.append(Check("attach", None, "no cookie, so nothing live could be tried"))
            print(format_report(checks), file=out, end="")
            return exit_code(checks)

    try:
        on_status("checking iTerm2's socket and cookie")
        preflight()
    except UiConnectionError as exc:
        checks.append(Check("attach", None, str(exc)))
        print(format_report(checks), file=out, end="")
        return exit_code(checks)

    iterm2 = import_iterm2()
    collected: list[Check] = []

    async def main(connection):
        if args.enable_status_bar or args.disable_status_bar:
            collected.extend(
                await set_status_bar(connection, shown=args.enable_status_bar, on_status=on_status)
            )
        if args.selftest:
            collected.extend(
                await run_checks(
                    connection,
                    session_id=args.session,
                    width=args.width,
                    on_status=on_status,
                )
            )

    try:
        iterm2.run_until_complete(main)
    except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
        checks.append(Check("attach", None, f"could not attach to iTerm2: {exc}"))
        print(format_report(checks), file=out, end="")
        return exit_code(checks)

    checks.append(Check("attach", True, "connected over iTerm2's Unix domain socket"))
    checks.extend(collected)
    print(format_report(checks), file=out, end="")
    return exit_code(checks)
