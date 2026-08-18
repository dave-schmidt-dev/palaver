"""`palaver ui`: manage pane pins and companion enablement without focusing panes.

The status-bar experiment was rejected. This command remains because an
explicit pane pin is useful to the future companion-pane surface when a
project directory has been renamed or moved.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TextIO

from palaver.ui.autolaunch import STATE_DIR
from palaver.ui.companion import CompanionController, make_metadata_reader
from palaver.ui.connection import UiConnectionError, import_iterm2, preflight
from palaver.ui.pane_join import CLAUDE_SOURCE, CODEX_SOURCE, PIN_VARIABLE

NAME = "ui"
HELP = "manage an iTerm2 pane's session pin or companion"

SetVariable = Callable[[str, str, Any], Awaitable[None]]


def _stderr_status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def make_variable_writer(connection: Any) -> SetVariable:
    """Return an authenticated writer for one named iTerm2 session variable."""
    import_iterm2()
    import iterm2.api_pb2  # noqa: PLC0415 - optional extra, checked above
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


async def set_session_pin(
    writer: SetVariable,
    pane_id: str,
    *,
    source: str | None = None,
    session_key: str | None = None,
) -> str:
    """Write or clear one pane's pin without selecting or focusing it."""
    if source is None and session_key is None:
        value = ""
    elif source in {CLAUDE_SOURCE, CODEX_SOURCE} and isinstance(session_key, str) and session_key:
        value = json.dumps({"source": source, "session_key": session_key}, separators=(",", ":"))
    else:
        raise ValueError("pin requires a supported source and non-empty session key")
    await writer(pane_id, PIN_VARIABLE, value)
    return value


def add_arguments(parser) -> None:
    """Register `ui`'s pane-pin flags."""
    parser.add_argument("--session", help="iTerm2 pane id to pin")
    parser.add_argument(
        "--pin",
        nargs=2,
        metavar=("SOURCE", "SESSION_KEY"),
        help="pin the pane to SOURCE and SESSION_KEY without focusing it",
    )
    parser.add_argument("--clear-pin", action="store_true", help="clear the pane's explicit pin")
    parser.add_argument(
        "--enable-companion", action="store_true", help="enable the companion for the named pane"
    )
    parser.add_argument(
        "--disable-companion", action="store_true", help="disable and close its companion"
    )


def run(args, *, out: TextIO | None = None) -> int:
    """Apply requested pane pin and companion lifecycle actions."""
    out = sys.stdout if out is None else out
    pin = args.pin
    clear_pin = args.clear_pin
    enable = args.enable_companion
    disable = args.disable_companion
    if pin and clear_pin:
        print("palaver ui: --pin and --clear-pin contradict each other", file=out)
        return 2
    if enable and disable:
        print("palaver ui: --enable-companion and --disable-companion contradict", file=out)
        return 2
    if not (pin or clear_pin or enable or disable):
        print("palaver ui: nothing to do; pass --session with a pin or companion action", file=out)
        return 2
    if not args.session:
        print("palaver ui: --pin/--clear-pin requires --session PANE_ID", file=out)
        return 2
    if pin and (pin[0] not in {CLAUDE_SOURCE, CODEX_SOURCE} or not pin[1]):
        print("palaver ui: --pin requires SOURCE claude-code/codex and SESSION_KEY", file=out)
        return 2

    try:
        preflight()
    except UiConnectionError as exc:
        print(f"palaver ui: {exc}", file=out)
        return 1

    writer_result: list[str] = []

    async def main(connection: Any) -> None:
        if clear_pin:
            writer = make_variable_writer(connection)
            await set_session_pin(writer, args.session)
            writer_result.append(f"cleared pin for pane {args.session}")
        elif pin:
            writer = make_variable_writer(connection)
            await set_session_pin(writer, args.session, source=pin[0], session_key=pin[1])
            writer_result.append(f"pinned pane {args.session} to {pin[0]} {pin[1]}")
        if enable or disable:
            iterm2 = import_iterm2()
            app = await iterm2.async_get_app(connection)
            controller = CompanionController(
                STATE_DIR,
                read_metadata=make_metadata_reader(connection),
                on_status=_stderr_status,
            )
            await controller.reconcile(app, create=False)
            changed = await controller.set_enabled(app, args.session, enabled=enable)
            if not changed:
                raise RuntimeError(f"pane {args.session} does not exist")
            if enable:
                await controller.reconcile(app, only_agent_id=args.session)
            writer_result.append(
                f"{'enabled' if enable else 'disabled'} companion for pane {args.session}"
            )

    try:
        import_iterm2().run_until_complete(main)
    except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
        print(f"palaver ui: could not attach to iTerm2: {exc}", file=out)
        return 1
    print("; ".join(writer_result), file=out)
    return 0
