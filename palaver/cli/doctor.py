"""`palaver doctor`: what the running `llama-server` actually is, right now.

Palaver does not manage the inference server (plan section 5), which means every
assumption it holds about that server — how many KV slots exist, how large the
context window is, whether slot save and restore are available — is an
assumption about a process someone else started. When one of those assumptions
is wrong the symptom is not an error. It is latency: requests queue behind a
slot count smaller than the one the scheduler planned for, or every tick
re-prefills a prompt the code believed it had paged out. This command turns that
silent mismatch into a printed line.

**Why the flag is `--server-cmdline` and the output is not a command line.**
llama.cpp's `/props` endpoint carries no invocation. Its keys are `bos_token`,
`build_info`, `chat_template`, `chat_template_caps`, `cors_proxy_enabled`,
`default_generation_settings`, `endpoint_metrics`, `endpoint_props`,
`endpoint_slots`, `eos_token`, `is_sleeping`, `media_marker`, `modalities`,
`model_alias`, `model_ftype`, `model_path`, `total_slots`, `ui`, and
`ui_settings` — verified against the running server and against
`tools/server/server-context.cpp` on 2026-08-15. There is no `argv`, no
`cmdline`, no `params`. Recovering the literal argv would mean going outside
HTTP entirely (port to pid, pid to `ps`), which is untestable without a real
local server, unavailable when the server is not on this machine, and prints
another process's full command line for no gain the report below does not
already deliver. So the flag keeps the plan's name and the report states plainly
what it is showing: the *effective* configuration, observed, not the invocation.
Every fact the flag was asked for — the slot count, and whether slot save and
restore work — is in it.

The command has one check today, so `palaver doctor` and `palaver doctor
--server-cmdline` do the same thing. The flag exists because the plan names it
and because a second check would need the distinction; it is not pretending to
select among checks that do not exist.

Output follows the CLI's two-stream contract: the report goes to stdout, and
per-request progress goes through `on_status` to stderr (INV-1), so a probe
against an unreachable server is a wait with visible reasons rather than a
silent one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from palaver.extract.client import ModelClientError
from palaver.extract.slots import ServerProperties, SlotClient, SlotSaveSupport, SlotState

NAME = "doctor"
HELP = "report the effective configuration of the running llama-server"

#: Printed above the observed values so no reader mistakes them for the
#: server's invocation. Kept as a constant because `tests/test_slots.py`
#: asserts the report carries it — a command named `--server-cmdline` that
#: silently reported something else would be the defect this note prevents.
NO_CMDLINE_NOTE = (
    "llama-server exposes no command line over HTTP (/props carries no argv, "
    "cmdline, or params), so every value below is an observed property of the "
    "running process, not the invocation that produced it."
)


def _render_properties(properties: ServerProperties) -> list[str]:
    context = "unreported" if properties.n_ctx is None else f"{properties.n_ctx} tokens"
    return [
        f"build: {properties.build_info or 'unreported'}",
        f"model: {properties.model_path or 'unreported'}",
        f"alias: {properties.model_alias or 'unreported'}",
        f"context: {context}",
        f"slots: {properties.total_slots}",
        f"slots endpoint: {'enabled' if properties.endpoint_slots else 'disabled'}",
    ]


def _render_slots(slots: tuple[SlotState, ...]) -> str:
    busy = sum(1 for slot in slots if slot.is_processing)
    return f"slots observed: {len(slots)} ({busy} processing)"


def _render_save_support(support: SlotSaveSupport | None, failure: str) -> list[str]:
    """Render the save/restore capability line, and the server's own words under it.

    `support` is `None` when the probe itself failed, which is reported as
    `unknown` rather than as `unavailable`: "the capability is absent" and "the
    question could not be asked" are different answers, and collapsing them
    would let a broken probe read as a configured server.
    """
    if support is None:
        return ["slot save/restore: unknown", f"  {failure}"]
    state = "available" if support.supported else "unavailable"
    return [f"slot save/restore: {state}", f"  {support.detail}"]


def render_report(
    *,
    host: str,
    port: int,
    properties: ServerProperties,
    slots: tuple[SlotState, ...],
    support: SlotSaveSupport | None,
    support_failure: str = "",
) -> str:
    """Render the doctor report as the command's stdout output.

    Args:
        host: Server host the report was gathered from.
        port: Server port.
        properties: Parsed `/props`.
        slots: Live slot state, or the single-slot fallback `probe_slots`
            returns when `/slots` could not be read.
        support: Slot save/restore capability, or `None` when the probe failed.
        support_failure: Why the probe failed, when `support` is `None`.

    Returns:
        The full report, newline-terminated.
    """
    lines = [
        "palaver doctor",
        f"server: {host}:{port}",
        "",
        NO_CMDLINE_NOTE,
        "",
        *_render_properties(properties),
        _render_slots(slots),
        *_render_save_support(support, support_failure),
    ]
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Register `doctor`'s arguments on its subparser."""
    parser.add_argument(
        "--server-cmdline",
        action="store_true",
        help=(
            "report the running llama-server's effective configuration "
            "(llama-server exposes no literal command line over HTTP)"
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="llama-server host (default: 127.0.0.1; INV-9 permits no other)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="llama-server port (default: 8090)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="seconds allowed per request (default: 10)",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver doctor`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when the server answered `/props`, and 1 when it could not be reached
        or its `/props` could not be trusted. A server that is running but was
        started *without* `--slot-save-path` is a successful run reporting an
        absent capability, not a failure — that report is the point of the
        command.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    client = SlotClient(host=args.host, port=args.port, timeout=args.timeout)

    try:
        properties = client.properties(on_status=on_status)
    except ModelClientError as exc:
        print(f"palaver doctor: {exc}", file=sys.stderr)
        return 1

    slots = client.probe_slots(on_status=on_status)

    support: SlotSaveSupport | None
    support_failure = ""
    try:
        support = client.slot_save_support(on_status=on_status)
    except ModelClientError as exc:
        support = None
        support_failure = str(exc)

    out.write(
        render_report(
            host=args.host,
            port=args.port,
            properties=properties,
            slots=slots,
            support=support,
            support_failure=support_failure,
        )
    )
    return 0
