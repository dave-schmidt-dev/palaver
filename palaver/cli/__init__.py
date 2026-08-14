"""Palaver's command line: one `argparse` root, one module per subcommand.

`main()` is the `palaver` console script declared in `pyproject.toml`. Adding
a subcommand means writing a module with `NAME`, `HELP`, `add_arguments()` and
`run()`, and listing it in `SUBCOMMANDS` — the root parser never learns
anything about a subcommand's own flags. Task 1.9 registers `status` and
`inspect` that way.

Two output rules hold for every subcommand, and they exist because INV-1's
progress channel and the CLI's output contract must not collide:

* **stdout is the command's result** — the report, the table, the answer.
  Only the subcommand's `out` stream writes there.
* **stderr is progress and diagnostics.** A subcommand that walks many
  session stores emits per-session progress through its `on_status` channel,
  which writes to stderr, so a long scan is never a silent wait and piping
  stdout stays clean.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from palaver.cli import diagnose, fixture_lint, inspect, status
from palaver.logging_setup import setup_logging

#: Every registered subcommand module, in help-listing order.
SUBCOMMANDS = (diagnose, fixture_lint, inspect, status)


def build_parser() -> argparse.ArgumentParser:
    """Build the `palaver` root parser with every registered subcommand.

    Returns:
        The configured parser. `--help` works with no subcommand, and each
        subcommand module supplies its own arguments and handler.
    """
    parser = argparse.ArgumentParser(
        prog="palaver",
        description="Local-first observer, memory, and situational awareness.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="log at DEBUG level to .logs/palaver.log",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for module in SUBCOMMANDS:
        subparser = subparsers.add_parser(
            module.NAME,
            help=module.HELP,
            description=module.HELP,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(handler=module.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the `palaver` CLI.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        The subcommand's exit status, or 2 when no subcommand was given (the
        conventional `argparse` usage-error status, printed with the help
        text so the invocation is not silently a no-op).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(debug=args.debug)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)
