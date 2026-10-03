"""The palaver fixture-lint entry surface: rendering, argparse wiring, and run()."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, TextIO

from palaver.progress import stderr_status

from .surfaces import LintReport
from .tree import lint_provenance, lint_tree


def render_report(report: LintReport) -> str:
    """Render a `LintReport` as the command's stdout output."""
    lines = [
        "palaver fixture-lint",
        f"corpus: {report.root}",
        f"files: {report.files}",
        f"records: {report.records}",
        f"strings: {report.strings}",
        f"rejected: {len(report.rejections)}",
    ]
    if report.rejections:
        lines.append("")
        for rejection in report.rejections:
            lines.append(f"{rejection.path}:{rejection.line}: {rejection.rule}: {rejection.detail}")
        lines.extend(
            [
                "",
                "A rejected record is not classified, and an unclassified record does",
                "not ship: fix the fixture, or add its shape to the allowlist in",
                "palaver/cli/fixture_lint.py deliberately.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "every record matched an allowlisted shape and carried only",
                "phrasebook text; every quoted string on every other surface was",
                "phrasebook, a command, or a structural token, and no file was",
                "skipped for its extension.",
            ]
        )
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Register `fixture-lint`'s arguments on its subparser."""
    parser.add_argument(
        "path",
        type=Path,
        help="fixture corpus directory (or a single .jsonl fixture) to check",
    )
    parser.add_argument(
        "--provenance-source",
        type=Path,
        action="append",
        default=[],
        help="source file or directory to compare for verbatim copied prose (opt-in)",
    )


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver fixture-lint`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when every record classified, 1 when any record was rejected, and 2
        for a usage failure — a missing path or a corpus with no fixtures in
        it. The two non-zero codes are distinct deliberately: a test that
        asserts "the linter rejected my poisoned record" must be able to fail
        when what actually happened was that the path was wrong.
    """
    out = sys.stdout if out is None else out
    on_status = stderr_status if on_status is None else on_status

    root = Path(args.path)
    if not root.exists():
        print(f"palaver fixture-lint: no such path: {root}", file=sys.stderr)
        return 2

    report = lint_tree(root, on_status=on_status)
    sources = tuple(args.provenance_source)
    if sources:
        try:
            provenance = lint_provenance(root, sources)
        except FileNotFoundError as exc:
            print(f"palaver fixture-lint: provenance source unavailable: {exc}", file=sys.stderr)
            return 2
        report = LintReport(
            root=report.root,
            files=report.files,
            records=report.records,
            strings=report.strings,
            rejections=(*report.rejections, *provenance),
        )
    if report.files == 0:
        print(f"palaver fixture-lint: no files under {root}", file=sys.stderr)
        return 2

    out.write(render_report(report))
    return 1 if report.rejections else 0
