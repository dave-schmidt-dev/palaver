"""`palaver eval [--report]`: score extraction quality, E4B against E2B (task 3.5).

Same split as every other subcommand in this package: this module owns
argparse wiring and stdout/stderr framing (INV-1: stdout is the result,
stderr is progress), and none of the harness/scoring logic, which lives in
`palaver.eval.harness` so it can be called directly from tests without going
through argument parsing or a real model server.

Runs the E4B leg against the pre-existing `llama-server` on port 8090
(never started or stopped here) and the E2B leg against a `llama-server`
this command starts on port 8091 and always tears down, via
`palaver.eval.harness.managed_e2b_server`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable, TextIO

from palaver.eval.harness import (
    DEFAULT_FIXTURES_DIR,
    DEFAULT_LABELS_PATH,
    E2B_LEG,
    E4B_LEG,
    EvalReport,
    LegMetrics,
    ModelClient,
    assert_legs_distinct,
    load_labels,
    managed_e2b_server,
    run_eval,
)
from palaver.extract.client import ModelClientError
from palaver.store.migrate import connect, migrate

NAME = "eval"
HELP = "score extraction quality across the E4B and E2B model legs on labelled fixtures"

#: Deliberately outside the repository tree, mirroring `replay.py`'s
#: `DEFAULT_DB_PATH` — this command's `model_runs` bookkeeping has no
#: project-local home yet either, and must never write into the repo.
DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "palaver-eval" / "eval.db"

_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("question_detection_accuracy", "question detection"),
    ("blocker_detection_accuracy", "blocker detection"),
    ("current_task_accuracy", "current-task extraction"),
    ("decision_retention_accuracy", "decision retention"),
    ("false_decision_rate", "false-decision rate"),
    ("completion_detection_accuracy", "completion detection"),
)


def add_arguments(parser) -> None:
    """Register `eval`'s flags on its subparser."""
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the per-metric table for both model legs (default: a one-line summary)",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=None,
        help="directory labelled fixture paths are resolved against (default: tests/fixtures)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="path to labels.json (default: tests/fixtures/eval/labels.json)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"database file for model_runs bookkeeping (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the E2B server to report healthy (default: 120)",
    )


def render_report(report: EvalReport) -> str:
    """Render a per-metric table for both legs — `--report`'s output."""
    header = f"{'metric':<28}{'E4B':>12}{'E2B':>12}"
    lines = [header, "-" * len(header)]
    for field_name, label in _METRIC_LABELS:
        e4b_value = _format_metric(report.per_leg["E4B"], field_name)
        e2b_value = _format_metric(report.per_leg["E2B"], field_name)
        lines.append(f"{label:<28}{e4b_value:>12}{e2b_value:>12}")
    lines.append("")
    for leg_name in ("E4B", "E2B"):
        metrics = report.per_leg[leg_name]
        lines.append(
            f"{leg_name}: {metrics.decisions_extracted} decisions extracted, "
            f"{metrics.false_decisions} false"
        )
    return "".join(f"{line}\n" for line in lines)


def _format_metric(metrics: LegMetrics, field_name: str) -> str:
    value = getattr(metrics, field_name)
    return f"{value:.2f}"


def render_summary(report: EvalReport) -> str:
    """Render the default (non `--report`) one-line summary."""
    return f"eval complete: {len(report.fixture_ids)} fixtures, 2 legs (E4B, E2B)\n"


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver eval`.

    Args:
        args: Parsed arguments from this subcommand's parser (`report`,
            `fixtures_dir`, `labels`, `db`, `health_timeout`).
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 having scored both legs and printed a result. 1 on a diagnosable
        setup failure (labels file missing, `llama-server` not on `PATH`,
        the E2B server never reporting healthy).
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status
    db_path = DEFAULT_DB_PATH if args.db is None else args.db

    try:
        assert_legs_distinct(E4B_LEG, E2B_LEG)
        labels_path = args.labels if args.labels is not None else DEFAULT_LABELS_PATH
        labels = load_labels(labels_path)
        fixtures_dir = args.fixtures_dir if args.fixtures_dir is not None else DEFAULT_FIXTURES_DIR
    except (OSError, ValueError) as exc:
        print(f"palaver eval: {exc}", file=sys.stderr)
        return 1

    db_path.parent.mkdir(parents=True, exist_ok=True)
    migrate(db_path)
    conn = connect(db_path)
    try:
        e4b_client = ModelClient(conn, host=E4B_LEG.host, port=E4B_LEG.port)
        try:
            with managed_e2b_server(
                E2B_LEG, health_timeout=args.health_timeout, on_status=on_status
            ):
                e2b_client = ModelClient(conn, host=E2B_LEG.host, port=E2B_LEG.port)
                report = run_eval(
                    labels,
                    fixtures_dir,
                    e4b_client=e4b_client,
                    e2b_client=e2b_client,
                    on_status=on_status,
                )
        except (OSError, TimeoutError, FileNotFoundError, ModelClientError) as exc:
            print(f"palaver eval: {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()

    out.write(render_report(report) if args.report else render_summary(report))
    return 0
