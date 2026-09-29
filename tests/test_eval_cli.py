"""
palaver.cli.eval's rendering (report/summary/aggregate) and its run() wiring, including
e2b teardown on failure.
"""

from __future__ import annotations

from palaver.cli import eval as eval_cli
from palaver.eval.harness import (
    EvalReport,
    LegMetrics,
)
from palaver.extract.client import ModelClientError

# =============================================================================
# palaver.cli.eval: rendering and wiring through the guaranteed-teardown context manager
# =============================================================================


def test_render_report_includes_every_metric_and_both_legs():
    report = EvalReport(
        fixture_ids=("a", "b"),
        per_leg={
            "E4B": LegMetrics(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2, 0),
            "E2B": LegMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 1),
        },
    )

    rendered = eval_cli.render_report(report)

    for _field_name, label in eval_cli._METRIC_LABELS:
        assert label in rendered
    assert "E4B" in rendered
    assert "E2B" in rendered
    assert "spread (max - min)" not in rendered


def test_render_summary_reports_fixture_and_leg_count():
    report = EvalReport(fixture_ids=("a", "b", "c"), per_leg={})
    assert eval_cli.render_summary(report) == "eval complete: 3 fixtures, 2 legs, 1 runs\n"


def test_aggregate_reports_includes_mean_and_observed_spread():
    first = EvalReport(
        fixture_ids=("a",),
        per_leg={
            "E4B": LegMetrics(1, 1, 1, 1, 0, 1, 2, 0),
            "E2B": LegMetrics(0, 0, 0, 0, 1, 0, 0, 0),
        },
    )
    second = EvalReport(
        fixture_ids=("a",),
        per_leg={
            "E4B": LegMetrics(0, 1, 1, 1, 0, 1, 0, 0),
            "E2B": LegMetrics(1, 0, 0, 0, 0, 0, 2, 1),
        },
    )

    report = eval_cli.aggregate_reports([first, second])

    assert report.run_count == 2
    assert report.per_leg["E4B"].question_detection_accuracy == 0.5
    assert report.spread_per_leg["E4B"]["question_detection_accuracy"] == 1
    assert "spread (max - min)" in eval_cli.render_report(report)


class _FakeArgs:
    def __init__(self, **overrides):
        self.report = False
        self.fixtures_dir = None
        self.labels = None
        self.db = None
        self.health_timeout = 30.0
        self.__dict__.update(overrides)


def test_cli_run_tears_down_e2b_server_even_when_run_eval_raises(monkeypatch, tmp_path):
    """CLI-level version of the Done-when teardown bullet: the CLI's own
    wiring goes through `managed_e2b_server`, so a failure inside the
    `with` block still tears the child process down."""
    teardown_calls = {"count": 0}

    from contextlib import contextmanager

    @contextmanager
    def _fake_managed_e2b_server(leg, *, health_timeout, on_status=None):
        try:
            yield object()
        finally:
            teardown_calls["count"] += 1

    def _fake_run_eval(*args, **kwargs):
        raise ModelClientError("simulated E2B leg failure")

    monkeypatch.setattr(eval_cli, "managed_e2b_server", _fake_managed_e2b_server)
    monkeypatch.setattr(eval_cli, "run_eval", _fake_run_eval)
    monkeypatch.setattr(eval_cli, "ModelClient", lambda conn, **kwargs: object())

    args = _FakeArgs(db=tmp_path / "eval.db")
    import io

    out = io.StringIO()
    status_lines: list[str] = []

    exit_code = eval_cli.run(args, out=out, on_status=status_lines.append)

    assert exit_code == 1
    assert teardown_calls["count"] == 1


def test_cli_run_reports_success_when_both_legs_complete(monkeypatch, tmp_path):
    """Positive control for the test above: a clean run exits 0 and prints a result."""

    from contextlib import contextmanager

    @contextmanager
    def _fake_managed_e2b_server(leg, *, health_timeout, on_status=None):
        yield object()

    fake_report = EvalReport(
        fixture_ids=("a",),
        per_leg={
            "E4B": LegMetrics(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0, 0),
            "E2B": LegMetrics(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0, 0),
        },
    )

    monkeypatch.setattr(eval_cli, "managed_e2b_server", _fake_managed_e2b_server)
    monkeypatch.setattr(eval_cli, "run_eval", lambda *a, **k: fake_report)
    monkeypatch.setattr(eval_cli, "ModelClient", lambda conn, **kwargs: object())

    args = _FakeArgs(db=tmp_path / "eval.db", report=True)
    import io

    out = io.StringIO()

    exit_code = eval_cli.run(args, out=out, on_status=lambda _msg: None)

    assert exit_code == 0
    assert "question detection" in out.getvalue()
