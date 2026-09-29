"""
The labelled fixture corpus and run_eval end to end against the stub model servers, plus
the real-model anti-degeneracy check.
"""

from __future__ import annotations

import json

import pytest

from palaver.eval.harness import (
    DEFAULT_FIXTURES_DIR,
    DEFAULT_LABELS_PATH,
    FixtureLabel,
    build_prompt,
    is_degenerate_extraction,
    load_labels,
    run_eval,
)
from palaver.extract.client import ModelClient, ModelClientError
from tests._eval_support import _CONFORMING_EXTRACTION, FIXTURES_DIR, _wrap
from tests._eval_support import store_conn as store_conn
from tests._eval_support import stub_server as stub_server

# =============================================================================
# Labelled fixture corpus integration
# =============================================================================


def test_load_labels_loads_the_real_labels_json_and_every_fixture_exists():
    labels = load_labels(DEFAULT_LABELS_PATH)

    assert len(labels) >= 6
    ids = [label.id for label in labels]
    assert len(ids) == len(set(ids)), "fixture ids must be unique"
    for label in labels:
        fixture_path = DEFAULT_FIXTURES_DIR / label.path
        assert fixture_path.is_file(), f"{label.path} referenced by labels.json does not exist"


def test_default_fixtures_dir_matches_the_tests_fixtures_directory():
    assert DEFAULT_FIXTURES_DIR == FIXTURES_DIR


def test_build_prompt_includes_transcript_and_channel_legend():
    prompt = build_prompt("HUMAN: refactor the auth module\n")
    assert "HUMAN: refactor the auth module" in prompt
    assert "INJECTED" in prompt
    assert "AGENT" in prompt


def test_build_prompt_handles_empty_transcript():
    prompt = build_prompt("")
    assert "(empty transcript)" in prompt


# =============================================================================
# run_eval: end-to-end wiring against stub servers, at least two fixtures
# =============================================================================


def test_run_eval_scores_both_legs_over_the_labelled_corpus(stub_server, store_conn):
    labels = load_labels(DEFAULT_LABELS_PATH)

    e4b_requests: list[tuple[str, str]] = []
    e2b_requests: list[tuple[str, str]] = []
    e4b_port = stub_server(e4b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))
    e2b_port = stub_server(e2b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))
    e4b_client = ModelClient(store_conn, port=e4b_port, timeout=5.0)
    e2b_client = ModelClient(store_conn, port=e2b_port, timeout=5.0)

    report = run_eval(labels, DEFAULT_FIXTURES_DIR, e4b_client=e4b_client, e2b_client=e2b_client)

    assert set(report.per_leg) == {"E4B", "E2B"}
    assert len(e4b_requests) == len(labels)
    assert len(e2b_requests) == len(labels)
    for metrics in report.per_leg.values():
        assert 0.0 <= metrics.question_detection_accuracy <= 1.0
        assert 0.0 <= metrics.false_decision_rate <= 1.0

    run_count = store_conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
    assert run_count == 2 * len(labels)


def test_run_eval_records_model_run_failure_and_raises(stub_server, store_conn):
    """Positive control for the model_runs bookkeeping above: a failing call
    must still be visible in `model_runs` (status="error"), not silently dropped."""
    e4b_requests: list[tuple[str, str]] = []

    def _bad_response():
        return {"choices": [{"message": {"content": "not json at all {{{"}}]}

    e4b_port = stub_server(e4b_requests, _bad_response)
    e2b_port = stub_server([], lambda: _wrap(_CONFORMING_EXTRACTION))
    e4b_client = ModelClient(store_conn, port=e4b_port, timeout=5.0)
    e2b_client = ModelClient(store_conn, port=e2b_port, timeout=5.0)
    labels = (
        FixtureLabel(
            id="bookkeeping-only",
            path="bookkeeping-only.jsonl",
            expect_question=False,
            expect_blocker=False,
            expect_current_task=False,
            expect_decision=False,
            expect_completion=False,
        ),
    )

    with pytest.raises(ModelClientError):
        run_eval(labels, FIXTURES_DIR, e4b_client=e4b_client, e2b_client=e2b_client)

    statuses = [row[0] for row in store_conn.execute("SELECT status FROM model_runs").fetchall()]
    assert "error" in statuses


# =============================================================================
# test_extraction_is_not_degenerate -- named in the Phase 3 acceptance line
# =============================================================================


def test_extraction_is_not_degenerate():
    """On every labelled fixture where a task, blocker, or question demonstrably
    exists, the extractor must return a non-empty field of that kind.

    Replays a committed snapshot of a real E4B run's raw extraction JSON
    (`tests/fixtures/eval/e4b_snapshot.json`, captured by an actual
    `palaver eval` run against the live model on port 8090) through the real
    parsing and scoring path (`extraction_from_json` -> `is_degenerate_extraction`),
    so this test needs no running model server and exercises the harness's
    own pipeline code rather than a hand-invented stub extraction. It cannot
    catch a prompt-wording regression that changes what the live model
    outputs -- only a genuine end-to-end `palaver eval --report` run does
    that.
    """
    from palaver.eval.harness import extraction_from_json

    snapshot_path = DEFAULT_FIXTURES_DIR / "eval" / "e4b_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    labels = load_labels(DEFAULT_LABELS_PATH)

    extractions = {
        label.id: extraction_from_json(snapshot[label.id])
        for label in labels
        if label.id in snapshot
    }
    assert extractions, "snapshot must cover at least one labelled fixture"

    scored_labels = [label for label in labels if label.id in extractions]
    offenders = is_degenerate_extraction(scored_labels, extractions)

    assert offenders == [], f"degenerate extraction on: {offenders}"
