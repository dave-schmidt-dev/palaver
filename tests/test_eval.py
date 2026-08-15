"""Tests for `palaver.eval.harness` and `palaver.cli.eval` (task 3.5).

No test here ever points a client at the real port 8090 or 8091, and none
starts a real `llama-server` subprocess or needs a real GGUF file --
`managed_e2b_server`'s `popen`/`check_health` and `run_eval`'s `ModelClient`
arguments are all injectable for exactly this reason, the same discipline
`tests/test_model_client.py` documents for task 3.2's client.

Every negative assertion here is paired with a positive control on the same
input shape, per the plan's standing rule and task 3.5's own explicit
example: "no shutdown call against port 8090" is checked together with "at
least one ordinary inference request to port 8090" in the same test, because
the first half alone passes vacuously if client wiring never reaches port
8090 at all.

INV-9: every fixture referenced here is one of Phase 1's already-vetted
`tests/fixtures/*.jsonl` files or `tests/fixtures/eval/decision-database-choice.jsonl`,
which reuses only phrasebook strings from `palaver/cli/fixture_lint.py`'s
`SYNTHESIZED_TEXT` allowlist -- no prose here was copied from a real session.
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from palaver.cli import eval as eval_cli
from palaver.eval.harness import (
    DEFAULT_FIXTURES_DIR,
    DEFAULT_LABELS_PATH,
    E2B_LEG,
    E4B_LEG,
    Decision,
    EvalReport,
    Extraction,
    FixtureLabel,
    LegConfig,
    LegMetrics,
    assert_legs_distinct,
    build_prompt,
    is_decision_grounded,
    is_degenerate_extraction,
    load_labels,
    managed_e2b_server,
    resolve_llama_server_binary,
    run_eval,
    score_leg,
)
from palaver.extract.client import ModelClient, ModelClientError
from palaver.extract.normalize import normalize_path
from palaver.extract.quote_gate import ground_quote
from palaver.replay import replay
from palaver.store.migrate import connect

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_CONFORMING_EXTRACTION = {
    "current_task": "",
    "blockers_now": [],
    "questions_for_user": [],
    "user_decisions": [],
    "session_complete": False,
}


# =============================================================================
# Stub HTTP server (mirrors tests/test_model_client.py's pattern)
# =============================================================================


class _RecordingHandler(BaseHTTPRequestHandler):
    def __init__(self, requests, response_body, *args, **kwargs):
        self._requests = requests
        self._response_body = response_body
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self._requests.append(("POST", self.path))
        body = json.dumps(self._response_body()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._requests.append(("GET", self.path))
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self) -> None:
        self._requests.append(("DELETE", self.path))
        self.send_response(404)
        self.end_headers()


class _StubServer:
    def __init__(self, requests, response_body):
        def _factory(*args, **kwargs):
            return _RecordingHandler(requests, response_body, *args, **kwargs)

        self._server = HTTPServer(("127.0.0.1", 0), _factory)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_server():
    """Yield `start(requests, response_body) -> port`; every server is closed at teardown."""
    servers = []

    def _start(requests, response_body=lambda: _wrap(_CONFORMING_EXTRACTION)) -> int:
        server = _StubServer(requests, response_body)
        servers.append(server)
        return server.port

    yield _start

    for server in servers:
        server.close()


def _wrap(extraction: dict, *, prompt_tokens: int = 100) -> dict:
    content = json.dumps(extraction)
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens},
    }


@pytest.fixture
def store_conn(tmp_path):
    from palaver.store.migrate import migrate

    db_path = tmp_path / "palaver.db"
    migrate(db_path)
    conn = connect(db_path)
    yield conn
    conn.close()


# =============================================================================
# Leg identity, structural pins (advisor point 5, half 2: leg identity, not port)
# =============================================================================


def test_e4b_leg_is_pinned_to_8090_and_unmanaged():
    """E4B is the pre-existing server: fixed port, and this module never holds a Popen for it."""
    assert E4B_LEG.port == 8090
    assert E4B_LEG.host == "127.0.0.1"
    assert E4B_LEG.managed is False


def test_e2b_leg_is_pinned_to_8091_and_managed():
    """E2B is the leg this harness starts and stops itself."""
    assert E2B_LEG.port == 8091
    assert E2B_LEG.host == "127.0.0.1"
    assert E2B_LEG.managed is True


# =============================================================================
# assert_legs_distinct: mutation guard for "point both legs at the same GGUF"
# =============================================================================


def test_assert_legs_distinct_accepts_the_real_pinned_pair():
    """Positive control: the actual E4B/E2B pair task 3.5 pins must pass this guard."""
    assert_legs_distinct(E4B_LEG, E2B_LEG)  # must not raise


def test_assert_legs_distinct_rejects_same_model_path():
    same_path = E4B_LEG.model_path
    e2b_pointed_at_e4b = LegConfig(
        name="E2B", host="127.0.0.1", port=8091, model_path=same_path, model_name="x", managed=True
    )
    with pytest.raises(ValueError, match="different GGUFs"):
        assert_legs_distinct(E4B_LEG, e2b_pointed_at_e4b)


def test_assert_legs_distinct_rejects_same_port():
    e2b_on_e4b_port = LegConfig(
        name="E2B",
        host="127.0.0.1",
        port=8090,
        model_path=E2B_LEG.model_path,
        model_name="x",
        managed=True,
    )
    with pytest.raises(ValueError, match="different ports"):
        assert_legs_distinct(E4B_LEG, e2b_on_e4b_port)


def test_assert_legs_distinct_rejects_non_eval_gguf():
    bogus = LegConfig(
        name="E2B",
        host="127.0.0.1",
        port=8091,
        model_path=Path("/Users/dave/models/gemma-4-repaired/gemma-4-26B_q4_0-it.gguf"),
        model_name="x",
        managed=True,
    )
    with pytest.raises(ValueError, match="not an eval leg"):
        assert_legs_distinct(E4B_LEG, bogus)


# =============================================================================
# resolve_llama_server_binary
# =============================================================================


def test_resolve_llama_server_binary_returns_which_result(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/fake/bin/llama-server")
    assert resolve_llama_server_binary() == "/fake/bin/llama-server"


def test_resolve_llama_server_binary_raises_when_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        resolve_llama_server_binary()


# =============================================================================
# managed_e2b_server: guaranteed teardown (task 3.5's Done-when bullet)
# =============================================================================


class _FakeProcess:
    def __init__(self):
        self.terminate_calls = 0
        self.wait_calls = 0
        self.kill_calls = 0

    def terminate(self):
        self.terminate_calls += 1

    def wait(self, timeout=None):
        self.wait_calls += 1

    def kill(self):
        self.kill_calls += 1


def test_managed_e2b_server_terminates_process_when_body_raises():
    """The Done-when bullet: teardown happens even when the E2B leg raises."""
    fake_process = _FakeProcess()

    with pytest.raises(RuntimeError, match="boom"):
        with managed_e2b_server(
            E2B_LEG,
            binary="fake-llama-server",
            popen=lambda *a, **k: fake_process,
            check_health=lambda host, port: True,
        ):
            raise RuntimeError("boom")

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1


def test_managed_e2b_server_terminates_process_on_clean_exit():
    """Positive control for the above: teardown is not an artifact of the exception path."""
    fake_process = _FakeProcess()

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=lambda host, port: True,
    ) as process:
        assert process is fake_process

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1


def test_managed_e2b_server_raises_timeout_and_still_tears_down():
    fake_process = _FakeProcess()

    with pytest.raises(TimeoutError):
        with managed_e2b_server(
            E2B_LEG,
            binary="fake-llama-server",
            popen=lambda *a, **k: fake_process,
            check_health=lambda host, port: False,
            health_timeout=0.05,
            poll_interval=0.01,
        ):
            raise AssertionError("body must never run when health never reports ok")

    assert fake_process.terminate_calls == 1


def test_managed_e2b_server_polls_health_until_ready():
    fake_process = _FakeProcess()
    calls = {"count": 0}

    def flaky_health(host, port):
        calls["count"] += 1
        return calls["count"] >= 3

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=flaky_health,
        poll_interval=0.01,
    ):
        pass

    assert calls["count"] >= 3
    assert fake_process.terminate_calls == 1


def test_managed_e2b_server_never_calls_the_real_subprocess_module(monkeypatch):
    """Positive control that the injected `popen` is actually what gets called.

    Without this, a `managed_e2b_server` that silently ignored the `popen`
    argument and shelled out to the real `subprocess.Popen` would still pass
    every test above, since none of them inspect what started the process.
    """

    def _real_popen_must_not_be_called(*args, **kwargs):
        raise AssertionError("the real subprocess.Popen must not be called in this test")

    monkeypatch.setattr(subprocess, "Popen", _real_popen_must_not_be_called)
    fake_process = _FakeProcess()

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=lambda host, port: True,
    ):
        pass

    assert fake_process.terminate_calls == 1


# =============================================================================
# Anti-vacuity: E4B leg gets ordinary inference, never a shutdown-shaped call
# =============================================================================


def test_e4b_leg_receives_inference_but_no_shutdown_call(stub_server, store_conn):
    """Task 3.5's named vacuity trap, addressed directly.

    Checking only "no shutdown call was recorded" would pass trivially if
    `run_eval`'s client wiring never reached the E4B leg's server at all.
    This test also asserts the positive half in the same body: the E4B
    leg's recorded request set contains at least one
    `/v1/chat/completions` POST, so the negative half is meaningful.

    The E4B leg's stub stands in for port 8090 by *role* (it is passed as
    `e4b_client` to `run_eval`, the same parameter production code points at
    `E4B_LEG`), not by literal port number -- `test_e4b_leg_is_pinned_to_8090_and_unmanaged`
    above pins the real port/managed-ness separately, since binding an
    ephemeral test port to 8090 itself is not possible.
    """
    e4b_requests: list[tuple[str, str]] = []
    e2b_requests: list[tuple[str, str]] = []
    e4b_port = stub_server(e4b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))
    e2b_port = stub_server(e2b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))

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
    e4b_client = ModelClient(store_conn, port=e4b_port, timeout=5.0)
    e2b_client = ModelClient(store_conn, port=e2b_port, timeout=5.0)

    run_eval(labels, FIXTURES_DIR, e4b_client=e4b_client, e2b_client=e2b_client)

    e4b_paths = {path for _method, path in e4b_requests}
    # Positive control: the wiring actually reached the E4B leg's server.
    assert len(e4b_requests) >= 1
    # Negative: an allowlist of paths actually seen, not a blacklist of
    # shutdown-ish names -- the only path a `ModelClient` ever POSTs to.
    assert e4b_paths == {"/v1/chat/completions"}
    assert all(method == "POST" for method, _path in e4b_requests)


# =============================================================================
# is_decision_grounded: fabrication / misattribution check
# =============================================================================


def test_is_decision_grounded_true_for_human_line():
    transcript = "HUMAN: refactor the auth module\nAGENT: the auth module is refactored\n"
    assert is_decision_grounded(transcript, "refactor the auth module") is True


def test_is_decision_grounded_false_for_injected_line():
    """Misattribution: the quote is real text, but on an INJECTED, not HUMAN, line."""
    transcript = "HUMAN: refactor the auth module\nINJECTED: <command-name>/status</command-name>\n"
    assert is_decision_grounded(transcript, "<command-name>/status</command-name>") is False


def test_is_decision_grounded_false_for_agent_line():
    """Fabrication: a real AGENT statement must never ground a user decision."""
    transcript = "HUMAN: refactor the auth module\nAGENT: the auth module is refactored\n"
    assert is_decision_grounded(transcript, "the auth module is refactored") is False


def test_is_decision_grounded_false_for_tool_result_line():
    transcript = "HUMAN: run the test suite\n  result[error]> command not found\n"
    assert is_decision_grounded(transcript, "command not found") is False


def test_is_decision_grounded_false_for_empty_quote():
    transcript = "HUMAN: refactor the auth module\n"
    assert is_decision_grounded(transcript, "") is False


def test_is_decision_grounded_parity_with_quote_gate_ground_quote(tmp_path):
    """Pins the harness's decoupled grounding check to the production gate.

    Replays two real fixtures through the actual pipeline (adapter ->
    `classify_channel` -> normalizer -> `transcript_chunks`), same pattern
    `tests/test_extraction.py` uses, so both checks run against a chunk the
    real classifier produced -- not a hand-typed string that could drift
    from what `classify_channel` actually does.
    """
    db_path = tmp_path / "palaver.db"

    human_result = replay(FIXTURES_DIR / "slash-command-after-reply.jsonl", db_path)
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, content FROM transcript_chunks WHERE session_id = ? ORDER BY seq",
            (human_result.session_id,),
        ).fetchall()
        human_chunk_id, human_content = rows[0]
        injected_chunk_id, injected_content = rows[2]
        assert human_content.startswith("HUMAN: ")
        assert injected_content.startswith("INJECTED: ")

        human_quote = "refactor the auth module"
        gate_verdict = ground_quote(
            conn, transcript_chunk_id=human_chunk_id, quote=human_quote, statement=human_quote
        )
        harness_verdict = is_decision_grounded(
            normalize_path(FIXTURES_DIR / "slash-command-after-reply.jsonl"), human_quote
        )
        assert gate_verdict.is_tier_one is True
        assert harness_verdict is True

        injected_quote = "<command-name>/status</command-name>"
        gate_verdict_injected = ground_quote(
            conn,
            transcript_chunk_id=injected_chunk_id,
            quote=injected_quote,
            statement=injected_quote,
        )
        harness_verdict_injected = is_decision_grounded(
            normalize_path(FIXTURES_DIR / "slash-command-after-reply.jsonl"), injected_quote
        )
        assert gate_verdict_injected.is_tier_one is False
        assert harness_verdict_injected is False
    finally:
        conn.close()


# =============================================================================
# score_leg
# =============================================================================


def _label(id_, **overrides):
    defaults = dict(
        path=f"{id_}.jsonl",
        expect_question=False,
        expect_blocker=False,
        expect_current_task=False,
        expect_decision=False,
        expect_completion=False,
        forbidden_quote_substrings=(),
    )
    defaults.update(overrides)
    return FixtureLabel(id=id_, **defaults)


def test_score_leg_perfect_extraction_scores_full_marks():
    labels = (
        _label("has-task", expect_current_task=True),
        _label("no-signal"),
    )
    extractions = {
        "has-task": Extraction("do the thing", (), (), (), False),
        "no-signal": Extraction("", (), (), (), False),
    }
    transcripts = {"has-task": "HUMAN: x\n", "no-signal": ""}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.current_task_accuracy == 1.0
    assert metrics.question_detection_accuracy == 1.0
    assert metrics.blocker_detection_accuracy == 1.0
    assert metrics.decision_retention_accuracy == 1.0
    assert metrics.completion_detection_accuracy == 1.0
    assert metrics.false_decision_rate == 0.0
    assert metrics.decisions_extracted == 0


def test_score_leg_penalizes_ungrounded_decision_as_false():
    labels = (_label("with-decision", expect_decision=True),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="chose postgres", quote="postgres"),), False
        )
    }
    # "postgres" never appears on a HUMAN line here -- ungrounded.
    transcripts = {"with-decision": "AGENT: we will use postgres\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.decisions_extracted == 1
    assert metrics.false_decisions == 1
    assert metrics.false_decision_rate == 1.0


def test_score_leg_grounded_decision_is_not_false():
    """Positive control for the test above: a correctly grounded decision must not be penalized."""
    labels = (_label("with-decision", expect_decision=True),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="chose postgres", quote="postgres"),), False
        )
    }
    transcripts = {"with-decision": "HUMAN: postgres\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.false_decisions == 0
    assert metrics.false_decision_rate == 0.0


def test_score_leg_forbidden_substring_is_false_even_if_grounded():
    """`forbidden_quote_substrings` catches a quote that is technically HUMAN-grounded
    but named as a known-bad attribution target for a specific fixture."""
    labels = (_label("with-decision", expect_decision=True, forbidden_quote_substrings=("no",)),)
    extractions = {
        "with-decision": Extraction(
            "", (), (), (Decision(statement="declined", quote="no"),), False
        )
    }
    transcripts = {"with-decision": "HUMAN: no\n"}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.false_decisions == 1


def test_score_leg_zero_decisions_extracted_gives_zero_rate_not_a_crash():
    labels = (_label("no-signal"),)
    extractions = {"no-signal": Extraction("", (), (), (), False)}
    transcripts = {"no-signal": ""}

    metrics = score_leg(labels, extractions, transcripts)

    assert metrics.decisions_extracted == 0
    assert metrics.false_decision_rate == 0.0


# =============================================================================
# is_degenerate_extraction
# =============================================================================


def test_is_degenerate_extraction_flags_empty_field_when_demonstrably_present():
    labels = (_label("has-blocker", expect_blocker=True),)
    extractions = {"has-blocker": Extraction("", (), (), (), False)}

    offenders = is_degenerate_extraction(labels, extractions)

    assert offenders == ["has-blocker"]


def test_is_degenerate_extraction_empty_when_fields_are_present():
    """Positive control: a non-degenerate extractor produces no offenders."""
    labels = (_label("has-blocker", expect_blocker=True),)
    extractions = {"has-blocker": Extraction("", ("tests fail",), (), (), False)}

    offenders = is_degenerate_extraction(labels, extractions)

    assert offenders == []


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


def test_render_summary_reports_fixture_and_leg_count():
    report = EvalReport(fixture_ids=("a", "b", "c"), per_leg={})
    assert eval_cli.render_summary(report) == "eval complete: 3 fixtures, 2 legs (E4B, E2B)\n"


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
