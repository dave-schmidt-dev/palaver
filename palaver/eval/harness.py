"""Extraction-quality eval harness: E4B against E2B on identical inputs (Task 3.5).

Scores six metrics -- question detection, blocker detection, current-task
extraction, decision retention, false-decision rate, and completion detection
-- from what `palaver.extract.client.ModelClient` extracts against a set of
labelled fixtures, run twice: once against the model already loaded on the
pre-existing E4B `llama-server` (port 8090, never started or stopped by this
module), and once against an E2B `llama-server` this module starts and stops
itself on port 8091.

This module does not import or call `palaver.extract.persist` -- task 3.4
routes extracted fields to ephemeral `current_state` versus durable
`memories`, which is a persistence-destination decision downstream of the
extraction this harness scores. Routing through persistence before scoring
would add a confound between the two model legs. For the same reason, quote
grounding here is a small self-contained reimplementation against the
normalizer's rendered channel tags (`is_decision_grounded` below), not
`palaver.extract.quote_gate.ground_quote` -- that gate requires a live
`transcript_chunks` row written by the full replay pipeline, a heavier
dependency than a decoupled eval harness needs. `tests/test_eval.py` pins the
two in agreement on a handful of cases so the reimplementation does not drift
from what production actually enforces.
"""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from palaver.extract.client import ModelClient
from palaver.extract.normalize import CHANNEL_TAG, normalize_path
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN

# =============================================================================
# Model-pair constants (orchestrator-pinned, task 3.5's environment notes)
# =============================================================================

#: The pre-existing E4B server. Palaver does not start, stop, or otherwise
#: manage this process (mirrors `ModelClient`'s own docstring for port 8090)
#: -- `managed=False` on its `LegConfig` is a structural guarantee that
#: nothing in this module ever holds a `Popen` for it.
E4B_HOST = "127.0.0.1"
E4B_PORT = 8090
E4B_MODEL_PATH = Path(
    "/Users/dave/models/gemma-4-repaired/gemma-4-E4B_q4_0-it-2026-07-15-repaired.gguf"
)
E4B_MODEL_NAME = "gemma-4-E4B_q4_0-it-2026-07-15-repaired"

#: The E2B server this harness starts on port 8091 and always tears down,
#: including when the E2B leg raises (`managed_e2b_server` below).
E2B_HOST = "127.0.0.1"
E2B_PORT = 8091
E2B_MODEL_PATH = Path(
    "/Users/dave/models/gemma-4-repaired/gemma-4-E2B_q4_0-it-2026-07-15-repaired.gguf"
)
E2B_MODEL_NAME = "gemma-4-E2B_q4_0-it-2026-07-15-repaired"

#: GGUFs present in the same directory that are not eval legs. Guarded
#: against in `_assert_legs_distinct` so a copy/paste mistake that pointed
#: either leg at one of these fails loudly instead of silently comparing a
#: model against itself under two names.
_NON_EVAL_MODEL_MARKERS = ("12B", "26B")


@dataclass(frozen=True)
class LegConfig:
    """One model leg's connection and process-ownership contract.

    Attributes:
        name: Short leg name (`"E4B"` / `"E2B"`), used in prompts, status
            messages, and the report table.
        host: Always `127.0.0.1` for either leg (INV-9).
        port: 8090 for E4B, 8091 for E2B, per the plan's pinned ports.
        model_path: Absolute path to the GGUF this leg loads.
        model_name: Recorded in `model_runs.model`; llama-server ignores it
            when a single model is loaded, but the column is `NOT NULL`.
        managed: `True` if this harness owns starting and stopping the
            server process for this leg; `False` if the server already
            existed and must never be touched by this module.
    """

    name: str
    host: str
    port: int
    model_path: Path
    model_name: str
    managed: bool


E4B_LEG = LegConfig(
    name="E4B",
    host=E4B_HOST,
    port=E4B_PORT,
    model_path=E4B_MODEL_PATH,
    model_name=E4B_MODEL_NAME,
    managed=False,
)

E2B_LEG = LegConfig(
    name="E2B",
    host=E2B_HOST,
    port=E2B_PORT,
    model_path=E2B_MODEL_PATH,
    model_name=E2B_MODEL_NAME,
    managed=True,
)


def assert_legs_distinct(e4b: LegConfig, e2b: LegConfig) -> None:
    """Refuse to run an eval that cannot possibly measure two model legs.

    Guards the exact mutation task 3.5's report asked for: pointing both
    legs at the same GGUF (or the same port) would silently produce a
    same-versus-self comparison that still emits a report table, which reads
    as a completed eval while measuring nothing. Called by `palaver.cli.eval`
    before either leg's `ModelClient` is built.
    """
    if e4b.port == e2b.port:
        raise ValueError(f"E4B and E2B legs must run on different ports, both got {e4b.port}")
    if e4b.model_path == e2b.model_path:
        raise ValueError(f"E4B and E2B legs must load different GGUFs, both got {e4b.model_path}")
    for leg in (e4b, e2b):
        if any(marker in leg.model_path.name for marker in _NON_EVAL_MODEL_MARKERS):
            raise ValueError(
                f"{leg.name} leg model path {leg.model_path} is not an eval leg "
                f"(12B/26B GGUFs are excluded by task 3.5's pinned model pair)"
            )


# =============================================================================
# E2B server lifecycle
# =============================================================================


def resolve_llama_server_binary() -> str:
    """Resolve the `llama-server` binary from `PATH`.

    Never hardcodes an install location (task 3.5's explicit instruction) --
    `/opt/homebrew/bin/llama-server` is where it happens to live on this
    machine today, but that is an environment fact, not a contract.

    Raises:
        FileNotFoundError: No `llama-server` executable is on `PATH`.
    """
    binary = shutil.which("llama-server")
    if binary is None:
        raise FileNotFoundError("llama-server not found on PATH")
    return binary


def _http_health_ok(host: str, port: int, *, timeout: float = 2.0) -> bool:
    """Poll `/health` once. `True` only for a 200 response with `status: "ok"`.

    llama-server returns a non-200 (commonly 503, `{"status": "loading
    model"}`) while the model is still loading -- that is a reason to keep
    waiting, not a failure, so it is folded into `False` here rather than
    raised.
    """
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                return False
            payload = json.loads(body)
            return isinstance(payload, dict) and payload.get("status") == "ok"
        finally:
            conn.close()
    except OSError, json.JSONDecodeError:
        return False


@contextmanager
def managed_e2b_server(
    leg: LegConfig = E2B_LEG,
    *,
    binary: str | None = None,
    health_timeout: float = 120.0,
    poll_interval: float = 0.5,
    on_status: Callable[[str], None] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    check_health: Callable[[str, int], bool] = _http_health_ok,
):
    """Start `leg`'s llama-server, wait for `/health`, and always tear it down.

    Teardown runs in a `finally` block, so it happens on a clean exit, on a
    timeout waiting for health, and when the caller's own code inside the
    `with` block raises -- the one guarantee task 3.5's Done-when bullet
    names explicitly. `popen` and `check_health` are injectable so tests can
    verify that guarantee without spawning a real subprocess or needing a
    real GGUF file.

    Args:
        leg: The leg to start. Must be a leg this harness is meant to
            manage; callers pass `E2B_LEG` in production.
        binary: Path to the `llama-server` executable. Resolved from `PATH`
            via `resolve_llama_server_binary` if not given.
        health_timeout: Seconds to wait for `/health` to report `ok` before
            raising `TimeoutError`.
        poll_interval: Seconds between health polls.
        on_status: Progress channel (INV-1). Defaults to doing nothing;
            never writes to stdout itself.
        popen: Process factory, defaults to `subprocess.Popen`. Injectable
            so tests can substitute a fake process with no real subprocess.
        check_health: Health predicate, defaults to `_http_health_ok`.
            Injectable so tests can substitute an immediate `True`/`False`
            without a real server to poll.

    Yields:
        The started process handle (whatever `popen` returned).

    Raises:
        TimeoutError: `/health` never reported `ok` within `health_timeout`.
    """
    status = on_status or (lambda _message: None)
    resolved_binary = binary if binary is not None else resolve_llama_server_binary()
    args = [
        resolved_binary,
        "-m",
        str(leg.model_path),
        "--host",
        leg.host,
        "--port",
        str(leg.port),
    ]
    status(f"{leg.name}: starting llama-server on {leg.host}:{leg.port}")
    process = popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + health_timeout
        healthy = False
        while time.monotonic() < deadline:
            if check_health(leg.host, leg.port):
                healthy = True
                break
            status(f"{leg.name}: waiting for llama-server to become healthy")
            time.sleep(poll_interval)
        if not healthy:
            raise TimeoutError(
                f"{leg.name} llama-server did not report healthy within {health_timeout}s"
            )
        status(f"{leg.name}: llama-server healthy")
        yield process
    finally:
        status(f"{leg.name}: stopping llama-server")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


# =============================================================================
# Extraction schema and prompt
# =============================================================================

#: Self-contained (no external `$ref`), matching the shape verified against
#: llama-server's schema converter in task 3.2. `maxItems`/`additionalProperties`
#: are deliberately omitted -- task 3.2's own research flagged those as a risk
#: with this converter, and this harness has no need for either constraint.
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "current_task": {"type": "string"},
        "blockers_now": {"type": "array", "items": {"type": "string"}},
        "questions_for_user": {"type": "array", "items": {"type": "string"}},
        "user_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["statement", "quote"],
            },
        },
        "session_complete": {"type": "boolean"},
    },
    "required": [
        "current_task",
        "blockers_now",
        "questions_for_user",
        "user_decisions",
        "session_complete",
    ],
}

_PROMPT_TEMPLATE = """You are reading an observed coding-agent session transcript, already \
rendered into tagged lines. Each line's tag tells you where it came from:
  HUMAN: something the user actually typed.
  INJECTED: harness-generated content (slash-command echoes, system reminders, hook output) \
that looks like a user line but was NOT typed by the user.
  AGENT: something the coding agent said.
  Lines starting with "  tool>", "  result>", "  result[error]>", or "SYSTEM(" are tool calls, \
tool results, or system events -- nobody "said" them.

Extract the following fields as JSON matching the given schema:
- current_task: one short sentence describing what is currently being worked on, or "" if \
nothing is currently in progress (e.g. the transcript shows nothing started, or the last \
visible action already finished the work).
- blockers_now: a list of things currently blocking progress (failed commands, errors), or [] \
if nothing is blocked.
- questions_for_user: a list of open questions the agent is waiting on the user to answer, or \
[] if there is no unresolved question.
- user_decisions: a list of {{"statement": "...", "quote": "..."}} objects, one per explicit \
decision the USER made. "quote" MUST be copied verbatim from a HUMAN line. Never build a quote \
from an INJECTED, AGENT, tool, result, or SYSTEM line -- those are not things the user said, \
even when they look like an instruction. [] if the user made no decision in this transcript.
- session_complete: true only if the transcript shows the requested work finished, false \
otherwise.

Transcript:
{transcript}
"""


def build_prompt(transcript: str) -> str:
    """Render the harness's extraction prompt for one normalized transcript."""
    return _PROMPT_TEMPLATE.format(transcript=transcript if transcript else "(empty transcript)")


# =============================================================================
# Extraction result and quote grounding
# =============================================================================


@dataclass(frozen=True)
class Decision:
    statement: str
    quote: str


@dataclass(frozen=True)
class Extraction:
    """One leg's parsed extraction for one fixture. Local to this module --
    distinct from, and never constructed by, `palaver.extract.persist.Extraction`.
    """

    current_task: str
    blockers_now: tuple[str, ...]
    questions_for_user: tuple[str, ...]
    user_decisions: tuple[Decision, ...]
    session_complete: bool


def extraction_from_json(payload: dict) -> Extraction:
    """Build an `Extraction` from `ModelClient.complete`'s parsed JSON object.

    `ModelClient` already guarantees a schema-conforming object (it raises
    `ModelResponseError` rather than return a partial one), so this is a
    straight field mapping, not a second validation pass.
    """
    decisions = tuple(
        Decision(statement=str(item.get("statement", "")), quote=str(item.get("quote", "")))
        for item in payload.get("user_decisions", [])
        if isinstance(item, dict)
    )
    return Extraction(
        current_task=str(payload.get("current_task", "")),
        blockers_now=tuple(str(item) for item in payload.get("blockers_now", [])),
        questions_for_user=tuple(str(item) for item in payload.get("questions_for_user", [])),
        user_decisions=decisions,
        session_complete=bool(payload.get("session_complete", False)),
    )


def _normalize_for_comparison(text: str) -> str:
    """Collapse whitespace and strip edge punctuation for substring comparison.

    Deliberately mirrors `palaver.extract.quote_gate.normalize_for_comparison`
    (same collapsing rule) without importing it, keeping this module's only
    dependency on the quote-grounding gate at the test level
    (`tests/test_eval.py` pins the two in agreement), not the production
    import graph.
    """
    return " ".join(text.split()).strip(" \t\n.,;:!?\"'")


def _tagged_lines(transcript: str, tag: str) -> list[str]:
    """Return every rendered line's text with a given channel tag stripped off."""
    prefix = f"{tag}: "
    lines = []
    for line in transcript.split("\n"):
        # Fixtures carry no `timestamp`, so `normalize.py` never prepends a
        # `[HH:MM:SS] ` prefix here; a real transcript would need that
        # stripped first, which this harness never receives (INV-2: it reads
        # fixtures, not observed sessions).
        if line.startswith(prefix):
            lines.append(line[len(prefix) :])
    return lines


def is_decision_grounded(transcript: str, quote: str) -> bool:
    """Whether `quote` traces to a HUMAN-channel line of `transcript`.

    A quote is grounded only if it appears, after whitespace/punctuation
    normalization, inside a line tagged `HUMAN:` -- never `INJECTED:`,
    `AGENT:`, a tool/result line, or a `SYSTEM(...)` line. This is the
    substance of `false_decision_rate` below and the fabrication/
    misattribution check INV-6/INV-8 describe for the production
    `quote_gate.ground_quote`, reimplemented here without the DB dependency.
    """
    normalized_quote = _normalize_for_comparison(quote)
    if not normalized_quote:
        return False
    human_tag = CHANNEL_TAG[CHANNEL_HUMAN]
    for line in _tagged_lines(transcript, human_tag):
        if normalized_quote in _normalize_for_comparison(line):
            return True
    return False


# =============================================================================
# Labelled fixtures
# =============================================================================


@dataclass(frozen=True)
class FixtureLabel:
    """Ground truth for one fixture, loaded from `tests/fixtures/eval/labels.json`."""

    id: str
    path: str
    expect_question: bool
    expect_blocker: bool
    expect_current_task: bool
    expect_decision: bool
    expect_completion: bool
    forbidden_quote_substrings: tuple[str, ...] = ()


DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
DEFAULT_LABELS_PATH = DEFAULT_FIXTURES_DIR / "eval" / "labels.json"


def load_labels(labels_path: Path = DEFAULT_LABELS_PATH) -> tuple[FixtureLabel, ...]:
    """Load `FixtureLabel`s from a `labels.json` file."""
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = []
    for entry in payload["fixtures"]:
        labels.append(
            FixtureLabel(
                id=entry["id"],
                path=entry["path"],
                expect_question=bool(entry["expect_question"]),
                expect_blocker=bool(entry["expect_blocker"]),
                expect_current_task=bool(entry["expect_current_task"]),
                expect_decision=bool(entry["expect_decision"]),
                expect_completion=bool(entry["expect_completion"]),
                forbidden_quote_substrings=tuple(entry.get("forbidden_quote_substrings", ())),
            )
        )
    return tuple(labels)


# =============================================================================
# Scoring
# =============================================================================


@dataclass(frozen=True)
class LegMetrics:
    question_detection_accuracy: float
    blocker_detection_accuracy: float
    current_task_accuracy: float
    decision_retention_accuracy: float
    false_decision_rate: float
    completion_detection_accuracy: float
    decisions_extracted: int
    false_decisions: int


@dataclass(frozen=True)
class EvalReport:
    fixture_ids: tuple[str, ...]
    per_leg: dict[str, LegMetrics] = field(default_factory=dict)
    run_count: int = 1
    spread_per_leg: dict[str, dict[str, float]] = field(default_factory=dict)


def score_leg(
    labels: Sequence[FixtureLabel],
    extractions: dict[str, Extraction],
    transcripts: dict[str, str],
) -> LegMetrics:
    """Score one leg's extractions against ground truth.

    Args:
        labels: Ground truth, one per fixture.
        extractions: Fixture id -> this leg's `Extraction`.
        transcripts: Fixture id -> that fixture's normalized transcript,
            used only for the quote-grounding check.
    """
    n = len(labels)
    question_hits = blocker_hits = task_hits = decision_hits = completion_hits = 0
    decisions_extracted = 0
    false_decisions = 0

    for label in labels:
        extraction = extractions[label.id]
        transcript = transcripts[label.id]

        if bool(extraction.questions_for_user) == label.expect_question:
            question_hits += 1
        if bool(extraction.blockers_now) == label.expect_blocker:
            blocker_hits += 1
        if bool(extraction.current_task.strip()) == label.expect_current_task:
            task_hits += 1
        if bool(extraction.user_decisions) == label.expect_decision:
            decision_hits += 1
        if extraction.session_complete == label.expect_completion:
            completion_hits += 1

        for decision in extraction.user_decisions:
            decisions_extracted += 1
            grounded = is_decision_grounded(transcript, decision.quote)
            forbidden = any(
                forbidden_text in decision.quote
                for forbidden_text in label.forbidden_quote_substrings
            )
            if not grounded or forbidden:
                false_decisions += 1

    false_decision_rate = (false_decisions / decisions_extracted) if decisions_extracted else 0.0

    return LegMetrics(
        question_detection_accuracy=question_hits / n,
        blocker_detection_accuracy=blocker_hits / n,
        current_task_accuracy=task_hits / n,
        decision_retention_accuracy=decision_hits / n,
        false_decision_rate=false_decision_rate,
        completion_detection_accuracy=completion_hits / n,
        decisions_extracted=decisions_extracted,
        false_decisions=false_decisions,
    )


def is_degenerate_extraction(
    labels: Iterable[FixtureLabel], extractions: dict[str, Extraction]
) -> list[str]:
    """Return fixture ids where a demonstrably-present field extracted empty.

    Backs `test_extraction_is_not_degenerate`: for every fixture whose label
    says a task, blocker, or question demonstrably exists, the corresponding
    extracted field must be non-empty. Returns the offending fixture ids
    (empty list means none are degenerate).
    """
    offenders = []
    for label in labels:
        extraction = extractions[label.id]
        if label.expect_current_task and not extraction.current_task.strip():
            offenders.append(label.id)
        elif label.expect_blocker and not extraction.blockers_now:
            offenders.append(label.id)
        elif label.expect_question and not extraction.questions_for_user:
            offenders.append(label.id)
    return offenders


# =============================================================================
# Orchestration
# =============================================================================


def run_eval(
    labels: Sequence[FixtureLabel],
    fixtures_dir: Path,
    *,
    e4b_client: ModelClient,
    e2b_client: ModelClient,
    e4b_model_name: str = E4B_MODEL_NAME,
    e2b_model_name: str = E2B_MODEL_NAME,
    on_status: Callable[[str], None] | None = None,
) -> EvalReport:
    """Run both legs over the same labelled fixtures and score each.

    Deliberately takes already-constructed `ModelClient`s rather than
    `LegConfig`s: process lifecycle (starting/stopping the E2B server) is
    entirely `managed_e2b_server`'s concern, kept separate so this function
    can be exercised hermetically against stub HTTP servers in tests without
    any subprocess involved.

    Args:
        labels: Ground truth for every fixture to run.
        fixtures_dir: Directory `label.path` is resolved against.
        e4b_client: Client already pointed at the E4B leg.
        e2b_client: Client already pointed at the E2B leg.
        e4b_model_name: Recorded in `model_runs.model` for the E4B leg.
        e2b_model_name: Recorded in `model_runs.model` for the E2B leg.
        on_status: Progress channel (INV-1), forwarded into every
            `ModelClient.complete` call. Never writes to stdout.
    """
    status = on_status or (lambda _message: None)

    transcripts: dict[str, str] = {}
    for label in labels:
        transcripts[label.id] = normalize_path(fixtures_dir / label.path)

    per_leg: dict[str, LegMetrics] = {}
    for leg_name, client, model_name in (
        ("E4B", e4b_client, e4b_model_name),
        ("E2B", e2b_client, e2b_model_name),
    ):
        extractions: dict[str, Extraction] = {}
        for label in labels:
            status(f"{leg_name}: extracting {label.id}")
            prompt = build_prompt(transcripts[label.id])
            raw = client.complete(
                model=model_name,
                purpose="eval-extraction",
                prompt=prompt,
                schema=EXTRACTION_SCHEMA,
                on_status=status,
            )
            extractions[label.id] = extraction_from_json(raw)
        per_leg[leg_name] = score_leg(labels, extractions, transcripts)

    return EvalReport(fixture_ids=tuple(label.id for label in labels), per_leg=per_leg)
