"""Tests for the Codex rollout adapter and its tier-4 cap (task 7.1).

Every rollout this module tails is a JSONL file it writes itself under
pytest's `tmp_path`, with prose invented for the test — no real
`~/.codex/sessions/` store is ever opened, globbed, or read (INV-2, INV-9).
`CodexAdapter` is always constructed with an explicit `root` pointing into
`tmp_path`.

The two committed artifacts (`tests/fixtures/labels/codex-role-labels.jsonl` and
`tests/fixtures/labels/codex-role-class-measurement.json`) are read where the
measurement itself is under test, because their *content* is the claim being
checked.

**Why the positive controls matter here more than usual.** Most of this file
asserts that something is refused: the cap holds, the channel is harness, the
tier is 4. A suite of refusals passes trivially against an implementation
that refuses everything — `cap_codex_tier` could be `return 4` and every
negative test would stay green. So each refusal has a sibling proving the
code can do the other thing under the right conditions: the cap lifts against
a synthetic passing measurement, the classifier returns the human channel for
a bare-prose record, and a non-boundary event does not produce a
turn-boundary kind.
"""

import json
import subprocess
from pathlib import Path

import pytest

from palaver.ingest.adapters.base import Event
from palaver.ingest.adapters.codex import (
    CHANNEL_HUMAN,
    CHANNEL_INJECTED,
    INJECTED_TEXT_PREFIXES,
    KIND_COMPACTION,
    KIND_ERROR,
    KIND_MESSAGE,
    KIND_SESSION_META,
    KIND_TURN_BOUNDARY,
    LABELS_PATH,
    MEASUREMENT_PATH,
    REQUIRED_LABELLED_RECORDS,
    CodexAdapter,
    CodexTierCapError,
    RoleClassMeasurement,
    cap_codex_tier,
    codex_role_class,
    codex_tier_cap_lifted,
    load_measurement,
    measure_role_class,
    message_role,
    message_text,
    order_records,
    record_ordinal,
    require_codex_tier,
)
from palaver.ingest.cursors import Cursor
from palaver.memory.tiers import (
    TIER_AGENT_CONCLUSION,
    TIER_OBSERVED_RESULT,
    TIER_OBSERVER_INFERENCE,
    TIER_OBSERVER_SPECULATION,
    TIER_USER_INSTRUCTION,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- helpers ----------------------------------------------------------------


def _write_rollout(root: Path, name: str, records: list[dict]) -> Path:
    """Write a rollout file into a date-partitioned tree under `root`."""
    day = root / "2026" / "08" / "14"
    day.mkdir(parents=True, exist_ok=True)
    path = day / name
    path.write_bytes(b"".join((json.dumps(r) + "\n").encode("utf-8") for r in records))
    return path


def _session_meta(**overrides) -> dict:
    payload = {
        "id": "fixture-thread-1",
        "session_id": "fixture-thread-1",
        "cwd": "/tmp/fixture-codex-project",
    }
    payload.update(overrides)
    return {"type": "session_meta", "payload": payload}


def _message(role: str, text: str, block_type: str = "input_text") -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": block_type, "text": text}],
        },
    }


def _event(event_type: str, **payload) -> dict:
    return {"type": "event_msg", "payload": {"type": event_type, **payload}}


def _function_call(call_id: str | None = "call-1") -> dict:
    payload = {"type": "function_call", "name": "shell"}
    if call_id is not None:
        payload["call_id"] = call_id
    return {"type": "response_item", "payload": payload}


def _function_call_output(call_id: str | None = "call-1") -> dict:
    payload = {"type": "function_call_output"}
    if call_id is not None:
        payload["call_id"] = call_id
    return {"type": "response_item", "payload": payload}


def _write_measurement(path: Path, **fields) -> Path:
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def _passing_measurement(tmp_path: Path) -> Path:
    """A synthetic measurement that legitimately lifts the cap.

    This is the positive control the whole cap suite rests on: without it,
    every "the cap holds" assertion below would also pass against a
    `cap_codex_tier` that ignored its measurement entirely.
    """
    return _write_measurement(
        tmp_path / "passing.json",
        n_records=REQUIRED_LABELLED_RECORDS,
        n_errors=0,
        threshold_met=True,
    )


# --- codex_role_class: channel classification (INV-8) -----------------------


@pytest.mark.inv8
def test_developer_role_record_is_the_harness_channel():
    """Done-when: a `role == "developer"` record returns the harness channel.

    Codex has no `isMeta`, and `developer` is the one role research §2 found
    reliably harness across 817 sampled records. It is the strongest
    structural signal the source offers.
    """
    record = _message("developer", "you are operating inside a sandboxed fixture container")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_bare_user_prose_is_the_human_channel():
    """Positive control for every harness assertion in this file.

    Without this, `codex_role_class` could be `return CHANNEL_INJECTED` and
    the developer test, the prefix tests, and the role-less tests would all
    still pass.
    """
    record = _message("user", "check the staging deploy status")
    assert codex_role_class(record) == CHANNEL_HUMAN


@pytest.mark.inv8
@pytest.mark.parametrize("prefix", INJECTED_TEXT_PREFIXES)
def test_every_injected_prefix_is_classified_harness(prefix):
    """Each entry in the prefix table actually fires.

    Parametrized over the table itself rather than a hand-copied list, so a
    prefix added to the table without being reachable — a typo, a stray
    leading space — fails here instead of silently classifying real harness
    content as human.
    """
    record = _message("user", f"{prefix} trailing fixture text")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_leading_whitespace_cannot_launder_injected_content():
    """A harness block behind a newline is still harness.

    `str.startswith` on the raw text would classify this as human, which is
    the exact INV-8 failure: injected content wearing the human channel, and
    therefore quotable as a tier-1 instruction.
    """
    record = _message("user", "\n\n  <environment_context>fixture sandbox</environment_context>")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
@pytest.mark.parametrize(
    "record",
    [
        _session_meta(),
        _event("task_complete", last_agent_message=None),
        _event("context_compacted"),
        {"type": "compacted", "payload": {"replacement_history": []}},
        _function_call(),
        {"type": "some_future_record_type", "payload": {}},
        {},
    ],
    ids=[
        "session_meta",
        "task_complete",
        "context_compacted",
        "compacted",
        "function_call",
        "unknown_type",
        "empty",
    ],
)
def test_records_without_a_message_role_are_never_human(record):
    """Rule 1 is structural, and the narrowed measurement denominator rests on it.

    `measure_role_class` counts only role-bearing records toward the
    200-record threshold, on the grounds that a role-less record cannot be
    misclassified. That is only sound if it is actually impossible, so it is
    asserted rather than assumed.
    """
    assert message_role(record) is None
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_assistant_output_is_not_the_human_channel():
    record = _message("assistant", "the staging deploy is healthy", block_type="output_text")
    assert message_role(record) == "assistant"
    assert codex_role_class(record) == CHANNEL_INJECTED


def test_message_text_concatenates_only_recognized_text_blocks():
    record = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_image", "image_url": "ignored"},
                {"type": "input_text", "text": " second"},
            ],
        },
    }
    assert message_text(record) == "first second"


def test_message_text_of_a_non_message_record_is_empty():
    assert message_text(_event("task_complete", last_agent_message=None)) == ""
    assert message_text(_function_call()) == ""


# --- the tier cap -----------------------------------------------------------


def test_decision_is_tier_four_when_no_measurement_record_exists(tmp_path):
    """Done-when: a Codex-sourced decision returns tier-4 with no measurement."""
    absent = tmp_path / "does-not-exist.json"
    assert load_measurement(absent) is None
    assert codex_tier_cap_lifted(measurement_path=absent) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=absent) == (
        TIER_OBSERVER_INFERENCE
    )


def test_cap_remains_against_a_passing_measurement(tmp_path):
    """A diagnostic measurement cannot weaken Codex's permanent safety cap."""
    passing = _passing_measurement(tmp_path)
    assert codex_tier_cap_lifted(measurement_path=passing) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=passing) == (
        TIER_OBSERVER_INFERENCE
    )
    with pytest.raises(CodexTierCapError):
        require_codex_tier(TIER_USER_INSTRUCTION, measurement_path=passing)


@pytest.mark.parametrize(
    "fields,why",
    [
        (
            {"n_records": REQUIRED_LABELLED_RECORDS - 1, "n_errors": 0, "threshold_met": True},
            "sample one record short of the threshold",
        ),
        (
            {"n_records": REQUIRED_LABELLED_RECORDS, "n_errors": 1, "threshold_met": True},
            "one harness record classified as user",
        ),
        (
            {"n_records": REQUIRED_LABELLED_RECORDS, "n_errors": 0, "threshold_met": False},
            "threshold_met false despite passing counts",
        ),
    ],
    ids=["n_records_below_200", "n_errors_above_zero", "threshold_met_false"],
)
def test_each_failing_condition_independently_holds_the_cap(tmp_path, fields, why):
    """Done-when: tier-4 when `n_records` < 200, `n_errors` > 0, or `threshold_met` false.

    Each variant differs from the passing control in exactly one field, so a
    conjunction that dropped a term (`and self.threshold_met` deleted, say)
    fails on precisely the variant that term guards, and the failure names
    which condition stopped being enforced.
    """
    path = _write_measurement(tmp_path / "failing.json", **fields)
    assert codex_tier_cap_lifted(measurement_path=path) is False, why
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=path) == (TIER_OBSERVER_INFERENCE)


def test_requesting_tier_one_for_a_codex_source_raises(tmp_path):
    """Done-when: the tier cap raises when a caller requests tier-1.

    `cap_codex_tier` demotes silently, which is right for a tier the
    pipeline derived. An explicit tier-1 request is a caller asserting "the
    user said this" on the strength of a prefix heuristic, and silently
    handing it a 4 would hide that.
    """
    absent = tmp_path / "does-not-exist.json"
    with pytest.raises(CodexTierCapError) as excinfo:
        require_codex_tier(TIER_USER_INSTRUCTION, measurement_path=absent)
    message = str(excinfo.value)
    assert "tier 1" in message
    assert "user_instruction" in message
    assert "permanent" in message


@pytest.mark.parametrize(
    "tier", [TIER_AGENT_CONCLUSION, TIER_OBSERVED_RESULT], ids=["tier2", "tier3"]
)
def test_every_tier_above_the_cap_is_refused(tmp_path, tier):
    """The cap is not a tier-1 special case.

    Tier-2 ("the main agent concluded this") and tier-3 ("this tool result
    was observed") both rest on reading the transcript correctly, which for
    Codex means the same heuristic channel split.
    """
    absent = tmp_path / "does-not-exist.json"
    assert cap_codex_tier(tier, measurement_path=absent) == TIER_OBSERVER_INFERENCE
    with pytest.raises(CodexTierCapError):
        require_codex_tier(tier, measurement_path=absent)


def test_cap_never_promotes_a_weaker_tier(tmp_path):
    """Tier-5 speculation stays tier-5; the cap is a ceiling, not a floor."""
    absent = tmp_path / "does-not-exist.json"
    assert cap_codex_tier(TIER_OBSERVER_SPECULATION, measurement_path=absent) == (
        TIER_OBSERVER_SPECULATION
    )
    assert require_codex_tier(TIER_OBSERVER_SPECULATION, measurement_path=absent) == (
        TIER_OBSERVER_SPECULATION
    )
    assert require_codex_tier(TIER_OBSERVER_INFERENCE, measurement_path=absent) == (
        TIER_OBSERVER_INFERENCE
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        "[1, 2, 3]",
        '"a string"',
        "{}",
        '{"n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "threshold_met": true}',
        '{"n_records": 200, "n_errors": 0}',
        '{"n_records": "200", "n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "n_errors": 0, "threshold_met": 1}',
        '{"n_records": true, "n_errors": 0, "threshold_met": true}',
        '{"n_records": 200, "n_errors": false, "threshold_met": true}',
        '{"n_records": null, "n_errors": null, "threshold_met": null}',
    ],
    ids=[
        "empty",
        "not_json",
        "json_array",
        "json_string",
        "empty_object",
        "missing_n_records",
        "missing_n_errors",
        "missing_threshold_met",
        "n_records_as_string",
        "threshold_met_as_int",
        "n_records_as_bool",
        "n_errors_as_bool",
        "all_null",
    ],
)
def test_a_malformed_measurement_holds_the_cap(tmp_path, content):
    """Every unreadable measurement fails closed, and none of them lift the cap.

    `threshold_met: 1` and `n_records: true` are in here specifically because
    `isinstance(True, int)` is true in Python: a naive type check would admit
    both and lift a cap on a file that never passed a measurement.
    """
    path = tmp_path / "malformed.json"
    path.write_text(content, encoding="utf-8")
    assert load_measurement(path) is None
    assert codex_tier_cap_lifted(measurement_path=path) is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION, measurement_path=path) == (TIER_OBSERVER_INFERENCE)


def test_lifts_tier_cap_requires_all_three_conditions():
    """The conjunction, asserted directly on the dataclass.

    A truth table rather than a single happy path, so a term dropped from
    `lifts_tier_cap` cannot pass by coincidence.
    """
    n = REQUIRED_LABELLED_RECORDS
    assert RoleClassMeasurement(n, 0, True).lifts_tier_cap is True
    assert RoleClassMeasurement(n - 1, 0, True).lifts_tier_cap is False
    assert RoleClassMeasurement(n, 1, True).lifts_tier_cap is False
    assert RoleClassMeasurement(n, 0, False).lifts_tier_cap is False
    assert RoleClassMeasurement(0, 0, True).lifts_tier_cap is False


def test_the_committed_measurement_holds_the_cap_today():
    """The shipped state: 19 labelled records is not 200, so Codex stays tier-4.

    This is the honest negative result task 7.1 exists to record. It is not a
    failure of the mechanism — the mechanism is working, and this assertion
    is what proves it is wired to the real file rather than only to
    `tmp_path` fixtures.
    """
    measurement = load_measurement()
    assert measurement is not None, f"{MEASUREMENT_PATH} is missing or malformed"
    assert measurement.lifts_tier_cap is False
    assert codex_tier_cap_lifted() is False
    assert cap_codex_tier(TIER_USER_INSTRUCTION) == TIER_OBSERVER_INFERENCE
    with pytest.raises(CodexTierCapError):
        require_codex_tier(TIER_USER_INSTRUCTION)


# --- the blind measurement --------------------------------------------------


def test_committed_measurement_matches_a_live_recomputation():
    """The measurement file cannot be forged by editing it.

    Every count in the committed JSON is recomputed here from the committed
    labels and the committed corpus. Hand-editing `n_records` to 200 or
    `threshold_met` to true fails this test, which is what stops the file
    from being a claim rather than a measurement.
    """
    assert LABELS_PATH.is_file(), f"{LABELS_PATH} is missing"
    recomputed = measure_role_class()
    committed = json.loads(MEASUREMENT_PATH.read_text(encoding="utf-8"))
    assert committed == recomputed

    # The four counts below are properties of the immutable labels file, not
    # of `codex_role_class`. Pinning them anchors the recomputation to the
    # corpus that was labelled blind: if a later change edits, reorders or
    # re-labels `codex-role-labels.jsonl`, this fails even though the
    # classifier and the committed JSON would still agree with each other.
    # They are sized to today's 19-row corpus: when the corpus legitimately
    # grows toward the 200-record threshold, update these alongside the
    # regenerated measurement file rather than reading the failure as tampering.
    assert recomputed["n_labelled_rows"] == 19
    assert recomputed["n_records"] == 8
    assert recomputed["n_human_labels"] == 4
    assert recomputed["n_injected_labels"] == 15
    assert recomputed["n_discriminating"] == 1


def test_measurement_reports_the_honest_negative_result():
    """19 labelled records against a 200-record threshold: the cap stays on."""
    result = measure_role_class()
    assert result["n_labelled_rows"] == 19
    assert result["n_records"] < REQUIRED_LABELLED_RECORDS
    assert result["threshold_met"] is False


def test_measurement_records_zero_harness_classified_as_user_errors():
    """The measured disagreement, recorded rather than asserted as validation.

    Zero errors over this sample is weak evidence, and `n_discriminating`
    quantifies how weak. Against naive role-mapping — `user` is human,
    everything else is harness — exactly **one** of the nineteen rows
    disagrees: the `<environment_context>` record that wears `role: "user"`.
    Every other row, the `developer` record included, is one the trivial
    classifier also gets right, so it exercises nothing.

    A perfect score over a single discriminating case is not validation of
    the prefix table. The 200-record threshold is what would make a zero
    meaningful, and it is not met.
    """
    result = measure_role_class()
    assert result["n_errors"] == 0
    assert result["n_disagreements"] == 0
    assert result["n_discriminating"] == 1


def test_the_label_sample_contains_a_user_role_record_labelled_harness():
    """The sample is not trivially satisfiable.

    Five records carry `role: "user"` and only four are labelled human, so
    the corpus contains at least one record where the role and the true
    channel disagree — the case a role-only classifier would get wrong.
    A sample without it could be swept by `role == "user" -> human`.
    """
    rows = [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()]
    user_rows = [row for row in rows if row.get("role") == "user"]
    harness_user_rows = [row for row in user_rows if row["channel"] == CHANNEL_INJECTED]
    assert len(user_rows) == 5
    assert len(harness_user_rows) == 1

    # And the naive role-only classifier really would fail on it, which is
    # what makes the previous two assertions meaningful rather than trivia.
    assert harness_user_rows[0]["channel"] != CHANNEL_HUMAN


def test_every_label_row_resolves_to_a_real_corpus_record():
    """No label points at a record that does not exist.

    A stale `file`/`index` pair would silently shrink the measured sample or
    crash the measurement; either way the count that gates the cap would stop
    meaning what it says.
    """
    rows = [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()]
    assert rows, "the labels file is empty"
    for row in rows:
        source = REPO_ROOT / row["file"]
        assert source.is_file(), f"{row['file']} does not exist"
        lines = source.read_text(encoding="utf-8").splitlines()
        assert 0 <= row["index"] < len(lines), f"{row['file']}:{row['index']} is out of range"
        assert row["channel"] in {CHANNEL_HUMAN, CHANNEL_INJECTED}


def test_measurement_denominator_counts_only_role_bearing_records():
    """`n_records` is narrower than the file, and visibly so.

    Counting role-less envelopes toward the 200-record threshold would let a
    future corpus reach it on records `codex_role_class` cannot get wrong.
    `n_labelled_rows` is reported alongside so the narrowing is auditable
    rather than hidden in the denominator.
    """
    result = measure_role_class()
    rows = [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()]
    role_bearing = [row for row in rows if row.get("role") is not None]
    assert result["n_records"] == len(role_bearing)
    assert result["n_records"] < result["n_labelled_rows"]


def test_measurement_counts_an_injected_record_misread_as_human(tmp_path):
    """Positive control for `n_errors`: the counter can be non-zero.

    Every measurement assertion above is "the count is zero", which a broken
    counter satisfies for free. This points the same code at a deliberately
    wrong label file and requires the error to be counted, and counted in the
    harness-classified-as-user direction specifically.
    """
    corpus = tmp_path / "tests" / "fixtures" / "codex"
    corpus.mkdir(parents=True)
    (corpus / "mislabelled.jsonl").write_bytes(
        (json.dumps(_message("user", "check the staging deploy status")) + "\n").encode("utf-8")
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "file": "tests/fixtures/codex/mislabelled.jsonl",
                "index": 0,
                "role": "user",
                "channel": CHANNEL_INJECTED,
                "basis": "deliberately wrong, to prove the error counter fires",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = measure_role_class(labels_path=labels, corpus_root=tmp_path)
    assert result["n_errors"] == 1
    assert result["n_disagreements"] == 1
    assert result["threshold_met"] is False


def test_a_human_record_misread_as_harness_is_a_disagreement_not_an_error(tmp_path):
    """The two counters mean different things and are not aliases.

    Classifying a human record as harness loses a memory. Classifying a
    harness record as human mints a false tier-1, which under INV-4 cannot be
    retracted. Only the second blocks the cap, so only the second increments
    `n_errors`.
    """
    corpus = tmp_path / "tests" / "fixtures" / "codex"
    corpus.mkdir(parents=True)
    (corpus / "mislabelled.jsonl").write_bytes(
        (json.dumps(_message("developer", "a harness preamble")) + "\n").encode("utf-8")
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "file": "tests/fixtures/codex/mislabelled.jsonl",
                "index": 0,
                "role": "developer",
                "channel": CHANNEL_HUMAN,
                "basis": "deliberately wrong in the harmless direction",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = measure_role_class(labels_path=labels, corpus_root=tmp_path)
    assert result["n_disagreements"] == 1
    assert result["n_errors"] == 0


# --- commit ordering: the labels were authored blind ------------------------


#: Subcommands that write. They are legitimate against the throwaway
#: repositories the ordering positive-control builds under `tmp_path`, and
#: never against the repository under test.
_MUTATING_GIT_SUBCOMMANDS = frozenset({"init", "config", "add", "commit", "checkout", "reset"})


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Run a git command and return its stripped stdout.

    Guards the repository under test: a mutating subcommand is refused
    outright when `cwd` is `REPO_ROOT`. The ordering positive-control needs
    real commits to have a history to check, so it builds them in a
    `tmp_path` repository — and a missing `cwd=` argument there would
    otherwise silently commit into the developer's own working tree.
    """
    if args and args[0] in _MUTATING_GIT_SUBCOMMANDS and Path(cwd) == REPO_ROOT:
        raise AssertionError(
            f"refusing to run mutating `git {args[0]}` against the repository "
            f"under test; pass cwd=<tmp_path repo>"
        )
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _adding_commit(path: Path, cwd: Path = REPO_ROOT) -> str | None:
    """Return the single commit that added `path`, or `None` if untracked.

    `tests/fixtures/labels/` is a fresh path with no prior history — the
    earlier flat-root labels commit was reverted and never used this
    directory — so `--diff-filter=A` matches exactly one commit and there is
    no "which add did they mean" ambiguity to resolve. That is asserted
    rather than assumed: a path with two adds would make the ordering claim
    below depend on which one the helper happened to pick.
    """
    relative = path.relative_to(cwd).as_posix()
    out = _git("log", "--diff-filter=A", "--format=%H", "--", relative, cwd=cwd)
    commits = out.split()
    assert len(commits) <= 1, (
        f"{relative} was added in {len(commits)} commits ({commits}); the "
        f"ordering proof needs an unambiguous add"
    )
    return commits[0] if commits else None


def _path_exists_at(commit: str, relative: str, cwd: Path = REPO_ROOT) -> bool:
    """Whether `relative` exists in the tree of `commit`."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _is_ancestor(earlier: str, later: str, cwd: Path = REPO_ROOT) -> bool:
    """Whether `earlier` is a strict ancestor of `later`."""
    if earlier == later:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", earlier, later],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_labels_were_committed_before_the_measurement():
    """Done-when: the labels file must be the earlier commit.

    Ancestry, not timestamp comparison. Git commit timestamps have
    second resolution, so two commits made moments apart can tie and a
    strict `labels_time < measurement_time` assertion would flake; ancestry
    is exact and is what "committed first" actually means on a linear
    history.

    This is the ordering that makes the measurement meaningful. Labels
    written after seeing the classifier's output would validate the heuristic
    against labels the heuristic produced.
    """
    labels_commit = _adding_commit(LABELS_PATH)
    measurement_commit = _adding_commit(MEASUREMENT_PATH)

    if measurement_commit is None and labels_commit is None:
        pytest.skip(
            "neither artifact is committed yet; the orchestrator commits the "
            "labels alone first, then the measurement"
        )
    assert labels_commit is not None, (
        "the measurement is committed but the labels are not: the ordering "
        "proof has nothing to stand on"
    )
    if measurement_commit is None:
        pytest.skip("the measurement file is not committed yet")

    assert labels_commit != measurement_commit, (
        "labels and measurement landed in one commit, so nothing evidences "
        "that the labels predate the classifier"
    )
    assert _is_ancestor(labels_commit, measurement_commit), (
        f"labels commit {labels_commit} is not an ancestor of measurement "
        f"commit {measurement_commit}"
    )
    # Ordering shows order; absence is what carries the blindness claim. If
    # the classifier did not exist in the tree when the labels landed, it
    # cannot have informed them — no argument about intent required.
    assert not _path_exists_at(labels_commit, "palaver/ingest/adapters/codex.py"), (
        f"codex.py exists at the labels commit {labels_commit}, so the labels "
        f"could have been written against the classifier's output"
    )


def test_the_ordering_check_fails_on_a_wrong_order_history(tmp_path):
    """Positive control: the ordering check is live, and it is not vacuous.

    The real ordering test above skips while the artifacts are uncommitted,
    which is exactly when a broken check would go unnoticed. This builds a
    throwaway repository whose history has the two files in the *wrong*
    order and requires the same helpers to detect it — then rebuilds it in
    the right order and requires them to accept it.
    """
    repo = tmp_path / "repo"
    (repo / "tests" / "fixtures" / "labels").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "fixture@example.invalid", cwd=repo)
    _git("config", "user.name", "Fixture", cwd=repo)

    measurement = repo / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    labels = repo / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"

    # Wrong order: the measurement lands first.
    measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "measurement first", cwd=repo)
    labels.write_text("{}\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "labels second", cwd=repo)

    labels_commit = _adding_commit(labels, cwd=repo)
    measurement_commit = _adding_commit(measurement, cwd=repo)
    assert labels_commit and measurement_commit
    assert not _is_ancestor(labels_commit, measurement_commit, cwd=repo), (
        "the ordering check accepted a history where the labels were "
        "committed after the measurement"
    )
    # ...and the violation is detectable in the direction that matters.
    assert _is_ancestor(measurement_commit, labels_commit, cwd=repo)

    # Right order, in a second repository: the helpers accept it.
    good = tmp_path / "good"
    (good / "tests" / "fixtures" / "labels").mkdir(parents=True)
    _git("init", "-q", cwd=good)
    _git("config", "user.email", "fixture@example.invalid", cwd=good)
    _git("config", "user.name", "Fixture", cwd=good)
    good_labels = good / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"
    good_measurement = good / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    good_labels.write_text("{}\n", encoding="utf-8")
    _git("add", "-A", cwd=good)
    _git("commit", "-qm", "labels first", cwd=good)
    good_measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=good)
    _git("commit", "-qm", "measurement second", cwd=good)

    assert _is_ancestor(
        _adding_commit(good_labels, cwd=good),
        _adding_commit(good_measurement, cwd=good),
        cwd=good,
    )
    assert not _path_exists_at(
        _adding_commit(good_labels, cwd=good),
        "palaver/ingest/adapters/codex.py",
        cwd=good,
    )


def test_the_absence_check_catches_a_classifier_present_at_the_labels_commit(tmp_path):
    """Positive control for the absence assertion specifically.

    The wrong-order control above cannot exercise this: a history that fails
    on ordering never reaches the absence check, and a correctly-ordered
    history passes both. So this builds the one history where ordering is
    satisfied but the property is still violated — labels and classifier
    landing in the *same* commit, measurement after. Ordering says that is
    fine. Only absence catches it, which is why absence is the assertion the
    blindness claim actually rests on.
    """
    repo = tmp_path / "same-commit"
    (repo / "tests" / "fixtures" / "labels").mkdir(parents=True)
    (repo / "palaver" / "ingest" / "adapters").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "fixture@example.invalid", cwd=repo)
    _git("config", "user.name", "Fixture", cwd=repo)

    labels = repo / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"
    measurement = repo / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    classifier = repo / "palaver" / "ingest" / "adapters" / "codex.py"

    labels.write_text("{}\n", encoding="utf-8")
    classifier.write_text("# a classifier that already existed\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "labels and classifier together", cwd=repo)
    measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "measurement second", cwd=repo)

    labels_commit = _adding_commit(labels, cwd=repo)
    measurement_commit = _adding_commit(measurement, cwd=repo)

    # Ordering is satisfied, so it cannot be what rejects this history.
    assert labels_commit != measurement_commit
    assert _is_ancestor(labels_commit, measurement_commit, cwd=repo)

    # The absence check is the one that fires.
    assert _path_exists_at(labels_commit, "palaver/ingest/adapters/codex.py", cwd=repo), (
        "the absence check failed to see a classifier that was committed alongside the labels"
    )


def test_adding_commit_reports_none_for_an_untracked_path(tmp_path):
    """The untracked branch of `_adding_commit` is exercised, not assumed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    untracked = repo / "untracked.json"
    untracked.write_text("{}", encoding="utf-8")
    assert _adding_commit(untracked, cwd=repo) is None


# --- turn boundary ----------------------------------------------------------


def test_task_complete_emits_a_turn_boundary_event(tmp_path):
    """Done-when: `task_complete` emits a turn-boundary event."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-00-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _message("assistant", "the staging deploy is healthy", block_type="output_text"),
            _event("task_complete", last_agent_message="the fixture worker finished"),
        ],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    kinds = [event.kind for event in events]
    assert kinds == [KIND_SESSION_META, KIND_MESSAGE, KIND_MESSAGE, KIND_TURN_BOUNDARY]
    assert events[-1].payload["payload"]["type"] == "task_complete"


def test_turn_aborted_also_emits_a_turn_boundary_event(tmp_path):
    """Codex closes a turn two ways, and neither is the sole signal."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-01-00-fixture.jsonl",
        [_session_meta(), _event("turn_aborted", reason="interrupted")],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert events[-1].kind == KIND_TURN_BOUNDARY


def test_a_non_boundary_event_is_not_a_turn_boundary(tmp_path):
    """Positive control for the boundary kind.

    Without this, `_event_msg_kind` could return `KIND_TURN_BOUNDARY` for
    every `event_msg` and both boundary tests above would still pass.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-02-00-fixture.jsonl",
        [
            _session_meta(),
            _event("agent_reasoning_delta"),
            _event("context_compacted"),
            _event("error", message="a transient fixture timeout", codex_error_info="x"),
        ],
    )
    kinds = [event.kind for event in CodexAdapter(root=root).tail(path, Cursor()).events]
    assert KIND_TURN_BOUNDARY not in kinds
    assert kinds == [KIND_SESSION_META, "agent_reasoning_delta", KIND_COMPACTION, KIND_ERROR]


# --- compaction -------------------------------------------------------------


def test_the_paired_compaction_marker_produces_two_compaction_events(tmp_path):
    """Both halves of Codex's compaction pair are recognized independently.

    Keying only on the `compacted` envelope, or only on `context_compacted`,
    would leave a release that emits one of them invisible.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-03-00-fixture.jsonl",
        [
            _session_meta(),
            {
                "type": "compacted",
                "payload": {
                    "replacement_history": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "earlier turn"}],
                        }
                    ]
                },
            },
            _event("context_compacted"),
        ],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    compactions = [event for event in events if event.kind == KIND_COMPACTION]
    assert len(compactions) == 2
    assert compactions[0].payload["type"] == "compacted"
    assert compactions[1].payload["payload"]["type"] == "context_compacted"
    # The replacement history survives byte-for-byte, so INV-6 evidence
    # anchored into it still indexes the record as written.
    assert compactions[0].payload["payload"]["replacement_history"][0]["role"] == "user"


# --- errors, at all three layers -------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        _event("exec_command_end", exit_code=1, status="completed"),
        _event("exec_command_end", exit_code=0, status="failed"),
        _event("exec_command_end", exit_code=127, status="failed"),
        _event("patch_apply_end", success=False),
        _event("error", message="a transient fixture timeout", codex_error_info="x"),
    ],
    ids=["nonzero_exit", "failed_status", "both", "patch_failed", "event_error"],
)
def test_every_error_layer_maps_to_the_error_kind(tmp_path, record):
    """Research §2 names three error layers; all three are read, independently.

    `exit_code` and `status` are checked separately rather than conjunctively
    — requiring both would go blind if a release stopped emitting one.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-04-00-fixture.jsonl", [record])
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert [event.kind for event in events] == [KIND_ERROR]


@pytest.mark.parametrize(
    "record",
    [
        _event("exec_command_end", exit_code=0, status="completed"),
        _event("patch_apply_end", success=True),
    ],
    ids=["clean_exec", "clean_patch"],
)
def test_a_successful_command_is_not_an_error(tmp_path, record):
    """Positive control: the error mapping discriminates.

    A success still produces an event — nothing is dropped — but under its
    own kind. Without this, `_event_msg_kind` could return `KIND_ERROR` for
    every `exec_command_end` and the five tests above would pass.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-05-00-fixture.jsonl", [record])
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert len(events) == 1
    assert events[0].kind != KIND_ERROR
    assert events[0].kind == record["payload"]["type"]


def test_a_non_integer_exit_code_is_not_read_as_a_failure(tmp_path):
    """`exit_code: true` is not exit code 1.

    `isinstance(True, int)` is true in Python, so a bare integer check would
    read a boolean as a non-zero exit status.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-06-00-fixture.jsonl",
        [_event("exec_command_end", exit_code=True, status="completed")],
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert events[0].kind != KIND_ERROR


# --- identity ---------------------------------------------------------------


def test_identity_is_read_from_session_meta(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-07-00-fixture.jsonl",
        [
            _session_meta(
                id="fixture-thread-child",
                session_id="fixture-thread-root",
                parent_thread_id="fixture-thread-root",
            ),
            _message("user", "check the staging deploy status"),
        ],
    )
    adapter = CodexAdapter(root=root)
    identity = adapter.read_identity(path)
    assert identity is not None
    assert identity.cwd == "/tmp/fixture-codex-project"
    assert identity.id == "fixture-thread-child"
    assert identity.session_id == "fixture-thread-root"
    assert identity.parent_thread_id == "fixture-thread-root"
    assert identity.is_subagent is True
    assert adapter.project_key_for(path) == "/tmp/fixture-codex-project"


def test_a_root_session_is_not_a_subagent(tmp_path):
    """Positive control for `is_subagent`: it can be False."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-08-00-fixture.jsonl", [_session_meta()])
    identity = CodexAdapter(root=root).read_identity(path)
    assert identity is not None
    assert identity.is_subagent is False


def test_a_differing_session_id_alone_marks_a_subagent(tmp_path):
    """`parent_thread_id` is not the only linkage; `.id != .session_id` is too."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-09-00-fixture.jsonl",
        [_session_meta(id="fixture-thread-child", session_id="fixture-thread-root")],
    )
    identity = CodexAdapter(root=root).read_identity(path)
    assert identity is not None
    assert identity.parent_thread_id is None
    assert identity.is_subagent is True


def test_identity_is_none_when_the_file_has_no_session_meta(tmp_path):
    """A truncated or not-yet-flushed rollout is a legitimate state, not a crash."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-10-00-fixture.jsonl",
        [_message("user", "check the staging deploy status")],
    )
    adapter = CodexAdapter(root=root)
    assert adapter.read_identity(path) is None
    assert adapter.project_key_for(path) is None


def test_session_key_is_derived_from_the_path_without_opening_the_file(tmp_path):
    """`discover_sessions` calls this for paths it will never open."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-11-00-fixture.jsonl", [_session_meta()])
    adapter = CodexAdapter(root=root)
    assert adapter.session_key_for(path) == "rollout-2026-08-14T10-11-00-fixture"
    assert adapter.session_key_for(Path("/nonexistent/rollout-x.jsonl")) == "rollout-x"


# --- discovery --------------------------------------------------------------


def test_store_discovery_is_recursive_and_prefix_scoped(tmp_path):
    """Codex partitions by `YYYY/MM/DD`, so the glob has to be recursive.

    The `rollout-` prefix keeps any other `.jsonl` in that tree — a cache, a
    sidecar a future release drops in — from being mistaken for a session
    store and handed a bogus session key.
    """
    root = tmp_path / "sessions"
    _write_rollout(root, "rollout-2026-08-14T10-12-00-a.jsonl", [_session_meta()])
    deep = root / "2026" / "08" / "15"
    deep.mkdir(parents=True)
    (deep / "rollout-2026-08-15T09-00-00-b.jsonl").write_text("", encoding="utf-8")
    (deep / "not-a-rollout.jsonl").write_text("", encoding="utf-8")

    paths = list(CodexAdapter(root=root).list_store_paths())
    names = [path.name for path in paths]
    assert names == [
        "rollout-2026-08-14T10-12-00-a.jsonl",
        "rollout-2026-08-15T09-00-00-b.jsonl",
    ]


def test_discovery_of_a_missing_root_is_empty_not_an_error(tmp_path):
    assert list(CodexAdapter(root=tmp_path / "absent").list_store_paths()) == []


# --- unresolved trailing tool use (the Codex inversion) ---------------------


def test_a_turn_boundary_clears_a_dangling_tool_call(tmp_path):
    """The case that proves Claude Code's logic was not copied.

    For Codex the last line usually *is* the turn boundary, so a
    `function_call` with no matching output is resolved by the boundary that
    followed it. Reading it as unresolved would pin a finished session into
    `discover_sessions`'s always-include path for as long as the file exists.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-13-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _function_call("call-1"),
            _event("task_complete", last_agent_message=None),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False


def test_an_unanswered_tool_call_with_no_boundary_is_unresolved(tmp_path):
    """Positive control: the check can return True."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-14-00-fixture.jsonl",
        [
            _session_meta(),
            _message("user", "check the staging deploy status"),
            _function_call("call-1"),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_matching_output_resolves_a_tool_call(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-15-00-fixture.jsonl",
        [_session_meta(), _function_call("call-1"), _function_call_output("call-1")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False


def test_a_mismatched_output_does_not_resolve_a_tool_call(tmp_path):
    """Correlation is by `call_id`, not by arrival order."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-16-00-fixture.jsonl",
        [_session_meta(), _function_call("call-1"), _function_call_output("call-2")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_call_reopened_after_a_boundary_is_unresolved_again(tmp_path):
    """A boundary clears the calls before it, not the ones after it."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-17-00-fixture.jsonl",
        [
            _session_meta(),
            _function_call("call-1"),
            _event("task_complete", last_agent_message=None),
            _function_call("call-2"),
        ],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_call_without_a_correlation_id_still_counts_as_pending(tmp_path):
    """An unkeyed call cannot be matched, so ignoring it would under-report."""
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-18-00-fixture.jsonl",
        [_session_meta(), _function_call(None)],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is True


def test_a_session_with_no_tool_calls_is_resolved(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-19-00-fixture.jsonl",
        [_session_meta(), _message("user", "check the staging deploy status")],
    )
    assert CodexAdapter(root=root).has_unresolved_trailing_tool_use(path) is False


# --- cursor and ordinals ----------------------------------------------------


def test_record_ordinal_prefers_the_records_own_number():
    assert record_ordinal({"ordinal": 7}, 0) == 7
    assert record_ordinal({}, 3) == 3
    assert record_ordinal({"ordinal": "7"}, 3) == 3
    assert record_ordinal({"ordinal": True}, 3) == 3, "a boolean is not an ordinal"
    assert record_ordinal({"ordinal": 0}, 5) == 0


def test_records_are_ordered_by_ordinal_when_every_record_has_one():
    records = [{"ordinal": 2, "n": "c"}, {"ordinal": 0, "n": "a"}, {"ordinal": 1, "n": "b"}]
    assert [r["n"] for r in order_records(records)] == ["a", "b", "c"]


def test_records_keep_file_order_when_ordinals_are_absent_or_partial():
    """A batch with no shared coordinate system is left alone.

    Mixing real ordinals with fallback line indices would interleave the two
    arbitrarily, which is worse than the arrival order it replaced.
    """
    no_ordinals = [{"n": "a"}, {"n": "b"}, {"n": "c"}]
    assert [r["n"] for r in order_records(no_ordinals)] == ["a", "b", "c"]

    partial = [{"ordinal": 9, "n": "a"}, {"n": "b"}, {"ordinal": 1, "n": "c"}]
    assert [r["n"] for r in order_records(partial)] == ["a", "b", "c"]


def test_ordering_is_stable_for_equal_ordinals():
    records = [{"ordinal": 1, "n": "a"}, {"ordinal": 1, "n": "b"}]
    assert [r["n"] for r in order_records(records)] == ["a", "b"]


def test_tail_resumes_from_its_cursor_without_re_reading(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-20-00-fixture.jsonl",
        [_session_meta(), _message("user", "check the staging deploy status")],
    )
    adapter = CodexAdapter(root=root)
    first = adapter.tail(path, Cursor())
    assert len(first.events) == 2
    assert first.cursor.offset == path.stat().st_size

    second = adapter.tail(path, first.cursor)
    assert second.events == ()
    assert second.cursor.offset == first.cursor.offset

    with path.open("ab") as handle:
        handle.write(
            (json.dumps(_event("task_complete", last_agent_message=None)) + "\n").encode("utf-8")
        )
    third = adapter.tail(path, second.cursor)
    assert [event.kind for event in third.events] == [KIND_TURN_BOUNDARY]


def test_tail_does_not_advance_past_a_torn_write(tmp_path):
    """A record the agent is mid-way through flushing is not ingested.

    Codex appends to these files while Palaver reads them, so a partial line
    is expected, not exceptional.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-21-00-fixture.jsonl", [_session_meta()])
    complete_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"type": "response_item", "payl')

    result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META]
    assert result.cursor.offset == complete_size
    assert result.malformed_records == 0


def test_tail_counts_complete_malformed_records_without_logging_source_content(tmp_path, caplog):
    """A corrupt line must not crash the tail, and must not vanish silently."""
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-22-00-fixture.jsonl", [_session_meta()])
    with path.open("ab") as handle:
        handle.write(b"{not json secret-source-content}\n")
        handle.write((json.dumps(_event("task_complete", last_agent_message=None)) + "\n").encode())

    with caplog.at_level("WARNING"):
        result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META, KIND_TURN_BOUNDARY]
    assert result.malformed_records == 1
    assert any("Unparseable Codex rollout record" in record.message for record in caplog.records)
    assert "secret-source-content" not in caplog.text


def test_a_non_object_record_is_skipped(tmp_path, caplog):
    root = tmp_path / "sessions"
    path = _write_rollout(root, "rollout-2026-08-14T10-23-00-fixture.jsonl", [_session_meta()])
    with path.open("ab") as handle:
        handle.write(b"[1, 2, 3]\n")

    with caplog.at_level("WARNING"):
        result = CodexAdapter(root=root).tail(path, Cursor())
    assert [event.kind for event in result.events] == [KIND_SESSION_META]
    assert result.malformed_records == 1
    assert any("Non-object Codex rollout record" in record.message for record in caplog.records)


# --- tail payload fidelity --------------------------------------------------


def test_tail_payloads_are_the_source_records_byte_for_byte(tmp_path):
    """INV-6: an evidence anchor must index the record as written.

    A reshaped payload would make the anchor point at Palaver's rendering of
    a record rather than the record, and the quote-grounding gate would then
    be checking a quote against text the source never contained.
    """
    root = tmp_path / "sessions"
    records = [
        _session_meta(),
        _message("user", "check the staging deploy status"),
        _event("error", message="a transient fixture timeout", codex_error_info="x"),
    ]
    path = _write_rollout(root, "rollout-2026-08-14T10-24-00-fixture.jsonl", records)
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert [event.payload for event in events] == records


def test_tail_stamps_every_event_with_the_session_key(tmp_path):
    root = tmp_path / "sessions"
    path = _write_rollout(
        root, "rollout-2026-08-14T10-25-00-fixture.jsonl", [_session_meta(), _function_call()]
    )
    events = CodexAdapter(root=root).tail(path, Cursor()).events
    assert {event.session_key for event in events} == {"rollout-2026-08-14T10-25-00-fixture"}
    assert all(isinstance(event, Event) for event in events)


def test_an_unrecognized_record_type_is_ingested_under_its_own_kind(tmp_path):
    """A future Codex release's records stay visible as evidence.

    Dropping them would make the transcript Palaver stores quietly
    incomplete, and INV-7's `UNKNOWN` status depends on absent signals being
    absent for a reason rather than by omission.
    """
    root = tmp_path / "sessions"
    path = _write_rollout(
        root,
        "rollout-2026-08-14T10-26-00-fixture.jsonl",
        [{"type": "world_state", "payload": {"anything": 1}}, {"payload": {}}],
    )
    kinds = [event.kind for event in CodexAdapter(root=root).tail(path, Cursor()).events]
    assert kinds == ["world_state", "unknown"]
