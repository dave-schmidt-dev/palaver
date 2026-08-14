"""Tests for `palaver fixture-lint` and the ground truth of the fixture corpus.

Two things are defended here, and they fail in opposite directions.

**The linter (INV-9, git clause).** `tests/fixtures/` is committed to a public
remote, so a record that reaches it has left the machine irrecoverably. The
linter is an allowlist and its failure mode is *silent acceptance*, which no
amount of "the corpus passes" can rule out. So the tests that matter here are
the negative ones, and each poisons a record in exactly **one** dimension and
asserts on the *rule name* the linter reported — a poisoned record that fails
because three rules fired at once proves nothing about any of them. Above
them sits `test_poisoned_record_rejection_comes_from_the_classifier`, which
stubs `classify_record` to accept everything and requires the same run to exit
0: without it, a linter that rejected every path it was handed would pass the
whole negative suite.

`test_committed_corpus_passes_the_linter` proves the corpus and the linter
agree with each other. It does not prove the corpus is safe; the negative
tests are what prove that.

**The ground truth (accuracy).** Coverage counts the sessions a signal was
determinable for and a uniformly wrong classifier scores 100% at it. These
tests assert derived status against labels in `tests/fixtures/README.md`, and
assert that those labels state checkable structural facts rather than
authorial intent — "constructed to be WORKING" is circular and is rejected.

Ground truth and the derived value are tracked as separate columns on
purpose, and `KNOWN_DIVERGENCES` is currently empty: an unresolved
`AskUserQuestion` used to be a session blocked on its human that Phase 1
reported as WORKING, until `derive_turn_boundary` started reading the
`tool_use` block's `name` (task 4). The divergence set is still asserted to
be *exactly* `KNOWN_DIVERGENCES` — now the empty set — rather than dropped,
so a future regression that makes any fixture's derived status stop matching
its ground truth fails loudly here instead of being silently absorbed.

No real session store (`~/.claude/`, `~/.codex/`,
`~/.local/share/opencode/`) is opened, globbed, or read by this module or by
anything under `tests/fixtures/`. Every poisoned record is written under
pytest's `tmp_path` and every string in it was invented for the test.
"""

import json
import re
from pathlib import Path

from palaver.cli import SUBCOMMANDS, fixture_lint, main
from palaver.cli.fixture_lint import (
    ACCEPTED,
    RULE_BAD_IDENTIFIER,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_RECORD_TYPE,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_UNTERMINATED_FILE,
)
from palaver.observer.signals import PHASE1_STATUS_RANGE, Status, Tri, derive_status
from palaver.observer.turn_boundary import BASIS_ASSISTANT_FINAL, derive_signals

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

#: Every case the plan requires the corpus to cover. A fixture deleted or
#: retagged shows up here as a missing case rather than as a quietly smaller
#: corpus.
REQUIRED_CASES = frozenset(
    {
        "WORKING",
        "WAITING_FOR_USER",
        "QUESTION",
        "FINISHED",
        "ERROR",
        "COMPACT_BOUNDARY",
        "MID_TOOL_USE",
        "NO_STOP_HOOK",
        "SLASH_COMMAND",
    }
)

#: Fixtures whose ground truth and derived status disagree, named
#: exhaustively. Empty today — `question-askuserquestion-unresolved.jsonl`
#: was the one entry until task 4 fixed the underlying defect (an unresolved
#: `AskUserQuestion` derived WORKING instead of AWAITING_HUMAN). Kept as a
#: named, asserted-equal set rather than deleted: asserting the set is
#: *equal* to this — not that it contains it, and not just dropping the
#: check now that it is empty — is what stops a future regression from being
#: absorbed into "known divergence" without anyone deciding to accept it.
KNOWN_DIVERGENCES = frozenset()


# --- fixture corpus helpers --------------------------------------------------


def _records(path: Path) -> list[dict]:
    """Decode one fixture file into records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _status(records: list[dict]) -> Status:
    """Derive a status the way the observer does, with mtime withheld.

    mtime never moves a status — it is corroboration only — but withholding it
    keeps every assertion here independent of when the repository was checked
    out.
    """
    return derive_status(derive_signals(records, store_mtime=None).signals)


def _fixture_files() -> list[Path]:
    return sorted(FIXTURES.glob("*.jsonl"))


# --- poisoned-corpus helpers -------------------------------------------------


def _corpus(tmp_path: Path, records: list[dict], *, terminated: bool = True) -> Path:
    """Write a one-file corpus under `tmp_path` and return its directory."""
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    body = b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)
    if not terminated:
        body = body.rstrip(b"\n")
    (root / "poisoned.jsonl").write_bytes(body)
    return root


def _lint(root: Path) -> int:
    return main(["fixture-lint", str(root)])


def _valid_assistant(**overrides) -> dict:
    """A record the allowlist accepts, so a test can poison one thing about it."""
    record = {
        "type": "assistant",
        "sessionId": "fixture-poison-control",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "the deploy finished"}],
        },
    }
    record.update(overrides)
    return record


# --- README metadata parsing -------------------------------------------------

_HEADING = re.compile(r"^### `(?P<name>[^`]+)`\s*$")
_FIELD = re.compile(r"^- \*\*(?P<key>[^:*]+):\*\* (?P<value>.+)$")

#: Fields every entry must carry. `divergence` is required only where ground
#: truth and the derived value disagree.
REQUIRED_FIELDS = (
    "case",
    "ground truth",
    "derived today",
    "phase 3 target",
    "boundary basis",
    "last message-bearing record",
    "unresolved tool_use",
    "latest tool outcome",
    "channel",
    "derivation",
)

#: What a derivation must actually name for each derived status. These are the
#: facts that *fix* the label, and they differ by status: an ERROR is fixed by
#: the latest tool outcome and rule ordering, not by the turn boundary, and a
#: file with no conversational record has no last message-bearing record to
#: point at. A single marker set would let two entries pass while naming
#: nothing that decides them.
REQUIRED_MARKERS = {
    "WORKING": ("last message-bearing record", "tool_use"),
    "AWAITING_HUMAN": ("last message-bearing record", "tool_use"),
    "ERROR": ("tool_result", "is_error"),
    "UNKNOWN": ("no message-bearing record",),
}

#: Restatements of authorial intent. A derivation built from one of these is
#: circular: it explains the label by the fact that somebody chose it.
INTENT_PHRASES = (
    "constructed to be",
    "designed to be",
    "intended to be",
    "written to be",
    "built to be",
    "meant to be",
    "authored to be",
    "chosen to be",
)


def parse_metadata(text: str) -> dict[str, dict[str, str]]:
    """Parse `tests/fixtures/README.md` into one field mapping per fixture."""
    entries: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    key: str | None = None
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = {}
            entries[heading["name"]] = current
            key = None
            continue
        if current is None:
            continue
        field = _FIELD.match(line)
        if field:
            key = field["key"].strip()
            current[key] = field["value"].strip()
            continue
        if not line.strip():
            key = None
        elif key is not None and line.startswith("  "):
            current[key] = f"{current[key]} {line.strip()}"
    return entries


def derivation_problems(status: str, derivation: str) -> list[str]:
    """Report every reason `derivation` fails to fix `status` structurally.

    Args:
        status: The derived status the entry documents.
        derivation: The entry's derivation prose.

    Returns:
        A list of problems, empty when the derivation names the structural
        facts that decide `status` and restates no authorial intent.
    """
    problems = []
    lowered = derivation.lower()
    for phrase in INTENT_PHRASES:
        if phrase in lowered:
            problems.append(f"restates authorial intent: {phrase!r}")
    for marker in REQUIRED_MARKERS[status]:
        if marker.lower() not in lowered:
            problems.append(f"never names the structural fact {marker!r}")
    return problems


def _metadata() -> dict[str, dict[str, str]]:
    return parse_metadata((FIXTURES / "README.md").read_text(encoding="utf-8"))


# --- the linter: the corpus it ships with ------------------------------------


def test_committed_corpus_passes_the_linter(capsys):
    """Catches the corpus and the allowlist drifting apart.

    This is agreement, not safety: a linter that accepted everything would
    also pass. The negative tests below are the safety proof.
    """
    assert _lint(FIXTURES) == 0
    stdout = capsys.readouterr().out
    assert "rejected: 0" in stdout
    assert f"files: {len(_fixture_files())}" in stdout


def test_fixture_lint_is_registered_as_a_subcommand():
    """Catches a linter that exists but is not reachable from the CLI."""
    assert fixture_lint in SUBCOMMANDS
    assert fixture_lint.NAME == "fixture-lint"
    assert callable(fixture_lint.add_arguments)
    assert callable(fixture_lint.run)


def test_progress_goes_to_stderr_and_never_to_stdout(capsys):
    """Catches per-file progress leaking into the result stream (INV-1)."""
    _lint(FIXTURES)
    captured = capsys.readouterr()
    assert "linting 1/" in captured.err
    assert "linting" not in captured.out


# --- the linter: poisoned records, one dimension each ------------------------


def test_unclassified_record_fails(tmp_path, capsys):
    """Catches an unrecognized record shape being waved through (INV-9 gate).

    The poisoned record carries a `system` subtype nobody has classified. Its
    text is *phrasebook-approved* and its `sessionId` is well-formed, so the
    shape rule is the only rule that can fire — which makes the reported rule
    name an assertion about classification rather than a coincidence.
    """
    poisoned = {
        "type": "system",
        "subtype": "palaver_unclassified_subtype",
        "sessionId": "fixture-poison",
        "content": "the deploy finished",
    }
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNKNOWN_SYSTEM_SUBTYPE in capsys.readouterr().out

    # Positive control: the same record, the same phrasebook text, the same
    # path — only the subtype changes to one the adapter classifies.
    accepted = dict(poisoned, subtype="compact_boundary")
    assert _lint(_corpus(tmp_path, [accepted])) == 0


def test_prose_in_an_unexpected_field_fails(tmp_path, capsys):
    """Catches free text smuggled into a key no shape declares.

    The record is a valid `assistant` record and the added `cwd` value is
    phrasebook text, so nothing about the *content* can reject it. Only the
    key-set rule can, which is what makes a real pasted record — carrying
    `uuid`, `timestamp`, `cwd` — fail before its prose is ever read.

    The phrasebook value is deliberate but not what makes this single
    dimension: key-set validation runs ahead of every text check, so an
    unexpected key reports `unexpected-key` whatever it holds. Choosing an
    allowlisted string only removes the doubt about which rule fired.
    """
    poisoned = _valid_assistant(cwd="the deploy finished")
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    stdout = capsys.readouterr().out
    assert RULE_UNEXPECTED_KEY in stdout
    assert "cwd" in stdout

    # Positive control: identical record with the extra key removed.
    assert _lint(_corpus(tmp_path, [_valid_assistant()])) == 0


def test_empty_record_of_a_novel_type_fails(tmp_path, capsys):
    """Catches an allowlist that only inspects records it already understands.

    The record carries no text at all, so there is nothing to sanitize and
    nothing to grep for. It must still fail, because "carries no prose today"
    is not the same claim as "is a shape somebody reviewed".
    """
    assert _lint(_corpus(tmp_path, [{"type": "telemetry"}])) == 1
    assert RULE_UNKNOWN_RECORD_TYPE in capsys.readouterr().out

    # Positive control: an equally small record of a type the allowlist knows.
    assert _lint(_corpus(tmp_path, [_valid_assistant()])) == 0


def test_unallowlisted_prose_in_a_known_shape_fails(tmp_path, capsys):
    """Catches the actual leak: a real sentence inside a perfectly valid record.

    Shape is not enough. This record's type, keys, `sessionId`, and content
    block are all exactly what the allowlist wants; only the sentence is
    unreviewed, so the phrasebook rule is the only one that can fire.
    """
    poisoned = _valid_assistant(
        message={
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "the northwind reconciliation job is still failing on row 40191",
                }
            ],
        }
    )
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNALLOWLISTED_TEXT in capsys.readouterr().out

    # Positive control: same shape, same everything, phrasebook sentence.
    assert _lint(_corpus(tmp_path, [_valid_assistant()])) == 0


def test_prose_in_a_nested_tool_input_key_fails(tmp_path, capsys):
    """Catches free text smuggled into a JSON *key* rather than a value.

    A key is free text too, and a checker that only phrasebooks values walks
    straight past it. Everything else about this record is already allowlisted
    — type, keys, `sessionId`, tool name, `tool_use` id, and the leaf value —
    so `RULE_BAD_IDENTIFIER` at `input.questions[0]` is the only rule that can
    fire, which also separates it from the `sessionId` pattern check that
    reports the same rule. `AskUserQuestion`'s nested input is the one place in
    the committed corpus this recursive walk runs.
    """
    poisoned = _valid_assistant(
        message={
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "AskUserQuestion",
                    "input": {
                        "questions": [{"is the nightly reconciliation job still down": True}]
                    },
                }
            ],
        }
    )
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    stdout = capsys.readouterr().out
    assert RULE_BAD_IDENTIFIER in stdout
    assert "input.questions[0]" in stdout

    # Positive control: same record, same nesting depth, identifier-shaped key.
    control = _valid_assistant(
        message={
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu-1",
                    "name": "AskUserQuestion",
                    "input": {"questions": [{"multiSelect": True}]},
                }
            ],
        }
    )
    assert _lint(_corpus(tmp_path, [control])) == 0


def test_a_real_shaped_session_id_fails(tmp_path, capsys):
    """Catches a record pasted from a real store keeping its real session id.

    Claude Code session ids are UUIDs. Requiring `fixture-*` makes provenance
    a structural property of the value rather than a claim about it, so this
    is the rule a copy-paste dies on first.
    """
    poisoned = _valid_assistant(sessionId="9f1c2d34-5e6f-4a7b-8c9d-0e1f2a3b4c5d")
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_BAD_IDENTIFIER in capsys.readouterr().out

    # Positive control: identical record with a synthetic session id.
    assert _lint(_corpus(tmp_path, [_valid_assistant()])) == 0


def test_unterminated_final_line_fails(tmp_path, capsys):
    """Catches a record smuggled past the gate by omitting the trailing newline.

    JSONL's separator is the newline, so a reader using `read_complete_records`
    withholds an unterminated last line. A linter that used the same reader
    would classify everything *except* the one record somebody took the
    trouble to hide.
    """
    assert _lint(_corpus(tmp_path, [_valid_assistant()], terminated=False)) == 1
    assert RULE_UNTERMINATED_FILE in capsys.readouterr().out

    # Positive control: byte-identical corpus with the newline restored.
    assert _lint(_corpus(tmp_path, [_valid_assistant()], terminated=True)) == 0


def test_poisoned_record_rejection_comes_from_the_classifier(tmp_path, monkeypatch):
    """Catches a negative suite that passes for reasons other than classification.

    Every test above asserts a non-zero exit on a poisoned corpus. A linter
    that rejected any path it was given, or that crashed on startup, would
    satisfy all of them. This one pins the exit to the allowlist itself: with
    `classify_record` stubbed to accept everything, the *same* corpus at the
    *same* path through the *same* argument parsing must exit 0. If the
    rejection had come from argparse or a missing directory, the stubbed run
    would still fail and this test would fail with it.
    """
    poisoned = {
        "type": "system",
        "subtype": "palaver_unclassified_subtype",
        "sessionId": "fixture-poison",
        "content": "the deploy finished",
    }
    root = _corpus(tmp_path, [poisoned])
    assert _lint(root) == 1

    monkeypatch.setattr(fixture_lint, "classify_record", lambda record: ACCEPTED)
    assert _lint(root) == 0


def test_usage_failures_exit_two_not_one(tmp_path, monkeypatch):
    """Catches a missing path being reported as though a record were rejected.

    The two non-zero codes are distinct so "the linter rejected my record" and
    "the linter never saw my record" cannot be confused — including by the
    tests above, which assert exit 1 specifically.
    """
    assert _lint(tmp_path / "does-not-exist") == 2

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _lint(empty) == 2

    # Positive control: stubbing the classifier cannot rescue either case,
    # because neither one reached a classifier at all.
    monkeypatch.setattr(fixture_lint, "classify_record", lambda record: ACCEPTED)
    assert _lint(tmp_path / "does-not-exist") == 2
    assert _lint(empty) == 2


# --- the metadata: labels that can be checked against the file ---------------


def test_every_fixture_has_a_metadata_entry_naming_the_structural_facts():
    """Catches a fixture with no ground truth, or one labelled by assertion.

    Every entry must carry the full field set and a derivation that names the
    structural facts fixing its status — which record is last message-bearing,
    whether any `tool_use` is unresolved, what the latest tool outcome was.
    """
    entries = _metadata()
    names = {path.name for path in _fixture_files()}
    assert names, "the corpus is empty"
    assert set(entries) == names

    for name in sorted(names):
        entry = entries[name]
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        assert not missing, f"{name} is missing metadata fields {missing}"
        assert entry["derived today"] in REQUIRED_MARKERS, f"{name} documents an unknown status"
        problems = derivation_problems(entry["derived today"], entry["derivation"])
        assert not problems, f"{name} derivation is not checkable: {problems}"


def test_metadata_derivation_rejects_authorial_intent():
    """Catches a validator that would accept a circular derivation.

    "Constructed to be WORKING" explains the label by the fact somebody chose
    it, which is exactly what a ground-truth corpus must not rest on. The
    positive control is a real committed derivation, so this cannot pass by
    rejecting everything.
    """
    problems = derivation_problems("WORKING", "Constructed to be WORKING.")
    assert any("authorial intent" in problem for problem in problems)
    assert any("structural fact" in problem for problem in problems)

    # A derivation that names the facts but frames them as intent still fails.
    hedged = (
        "The last message-bearing record is an `assistant` record with an "
        "unresolved `tool_use`, and it was constructed to be WORKING."
    )
    assert derivation_problems("WORKING", hedged)

    # Positive control: the committed derivation for a real fixture passes.
    entry = _metadata()["working-mid-tool-use.jsonl"]
    assert derivation_problems(entry["derived today"], entry["derivation"]) == []


def test_required_cases_are_all_covered():
    """Catches a required case quietly leaving the corpus."""
    cases = {entry["case"] for entry in _metadata().values()}
    assert REQUIRED_CASES <= cases, f"uncovered cases: {sorted(REQUIRED_CASES - cases)}"


# --- accuracy: derived status against documented ground truth ----------------


def test_derived_status_matches_the_documented_derived_status():
    """Catches the observer changing what it reports without the corpus noticing."""
    entries = _metadata()
    for path in _fixture_files():
        documented = entries[path.name]["derived today"]
        actual = _status(_records(path))
        assert actual.value == documented, (
            f"{path.name}: derived {actual.value}, documented {documented}"
        )
        assert actual in PHASE1_STATUS_RANGE


def test_ground_truth_matches_derived_status_except_for_known_divergences():
    """Catches a new wrong answer being absorbed into "already documented".

    The divergence set is asserted equal to `KNOWN_DIVERGENCES`, not merely to
    contain it, so a second fixture whose derived status stops matching the
    truth fails here rather than being explained away in the README.
    """
    entries = _metadata()
    diverging = {
        name for name, entry in entries.items() if entry["ground truth"] != entry["derived today"]
    }
    assert diverging == set(KNOWN_DIVERGENCES)

    for name, entry in entries.items():
        if name in KNOWN_DIVERGENCES:
            assert "divergence" in entry, f"{name} diverges without saying why"
        else:
            assert entry["ground truth"] == _status(_records(FIXTURES / name)).value


def test_unresolved_askuserquestion_is_reported_as_awaiting_human():
    """Pins the fix: an unresolved `AskUserQuestion` now reports AWAITING_HUMAN.

    An unresolved `AskUserQuestion` is a session that has stopped and put a
    prompt in front of its human, not one still busy. Before task 4,
    `derive_turn_boundary` checked only that a `tool_use` block exists, never
    which tool it names, and reported WORKING — the costly direction, because
    the human saw no reason to look. The fix reads the tool name straight off
    the block, so this fixture's ground truth and derived status now agree
    and it no longer belongs in `KNOWN_DIVERGENCES`.
    """
    name = "question-askuserquestion-unresolved.jsonl"
    entry = _metadata()[name]
    assert entry["ground truth"] == Status.AWAITING_HUMAN.value
    assert entry["derived today"] == Status.AWAITING_HUMAN.value
    assert entry["phase 3 target"] == Status.QUESTION.value
    assert name not in KNOWN_DIVERGENCES

    records = _records(FIXTURES / name)
    assert _status(records) is Status.AWAITING_HUMAN

    # Non-vacuity: the tool name really is what the record carries, and it is
    # what the fix reads — not a relabelled fixture with the evidence gone.
    assert records[-1]["message"]["content"][0]["name"] == "AskUserQuestion"


def test_finished_session_is_awaiting_human_and_never_done():
    """Catches silence being read as completion.

    The work in this fixture is finished by any human reading and the file has
    stopped growing. Structure proves control returned to the human; it does
    not prove the work is done, and a confident wrong DONE tells the human a
    session needs nothing when it may be waiting on them.
    """
    status = _status(_records(FIXTURES / "finished-session.jsonl"))
    assert status is Status.AWAITING_HUMAN
    assert status is not Status.DONE
    assert Status.DONE not in PHASE1_STATUS_RANGE
    assert _metadata()["finished-session.jsonl"]["phase 3 target"] == Status.DONE.value


def test_slash_command_record_does_not_take_the_turn():
    """Catches a `<command-name>` record being read as a human turn (INV-8).

    The record is `isMeta: false`, so the structural flag cannot classify it
    and the injected-prefix table has to do the work. The positive control
    swaps only that record's text for an ordinary human sentence: the status
    flips to WORKING, which is what shows the AWAITING_HUMAN above came from
    the prefix table rather than from the fixture being short.
    """
    name = "slash-command-after-reply.jsonl"
    records = _records(FIXTURES / name)
    assert records[-1]["isMeta"] is False
    assert records[-1]["message"]["content"][0]["text"].startswith("<command-name>")

    observation = derive_signals(records, store_mtime=None)
    assert derive_status(observation.signals) is Status.AWAITING_HUMAN
    assert observation.boundary.basis == BASIS_ASSISTANT_FINAL
    assert _metadata()[name]["ground truth"] == Status.AWAITING_HUMAN.value

    # Positive control: same file, same length, only the prefix removed.
    records[-1]["message"]["content"][0]["text"] = "deploy status?"
    assert derive_status(derive_signals(records, store_mtime=None).signals) is Status.WORKING


def test_stop_hook_corroborates_without_moving_the_status():
    """Catches corroboration being folded into the boundary it corroborates.

    The pair differs by one trailing `system` record. Status and basis must be
    identical; only corroboration may change. A stop-hook record that moved
    the status would be a boundary resting on a signal that is absent from
    most real sessions.
    """
    without = derive_signals(_records(FIXTURES / "ended-without-stop-hook.jsonl"), store_mtime=None)
    with_hook = derive_signals(_records(FIXTURES / "ended-with-stop-hook.jsonl"), store_mtime=None)

    assert derive_status(without.signals) is derive_status(with_hook.signals)
    assert without.boundary.basis == with_hook.boundary.basis == BASIS_ASSISTANT_FINAL
    assert without.boundary.corroboration is Tri.UNKNOWN
    assert with_hook.boundary.corroboration is Tri.TRUE
