"""
README metadata parsing and the ground-truth-accuracy suite: every fixture's documented
derivation matches derive_status/derive_signals.
"""

import json
import re
from pathlib import Path

from palaver.observer.signals import PHASE1_STATUS_RANGE, Status, Tri, derive_status
from palaver.observer.turn_boundary import BASIS_ASSISTANT_FINAL, derive_signals
from tests._fixture_lint_support import FIXTURES

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
    """Claude Code's flat corpus, non-recursive on purpose.

    The metadata/ground-truth tests below (`test_every_fixture_has_a_
    metadata_entry_naming_the_structural_facts` and friends) are Claude-Code
    -specific — they call `derive_status`/`derive_signals`, which know nothing
    about Codex or OpenCode shapes — so this must never see `codex/` or
    `opencode/`. Use `_all_fixture_files()` for anything that means "every
    fixture the linter itself will read".
    """
    return sorted(FIXTURES.glob("*.jsonl"))


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
