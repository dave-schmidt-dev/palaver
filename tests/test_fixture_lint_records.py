"""
CLI registration, corpus-wide sanity, and Claude Code's poisoned-record negative suite
(the classify_record patch pin).
"""

from pathlib import Path

from palaver.cli import SUBCOMMANDS, fixture_lint
from palaver.cli.fixture_lint import (
    ACCEPTED,
    RULE_BAD_IDENTIFIER,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_RECORD_TYPE,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_UNTERMINATED_FILE,
)
from tests._fixture_lint_support import FIXTURES, _corpus, _lint


def _all_fixture_files() -> list[Path]:
    """Every file under the corpus, recursively — not every `.jsonl`.

    Mirrors `lint_tree`'s own discovery, so an assertion about "how many
    files did the linter read" stays correct as sources are added in
    subdirectories, without pulling those subdirectories into
    `_fixture_files()`'s Claude-Code-only ground-truth loop.

    This used to be `rglob("*.jsonl")`, matching a linter that only looked at
    `.jsonl`. That agreement was the problem: the helper and the linter shared
    a blind spot, so the count assertion below confirmed 21 of 28 files and
    read as full coverage.
    """
    return sorted(
        path
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name not in fixture_lint.IGNORED_NAMES
    )


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


# --- the linter: the corpus it ships with ------------------------------------


def test_committed_corpus_passes_the_linter(capsys):
    """Catches the corpus and the allowlist drifting apart.

    This is agreement, not safety: a linter that accepted everything would
    also pass. The negative tests below are the safety proof.
    """
    assert _lint(FIXTURES) == 0
    stdout = capsys.readouterr().out
    assert "rejected: 0" in stdout
    assert f"files: {len(_all_fixture_files())}" in stdout


def test_source_corpus_covers_all_three_sources(capsys):
    """Proves one recursive `fixture-lint` pass covers Claude Code, Codex,
    and OpenCode together, not three separately-green corpora someone forgot
    to wire into the same command.

    Done-when (task 7.0): "`uv run palaver fixture-lint tests/fixtures` exits
    0 across all three sources."
    """
    assert _lint(FIXTURES) == 0
    stdout = capsys.readouterr().out
    assert "rejected: 0" in stdout
    assert f"files: {len(_all_fixture_files())}" in stdout
    assert sorted((FIXTURES / "codex").glob("*.jsonl")), "no codex fixtures committed"
    assert sorted((FIXTURES / "opencode").glob("*.jsonl")), "no opencode fixtures committed"


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

    monkeypatch.setattr(fixture_lint.annotation, "classify_record", lambda record: ACCEPTED)
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
    monkeypatch.setattr(fixture_lint.annotation, "classify_record", lambda record: ACCEPTED)
    assert _lint(tmp_path / "does-not-exist") == 2
    assert _lint(empty) == 2
