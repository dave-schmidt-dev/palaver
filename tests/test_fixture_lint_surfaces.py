"""
Every non-JSONL-record regime: file-type routing, narrative/README phrasebook checks,
structural-token spans, data/JSON, golden files, and the rule-registration pin.
"""

import json
from pathlib import Path

from palaver.cli import fixture_lint, main
from tests._fixture_lint_support import FIXTURES

# ---------------------------------------------------------------------------
# Surfaces that are not JSONL records.
#
# Until 2026-08-15 discovery was `rglob("*.jsonl")`, so seven committed files
# were never opened by the gate that exists to read them — including one
# holding literal `HUMAN:`/`AGENT:` transcript lines. These tests cover the
# three new regimes and, more importantly, the property that makes the gap
# unrepeatable: the set of files read equals the set of files present.
# ---------------------------------------------------------------------------


def _write_corpus_file(tmp_path, name: str, text: str) -> Path:
    """Write one file into a throwaway corpus directory."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_every_file_under_the_committed_corpus_is_read():
    """The assertion the `*.jsonl` glob would have failed.

    Not "the linter found some files" but "the linter found *these* files":
    a future `.yaml` or `.csv` fixture trips this before it can ship
    unchecked, which a count-based assertion would not.
    """
    report = fixture_lint.lint_tree(FIXTURES)
    present = {
        path
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name not in fixture_lint.IGNORED_NAMES
    }
    assert report.files == len(present)
    assert report.files > len(list(FIXTURES.rglob("*.jsonl")))


def test_the_committed_corpus_passes_every_regime():
    """The real corpus, end to end, with no rejection of any kind."""
    report = fixture_lint.lint_tree(FIXTURES)
    assert report.rejections == ()
    assert report.records > 0
    assert report.strings > 0


def test_an_unclaimed_extension_is_a_rejection_not_a_skip(tmp_path):
    """The rule that makes the coverage property hold as the corpus grows."""
    _write_corpus_file(tmp_path, "notes.yaml", "greeting: hello\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert report.files == 1
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNKNOWN_FILE_TYPE]


def test_a_file_with_no_suffix_is_also_rejected(tmp_path):
    _write_corpus_file(tmp_path, "Makefile", "all:\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNKNOWN_FILE_TYPE]


def test_the_ignored_name_exemption_covers_only_exact_names(tmp_path):
    """`.DS_Store` is skipped; another dotfile is not.

    The exemption exists so Finder cannot break the gate. If it were a
    dotfile *pattern* it would also hide `.secret.jsonl`, which is the class
    of silent skip this whole change removes.
    """
    _write_corpus_file(tmp_path, ".DS_Store", "binary junk\n")
    _write_corpus_file(tmp_path, ".hidden", "junk\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert report.files == 1
    assert [r.path.name for r in report.rejections] == [".hidden"]


# --- narrative (.md) -------------------------------------------------------


def test_a_readme_carrying_a_real_session_uuid_is_rejected(tmp_path):
    _write_corpus_file(
        tmp_path,
        "README.md",
        "The corpus was sampled from session 3f2a1b9c-4d5e-4f6a-8b7c-9d0e1f2a3b4c.\n",
    )
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_PROVENANCE_MARKER]


def test_a_readme_carrying_an_absolute_home_path_is_rejected(tmp_path):
    _write_corpus_file(tmp_path, "README.md", "Sampled from /Users/someone/Projects/thing.\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_PROVENANCE_MARKER]


def test_a_readme_code_span_of_pasted_prose_is_rejected(tmp_path):
    """A quoted example is the realistic way real content reaches a README."""
    _write_corpus_file(tmp_path, "README.md", "For example `please ship the invoice import`.\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_readme_narrative_prose_is_not_phrasebook_checked(tmp_path):
    """The honest scope of the narrative regime, asserted rather than implied.

    Authored English outside code spans is checked for real-store markers
    only. Stating this as a test means the weaker check is a documented
    decision rather than something a reader has to infer from a passing run.
    """
    _write_corpus_file(tmp_path, "README.md", "This paragraph was written for the corpus.\n")
    assert fixture_lint.lint_tree(tmp_path).rejections == ()


def test_explicit_provenance_check_rejects_verbatim_source_prose(tmp_path):
    fixture_root = tmp_path / "fixtures"
    _write_corpus_file(
        fixture_root,
        "fixture.jsonl",
        '{"type":"fixture","message":"This line was copied from a source session."}\n',
    )
    source = _write_corpus_file(
        tmp_path / "source", "session.jsonl", "This line was copied from a source session.\n"
    )

    rejections = fixture_lint.lint_provenance(fixture_root, (source,))

    assert [rejection.rule for rejection in rejections] == [fixture_lint.RULE_SOURCE_PROVENANCE]


def test_explicit_provenance_check_requires_existing_source(tmp_path):
    _write_corpus_file(tmp_path, "README.md", "This paragraph was written for the corpus.\n")

    assert (
        main(["fixture-lint", str(tmp_path), "--provenance-source", str(tmp_path / "missing")]) == 2
    )


def test_a_code_span_that_wraps_a_line_is_parsed_as_one_span(tmp_path):
    """Regression: the per-line split inverted odd/even after a wrapped span.

    `tests/fixtures/codex/README.md` really does wrap `palaver/cli/
    fixture_lint.py` across a newline. Splitting per line put the opening and
    closing backticks in different splits, which handed the *prose* between
    two spans to the span check and let the real span go unchecked — a false
    rejection and a silent miss from one bug.
    """
    _write_corpus_file(tmp_path, "README.md", "See `palaver/cli/\nfixture_lint.py` and the rest.\n")
    assert fixture_lint.lint_tree(tmp_path).rejections == ()


def test_a_wrapped_span_still_has_its_contents_checked(tmp_path):
    """The other half: correct parsing must not mean skipping the span."""
    _write_corpus_file(tmp_path, "README.md", "See `please ship the\ninvoice import now` here.\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


# --- the structural-token rule --------------------------------------------


def test_a_long_sentence_containing_a_slash_is_not_a_structural_token(tmp_path):
    """Regression: `STRUCTURAL_CHARACTERS` alone passed an 80-word paragraph.

    Found on the first real run. `labels.json`'s `$comment` was accepted as
    "structural" because the paragraph contains a `/`. Any prose mentioning a
    path or a `key: value` pair would have done the same, which is exactly
    the heuristic `SYNTHESIZED_TEXT` warns against.
    """
    span = "we should probably move the report into tests/fixtures before the review"
    verdict = fixture_lint.check_span(span, "where")
    assert not verdict.ok
    assert verdict.rule == fixture_lint.RULE_UNALLOWLISTED_SPAN


def test_a_short_structural_token_is_still_accepted():
    """Positive control: the cap must not swallow what the rule is for."""
    for span in ('role: "user"', 'message.data.finish == "stop"', '{"type", "payload"}'):
        assert fixture_lint.check_span(span, "where").ok


def test_a_documented_command_line_is_accepted():
    """Commands carry no structural character and are covered on purpose."""
    for span in ("palaver fixture-lint", "uv run palaver fixture-lint tests/fixtures"):
        assert fixture_lint.check_span(span, "where").ok


def test_a_sentence_opening_with_the_word_palaver_is_not_a_command():
    verdict = fixture_lint.check_span("palaver watches the sessions you run", "where")
    assert not verdict.ok


def test_a_provenance_marker_beats_every_acceptance_path():
    """The marker scan runs first, so an identifier-shaped id cannot pass.

    A bare UUID has no whitespace, so the single-token rule would otherwise
    accept it — which is the one shape a real session id actually takes.
    """
    verdict = fixture_lint.check_span("3f2a1b9c-4d5e-4f6a-8b7c-9d0e1f2a3b4c", "where")
    assert not verdict.ok
    assert verdict.rule == fixture_lint.RULE_PROVENANCE_MARKER


# --- data (.json) ----------------------------------------------------------


def test_a_json_string_outside_every_allowlist_is_rejected(tmp_path):
    _write_corpus_file(tmp_path, "labels.json", json.dumps({"note": "ship the invoice import"}))
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_the_comment_key_is_not_an_escape_hatch(tmp_path):
    """`$comment` is allowlisted by equality, not by key name.

    A key-name exemption would be an escape hatch inside the strict regime:
    the next string someone would rather not justify becomes a `$comment`
    too. The committed paragraph passes; a different one does not.
    """
    _write_corpus_file(
        tmp_path, "labels.json", json.dumps({"$comment": "anything I like, at length"})
    )
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_the_committed_comment_paragraph_is_allowlisted_verbatim():
    """The two copies of the paragraph cannot drift apart unnoticed."""
    committed = json.loads((FIXTURES / "eval" / "labels.json").read_text())["$comment"]
    assert committed in fixture_lint.DOCUMENTATION_TEXT


def test_the_committed_comment_no_longer_claims_tree_wide_coverage():
    """The false claim this change was found by, asserted gone.

    It said "fixture-lint's allowlist applies tree-wide" while the linter
    read only `.jsonl` — a false statement about the gate, inside the corpus
    the gate covers, on a public remote.
    """
    committed = json.loads((FIXTURES / "eval" / "labels.json").read_text())["$comment"]
    assert "allowlist applies tree-wide" not in committed


def test_json_keys_are_checked_too(tmp_path):
    _write_corpus_file(tmp_path, "labels.json", json.dumps({"ship the invoice import now": 1}))
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_undecodable_json_is_rejected_rather_than_skipped(tmp_path):
    _write_corpus_file(tmp_path, "labels.json", "{not json\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNDECODABLE]


# --- golden (.txt) ---------------------------------------------------------


def test_a_golden_line_with_non_phrasebook_text_is_rejected(tmp_path):
    _write_corpus_file(tmp_path, "out.txt", "HUMAN: ship the invoice import\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_a_golden_line_without_a_label_is_rejected(tmp_path):
    _write_corpus_file(tmp_path, "out.txt", "just some text\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_BAD_GOLDEN_LABEL]


def test_a_golden_line_of_phrasebook_text_is_accepted(tmp_path):
    _write_corpus_file(
        tmp_path,
        "out.txt",
        "HUMAN: refactor the auth module\nSYSTEM(compaction): earlier notes trimmed\n",
    )
    assert fixture_lint.lint_tree(tmp_path).rejections == ()


# --- namespace separation --------------------------------------------------


def test_model_output_is_a_separate_namespace_from_transcript_prose():
    """A paraphrase must not satisfy a record's text check.

    `EXTRACTION_TEXT` holds what the local model wrote *about* the fixtures
    ("Running the test suite"), not what any fixture says ("run the test
    suite"). Merging the two would blur the only distinction the allowlist
    exists to draw — the same reasoning that keeps `ANNOTATION_TEXT` apart.
    """
    assert not (fixture_lint.EXTRACTION_TEXT & fixture_lint.SYNTHESIZED_TEXT)
    assert not (fixture_lint.DOCUMENTATION_TEXT & fixture_lint.SYNTHESIZED_TEXT)
    assert not (fixture_lint.EXTRACTION_TEXT & fixture_lint.ANNOTATION_TEXT)


def test_model_output_text_is_unreachable_from_narrative_and_records(tmp_path):
    """`extra` is a parameter, so widening one surface cannot widen the rest."""
    sample = sorted(fixture_lint.EXTRACTION_TEXT)[0]
    assert not fixture_lint.check_span(sample, "where").ok
    _write_corpus_file(tmp_path, "README.md", f"For example `{sample}`.\n")
    report = fixture_lint.lint_tree(tmp_path)
    assert [r.rule for r in report.rejections] == [fixture_lint.RULE_UNALLOWLISTED_SPAN]


def test_every_new_rule_is_registered_in_the_rule_table():
    """A rule absent from `RULE_NAMES` has no legend and no test hook."""
    for rule in (
        fixture_lint.RULE_UNKNOWN_FILE_TYPE,
        fixture_lint.RULE_UNALLOWLISTED_SPAN,
        fixture_lint.RULE_PROVENANCE_MARKER,
        fixture_lint.RULE_BAD_GOLDEN_LABEL,
    ):
        assert rule in fixture_lint.RULE_NAMES
