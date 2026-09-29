"""
The codex_role_label blind-annotation shape: committed label rows, its separate prose
namespace, and its structural-identifier-only overlap with the corpus.
"""

import json

from palaver.cli.fixture_lint import (
    ACCEPTED,
    ANNOTATION_TEXT,
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNKNOWN_RECORD_TYPE,
    SYNTHESIZED_TEXT,
    classify_record,
)
from tests._fixture_lint_support import FIXTURES

# --- codex_role_label annotation rows (task 7.1) -----------------------------
#
# These rows are the blind channel labels. They carry no `type` field, because
# they are byte-identical to the file committed before any classifier existed
# and re-encoding them would forfeit that provenance. `classify_record`
# therefore dispatches them on key set, *before* the `type` lookup — and the
# whole point of asserting rule names below is that a branch which never runs
# still rejects these rows, just as `unknown-record-type` instead. A test that
# only checked "the linter said no" would pass against an unreachable branch.

LABELS_PATH = FIXTURES / "labels" / "codex-role-labels.jsonl"

#: A representative label row, written here rather than read from the corpus so
#: the poisoning tests below perturb exactly one field of a known-good record.
GOOD_LABEL = {
    "file": "tests/fixtures/codex/session-finished.jsonl",
    "index": 0,
    "role": None,
    "channel": "injected",
    "basis": "session_meta envelope: harness-written session header, no typed content",
}


def _label(**overrides) -> dict:
    row = dict(GOOD_LABEL)
    row.update(overrides)
    return row


def test_a_blind_label_row_is_accepted_on_shape_not_on_filename():
    """Positive control: without this, every rejection below is vacuous.

    Acceptance is by key set and value domain, so the same row would be
    accepted at any path. That is deliberate — a by-filename exemption would
    leave INV-9 open for anything else written to the exempt name.
    """
    assert classify_record(GOOD_LABEL) == ACCEPTED


def test_an_unallowlisted_basis_is_rejected_as_unallowlisted_text():
    """The test that proves the annotation branch is actually reached.

    If `classify_record` short-circuited before the key-set dispatch, this row
    would still be rejected — as `unknown-record-type`, since it has no `type`
    field. Asserting the *rule* is what distinguishes "the phrasebook refused
    this prose" from "the linter never looked at it".
    """
    verdict = classify_record(_label(basis="the operator seemed frustrated about the outage"))
    assert not verdict.ok
    assert verdict.rule == RULE_UNALLOWLISTED_TEXT


def test_every_committed_label_row_is_accepted():
    """The committed labels and the allowlist agree, row by row.

    Reads the real file, so a future edit to either side that breaks the
    correspondence fails here with the offending row named.
    """
    rows = [
        json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 19
    for index, row in enumerate(rows):
        verdict = classify_record(row)
        assert verdict.ok, f"row {index} rejected as {verdict.rule}: {verdict.detail}"


def test_annotation_text_is_exactly_the_committed_basis_vocabulary():
    """No unreachable entries, no missing ones.

    An unused entry is prose hardcoded into the linter that no fixture
    justifies, which is exactly the drift INV-9's git clause is meant to make
    visible in review.
    """
    rows = [
        json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    assert {row["basis"] for row in rows} == set(ANNOTATION_TEXT)


def test_annotation_prose_and_transcript_prose_are_separate_namespaces():
    """An annotation sentence must never satisfy a transcript record's text check.

    They are different kinds of claim: `SYNTHESIZED_TEXT` is invented content a
    fixture may *contain*, `ANNOTATION_TEXT` is commentary *about* fixtures.
    Merging them would let either one launder the other.
    """
    assert not (ANNOTATION_TEXT & SYNTHESIZED_TEXT)
    for basis in ANNOTATION_TEXT:
        verdict = classify_record(
            {
                "type": "message",
                "sessionId": "fixture-a",
                "message": {"role": "user", "content": [{"type": "text", "text": basis}]},
            }
        )
        assert not verdict.ok, "annotation prose was accepted as transcript content"


def test_annotation_overlap_with_the_corpus_is_only_structural_identifiers():
    """Re-derives the INV-9 claim the `ANNOTATION_TEXT` comment records.

    The comment asserts a *bounded* overlap, not zero overlap: annotation
    sentences name Codex's schema identifiers, so those tokens necessarily
    recur in the fixtures that carry them. This recomputes the scan and fails
    if any overlap appears that is not one of those identifiers — which is how
    a future annotation quoting real transcript prose would be caught.
    """
    window = 14
    identifiers = (
        "last_agent_message",
        "codex_error_info",
        "context_compacted",
        "replacement_history",
        "environment_context",
        # A coincidental English collision, not a quotation: the annotation's
        # "an operating instruction" against the fixture's "you are operating
        # inside a sandboxed fixture container". Both sides are invented.
        " operating ins",
    )

    def mask(text: str) -> str:
        for token in identifiers:
            text = text.replace(token, "\x00" * len(token))
        return text

    corpus = [
        mask(path.read_text(encoding="utf-8"))
        for path in sorted(FIXTURES.rglob("*.jsonl"))
        if path != LABELS_PATH
    ]
    assert corpus, "no fixtures found to scan"

    for basis in ANNOTATION_TEXT:
        assert len(basis) >= window, f"{basis!r} is too short to scan, so it passes vacuously"
        masked = mask(basis)
        for start in range(len(masked) - window + 1):
            segment = masked[start : start + window]
            if "\x00" in segment:
                continue
            hit = next((text for text in corpus if segment in text), None)
            assert hit is None, f"annotation segment {segment!r} appears verbatim in the corpus"


def test_a_label_row_cannot_point_outside_the_committed_corpus():
    """A label naming an unreviewed path is describing something nobody vetted."""
    verdict = classify_record(_label(file="/Users/someone/.codex/sessions/rollout.jsonl"))
    assert not verdict.ok
    assert verdict.rule == RULE_BAD_IDENTIFIER


def test_a_label_row_rejects_an_unknown_channel():
    verdict = classify_record(_label(channel="maybe-human"))
    assert not verdict.ok
    assert verdict.rule == RULE_BAD_VALUE


def test_a_label_row_rejects_an_unknown_role():
    verdict = classify_record(_label(role="operator"))
    assert not verdict.ok
    assert verdict.rule == RULE_BAD_VALUE


def test_a_label_row_rejects_a_boolean_index():
    """`isinstance(True, int)` is True in Python, so this needs its own guard."""
    verdict = classify_record(_label(index=True))
    assert not verdict.ok
    assert verdict.rule == RULE_BAD_VALUE


def test_an_extra_key_stops_a_record_being_treated_as_an_annotation():
    """Key-set dispatch is exact, so a near-miss falls through to `type`.

    This is the safe direction: an unrecognized record is rejected, never
    silently classified under a shape it does not have.
    """
    verdict = classify_record({**GOOD_LABEL, "note": "extra"})
    assert not verdict.ok
    assert verdict.rule == RULE_UNKNOWN_RECORD_TYPE
