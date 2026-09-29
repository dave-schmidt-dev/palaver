"""
measure_role_class scored against the committed blind labels and the corpus they index.
"""

import json

from palaver.ingest.adapters.codex import (
    CHANNEL_HUMAN,
    CHANNEL_INJECTED,
    LABELS_PATH,
    MEASUREMENT_PATH,
    REQUIRED_LABELLED_RECORDS,
    measure_role_class,
)
from tests._adapter_codex_support import REPO_ROOT, _message

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
