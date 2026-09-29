"""Scoring codex_role_class against the committed blind labels."""

from __future__ import annotations

import json
from pathlib import Path

from palaver.ingest.adapters.base import read_complete_records
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED

from .classify import codex_role_class
from .constants import CORPUS_ROOT, HUMAN_CANDIDATE_ROLE, LABELS_PATH, REQUIRED_LABELLED_RECORDS
from .records import message_role

# --- Blind measurement ------------------------------------------------------


def load_labels(labels_path: Path | None = None) -> list[dict]:
    """Read the committed blind channel labels.

    Args:
        labels_path: Labels file. Defaults to `LABELS_PATH`.

    Returns:
        One decoded label row per line, in file order.

    Raises:
        ValueError: A line is not a JSON object. The labels file is a
            committed artifact under test, so a malformed row is a defect to
            surface, not a condition to skip past.
    """
    labels_path = LABELS_PATH if labels_path is None else Path(labels_path)
    raw_rows, _ = read_complete_records(labels_path, 0)
    rows = []
    for index, raw in enumerate(raw_rows):
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{labels_path}:{index} is not a JSON object")
        rows.append(row)
    return rows


def _labelled_record(row: dict, corpus_root: Path) -> dict:
    """Load the single rollout record one label row points at."""
    path = corpus_root / str(row["file"])
    raw_rows, _ = read_complete_records(path, 0)
    index = int(row["index"])
    record = json.loads(raw_rows[index])
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{index} is not a JSON object")
    return record


def measure_role_class(
    labels_path: Path | None = None, corpus_root: Path | None = None
) -> dict[str, int | bool]:
    """Score `codex_role_class` against the committed blind labels.

    The labels were authored from the raw records before this module existed
    and committed in their own commit; this function reads them back and
    counts disagreements. It does not read the committed measurement file, so
    the measurement cannot be forged by editing that file — a test recomputes
    these counts and compares.

    The measured domain is the *role-bearing* records. `codex_role_class`
    returns the harness channel structurally for every record with no message
    role (rule 1), so those rows are vacuously correct and counting them
    would inflate the sample toward the 200-record threshold with records the
    classifier cannot get wrong. `n_labelled_rows` reports the full file size
    alongside, so the narrowing is visible rather than implied.

    `n_errors` is counted over *every* row, not just the role-bearing ones —
    the narrowing applies to the denominator that gates the cap, never to the
    error count, because a false human classification anywhere is the failure
    INV-8 exists to prevent.

    Args:
        labels_path: Labels file. Defaults to `LABELS_PATH`.
        corpus_root: Directory the labels' `file` paths are relative to.
            Defaults to the repository root.

    Returns:
        A counts-only mapping — no transcript content of any kind, so it is
        INV-9-clean by construction (INV-9's git clause):

        - `n_records`: role-bearing labelled records, the gated denominator.
        - `n_labelled_rows`: every row in the labels file.
        - `n_errors`: rows labelled harness that the classifier called human.
        - `n_disagreements`: rows where label and classification differ at
          all, in either direction.
        - `n_discriminating`: role-bearing rows whose blind label differs
          from naive role-mapping (`user` is human, everything else is
          harness). This is the number that says whether the sample tests
          anything: a measurement over rows that all agree with the trivial
          classifier would score perfectly while proving nothing.
        - `n_human_labels` / `n_injected_labels`: the label distribution.
        - `threshold_met`: the full conjunction the cap requires.
    """
    corpus_root = CORPUS_ROOT if corpus_root is None else Path(corpus_root)
    rows = load_labels(labels_path)

    n_records = 0
    n_errors = 0
    n_disagreements = 0
    n_human_labels = 0
    n_injected_labels = 0
    n_discriminating = 0

    for row in rows:
        record = _labelled_record(row, corpus_root)
        label = row["channel"]
        classified = codex_role_class(record)
        role = message_role(record)
        if role is not None:
            n_records += 1
            # Naive role-mapping is the trivial classifier this measurement
            # has to beat: `user` means human, anything else means harness. A
            # row where the blind label agrees with it tests nothing, however
            # it is classified, so the count of rows that *disagree* is the
            # honest measure of how much signal the sample carries.
            if (label == CHANNEL_HUMAN) != (role == HUMAN_CANDIDATE_ROLE):
                n_discriminating += 1
        if label == CHANNEL_HUMAN:
            n_human_labels += 1
        else:
            n_injected_labels += 1
        if classified != label:
            n_disagreements += 1
            if label == CHANNEL_INJECTED and classified == CHANNEL_HUMAN:
                n_errors += 1

    return {
        "n_records": n_records,
        "n_labelled_rows": len(rows),
        "n_errors": n_errors,
        "n_disagreements": n_disagreements,
        "n_discriminating": n_discriminating,
        "n_human_labels": n_human_labels,
        "n_injected_labels": n_injected_labels,
        "threshold_met": n_records >= REQUIRED_LABELLED_RECORDS and n_errors == 0,
    }
