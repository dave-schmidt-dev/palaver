"""
Tree-level orchestration: lint_file (one JSONL file), lint_tree (every file under a
root, routed by regime), and the opt-in lint_provenance check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from . import annotation
from .checks import Verdict
from .constants import (
    RULE_SOURCE_PROVENANCE,
    RULE_UNDECODABLE,
    RULE_UNKNOWN_FILE_TYPE,
    RULE_UNTERMINATED_FILE,
)
from .surfaces import (
    DATA_SUFFIXES,
    GOLDEN_SUFFIXES,
    IGNORED_NAMES,
    KNOWN_SUFFIXES,
    NARRATIVE_SUFFIXES,
    RECORD_SUFFIXES,
    LintReport,
    Rejection,
    _line_of,
    _walk_strings,
    lint_data_file,
    lint_golden_file,
    lint_narrative_file,
)


def lint_file(
    path: Path, *, classify: Callable[[object], Verdict] | None = None
) -> tuple[int, list[Rejection]]:
    """Classify every record in one fixture file.

    A file whose last byte is not a newline is itself a rejection: JSONL's
    record separator is the newline, and a trailing unterminated line is a
    record that a reader using `read_complete_records` would withhold — which
    would let it ride into git having never been classified.

    Args:
        path: The fixture file.
        classify: Classifier override; defaults to `classify_record`, resolved
            at call time so a test can substitute the module attribute.

    Returns:
        `(records_classified, rejections)`.
    """
    classify = annotation.classify_record if classify is None else classify
    raw = path.read_bytes()
    rejections: list[Rejection] = []
    if raw and not raw.endswith(b"\n"):
        rejections.append(
            Rejection(
                path=path,
                line=raw.count(b"\n") + 1,
                rule=RULE_UNTERMINATED_FILE,
                detail="file does not end in a newline, so its last record is unterminated",
            )
        )
    counted = 0
    for number, line in enumerate(raw.split(b"\n"), start=1):
        if number > raw.count(b"\n") and not line:
            continue  # the empty string after the final newline
        counted += 1
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            rejections.append(
                Rejection(path=path, line=number, rule=RULE_UNDECODABLE, detail=str(exc))
            )
            continue
        verdict = classify(record)
        if not verdict.ok:
            rejections.append(
                Rejection(path=path, line=number, rule=verdict.rule, detail=verdict.detail)
            )
    return counted, rejections


def lint_tree(
    root: Path,
    *,
    classify: Callable[[object], Verdict] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> LintReport:
    """Check every file under `root`, routing each to its regime by extension.

    Discovery is `rglob("*")` rather than `rglob("*.jsonl")`, and a suffix no
    regime claims is reported as `RULE_UNKNOWN_FILE_TYPE`. Those two together
    are the property worth having: the set of files this function reads equals
    the set of files present, so a fixture cannot be added in a format the
    linter does not understand and be counted as passing.

    Args:
        root: Corpus directory, or a single fixture file of any known type.
        classify: Record-classifier override; defaults to `classify_record`.
            Applies to `.jsonl` only — the other regimes have no per-record
            classifier to substitute.
        on_status: Progress channel, called once per file before it is read
            (INV-1). Never writes to stdout.

    Returns:
        A `LintReport` over everything read.
    """
    root = Path(root)
    paths = (
        [root]
        if root.is_file()
        else sorted(p for p in root.rglob("*") if p.is_file() and p.name not in IGNORED_NAMES)
    )
    records = 0
    strings = 0
    rejections: list[Rejection] = []
    for index, path in enumerate(paths, start=1):
        if on_status is not None:
            on_status(f"linting {index}/{len(paths)}: {path.name}")
        suffix = path.suffix
        if suffix in RECORD_SUFFIXES:
            counted, found = lint_file(path, classify=classify)
            records += counted
        elif suffix in DATA_SUFFIXES:
            counted, found = lint_data_file(path)
            strings += counted
        elif suffix in GOLDEN_SUFFIXES:
            counted, found = lint_golden_file(path)
            strings += counted
        elif suffix in NARRATIVE_SUFFIXES:
            counted, found = lint_narrative_file(path)
            strings += counted
        else:
            found = [
                Rejection(
                    path=path,
                    line=1,
                    rule=RULE_UNKNOWN_FILE_TYPE,
                    detail=(
                        f"no checker claims {suffix or 'a file with no suffix'}; "
                        f"known types are {sorted(KNOWN_SUFFIXES)}"
                    ),
                )
            ]
        rejections.extend(found)
    return LintReport(
        root=root,
        files=len(paths),
        records=records,
        strings=strings,
        rejections=tuple(rejections),
    )


def lint_provenance(root: Path, source_roots: tuple[Path, ...]) -> tuple[Rejection, ...]:
    """Reject fixture text copied verbatim from explicitly supplied source stores.

    This intentionally has no implicit default: a normal lint must never open
    a real transcript store, while a requested provenance check must not pass
    merely because its comparison corpus is absent.
    """
    source_files: list[Path] = []
    for source_root in source_roots:
        if not source_root.exists():
            raise FileNotFoundError(source_root)
        source_files.extend(
            [source_root]
            if source_root.is_file()
            else sorted(path for path in source_root.rglob("*") if path.is_file())
        )
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in source_files
    )
    rejections: list[Rejection] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        raw = path.read_text(encoding="utf-8")
        spans: list[tuple[int, str]] = []
        if path.suffix in RECORD_SUFFIXES:
            for line_number, line in enumerate(raw.splitlines(), start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                spans.extend((_line_of(raw, text), text) for text, _where in _walk_strings(record))
        elif path.suffix in DATA_SUFFIXES:
            try:
                document = json.loads(raw)
            except json.JSONDecodeError:
                document = None
            if document is not None:
                spans.extend(
                    (_line_of(raw, text), text) for text, _where in _walk_strings(document)
                )
        else:
            spans.extend(enumerate(raw.splitlines(), start=1))
        for line_number, text in spans:
            text = text.strip()
            if len(text) >= 20 and text in source_text:
                rejections.append(
                    Rejection(
                        path, line_number, RULE_SOURCE_PROVENANCE, "verbatim source-store text"
                    )
                )
    return tuple(rejections)
