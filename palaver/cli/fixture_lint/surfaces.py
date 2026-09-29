"""
Rejection/LintReport, the per-extension regime routing constants, and the three non-
record file checkers (data/golden/narrative) plus their string-walking helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    RULE_BAD_GOLDEN_LABEL,
    RULE_PROVENANCE_MARKER,
    RULE_UNALLOWLISTED_SPAN,
    RULE_UNDECODABLE,
    SYNTHESIZED_TEXT,
)
from .spans import GOLDEN_LINE, check_span, provenance_markers


@dataclass(frozen=True)
class Rejection:
    """One record the allowlist refused, located precisely enough to fix."""

    path: Path
    line: int
    rule: str
    detail: str


@dataclass(frozen=True)
class LintReport:
    """The outcome of one `fixture-lint` run.

    Attributes:
        root: Directory the corpus was read from.
        files: How many files were read — every file under `root`, not every
            file of some extension. A count that moved when a checker was
            added would hide the thing this field is for.
        records: How many `.jsonl` records were classified.
        strings: How many quoted strings, golden lines, and code spans were
            checked on the non-record surfaces. Reported beside `records`
            because a regime that silently checked nothing would otherwise
            look identical to one that passed.
        rejections: Everything the allowlist refused, in file order.
    """

    root: Path
    files: int
    records: int
    strings: int
    rejections: tuple[Rejection, ...]


#: Which regime reads which extension. Exhaustive by construction: `lint_tree`
#: rejects any suffix absent from this mapping, so adding a `.yaml` fixture is
#: a decision someone has to make here rather than a file that quietly ships
#: unchecked.
#: Filesystem droppings that are not fixtures and never will be. Matched by
#: **exact name**, not by a dotfile or wildcard rule: `.DS_Store` appears the
#: moment anyone opens the corpus in Finder, and failing the gate on it would
#: train people to reach for a skip. Any other unexpected file — including any
#: other dotfile — is still a rejection, so this exemption cannot grow by
#: accident the way a `.*` pattern would.
IGNORED_NAMES = frozenset({".DS_Store"})

RECORD_SUFFIXES = frozenset({".jsonl"})
DATA_SUFFIXES = frozenset({".json"})
GOLDEN_SUFFIXES = frozenset({".txt"})
NARRATIVE_SUFFIXES = frozenset({".md"})
KNOWN_SUFFIXES = RECORD_SUFFIXES | DATA_SUFFIXES | GOLDEN_SUFFIXES | NARRATIVE_SUFFIXES


#: Local-model output over the sanitized corpus, kept in a **third** namespace
#: for the same reason `ANNOTATION_TEXT` is kept out of `SYNTHESIZED_TEXT`: a
#: model's paraphrase of a fixture is not fixture content, and letting one
#: satisfy a transcript record's text check would blur the only distinction
#: this allowlist exists to draw.
#:
#: These are the extraction results in `tests/fixtures/eval/e4b_snapshot.json`,
#: which the eval harness produced by running the local model over the linted
#: `.jsonl` corpus. They are paraphrases rather than quotes — the model
#: capitalized and re-tensed ("run the test suite" became "Running the test
#: suite"), which is exactly why they fail an equality check against the
#: phrasebook and why they cannot simply be added to it.
#:
#: Reachable only from `lint_data_file`. A README cannot quote one, and no
#: `.jsonl` record can carry one, because `classify_record` never consults
#: this set.
EXTRACTION_TEXT = frozenset(
    {
        "Running the test suite",
        "Refactor the auth module",
        "The user requested the agent run the test suite.",
    }
)

#: Prose that documents the corpus from *inside* a data file — today just the
#: `$comment` in `tests/fixtures/eval/labels.json`.
#:
#: A fourth namespace rather than a fifteenth `ANNOTATION_TEXT` entry, because
#: that set has a narrower documented meaning (commentary carried by
#: `codex_role_label` records, with a mechanical overlap analysis over exactly
#: 14 strings) and a test that re-derives it. Adding an unrelated paragraph
#: there would falsify both.
#:
#: Allowlisted by equality rather than by exempting the `$comment` key. A key
#: exemption is an escape hatch inside the strict regime: the next string
#: someone would rather not justify becomes a `$comment` too. Equality means
#: editing this paragraph is a two-file change that shows up in review, which
#: is the deliberate act INV-9's git clause is about.
DOCUMENTATION_TEXT = frozenset(
    {
        "Ground truth for task 3.5's eval harness. `path` is relative to "
        "tests/fixtures/. Most entries reuse Phase 1's existing labelled-fixture "
        "corpus (chosen over authoring a parallel transcript corpus, since those "
        "fixtures are already vetted); only decision-database-choice.jsonl is new, "
        "added here under tests/fixtures/eval/ because no existing fixture "
        "demonstrates a resolved user decision. forbidden_quote_substrings names "
        "text a correct extractor must never cite as a user decision's quote -- "
        "either because it is INJECTED-channel (misattribution) or AGENT-channel "
        "(fabrication), never HUMAN-channel. An earlier version of this note said "
        "fixture-lint's allowlist applied tree-wide. That was false when written: "
        "discovery was rglob(*.jsonl), so this file was one of seven the linter "
        "never opened. It became true on 2026-08-15, when every file under "
        "tests/fixtures/ was brought under a checker and an unclaimed extension "
        "became a rejection rather than a skip."
    }
)


def _line_of(raw: str, needle: str) -> int:
    """Best-effort source line for a string found inside a parsed file.

    `json.loads` discards positions, so a rejection in a `.json` fixture would
    otherwise have to report line 1 and leave the reader searching. Falls back
    to 1 when the string cannot be located verbatim (it was escaped, or spans
    a line break).
    """
    index = raw.find(needle)
    if index < 0:
        return 1
    return raw.count("\n", 0, index) + 1


def _walk_strings(value: object, path: str = "$") -> list[tuple[str, str]]:
    """Every string in a decoded JSON document, with its dotted location."""
    if isinstance(value, str):
        return [(value, path)]
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key, item in value.items():
            found.append((key, f"{path} key"))
            found.extend(_walk_strings(item, f"{path}.{key}"))
        return found
    if isinstance(value, list):
        found = []
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, f"{path}[{index}]"))
        return found
    return []


def lint_data_file(path: Path) -> tuple[int, list[Rejection]]:
    """Check every string in a `.json` fixture — keys included.

    Labels, eval snapshots, and measurement files are data, not prose, so they
    get the strict regime: each string must be phrasebook, a command, or a
    structural token. There is deliberately no key that opts a string out.
    `tests/fixtures/eval/labels.json` carries a long `$comment`, and it is
    allowlisted by *equality* in `ANNOTATION_TEXT` rather than by a wildcard
    on the key name, because a wildcard is an escape hatch inside the strict
    regime and the next person who wants to dodge the phrasebook adds one.
    """
    raw = path.read_text()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        return 0, [Rejection(path=path, line=1, rule=RULE_UNDECODABLE, detail=str(exc))]
    rejections: list[Rejection] = []
    strings = _walk_strings(document)
    for value, where in strings:
        verdict = check_span(value, where, extra=EXTRACTION_TEXT | DOCUMENTATION_TEXT)
        if not verdict.ok:
            rejections.append(
                Rejection(
                    path=path,
                    line=_line_of(raw, value),
                    rule=verdict.rule,
                    detail=verdict.detail,
                )
            )
    return len(strings), rejections


def lint_golden_file(path: Path) -> tuple[int, list[Rejection]]:
    """Check a `.txt` golden output line by line.

    Each line is a channel label and a text half. The label is checked by
    shape and the text by the phrasebook, because the text half is the part
    that came out of a transcript.
    """
    rejections: list[Rejection] = []
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = GOLDEN_LINE.match(line)
        if match is None:
            rejections.append(
                Rejection(
                    path=path,
                    line=number,
                    rule=RULE_BAD_GOLDEN_LABEL,
                    detail=f"line is not <label>: <text>: {line[:40]!r}",
                )
            )
            continue
        text = match.group("text")
        if text.strip() and text.strip() not in SYNTHESIZED_TEXT:
            rejections.append(
                Rejection(
                    path=path,
                    line=number,
                    rule=RULE_UNALLOWLISTED_SPAN,
                    detail=f"golden text is not phrasebook: {text[:40]!r}",
                )
            )
    return len(lines), rejections


def lint_narrative_file(path: Path) -> tuple[int, list[Rejection]]:
    """Check a `.md` file: quoted content strictly, authored English by marker.

    A README's purpose *is* authored English, so narrative is the default here
    and quoted content is the exception — the inverse of every other regime.
    The risk a corpus README actually carries is a real record pasted in as an
    example, so fenced blocks and backtick spans go through `check_span` like
    any other quoted string, while the prose around them gets the marker scan.

    That asymmetry is the honest scope of this check and is stated rather than
    hidden: narrative is not proven invented, only proven free of real-store
    artefacts.
    """
    rejections: list[Rejection] = []
    checked = 0
    in_fence = False
    # Narrative is reassembled into one document before the backtick split,
    # because an inline span may wrap a line: this corpus contains both
    # `palaver/cli/\nfixture_lint.py` and `session_meta.payload.{id,\n
    # session_id}`. Splitting per line puts the opening and closing backticks
    # in different splits, which inverts odd/even from that point on and hands
    # the *prose* to `check_span` while the real span goes unchecked — a
    # false rejection and a silent miss from the same bug.
    narrative: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            narrative.append("")
            continue
        if in_fence:
            checked += 1
            narrative.append("")
            verdict = check_span(line, f"fenced line {number}")
            if not verdict.ok:
                rejections.append(
                    Rejection(path=path, line=number, rule=verdict.rule, detail=verdict.detail)
                )
            continue
        narrative.append(line)

    body = "\n".join(narrative)
    # Inline spans are the odd-index segments of a backtick split. Matching
    # the pattern `` `[^`]*` `` instead would also match the gap *between* two
    # adjacent spans and report the prose in it as quoted content.
    offset = 0
    for index, segment in enumerate(body.split("`")):
        if index % 2 == 1:
            checked += 1
            line_number = body.count("\n", 0, offset) + 1
            verdict = check_span(segment, f"code span on line {line_number}")
            if not verdict.ok:
                rejections.append(
                    Rejection(path=path, line=line_number, rule=verdict.rule, detail=verdict.detail)
                )
        offset += len(segment) + 1

    for number, line in enumerate(body.splitlines(), start=1):
        for reason in provenance_markers(line):
            rejections.append(
                Rejection(
                    path=path,
                    line=number,
                    rule=RULE_PROVENANCE_MARKER,
                    detail=f"narrative carries {reason}",
                )
            )
    return checked, rejections
