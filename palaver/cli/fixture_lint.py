"""`palaver fixture-lint`: the allowlist gate on the committed fixture corpus.

This is INV-9's second half — the git half — and it is the last automated gate
before a transcript-shaped file enters a public repository's history. A fixture
pushed to a public remote leaves the machine as surely as an HTTP POST does,
and unlike a POST it cannot be recalled, so the failure mode this module exists
to prevent is *silent acceptance*, not noisy rejection.

**It is an allowlist, in two layers, and both live behind `classify_record`.**

1. **Shape.** A record ships only if its `type` (and, for `system`, its
   `subtype`) names an entry in the table below, every key it carries is one
   that entry declares, and every structural value is a literal that entry
   permits. Unknown type, unknown subtype, unknown key, unknown content block,
   unknown tool name: rejected. There is no fallthrough branch and no
   "probably fine" case.

2. **Free text.** Every string that reaches a free-text position must be an
   exact member of `SYNTHESIZED_TEXT`, the corpus's phrasebook. The linter
   cannot verify authorship, so it does not try; it verifies membership in a
   set that a human wrote deliberately. Adding a sentence to the corpus
   therefore requires editing this module, which is the point — the gate is
   the edit, and the edit is visible in a diff.

Why not a denylist cross-grep against the real stores. A grep answers "does
this fixture contain a string I already know about", which requires reading the
real stores to build the query, misses everything paraphrased or truncated, and
fails *open* on anything it has not seen. Its false negatives are silent. An
allowlist fails closed: an unclassified record is a failure, and the reason is
named.

What this buys concretely: a record copied out of a real Claude Code transcript
carries `uuid`, `parentUuid`, `timestamp`, `cwd`, `gitBranch`, and `version`
keys that no shape here declares, and a `sessionId` that is a UUID rather than
the required `fixture-*`. It is rejected on the first of those, before its
prose is ever considered.

`SYSTEM_SUBTYPE_KINDS` is imported from the Claude Code adapter rather than
restated, so the set of `system` subtypes the corpus may contain is exactly the
set the adapter claims to understand. A subtype the adapter has never heard of
is, by construction, one nobody has classified.

Output follows the CLI's two-stream contract: the result (the report, with one
line per rejection) goes to stdout, per-file progress goes to stderr (INV-1).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from palaver.ingest.adapters.claude_code import SYSTEM_SUBTYPE_KINDS

NAME = "fixture-lint"
HELP = "check every committed fixture record against the sanitization allowlist"

# Rejection rules. Named constants rather than free-form strings because the
# tests assert *which* rule fired: a poisoned record that fails for the wrong
# reason is a test that proves nothing, and a rule name in the report is the
# difference between "the linter rejected it" and "the linter rejected it for
# the reason under test".
RULE_UNDECODABLE = "undecodable-record"
RULE_NOT_AN_OBJECT = "not-an-object"
RULE_UNKNOWN_RECORD_TYPE = "unknown-record-type"
RULE_UNKNOWN_SYSTEM_SUBTYPE = "unknown-system-subtype"
RULE_MISSING_KEY = "missing-key"
RULE_UNEXPECTED_KEY = "unexpected-key"
RULE_UNKNOWN_CONTENT_BLOCK = "unknown-content-block"
RULE_UNKNOWN_TOOL = "unknown-tool"
RULE_BAD_IDENTIFIER = "bad-identifier"
RULE_BAD_VALUE = "bad-value"
RULE_UNALLOWLISTED_TEXT = "unallowlisted-text"
RULE_UNTERMINATED_FILE = "unterminated-file"

#: Every rule `classify_record` and `lint_tree` can report, for the report
#: legend and for the tests' "this rule exists" assertions.
RULE_NAMES: tuple[str, ...] = (
    RULE_UNDECODABLE,
    RULE_NOT_AN_OBJECT,
    RULE_UNKNOWN_RECORD_TYPE,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_MISSING_KEY,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_CONTENT_BLOCK,
    RULE_UNKNOWN_TOOL,
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNTERMINATED_FILE,
)

#: The corpus phrasebook: every free-text string any committed fixture may
#: contain, exactly. Written for the fixtures, about invented work, by a human
#: who was not looking at a real transcript while writing them. Membership is
#: checked by equality, not by pattern, because a pattern is a heuristic and a
#: heuristic that admits a real sentence fails silently.
#:
#: Adding an entry here is the deliberate act INV-9's git clause is about. It
#: should be rare, and it should be obvious in review that the new string is
#: invented.
SYNTHESIZED_TEXT = frozenset(
    {
        # Human-channel turns.
        "refactor the auth module",
        "run the test suite",
        "deploy status?",
        "widen the retry window",
        # Assistant replies.
        "the auth module is refactored",
        "the test suite is green",
        "the deploy finished",
        "the retry window is now thirty seconds",
        "should i also rename the helper?",
        # Tool results.
        "ok",
        "command not found",
        # `AskUserQuestion` input.
        "which database should the worker use?",
        "Database",
        "postgres",
        "sqlite",
        "the shared instance",
        "a file next to the worker",
        # Harness-written content.
        "earlier notes trimmed",
        "hook ran",
        "test suite run",
        "<command-name>/status</command-name>",
    }
)

#: A fixture's `sessionId`. Deliberately not "any UUID": a real Claude Code
#: session id *is* a UUID, so a pattern that admitted one would admit a record
#: pasted from a real store. Requiring a `fixture-` prefix makes provenance a
#: structural property of the value rather than a claim about it.
SESSION_ID = re.compile(r"^fixture-[a-z0-9-]{1,48}$")

#: A fixture's `tool_use_id` / `tool_use.id`. Same reasoning: real ids are
#: `toolu_…` opaque strings, and none of them match this.
TOOL_USE_ID = re.compile(r"^tu-[0-9]{1,3}$")

#: Keys permitted inside a `tool_use` block's `input` map. A plain identifier
#: carries no prose; anything else is free text wearing a key's clothing.
INPUT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")

#: Tool names a fixture may name. Small and explicit: a tool name is the one
#: piece of a `tool_use` block that could otherwise carry an unreviewed string.
TOOL_NAMES = frozenset({"Bash", "Read", "Edit", "AskUserQuestion"})

#: `mode` record values. Structural literals, so no phrasebook check applies.
MODE_VALUES = frozenset({"default", "plan", "acceptEdits"})

#: How deep a `tool_use` input map may nest before the linter stops walking.
#: `AskUserQuestion`'s real input reaches five levels (input → questions →
#: question → options → option → label), so this is that plus headroom, not an
#: arbitrary number.
MAX_INPUT_DEPTH = 8


@dataclass(frozen=True)
class Verdict:
    """The classifier's answer for one record.

    Attributes:
        ok: True when the record matched an allowlisted shape and every
            free-text payload it carried was in the phrasebook.
        rule: Which rule rejected it, from `RULE_NAMES`; empty when `ok`.
        detail: Human-readable specifics — which key, which value, which
            block. Never echoes more than a truncated prefix of an offending
            string, so a linter failure does not itself print a transcript.
    """

    ok: bool
    rule: str = ""
    detail: str = ""


#: The accepting verdict, as a singleton so a stub classifier in a test can
#: return exactly what the real one returns on success.
ACCEPTED = Verdict(ok=True)


def _reject(rule: str, detail: str) -> Verdict:
    return Verdict(ok=False, rule=rule, detail=detail)


def _check_keys(
    record: dict, required: frozenset[str], optional: frozenset[str], where: str
) -> Verdict | None:
    """Reject a mapping whose key set is not exactly what its shape declares."""
    keys = set(record)
    missing = required - keys
    if missing:
        return _reject(RULE_MISSING_KEY, f"{where} is missing {sorted(missing)}")
    unexpected = keys - required - optional
    if unexpected:
        return _reject(
            RULE_UNEXPECTED_KEY, f"{where} carries unexpected key(s) {sorted(unexpected)}"
        )
    return None


def _check_literal(value: object, allowed: frozenset[str], where: str) -> Verdict | None:
    """Reject a structural value that is not one of a fixed set of literals."""
    if not isinstance(value, str) or value not in allowed:
        return _reject(RULE_BAD_VALUE, f"{where} must be one of {sorted(allowed)}, got {value!r}")
    return None


def _check_pattern(value: object, pattern: re.Pattern[str], where: str) -> Verdict | None:
    """Reject an identifier that does not match its declared synthetic shape."""
    if not isinstance(value, str) or not pattern.match(value):
        return _reject(
            RULE_BAD_IDENTIFIER,
            f"{where} must match {pattern.pattern}, got {str(value)[:40]!r}",
        )
    return None


def _check_bool(value: object, where: str) -> Verdict | None:
    if not isinstance(value, bool):
        return _reject(RULE_BAD_VALUE, f"{where} must be a boolean, got {type(value).__name__}")
    return None


def _check_text(value: object, where: str) -> Verdict | None:
    """Reject any free-text payload that is not in the corpus phrasebook.

    This is the second allowlist layer. It is exact-match on purpose: a length
    bound, a character class, or a "looks synthetic" heuristic would each admit
    some real sentence, and every one of those admissions is silent.
    """
    if not isinstance(value, str):
        return _reject(RULE_BAD_VALUE, f"{where} must be a string, got {type(value).__name__}")
    if value not in SYNTHESIZED_TEXT:
        return _reject(
            RULE_UNALLOWLISTED_TEXT,
            f"{where} is not in the synthesized phrasebook: {value[:48]!r}",
        )
    return None


def _check_input_value(value: object, where: str, depth: int) -> Verdict | None:
    """Walk a `tool_use` input map, allowlisting every leaf it reaches.

    Booleans and nulls carry no prose and pass structurally. Strings go through
    the phrasebook. Numbers are *not* admitted: no fixture needs one, and every
    value type this function accepts is one somebody decided to accept.
    """
    if depth > MAX_INPUT_DEPTH:
        return _reject(RULE_BAD_VALUE, f"{where} nests deeper than {MAX_INPUT_DEPTH} levels")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return _check_text(value, where)
    if isinstance(value, list):
        for index, item in enumerate(value):
            verdict = _check_input_value(item, f"{where}[{index}]", depth + 1)
            if verdict is not None:
                return verdict
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not INPUT_KEY.match(key):
                return _reject(
                    RULE_BAD_IDENTIFIER,
                    f"{where} has a key that is not a plain identifier: {str(key)[:40]!r}",
                )
            verdict = _check_input_value(item, f"{where}.{key}", depth + 1)
            if verdict is not None:
                return verdict
        return None
    return _reject(RULE_BAD_VALUE, f"{where} has unsupported type {type(value).__name__}")


def _check_assistant_block(block: object, where: str) -> Verdict | None:
    """Allowlist one content block of an `assistant` record."""
    if not isinstance(block, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    block_type = block.get("type")
    if block_type == "text":
        verdict = _check_keys(block, frozenset({"type", "text"}), frozenset(), where)
        return verdict if verdict is not None else _check_text(block["text"], f"{where}.text")
    if block_type == "tool_use":
        verdict = _check_keys(block, frozenset({"type", "id", "name", "input"}), frozenset(), where)
        if verdict is not None:
            return verdict
        verdict = _check_pattern(block["id"], TOOL_USE_ID, f"{where}.id")
        if verdict is not None:
            return verdict
        if not isinstance(block["name"], str) or block["name"] not in TOOL_NAMES:
            return _reject(
                RULE_UNKNOWN_TOOL,
                f"{where}.name must be one of {sorted(TOOL_NAMES)}, got {block['name']!r}",
            )
        if not isinstance(block["input"], dict):
            return _reject(RULE_BAD_VALUE, f"{where}.input must be an object")
        return _check_input_value(block["input"], f"{where}.input", 0)
    return _reject(
        RULE_UNKNOWN_CONTENT_BLOCK,
        f"{where} has block type {block_type!r}, which no assistant shape declares",
    )


def _check_user_block(block: object, where: str) -> Verdict | None:
    """Allowlist one content block of a `user` record."""
    if not isinstance(block, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"{where} is not an object")
    block_type = block.get("type")
    if block_type == "text":
        verdict = _check_keys(block, frozenset({"type", "text"}), frozenset(), where)
        return verdict if verdict is not None else _check_text(block["text"], f"{where}.text")
    if block_type == "tool_result":
        verdict = _check_keys(
            block,
            frozenset({"type", "tool_use_id", "is_error", "content"}),
            frozenset(),
            where,
        )
        if verdict is not None:
            return verdict
        verdict = _check_pattern(block["tool_use_id"], TOOL_USE_ID, f"{where}.tool_use_id")
        if verdict is not None:
            return verdict
        verdict = _check_bool(block["is_error"], f"{where}.is_error")
        if verdict is not None:
            return verdict
        return _check_text(block["content"], f"{where}.content")
    return _reject(
        RULE_UNKNOWN_CONTENT_BLOCK,
        f"{where} has block type {block_type!r}, which no user shape declares",
    )


def _check_message(
    record: dict, role: str, block_checker: Callable[[object, str], Verdict | None]
) -> Verdict | None:
    """Allowlist a record's `message` envelope and every content block in it."""
    message = record.get("message")
    if not isinstance(message, dict):
        return _reject(RULE_NOT_AN_OBJECT, "message is not an object")
    verdict = _check_keys(message, frozenset({"role", "content"}), frozenset(), "message")
    if verdict is not None:
        return verdict
    verdict = _check_literal(message["role"], frozenset({role}), "message.role")
    if verdict is not None:
        return verdict
    content = message["content"]
    if not isinstance(content, list):
        return _reject(RULE_BAD_VALUE, "message.content must be a list of blocks")
    if not content:
        return _reject(RULE_BAD_VALUE, "message.content is empty")
    for index, block in enumerate(content):
        verdict = block_checker(block, f"message.content[{index}]")
        if verdict is not None:
            return verdict
    return None


def _classify_user(record: dict) -> Verdict:
    verdict = _check_keys(
        record,
        frozenset({"type", "sessionId", "isMeta", "message"}),
        frozenset(),
        "record",
    )
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_bool(record["isMeta"], "record.isMeta")
    if verdict is None:
        verdict = _check_message(record, "user", _check_user_block)
    return verdict if verdict is not None else ACCEPTED


def _classify_assistant(record: dict) -> Verdict:
    verdict = _check_keys(
        record, frozenset({"type", "sessionId", "message"}), frozenset(), "record"
    )
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_message(record, "assistant", _check_assistant_block)
    return verdict if verdict is not None else ACCEPTED


def _classify_system(record: dict) -> Verdict:
    verdict = _check_keys(
        record,
        frozenset({"type", "subtype", "sessionId"}),
        frozenset({"content", "summary"}),
        "record",
    )
    if verdict is not None:
        return verdict
    subtype = record["subtype"]
    if not isinstance(subtype, str) or subtype not in SYSTEM_SUBTYPE_KINDS:
        return _reject(
            RULE_UNKNOWN_SYSTEM_SUBTYPE,
            f"record.subtype {str(subtype)[:40]!r} is not a subtype the Claude Code "
            f"adapter classifies ({sorted(SYSTEM_SUBTYPE_KINDS)})",
        )
    verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is not None:
        return verdict
    for key in ("content", "summary"):
        if key in record:
            verdict = _check_text(record[key], f"record.{key}")
            if verdict is not None:
                return verdict
    return ACCEPTED


def _classify_mode(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "sessionId", "mode"}), frozenset(), "record")
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_literal(record["mode"], MODE_VALUES, "record.mode")
    return verdict if verdict is not None else ACCEPTED


def _classify_ai_title(record: dict) -> Verdict:
    verdict = _check_keys(record, frozenset({"type", "sessionId", "title"}), frozenset(), "record")
    if verdict is None:
        verdict = _check_pattern(record["sessionId"], SESSION_ID, "record.sessionId")
    if verdict is None:
        verdict = _check_text(record["title"], "record.title")
    return verdict if verdict is not None else ACCEPTED


#: The shape allowlist. A record type absent from this mapping is unclassified
#: and fails — including every bookkeeping type the adapter tolerates at
#: runtime (`attachment`, `last-prompt`, `bridge-session`, `summary`). The
#: adapter may safely *ignore* a record type it does not model; the corpus may
#: not safely *ship* one nobody has reviewed for content.
RECORD_SHAPES: dict[str, Callable[[dict], Verdict]] = {
    "user": _classify_user,
    "assistant": _classify_assistant,
    "system": _classify_system,
    "mode": _classify_mode,
    "ai-title": _classify_ai_title,
}


def classify_record(record: object) -> Verdict:
    """Classify one decoded fixture record against both allowlist layers.

    This is the single seam the whole gate rests on: shape *and* phrasebook are
    decided here, so stubbing it to accept everything must make the linter
    accept everything. A test that poisons a record and asserts a non-zero exit
    proves nothing unless stubbing this function flips that exit to zero.

    Args:
        record: A decoded JSONL record.

    Returns:
        `ACCEPTED`, or a `Verdict` naming the rule that rejected it.
    """
    if not isinstance(record, dict):
        return _reject(RULE_NOT_AN_OBJECT, f"record is a {type(record).__name__}, not an object")
    record_type = record.get("type")
    shape = RECORD_SHAPES.get(record_type) if isinstance(record_type, str) else None
    if shape is None:
        return _reject(
            RULE_UNKNOWN_RECORD_TYPE,
            f"record type {str(record_type)[:40]!r} is not in the shape allowlist "
            f"({sorted(RECORD_SHAPES)})",
        )
    return shape(record)


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
        files: How many `.jsonl` fixtures were read.
        records: How many records were classified.
        rejections: Every record the allowlist refused, in file order.
    """

    root: Path
    files: int
    records: int
    rejections: tuple[Rejection, ...]


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
    classify = classify_record if classify is None else classify
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
    """Classify every record of every `.jsonl` fixture under `root`.

    Args:
        root: Corpus directory (or a single `.jsonl` file).
        classify: Classifier override; defaults to `classify_record`.
        on_status: Progress channel, called once per file before it is read
            (INV-1). Never writes to stdout.

    Returns:
        A `LintReport` over everything read.
    """
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    records = 0
    rejections: list[Rejection] = []
    for index, path in enumerate(paths, start=1):
        if on_status is not None:
            on_status(f"linting {index}/{len(paths)}: {path.name}")
        counted, found = lint_file(path, classify=classify)
        records += counted
        rejections.extend(found)
    return LintReport(root=root, files=len(paths), records=records, rejections=tuple(rejections))


def render_report(report: LintReport) -> str:
    """Render a `LintReport` as the command's stdout output."""
    lines = [
        "palaver fixture-lint",
        f"corpus: {report.root}",
        f"files: {report.files}",
        f"records: {report.records}",
        f"rejected: {len(report.rejections)}",
    ]
    if report.rejections:
        lines.append("")
        for rejection in report.rejections:
            lines.append(f"{rejection.path}:{rejection.line}: {rejection.rule}: {rejection.detail}")
        lines.extend(
            [
                "",
                "A rejected record is not classified, and an unclassified record does",
                "not ship: fix the fixture, or add its shape to the allowlist in",
                "palaver/cli/fixture_lint.py deliberately.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "every record matched an allowlisted shape and carried only",
                "phrasebook text.",
            ]
        )
    return "\n".join(lines) + "\n"


def add_arguments(parser) -> None:
    """Register `fixture-lint`'s arguments on its subparser."""
    parser.add_argument(
        "path",
        type=Path,
        help="fixture corpus directory (or a single .jsonl fixture) to check",
    )


def _stderr_status(message: str) -> None:
    """Write one progress line to stderr, keeping stdout the result channel."""
    print(message, file=sys.stderr, flush=True)


def run(
    args,
    *,
    out: TextIO | None = None,
    on_status: Callable[[str], None] | None = None,
) -> int:
    """Run `palaver fixture-lint`.

    Args:
        args: Parsed arguments from this subcommand's parser.
        out: Result stream, defaulting to stdout.
        on_status: Progress channel, defaulting to a stderr writer (INV-1).

    Returns:
        0 when every record classified, 1 when any record was rejected, and 2
        for a usage failure — a missing path or a corpus with no fixtures in
        it. The two non-zero codes are distinct deliberately: a test that
        asserts "the linter rejected my poisoned record" must be able to fail
        when what actually happened was that the path was wrong.
    """
    out = sys.stdout if out is None else out
    on_status = _stderr_status if on_status is None else on_status

    root = Path(args.path)
    if not root.exists():
        print(f"palaver fixture-lint: no such path: {root}", file=sys.stderr)
        return 2

    report = lint_tree(root, on_status=on_status)
    if report.files == 0:
        print(f"palaver fixture-lint: no .jsonl fixtures under {root}", file=sys.stderr)
        return 2

    out.write(render_report(report))
    return 1 if report.rejections else 0
