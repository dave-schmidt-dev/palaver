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

**Three sources, one allowlist discipline.** `RECORD_SHAPES` (Claude Code),
`CODEX_RECORD_SHAPES`, and `OPENCODE_RECORD_SHAPES` are three separate tables,
one per source, because task 7.0 asked for per-source shape tables rather than
one table folding all three vocabularies together — the point being that a
reviewer can delete one source's table and watch only that source's corpus
fail (`tests/test_fixture_lint.py`'s
`test_codex_source_and_opencode_source_corpora_require_their_shape_tables`
does exactly that). `classify_record` dispatches by trying each table in turn
against the record's `type`; this is safe only because the three sources' top
level `type` vocabularies are disjoint today
(`test_source_shape_tables_are_pairwise_disjoint` pins that as an assertion,
not an assumption). Neither Codex nor OpenCode has an adapter yet (tasks 7.1
and 7.2), so unlike `SYSTEM_SUBTYPE_KINDS` there is nothing to import the
allowlisted sub-values from; `CODEX_EVENT_TYPES` and `OPENCODE_PART_TYPES`
below are authored from `docs/research.md` and are expected to become the
values those adapters import back out of this module, mirroring the existing
direction in reverse.

OpenCode has no natural JSONL representation — its real store is SQLite rows
in `message`/`part` with a JSON `data` column, not a line-delimited transcript
file. `opencode_message` / `opencode_part` (the `type` values in
`OPENCODE_RECORD_SHAPES`) are this module's own fixture-format invention: one
JSON object per row, wrapping the columns a future adapter reads. A binary
`.db` fixture would not be reviewable by a human before it reaches a public
remote, which is the exact thing INV-9's git clause exists to make possible.

Output follows the CLI's two-stream contract: the result (the report, with one
line per rejection) goes to stdout, per-file progress goes to stderr (INV-1).
"""

from __future__ import annotations

import json  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Callable, TextIO  # noqa: F401

from palaver.ingest.adapters.claude_code import SYSTEM_SUBTYPE_KINDS  # noqa: F401

from .annotation import ANNOTATION_CHANNELS as ANNOTATION_CHANNELS
from .annotation import ANNOTATION_FILE as ANNOTATION_FILE
from .annotation import ANNOTATION_KEYS as ANNOTATION_KEYS
from .annotation import ANNOTATION_ROLES as ANNOTATION_ROLES
from .annotation import ANNOTATION_TEXT as ANNOTATION_TEXT
from .annotation import _shape_codex_role_label as _shape_codex_role_label
from .annotation import classify_record as classify_record
from .checks import ACCEPTED as ACCEPTED
from .checks import INPUT_KEY as INPUT_KEY
from .checks import MAX_INPUT_DEPTH as MAX_INPUT_DEPTH
from .checks import Verdict as Verdict
from .checks import _check_bool as _check_bool
from .checks import _check_input_value as _check_input_value
from .checks import _check_keys as _check_keys
from .checks import _check_literal as _check_literal
from .checks import _check_pattern as _check_pattern
from .checks import _check_text as _check_text
from .checks import _reject as _reject
from .claude_code import MODE_VALUES as MODE_VALUES
from .claude_code import RECORD_SHAPES as RECORD_SHAPES
from .claude_code import TOOL_NAMES as TOOL_NAMES
from .claude_code import _check_assistant_block as _check_assistant_block
from .claude_code import _check_message as _check_message
from .claude_code import _check_user_block as _check_user_block
from .claude_code import _classify_ai_title as _classify_ai_title
from .claude_code import _classify_assistant as _classify_assistant
from .claude_code import _classify_mode as _classify_mode
from .claude_code import _classify_system as _classify_system
from .claude_code import _classify_user as _classify_user
from .cli import _stderr_status as _stderr_status
from .cli import add_arguments as add_arguments
from .cli import render_report as render_report
from .cli import run as run
from .codex import CODEX_CONTENT_BLOCK_TYPES as CODEX_CONTENT_BLOCK_TYPES
from .codex import CODEX_CWD as CODEX_CWD
from .codex import CODEX_ERROR_CODES as CODEX_ERROR_CODES
from .codex import CODEX_EVENT_TYPES as CODEX_EVENT_TYPES
from .codex import CODEX_RECORD_SHAPES as CODEX_RECORD_SHAPES
from .codex import CODEX_ROLES as CODEX_ROLES
from .codex import CODEX_TURN_ABORTED_REASONS as CODEX_TURN_ABORTED_REASONS
from .codex import _check_codex_content_block as _check_codex_content_block
from .codex import _check_codex_message as _check_codex_message
from .codex import _classify_codex_compacted as _classify_codex_compacted
from .codex import _classify_codex_event_msg as _classify_codex_event_msg
from .codex import _classify_codex_response_item as _classify_codex_response_item
from .codex import _classify_codex_session_meta as _classify_codex_session_meta
from .constants import HELP as HELP
from .constants import NAME as NAME
from .constants import RULE_BAD_GOLDEN_LABEL as RULE_BAD_GOLDEN_LABEL
from .constants import RULE_BAD_IDENTIFIER as RULE_BAD_IDENTIFIER
from .constants import RULE_BAD_VALUE as RULE_BAD_VALUE
from .constants import RULE_MISSING_KEY as RULE_MISSING_KEY
from .constants import RULE_NAMES as RULE_NAMES
from .constants import RULE_NOT_AN_OBJECT as RULE_NOT_AN_OBJECT
from .constants import RULE_PROVENANCE_MARKER as RULE_PROVENANCE_MARKER
from .constants import RULE_SOURCE_PROVENANCE as RULE_SOURCE_PROVENANCE
from .constants import RULE_UNALLOWLISTED_SPAN as RULE_UNALLOWLISTED_SPAN
from .constants import RULE_UNALLOWLISTED_TEXT as RULE_UNALLOWLISTED_TEXT
from .constants import RULE_UNDECODABLE as RULE_UNDECODABLE
from .constants import RULE_UNEXPECTED_KEY as RULE_UNEXPECTED_KEY
from .constants import RULE_UNKNOWN_CONTENT_BLOCK as RULE_UNKNOWN_CONTENT_BLOCK
from .constants import RULE_UNKNOWN_FILE_TYPE as RULE_UNKNOWN_FILE_TYPE
from .constants import RULE_UNKNOWN_RECORD_TYPE as RULE_UNKNOWN_RECORD_TYPE
from .constants import RULE_UNKNOWN_SUBTYPE as RULE_UNKNOWN_SUBTYPE
from .constants import RULE_UNKNOWN_SYSTEM_SUBTYPE as RULE_UNKNOWN_SYSTEM_SUBTYPE
from .constants import RULE_UNKNOWN_TOOL as RULE_UNKNOWN_TOOL
from .constants import RULE_UNTERMINATED_FILE as RULE_UNTERMINATED_FILE
from .constants import SESSION_ID as SESSION_ID
from .constants import SYNTHESIZED_TEXT as SYNTHESIZED_TEXT
from .constants import TOOL_USE_ID as TOOL_USE_ID
from .opencode import OPENCODE_FINISH_VALUES as OPENCODE_FINISH_VALUES
from .opencode import OPENCODE_PART_TYPES as OPENCODE_PART_TYPES
from .opencode import OPENCODE_RECORD_SHAPES as OPENCODE_RECORD_SHAPES
from .opencode import OPENCODE_TOOL_NAMES as OPENCODE_TOOL_NAMES
from .opencode import OPENCODE_TOOL_STATUSES as OPENCODE_TOOL_STATUSES
from .opencode import _check_opencode_tool_state as _check_opencode_tool_state
from .opencode import _classify_opencode_message as _classify_opencode_message
from .opencode import _classify_opencode_part as _classify_opencode_part
from .spans import COMMAND_SPAN as COMMAND_SPAN
from .spans import GOLDEN_LINE as GOLDEN_LINE
from .spans import PROVENANCE_MARKERS as PROVENANCE_MARKERS
from .spans import STRUCTURAL_CHARACTERS as STRUCTURAL_CHARACTERS
from .spans import STRUCTURAL_MAX_WORDS as STRUCTURAL_MAX_WORDS
from .spans import check_span as check_span
from .spans import provenance_markers as provenance_markers
from .surfaces import DATA_SUFFIXES as DATA_SUFFIXES
from .surfaces import DOCUMENTATION_TEXT as DOCUMENTATION_TEXT
from .surfaces import EXTRACTION_TEXT as EXTRACTION_TEXT
from .surfaces import GOLDEN_SUFFIXES as GOLDEN_SUFFIXES
from .surfaces import IGNORED_NAMES as IGNORED_NAMES
from .surfaces import KNOWN_SUFFIXES as KNOWN_SUFFIXES
from .surfaces import NARRATIVE_SUFFIXES as NARRATIVE_SUFFIXES
from .surfaces import RECORD_SUFFIXES as RECORD_SUFFIXES
from .surfaces import LintReport as LintReport
from .surfaces import Rejection as Rejection
from .surfaces import _line_of as _line_of
from .surfaces import _walk_strings as _walk_strings
from .surfaces import lint_data_file as lint_data_file
from .surfaces import lint_golden_file as lint_golden_file
from .surfaces import lint_narrative_file as lint_narrative_file
from .tree import lint_file as lint_file
from .tree import lint_provenance as lint_provenance
from .tree import lint_tree as lint_tree

__all__ = [
    "ACCEPTED",
    "ANNOTATION_CHANNELS",
    "ANNOTATION_FILE",
    "ANNOTATION_KEYS",
    "ANNOTATION_ROLES",
    "ANNOTATION_TEXT",
    "CODEX_CONTENT_BLOCK_TYPES",
    "CODEX_CWD",
    "CODEX_ERROR_CODES",
    "CODEX_EVENT_TYPES",
    "CODEX_RECORD_SHAPES",
    "CODEX_ROLES",
    "CODEX_TURN_ABORTED_REASONS",
    "COMMAND_SPAN",
    "DATA_SUFFIXES",
    "DOCUMENTATION_TEXT",
    "EXTRACTION_TEXT",
    "GOLDEN_LINE",
    "GOLDEN_SUFFIXES",
    "HELP",
    "IGNORED_NAMES",
    "INPUT_KEY",
    "KNOWN_SUFFIXES",
    "MAX_INPUT_DEPTH",
    "MODE_VALUES",
    "NAME",
    "NARRATIVE_SUFFIXES",
    "OPENCODE_FINISH_VALUES",
    "OPENCODE_PART_TYPES",
    "OPENCODE_RECORD_SHAPES",
    "OPENCODE_TOOL_NAMES",
    "OPENCODE_TOOL_STATUSES",
    "PROVENANCE_MARKERS",
    "RECORD_SHAPES",
    "RECORD_SUFFIXES",
    "RULE_BAD_GOLDEN_LABEL",
    "RULE_BAD_IDENTIFIER",
    "RULE_BAD_VALUE",
    "RULE_MISSING_KEY",
    "RULE_NAMES",
    "RULE_NOT_AN_OBJECT",
    "RULE_PROVENANCE_MARKER",
    "RULE_SOURCE_PROVENANCE",
    "RULE_UNALLOWLISTED_SPAN",
    "RULE_UNALLOWLISTED_TEXT",
    "RULE_UNDECODABLE",
    "RULE_UNEXPECTED_KEY",
    "RULE_UNKNOWN_CONTENT_BLOCK",
    "RULE_UNKNOWN_FILE_TYPE",
    "RULE_UNKNOWN_RECORD_TYPE",
    "RULE_UNKNOWN_SUBTYPE",
    "RULE_UNKNOWN_SYSTEM_SUBTYPE",
    "RULE_UNKNOWN_TOOL",
    "RULE_UNTERMINATED_FILE",
    "SESSION_ID",
    "STRUCTURAL_CHARACTERS",
    "STRUCTURAL_MAX_WORDS",
    "SYNTHESIZED_TEXT",
    "TOOL_NAMES",
    "TOOL_USE_ID",
    "LintReport",
    "Rejection",
    "Verdict",
    "_check_assistant_block",
    "_check_bool",
    "_check_codex_content_block",
    "_check_codex_message",
    "_check_input_value",
    "_check_keys",
    "_check_literal",
    "_check_message",
    "_check_opencode_tool_state",
    "_check_pattern",
    "_check_text",
    "_check_user_block",
    "_classify_ai_title",
    "_classify_assistant",
    "_classify_codex_compacted",
    "_classify_codex_event_msg",
    "_classify_codex_response_item",
    "_classify_codex_session_meta",
    "_classify_mode",
    "_classify_opencode_message",
    "_classify_opencode_part",
    "_classify_system",
    "_classify_user",
    "_line_of",
    "_reject",
    "_shape_codex_role_label",
    "_stderr_status",
    "_walk_strings",
    "add_arguments",
    "check_span",
    "classify_record",
    "lint_data_file",
    "lint_file",
    "lint_golden_file",
    "lint_narrative_file",
    "lint_provenance",
    "lint_tree",
    "provenance_markers",
    "render_report",
    "run",
]
