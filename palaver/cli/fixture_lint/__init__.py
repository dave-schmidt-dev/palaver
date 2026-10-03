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

from .annotation import ANNOTATION_TEXT as ANNOTATION_TEXT
from .annotation import classify_record as classify_record
from .checks import ACCEPTED as ACCEPTED
from .claude_code import RECORD_SHAPES as RECORD_SHAPES
from .cli import add_arguments as add_arguments
from .cli import run as run
from .codex import CODEX_RECORD_SHAPES as CODEX_RECORD_SHAPES
from .constants import HELP as HELP
from .constants import NAME as NAME
from .constants import RULE_BAD_GOLDEN_LABEL as RULE_BAD_GOLDEN_LABEL
from .constants import RULE_BAD_IDENTIFIER as RULE_BAD_IDENTIFIER
from .constants import RULE_BAD_VALUE as RULE_BAD_VALUE
from .constants import RULE_NAMES as RULE_NAMES
from .constants import RULE_PROVENANCE_MARKER as RULE_PROVENANCE_MARKER
from .constants import RULE_SOURCE_PROVENANCE as RULE_SOURCE_PROVENANCE
from .constants import RULE_UNALLOWLISTED_SPAN as RULE_UNALLOWLISTED_SPAN
from .constants import RULE_UNALLOWLISTED_TEXT as RULE_UNALLOWLISTED_TEXT
from .constants import RULE_UNDECODABLE as RULE_UNDECODABLE
from .constants import RULE_UNEXPECTED_KEY as RULE_UNEXPECTED_KEY
from .constants import RULE_UNKNOWN_FILE_TYPE as RULE_UNKNOWN_FILE_TYPE
from .constants import RULE_UNKNOWN_RECORD_TYPE as RULE_UNKNOWN_RECORD_TYPE
from .constants import RULE_UNKNOWN_SUBTYPE as RULE_UNKNOWN_SUBTYPE
from .constants import RULE_UNKNOWN_SYSTEM_SUBTYPE as RULE_UNKNOWN_SYSTEM_SUBTYPE
from .constants import RULE_UNTERMINATED_FILE as RULE_UNTERMINATED_FILE
from .constants import SYNTHESIZED_TEXT as SYNTHESIZED_TEXT
from .opencode import OPENCODE_RECORD_SHAPES as OPENCODE_RECORD_SHAPES
from .spans import check_span as check_span
from .surfaces import DOCUMENTATION_TEXT as DOCUMENTATION_TEXT
from .surfaces import EXTRACTION_TEXT as EXTRACTION_TEXT
from .surfaces import IGNORED_NAMES as IGNORED_NAMES
from .tree import lint_provenance as lint_provenance
from .tree import lint_tree as lint_tree

__all__ = [
    "ACCEPTED",
    "ANNOTATION_TEXT",
    "CODEX_RECORD_SHAPES",
    "DOCUMENTATION_TEXT",
    "EXTRACTION_TEXT",
    "HELP",
    "IGNORED_NAMES",
    "NAME",
    "OPENCODE_RECORD_SHAPES",
    "RECORD_SHAPES",
    "RULE_BAD_GOLDEN_LABEL",
    "RULE_BAD_IDENTIFIER",
    "RULE_BAD_VALUE",
    "RULE_NAMES",
    "RULE_PROVENANCE_MARKER",
    "RULE_SOURCE_PROVENANCE",
    "RULE_UNALLOWLISTED_SPAN",
    "RULE_UNALLOWLISTED_TEXT",
    "RULE_UNDECODABLE",
    "RULE_UNEXPECTED_KEY",
    "RULE_UNKNOWN_FILE_TYPE",
    "RULE_UNKNOWN_RECORD_TYPE",
    "RULE_UNKNOWN_SUBTYPE",
    "RULE_UNKNOWN_SYSTEM_SUBTYPE",
    "RULE_UNTERMINATED_FILE",
    "SYNTHESIZED_TEXT",
    "add_arguments",
    "check_span",
    "classify_record",
    "lint_provenance",
    "lint_tree",
    "run",
]
