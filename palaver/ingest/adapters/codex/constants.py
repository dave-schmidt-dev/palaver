"""
Rollout-store constants, the event/role vocabularies, and the derived fixture paths
(measurement, labels, corpus).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("palaver.ingest.adapters.codex")

#: Rollout files are named `rollout-<timestamp>-<uuid>.jsonl` and live three
#: levels under the sessions root, partitioned `YYYY/MM/DD`.
STORE_GLOB = "rollout-*.jsonl"

#: `payload.type` values of an `event_msg` that closes a turn.
TURN_BOUNDARY_EVENT_TYPES = frozenset({"task_complete", "turn_aborted"})

#: The `event_msg` half of the compaction pair. The other half is the
#: top-level `compacted` envelope.
COMPACTION_EVENT_TYPE = "context_compacted"

#: Top-level record type carrying `payload.replacement_history`.
COMPACTED_RECORD_TYPE = "compacted"

#: `event_msg` types this adapter inspects for a failure outcome. An
#: `exec_command_end` or `patch_apply_end` that did *not* fail still produces
#: an event — under its own kind, not `error` — so nothing is dropped.
EXEC_END_EVENT_TYPE = "exec_command_end"
PATCH_END_EVENT_TYPE = "patch_apply_end"
ERROR_EVENT_TYPE = "error"

#: Canonical event kinds this adapter emits, beyond the passthrough kinds it
#: derives from a record's own type.
KIND_MESSAGE = "message"
KIND_TURN_BOUNDARY = "turn_boundary"
KIND_COMPACTION = "compaction"
KIND_ERROR = "error"
KIND_SESSION_META = "session_meta"

#: Content-block types that carry text on a Codex `message` payload. Matches
#: `palaver.cli.fixture_lint.CODEX_CONTENT_BLOCK_TYPES`, plus the bare
#: `"text"` a future release could use; a block type absent here contributes
#: no text, which is the fail-closed direction (unrecognized content cannot
#: dilute a prefix match at position zero).
TEXT_BLOCK_TYPES = frozenset({"input_text", "output_text", "text"})

# Codex serializes an attached image into the otherwise human-authored text
# stream. This shape is transport metadata, not a user request. Keep the
# match deliberately narrow: nearby angle-bracket text remains user content.
CODEX_IMAGE_ATTACHMENT_MARKER = re.compile(r'<image name=\[Image #\d+\] path="[^"\r\n]*">')

#: The only role whose content can ever be the human channel. Every other
#: role — and every record with no role at all — is harness by construction.
HUMAN_CANDIDATE_ROLE = "user"

#: Roles that are always harness. `developer` is named explicitly because
#: the task makes it a contract rather than an inference, even though the
#: `HUMAN_CANDIDATE_ROLE` test below would already exclude it: a rule that
#: holds only as a side effect of another rule is a rule that a later
#: refactor can delete without noticing.
HARNESS_ROLES = frozenset({"developer"})

#: INV-8's only available signal for a `role: "user"` record, since Codex has
#: no `isMeta` equivalent. Cited from `docs/research.md` §2, which measured
#: these six prefixes structurally across 4,884 rollout files. This table was
#: written from that source and not adjusted after the measurement ran.
#:
#: `<codex_internal_context` is deliberately matched without its closing
#: bracket: the real tag carries attributes (`source=...`), so anchoring on
#: `>` would miss every instance of it.
INJECTED_TEXT_PREFIXES = (
    "<environment_context>",
    "<recommended_plugins>",
    "<codex_internal_context",
    "<subagent_notification>",
    "# AGENTS.md instructions",
    "Automated daily window start.",
)

#: Labelled records required before the tier-4 cap may lift. The sample must
#: also be measured at zero harness-classified-as-user errors — see
#: `RoleClassMeasurement.lifts_tier_cap`, which requires the full
#: conjunction.
REQUIRED_LABELLED_RECORDS = 200

# `palaver/ingest/adapters/codex.py` -> `palaver/ingest/adapters` ->
# `palaver/ingest` -> `palaver` -> the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

#: The committed measurement record. It lives under `tests/fixtures/` rather
#: than `HISTORY.md` or `docs/` because both of those are gitignored in this
#: public repo, so neither could carry the blame data the two-commit ordering
#: proof reads (orchestrator amendments 1 and 2 to task 7.1). It holds counts
#: only — no prose, no transcript content — which keeps it INV-9-clean by
#: construction rather than by review.
MEASUREMENT_PATH = (
    _REPO_ROOT / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
)

#: The blind labels the measurement scores against, and the corpus they index.
LABELS_PATH = _REPO_ROOT / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"
CORPUS_ROOT = _REPO_ROOT
