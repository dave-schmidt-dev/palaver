"""Codex rollout JSONL adapter, fail-closed at tier-4 (task 7.1).

Reads `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` — one file per session
thread, nested three levels under a date-partitioned root, append-only while
the session is live. Every structural fact this module encodes was measured
in `docs/research.md` §2 against 4,884 real rollout files, structurally: no
free text was read out of them, and nothing in this module's docstrings comes
from an observed session (INV-9).

**Turn boundary — structural and strong.** An `event_msg` whose
`payload.type` is `task_complete` or `turn_aborted` closes a turn. This is
the opposite of Claude Code, where the last line of a file is bookkeeping and
the real turn ends earlier: for Codex the last line usually *is* the boundary
(296/300 sampled finished files). `has_unresolved_trailing_tool_use` is built
around that inversion — a boundary event clears every pending tool call,
because the turn it belonged to is over.

**Compaction — an always-paired two-record marker.** A top-level
`type: "compacted"` envelope carrying `payload.replacement_history`,
immediately followed by an `event_msg` with `payload.type ==
"context_compacted"`. Both map to the `compaction` event kind; neither is
treated as the sole signal, so a future release that emits only one half is
still seen.

**Errors are distinguishable at three layers**, and all three are mapped to
the `error` kind: `exec_command_end` with a non-zero `exit_code` or a
`status` of `"failed"`, `patch_apply_end` with `success` false, and an
`event_msg` of type `error` carrying `message` and `codex_error_info`.

**Identity.** `session_meta.payload.cwd` is the working directory,
`.id` is this thread's own stable id (it matches the filename's UUID),
`.session_id` is the root session — differing from `.id` only for subagent
thread-spawns — and `parent_thread_id` links a subagent file to its parent.
`session_key_for` stays path-derived, per the `Adapter` contract's "derive
identity from the store path" requirement; `read_identity` supplies the
richer identity for callers that need `cwd` or the parent linkage, and pays
a file read for it.

**Cursor.** The durable resume position stays the byte offset the base
contract defines (`Cursor.offset`), because that is what
`read_complete_records` can advance without ever passing a torn write.
Within a batch, records are ordered by their own `ordinal` field when every
record in the batch carries one, and by file order otherwise
(`order_records`) — early rollout files add `ordinal` to the envelope and
later ones do not, and a source that numbers its own records should be
believed over the order they happen to arrive in.

**Channel classification is a heuristic here, and that is why the tier cap
exists (INV-8).** Claude Code has `isMeta`, a structural flag the harness
sets. Codex has no equivalent anywhere: `role: "user"` response items mix
genuine human text with harness content, and the only available
discrimination is `role == "developer"` (reliably harness across 817 sampled
records) plus a fixed text-prefix table. A prefix table cannot see a
harness-injected shape whose prefix is not yet known, so a Codex record
classified as the human channel is a *belief*, not an observation — and
tier-1 means "the user said this", the tier every other tier defers to under
INV-5. **Every Codex-sourced decision is therefore capped at
`TIER_OBSERVER_INFERENCE` (4)** until `codex_role_class` has been measured
over a labelled sample of at least `REQUIRED_LABELLED_RECORDS` records at
zero harness-classified-as-user errors. `cap_codex_tier` demotes; the cap
lifts only on the full conjunction in `RoleClassMeasurement.lifts_tier_cap`,
never on the measurement file's mere presence.

The labels that measurement reads were authored blind, before this module
existed, and committed in their own commit — see
`tests/fixtures/labels/codex-role-labels.jsonl` and `measure_role_class`. The
prefix table below is cited from `docs/research.md` §2 and was not adjusted
after seeing which records disagreed; adjusting it post-hoc would validate
the heuristic against labels the heuristic produced, which is the
circularity the two-commit protocol exists to prevent.

Every session record read here goes through `read_complete_records` (in
turn, `open_source_readonly`) — this module never opens a session store
itself (INV-2).

No `on_status` channel (INV-1): every operation here is bounded local file
IO and pure string work, with no network call, subprocess, model inference,
or stall-prone wait to surface progress for. This matches `claude_code.py`,
the sibling adapter, rather than diverging from it.
"""

from __future__ import annotations

# Imported rather than redefined: `CHANNEL_HUMAN`/`CHANNEL_INJECTED` are
# INV-8's vocabulary, not Claude Code's private spelling, and
# `palaver.extract.normalize` and `palaver.extract.quote_gate` already read
# them from there. A second definition here would let the two sources drift
# into disagreeing about what "human" means.
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED  # noqa: F401

from .adapter import CodexAdapter as CodexAdapter
from .classify import codex_role_class as codex_role_class
from .constants import INJECTED_TEXT_PREFIXES as INJECTED_TEXT_PREFIXES
from .constants import KIND_COMPACTION as KIND_COMPACTION
from .constants import KIND_ERROR as KIND_ERROR
from .constants import KIND_MESSAGE as KIND_MESSAGE
from .constants import KIND_SESSION_META as KIND_SESSION_META
from .constants import KIND_TURN_BOUNDARY as KIND_TURN_BOUNDARY
from .constants import LABELS_PATH as LABELS_PATH
from .constants import MEASUREMENT_PATH as MEASUREMENT_PATH
from .constants import REQUIRED_LABELLED_RECORDS as REQUIRED_LABELLED_RECORDS
from .measurement import measure_role_class as measure_role_class
from .ordering import order_records as order_records
from .ordering import record_ordinal as record_ordinal
from .records import CodexTierCapError as CodexTierCapError
from .records import RoleClassMeasurement as RoleClassMeasurement
from .records import message_role as message_role
from .records import message_text as message_text
from .records import strip_codex_image_attachment_markers as strip_codex_image_attachment_markers
from .tier_cap import cap_codex_tier as cap_codex_tier
from .tier_cap import codex_tier_cap_lifted as codex_tier_cap_lifted
from .tier_cap import load_measurement as load_measurement
from .tier_cap import require_codex_tier as require_codex_tier

__all__ = [
    "INJECTED_TEXT_PREFIXES",
    "KIND_COMPACTION",
    "KIND_ERROR",
    "KIND_MESSAGE",
    "KIND_SESSION_META",
    "KIND_TURN_BOUNDARY",
    "LABELS_PATH",
    "MEASUREMENT_PATH",
    "REQUIRED_LABELLED_RECORDS",
    "CodexAdapter",
    "CodexTierCapError",
    "RoleClassMeasurement",
    "cap_codex_tier",
    "codex_role_class",
    "codex_tier_cap_lifted",
    "load_measurement",
    "measure_role_class",
    "message_role",
    "message_text",
    "order_records",
    "record_ordinal",
    "require_codex_tier",
    "strip_codex_image_attachment_markers",
]
