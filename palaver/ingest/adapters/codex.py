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

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from palaver.ingest.adapters.base import Adapter, Event, TailResult, read_complete_records

# Imported rather than redefined: `CHANNEL_HUMAN`/`CHANNEL_INJECTED` are
# INV-8's vocabulary, not Claude Code's private spelling, and
# `palaver.extract.normalize` and `palaver.extract.quote_gate` already read
# them from there. A second definition here would let the two sources drift
# into disagreeing about what "human" means.
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED
from palaver.ingest.cursors import Cursor
from palaver.memory.tiers import TIER_OBSERVER_INFERENCE, tier_name

logger = logging.getLogger(__name__)

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
_REPO_ROOT = Path(__file__).resolve().parents[3]

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


class CodexTierCapError(RuntimeError):
    """Raised when a caller explicitly requests a tier the Codex cap forbids.

    `cap_codex_tier` demotes silently, which is right for a tier derived by
    the extraction pipeline. An *explicit* request for a tier above the cap
    is a different thing: silently demoting it would hide a caller that
    believes it is minting tier-1 provenance out of a heuristic channel
    classification. That caller should fail loudly instead.
    """


@dataclass(frozen=True)
class RoleClassMeasurement:
    """The committed counts from measuring `codex_role_class` against labels.

    Attributes:
        n_records: Labelled records in the classifier's measured domain.
        n_errors: Harness-labelled records the classifier called human. This
            is the error that matters: it is the one that mints a false
            tier-1, and it is the direction INV-8 exists to prevent.
        threshold_met: The measurement's own verdict, recomputed by
            `measure_role_class` rather than hand-written.
    """

    n_records: int
    n_errors: int
    threshold_met: bool

    @property
    def lifts_tier_cap(self) -> bool:
        """Whether this measurement is strong enough to lift the tier-4 cap.

        All three conditions, conjunctively. A measurement file that merely
        exists lifts nothing, and neither does one that reports a clean zero
        over too small a sample.
        """
        return (
            self.n_records >= REQUIRED_LABELLED_RECORDS
            and self.n_errors == 0
            and self.threshold_met
        )


@dataclass(frozen=True)
class CodexIdentity:
    """Identity read out of a rollout's `session_meta` record.

    Attributes:
        cwd: The session's working directory, the basis for project scoping.
        id: This thread's own stable id; matches the filename's UUID.
        session_id: The root session's id. Equal to `id` except for a
            subagent thread-spawn.
        parent_thread_id: The parent thread this file was spawned from, or
            `None` for a root session.
    """

    cwd: str | None
    id: str | None
    session_id: str | None
    parent_thread_id: str | None

    @property
    def is_subagent(self) -> bool:
        """Whether this rollout is a subagent thread rather than a root session."""
        if self.parent_thread_id is not None:
            return True
        return self.id is not None and self.session_id is not None and self.id != self.session_id


def _parse_record(raw: bytes, path: Path) -> dict | None:
    """Decode one complete JSONL line, logging and skipping on failure.

    Args:
        raw: A complete, newline-stripped line from `read_complete_records`.
        path: Source path, for the warning message only. The record's own
            bytes are truncated in the log and never the whole line (INV-9).

    Returns:
        The decoded record, or `None` if `raw` was not a parseable JSON
        object. A corrupt record is logged at WARNING rather than silently
        dropped: silent data loss is invisible, and the tail must not crash
        on one bad line either.
    """
    try:
        record = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        logger.warning("Unparseable Codex rollout record in %s", path)
        return None
    if not isinstance(record, dict):
        logger.warning("Non-object Codex rollout record in %s", path)
        return None
    return record


def _payload(record: dict) -> dict:
    """Return a record's `payload` object, or an empty dict if it has none."""
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _message_payload(record: dict) -> dict | None:
    """Return the `message` payload of a `response_item`, or `None`.

    A `response_item` can carry `function_call`, `reasoning`, and other
    payload types that are not messages and have no role to classify.
    """
    if record.get("type") != "response_item":
        return None
    payload = _payload(record)
    if payload.get("type") != "message":
        return None
    return payload


def message_role(record: dict) -> str | None:
    """Return the role of a `response_item` message record, or `None`.

    `None` means "this record carries no role at all" — a `session_meta`, an
    `event_msg`, a `compacted` envelope, or a non-message `response_item`.
    That is a different answer from an unrecognized role string, and callers
    depend on the distinction: a role-less record is never the human channel,
    but it is also outside the domain `codex_role_class` is *measured* over.

    Args:
        record: A decoded rollout record.

    Returns:
        The role string, or `None` if the record carries no message role.
    """
    payload = _message_payload(record)
    if payload is None:
        return None
    role = payload.get("role")
    return role if isinstance(role, str) else None


def message_text(record: dict) -> str:
    """Flatten a message record's content blocks to plain text.

    Args:
        record: A decoded rollout record.

    Returns:
        The concatenated text of every recognized text block, or `""` for a
        record that is not a message or carries no text.
    """
    payload = _message_payload(record)
    if payload is None:
        return ""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") in TEXT_BLOCK_TYPES
    )


def codex_role_class(record: dict) -> str:
    """Classify a Codex record as the human or the harness channel (INV-8).

    A documented heuristic, not a flag — Codex has no `isMeta` equivalent.
    The rules, in order:

    1. A record with no message role at all is harness. This is structural,
       not a guess: a `session_meta`, an `event_msg`, or a `compacted`
       envelope is something the harness wrote about the session, and no
       amount of text in it was typed by a person.
    2. `role == "developer"` is harness, always (`HARNESS_ROLES`).
    3. Any role other than `user` is harness — `assistant` output is the
       model's, not the human's.
    4. A `user` record whose text begins with a known injected prefix is
       harness (`INJECTED_TEXT_PREFIXES`). Leading whitespace is stripped
       before the comparison, so a harness block preceded by a newline
       cannot launder itself into the human channel.
    5. Everything else is the human channel.

    Rules 1 through 3 are structural and cannot be wrong in the direction
    that matters. Rule 4 is the heuristic, and it is why every Codex-sourced
    decision is capped at tier-4 (`cap_codex_tier`): the table cannot see an
    injected shape whose prefix is not yet known, so a `user` record
    classified human is a belief about an absent signal.

    Args:
        record: A decoded rollout record.

    Returns:
        `CHANNEL_HUMAN` or `CHANNEL_INJECTED`.
    """
    role = message_role(record)
    if role is None:
        return CHANNEL_INJECTED
    if role in HARNESS_ROLES:
        return CHANNEL_INJECTED
    if role != HUMAN_CANDIDATE_ROLE:
        return CHANNEL_INJECTED
    text = message_text(record).lstrip()
    if text.startswith(INJECTED_TEXT_PREFIXES):
        return CHANNEL_INJECTED
    return CHANNEL_HUMAN


def record_ordinal(record: dict, line_index: int) -> int:
    """Return a record's cursor position: its `ordinal`, or its line index.

    Early rollout files add an `ordinal` to the envelope and later ones do
    not, so neither signal is available everywhere and the fallback is not
    an error path.

    Args:
        record: A decoded rollout record.
        line_index: The record's position in the batch it was read in.

    Returns:
        `record["ordinal"]` when it is a genuine integer, `line_index`
        otherwise. A boolean is rejected explicitly — `isinstance(True, int)`
        is true in Python, and `True` is not an ordinal.
    """
    ordinal = record.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        return line_index
    return ordinal


def order_records(records: Sequence[dict]) -> list[dict]:
    """Return `records` in cursor order: by `ordinal`, or by file order.

    The source's own numbering is believed over arrival order, but only when
    the whole batch carries it. A batch where some records have an `ordinal`
    and some do not has no single coordinate system to sort in — mixing real
    ordinals with fallback line indices would interleave them arbitrarily —
    so that batch keeps file order untouched.

    The sort is stable, so records sharing an ordinal keep their file order
    relative to each other.

    Args:
        records: Decoded rollout records, in file order.

    Returns:
        A new list in cursor order.
    """
    if not all(
        isinstance(record.get("ordinal"), int) and not isinstance(record.get("ordinal"), bool)
        for record in records
    ):
        return list(records)
    return sorted(records, key=lambda record: record_ordinal(record, 0))


# --- Tier cap ---------------------------------------------------------------


def load_measurement(path: Path | None = None) -> RoleClassMeasurement | None:
    """Read the committed measurement record, or `None` if it is unusable.

    Every failure mode returns `None`, which holds the cap: an absent file, a
    file that is not JSON, a file that is not an object, a missing key, and a
    key of the wrong type. Fail-closed is the only safe direction here,
    because the thing a malformed measurement file would otherwise unlock is
    tier-1 provenance, which under INV-4 cannot be retracted once written.

    Args:
        path: Measurement file to read. Defaults to `MEASUREMENT_PATH`; tests
            pass a `tmp_path` file to exercise both sides of the cap.

    Returns:
        The parsed measurement, or `None`.
    """
    path = MEASUREMENT_PATH if path is None else Path(path)
    if not path.is_file():
        return None
    # `Path.read_text`, not `open_source_readonly`: this is Palaver's own
    # committed artifact, not an observed agent's session store. INV-2's
    # chokepoint governs the stores this adapter observes, and routing a
    # repo file through it would misrepresent what that chokepoint is for.
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Codex role-class measurement at %s could not be read; cap holds", path)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        logger.warning("Codex role-class measurement at %s is not valid JSON; cap holds", path)
        return None
    if not isinstance(data, dict):
        logger.warning("Codex role-class measurement at %s is not an object; cap holds", path)
        return None

    n_records = data.get("n_records")
    n_errors = data.get("n_errors")
    threshold_met = data.get("threshold_met")
    if isinstance(n_records, bool) or not isinstance(n_records, int):
        logger.warning("Codex measurement %s has no integer n_records; cap holds", path)
        return None
    if isinstance(n_errors, bool) or not isinstance(n_errors, int):
        logger.warning("Codex measurement %s has no integer n_errors; cap holds", path)
        return None
    if not isinstance(threshold_met, bool):
        logger.warning("Codex measurement %s has no boolean threshold_met; cap holds", path)
        return None
    return RoleClassMeasurement(n_records=n_records, n_errors=n_errors, threshold_met=threshold_met)


def codex_tier_cap_lifted(*, measurement_path: Path | None = None) -> bool:
    """Return whether the Codex tier-4 cap can be lifted.

    Args:
        measurement_path: Measurement file to consult. Defaults to
            `MEASUREMENT_PATH`.

    Returns:
        Always false. Codex exposes no structural equivalent of Claude Code's
        ``isMeta`` marker, so its prefix classifier is never strong enough to
        mint tier-1 through tier-3 durable claims. Measurements remain useful
        diagnostics but cannot weaken this fail-closed boundary.
    """
    del measurement_path
    return False


def cap_codex_tier(tier: int, *, measurement_path: Path | None = None) -> int:
    """Demote `tier` to the Codex cap when the cap is in force.

    Args:
        tier: The provenance tier a decision would otherwise be written at.
        measurement_path: Measurement file to consult. Defaults to
            `MEASUREMENT_PATH`.

    Returns:
        `tier` unchanged when it is already at or below the cap's confidence.
        Otherwise `TIER_OBSERVER_INFERENCE` (4). Tiers are numbered
        highest-confidence-first, so the demotion is a `max`, and a tier-5
        speculation is never *promoted* to 4.
    """
    if codex_tier_cap_lifted(measurement_path=measurement_path):
        return tier
    return max(tier, TIER_OBSERVER_INFERENCE)


def require_codex_tier(tier: int, *, measurement_path: Path | None = None) -> int:
    """Return `tier` for a Codex source, or raise if the cap forbids it.

    Args:
        tier: The provenance tier the caller is explicitly asking for.
        measurement_path: Measurement file to consult. Defaults to
            `MEASUREMENT_PATH`.

    Returns:
        `tier`, unchanged, when the cap permits it.

    Raises:
        CodexTierCapError: `tier` is above the cap. The message names the
            requested tier, the cap, and the measurement thresholds, and
            carries no transcript content (INV-9).
    """
    capped = cap_codex_tier(tier, measurement_path=measurement_path)
    if capped != tier:
        raise CodexTierCapError(
            f"Codex-sourced decisions are capped at tier {capped} "
            f"({tier_name(capped)}); tier {tier} ({tier_name(tier)}) was requested. "
            "The cap is permanent because Codex has no structural human-content marker."
        )
    return tier


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


# --- Adapter ----------------------------------------------------------------


def _event_kind(record: dict) -> str:
    """Map one record to its canonical event kind.

    Unrecognized record types fall through to the type's own name rather than
    being dropped, so a future Codex release's records are still ingested and
    still visible as evidence, just not specially interpreted.
    """
    rtype = record.get("type")

    if rtype == "session_meta":
        return KIND_SESSION_META
    if rtype == COMPACTED_RECORD_TYPE:
        return KIND_COMPACTION
    if rtype == "response_item":
        payload_type = _payload(record).get("type")
        if payload_type == "message":
            return KIND_MESSAGE
        return str(payload_type) if isinstance(payload_type, str) and payload_type else "unknown"
    if rtype == "event_msg":
        return _event_msg_kind(_payload(record))

    return str(rtype) if isinstance(rtype, str) and rtype else "unknown"


def _event_msg_kind(payload: dict) -> str:
    """Map an `event_msg` payload to its kind, checking all three error layers."""
    event_type = payload.get("type")

    if event_type in TURN_BOUNDARY_EVENT_TYPES:
        return KIND_TURN_BOUNDARY
    if event_type == COMPACTION_EVENT_TYPE:
        return KIND_COMPACTION
    if event_type == ERROR_EVENT_TYPE:
        return KIND_ERROR
    if event_type == EXEC_END_EVENT_TYPE and _exec_failed(payload):
        return KIND_ERROR
    if event_type == PATCH_END_EVENT_TYPE and payload.get("success") is False:
        return KIND_ERROR

    return str(event_type) if isinstance(event_type, str) and event_type else "unknown"


def _exec_failed(payload: dict) -> bool:
    """Whether an `exec_command_end` payload reports a failure.

    Both layers research §2 names are checked, independently: a non-zero
    `exit_code` and a `status` of `"failed"`. Requiring both would miss a
    release that stopped emitting one of them, and either alone is
    sufficient evidence of a failed command.
    """
    if payload.get("status") == "failed":
        return True
    exit_code = payload.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return False
    return exit_code != 0


def _tool_call_id(record: dict) -> str | None:
    """Return the tool-call correlation id a record carries, if any."""
    payload = _payload(record)
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


class CodexAdapter(Adapter):
    """Adapter over Codex's `~/.codex/sessions/**/rollout-*.jsonl` stores."""

    source = "codex"

    def __init__(self, root: Path | None = None) -> None:
        """Create the adapter.

        Args:
            root: Directory holding Codex's date-partitioned rollout files.
                Defaults to `~/.codex/sessions`; tests must always pass a
                `tmp_path` root instead (INV-2/INV-3) — this default is
                production-only and is never exercised by this module's own
                test suite.
        """
        self.root = Path(root) if root is not None else Path.home() / ".codex" / "sessions"

    def list_store_paths(self) -> Iterable[Path]:
        """Enumerate rollout files without opening any of them.

        Recursive, unlike `ClaudeCodeAdapter.list_store_paths`: Codex
        partitions its sessions root by `YYYY/MM/DD`, so the files are three
        levels down rather than one, and the depth carries no identity — the
        filename does. The `rollout-*` prefix keeps any other `.jsonl` a
        future release drops in that tree from being mistaken for a session
        store.
        """
        if not self.root.exists():
            return []
        return sorted(self.root.rglob(STORE_GLOB))

    def session_key_for(self, path: Path) -> str:
        """Derive session identity from the file path alone.

        A rollout filename embeds the thread's UUID, which
        `session_meta.payload.id` repeats, so the stem is a stable identity
        that costs no file read — and `discover_sessions` calls this for
        every path it enumerates, including ones it will never open.
        `read_identity` is the richer, file-reading counterpart for callers
        that need `cwd` or the parent-thread linkage.
        """
        return path.stem

    def project_key_for(self, path: Path) -> str | None:
        """Return the working directory this session ran in, or `None`.

        Unlike Claude Code, where the project is encoded in the containing
        directory name, Codex's date-partitioned layout carries no project
        information at all — `cwd` lives inside `session_meta`, so this must
        open the file.
        """
        identity = self.read_identity(path)
        return None if identity is None else identity.cwd

    def read_identity(self, path: Path) -> CodexIdentity | None:
        """Read `session_meta` identity out of a rollout file.

        Args:
            path: The rollout file.

        Returns:
            The session's identity, or `None` if the file carries no
            `session_meta` record — which a truncated or not-yet-flushed
            file legitimately may.
        """
        for record in self._records(path):
            if record.get("type") != "session_meta":
                continue
            payload = _payload(record)
            return CodexIdentity(
                cwd=_optional_str(payload.get("cwd")),
                id=_optional_str(payload.get("id")),
                session_id=_optional_str(payload.get("session_id")),
                parent_thread_id=_optional_str(payload.get("parent_thread_id")),
            )
        return None

    def has_unresolved_trailing_tool_use(self, path: Path) -> bool:
        """Report whether the session ended with a tool call still outstanding.

        Codex's turn boundary is the inversion that makes this different from
        Claude Code's version. There, the last line is bookkeeping and the
        check has to read *past* it to find the last real turn. Here, the
        last line usually *is* the boundary (296/300 sampled files), and a
        boundary means the turn is over — so `task_complete` and
        `turn_aborted` clear every pending call, and a file ending on one of
        them reports False no matter how many calls it opened along the way.
        Reading a dangling `function_call` as unresolved after its turn had
        already closed would pin a finished session into
        `discover_sessions`'s always-include path forever.

        Args:
            path: The rollout file.

        Returns:
            True when at least one tool call was opened, never answered, and
            never closed out by a turn boundary.
        """
        pending: set[str] = set()
        unkeyed_calls = 0
        for record in self._records(path):
            kind = _event_kind(record)
            if kind == KIND_TURN_BOUNDARY:
                pending.clear()
                unkeyed_calls = 0
                continue
            if kind == "function_call":
                call_id = _tool_call_id(record)
                if call_id is None:
                    # A call with no correlation id cannot be matched to its
                    # output, so it is counted rather than tracked. Ignoring
                    # it would under-report; the turn boundary still clears it.
                    unkeyed_calls += 1
                else:
                    pending.add(call_id)
                continue
            if kind == "function_call_output":
                call_id = _tool_call_id(record)
                if call_id is None:
                    unkeyed_calls = max(0, unkeyed_calls - 1)
                else:
                    pending.discard(call_id)
        return bool(pending) or unkeyed_calls > 0

    def tail(self, path: Path, cursor: Cursor) -> TailResult:
        """Read every complete record appended after `cursor` into canonical events.

        Args:
            path: The rollout file.
            cursor: The durable byte-offset cursor from the last tail.

        Returns:
            A `TailResult` whose events are in cursor order (`order_records`)
            and whose payloads are the decoded source records byte-for-byte,
            so the evidence INV-6 anchors into is the record as written, not
            a reshaped view of it.
        """
        raw_records, new_offset = read_complete_records(path, cursor.offset)
        session_key = self.session_key_for(path)
        records = []
        malformed_records = 0
        for raw in raw_records:
            record = _parse_record(raw, path)
            if record is not None:
                records.append(record)
            else:
                malformed_records += 1
        events = tuple(
            Event(session_key=session_key, kind=_event_kind(record), payload=record)
            for record in order_records(records)
        )
        return TailResult(
            events=events,
            cursor=Cursor(offset=new_offset),
            malformed_records=malformed_records,
        )

    def _records(self, path: Path) -> list[dict]:
        """Decode every complete record in `path`, in cursor order."""
        raw_records, _ = read_complete_records(path, 0)
        records = []
        for raw in raw_records:
            record = _parse_record(raw, path)
            if record is not None:
                records.append(record)
        return order_records(records)


def _optional_str(value: object) -> str | None:
    """Return `value` if it is a non-empty string, else `None`."""
    return value if isinstance(value, str) and value else None
