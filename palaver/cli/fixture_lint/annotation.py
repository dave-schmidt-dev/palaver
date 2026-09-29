"""
The codex_role_label blind-annotation shape and classify_record, the single dispatch
seam that tries the annotation shape then all three sources' RECORD_SHAPES tables.
"""

from __future__ import annotations

import re

from . import codex, opencode
from .checks import ACCEPTED, Verdict, _check_keys, _check_literal, _reject
from .claude_code import RECORD_SHAPES
from .constants import (
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_NOT_AN_OBJECT,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNKNOWN_RECORD_TYPE,
)

#: The annotation phrasebook, kept in a **separate namespace** from
#: `SYNTHESIZED_TEXT` on purpose: these strings describe fixtures, they are not
#: fixture content. Merging the two would let an annotation sentence satisfy a
#: transcript record's text check, which is precisely the confusion INV-9's
#: allowlist exists to prevent. Only `codex_role_label` records may carry them.
#:
#: **Why hardcoding prose here is safe (verified mechanically, 2026-08-14).**
#: These 14 strings were authored as commentary about record *structure*, not
#: copied from any transcript. A sliding 14-character window over all 14
#: strings (1,609 windows) was tested against every committed fixture under
#: `tests/fixtures/` (20 files). Zero are shorter than 14 characters, so no
#: string passes vacuously. The scan found 29 distinct overlapping segments,
#: and every one is accounted for:
#:
#:   * 28 are fragments of five Codex **schema identifiers** — `event_msg`
#:     field and tag names (`last_agent_message`, `codex_error_info`,
#:     `context_compacted`, `replacement_history`, `environment_context`).
#:     Naming the discriminator is what a `basis` annotation is *for*, so this
#:     overlap is expected and carries no session content.
#:   * 1 is a coincidental English collision: `" operating ins"`, where the
#:     annotation's "an operating instruction" meets the fixture's "you are
#:     operating inside…". Both sides are invented text.
#:
#: No corpus utterance is reproduced inside an annotation except the enum value
#: `interrupted` (`turn_aborted.reason`), which is a structural literal, and no
#: annotation appears inside a corpus utterance. `test_fixture_lint.py`
#: re-derives all of this rather than trusting this comment.
#:
#: Note the honest scope: this is *not* the stronger "zero overlap at 14
#: characters" property. That property is false for this corpus, and stating it
#: would be a claim no one could reproduce.
ANNOTATION_TEXT = frozenset(
    {
        # Role-less envelopes: harness bookkeeping with no typed content.
        "session_meta envelope: harness-written session header, no typed content",
        "session_meta envelope: harness-written session header (id, session_id, cwd), "
        "carries no typed content at all",
        "event_msg task_complete: a turn-lifecycle event the harness emits, not content",
        "event_msg task_complete carrying last_agent_message: a harness lifecycle event "
        "that quotes the model, not the human",
        "event_msg context_compacted: a lifecycle marker the harness emits about its own "
        "context management",
        "event_msg error with message and codex_error_info: a harness-reported failure "
        "about the run itself",
        "event_msg turn_aborted with reason interrupted: a harness record OF a human "
        "action (the interrupt), not text the human typed - the same distinction INV-8 "
        "draws for [Request interrupted by user]",
        "type compacted: a harness-synthesized replacement_history that rewrites the "
        "context window; the nested user turn inside it is quoted history, not a fresh "
        "human keystroke",
        # Role-bearing records: the classifier's actual domain.
        "role user, bare imperative request in ordinary prose with no wrapper or tag",
        "role user, bare imperative request in ordinary prose with no wrapper, tag, or "
        "preamble - an operator asking for work",
        "role assistant with output_text: model-generated reply, not typed by the human",
        "role assistant with output_text: model-generated reply, not typed by the human; "
        "non-human by role alone",
        "role developer, and the text describes the model's own execution environment "
        "(sandboxed container, no network) in the second person - an operating "
        "instruction addressed to the model, which is the harness speaking, not the "
        "operator",
        "role says user, but the whole text is a single machine-generated XML block "
        "<environment_context>...</environment_context> stating the platform and shell; "
        "a person asking for work does not hand-type a self-closing environment "
        "descriptor as their entire message",
    }
)

#: Exact key set of a `codex_role_label` record. These rows have no `type`
#: field — they are byte-identical to the blind labels committed before any
#: classifier existed, and re-encoding them would forfeit that provenance — so
#: the shape is recognized by key set instead. See `classify_record`.
ANNOTATION_KEYS = frozenset({"file", "index", "role", "channel", "basis"})

#: Channels a label row may assign, matching the adapter's own two constants.
ANNOTATION_CHANNELS = frozenset({"human", "injected"})

#: Roles a label row may record. `None` is the role-less case (a `session_meta`
#: envelope or an `event_msg`), which is most of the corpus.
ANNOTATION_ROLES = frozenset({"user", "assistant", "developer"})

#: Which fixture a label row may point at. Confines annotations to the
#: committed corpus: a label naming a path outside `tests/fixtures/` would be
#: describing something unreviewed.
ANNOTATION_FILE = re.compile(r"^tests/fixtures/[a-z0-9][a-z0-9/-]{0,80}\.jsonl$")


def _shape_codex_role_label(record: dict) -> Verdict:
    """Classify one blind channel-label row (task 7.1).

    A label row annotates a committed fixture record: which file, which record
    index, what role it carried, and which channel a human judged it to be.
    Every field is either a structural literal or an exact match against
    `ANNOTATION_TEXT`, so a row cannot smuggle unreviewed prose.

    Args:
        record: A decoded row whose key set is exactly `ANNOTATION_KEYS`.

    Returns:
        `ACCEPTED`, or the `Verdict` naming the first rule that rejected it.
    """
    verdict = _check_keys(record, frozenset(ANNOTATION_KEYS), frozenset(), "codex_role_label")
    if verdict is not None:
        return verdict

    path = record["file"]
    if not isinstance(path, str) or not ANNOTATION_FILE.fullmatch(path):
        return _reject(
            RULE_BAD_IDENTIFIER,
            f"file {str(path)[:60]!r} is not a committed fixture path",
        )

    index = record["index"]
    # `isinstance(True, int)` is True in Python, so booleans are rejected
    # explicitly rather than sliding through as 0 and 1.
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        return _reject(RULE_BAD_VALUE, f"index {index!r} is not a non-negative integer")

    role = record["role"]
    if role is not None:
        verdict = _check_literal(role, frozenset(ANNOTATION_ROLES), "codex_role_label.role")
        if verdict is not None:
            return verdict

    verdict = _check_literal(
        record["channel"], frozenset(ANNOTATION_CHANNELS), "codex_role_label.channel"
    )
    if verdict is not None:
        return verdict

    basis = record["basis"]
    if not isinstance(basis, str) or basis not in ANNOTATION_TEXT:
        return _reject(
            RULE_UNALLOWLISTED_TEXT,
            f"basis {str(basis)[:40]!r} is not in the annotation phrasebook",
        )
    return ACCEPTED


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
    # Annotation rows are dispatched on key set, before the `type` lookup below,
    # because they deliberately carry no `type` field. Order matters: after the
    # lookup they would already have been rejected as `unknown-record-type`,
    # which is why the tests assert the *rule name* a poisoned annotation
    # produces and not merely that the linter exited non-zero.
    if set(record) == ANNOTATION_KEYS:
        return _shape_codex_role_label(record)
    record_type = record.get("type")
    shape = None
    if isinstance(record_type, str):
        for shapes in (RECORD_SHAPES, codex.CODEX_RECORD_SHAPES, opencode.OPENCODE_RECORD_SHAPES):
            shape = shapes.get(record_type)
            if shape is not None:
                break
    if shape is None:
        allowed = sorted(
            set(RECORD_SHAPES)
            | set(codex.CODEX_RECORD_SHAPES)
            | set(opencode.OPENCODE_RECORD_SHAPES)
        )
        return _reject(
            RULE_UNKNOWN_RECORD_TYPE,
            f"record type {str(record_type)[:40]!r} is not in the shape allowlist ({allowed})",
        )
    return shape(record)
