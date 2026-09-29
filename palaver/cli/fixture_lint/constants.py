"""
NAME/HELP, every RULE_* rejection-rule constant and RULE_NAMES, the SYNTHESIZED_TEXT
phrasebook, and the two identifier patterns (SESSION_ID, TOOL_USE_ID) shared by all
three sources.
"""

from __future__ import annotations

import re

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
#: The Codex/OpenCode analogue of `RULE_UNKNOWN_SYSTEM_SUBTYPE`: a nested
#: discriminator (Codex's `event_msg.payload.type`, OpenCode's
#: `part.data.type`) inside an otherwise-recognized record shape, whose value
#: this corpus does not classify. One shared rule rather than one per source,
#: because it is the same dimension — "which sub-shape applies" — recurring in
#: a second and third source, not a new kind of failure.
RULE_UNKNOWN_SUBTYPE = "unknown-subtype"
RULE_MISSING_KEY = "missing-key"
RULE_UNEXPECTED_KEY = "unexpected-key"
RULE_UNKNOWN_CONTENT_BLOCK = "unknown-content-block"
RULE_UNKNOWN_TOOL = "unknown-tool"
RULE_BAD_IDENTIFIER = "bad-identifier"
RULE_BAD_VALUE = "bad-value"
RULE_UNALLOWLISTED_TEXT = "unallowlisted-text"
RULE_UNTERMINATED_FILE = "unterminated-file"

#: A file under the corpus whose extension no checker claims. Rejected rather
#: than skipped, which is the whole point: until 2026-08-15 discovery was
#: `rglob("*.jsonl")`, so seven committed files — two of them carrying
#: session-shaped prose — were never opened by the gate that exists to read
#: them, and nothing said so. A corpus cannot be checked by a linter that
#: decides for itself which files count.
RULE_UNKNOWN_FILE_TYPE = "unknown-file-type"

#: A quoted string outside a `.jsonl` record — a JSON string value, a markdown
#: code span or fenced line, or the text half of a golden line — that is
#: neither phrasebook nor a structural token. The record analogue is
#: `RULE_UNALLOWLISTED_TEXT`; kept separate so a rejection names which surface
#: it came from.
RULE_UNALLOWLISTED_SPAN = "unallowlisted-span"
RULE_SOURCE_PROVENANCE = "source-provenance"

#: Text of any kind carrying a marker only a real session store produces — a
#: bare UUID, an opaque `toolu_`/`msg_` id, an absolute home path, an email
#: address. This is the only check applied to authored narrative, and it is a
#: marker scan rather than an allowlist because a 17 KB README cannot be
#: equality-checked against anything but itself.
RULE_PROVENANCE_MARKER = "provenance-marker"

#: A golden-output line whose label is not label-shaped. The text half is
#: checked against the phrasebook like any other quoted string; this rule
#: covers the half that names the channel.
RULE_BAD_GOLDEN_LABEL = "bad-golden-label"

#: Every rule `classify_record` and `lint_tree` can report, for the report
#: legend and for the tests' "this rule exists" assertions.
RULE_NAMES: tuple[str, ...] = (
    RULE_UNDECODABLE,
    RULE_NOT_AN_OBJECT,
    RULE_UNKNOWN_RECORD_TYPE,
    RULE_UNKNOWN_SYSTEM_SUBTYPE,
    RULE_UNKNOWN_SUBTYPE,
    RULE_MISSING_KEY,
    RULE_UNEXPECTED_KEY,
    RULE_UNKNOWN_CONTENT_BLOCK,
    RULE_UNKNOWN_TOOL,
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNTERMINATED_FILE,
    RULE_UNKNOWN_FILE_TYPE,
    RULE_UNALLOWLISTED_SPAN,
    RULE_PROVENANCE_MARKER,
    RULE_BAD_GOLDEN_LABEL,
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
        # Codex: human-channel turns and assistant replies.
        "check the staging deploy status",
        "the staging deploy is healthy",
        # Codex: harness-written content on the one reliably-harness channel
        # (`role: "developer"`, `docs/research.md` §2).
        "you are operating inside a sandboxed fixture container with no network access",
        # Codex: `role: "user"` wearing an injected prefix. Codex has no
        # `isMeta` equivalent, so this is exactly the shape the prefix
        # heuristic in a future adapter (task 7.1) has to see in the corpus.
        "<environment_context>fixture sandbox: bash on linux</environment_context>",
        # Codex: a turn's final assistant message, and an error message.
        "the fixture worker finished the requested change",
        "the fixture worker is retrying after a transient timeout",
        # OpenCode: human-channel turns and assistant replies.
        "restart the worker queue",
        "the worker queue is restarted",
        # OpenCode: a tool-part error message (`state.error`).
        "fixture tool exited with a non-zero status",
        # OpenCode: a synthetic (harness-injected) text part attached to a
        # `role: "user"` message — the same channel-ambiguity lesson INV-8
        # names for Claude Code, reproduced at the *part* level.
        "session continuation: resuming after context compaction",
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
