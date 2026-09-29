"""Classifying a message record's channel: human prose vs. harness-injected text."""

from __future__ import annotations

from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED

from .constants import HARNESS_ROLES, HUMAN_CANDIDATE_ROLE, INJECTED_TEXT_PREFIXES
from .records import message_role, message_text


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
