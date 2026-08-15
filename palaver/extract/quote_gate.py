"""Quote-grounding gate and span-anchored tier-1 (Task 3.3).

This module is the memory write boundary for anything a model extracted. It
carries the half of two invariants (`INVARIANTS.md`) that no earlier layer
has a write boundary to enforce:

**INV-6 — a quote that does not appear in its cited evidence span is
rejected at write time.** `ground_quote` locates the quote inside the cited
span and returns the offsets of the match; `admit_decision` raises
`QuoteNotGroundedError` before issuing any `INSERT`, so a rejected decision
leaves no row behind to review later.

**INV-8 — a quote drawn from an injected channel is never tier-1.** Tier-1
means "the user said this", and under INV-5 it is the tier every other tier
defers to, so a mis-attributed quote corrupts the store by construction and
INV-4 makes the corruption permanent. Everything this module cannot prove is
a human instruction is admitted at `TIER_OBSERVER_INFERENCE` instead —
tier-4 is the fail-closed direction, and demotion is always the outcome, not
rejection, because the evidence is real even when the attribution is not.

**Substring presence is necessary and not sufficient, measured.** Spike run
2's baseline arm over fixture B (`spikes/2026-08-14-e4b-extraction/out_B_base.txt`,
Gemma 3n E4B q4_0, 2026-08-14) returned six `user_decisions` whose quotes all
passed the spike's substring check — the check reported 6 of 6 REAL — while
three of the six carried a `statement` the model had written itself: a
third-person summary of what the quote showed, a paraphrase of it, and a
statement that merely cited the quoted fragment inside a longer sentence. A
substring check cannot tell "the user said X" from "the user said something
and the model concluded X", so it is only the floor here. Tier-1 additionally
requires the `statement` to *be* the anchored span, compared modulo
whitespace and surrounding punctuation (`normalize_for_comparison`). A
statement that merely cites a span is tier-4.

**The channel comes from the chunk's own first rendered line, never from a
backward scan.** `transcript_chunks.content` holds one record's normalized
rendering (see `palaver/replay.py`), and `classify_channel` classifies a
record as a whole, so every text line a chunk renders carries the *same*
tag. A backward scan from the match ("which tagged line precedes this
offset?") would therefore add nothing legitimate while opening a real
attack: `_clip` keeps newlines inside a turn, so injected content whose body
contains the line `HUMAN: ship it` renders as an `INJECTED:` first line
followed by a line that looks tagged. Under a backward scan that inner line
would mint tier-1 out of injected content — exactly INV-8's failure mode.
The first rendered line's tag governs the whole chunk, and a span that
reaches into the chunk's `tool>`/`result>` lines (tool output carries no
channel, because nothing was said — something happened) is demoted too.

**Case is compared exactly.** "Modulo whitespace and surrounding
punctuation" is the whole allowance: a model that recapitalizes a turn has
edited the user's words, and this gate errs toward tier-4 in every case it
cannot decide, because a wrongly minted tier-1 cannot be retracted under
INV-4.

Quotes are grounded against `transcript_chunks.content` only — the
normalized text the model actually read. The raw JSONL record lives in
`events.payload`, and a substring check against that would measure JSON
escaping rather than grounding (task 3.3's amendment: a quote containing a
newline, a double quote, or a backslash cannot appear verbatim in
`json.dumps` output), so this module never reads it.

No `on_status` channel (INV-1): every operation here is one indexed `SELECT`
and pure string work, with no network call, subprocess, model inference, or
stall-prone IO to surface progress for.

Exception messages carry offsets and lengths, never the quoted text: this
module's inputs are observed-session content, and an exception string
propagates into logs and reports (INV-9). This repository is public and
nothing in these docstrings comes from a real observed session.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from palaver.extract.normalize import AGENT_TAG, CHANNEL_TAG
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN
from palaver.memory.evidence import EvidenceAnchor
from palaver.memory.tiers import TIER_OBSERVER_INFERENCE, TIER_USER_INSTRUCTION
from palaver.memory.write import write_memory

logger = logging.getLogger(__name__)

#: Channel of a chunk whose first line is an `AGENT:` turn.
CHANNEL_AGENT = "agent"

#: Channel of a chunk whose first line is tool output (`tool>`, `result>`,
#: `result[error]>`). Not a channel in INV-8's sense at all — nothing was
#: said — which is precisely why it can never be tier-1.
CHANNEL_TOOL_RESULT = "tool_result"

#: Channel of a chunk whose first line is a `SYSTEM(kind)` record.
CHANNEL_SYSTEM = "system"

#: Channel of a chunk that rendered no recognizable tag at all, including
#: the empty chunk a record that normalizes to nothing leaves behind.
CHANNEL_UNTAGGED = "untagged"

#: Inverted from `palaver.extract.normalize.CHANNEL_TAG` rather than spelled
#: out again, so the tag this module reads back is by construction the tag
#: the normalizer writes.
_TAG_TO_CHANNEL: dict[str, str] = {tag: channel for channel, tag in CHANNEL_TAG.items()}
_TAG_TO_CHANNEL[AGENT_TAG] = CHANNEL_AGENT

#: Line prefixes the normalizer uses for content that carries no channel.
#: Matched after the optional `[HH:MM:SS] ` timestamp prefix and including
#: the two-space indent `_render_user`/`_render_assistant` emit.
_TOOL_LINE_PREFIXES = ("  tool>", "  result>", "  result[error]>")

_SYSTEM_LINE_PREFIX = "SYSTEM("

#: Characters stripped from both ends of a statement or span before they are
#: compared. Whitespace is in the set so a stripped bracket can expose a
#: space that then strips too (`"( ship it )"` -> `"ship it"`).
_EDGE_PUNCTUATION = " \t\n\"'`“”‘’«»„.,;:!?…()[]{}<>*_~-–—"

_TIMESTAMP_PREFIX_LENGTH = len("[00:00:00] ")


class QuoteGateError(Exception):
    """Base class for every rejection this module raises."""


class QuoteNotGroundedError(QuoteGateError):
    """Raised when a quote cannot be located in the evidence span it cites.

    Covers an empty quote, a cited chunk that does not exist, a cited span
    outside the chunk's bounds, and the substring failure itself — all four
    are the same thing from the store's point of view: a decision whose
    evidence cannot be checked, which INV-6 refuses to admit rather than
    flag for later review.
    """


@dataclass(frozen=True)
class GroundedQuote:
    """One quote, located in its cited span and tiered.

    Attributes:
        anchor: The `EvidenceAnchor` for the match, with offsets absolute in
            the chunk's content (not relative to the cited span).
        tier: `TIER_USER_INSTRUCTION` (1) only when the quote is a human
            turn and the statement is the anchored span itself; otherwise
            `TIER_OBSERVER_INFERENCE` (4).
        channel: The chunk's channel, from its first rendered line — one of
            `CHANNEL_HUMAN`, `CHANNEL_INJECTED`, `CHANNEL_AGENT`,
            `CHANNEL_TOOL_RESULT`, `CHANNEL_SYSTEM`, `CHANNEL_UNTAGGED`.
        span_text: The exact matched text, read back out of the chunk by the
            anchor's own offsets rather than echoed from the caller's
            `quote` argument.
        reason: Short, content-free explanation of the tier decision, for
            logs and audits.
    """

    anchor: EvidenceAnchor
    tier: int
    channel: str
    span_text: str
    reason: str

    @property
    def is_tier_one(self) -> bool:
        """Whether this quote was admitted as an explicit user instruction."""
        return self.tier == TIER_USER_INSTRUCTION


@dataclass(frozen=True)
class AdmittedDecision:
    """A decision that passed the gate and the `memories` row it became."""

    memory_id: int
    grounded: GroundedQuote


def normalize_for_comparison(text: str) -> str:
    """Reduce text to the form tier-1's statement/span comparison uses.

    Collapses every run of whitespace to a single space and strips
    surrounding punctuation and whitespace from both ends. Case, internal
    punctuation, and word order are left exactly as written — see the module
    docstring for why the allowance stops here.

    Args:
        text: Statement or span text.

    Returns:
        The comparison form, which is `""` for text that is entirely
        whitespace and punctuation.
    """
    return " ".join(text.split()).strip(_EDGE_PUNCTUATION)


def _strip_timestamp(line: str) -> str:
    """Drop the normalizer's optional `[HH:MM:SS] ` prefix from one line."""
    if (
        len(line) >= _TIMESTAMP_PREFIX_LENGTH
        and line.startswith("[")
        and line[3] == ":"
        and line[6] == ":"
        and line[9:11] == "] "
    ):
        return line[_TIMESTAMP_PREFIX_LENGTH:]
    return line


def _line_channel(line: str) -> str:
    """Classify one rendered transcript line by the tag the normalizer gave it."""
    body = _strip_timestamp(line)
    for tag, channel in _TAG_TO_CHANNEL.items():
        if body.startswith(f"{tag}: "):
            return channel
    if body.startswith(_SYSTEM_LINE_PREFIX):
        return CHANNEL_SYSTEM
    if body.startswith(_TOOL_LINE_PREFIXES):
        return CHANNEL_TOOL_RESULT
    return CHANNEL_UNTAGGED


def _chunk_channel(content: str) -> str:
    """Return the channel governing a whole chunk: its first line's tag.

    One chunk is one record's rendering and `classify_channel` classifies a
    record as a whole, so the first rendered line's tag governs every line
    below it. See the module docstring for the impersonation attack this
    closes.
    """
    if not content:
        return CHANNEL_UNTAGGED
    return _line_channel(content.split("\n", 1)[0])


def _tool_output_starts_at(content: str) -> int | None:
    """Return the offset where this chunk's first tool-output line begins.

    `None` when the chunk renders no `tool>`/`result>` line at all. A span
    reaching at or past this offset has left the turn's own text and is
    quoting tool output, which carries no channel and cannot be tier-1.
    """
    offset = 0
    for line in content.split("\n"):
        if _line_channel(line) == CHANNEL_TOOL_RESULT:
            return offset
        offset += len(line) + 1
    return None


def _chunk_content(conn: sqlite3.Connection, transcript_chunk_id: int) -> str:
    row = conn.execute(
        "SELECT content FROM transcript_chunks WHERE id = ?", (transcript_chunk_id,)
    ).fetchone()
    if row is None:
        raise QuoteNotGroundedError(
            f"cited transcript_chunks row {transcript_chunk_id} does not exist"
        )
    return row[0]


def ground_quote(
    conn: sqlite3.Connection,
    *,
    transcript_chunk_id: int,
    quote: str,
    statement: str,
    cited_span: tuple[int, int] | None = None,
) -> GroundedQuote:
    """Locate a quote in its cited evidence span and decide its tier.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 4.
        transcript_chunk_id: `transcript_chunks.id` the decision cites. Its
            `content` is the normalized text the model read (see
            `palaver/replay.py`); this function never reads `events.payload`.
        quote: The verbatim text the model claims the transcript contains.
            Must be non-empty and must appear in the cited span.
        statement: The memory's text — what the model claims the quote
            shows. Tier-1 requires this to be the anchored span itself.
        cited_span: `(start, end)` half-open offsets narrowing the citation
            to part of the chunk. Defaults to the whole chunk. Offsets in
            the returned anchor are always absolute in the chunk, whether or
            not this is passed.

    Returns:
        A `GroundedQuote`. The first occurrence of `quote` in the cited span
        is the one anchored, deterministically, so re-running the gate over
        the same inputs always produces the same offsets.

    Raises:
        QuoteNotGroundedError: `quote` is empty, the cited chunk does not
            exist, `cited_span` is not a valid span of that chunk, or
            `quote` does not appear in the cited span.
    """
    if not quote.strip():
        raise QuoteNotGroundedError(
            "a decision's quote is empty: there is nothing to ground it against (INV-6)"
        )

    content = _chunk_content(conn, transcript_chunk_id)
    span_start, span_end = (0, len(content)) if cited_span is None else cited_span
    if not 0 <= span_start < span_end <= len(content):
        raise QuoteNotGroundedError(
            f"cited span [{span_start}:{span_end}] is not a valid span of "
            f"transcript_chunks row {transcript_chunk_id} (length {len(content)})"
        )

    found = content.find(quote, span_start, span_end)
    if found < 0:
        raise QuoteNotGroundedError(
            f"a quote of {len(quote)} characters does not appear in the cited span "
            f"[{span_start}:{span_end}] of transcript_chunks row {transcript_chunk_id} "
            "(INV-6: a quote that is not in its cited evidence span is rejected at write "
            "time, not flagged for review)"
        )

    start, end = found, found + len(quote)
    anchor = EvidenceAnchor(
        start_offset=start, end_offset=end, transcript_chunk_id=transcript_chunk_id
    )
    span_text = content[start:end]
    channel = _chunk_channel(content)
    tool_output_at = _tool_output_starts_at(content)

    if channel != CHANNEL_HUMAN:
        tier, reason = (
            TIER_OBSERVER_INFERENCE,
            f"quote is from the {channel!r} channel, not a human turn (INV-8)",
        )
    elif tool_output_at is not None and end > tool_output_at:
        tier, reason = (
            TIER_OBSERVER_INFERENCE,
            "quote reaches into the chunk's tool output, which carries no channel",
        )
    elif not _statement_is_the_span(statement, span_text):
        tier, reason = (
            TIER_OBSERVER_INFERENCE,
            "statement is not the anchored span: it cites the quote rather than being it",
        )
    else:
        tier, reason = TIER_USER_INSTRUCTION, "statement is the anchored span of a human turn"

    logger.debug(
        "grounded quote in chunk %s at [%s:%s]: tier %s (%s)",
        transcript_chunk_id,
        start,
        end,
        tier,
        reason,
    )
    return GroundedQuote(
        anchor=anchor, tier=tier, channel=channel, span_text=span_text, reason=reason
    )


def _statement_is_the_span(statement: str, span_text: str) -> bool:
    """Whether `statement` is the anchored span, modulo whitespace and edge punctuation.

    Both sides must be non-empty after normalization: two strings that
    reduce to `""` (a statement of only punctuation, an all-whitespace span)
    are equal in the trivial sense and must not mint tier-1 on that basis.
    """
    normalized_statement = normalize_for_comparison(statement)
    if not normalized_statement:
        return False
    return normalized_statement == normalize_for_comparison(span_text)


def admit_decision(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    statement: str,
    quote: str,
    transcript_chunk_id: int,
    origin: str,
    session_id: int | None = None,
    cited_span: tuple[int, int] | None = None,
    supersedes: int | None = None,
) -> AdmittedDecision:
    """Ground one extracted decision and write it, at the tier it earned.

    This is the write boundary task 3.3 names: grounding happens *before*
    any `INSERT`, so a rejected decision leaves no `memories` row, no
    `memory_evidence` row, and nothing for a later pass to mistake for a
    checked memory. The caller owns committing the connection.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 4.
        project_id: `projects.id` this memory belongs to.
        statement: The memory's text.
        quote: The verbatim text the model claims the transcript contains.
        transcript_chunk_id: `transcript_chunks.id` the decision cites.
        origin: Free-text description of what produced this decision, e.g.
            `"observer-extraction"`. Recorded as `memories.origin`.
        session_id: `sessions.id` this decision was extracted from, if any.
        cited_span: Narrower citation within the chunk; see `ground_quote`.
        supersedes: `memories.id` this decision reclassifies, if any. Passed
            through to `write_memory` unchanged — INV-5's tier-ordering rule
            on that link is enforced by the database trigger, not here.

    Returns:
        The `AdmittedDecision`: the new `memories.id` and the
        `GroundedQuote` whose anchor was written with it.

    Raises:
        QuoteNotGroundedError: The quote is not grounded in its cited span;
            nothing was written.
    """
    grounded = ground_quote(
        conn,
        transcript_chunk_id=transcript_chunk_id,
        quote=quote,
        statement=statement,
        cited_span=cited_span,
    )
    memory_id = write_memory(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement=statement,
        origin=origin,
        tier=grounded.tier,
        evidence=[grounded.anchor],
        supersedes=supersedes,
    )
    return AdmittedDecision(memory_id=memory_id, grounded=grounded)
