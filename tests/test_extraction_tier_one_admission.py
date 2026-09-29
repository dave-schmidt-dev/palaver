"""
Tier-1 admission: statement-equals-span (modulo whitespace/case) and INV-6 grounding of
the cited span.
"""

from __future__ import annotations

import pytest

from palaver.extract.quote_gate import (
    QuoteNotGroundedError,
    ground_quote,
    normalize_for_comparison,
)
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN
from palaver.memory.tiers import (
    TIER_OBSERVER_INFERENCE,
    TIER_USER_INSTRUCTION,
)
from tests._extraction_support import _chunk_id, _content, _replayed, _user_record

# --- tier-1 admission: the statement must BE the anchored span --------------


def test_statement_equal_to_its_span_modulo_whitespace_is_tier_one(tmp_path):
    """A statement that is the anchored span, differing only in whitespace and
    surrounding punctuation, is admitted as tier-1; one extra word is not.

    The pair is the whole point: the same chunk, the same quote, the same
    human channel, and the only variable is whether the statement *is* the
    span or merely resembles it.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("hold the release until the audit clears")])
    try:
        chunk_id = _chunk_id(conn, 1)
        quote = "hold the release until the audit clears"

        admitted = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote=quote,
            statement="  hold the  release until\nthe audit clears.  ",
        )
        assert admitted.tier == TIER_USER_INSTRUCTION
        assert admitted.is_tier_one
        assert admitted.channel == CHANNEL_HUMAN

        one_word_more = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote=quote,
            statement="hold the next release until the audit clears",
        )
        assert one_word_more.tier == TIER_OBSERVER_INFERENCE
    finally:
        conn.close()


def test_recapitalized_statement_is_not_tier_one(tmp_path):
    """Case is not part of the allowance, and that is deliberate.

    "Modulo whitespace and surrounding punctuation" is the whole latitude
    task 3.3 grants; a statement that recapitalizes the user's turn has
    edited it, and tier-1 is irrecoverable under INV-4/INV-5, so the gate
    fails closed to tier-4. This test exists so the choice is a pinned,
    visible property rather than an accident of the comparison function —
    if the project later decides case-insensitive is the better tradeoff,
    this is the test that has to change on purpose. The control differs
    from the demoted call by capitalization alone.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("freeze the schema")])
    try:
        chunk_id = _chunk_id(conn, 1)

        recapitalized = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="freeze the schema",
            statement="Freeze the schema.",
        )
        assert recapitalized.tier == TIER_OBSERVER_INFERENCE

        control = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="freeze the schema",
            statement="freeze the schema.",
        )
        assert control.tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()


def test_real_quote_wrong_statement_is_not_tier_one(tmp_path):
    """A verbatim quote carrying a statement the model wrote itself is tier-4.

    Input shape harvested from the archived baseline experiment. That run returned six
    `user_decisions`; the spike's substring check reported 6 of 6 quotes
    REAL, and three of the six nonetheless carried a `statement` the model
    had composed rather than quoted — including one whose quote was a bare
    identifier the human had typed and whose statement was a third-person
    summary of what handing over that identifier meant. That is the shape
    reproduced here. Per INV-9 the content is invented for this test: the
    real session's prose is not committed to this public repository, and the
    run is named instead.

    The positive control is the same chunk and the same quote with the
    statement set to the span itself, so this asserts the gate discriminates
    statements rather than distrusting this chunk.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("the deploy key is DK-4417, use that one")])
    try:
        chunk_id = _chunk_id(conn, 1)

        summarized = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="DK-4417",
            statement="Provided the deploy key for the staging cluster",
        )
        assert summarized.tier == TIER_OBSERVER_INFERENCE
        assert summarized.channel == CHANNEL_HUMAN  # the quote is real and human, only the
        assert summarized.span_text == "DK-4417"  # statement is the model's own words

        control = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="DK-4417",
            statement="DK-4417",
        )
        assert control.tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()


def test_statement_that_merely_cites_the_span_is_not_tier_one(tmp_path):
    """A statement quoting a span inside a longer sentence is tier-4, not tier-1.

    Also a shape from the run named in `test_real_quote_wrong_statement_is_not_tier_one`:
    the model wrapped the human's words in a sentence of its own
    (`<topic>: chose "<quote>"`). The quote is real and the substring check
    passes; the memory still is not the user's own words.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("keep the retry window at ten seconds")])
    try:
        chunk_id = _chunk_id(conn, 1)
        quote = "keep the retry window at ten seconds"

        cited = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote=quote,
            statement=f'Retry policy: chose "{quote}"',
        )
        assert cited.tier == TIER_OBSERVER_INFERENCE

        control = ground_quote(conn, transcript_chunk_id=chunk_id, quote=quote, statement=quote)
        assert control.tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()


def test_statement_and_span_that_both_reduce_to_nothing_are_not_tier_one(tmp_path):
    """Two strings that normalize to `""` are equal only trivially, never tier-1.

    Without the non-empty guard, a punctuation-only statement would match a
    punctuation-only span and mint tier-1 out of nothing at all. The control
    uses the same chunk, so the chunk itself is demonstrably tier-1-capable.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("??? ship it")])
    try:
        chunk_id = _chunk_id(conn, 1)

        assert normalize_for_comparison("???") == ""
        empty_match = ground_quote(conn, transcript_chunk_id=chunk_id, quote="???", statement="...")
        assert empty_match.tier == TIER_OBSERVER_INFERENCE

        control = ground_quote(
            conn, transcript_chunk_id=chunk_id, quote="ship it", statement="ship it"
        )
        assert control.tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()


# --- INV-6: the quote must be in its cited evidence span --------------------


@pytest.mark.inv6
def test_paraphrase_of_a_real_quote_fails_the_substring_check(tmp_path):
    """A paraphrase of what the human said is not a quote and is rejected.

    The paraphrase preserves the meaning and most of the words; the control
    passes the verbatim text through the same call, so this measures the
    substring check rather than a chunk that could not be quoted at all.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("move the nightly job to 03:00 UTC")])
    try:
        chunk_id = _chunk_id(conn, 1)

        with pytest.raises(QuoteNotGroundedError):
            ground_quote(
                conn,
                transcript_chunk_id=chunk_id,
                quote="move the nightly job to 3am UTC",
                statement="move the nightly job to 3am UTC",
            )

        control = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="move the nightly job to 03:00 UTC",
            statement="move the nightly job to 03:00 UTC",
        )
        assert control.tier == TIER_USER_INSTRUCTION
    finally:
        conn.close()


@pytest.mark.inv6
def test_quote_outside_the_cited_span_is_rejected_even_though_it_is_in_the_chunk(tmp_path):
    """Grounding is against the *cited* span, not the whole transcript.

    Both calls use a quote that really is in the chunk; only the citation
    differs, so this pins that `cited_span` is honoured rather than ignored.
    """
    conn, _, _ = _replayed(tmp_path, [_user_record("first drop the flag, then rerun the suite")])
    try:
        chunk_id = _chunk_id(conn, 1)
        content = _content(conn, chunk_id)
        split = content.index("then")

        with pytest.raises(QuoteNotGroundedError):
            ground_quote(
                conn,
                transcript_chunk_id=chunk_id,
                quote="rerun the suite",
                statement="rerun the suite",
                cited_span=(0, split),
            )

        control = ground_quote(
            conn,
            transcript_chunk_id=chunk_id,
            quote="rerun the suite",
            statement="rerun the suite",
            cited_span=(split, len(content)),
        )
        assert control.anchor.start_offset >= split
        assert (
            _content(conn, chunk_id)[control.anchor.start_offset : control.anchor.end_offset]
            == "rerun the suite"
        )
    finally:
        conn.close()
