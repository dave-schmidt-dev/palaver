"""
Prompt sizing: fit to one slot, fall back on an unreadable budget, derive from the live
server, or take an explicit override.
"""

from __future__ import annotations

from palaver.bench import (
    DEFAULT_PROMPT_FRACTION,
    FALLBACK_PROMPT_WORDS,
    TOKENS_PER_WORD,
    resolve_prompt_words,
    run_bench,
)
from tests._bench_support import SESSIONS, STUB_N_CTX, STUB_SLOTS, TEST_PROMPT_WORDS, _run
from tests._bench_support import stub_server as stub_server

# =============================================================================
# Prompt sizing: the first live run 500'd because a fixed size cannot be right
# =============================================================================


def test_the_prompt_is_sized_to_one_slot_not_the_whole_reported_context():
    """`n_ctx` is the server's total, shared across its slots.

    Measured 2026-08-15 against the observed server: a ~19,700-token prompt to
    a four-slot server reporting `n_ctx: 32768` came back
    `500 Context size has been exceeded`, which is impossible against a
    32,768-token slot and expected against an 8,192-token one.
    """
    words = resolve_prompt_words(32768, 4)
    per_slot_tokens = 32768 // 4

    assert words * TOKENS_PER_WORD <= per_slot_tokens
    # And it is not merely small: it uses the share it was told it could.
    assert words * TOKENS_PER_WORD >= per_slot_tokens * DEFAULT_PROMPT_FRACTION * 0.9
    # The control that pins the division: the same context on one slot allows
    # roughly four times the prompt. Without it, a helper that ignored
    # `total_slots` and just returned something small would pass above.
    assert resolve_prompt_words(32768, 1) > words * 3
    # And the size that actually failed live is excluded.
    assert words < 12000


def test_an_unreadable_context_budget_falls_back_to_a_small_prompt():
    assert resolve_prompt_words(None, 4) == FALLBACK_PROMPT_WORDS
    assert resolve_prompt_words(0, 4) == FALLBACK_PROMPT_WORDS
    # Positive control: a real budget is not the fallback.
    assert resolve_prompt_words(32768, 4) != FALLBACK_PROMPT_WORDS


def test_run_bench_derives_the_prompt_size_from_the_server(stub_server, tmp_path):
    """The derivation is wired to the round, not merely available beside it."""
    handle = stub_server(barrier_parties=SESSIONS)

    report = run_bench(
        db_path=tmp_path / "bench.db",
        sessions=SESSIONS,
        host=handle.host,
        port=handle.port,
        timeout=30.0,
    )

    assert report.ok
    expected = resolve_prompt_words(STUB_N_CTX, STUB_SLOTS)
    sent = [len(prompt.split()) for prompt in handle.prompts]
    assert len(sent) == SESSIONS
    # Within a line's worth of words of the derived size, since the prompt is
    # built from whole repetitions of one line.
    assert all(abs(count - expected) < 20 for count in sent), sent
    # And it is the *derived* size, not the fallback the code uses when it
    # cannot read /props.
    assert expected != FALLBACK_PROMPT_WORDS


def test_an_explicit_prompt_size_overrides_the_derivation(stub_server, tmp_path):
    handle = stub_server(barrier_parties=SESSIONS)

    report = _run(handle, tmp_path / "bench.db")

    assert report.ok
    sent = [len(prompt.split()) for prompt in handle.prompts]
    assert all(abs(count - TEST_PROMPT_WORDS) < 20 for count in sent), sent
