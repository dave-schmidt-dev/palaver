"""Sizing and synthesizing the benchmark prompt to fit one of the server's KV slots."""

from __future__ import annotations

from collections.abc import Callable

from palaver.extract.client import (
    ModelClientError,
)
from palaver.extract.slots import SlotClient

#: Tokens per word for the synthesized prompt, measured against the observed
#: server on 2026-08-15: 1,000 words produced 1,572 prompt tokens and 4,000
#: produced 6,567 (`usage.prompt_tokens`, both requests). The higher of the two
#: ratios is used so the estimate errs toward a shorter prompt.
TOKENS_PER_WORD = 1.65

#: Share of one slot's context the synthesized prompt may fill, leaving the
#: rest for the response. Not a measurement — a margin.
DEFAULT_PROMPT_FRACTION = 0.6

#: Used only when the server's context budget could not be read, which in
#: practice means the server is unreachable and every request is about to fail
#: anyway. Small enough to fit any plausible slot.
FALLBACK_PROMPT_WORDS = 1000


#: The one invented line the synthesized prompt is built from. No observed
#: session content ever reaches the model from this module (INV-9).
_PROMPT_LINE = "fixture: invented benchmark transcript line carrying no observed content"


def resolve_prompt_words(
    n_ctx: int | None, total_slots: int, *, fraction: float = DEFAULT_PROMPT_FRACTION
) -> int:
    """Size the synthesized prompt to fit one of the server's KV slots.

    **`n_ctx` is the whole server's context, shared across its slots, not one
    slot's.** Measured on 2026-08-15: a server reporting `total_slots: 4` and
    `default_generation_settings.n_ctx: 32768` rejected a ~19,700-token prompt
    with `{"error":{"code":500,"message":"Context size has been
    exceeded."}}` — which cannot happen against a 32,768-token slot, and is
    exactly what a 8,192-token slot does. A benchmark whose every request 500s
    measures nothing, so the size is derived here rather than guessed.

    Args:
        n_ctx: The server's reported context length, or `None` when it did not
            report one.
        total_slots: The server's slot count; values below 1 are treated as 1.
        fraction: Share of the per-slot budget the prompt may occupy.

    Returns:
        A positive word count. `FALLBACK_PROMPT_WORDS` when `n_ctx` is unknown
        or not positive.
    """
    if not n_ctx or n_ctx <= 0:
        return FALLBACK_PROMPT_WORDS
    per_slot_tokens = n_ctx // max(1, total_slots)
    prompt_tokens = per_slot_tokens * fraction
    return max(1, int(prompt_tokens / TOKENS_PER_WORD))


def synthetic_prompt(words: int, *, label: str = "") -> str:
    """Build an invented prompt of roughly `words` words.

    Deterministic, so two runs of the benchmark send the same number of tokens
    and their latencies are comparable. `label` is prefixed so each concurrent
    request differs by at least one token, which keeps a server-side prompt
    cache from making five of the six requests trivially fast and the
    measurement meaningless.

    Args:
        words: Approximate word count. Must be positive.
        label: Optional per-session prefix.

    Returns:
        The prompt text.

    Raises:
        ValueError: If `words` is not positive — a zero-word prompt would
            measure the server's overhead, not its inference.
    """
    if words <= 0:
        raise ValueError(f"words must be positive, got {words}")
    line_words = len(_PROMPT_LINE.split())
    # `+ 1` for the index appended to each line. Counting only `_PROMPT_LINE`'s
    # own words overshot the requested size by about 10%, which eats into the
    # margin `resolve_prompt_words` leaves for the response.
    repeats = max(1, words // (line_words + 1))
    body = "\n".join(f"{_PROMPT_LINE} {index}" for index in range(repeats))
    return f"{label}\n{body}" if label else body


def _derive_prompt_words(
    *,
    host: str,
    port: int,
    timeout: float,
    fraction: float,
    on_status: Callable[[str], None] | None,
) -> int:
    """Ask the server for its context budget and size the prompt to one slot.

    Reuses task 4.2's `SlotClient` rather than opening its own socket, so the
    benchmark and `palaver doctor` read the same `/props` through the same
    parser. A failure here is not fatal: an unreachable server means every
    request is about to fail with a connection error, and that report is more
    useful than an exception raised before the round even started.
    """
    try:
        properties = SlotClient(host=host, port=port, timeout=timeout).properties(
            on_status=on_status
        )
    except ModelClientError as exc:
        if on_status is not None:
            on_status(f"could not read the server's context budget ({exc}); using a small prompt")
        return FALLBACK_PROMPT_WORDS
    words = resolve_prompt_words(properties.n_ctx, properties.total_slots, fraction=fraction)
    if on_status is not None:
        on_status(
            f"server reports n_ctx {properties.n_ctx} across {properties.total_slots} slot(s); "
            f"synthesizing a ~{words}-word prompt"
        )
    return words
