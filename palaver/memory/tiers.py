"""Provenance tiers for `memories` rows (INV-5), highest confidence first.

Five tiers, per `INVARIANTS.md` INV-5: (1) explicit user instruction or
correction, (2) explicit main-agent conclusion, (3) observed tool or command
result, (4) observer inference, (5) observer speculation. A tier is a claim
about *how a statement was produced*, not about its subject matter. It is
assigned once, at insert (`palaver.memory.write.write_memory`), and never
changed afterward — the database-level enforcement of that immutability
lives in `palaver/store/schema.py` migration 3 (`memories_tier_immutable`),
not here. This module only names the tiers so callers and tests share one
vocabulary instead of scattering the integers 1-5 through the codebase.

`memories.tier` is CHECK-constrained to 1-5 in `schema.py`; `ALL_TIERS` here
expresses the same range as a Python tuple.
"""

from __future__ import annotations

TIER_USER_INSTRUCTION = 1
TIER_AGENT_CONCLUSION = 2
TIER_OBSERVED_RESULT = 3
TIER_OBSERVER_INFERENCE = 4
TIER_OBSERVER_SPECULATION = 5

TIER_NAMES: dict[int, str] = {
    TIER_USER_INSTRUCTION: "user_instruction",
    TIER_AGENT_CONCLUSION: "agent_conclusion",
    TIER_OBSERVED_RESULT: "observed_result",
    TIER_OBSERVER_INFERENCE: "observer_inference",
    TIER_OBSERVER_SPECULATION: "observer_speculation",
}

ALL_TIERS: tuple[int, ...] = tuple(sorted(TIER_NAMES))


def tier_name(tier: int) -> str:
    """Return the human-readable name for a provenance tier.

    Args:
        tier: A provenance tier, 1 (highest confidence) through 5 (lowest).

    Returns:
        The tier's name, e.g. `"user_instruction"` for tier 1.

    Raises:
        ValueError: `tier` is not one of the five defined tiers.
    """
    try:
        return TIER_NAMES[tier]
    except KeyError:
        raise ValueError(f"unknown tier: {tier!r} (expected one of {ALL_TIERS})") from None
