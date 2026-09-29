"""The tier-4 observer-inference cap and the committed measurement that may lift it."""

from __future__ import annotations

import json
from pathlib import Path

from palaver.memory.tiers import TIER_OBSERVER_INFERENCE, tier_name

from .constants import MEASUREMENT_PATH, logger
from .records import CodexTierCapError, RoleClassMeasurement

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
