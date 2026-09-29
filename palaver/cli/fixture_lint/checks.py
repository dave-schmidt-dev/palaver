"""
The Verdict record and the generic, source-agnostic structural checkers every classifier
calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import (
    RULE_BAD_IDENTIFIER,
    RULE_BAD_VALUE,
    RULE_MISSING_KEY,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNEXPECTED_KEY,
    SYNTHESIZED_TEXT,
)

#: Keys permitted inside a `tool_use` block's `input` map. A plain identifier
#: carries no prose; anything else is free text wearing a key's clothing.
INPUT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")


#: How deep a `tool_use` input map may nest before the linter stops walking.
#: `AskUserQuestion`'s real input reaches five levels (input → questions →
#: question → options → option → label), so this is that plus headroom, not an
#: arbitrary number.
MAX_INPUT_DEPTH = 8


@dataclass(frozen=True)
class Verdict:
    """The classifier's answer for one record.

    Attributes:
        ok: True when the record matched an allowlisted shape and every
            free-text payload it carried was in the phrasebook.
        rule: Which rule rejected it, from `RULE_NAMES`; empty when `ok`.
        detail: Human-readable specifics — which key, which value, which
            block. Never echoes more than a truncated prefix of an offending
            string, so a linter failure does not itself print a transcript.
    """

    ok: bool
    rule: str = ""
    detail: str = ""


#: The accepting verdict, as a singleton so a stub classifier in a test can
#: return exactly what the real one returns on success.
ACCEPTED = Verdict(ok=True)


def _reject(rule: str, detail: str) -> Verdict:
    return Verdict(ok=False, rule=rule, detail=detail)


def _check_keys(
    record: dict, required: frozenset[str], optional: frozenset[str], where: str
) -> Verdict | None:
    """Reject a mapping whose key set is not exactly what its shape declares."""
    keys = set(record)
    missing = required - keys
    if missing:
        return _reject(RULE_MISSING_KEY, f"{where} is missing {sorted(missing)}")
    unexpected = keys - required - optional
    if unexpected:
        return _reject(
            RULE_UNEXPECTED_KEY, f"{where} carries unexpected key(s) {sorted(unexpected)}"
        )
    return None


def _check_literal(value: object, allowed: frozenset[str], where: str) -> Verdict | None:
    """Reject a structural value that is not one of a fixed set of literals."""
    if not isinstance(value, str) or value not in allowed:
        return _reject(RULE_BAD_VALUE, f"{where} must be one of {sorted(allowed)}, got {value!r}")
    return None


def _check_pattern(value: object, pattern: re.Pattern[str], where: str) -> Verdict | None:
    """Reject an identifier that does not match its declared synthetic shape."""
    if not isinstance(value, str) or not pattern.match(value):
        return _reject(
            RULE_BAD_IDENTIFIER,
            f"{where} must match {pattern.pattern}, got {str(value)[:40]!r}",
        )
    return None


def _check_bool(value: object, where: str) -> Verdict | None:
    if not isinstance(value, bool):
        return _reject(RULE_BAD_VALUE, f"{where} must be a boolean, got {type(value).__name__}")
    return None


def _check_text(value: object, where: str) -> Verdict | None:
    """Reject any free-text payload that is not in the corpus phrasebook.

    This is the second allowlist layer. It is exact-match on purpose: a length
    bound, a character class, or a "looks synthetic" heuristic would each admit
    some real sentence, and every one of those admissions is silent.
    """
    if not isinstance(value, str):
        return _reject(RULE_BAD_VALUE, f"{where} must be a string, got {type(value).__name__}")
    if value not in SYNTHESIZED_TEXT:
        return _reject(
            RULE_UNALLOWLISTED_TEXT,
            f"{where} is not in the synthesized phrasebook: {value[:48]!r}",
        )
    return None


def _check_input_value(value: object, where: str, depth: int) -> Verdict | None:
    """Walk a `tool_use` input map, allowlisting every leaf it reaches.

    Booleans and nulls carry no prose and pass structurally. Strings go through
    the phrasebook. Numbers are *not* admitted: no fixture needs one, and every
    value type this function accepts is one somebody decided to accept.
    """
    if depth > MAX_INPUT_DEPTH:
        return _reject(RULE_BAD_VALUE, f"{where} nests deeper than {MAX_INPUT_DEPTH} levels")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return _check_text(value, where)
    if isinstance(value, list):
        for index, item in enumerate(value):
            verdict = _check_input_value(item, f"{where}[{index}]", depth + 1)
            if verdict is not None:
                return verdict
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not INPUT_KEY.match(key):
                return _reject(
                    RULE_BAD_IDENTIFIER,
                    f"{where} has a key that is not a plain identifier: {str(key)[:40]!r}",
                )
            verdict = _check_input_value(item, f"{where}.{key}", depth + 1)
            if verdict is not None:
                return verdict
        return None
    return _reject(RULE_BAD_VALUE, f"{where} has unsupported type {type(value).__name__}")
