"""
check_span and its supporting regexes/phrasebook-adjacent constants: judging one quoted
string from a non-record surface.
"""

from __future__ import annotations

import re

from .annotation import ANNOTATION_TEXT
from .checks import ACCEPTED, Verdict, _reject
from .constants import RULE_PROVENANCE_MARKER, RULE_UNALLOWLISTED_SPAN, SYNTHESIZED_TEXT

# ---------------------------------------------------------------------------
# Surfaces that are not JSONL records.
#
# Discovery was `rglob("*.jsonl")` until 2026-08-15, so seven committed files
# were never opened: three corpus READMEs, three JSON label/measurement files,
# and one golden normalizer output that is literally `HUMAN:` / `AGENT:`
# transcript lines. They were clean, but clean *by derivation* from the linted
# `.jsonl` sources — an argument, not a gate. `tests/fixtures/eval/labels.json`
# even asserted in its own `$comment` that "fixture-lint's allowlist applies
# tree-wide", which was false, sitting inside the corpus the gate is meant to
# cover, on a public remote.
#
# Everything under the corpus root is now checked, under one of three regimes,
# and an extension no regime claims is a rejection rather than a skip.
# ---------------------------------------------------------------------------

#: Characters ordinary English prose does not use, and which therefore mark a
#: quoted string as a structural token — a field path, a JSON literal, a
#: comparison, a filename — rather than something a person said.
#:
#: `.` `-` `?` `!` `,` `'` `(` `)` are deliberately absent. Prose uses all of
#: them, so admitting them would make this a heuristic that passes real
#: sentences, which is exactly what `SYNTHESIZED_TEXT`'s "equality, not
#: pattern" comment warns against. The price is that `palaver fixture-lint`
#: and `palaver diagnose --coverage` contain no structural character at all;
#: those are covered by `COMMAND_SPAN` rather than by widening this set,
#: because "it is a documented command line" is a different claim from "it is
#: not a sentence" and should not be smuggled in as one.
STRUCTURAL_CHARACTERS = frozenset('_:=/<>{}[]"|*\\')

#: How many words a structural-character span may run to before it stops being
#: a token and starts being a sentence.
#:
#: Without this the rule is exactly the heuristic `SYNTHESIZED_TEXT` warns
#: about, and it failed on the first real file it met: the 80-word `$comment`
#: in `tests/fixtures/eval/labels.json` passed as "structural" because the
#: paragraph happens to contain a `/`. Any prose mentioning a path or a
#: `key: value` pair would have done the same.
#:
#: Six is twice the observed maximum. Every span in this corpus that actually
#: relies on the structural rule is three words or fewer
#: (`message.data.finish == "stop"`, `session_meta.payload.{id, session_id}`),
#: and the longest legitimate multi-word quotes are command lines, which
#: `COMMAND_SPAN` handles separately. The gap between 3 and 80 is what makes a
#: cap here decisive rather than arbitrary.
STRUCTURAL_MAX_WORDS = 6

#: A documented `palaver` invocation. The corpus READMEs quote commands a
#: reader is meant to run, and those are neither phrasebook utterances nor
#: structural tokens. Anchored and restricted to lowercase subcommand and flag
#: shapes so it cannot be satisfied by a sentence that happens to open with
#: the word "palaver".
#: Every token after the subcommand must be a flag or a path. An earlier
#: version allowed any lowercase word there, which made "palaver watches the
#: sessions you run" a valid command line — a sentence passing as a command is
#: the same failure as a sentence passing as a structural token, and its own
#: test caught it.
COMMAND_SPAN = re.compile(
    r"^(?:uv run )?palaver [a-z][a-z-]*"
    r"(?:\s(?:--[a-z][a-z-]*|[a-z0-9_*-]*[./][a-z0-9_.*/-]*))*$"
)

#: The channel label on a line of normalizer golden output — `HUMAN`, `AGENT`,
#: `SYSTEM(compaction)`, `result`. Matched by shape rather than against a
#: hardcoded vocabulary so the linter does not have to be edited every time
#: `palaver.extract.normalize` learns a new record kind; the *text* half of the
#: line is what carries session content, and that is phrasebook-checked.
GOLDEN_LINE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z_]*(?:\([a-z_]+\))?)(?:: |> )(?P<text>.*)$")

#: Markers only a real session store produces. This is a marker scan rather
#: than an allowlist, and that is a real difference in strength: it answers
#: "does this carry an artefact of a real machine", not "was every word of
#: this invented". It is the only check applied to authored narrative, because
#: a 17 KB README cannot be equality-checked against anything but itself —
#: every other surface gets the allowlist as well.
PROVENANCE_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a bare UUID, which is the shape of a real Claude Code session id",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
    ),
    ("an opaque tool-use id from a real transcript", re.compile(r"\btoolu_[A-Za-z0-9]{6,}")),
    ("an opaque message id from a real transcript", re.compile(r"\bmsg_[A-Za-z0-9]{10,}")),
    (
        "an absolute home path, which names a real machine and a real user",
        re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+"),
    ),
    ("an email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "a Claude Code project directory key, which encodes a real path",
        re.compile(r"-(?:Users|home)-[A-Za-z0-9]+-"),
    ),
)


def provenance_markers(text: str) -> tuple[str, ...]:
    """Every real-store marker `text` carries, as human-readable reasons.

    Args:
        text: Any string from any corpus surface.

    Returns:
        One reason per marker found, empty when the text is clean.
    """
    return tuple(reason for reason, pattern in PROVENANCE_MARKERS if pattern.search(text))


def check_span(span: str, where: str, *, extra: frozenset[str] = frozenset()) -> Verdict:
    """Judge one quoted string from a non-record surface.

    Accepts, in order: phrasebook or annotation text by equality, a documented
    command line, a single whitespace-free token (an identifier by shape — a
    field name, filename, or path cannot be a sentence), and a multi-word
    string carrying a structural character. Everything else is a rejection.

    The marker scan runs first and applies to all of them, so a span that
    would otherwise pass as a harmless identifier still fails if it is a real
    session's UUID.

    Args:
        span: The string to judge.
        where: Human-readable location, used only in the rejection detail.
        extra: An additional equality allowlist for surfaces that carry a
            content class the phrasebook does not cover. Passed only by
            `lint_data_file`, and only `EXTRACTION_TEXT` — a parameter rather
            than a module-level union so that widening it for one surface
            cannot widen it for every surface.

    Returns:
        `ACCEPTED`, or a rejecting `Verdict` naming the rule.
    """
    stripped = span.strip()
    if not stripped:
        return ACCEPTED
    markers = provenance_markers(stripped)
    if markers:
        return _reject(RULE_PROVENANCE_MARKER, f"{where} carries {markers[0]}: {stripped[:40]!r}")
    if stripped in SYNTHESIZED_TEXT or stripped in ANNOTATION_TEXT or stripped in extra:
        return ACCEPTED
    if COMMAND_SPAN.match(stripped):
        return ACCEPTED
    if not any(character.isspace() for character in stripped):
        return ACCEPTED
    if STRUCTURAL_CHARACTERS & set(stripped) and len(stripped.split()) <= STRUCTURAL_MAX_WORDS:
        return ACCEPTED
    return _reject(
        RULE_UNALLOWLISTED_SPAN,
        f"{where} is neither phrasebook text nor a structural token: {stripped[:40]!r}",
    )
