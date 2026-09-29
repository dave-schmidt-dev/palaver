"""
INVARIANTS.md parses into complete blocks whose gate_test and area entries resolve to
real code, and ledger.yaml tracks exactly the charter's invariants.
"""

from __future__ import annotations

import ast
import glob as globlib
import re
from pathlib import Path

import pytest

from tests._test_invariants_support import PALAVER_ROOT

REPO_ROOT = PALAVER_ROOT.parent
CHARTER_PATH = REPO_ROOT / "INVARIANTS.md"

# Mirrors the field grammar in `~/.agent/harvest/charter.py` — the charter is
# not just prose, it is input to `harvest`, which maps HISTORY bug entries to
# invariants through `area:` and computes the foundation-freeze verdict from
# `threshold:`. A field this file gets wrong is a verdict that tool gets wrong.
CHARTER_HEADING = re.compile(r"^### (?P<id>INV-\d+) — ")
CHARTER_FIELD = re.compile(
    r"^(?P<key>area|gate_test|threshold|rationale|always_active): *(?P<val>.+?) *$"
)


# --------------------------------------------------------------------------
# The charter itself. Every test above enforces one invariant; nothing until
# now enforced that `INVARIANTS.md` still describes the code it governs.
#
# It had drifted. Three `area:` entries named files that were never written —
# `palaver/memory/provenance.py`, `palaver/memory/schema.py`,
# `palaver/observer/state.py` — because the charter was authored on
# 2026-08-14 before implementation and the modules landed under other names.
# That is not a documentation nit: `area:` is how `harvest` maps a HISTORY
# bug entry to an invariant, so INV-5's area matched nothing and no bug
# against it could ever count toward its recurrence threshold. The invariant
# read as clean because it was unreachable, which is the same failure the
# `*.jsonl` glob produced in `fixture-lint` and the same one a gate test
# naming a deleted function would produce here.
# --------------------------------------------------------------------------


def parse_charter(text: str) -> dict[str, dict[str, list[str]]]:
    """Split `INVARIANTS.md` into one field bag per `### INV-N` heading.

    Collects repeated keys into a list rather than overwriting. `harvest`'s
    own parser keeps only the last value per key, which is why INV-9's
    multiple `gate_test:` lines are invisible to it; that is a limitation of
    a tool this repo does not own, and the reason this parser is written
    separately instead of imported.

    Args:
        text: Full contents of the charter.

    Returns:
        Mapping of invariant id to a mapping of field name to every value
        given for that field, in file order.
    """
    blocks: dict[str, dict[str, list[str]]] = {}
    current: dict[str, list[str]] | None = None
    for line in text.splitlines():
        heading = CHARTER_HEADING.match(line)
        if heading:
            current = blocks.setdefault(heading.group("id"), {})
            continue
        if current is None:
            continue
        field = CHARTER_FIELD.match(line)
        if field:
            current.setdefault(field.group("key"), []).append(field.group("val"))
    return blocks


def _parse_area_globs(raw: str) -> list[str]:
    """Read one `area:` line's JSON-ish list into its component globs."""
    return [piece.strip().strip('"') for piece in raw.strip("[]").split(",") if piece.strip()]


def _defined_test_names(path: Path) -> set[str]:
    """Every function name defined at module level or inside a class in `path`.

    AST rather than a `def <name>(` text search, matching the discipline
    `count_http_client_references` already applies: a name inside a comment,
    a docstring, or a nested closure is not a collectible test, and a grep
    cannot tell the difference.

    Args:
        path: Python source file to parse.

    Returns:
        Set of function names pytest could collect from that file.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.update(
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            )
    return names


CHARTER = parse_charter(CHARTER_PATH.read_text(encoding="utf-8"))
CHARTER_IDS = sorted(CHARTER, key=lambda name: int(name.removeprefix("INV-")))


def test_the_charter_parses_into_invariant_blocks():
    """Positive control: the parser finds blocks at all.

    Without this, every parametrized test below would collect zero cases and
    the whole section would pass vacuously — a heading style change in the
    charter would silently disarm its own enforcement.
    """
    assert len(CHARTER_IDS) >= 9
    assert CHARTER_IDS[0] == "INV-1"


@pytest.mark.parametrize("invariant", CHARTER_IDS, ids=CHARTER_IDS)
def test_every_invariant_carries_the_fields_harvest_reads(invariant):
    """Done-when: each block has `area:`, `gate_test:`, and an integer `threshold:`.

    A block missing `area:` is invisible to harvest's mapping; one missing
    `threshold:` silently takes the default 3 rather than the value its
    author intended.
    """
    fields = CHARTER[invariant]
    assert fields.get("area"), f"{invariant} has no area: glob"
    assert fields.get("gate_test"), f"{invariant} names no gate_test:"
    assert len(fields.get("threshold", [])) == 1, f"{invariant} needs exactly one threshold:"
    assert int(fields["threshold"][0]) >= 1


@pytest.mark.parametrize("invariant", CHARTER_IDS, ids=CHARTER_IDS)
def test_every_gate_test_named_by_the_charter_exists(invariant):
    """Done-when: every `gate_test:` resolves to a real, collectible test.

    A renamed or deleted test leaves the charter naming an enforcement that
    is not there, and the suite stays green because nothing reads the
    charter. This is the test that reads it.
    """
    for reference in CHARTER[invariant]["gate_test"]:
        relative, _, name = reference.partition("::")
        path = REPO_ROOT / relative
        assert path.is_file(), f"{invariant} names a missing file: {relative}"
        assert name, f"{invariant} names a file with no test: {reference}"
        assert name in _defined_test_names(path), f"{invariant} names a missing test: {reference}"


@pytest.mark.parametrize("invariant", CHARTER_IDS, ids=CHARTER_IDS)
def test_every_area_glob_matches_something_on_disk(invariant):
    """Done-when: each `area:` glob matches at least one existing path.

    Asserts presence, never a count — `palaver/**/*.py` matches 60-odd files
    today and that number is meant to move. What must not happen is a glob
    matching zero, which reads in `harvest` output as an invariant nothing
    can violate rather than as a broken pattern.
    """
    for line in CHARTER[invariant]["area"]:
        for pattern in _parse_area_globs(line):
            matches = globlib.glob(pattern, root_dir=REPO_ROOT, recursive=True)
            assert matches, f"{invariant} area glob matches nothing: {pattern}"


def test_the_area_glob_check_is_not_vacuous():
    """Negative control: the same matcher returns empty for a path that is gone.

    Named for the file the real charter used to point at. Without this, a
    `glob` call that silently returned a non-empty list for everything would
    leave the test above passing over a charter full of phantom paths.
    """
    assert not globlib.glob("palaver/memory/provenance.py", root_dir=REPO_ROOT, recursive=True)
    assert globlib.glob("palaver/store/schema.py", root_dir=REPO_ROOT, recursive=True)


def test_the_gate_test_check_reads_definitions_not_text():
    """Negative control: a name that appears only as text is not a definition.

    `_defined_test_names` is the reason the charter cannot be satisfied by a
    test name mentioned in a docstring or comment. This file mentions
    `test_status_is_never_model_supplied` in prose; it is defined in
    `tests/test_signals.py`, and the checker must not find it here.
    """
    here = _defined_test_names(Path(__file__))
    assert "test_status_is_never_model_supplied" not in here
    assert "test_every_area_glob_matches_something_on_disk" in here


# =============================================================================
# The ledger tracks the charter
#
# `ledger.yaml` is the other half of the machine state `harvest` reads: the
# charter says what the invariants are, the ledger says what state each is in.
# The failure this guards is quiet — an INV-10 added to the charter with no
# ledger entry is not an error to `harvest`, it is an invariant whose
# `gate_test_status` is simply unknown, reported as nothing at all.
#
# Only that one correspondence is asserted. `resolutions` and `maturity` are
# owner decisions the ledger exists to record, and a test that failed when
# `maturity.status` flipped to `mvp` would fire on the correct action.
#
# Parsed by regex rather than `yaml.safe_load` because pyyaml is not a
# dependency of this project and adding one for a single assertion is a worse
# trade than hand-parsing a flat block — the same call already made for
# `INVARIANTS.md` above. The cost is that this does not prove the file is
# well-formed YAML; `harvest` raises on read if it is not, which is loud.
# =============================================================================

LEDGER_PATH = REPO_ROOT / "ledger.yaml"
LEDGER_BLOCK_KEY = re.compile(r"^(?P<key>\w+):\s*$")
LEDGER_INVARIANT_KEY = re.compile(r"^\s+(?P<id>INV-[\w.-]+):")


def _ledger_invariant_ids(text: str) -> set[str]:
    """The `INV-*` keys nested under the ledger's `invariants:` block.

    Scoped to that block on purpose: `resolutions:` carries the same key
    shape, so an unscoped scan would report an invariant as present in the
    ledger on the strength of a resolution entry alone.
    """
    ids: set[str] = set()
    in_block = False
    for line in text.splitlines():
        block = LEDGER_BLOCK_KEY.match(line)
        if block:
            in_block = block.group("key") == "invariants"
            continue
        if not in_block:
            continue
        key = LEDGER_INVARIANT_KEY.match(line)
        if key:
            ids.add(key.group("id"))
    return ids


def test_the_ledger_carries_an_entry_for_every_charter_invariant():
    """Done-when: `ledger.yaml`'s `invariants:` keys equal the charter's.

    Equality both ways. A charter invariant missing from the ledger has no
    tracked state; a ledger entry naming an invariant the charter dropped is
    state for something that no longer exists.
    """
    assert LEDGER_PATH.is_file(), "ledger.yaml is missing — harvest has no state to read"
    ledger_ids = _ledger_invariant_ids(LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger_ids == set(CHARTER_IDS), (
        f"ledger.yaml and INVARIANTS.md disagree: {ledger_ids ^ set(CHARTER_IDS)}"
    )


def test_the_ledger_scan_is_scoped_to_the_invariants_block():
    """Negative control: keys under `resolutions:` are not counted as entries.

    Without the block scope the check above would pass on a ledger that
    tracked an invariant's resolution date and nothing else, which is the
    field it is not supposed to be reading.
    """
    text = (
        "invariants:\n"
        "  INV-1: {gate_test_status: covered}\n"
        "resolutions:\n"
        '  INV-2: {resolved_at_date: "2026-08-15"}\n'
        "maturity:\n"
        "  status: pre-mvp\n"
    )
    assert _ledger_invariant_ids(text) == {"INV-1"}
