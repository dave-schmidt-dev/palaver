"""
INV-2: no module under palaver/ingest/adapters/ other than base.py opens a file
directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._test_invariants_support import PALAVER_ROOT

ADAPTERS_DIR = PALAVER_ROOT / "ingest" / "adapters"


# =============================================================================
# INV-2 (chokepoint) — every adapter read goes through open_source_readonly
# =============================================================================


def _direct_open_call_lines(path: Path) -> list[int]:
    """Line numbers of direct `open(...)` or `os.open(...)` calls in `path`.

    A structural AST-`Call`-node check, not a text grep — a call spread
    across lines or preceded by unrelated text is still found, and a
    substring `"open"` appearing inside a string literal or a comment is
    not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            hits.append(node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.inv2
def test_adapters_route_every_read_through_the_chokepoint(tmp_path):
    """No module under `palaver/ingest/adapters/` other than `base.py` opens a file directly.

    `base.py` owns `open_source_readonly`, the one chokepoint every adapter
    read must go through (INV-2's docstring, `palaver/ingest/adapters/base.py`).
    A sibling module opening a file itself could quietly request write
    access from a path `open_source_readonly`'s own tests never see.
    """
    scanned = sorted(ADAPTERS_DIR.rglob("*.py"))
    # Enumeration itself is part of the contract: an empty or wrong-directory
    # scan would make `violations == {}` pass vacuously. `rglob`, not `glob`,
    # so a future subpackage under `adapters/` (e.g. task 7.2's OpenCode
    # adapter) is not silently skipped the way a flat `glob` would skip it.
    assert any(path.name == "claude_code.py" for path in scanned)

    violations = {}
    for path in scanned:
        if path.name == "base.py":
            continue
        hits = _direct_open_call_lines(path)
        if hits:
            violations[path.name] = hits
    assert violations == {}

    # Positive control: prove the detector is live by pointing it at modules
    # that do call open()/os.open() directly.
    poisoned_open = tmp_path / "poisoned_open_adapter.py"
    poisoned_open.write_text(
        "def read_it(path):\n    with open(path) as f:\n        return f.read()\n"
    )
    assert _direct_open_call_lines(poisoned_open) == [2]

    poisoned_os_open = tmp_path / "poisoned_os_open_adapter.py"
    poisoned_os_open.write_text(
        "import os\n\ndef read_it(path):\n    return os.open(path, os.O_RDONLY)\n"
    )
    assert _direct_open_call_lines(poisoned_os_open) == [4]
