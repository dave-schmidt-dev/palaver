"""Tests for `palaver fixture-lint` and the ground truth of the fixture corpus.

Two things are defended here, and they fail in opposite directions.

**The linter (INV-9, git clause).** `tests/fixtures/` is committed to a public
remote, so a record that reaches it has left the machine irrecoverably. The
linter is an allowlist and its failure mode is *silent acceptance*, which no
amount of "the corpus passes" can rule out. So the tests that matter here are
the negative ones, and each poisons a record in exactly **one** dimension and
asserts on the *rule name* the linter reported — a poisoned record that fails
because three rules fired at once proves nothing about any of them. Above
them sits `test_poisoned_record_rejection_comes_from_the_classifier`, which
stubs `classify_record` to accept everything and requires the same run to exit
0: without it, a linter that rejected every path it was handed would pass the
whole negative suite.

`test_committed_corpus_passes_the_linter` proves the corpus and the linter
agree with each other. It does not prove the corpus is safe; the negative
tests are what prove that.

**The ground truth (accuracy).** Coverage counts the sessions a signal was
determinable for and a uniformly wrong classifier scores 100% at it. These
tests assert derived status against labels in `tests/fixtures/README.md`, and
assert that those labels state checkable structural facts rather than
authorial intent — "constructed to be WORKING" is circular and is rejected.

Ground truth and the derived value are tracked as separate columns on
purpose, and `KNOWN_DIVERGENCES` is currently empty: an unresolved
`AskUserQuestion` used to be a session blocked on its human that Phase 1
reported as WORKING, until `derive_turn_boundary` started reading the
`tool_use` block's `name` (task 4). The divergence set is still asserted to
be *exactly* `KNOWN_DIVERGENCES` — now the empty set — rather than dropped,
so a future regression that makes any fixture's derived status stop matching
its ground truth fails loudly here instead of being silently absorbed.

No real session store (`~/.claude/`, `~/.codex/`,
`~/.local/share/opencode/`) is opened, globbed, or read by this module or by
anything under `tests/fixtures/`. Every poisoned record is written under
pytest's `tmp_path` and every string in it was invented for the test.
"""

import json
from pathlib import Path

from palaver.cli import main

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"


# --- poisoned-corpus helpers -------------------------------------------------


def _corpus(tmp_path: Path, records: list[dict], *, terminated: bool = True) -> Path:
    """Write a one-file corpus under `tmp_path` and return its directory."""
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    body = b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)
    if not terminated:
        body = body.rstrip(b"\n")
    (root / "poisoned.jsonl").write_bytes(body)
    return root


def _lint(root: Path) -> int:
    return main(["fixture-lint", str(root)])
