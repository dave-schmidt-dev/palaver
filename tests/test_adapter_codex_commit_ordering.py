"""
Git-history proof that the blind labels were committed before the measurement that
scores against them.
"""

import subprocess
from pathlib import Path

import pytest

from palaver.ingest.adapters.codex import (
    LABELS_PATH,
    MEASUREMENT_PATH,
)
from tests._adapter_codex_support import REPO_ROOT

# --- commit ordering: the labels were authored blind ------------------------


#: Subcommands that write. They are legitimate against the throwaway
#: repositories the ordering positive-control builds under `tmp_path`, and
#: never against the repository under test.
_MUTATING_GIT_SUBCOMMANDS = frozenset({"init", "config", "add", "commit", "checkout", "reset"})


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Run a git command and return its stripped stdout.

    Guards the repository under test: a mutating subcommand is refused
    outright when `cwd` is `REPO_ROOT`. The ordering positive-control needs
    real commits to have a history to check, so it builds them in a
    `tmp_path` repository — and a missing `cwd=` argument there would
    otherwise silently commit into the developer's own working tree.
    """
    if args and args[0] in _MUTATING_GIT_SUBCOMMANDS and Path(cwd) == REPO_ROOT:
        raise AssertionError(
            f"refusing to run mutating `git {args[0]}` against the repository "
            f"under test; pass cwd=<tmp_path repo>"
        )
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _adding_commit(path: Path, cwd: Path = REPO_ROOT) -> str | None:
    """Return the single commit that added `path`, or `None` if untracked.

    `tests/fixtures/labels/` is a fresh path with no prior history — the
    earlier flat-root labels commit was reverted and never used this
    directory — so `--diff-filter=A` matches exactly one commit and there is
    no "which add did they mean" ambiguity to resolve. That is asserted
    rather than assumed: a path with two adds would make the ordering claim
    below depend on which one the helper happened to pick.
    """
    relative = path.relative_to(cwd).as_posix()
    out = _git("log", "--diff-filter=A", "--format=%H", "--", relative, cwd=cwd)
    commits = out.split()
    assert len(commits) <= 1, (
        f"{relative} was added in {len(commits)} commits ({commits}); the "
        f"ordering proof needs an unambiguous add"
    )
    return commits[0] if commits else None


def _path_exists_at(commit: str, relative: str, cwd: Path = REPO_ROOT) -> bool:
    """Whether `relative` exists in the tree of `commit`."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative}"],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _is_ancestor(earlier: str, later: str, cwd: Path = REPO_ROOT) -> bool:
    """Whether `earlier` is a strict ancestor of `later`."""
    if earlier == later:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", earlier, later],
        cwd=cwd,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_labels_were_committed_before_the_measurement():
    """Done-when: the labels file must be the earlier commit.

    Ancestry, not timestamp comparison. Git commit timestamps have
    second resolution, so two commits made moments apart can tie and a
    strict `labels_time < measurement_time` assertion would flake; ancestry
    is exact and is what "committed first" actually means on a linear
    history.

    This is the ordering that makes the measurement meaningful. Labels
    written after seeing the classifier's output would validate the heuristic
    against labels the heuristic produced.
    """
    labels_commit = _adding_commit(LABELS_PATH)
    measurement_commit = _adding_commit(MEASUREMENT_PATH)

    if measurement_commit is None and labels_commit is None:
        pytest.skip(
            "neither artifact is committed yet; the orchestrator commits the "
            "labels alone first, then the measurement"
        )
    assert labels_commit is not None, (
        "the measurement is committed but the labels are not: the ordering "
        "proof has nothing to stand on"
    )
    if measurement_commit is None:
        pytest.skip("the measurement file is not committed yet")

    assert labels_commit != measurement_commit, (
        "labels and measurement landed in one commit, so nothing evidences "
        "that the labels predate the classifier"
    )
    assert _is_ancestor(labels_commit, measurement_commit), (
        f"labels commit {labels_commit} is not an ancestor of measurement "
        f"commit {measurement_commit}"
    )
    # Ordering shows order; absence is what carries the blindness claim. If
    # the classifier did not exist in the tree when the labels landed, it
    # cannot have informed them — no argument about intent required.
    assert not _path_exists_at(labels_commit, "palaver/ingest/adapters/codex.py"), (
        f"codex.py exists at the labels commit {labels_commit}, so the labels "
        f"could have been written against the classifier's output"
    )


def test_the_ordering_check_fails_on_a_wrong_order_history(tmp_path):
    """Positive control: the ordering check is live, and it is not vacuous.

    The real ordering test above skips while the artifacts are uncommitted,
    which is exactly when a broken check would go unnoticed. This builds a
    throwaway repository whose history has the two files in the *wrong*
    order and requires the same helpers to detect it — then rebuilds it in
    the right order and requires them to accept it.
    """
    repo = tmp_path / "repo"
    (repo / "tests" / "fixtures" / "labels").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "fixture@example.invalid", cwd=repo)
    _git("config", "user.name", "Fixture", cwd=repo)

    measurement = repo / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    labels = repo / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"

    # Wrong order: the measurement lands first.
    measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "measurement first", cwd=repo)
    labels.write_text("{}\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "labels second", cwd=repo)

    labels_commit = _adding_commit(labels, cwd=repo)
    measurement_commit = _adding_commit(measurement, cwd=repo)
    assert labels_commit and measurement_commit
    assert not _is_ancestor(labels_commit, measurement_commit, cwd=repo), (
        "the ordering check accepted a history where the labels were "
        "committed after the measurement"
    )
    # ...and the violation is detectable in the direction that matters.
    assert _is_ancestor(measurement_commit, labels_commit, cwd=repo)

    # Right order, in a second repository: the helpers accept it.
    good = tmp_path / "good"
    (good / "tests" / "fixtures" / "labels").mkdir(parents=True)
    _git("init", "-q", cwd=good)
    _git("config", "user.email", "fixture@example.invalid", cwd=good)
    _git("config", "user.name", "Fixture", cwd=good)
    good_labels = good / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"
    good_measurement = good / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    good_labels.write_text("{}\n", encoding="utf-8")
    _git("add", "-A", cwd=good)
    _git("commit", "-qm", "labels first", cwd=good)
    good_measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=good)
    _git("commit", "-qm", "measurement second", cwd=good)

    assert _is_ancestor(
        _adding_commit(good_labels, cwd=good),
        _adding_commit(good_measurement, cwd=good),
        cwd=good,
    )
    assert not _path_exists_at(
        _adding_commit(good_labels, cwd=good),
        "palaver/ingest/adapters/codex.py",
        cwd=good,
    )


def test_the_absence_check_catches_a_classifier_present_at_the_labels_commit(tmp_path):
    """Positive control for the absence assertion specifically.

    The wrong-order control above cannot exercise this: a history that fails
    on ordering never reaches the absence check, and a correctly-ordered
    history passes both. So this builds the one history where ordering is
    satisfied but the property is still violated — labels and classifier
    landing in the *same* commit, measurement after. Ordering says that is
    fine. Only absence catches it, which is why absence is the assertion the
    blindness claim actually rests on.
    """
    repo = tmp_path / "same-commit"
    (repo / "tests" / "fixtures" / "labels").mkdir(parents=True)
    (repo / "palaver" / "ingest" / "adapters").mkdir(parents=True)
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "fixture@example.invalid", cwd=repo)
    _git("config", "user.name", "Fixture", cwd=repo)

    labels = repo / "tests" / "fixtures" / "labels" / "codex-role-labels.jsonl"
    measurement = repo / "tests" / "fixtures" / "labels" / "codex-role-class-measurement.json"
    classifier = repo / "palaver" / "ingest" / "adapters" / "codex.py"

    labels.write_text("{}\n", encoding="utf-8")
    classifier.write_text("# a classifier that already existed\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "labels and classifier together", cwd=repo)
    measurement.write_text("{}", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "measurement second", cwd=repo)

    labels_commit = _adding_commit(labels, cwd=repo)
    measurement_commit = _adding_commit(measurement, cwd=repo)

    # Ordering is satisfied, so it cannot be what rejects this history.
    assert labels_commit != measurement_commit
    assert _is_ancestor(labels_commit, measurement_commit, cwd=repo)

    # The absence check is the one that fires.
    assert _path_exists_at(labels_commit, "palaver/ingest/adapters/codex.py", cwd=repo), (
        "the absence check failed to see a classifier that was committed alongside the labels"
    )


def test_adding_commit_reports_none_for_an_untracked_path(tmp_path):
    """The untracked branch of `_adding_commit` is exercised, not assumed."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    untracked = repo / "untracked.json"
    untracked.write_text("{}", encoding="utf-8")
    assert _adding_commit(untracked, cwd=repo) is None
