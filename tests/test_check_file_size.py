"""Behavioral tests for the staged and working-tree file-size policy."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_file_size

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> None:
    """Run a Git command in an isolated temporary repository."""
    subprocess.run(["git", *arguments], cwd=repo, env=env, check=True, capture_output=True)


def repo_at(path: Path) -> Path:
    """Create a local repository with a copy of the checker under test."""
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Palaver Test")
    (path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/check_file_size.py", path / "scripts/check_file_size.py")
    return path


def source(repo: Path, name: str, lines: int, *, final_newline: bool = True) -> Path:
    """Write a source file of a precise byte-line length."""
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"x\n" * (lines - int(not final_newline)) + (b"x" if not final_newline else b"")
    )
    return path


def check(
    repo: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the copied checker with the repository as its working directory."""
    return subprocess.run(
        [sys.executable, "scripts/check_file_size.py", *arguments],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("lines", "expected_code", "warning"),
    [(500, 0, False), (501, 0, True), (800, 0, True), (801, 1, False)],
)
def test_file_thresholds(tmp_path: Path, lines: int, expected_code: int, warning: bool) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "app.py", lines)
    result = check(repo, "app.py")
    assert result.returncode == expected_code
    assert ("target 500" in result.stdout) is warning
    if expected_code:
        assert "app.py has 801 lines" in result.stderr
    else:
        assert result.stderr == ""


def test_file_selection_and_unterminated_last_line(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "README.md", 900)
    source(repo, "metadata.json", 900)
    source(repo, "last.sh", 801, final_newline=False)
    assert check(repo, "README.md", "metadata.json", "missing.py").returncode == 0
    result = check(repo, "last.sh")
    assert result.returncode == 1
    assert "last.sh has 801 lines" in result.stderr


def test_reasoned_exceptions_and_removal_note(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "large.py", 801)
    source(repo, "small.py", 10)
    (repo / ".file-size-exceptions").write_text(
        "# reasoned exceptions\nlarge.py legacy source\nsmall.py now small\n"
    )
    result = check(repo, "large.py", "small.py")
    assert result.returncode == 0
    assert "remove its exception" in result.stdout
    assert "small.py has 10 lines" in result.stdout


def test_legacy_exception_notice_in_file_mode(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "old.py", 850)
    (repo / ".file-size-exceptions").write_text("old.py legacy source\n")

    result = check(repo, "old.py")

    assert result.returncode == 0
    assert result.stdout == (
        "file-size: old.py is a legacy exception (850 lines); "
        "extract a clean seam from it in this piece of work\n"
    )
    assert result.stderr == ""


def test_staged_legacy_exception_notice_only_for_changed_source_paths(
    tmp_path: Path,
) -> None:
    repo = committed_exception_repo(tmp_path / "repo")
    source(repo, "old.py", 851)
    git(repo, "add", "old.py")

    result = check(repo, "--staged")

    assert result.returncode == 0
    assert result.stdout == (
        "file-size: old.py is a legacy exception (851 lines); "
        "extract a clean seam from it in this piece of work\n"
    )
    assert result.stderr == ""


def test_exceptions_only_staged_does_not_notice_unchanged_legacy_file(
    tmp_path: Path,
) -> None:
    repo = committed_exception_repo(tmp_path / "repo")
    (repo / ".file-size-exceptions").write_text("# refreshed policy\nold.py legacy source\n")
    git(repo, "add", ".file-size-exceptions")

    result = check(repo, "--staged")

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_all_and_nonlegacy_exception_never_print_legacy_notice(
    tmp_path: Path,
) -> None:
    repo = committed_exception_repo(tmp_path / "repo")

    all_result = check(repo, "--all")
    assert all_result.returncode == 0
    assert all_result.stdout == ""

    (repo / ".file-size-exceptions").write_text("old.py retained generated interface\n")
    file_result = check(repo, "old.py")
    assert file_result.returncode == 0
    assert file_result.stdout == ""


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ("a.py\n", "needs a reason"),
        ("a.py 600 legacy\n", "line caps are no longer supported; remove the cap"),
        ("a.py reason\na.py another\n", "duplicate exception path"),
    ],
)
def test_exception_format_is_checked_without_named_files(
    tmp_path: Path, entry: str, error: str
) -> None:
    repo = repo_at(tmp_path / "repo")
    (repo / ".file-size-exceptions").write_text(entry)
    result = check(repo, "--all")
    assert result.returncode == 1
    assert error in result.stderr


def test_staged_blob_isolated_from_worktree_and_untracked_files(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "staged.sh", 800)
    git(repo, "add", "staged.sh")
    source(repo, "staged.sh", 801)
    source(repo, "untracked.py", 801)
    assert check(repo, "--staged").returncode == 0
    git(repo, "add", "staged.sh")
    result = check(repo, "--staged")
    assert result.returncode == 1
    assert "staged.sh has 801 lines" in result.stderr
    assert "untracked.py" not in result.stderr


def test_staged_deletion_skips_oversized_file(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "deleted.sh", 801)
    git(repo, "add", "deleted.sh")
    git(repo, "commit", "-qm", "oversized baseline")
    git(repo, "rm", "-q", "--", "deleted.sh")
    assert check(repo, "--staged").returncode == 0


def test_staged_mode_honors_alternate_index(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "base.txt", 1)
    git(repo, "add", "base.txt")
    git(repo, "commit", "-qm", "base")
    source(repo, "alternate.py", 801)
    alternate = os.environ.copy()
    alternate["GIT_INDEX_FILE"] = str(repo / "alternate-index")
    git(repo, "read-tree", "HEAD", env=alternate)
    git(repo, "add", "alternate.py", env=alternate)
    assert check(repo, "--staged").returncode == 0
    result = check(repo, "--staged", env=alternate)
    assert result.returncode == 1
    assert "alternate.py has 801 lines" in result.stderr


def committed_exception_repo(path: Path) -> Path:
    """Make a repository with an unchanged oversized file and valid entry."""
    repo = repo_at(path)
    source(repo, "old.py", 850)
    (repo / ".file-size-exceptions").write_text("old.py legacy source\n")
    git(repo, "add", "old.py", ".file-size-exceptions")
    git(repo, "commit", "-qm", "baseline")
    return repo


def test_staged_exception_removal_rechecks_unchanged_file(tmp_path: Path) -> None:
    repo = committed_exception_repo(tmp_path / "repo")
    (repo / ".file-size-exceptions").write_text("")
    git(repo, "add", ".file-size-exceptions")
    result = check(repo, "--staged")
    assert result.returncode == 1
    assert "old.py has 850 lines" in result.stderr


def test_staged_exception_deletion_rechecks_unchanged_file(tmp_path: Path) -> None:
    repo = committed_exception_repo(tmp_path / "repo")
    git(repo, "rm", "-q", ".file-size-exceptions")
    result = check(repo, "--staged")
    assert result.returncode == 1
    assert "old.py has 850 lines" in result.stderr


def test_staged_added_exception_passes(tmp_path: Path) -> None:
    repo = committed_exception_repo(tmp_path / "repo")
    source(repo, "new.py", 801)
    (repo / ".file-size-exceptions").write_text("old.py legacy source\nnew.py legacy source\n")
    git(repo, "add", "new.py", ".file-size-exceptions")
    assert check(repo, "--staged").returncode == 0


def test_all_sees_untracked_but_not_ignored_files(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    source(repo, "visible.py", 801)
    source(repo, "ignored.py", 801)
    (repo / ".gitignore").write_text("ignored.py\n")
    result = check(repo, "--all")
    assert result.returncode == 1
    assert "visible.py" in result.stderr
    assert "ignored.py" not in result.stderr


def test_usage_requires_one_mode(tmp_path: Path) -> None:
    repo = repo_at(tmp_path / "repo")
    assert check(repo).returncode == 2
    assert check(repo, "--all", "file.py").returncode == 2


def test_pre_commit_runs_size_check_before_ruff() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "- id: check-file-size" in config
    assert "- id: ruff-check\n" in config
    start = config.index("- id: check-file-size")
    assert start < config.index("- id: ruff-check\n")
    block = config[start : config.index("- id: ruff-check\n")]
    assert "entry: uv run python scripts/check_file_size.py --staged" in block
    assert "verbose: true" in block
    assert "stages: [pre-commit]" in block


def test_repository_audit_is_green() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_file_size.py", "--all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_imported_checker_counts_without_decoding() -> None:
    assert check_file_size.line_count(b"\xff\n\xfe") == 2
