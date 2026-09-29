"""Warn above the source-size target and enforce the commit ceiling."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

TARGET = 500
MAX_LINES = 800
EXCEPTIONS = ".file-size-exceptions"
SUFFIXES = (".py", ".sh")


def line_count(contents: bytes) -> int:
    """Count byte lines, including a final line without a newline."""
    return contents.count(b"\n") + int(bool(contents) and not contents.endswith(b"\n"))


def checked_path(path: str) -> bool:
    """Return whether a path is source covered by this policy."""
    return path.endswith(SUFFIXES)


def git_output(*arguments: str) -> bytes:
    """Run Git in the current repository and return its raw output."""
    result = subprocess.run(["git", *arguments], capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(detail or f"git {' '.join(arguments)} failed")
    return result.stdout


def paths_from_git(*arguments: str) -> list[str]:
    """Decode NUL-separated Git paths without losing spaces or newlines."""
    return [os.fsdecode(path) for path in git_output(*arguments).split(b"\0") if path]


def parse_exceptions(contents: bytes, source: str) -> tuple[dict[str, str], list[str]]:
    """Parse reasoned exception entries and report every format error."""
    try:
        lines = contents.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return {}, [f"{source}: exceptions file must be UTF-8"]

    entries: dict[str, str] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        path = fields[0]
        if path in seen:
            errors.append(f"{source}:{number}: duplicate exception path {path}")
            continue
        seen.add(path)
        if len(fields) == 1 or not fields[1].strip():
            errors.append(f"{source}:{number}: exception entry needs a reason")
        elif fields[1].split(maxsplit=1)[0].isdigit():
            errors.append(
                f"{source}:{number}: line caps are no longer supported; remove the cap"
            )
        else:
            entries[path] = fields[1].strip()
    return entries, errors


def working_exceptions(path: str) -> tuple[dict[str, str], list[str]]:
    """Load the working-tree exceptions file, if one exists."""
    try:
        return parse_exceptions(Path(path).read_bytes(), path)
    except FileNotFoundError:
        return {}, []
    except OSError as error:
        return {}, [f"{path}: {error}"]


def staged_exceptions() -> tuple[dict[str, str], list[str]]:
    """Load exceptions from the index being committed, if present."""
    if EXCEPTIONS not in paths_from_git("ls-files", "-z", "--", EXCEPTIONS):
        return {}, []
    contents = git_output("cat-file", "blob", f":{EXCEPTIONS}")
    return parse_exceptions(contents, EXCEPTIONS)


def regular_file_contents(path: str) -> bytes | None:
    """Read a working-tree regular file, skipping missing files and links."""
    file = Path(path)
    try:
        if not stat.S_ISREG(file.lstat().st_mode):
            return None
        return file.read_bytes()
    except FileNotFoundError:
        return None


def inspect_files(
    paths: list[str],
    exceptions: dict[str, str],
    target: int,
    maximum: int,
    exceptions_path: str,
    *,
    staged: bool,
    legacy_notice_paths: set[str],
) -> list[str]:
    """Print advisory notes and return hard-limit or file-read errors."""
    errors: list[str] = []
    for path in paths:
        if not checked_path(path):
            continue
        try:
            contents = (
                git_output("cat-file", "blob", f":{path}")
                if staged
                else regular_file_contents(path)
            )
        except (RuntimeError, OSError) as error:
            errors.append(
                f"{path}: unable to read {'staged blob' if staged else 'file'}: {error}"
            )
            continue
        if contents is None:
            continue
        count = line_count(contents)
        if target < count <= maximum:
            print(
                f"file-size: {path} has {count} lines (target {target}); "
                "split it when a clean seam exists"
            )
        if count > maximum and path not in exceptions:
            errors.append(
                f"file-size: {path} has {count} lines (maximum {maximum}); "
                f"add a reasoned entry to {exceptions_path}"
            )
        elif (
            count > maximum
            and exceptions.get(path, "").startswith("legacy ")
            and path in legacy_notice_paths
        ):
            print(
                f"file-size: {path} is a legacy exception ({count} lines); "
                "extract a clean seam from it in this piece of work"
            )
        elif path in exceptions and count <= maximum:
            print(
                f"file-size: {path} has {count} lines (at or under {maximum}); "
                f"remove its exception from {exceptions_path}"
            )
    return errors


def parse_args(arguments: list[str]) -> argparse.Namespace:
    """Require exactly one file selection mode and positive limits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--max-lines", type=int, default=MAX_LINES)
    parser.add_argument("--exceptions", default=EXCEPTIONS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("files", metavar="FILE", nargs="*")
    options = parser.parse_args(arguments)
    if (options.staged or options.all) == bool(options.files):
        parser.error("provide exactly one of --staged, --all, or FILE arguments")
    if options.target <= 0 or options.max_lines <= 0:
        parser.error("line limits must be positive integers")
    return options


def main(arguments: list[str] | None = None) -> int:
    """Check selected source files and return zero only without errors."""
    options = parse_args(sys.argv[1:] if arguments is None else arguments)
    errors: list[str] = []
    try:
        if options.staged:
            changed = git_output("diff", "--cached", "--name-only", "--", EXCEPTIONS)
            staged_paths = paths_from_git(
                "diff",
                "--cached",
                "--name-only",
                "-z",
                "--diff-filter=ACMR",
            )
            paths = paths_from_git("ls-files", "-z") if changed else staged_paths
            legacy_notice_paths = set(staged_paths)
            exceptions, format_errors = staged_exceptions()
            exceptions_path = EXCEPTIONS
        else:
            paths = (
                paths_from_git("ls-files", "-z", "-co", "--exclude-standard")
                if options.all
                else options.files
            )
            legacy_notice_paths = set() if options.all else set(paths)
            exceptions, format_errors = working_exceptions(options.exceptions)
            exceptions_path = options.exceptions
        errors.extend(format_errors)
        errors.extend(
            inspect_files(
                paths,
                exceptions,
                options.target,
                options.max_lines,
                exceptions_path,
                staged=options.staged,
                legacy_notice_paths=legacy_notice_paths,
            )
        )
    except RuntimeError as error:
        errors.append(str(error))
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
