"""Palaver Python sources that static tests read, whatever their file layout.

A module is named by its repository path without ``.py``: ``"palaver/observer/signals"``.
Unsplit, it is one file (``palaver/observer/signals.py``). Split, it is a package
(``palaver/observer/signals/__init__.py`` and its submodules). Every reader returns the
whole module in either layout, so a check written against the single file keeps covering
every line after a split. A module with no files raises ``FileNotFoundError``; no reader
returns part of a module.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module_files(module: str) -> list[Path]:
    """Every file of ``module`` in path order: the single file, or the package's files."""
    single = ROOT / f"{module}.py"
    if single.is_file():
        return [single]
    package = ROOT / module
    if (package / "__init__.py").is_file():
        return sorted(package.rglob("*.py"))
    raise FileNotFoundError(f"no source files for {module}")


def module_text(module: str) -> str:
    """The module's source as one text, its files joined in path order."""
    return "\n".join(path.read_text(encoding="utf-8") for path in module_files(module))


def _inside(name: str, dotted: str) -> bool:
    return name == dotted or name.startswith(f"{dotted}.")


def imported_modules(module: str) -> set[str]:
    """The full dotted name of every module that ``module`` imports from outside itself.

    Relative imports are resolved against the importing file's package, so
    ``from ..extract import client`` counts as ``palaver.extract``. An import that lands
    inside ``module`` itself (split submodules importing each other, or a facade alias
    such as ``from palaver.observer import signals``) is not counted: a single-file
    module reads its own names without importing them. For a single file this is the
    set its import statements name, exactly as written.
    """
    dotted = module.replace("/", ".")
    found: set[str] = set()
    for path in module_files(module):
        package = list(path.relative_to(ROOT).parts[:-1])
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if not _inside(a.name, dotted))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level - 1 > len(package):
                        raise ValueError(f"{path}: relative import beyond the repository")
                    base = package[: len(package) - (node.level - 1)]
                    name = ".".join([*base, *([node.module] if node.module else [])])
                elif node.module:
                    name = node.module
                else:
                    continue
                if _inside(name, dotted):
                    continue
                if all(_inside(f"{name}.{a.name}", dotted) for a in node.names):
                    continue
                found.add(name)
    return found
