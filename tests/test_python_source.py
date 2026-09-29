"""The layout-agnostic source readers in `tests/python_source.py` fail closed."""

from __future__ import annotations

import pytest

from tests import python_source


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(python_source, "ROOT", tmp_path)
    return tmp_path


def test_single_file_is_the_whole_module(root):
    _write(root, "palaver/mod.py", "A = 1\n")
    assert python_source.module_files("palaver/mod") == [root / "palaver/mod.py"]
    assert python_source.module_text("palaver/mod") == "A = 1\n"


def test_package_reads_every_submodule_in_path_order(root):
    _write(root, "palaver/mod/__init__.py", "from .b import B\n")
    _write(root, "palaver/mod/b.py", "B = 2\n")
    _write(root, "palaver/mod/a.py", "A = 1\n")
    files = python_source.module_files("palaver/mod")
    assert [path.name for path in files] == ["__init__.py", "a.py", "b.py"]
    text = python_source.module_text("palaver/mod")
    assert "A = 1" in text and "B = 2" in text


def test_a_missing_module_raises_instead_of_reading_nothing(root):
    with pytest.raises(FileNotFoundError):
        python_source.module_files("palaver/gone")
    with pytest.raises(FileNotFoundError):
        python_source.imported_modules("palaver/gone")


def test_single_file_imports_are_named_as_written(root):
    _write(
        root,
        "palaver/pkg/mod.py",
        "from __future__ import annotations\n"
        "import collections.abc\n"
        "from palaver.extract.persist import Extraction\n"
        "def late():\n"
        "    import json\n",
    )
    assert python_source.imported_modules("palaver/pkg/mod") == {
        "__future__",
        "collections.abc",
        "palaver.extract.persist",
        "json",
    }


def test_split_package_drops_only_imports_of_itself(root):
    """The same imports, split: internal imports vanish, external ones survive."""
    _write(
        root,
        "palaver/pkg/mod/__init__.py",
        "from __future__ import annotations\n"
        "from . import core\n"
        "from .core import A\n"
        "from palaver.pkg.mod.core import B\n",
    )
    _write(
        root,
        "palaver/pkg/mod/core.py",
        "import collections.abc\n"
        "from palaver.pkg import mod as _facade\n"
        "from palaver.extract.persist import Extraction\n"
        "A = B = 1\n",
    )
    assert python_source.imported_modules("palaver/pkg/mod") == {
        "__future__",
        "collections.abc",
        "palaver.extract.persist",
    }


def test_a_relative_import_that_leaves_the_module_is_counted(root):
    """Positive control: resolving relative imports must not hide an external one."""
    _write(root, "palaver/pkg/mod/__init__.py", "from .core import A\n")
    _write(
        root,
        "palaver/pkg/mod/core.py",
        "from ...extract import client\nfrom .. import sibling\nA = 1\n",
    )
    assert python_source.imported_modules("palaver/pkg/mod") == {"palaver.extract", "palaver.pkg"}
