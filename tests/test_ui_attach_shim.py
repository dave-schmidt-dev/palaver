"""
The generated AutoLaunch shim script itself (syntax, imports, cookie handling, backoff)
and installing/printing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from palaver.ui import autolaunch
from palaver.ui.autolaunch import (
    ADVISORY_NAME,
    AUTOLAUNCH_DIR,
    SHIM_NAME,
    install_shim,
    render_shim,
)
from palaver.ui.connection import (
    COOKIE_ENV,
    KEY_ENV,
)

# --- the shim --------------------------------------------------------------


def test_the_shim_is_valid_python():
    compile(render_shim(), "palaver.py", "exec")


def test_the_shim_imports_nothing_from_palaver():
    """It runs under iTerm2's managed Python, whose version iTerm2 chooses.

    An import of `palaver` would be a syntax error the day that runtime is
    older than 3.14, and the symptom would be a script that silently never
    starts.
    """
    source = render_shim()
    assert "import palaver" not in source
    assert "from palaver" not in source


def test_the_shim_uses_no_syntax_newer_than_the_oldest_runtime_it_may_meet():
    """f-strings and walruses are the two easy ways to break 3.5 compatibility.

    The bar is deliberately lower than any Python iTerm2 plausibly ships:
    the cost of staying conservative here is one `%` format, and the cost of
    guessing wrong is a surface that never appears and never says why.
    """
    source = render_shim()
    assert ":=" not in source
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith('f"'), line
        assert 'f"' not in stripped.replace('"""', ""), line


def test_the_shim_spawns_the_interpreter_palaver_is_installed_into():
    source = render_shim(python_executable=Path("/opt/pythons/3.14/bin/python3"))
    assert '"/opt/pythons/3.14/bin/python3"' in source or (
        "'/opt/pythons/3.14/bin/python3'" in source
    )
    assert '"-m", MODULE' in source


def test_the_shim_pins_one_interpreter_name_however_install_was_invoked(monkeypatch, tmp_path):
    """`--install` must render the same bytes from either spelling.

    `sys.executable` is whatever name started the process, and a virtualenv
    offers both `python` and `python3`, so without normalization one
    environment has two valid shims and no check can compare the installed
    file against the rendered one.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text("", encoding="utf-8")
    (bin_dir / "python3").symlink_to(bin_dir / "python")

    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    from_python3 = render_shim()
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    from_python = render_shim()

    assert from_python3 == from_python
    assert "PYTHON = " + repr(str(bin_dir / "python")) in from_python


def test_an_interpreter_with_no_python_alias_beside_it_is_left_alone(monkeypatch, tmp_path):
    """Normalizing to a name that does not exist would render a dead shim."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "executable", str(bin_dir / "python3"))
    assert "PYTHON = " + repr(str(bin_dir / "python3")) in render_shim()


def test_the_shim_pins_the_child_to_the_project_local_state_root():
    source = render_shim(project_root=Path("/project/palaver"))
    assert "CWD = '/project/palaver'" in source
    assert "cwd=CWD" in source


def test_the_shim_passes_the_cookie_by_environment_and_never_by_argv():
    """`ps` shows argv to every process on the machine.

    A cookie in an argument vector is a credential published to the whole
    system for the lifetime of the process.
    """
    source = render_shim()
    argv_line = next(line for line in source.splitlines() if "subprocess.call" in line)
    assert COOKIE_ENV not in argv_line
    assert "env=env" in argv_line
    assert f"env[{COOKIE_ENV!r}]" in source


def test_the_shim_never_prints_the_cookie():
    """Its stdout and stderr are a log file iTerm2 keeps on disk."""
    source = render_shim()
    for line in source.splitlines():
        if "write(" in line or "print(" in line:
            assert "cookie" not in line.lower(), line


def test_the_shim_backs_off_after_a_fast_failure_but_not_after_a_long_run():
    source = render_shim()
    assert "min(backoff * 2, MAX_BACKOFF)" in source
    assert "MIN_BACKOFF if ran_for > MAX_BACKOFF" in source


def test_the_shim_asks_iterm_for_a_fresh_cookie_each_time():
    """An inherited cookie is consumed by the first connection.

    Without a fresh request every restart after the first would fail
    authentication, which would look exactly like a crash loop.
    """
    source = render_shim()
    assert "request cookie and key" in source
    assert ADVISORY_NAME in source
    assert KEY_ENV in source


def test_installing_writes_the_shim_where_iterm_looks_for_it(tmp_path):
    path = install_shim(directory=tmp_path / "AutoLaunch")
    assert path.name == SHIM_NAME
    assert path.parent.name == "AutoLaunch"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_the_default_install_directory_is_iterms_autolaunch_directory():
    """A script anywhere else is simply never run by iTerm2."""
    assert AUTOLAUNCH_DIR.parts[-2:] == ("Scripts", "AutoLaunch")
    assert "iTerm2" in AUTOLAUNCH_DIR.parts


def test_printing_the_shim_writes_nothing(tmp_path, capsys):
    assert autolaunch.run(["--print"]) == 0
    assert capsys.readouterr().out.startswith('"""Palaver')
    assert not (tmp_path / SHIM_NAME).exists()


def test_the_entry_point_reports_a_setup_failure_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(COOKIE_ENV, raising=False)
    assert autolaunch.run([]) == 1
    assert "Enable Python API" in capsys.readouterr().err
