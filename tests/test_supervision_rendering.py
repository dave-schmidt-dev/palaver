"""
The rendered plist: something launchd can parse, both log streams under the project's
logs dir, restart-on-any-exit, XML-safe, and every refusal case.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    THROTTLE_INTERVAL_SECONDS,
    InstallAgentError,
    observe_program_arguments,
    render_plist,
)
from tests._supervision_support import _install_args, _run_capture

# --- rendering -------------------------------------------------------------


def test_the_rendered_plist_is_something_launchd_can_parse(tmp_path):
    plist = tmp_path / "agent.plist"
    plist.write_text(
        render_plist(
            program_arguments=["/usr/bin/true"],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["plutil", "-lint", str(plist)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_plist_writes_both_streams_under_the_projects_logs_directory(tmp_path):
    """The done-when's log-path check, asserted on the command's own default.

    Rendering with an explicit `--log-dir` would assert only that the argument
    is honoured. This goes through `run()` so it pins the default, which is
    the thing a reader of the plist would want to be true.
    """
    out = tmp_path / "captured.plist"
    args = _install_args(print_only=True)
    rendered = _run_capture(args)
    out.write_text(rendered, encoding="utf-8")
    paths = re.findall(r"<string>([^<]*\.log)</string>", rendered)
    assert len(paths) == 2
    assert all(Path(path).parent.name == ".logs" for path in paths), paths
    assert all(path.endswith((".out.log", ".err.log")) for path in paths), paths


def test_the_plist_restarts_the_daemon_on_any_exit_including_a_clean_one(tmp_path):
    rendered = render_plist(
        program_arguments=["/usr/bin/true"],
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
    )
    # A KeepAlive *dict* with SuccessfulExit would leave a cleanly exited
    # daemon dead, which is the failure this daemon must not have.
    assert "<key>KeepAlive</key>\n\t<true/>" in rendered
    assert "SuccessfulExit" not in rendered
    assert f"<key>ThrottleInterval</key>\n\t<integer>{THROTTLE_INTERVAL_SECONDS}</integer>" in (
        rendered
    )


def test_a_path_containing_xml_metacharacters_still_renders_a_valid_plist(tmp_path):
    """The escaping case that fails quietly rather than loudly.

    A directory named with `&` produces XML `plutil` rejects; one named with
    `<` produces a plist that parses fine and carries a *truncated* path. The
    second is the reason this is escaped rather than merely hoped about.
    """
    awkward = tmp_path / "a & b <c>"
    awkward.mkdir()
    plist = tmp_path / "agent.plist"
    plist.write_text(
        render_plist(
            program_arguments=[str(awkward / "palaver"), "observe"],
            stdout_path=awkward / "out.log",
            stderr_path=awkward / "err.log",
            working_directory=awkward,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["plutil", "-lint", str(plist)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The path survives the round trip intact, not merely legibly.
    converted = subprocess.run(
        ["plutil", "-extract", "WorkingDirectory", "raw", "-o", "-", str(plist)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert converted.stdout.strip() == str(awkward)


def test_an_empty_argv_is_refused_rather_than_rendered(tmp_path):
    with pytest.raises(InstallAgentError, match="ProgramArguments"):
        render_plist(
            program_arguments=[],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
        )


@pytest.mark.parametrize("label", ["", "com.zerodelta/../evil", "has space", ".leading-dot"])
def test_a_label_that_would_escape_its_filename_is_refused(tmp_path, label):
    with pytest.raises(InstallAgentError, match="usable launchd label"):
        render_plist(
            label=label,
            program_arguments=["/usr/bin/true"],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
        )


def test_a_missing_template_is_an_error_rather_than_an_empty_plist(tmp_path):
    with pytest.raises(InstallAgentError, match="cannot read the agent template"):
        render_plist(
            program_arguments=["/usr/bin/true"],
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            template_path=tmp_path / "absent.plist.tmpl",
        )


def test_unset_daemon_options_are_omitted_rather_than_duplicated():
    """The daemon owns its own defaults; the plist must not restate them."""
    bare = observe_program_arguments(executable=Path("/bin/palaver"))
    assert bare == ["/bin/palaver", "observe"]

    full = observe_program_arguments(
        executable=Path("/bin/palaver"),
        db_path=Path("/tmp/o.db"),
        cursor_root=Path("/tmp/cursors"),
        interval=45.0,
    )
    assert full == [
        "/bin/palaver",
        "observe",
        "--db",
        "/tmp/o.db",
        "--cursors",
        "/tmp/cursors",
        "--interval",
        "45",
    ]


def test_the_default_executable_is_the_console_script_not_the_interpreter():
    """A plist running `python3 observe` would crash-loop under KeepAlive.

    The venv's `bin/python3` is a symlink to the base interpreter, so
    resolving it first lands in a directory with no `palaver` script. This
    asserts the search order that avoids that.
    """
    found = install_agent._default_executable()
    assert found.name == "palaver", f"got {found}"
    assert found.exists()
