"""
The install-agent command: printing writes nothing to disk, a bad label or unwritable
path fails before anything is written, and it never boots out an existing job unasked.
"""

from __future__ import annotations

import subprocess

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    DEFAULT_LABEL,
    service_target,
)
from tests._supervision_support import _install_args, _run_capture


def test_printing_writes_nothing_to_disk(tmp_path):
    target = tmp_path / "never-written.plist"
    rendered = _run_capture(_install_args(print_only=True, plist_path=target))
    assert rendered.startswith("<?xml")
    assert not target.exists()


def test_writing_the_plist_does_not_load_it(tmp_path):
    """Loading changes the machine's running state, so it is opt-in."""
    target = tmp_path / "written.plist"
    output = _run_capture(_install_args(plist_path=target, log_dir=tmp_path / "logs"))
    assert target.exists()
    assert "launchctl bootstrap" in output
    assert f"launchctl bootout {service_target(DEFAULT_LABEL)}" in output


def test_an_unwritable_plist_path_fails_rather_than_reporting_success(tmp_path, capsys):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory", encoding="utf-8")
    args = _install_args(plist_path=blocked / "agent.plist", log_dir=tmp_path / "logs")
    assert install_agent.run(args, on_status=lambda _message: None) == 1
    assert "cannot write" in capsys.readouterr().err


def test_a_bad_label_fails_the_command_before_anything_is_written(tmp_path, capsys):
    target = tmp_path / "agent.plist"
    args = _install_args(label="not a label", plist_path=target, log_dir=tmp_path / "logs")
    assert install_agent.run(args, on_status=lambda _message: None) == 1
    assert not target.exists()
    assert "usable launchd label" in capsys.readouterr().err


def test_a_failed_bootstrap_is_reported_rather_than_swallowed(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        install_agent,
        "bootstrap",
        lambda plist, uid=None: subprocess.CompletedProcess([], 5, "", "Input/output error"),
    )
    args = _install_args(plist_path=tmp_path / "agent.plist", log_dir=tmp_path / "logs", load=True)
    assert install_agent.run(args, on_status=lambda _message: None) == 1
    assert "Input/output error" in capsys.readouterr().err


def test_loading_never_boots_out_an_existing_job_unless_asked(tmp_path, monkeypatch):
    """A plain --load must not tear down a daemon that is mid-tick."""
    booted_out: list[str] = []
    monkeypatch.setattr(install_agent, "bootout", lambda label, uid=None: booted_out.append(label))
    monkeypatch.setattr(
        install_agent,
        "bootstrap",
        lambda plist, uid=None: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 1234)

    base = dict(plist_path=tmp_path / "agent.plist", log_dir=tmp_path / "logs")
    assert install_agent.run(_install_args(load=True, **base), on_status=lambda _m: None) == 0
    assert booted_out == []

    # The positive control: --reload does what --load refuses to.
    assert install_agent.run(_install_args(reload=True, **base), on_status=lambda _m: None) == 0
    assert booted_out == [DEFAULT_LABEL]


def test_the_subcommand_is_registered_on_the_root_parser():
    from palaver.cli import SUBCOMMANDS, build_parser

    assert install_agent in SUBCOMMANDS
    parsed = build_parser().parse_args(["install-agent", "--print"])
    assert parsed.handler is install_agent.run
    assert parsed.print_only is True
