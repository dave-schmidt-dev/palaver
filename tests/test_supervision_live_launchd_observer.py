"""
Live launchd, the observer agent: loads and is found by launchctl print, and restarts
(or, with KeepAlive stripped, does not restart) when killed.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    DEFAULT_LABEL,
    RESTART_WINDOW_SECONDS,
    THROTTLE_INTERVAL_SECONDS,
    bootstrap,
    domain_target,
    print_service,
    render_plist,
    wait_for_new_pid,
)
from tests._supervision_support import (
    SELFTEST_LABEL,
    STARTUP_WINDOW_SECONDS,
    _await_pid,
    _bootout_and_settle,
    live,
)

#: Tick fast enough that a restarted job has visibly done something, slow
#: enough that a two-second test window does not accumulate ticks.
SELFTEST_INTERVAL = "5"


#: How long the no-KeepAlive control waits before concluding that nothing is
#: coming back. Longer than one throttle interval, because a job that *would*
#: restart cannot do so any sooner than that.
NO_RESTART_WINDOW_SECONDS = THROTTLE_INTERVAL_SECONDS + 3.0


def _render_selftest_plist(tmp_path: Path, *, template_path: Path | None = None) -> Path:
    """Render a plist that runs the real daemon against an empty sample root."""
    sample = tmp_path / "sample"
    sample.mkdir(exist_ok=True)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    argv = [
        str(install_agent._default_executable()),
        "observe",
        "--db",
        str(tmp_path / "selftest.db"),
        "--cursors",
        str(tmp_path / "cursors"),
        "--sample",
        str(sample),
        "--interval",
        SELFTEST_INTERVAL,
    ]
    rendered = render_plist(
        label=SELFTEST_LABEL,
        program_arguments=argv,
        stdout_path=logs / "out.log",
        stderr_path=logs / "err.log",
        working_directory=tmp_path,
        **({} if template_path is None else {"template_path": template_path}),
    )
    plist = tmp_path / f"{SELFTEST_LABEL}.plist"
    plist.write_text(rendered, encoding="utf-8")
    return plist


@pytest.fixture
def loaded_selftest_agent(tmp_path):
    """Bootstrap the selftest agent, yield its plist path, and always unload."""
    plist = _render_selftest_plist(tmp_path)
    _bootout_and_settle(SELFTEST_LABEL)
    result = bootstrap(plist)
    if result.returncode != 0:
        _bootout_and_settle(SELFTEST_LABEL, strict=False)
        pytest.fail(f"launchctl bootstrap failed: {(result.stderr or result.stdout).strip()}")
    try:
        yield plist
    finally:
        _bootout_and_settle(SELFTEST_LABEL, strict=False)


# --- live launchd ----------------------------------------------------------


@live
def test_the_observe_agent_loads_and_launchctl_print_reports_it(loaded_selftest_agent):
    """The done-when's load check: `launchctl print` on the loaded label."""
    result = print_service(SELFTEST_LABEL)
    assert result.returncode == 0, (result.stdout + result.stderr)[:2000]
    assert SELFTEST_LABEL in result.stdout

    # A label that was never loaded is the control. Without it, a `print`
    # that succeeded for every argument would pass this test unchanged.
    absent = print_service(f"{SELFTEST_LABEL}.absent")
    assert absent.returncode != 0


@live
def test_killing_the_observe_daemon_brings_back_a_different_pid(loaded_selftest_agent):
    """The done-when's restart check, against the real daemon and real launchd."""
    original = _await_pid(SELFTEST_LABEL)
    assert original is not None, (
        f"the job never reached a signalable pid within {STARTUP_WINDOW_SECONDS}s of bootstrap"
    )

    os.kill(original, signal.SIGKILL)
    restarted = wait_for_new_pid(SELFTEST_LABEL, original, on_status=lambda _message: None)

    assert restarted is not None, (
        f"no restart within {RESTART_WINDOW_SECONDS}s of killing pid {original}"
    )
    assert restarted != original


@live
def test_an_agent_without_keepalive_stays_dead_when_killed(tmp_path):
    """The control for the restart test.

    Same daemon, same launchd, same kill — only `KeepAlive` removed. If this
    job came back too, the restart test above would be measuring `RunAtLoad`,
    or a kill that never landed, rather than the supervision policy it claims
    to prove.
    """
    stripped = tmp_path / "no-keepalive.plist.tmpl"
    original_template = install_agent.TEMPLATE_PATH.read_text(encoding="utf-8")
    without = original_template.replace("<key>KeepAlive</key>\n\t<true/>\n", "")
    assert without != original_template, "the KeepAlive block moved; this control is now vacuous"
    stripped.write_text(without, encoding="utf-8")

    plist = _render_selftest_plist(tmp_path, template_path=stripped)
    _bootout_and_settle(SELFTEST_LABEL)
    result = bootstrap(plist)
    assert result.returncode == 0, (result.stderr or result.stdout).strip()
    try:
        original = _await_pid(SELFTEST_LABEL)
        assert original is not None, (
            f"the control job never reached a signalable pid within "
            f"{STARTUP_WINDOW_SECONDS}s of bootstrap, so the kill below would "
            f"prove nothing"
        )
        os.kill(original, signal.SIGKILL)
        revived = wait_for_new_pid(
            SELFTEST_LABEL,
            original,
            window=NO_RESTART_WINDOW_SECONDS,
            on_status=lambda _message: None,
        )
        assert revived is None, f"a job with no KeepAlive came back as pid {revived}"
    finally:
        _bootout_and_settle(SELFTEST_LABEL, strict=False)


@live
def test_the_selftest_label_is_never_the_one_a_user_installs():
    """Guards the isolation the rest of the live tests depend on."""
    assert SELFTEST_LABEL != DEFAULT_LABEL
    assert domain_target().startswith("gui/")
