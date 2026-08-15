"""Task 5.0: the launchd user agent that keeps `palaver observe` running.

Most of this file is headless rendering, but three tests load a real job into
the real launchd GUI domain, because nothing else can prove a plist is
*loadable* — `plutil -lint` proves only that it is well-formed XML, and a
plist can lint cleanly while launchd rejects it.

Four things keep those live tests from touching the machine's real state:

* **A distinct label.** `com.zerodelta.palaver.observe.selftest` is never the
  label `palaver install-agent` installs by default, so a live test cannot
  bootstrap over, or bootout, a daemon the user is actually running. It is a
  fixed name rather than a pid-suffixed one on purpose: a leaked job stays
  findable with `launchctl list | grep selftest`.
* **A plist under `tmp_path`.** launchd bootstraps from any absolute path,
  so nothing is written into `~/Library/LaunchAgents`, where it would be
  reloaded at every login.
* **An empty `--sample` root.** The job runs the *real* daemon — a stand-in
  `sleep` would prove KeepAlive restarts `sleep` — but pointed at a directory
  containing no sessions, so it discovers nothing, opens no real session
  store, and issues no inference request. Its store and cursors are under
  `tmp_path` too.
* **Unconditional teardown.** The fixture boots the label out in a `finally`,
  and boots it out again before loading, so a previous crashed run cannot
  leave a job that poisons the next one.

The restart test has a live control: `test_an_agent_without_keepalive_stays_
dead_when_killed` loads the same plist with the `KeepAlive` block stripped and
asserts no new pid appears. Without it, "a pid exists again after the kill"
could just as well mean the kill never landed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import json
import os
import plistlib
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli import mcp as mcp_cli
from palaver.cli.install_agent import (
    DEFAULT_LABEL,
    MCP_LABEL,
    MCP_TEMPLATE_PATH,
    OBSERVE_LABEL,
    OBSERVE_TEMPLATE_PATH,
    RESTART_WINDOW_SECONDS,
    THROTTLE_INTERVAL_SECONDS,
    InstallAgentError,
    bootout,
    bootstrap,
    domain_target,
    mcp_program_arguments,
    observe_program_arguments,
    pid_is_ours,
    print_service,
    render_plist,
    service_pid,
    service_target,
    wait_for_new_pid,
    wait_for_running_pid,
)
from palaver.observer import socket as writer_socket
from palaver.observer.socket import (
    DaemonAlreadyRunningError,
    NonLocalFilesystemError,
    single_writer,
)

#: Never `DEFAULT_LABEL`. See the module docstring.
SELFTEST_LABEL = "com.zerodelta.palaver.observe.selftest"

#: The same isolation rule for task 6.5's MCP agent: never `MCP_LABEL`, so a
#: live test cannot bootout the server an agent is currently registered
#: against. Its port is allocated per run rather than fixed, because unlike
#: the observer this job binds one — and 8787 is the port real clients hold.
MCP_SELFTEST_LABEL = "com.zerodelta.palaver.mcp.selftest"

#: Tick fast enough that a restarted job has visibly done something, slow
#: enough that a two-second test window does not accumulate ticks.
SELFTEST_INTERVAL = "5"

#: Seconds to wait for a freshly bootstrapped job to have a pid.
#:
#: One throttle interval plus margin, and derived from the constant rather
#: than written as a number, because the two are not independent. launchd
#: throttles per *label*, and every live test in this file bootstraps the
#: same selftest label — so by the time the last of them runs, the label's
#: throttle accounting is warm and a start can be held for most of an
#: interval before the job ever execs. This was 10.0, i.e. exactly one
#: interval with no margin, which passed this file in isolation and failed
#: roughly one whole-suite run in three.
STARTUP_WINDOW_SECONDS = THROTTLE_INTERVAL_SECONDS + 10.0

#: How long the no-KeepAlive control waits before concluding that nothing is
#: coming back. Longer than one throttle interval, because a job that *would*
#: restart cannot do so any sooner than that.
NO_RESTART_WINDOW_SECONDS = THROTTLE_INTERVAL_SECONDS + 3.0

live = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("launchctl") is None,
    reason="launchd user agents exist only on macOS with launchctl on PATH",
)


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


def _await_pid(label: str, *, window: float = STARTUP_WINDOW_SECONDS) -> int | None:
    """Wait for a job to reach a pid this test can actually signal.

    Not `service_pid`. launchd publishes the pid while it is still the root
    `xpcproxy` stub, and killing that raises `EPERM` — which is exactly how
    this test failed the first time it ran.
    """
    return wait_for_running_pid(label, window=window)


@pytest.fixture
def loaded_selftest_agent(tmp_path):
    """Bootstrap the selftest agent, yield its plist path, and always unload."""
    plist = _render_selftest_plist(tmp_path)
    bootout(SELFTEST_LABEL)
    result = bootstrap(plist)
    if result.returncode != 0:
        bootout(SELFTEST_LABEL)
        pytest.fail(f"launchctl bootstrap failed: {(result.stderr or result.stdout).strip()}")
    try:
        yield plist
    finally:
        bootout(SELFTEST_LABEL)


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


# --- pid parsing and the restart wait --------------------------------------


def test_a_loaded_but_not_running_job_reports_no_pid(monkeypatch):
    """`launchctl print` omits the pid line entirely between runs."""
    monkeypatch.setattr(
        install_agent,
        "print_service",
        lambda label, uid=None: subprocess.CompletedProcess([], 0, "state = not running\n", ""),
    )
    assert service_pid("whatever") is None


def test_a_running_job_reports_the_pid_launchctl_printed(monkeypatch):
    monkeypatch.setattr(
        install_agent,
        "print_service",
        lambda label, uid=None: subprocess.CompletedProcess(
            [], 0, "\tpid = 4242\n\tstate = r\n", ""
        ),
    )
    assert service_pid("whatever") == 4242


def test_an_unloaded_job_reports_no_pid_even_if_stdout_mentions_one(monkeypatch):
    """A non-zero `launchctl print` is authoritative over whatever it printed."""
    monkeypatch.setattr(
        install_agent,
        "print_service",
        lambda label, uid=None: subprocess.CompletedProcess([], 113, "pid = 9\n", "not found"),
    )
    assert service_pid("whatever") is None


def test_the_restart_wait_rejects_the_pid_it_was_told_to_replace(monkeypatch):
    """Seeing the old pid means the kill has not landed, not that it restarted."""
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 4242)
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: True)
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert wait_for_new_pid("whatever", 4242, window=0.2, poll_interval=0.01) is None


def test_the_restart_wait_returns_the_first_different_pid(monkeypatch):
    seen = iter([4242, 4242, 9001])
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: next(seen))
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: True)
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert wait_for_new_pid("whatever", 4242, window=5.0, poll_interval=0.01) == 9001


def test_our_own_process_is_signalable_and_a_free_pid_is_not():
    """The two live branches of `pid_is_ours`, with no mocking at all."""
    assert pid_is_ours(os.getpid())

    # A pid that cannot exist: ProcessLookupError, not EPERM.
    assert not pid_is_ours(2**30)


def test_a_process_this_user_may_not_signal_is_not_ours():
    """launchd's `xpcproxy` stub runs as root for a moment after every start.

    `launchd` itself is pid 1 and permanently uid 0, which makes it a stable
    stand-in for that transient state — no other process on the machine is
    guaranteed to be both alive and unsignalable.
    """
    assert not pid_is_ours(1)


def test_the_restart_wait_will_not_accept_launchds_root_stub(monkeypatch):
    """A pid that is new but not yet ours is not a restart.

    Without this the restart check reports success the instant launchd forks,
    which is before the daemon exists — the failure this suite hit live.
    """
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 9001)
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: False)
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert wait_for_new_pid("whatever", 4242, window=0.2, poll_interval=0.01) is None

    # The positive control: the same pid, once the stub has execed.
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: True)
    assert wait_for_new_pid("whatever", 4242, window=0.2, poll_interval=0.01) == 9001


def test_waiting_for_a_running_pid_holds_out_for_a_signalable_one(monkeypatch):
    ours = iter([False, False, True])
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 9001)
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: next(ours))
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert wait_for_running_pid("whatever", window=5.0, poll_interval=0.01) == 9001


def test_waiting_for_a_running_pid_gives_up_rather_than_returning_the_stub(monkeypatch):
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 9001)
    monkeypatch.setattr(install_agent, "pid_is_ours", lambda pid: False)
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert wait_for_running_pid("whatever", window=0.2, poll_interval=0.01) is None


def test_the_restart_window_is_wider_than_two_throttle_intervals():
    """One interval can already be partly spent when the process dies."""
    assert RESTART_WINDOW_SECONDS > 2 * THROTTLE_INTERVAL_SECONDS


# --- the command -----------------------------------------------------------


class _Args:
    def __init__(self, **fields):
        defaults = {
            # None, not DEFAULT_LABEL: `run` resolves an unset label to the
            # selected service's own, and hardcoding the observer's here would
            # mean the mcp cases silently installed under the observer's label
            # while appearing to test the default path.
            "service": "observe",
            "label": None,
            "db": None,
            "cursors": None,
            "interval": None,
            "host": None,
            "port": None,
            "executable": None,
            "log_dir": None,
            "plist_path": None,
            "print_only": False,
            "load": False,
            "reload": False,
        }
        for key, value in {**defaults, **fields}.items():
            setattr(self, key, value)


def _install_args(**fields) -> _Args:
    return _Args(**fields)


def _run_capture(args) -> str:
    import io

    out = io.StringIO()
    status = install_agent.run(args, out=out, on_status=lambda _message: None)
    assert status == 0
    return out.getvalue()


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


# --- the mcp agent (task 6.5) ----------------------------------------------
#
# Every name here carries `mcp_agent`, which is the selection the plan's quick
# check runs: `pytest -q tests/test_supervision.py -k mcp_agent`. A test whose
# name misses that substring is a test the gate does not run.


def _render_keys(template_path: Path, tmp_path: Path) -> dict:
    """Render a template with throwaway values and parse what launchd would see.

    The templates cannot be parsed directly — they are `string.Template`
    sources, and the unsubstituted placeholders are not valid inside the
    elements they sit in. Rendering first is the only way to compare them as
    plists rather than as text.
    """
    rendered = render_plist(
        label="com.zerodelta.palaver.render-probe",
        program_arguments=["/usr/bin/true"],
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        working_directory=tmp_path,
        template_path=template_path,
    )
    return plistlib.loads(rendered.encode("utf-8"))


def test_the_rendered_mcp_agent_plist_is_something_launchd_can_parse(tmp_path):
    plist = tmp_path / "mcp-agent.plist"
    plist.write_text(
        render_plist(
            label=MCP_SELFTEST_LABEL,
            program_arguments=mcp_program_arguments(
                executable=Path("/bin/palaver"), db_path=tmp_path / "m.db", port=9999
            ),
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            template_path=MCP_TEMPLATE_PATH,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["plutil", "-lint", str(plist)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # `plutil` alone is not enough, and this is not a hypothetical. The first
    # draft of this template wrote a command-line flag inside an XML comment.
    # A double hyphen is illegal there, `plutil -lint` accepted the file
    # regardless, and plistlib refused it. Parsing with a second, stricter
    # implementation is what turned that into a failure instead of a latent
    # portability bug in a file only launchd normally reads.
    parsed = plistlib.loads(plist.read_bytes())
    assert parsed["Label"] == MCP_SELFTEST_LABEL


def test_the_mcp_agent_is_not_throttled_the_way_a_background_job_is(tmp_path):
    """The finding that made task 6.5 more than a copy of task 5.0.

    `man launchd.plist` describes Background's resource limits as existing
    "to prevent them from disrupting the user experience". This server is
    *inside* the user experience: every cycle it spends is an agent's blocking
    tool call. So it is Standard, and it omits LowPriorityIO and Nice rather
    than setting them low.

    The observer assertions are the positive control. Without them, this test
    would pass just as happily against a template that had lost its scheduling
    keys entirely, which is a different bug wearing the same result.
    """
    mcp = _render_keys(MCP_TEMPLATE_PATH, tmp_path)
    assert mcp["ProcessType"] == "Standard"
    assert "LowPriorityIO" not in mcp
    assert "Nice" not in mcp

    observe = _render_keys(OBSERVE_TEMPLATE_PATH, tmp_path)
    assert observe["ProcessType"] == "Background"
    assert observe["LowPriorityIO"] is True
    assert observe["Nice"] == 5


def test_the_two_mcp_agent_templates_agree_on_everything_but_scheduling(tmp_path):
    """The guard that lets these be two files instead of one.

    Two static templates were chosen over one parameterized template because
    the MCP job *omits* keys the observer sets, and a placeholder can supply a
    value but cannot remove an element. The cost of that choice is ~60 lines
    of duplicated XML that could drift. This is what makes drift fail loudly:
    every key outside the scheduling set must be present in both and identical
    in both.
    """
    scheduling = {"ProcessType", "LowPriorityIO", "Nice"}
    observe = _render_keys(OBSERVE_TEMPLATE_PATH, tmp_path)
    mcp = _render_keys(MCP_TEMPLATE_PATH, tmp_path)

    assert set(observe) - scheduling == set(mcp) - scheduling
    for key in sorted(set(observe) - scheduling):
        assert observe[key] == mcp[key], f"{key} drifted between the two templates"

    # And the divergence is exactly the one that was intended, not merely
    # non-empty: naming both sides keeps this from passing if the MCP job
    # quietly grew a Nice key back.
    assert set(observe) & scheduling == scheduling
    assert set(mcp) & scheduling == {"ProcessType"}


def test_unset_mcp_agent_options_are_omitted_rather_than_duplicated():
    """The server owns its own defaults; the plist must not restate them."""
    bare = mcp_program_arguments(executable=Path("/bin/palaver"))
    assert bare == ["/bin/palaver", "mcp"]

    full = mcp_program_arguments(
        executable=Path("/bin/palaver"),
        db_path=Path("/tmp/m.db"),
        host="127.0.0.1",
        port=8787,
    )
    assert full == [
        "/bin/palaver",
        "mcp",
        "--db",
        "/tmp/m.db",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
    ]


def test_the_mcp_agent_and_the_observer_never_share_a_label():
    """Two jobs under one label is one job; launchd keys everything on it."""
    assert MCP_LABEL != OBSERVE_LABEL
    assert MCP_SELFTEST_LABEL not in {MCP_LABEL, OBSERVE_LABEL, SELFTEST_LABEL}
    assert install_agent.SERVICES["mcp"].label == MCP_LABEL
    assert install_agent.SERVICES["observe"].label == OBSERVE_LABEL
    assert install_agent.SERVICES["mcp"].template_path != (
        install_agent.SERVICES["observe"].template_path
    )


@pytest.mark.parametrize(
    ("service", "option", "value"),
    [
        ("mcp", "interval", 45.0),
        ("mcp", "cursors", Path("/tmp/cursors")),
        ("observe", "host", "127.0.0.1"),
        ("observe", "port", 9999),
    ],
)
def test_an_mcp_agent_option_meant_for_the_other_service_is_refused(
    tmp_path, capsys, service, option, value
):
    """Refused, not ignored.

    `--service mcp --interval 45` is a coherent sentence that means nothing.
    Dropping the flag silently would write a plist that loads, runs, and
    supervises something other than what was asked, with nothing anywhere
    recording that a flag was discarded.
    """
    import io

    target = tmp_path / "never-written.plist"
    args = _install_args(service=service, plist_path=target, **{option: value})
    status = install_agent.run(args, out=io.StringIO(), on_status=lambda _message: None)

    assert status == 2
    assert f"--{option}" in capsys.readouterr().err
    assert not target.exists(), "a refused invocation still wrote a plist"


def test_the_mcp_agent_service_is_reachable_from_the_real_parser(tmp_path):
    """The wiring check, and the reason it goes through `run` and not `render_plist`.

    Tasks 6.3 and 6.4 both produced a fully-tested unit that nothing called.
    Asserting on `render_plist(template_path=MCP_TEMPLATE_PATH)` would pass
    identically if `--service` had never been registered on the parser. So
    this drives the argv a user would actually type.
    """
    import io

    from palaver.cli import build_parser

    target = tmp_path / "from-parser.plist"
    parsed = build_parser().parse_args(
        [
            "install-agent",
            "--service",
            "mcp",
            "--port",
            "9999",
            "--plist-path",
            str(target),
        ]
    )
    assert parsed.service == "mcp"
    assert parsed.handler is install_agent.run

    status = parsed.handler(parsed, out=io.StringIO(), on_status=lambda _message: None)
    assert status == 0

    written = plistlib.loads(target.read_bytes())
    assert written["Label"] == MCP_LABEL
    assert written["ProcessType"] == "Standard"
    assert written["ProgramArguments"][1] == "mcp"
    assert written["ProgramArguments"][-2:] == ["--port", "9999"]


def test_the_default_mcp_agent_service_is_still_the_observer(tmp_path):
    """Task 5.0's invocation must keep meaning what it meant.

    `palaver install-agent` shipped before `--service` existed. If the default
    moved, an existing habit would silently install a different job.
    """
    import io

    from palaver.cli import build_parser

    target = tmp_path / "default.plist"
    parsed = build_parser().parse_args(["install-agent", "--plist-path", str(target)])
    assert parsed.service == "observe"

    status = install_agent.run(parsed, out=io.StringIO(), on_status=lambda _message: None)
    assert status == 0

    written = plistlib.loads(target.read_bytes())
    assert written["Label"] == OBSERVE_LABEL
    assert written["ProgramArguments"][1] == "observe"


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
    bootout(SELFTEST_LABEL)
    assert bootstrap(plist).returncode == 0
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
        bootout(SELFTEST_LABEL)


@live
def test_the_selftest_label_is_never_the_one_a_user_installs():
    """Guards the isolation the rest of the live tests depend on."""
    assert SELFTEST_LABEL != DEFAULT_LABEL
    assert domain_target().startswith("gui/")


# --- live launchd: the mcp agent (task 6.5) --------------------------------


def _accepting(port: int, *, window: float, host: str = "127.0.0.1") -> bool:
    """Wait until something actually accepts on the port.

    Not `wait_for_running_pid`, and the difference is the whole reason this
    helper exists. launchd publishes a pid the moment the process execs, but
    this one is a Python interpreter that must import, open the store, and
    only then bind. A client that proceeded on the pid alone would connect
    into that gap and read `ECONNREFUSED` as a failed restart.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _await_unloaded(label: str, *, window: float = STARTUP_WINDOW_SECONDS) -> bool:
    """Wait until launchd no longer knows the label at all.

    `bootout` returns as soon as launchd has *accepted* the request, not once
    the job is gone, and bootstrapping a label that is still being torn down
    fails with `Bootstrap failed: 5: Input/output error`. The observer's live
    tests never hit that because there are three of them; four MCP tests
    sharing one label hit it every run.

    It is worse for this job than for the observer's, too: these tests SIGKILL
    a process under `KeepAlive`, so at teardown launchd may be mid-restart —
    booting out a job that is in the middle of coming back is exactly the race
    this closes.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        if print_service(label).returncode != 0:
            return True
        time.sleep(0.2)
    return False


async def _recall_over_http(url: str) -> list[str]:
    """Read the seeded project through a *fresh* client, and return statements."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.call_tool("palaver_recall", {"scope": {"project": "demo"}})
            assert not result.is_error, result.content[0].text
            page = json.loads(result.content[0].text)
    return [memory["statement"] for memory in page["memories"]]


@pytest.fixture
def loaded_mcp_agent(tmp_path):
    """Bootstrap the MCP server under real launchd, on a port nobody else holds."""
    db_path = tmp_path / "mcp-selftest.db"
    _seed_memory(db_path)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    port = mcp_cli._free_port()

    plist = tmp_path / f"{MCP_SELFTEST_LABEL}.plist"
    plist.write_text(
        render_plist(
            label=MCP_SELFTEST_LABEL,
            program_arguments=mcp_program_arguments(
                executable=install_agent._default_executable(),
                db_path=db_path,
                host="127.0.0.1",
                port=port,
            ),
            stdout_path=logs / "out.log",
            stderr_path=logs / "err.log",
            working_directory=tmp_path,
            template_path=MCP_TEMPLATE_PATH,
        ),
        encoding="utf-8",
    )

    bootout(MCP_SELFTEST_LABEL)
    assert _await_unloaded(MCP_SELFTEST_LABEL), (
        "a previous run's job is still loaded; bootstrapping over it would fail with EIO"
    )
    result = bootstrap(plist)
    if result.returncode != 0:
        bootout(MCP_SELFTEST_LABEL)
        pytest.fail(f"launchctl bootstrap failed: {(result.stderr or result.stdout).strip()}")
    try:
        yield {"port": port, "url": f"http://127.0.0.1:{port}/mcp", "logs": logs}
    finally:
        bootout(MCP_SELFTEST_LABEL)
        # Blocking here rather than in the next test's setup: teardown is
        # where the job actually is, and leaving it half-removed makes the
        # *following* test fail for a reason that has nothing to do with it.
        _await_unloaded(MCP_SELFTEST_LABEL)


@live
def test_the_mcp_agent_loads_and_launchctl_print_reports_it(loaded_mcp_agent):
    """The done-when's load check: `launchctl print` on the MCP label."""
    result = print_service(MCP_SELFTEST_LABEL)
    assert result.returncode == 0, (result.stdout + result.stderr)[:2000]
    assert MCP_SELFTEST_LABEL in result.stdout

    # The same control the observer's load test carries: a `print` that
    # succeeded for any argument would pass the assertion above unchanged.
    absent = print_service(f"{MCP_SELFTEST_LABEL}.absent")
    assert absent.returncode != 0


@live
def test_killing_the_mcp_agent_brings_back_a_different_pid(loaded_mcp_agent):
    """The done-when's restart check, against the real server and real launchd."""
    assert _accepting(loaded_mcp_agent["port"], window=STARTUP_WINDOW_SECONDS), (
        f"the server never bound {loaded_mcp_agent['port']} within "
        f"{STARTUP_WINDOW_SECONDS}s of bootstrap"
    )
    original = _await_pid(MCP_SELFTEST_LABEL)
    assert original is not None

    os.kill(original, signal.SIGKILL)
    restarted = wait_for_new_pid(MCP_SELFTEST_LABEL, original, on_status=lambda _message: None)

    assert restarted is not None, (
        f"no restart within {RESTART_WINDOW_SECONDS}s of killing pid {original}"
    )
    assert restarted != original


@live
def test_a_client_reading_across_the_mcp_agent_restart_gets_its_answer(loaded_mcp_agent):
    """The done-when's reconnect check — corrected, because the premise was wrong.

    The plan and README both said a restart is "invisible" to clients because
    HTTP transports reconnect automatically. Half of that is true. The TCP
    connection is re-established, but the MCP *session* is not: the restarted
    server has never seen the `Mcp-Session-Id` the client is holding, answers
    404, and the SDK surfaces that as `Session terminated` rather than
    re-initializing (mcp/client/streamable_http.py). Re-establishing the
    session is the host application's job, not the transport's.

    So this asserts both halves, from one restart. The held session must fail
    — that is the negative control, and without it "a fresh client works"
    would pass even if the kill had never landed — and a client that
    reconnects must get the same answer as before, from the same store, with
    no connection error.
    """
    url = loaded_mcp_agent["url"]
    port = loaded_mcp_agent["port"]

    assert _accepting(port, window=STARTUP_WINDOW_SECONDS), "the server never bound its port"
    before = asyncio.run(_recall_over_http(url))
    assert before, "the fixture seeded no memory, so the reads below prove nothing"

    original = _await_pid(MCP_SELFTEST_LABEL)
    assert original is not None
    os.kill(original, signal.SIGKILL)
    restarted = wait_for_new_pid(MCP_SELFTEST_LABEL, original, on_status=lambda _message: None)
    assert restarted is not None and restarted != original

    # The pid is back; the socket need not be. Waiting on the port rather than
    # on the pid is what keeps this from being a flaky race.
    assert _accepting(port, window=RESTART_WINDOW_SECONDS), (
        f"pid {restarted} exists but nothing is accepting on {port}"
    )

    after = asyncio.run(_recall_over_http(url))
    assert after == before


@live
def test_a_held_session_does_not_survive_the_mcp_agent_restart(loaded_mcp_agent):
    """Documents what actually happens, so the README's claim stays honest.

    Kept separate from the reconnect test because it asserts the opposite
    outcome and would otherwise read as a bug. It is the measured behaviour of
    the SDK, and the reason the reconnect test opens a new client rather than
    reusing one.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = loaded_mcp_agent["port"]
    assert _accepting(port, window=STARTUP_WINDOW_SECONDS)

    async def hold_across_restart() -> str:
        async with streamable_http_client(loaded_mcp_agent["url"]) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                first = await session.call_tool("palaver_recall", {"scope": {"project": "demo"}})
                assert not first.is_error

                original = _await_pid(MCP_SELFTEST_LABEL)
                assert original is not None
                os.kill(original, signal.SIGKILL)
                assert (
                    wait_for_new_pid(MCP_SELFTEST_LABEL, original, on_status=lambda _message: None)
                    is not None
                )
                assert _accepting(port, window=RESTART_WINDOW_SECONDS)

                try:
                    await session.call_tool("palaver_recall", {"scope": {"project": "demo"}})
                except Exception as exc:  # noqa: BLE001 - the escape is the finding
                    return f"{type(exc).__name__}: {exc}"
                return ""

    outcome = asyncio.run(hold_across_restart())
    assert outcome, "the held session survived the restart; the README claim may now be true"
    assert "session" in outcome.lower(), outcome


# ---------------------------------------------------------------------------
# Task 6.3: the single-writer lock, the socket, and the order between them.
#
# The interesting failures here are all races, so most of these tests use real
# processes rather than monkeypatched primitives. A `flock` mocked out proves
# nothing about a `flock` -- the whole question is what the kernel does when
# two processes ask at once.
#
# None of them use `tmp_path`. pytest's is ~90 bytes before the test's own
# name is appended, and `sun_path` holds 103 -- so every socket test would
# fail on the path length rather than on the property it is checking. That is
# not a test-only quirk: it is the same limit a user with a deeply nested
# project hits, which is why `socket_path_for` reports it by name.
# ---------------------------------------------------------------------------


@pytest.fixture
def short_tmp():
    """A scratch directory short enough to hold an `AF_UNIX` socket."""
    directory = Path(tempfile.mkdtemp(prefix="plv", dir="/tmp"))  # noqa: S108
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


#: Holds the writer role and reports what happened, so a parent can assert on
#: a *second* process's outcome rather than on a same-process call that shares
#: this one's file descriptors. `flock` is per-open-file-description: two
#: `single_writer` calls inside one interpreter would each get their own
#: descriptor and would in fact conflict, but relying on that would leave the
#: cross-process case -- the only one that matters in production -- untested.
_HOLDER = """
import sys, time
from pathlib import Path
from palaver.observer.socket import single_writer

db_path = Path(sys.argv[1])
try:
    with single_writer(db_path) as server:
        print("HELD", flush=True)
        time.sleep(float(sys.argv[2]))
except Exception as exc:
    print(f"REFUSED {type(exc).__name__}: {exc}", flush=True)
    sys.exit(3)
"""


def _holder(db_path, seconds=30.0):
    """Start a writer-role holder and wait until it actually holds it."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(db_path), str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = _line_within(proc, 30.0)
    assert line == "HELD", f"holder did not take the role: {line!r}"
    return proc


def _line_within(proc, deadline):
    """One line of stdout, or a failure -- never an unbounded wait."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(proc.stdout.readline).result(timeout=deadline).strip()
    except concurrent.futures.TimeoutError:
        proc.kill()
        raise AssertionError(f"no output within {deadline}s") from None
    finally:
        pool.shutdown(wait=False)


def test_a_second_daemon_refuses_to_start_while_the_first_still_serves(short_tmp):
    """The property the whole architecture rests on.

    Two writers on one SQLite file is not a performance problem, it is a
    correctness one, and nothing downstream can detect it after the fact.
    """
    db_path = short_tmp / "palaver.db"
    first = _holder(db_path)
    try:
        second = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(db_path), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode != 0, "a second daemon started alongside the first"
        assert "DaemonAlreadyRunningError" in second.stdout
        assert first.poll() is None, "the first daemon died, so this proved nothing"
    finally:
        first.kill()
        first.wait(timeout=10)


def test_the_writer_role_is_released_when_the_first_daemon_exits(short_tmp):
    """The positive control for the test above.

    Without it, "the second process exited non-zero" would be equally
    consistent with a lock that can never be taken by anyone.
    """
    db_path = short_tmp / "palaver.db"
    first = _holder(db_path, seconds=0.1)
    first.wait(timeout=30)

    second = subprocess.run(
        [sys.executable, "-c", _HOLDER, str(db_path), "0.1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 0, f"the role never came back: {second.stdout}{second.stderr}"
    assert "HELD" in second.stdout


def test_a_stale_socket_node_is_unlinked_and_replaced_under_the_held_lock(short_tmp):
    """A crash leaves the node behind; the next daemon must not be blocked by it.

    The node is created by binding and abandoning a socket without ever
    listening, which is what a daemon killed between `bind` and `listen`
    leaves on disk: a filesystem entry that refuses every connect.
    """
    db_path = short_tmp / "palaver.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = writer_socket.socket_path_for(db_path)

    abandoned = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    abandoned.bind(str(socket_path))
    abandoned.close()  # the node outlives the socket
    assert socket_path.exists(), "the fixture did not leave a stale node"
    stale_inode = socket_path.stat().st_ino

    with single_writer(db_path) as server:
        assert socket_path.exists()
        assert socket_path.stat().st_ino != stale_inode, "the stale node was reused, not replaced"
        # Bound *and* listening: a node that exists but refuses is exactly the
        # state this test started from, so existence alone proves nothing.
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(5)
        try:
            client.connect(str(socket_path))
        finally:
            client.close()
        assert server.fileno() >= 0


def test_a_live_socket_is_never_unlinked_even_when_its_owner_holds_no_lock(short_tmp):
    """The reason the probe exists alongside the lock.

    A pathname socket keeps serving through its owner's descriptor no matter
    what happens to the name, so unlinking one that still has a listener does
    not stop it -- it just frees the name for a second listener nobody can
    see. This stands in for an older build, or a daemon started by hand.
    """
    db_path = short_tmp / "palaver.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path = writer_socket.socket_path_for(db_path)

    squatter = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    squatter.bind(str(socket_path))
    squatter.listen(4)
    inode = socket_path.stat().st_ino
    try:
        with pytest.raises(DaemonAlreadyRunningError, match="without holding"):
            with single_writer(db_path):
                pass
        assert socket_path.stat().st_ino == inode, "the live socket's node was unlinked"
    finally:
        squatter.close()


def test_a_data_directory_on_an_unverified_filesystem_stops_startup(short_tmp, monkeypatch):
    """`flock` returning success is not evidence that it locked anything.

    On NFS without lockd, and on some FUSE and SMB mounts, it is a no-op --
    which is indistinguishable from a working lock at the call site. The
    filesystem name is the only signal available, so an unrecognised one has
    to fail closed.
    """
    db_path = short_tmp / "palaver.db"
    monkeypatch.setattr(writer_socket, "filesystem_type", lambda _path: "nfs")

    with pytest.raises(NonLocalFilesystemError, match="nfs"):
        with single_writer(db_path):
            pass

    assert not writer_socket.socket_path_for(db_path).exists(), (
        "a socket was bound before the check"
    )


def test_the_filesystem_check_is_an_allowlist_not_a_denylist(short_tmp, monkeypatch):
    """A filesystem nobody here has tested must fail, not pass by omission.

    A denylist would admit every filesystem invented after this line was
    written, which is the population most likely to break `flock`.
    """
    db_path = short_tmp / "palaver.db"
    monkeypatch.setattr(writer_socket, "filesystem_type", lambda _path: "somethingnew")
    with pytest.raises(NonLocalFilesystemError, match="somethingnew"):
        with single_writer(db_path):
            pass


def test_this_repository_lives_on_a_filesystem_the_allowlist_accepts():
    """The positive control for both tests above.

    They monkeypatch `filesystem_type`, so together they would still pass if
    the real one returned garbage for every path. This one calls it for real.
    """
    fstype = writer_socket.filesystem_type(Path(__file__).parent)
    assert fstype in writer_socket.LOCAL_FILESYSTEMS, f"unexpected filesystem {fstype!r}"


def test_the_statfs_struct_matches_the_layout_it_was_checked_against():
    """A ctypes layout that drifts reads a plausible string from the wrong offset.

    `statfs` would still return 0, and the wrong bytes would still decode --
    a confident answer from the wrong field, which is INV-7's shape. The size
    is the cheapest check that catches it.
    """
    assert ctypes.sizeof(writer_socket._Statfs) == writer_socket._STATFS_SIZE


def test_a_socket_path_over_the_sun_path_limit_is_named_rather_than_raised_raw(short_tmp):
    """`OSError: AF_UNIX path too long` names neither the limit nor the path.

    `sun_path` is a fixed 104-byte array, so 103 bytes is the most that can
    be bound -- bisected, not assumed. A user whose project sits deep enough
    to cross that gets a kernel error naming none of: which path, how long,
    or what the ceiling is. Reported here instead, with all three.
    """
    deep = short_tmp / ("d" * 120)
    with pytest.raises(writer_socket.SocketPathTooLongError) as caught:
        writer_socket.socket_path_for(deep / "palaver.db")
    message = str(caught.value)
    assert str(writer_socket.MAX_SOCKET_PATH_BYTES) in message
    assert "palaver.sock" in message


def test_a_path_at_the_limit_binds_and_one_byte_over_does_not(short_tmp):
    """The positive control: the limit is the measured one, off by nothing.

    Without this, `MAX_SOCKET_PATH_BYTES` could be any conservative number
    and the test above would still pass -- including one so low it refused
    paths that work perfectly well.
    """
    limit = writer_socket.MAX_SOCKET_PATH_BYTES
    # `<short_tmp>` + `/` + padding + `/x` + `/palaver.sock`
    padding = limit - len(str(short_tmp)) - len("/") - len("/x") - len("/palaver.sock")
    assert padding > 0, "the scratch directory is already too long to test the boundary"

    at_limit = short_tmp / ("x" * padding) / "x"
    at_limit.mkdir(parents=True)
    path = writer_socket.socket_path_for(at_limit / "palaver.db")
    assert len(str(path)) == limit

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))  # the kernel agrees this length is fine
    finally:
        server.close()

    with pytest.raises(writer_socket.SocketPathTooLongError):
        writer_socket.socket_path_for(at_limit / "yy" / "palaver.db")


def test_a_store_too_deep_for_a_socket_still_gets_a_writer(tmp_path):
    """Degrade the request channel; never degrade the observing.

    `palaver observe` exists to keep watching. A path too deep for
    `sun_path` costs it corrections and a liveness probe -- it must not cost
    it the daemon. Refusing to start here would convert a missing feature
    into a total outage, and it would do so on exactly the machines whose
    projects are nested deepest.

    `tmp_path`, deliberately: pytest's own scratch path is already over the
    limit, so this is the real configuration rather than a contrived one.
    """
    said: list[str] = []
    with pytest.raises(writer_socket.SocketPathTooLongError):
        writer_socket.socket_path_for(tmp_path / "palaver.db")  # the premise

    with single_writer(tmp_path / "palaver.db", on_status=said.append) as server:
        assert server is None, "a socket was bound at a path the kernel cannot hold"
        # The role is still held, which is the whole point -- proven from a
        # second process, since `flock` is per-open-file-description and a
        # same-process retry would conflict for the wrong reason.
        refused = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(tmp_path / "palaver.db"), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert refused.returncode != 0, "the writer role was not held"
        assert "DaemonAlreadyRunningError" in refused.stdout

    assert any("write requests are disabled" in line for line in said), (
        f"the daemon went quiet about its missing request channel: {said}"
    )
    assert any(str(writer_socket.MAX_SOCKET_PATH_BYTES) in line for line in said), (
        "the warning named no limit, so nobody reading it knows what to fix"
    )


def test_serving_without_a_socket_sleeps_the_window_out_and_serves_nothing(tmp_path):
    """The idle window has to behave the same either way.

    A degraded daemon that returned from its idle window immediately would
    spin the tick loop at whatever rate the CPU allows -- observation would
    survive, but the machine would not.
    """
    slept: list[float] = []
    served = writer_socket.serve_until(None, None, 7.5, sleep=slept.append)
    assert served == 0
    assert slept == [7.5], "the idle window was not spent asleep"


def test_serving_with_a_socket_still_answers_a_request(short_tmp):
    """The positive control for the two tests above.

    Without it, `serve_until` could return 0 unconditionally and both
    degradation tests would still pass -- while no correction ever landed.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    conn = connect(db_path)
    try:
        with single_writer(db_path) as server:
            assert server is not None
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                pending = pool.submit(
                    writer_socket.request,
                    db_path,
                    {"op": "correct", "memory_id": memory_id, "statement": "served while idle"},
                )
                # Scripted clock: `serve_until` serves for the *whole*
                # window, so a real 30-second one would cost 30 seconds
                # after the request it is meant to prove. Third reading is
                # past the deadline, which also exercises the recomputation.
                ticks = iter([0.0, 0.0, 100.0])
                served = writer_socket.serve_until(
                    server, conn, 30.0, monotonic=lambda: next(ticks)
                )
                assert served == 1
                reply = pending.result(timeout=10)
                assert reply["ok"], reply
            finally:
                pool.shutdown(wait=False)
    finally:
        conn.close()
    assert _raw_row(db_path, reply["memory_id"])["statement"] == "served while idle"


# ---------------------------------------------------------------------------
# The write path: what it will do, what it refuses, and what it never touches.
# ---------------------------------------------------------------------------

#: A daemon that actually serves. Holds the writer role, opens the one
#: writable connection, and answers requests until told to stop -- which is
#: the only way to test `request()` against something that can really reply.
_SERVER = """
import sys
from pathlib import Path
from palaver.observer.socket import single_writer, serve_request
from palaver.store.migrate import connect

db_path = Path(sys.argv[1])
conn = connect(db_path)
try:
    with single_writer(db_path) as server:
        print("HELD", flush=True)
        while True:
            serve_request(server, conn)
finally:
    conn.close()
"""


def _seed_memory(db_path):
    """One project, one session, one chunk, one memory at tier 4."""
    from palaver.memory.evidence import EvidenceAnchor
    from palaver.memory.write import write_memory
    from palaver.store.migrate import connect, migrate

    migrate(db_path)
    conn = connect(db_path)
    project_id = conn.execute(
        "INSERT INTO projects (name, path) VALUES (?, ?) RETURNING id",
        ("demo", str(db_path.parent)),
    ).fetchone()[0]
    session_id = conn.execute(
        "INSERT INTO sessions (project_id, source, external_id) VALUES (?, ?, ?) RETURNING id",
        (project_id, "claude-code", "session-aaa"),
    ).fetchone()[0]
    chunk_id = conn.execute(
        "INSERT INTO transcript_chunks (session_id, seq, role, content) VALUES (?, ?, ?, ?) "
        "RETURNING id",
        (session_id, 0, "assistant", "the recorded evidence text"),
    ).fetchone()[0]
    memory_id = write_memory(
        conn,
        project_id=project_id,
        session_id=session_id,
        statement="the observer's original reading",
        origin="observer",
        tier=4,
        evidence=[EvidenceAnchor(start_offset=0, end_offset=8, transcript_chunk_id=chunk_id)],
    )
    conn.commit()
    conn.close()
    return memory_id


def _raw_row(db_path, memory_id):
    """Every column of one memory, read with a bare sqlite3 connection.

    Deliberately not through Palaver's own helpers: a test that asserts
    immutability using the same layer it is testing can be fooled by that
    layer. `sqlite3` sees what is actually on disk.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def test_a_correction_writes_a_new_row_and_leaves_the_original_byte_identical(short_tmp):
    """INV-4: supersede, never edit. Asserted column by column.

    Checking only the statement would miss a correction that rewrote the
    predecessor's tier or origin in passing, and INV-5 makes tier the field
    most worth watching.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    before = _raw_row(db_path, memory_id)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "what really happened"}
        )
    finally:
        conn.close()

    assert reply["ok"], reply
    assert reply["supersedes"] == memory_id
    assert _raw_row(db_path, memory_id) == before, "the corrected row was modified in place"

    successor = _raw_row(db_path, reply["memory_id"])
    assert successor["statement"] == "what really happened"
    assert successor["supersedes"] == memory_id
    assert successor["tier"] == 1, "a correction is a user instruction, the highest tier"
    assert successor["origin"] == writer_socket.CORRECTION_ORIGIN


def test_the_correction_inherits_the_evidence_rather_than_inventing_any(short_tmp):
    """INV-6 without a fabricated citation.

    A correction reinterprets the same span of transcript, so it points at
    the same anchors. Synthesising an anchor to satisfy the non-empty rule
    would put a citation in the store leading somewhere the statement does
    not come from -- worse than no citation, because it reads as grounded.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "corrected"}
        )
    finally:
        conn.close()

    raw = sqlite3.connect(db_path)
    try:
        original = raw.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (memory_id,),
        ).fetchall()
        inherited = raw.execute(
            "SELECT start_offset, end_offset, transcript_chunk_id, event_id "
            "FROM memory_evidence WHERE memory_id = ? ORDER BY id",
            (reply["memory_id"],),
        ).fetchall()
    finally:
        raw.close()
    assert inherited == original
    assert inherited, "a memory with no evidence would violate INV-6"


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "update", "memory_id": 1, "statement": "rewritten"},
        {"op": "delete", "memory_id": 1},
        {"op": "sql", "sql": "UPDATE memories SET statement = 'x' WHERE id = 1"},
        {"op": "sql", "sql": "DELETE FROM memories WHERE id = 1"},
        {"memory_id": 1, "statement": "no op at all"},
    ],
)
def test_an_update_or_delete_naming_a_memory_is_refused_by_the_write_path(short_tmp, payload):
    """There is no request shape that expresses an edit.

    The protocol carries an operation *name* and typed arguments, never SQL
    and never a table or column name, so this is not a filter that could be
    bypassed with different spelling -- there is nothing to spell.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    payload = {**payload, "memory_id": memory_id} if "memory_id" in payload else payload
    before = _raw_row(db_path, memory_id)

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(conn, payload)
    finally:
        conn.close()

    assert reply["ok"] is False
    assert reply["error"] == "UnsupportedOperationError"
    assert _raw_row(db_path, memory_id) == before
    assert _row_count(db_path) == 1, "a refused request still wrote something"


def test_an_ordinary_correction_lands_on_that_same_connection(short_tmp):
    """The positive control for the refusals above.

    Without it, every one of them would pass against a write path that
    refused *everything* -- including a connection opened read-only by
    mistake, or a store with no writable table at all.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    conn = connect(db_path)
    try:
        refused = writer_socket.apply_request(conn, {"op": "delete", "memory_id": memory_id})
        accepted = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "this one lands"}
        )
    finally:
        conn.close()

    assert refused["ok"] is False
    assert accepted["ok"] is True
    assert _row_count(db_path) == 2, "the correction did not write its row"


def _row_count(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    finally:
        conn.close()


def test_a_correction_travels_over_the_real_socket_to_a_real_daemon(short_tmp):
    """End to end: a separate process holds the writer role and applies it.

    Every in-process test above shares this interpreter's connection, which
    is exactly the arrangement production does not have. This one has the
    writer in another process, reached only through the socket.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line_within(proc, 30.0) == "HELD"
        assert writer_socket.daemon_running(db_path) is True

        reply = writer_socket.request(
            db_path, {"op": "correct", "memory_id": memory_id, "statement": "over the wire"}
        )
        assert reply["ok"], reply
        assert _raw_row(db_path, reply["memory_id"])["statement"] == "over the wire"

        refused = writer_socket.request(db_path, {"op": "delete", "memory_id": memory_id})
        assert refused["ok"] is False
        assert refused["error"] == "UnsupportedOperationError"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_a_write_with_no_daemon_fails_loudly_rather_than_opening_a_second_writer(short_tmp):
    """The refusal is the feature.

    A fallback to a direct write would be invisible and would end the
    single-writer guarantee precisely when the daemon is already unhealthy
    -- the moment it is least safe to have two processes writing.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    assert writer_socket.daemon_running(db_path) is False

    with pytest.raises(writer_socket.DaemonUnavailableError, match="second"):
        writer_socket.request(
            db_path, {"op": "correct", "memory_id": memory_id, "statement": "no daemon"}
        )
    assert _row_count(db_path) == 1, "a write happened with no daemon running"


#: The MCP server, as `palaver mcp` runs it.
_MCP_SERVE = """
import sys
from palaver.cli import main
db, port = sys.argv[1], sys.argv[2]
sys.argv = ["palaver", "mcp", "--db", db, "--port", port]
sys.exit(main())
"""


async def _correct_over_the_wire(url, memory_id, statement, *, approve):
    """Drive `palaver_correct` the way a real client would, sign-off included."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import ElicitResult

    prompts = []

    async def elicitation_callback(_ctx, params):
        prompts.append(params.message)
        return ElicitResult(action="accept", content={"approved": approve, "note": "checked"})

    async with streamable_http_client(url) as streams:
        async with ClientSession(
            streams[0], streams[1], elicitation_callback=elicitation_callback
        ) as session:
            await session.initialize()
            result = await session.call_tool(
                "palaver_correct", {"memory_id": memory_id, "statement": statement}
            )
            return result, prompts


@pytest.mark.parametrize("approve", [True, False])
def test_a_correction_crosses_the_real_transport_and_the_real_socket(short_tmp, approve):
    """The whole path, with nothing stubbed: client, server, daemon, store.

    Every other test here replaces at least one link -- an in-process
    connection, a stub context. This one has a real client eliciting over a
    real streamable-HTTP back-channel, a real MCP server holding a `mode=ro`
    connection, and a real daemon in a third process applying the write. It
    is the only test that would catch a break in how those three meet.

    Parametrised on the answer because the negative case is the one that
    matters: a sign-off gate that writes regardless is worse than none, and
    it looks identical from the accept side.
    """
    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    port = _free_port()

    daemon = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = None
    try:
        assert _line_within(daemon, 30.0) == "HELD"
        server = subprocess.Popen(
            [sys.executable, "-c", _MCP_SERVE, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        url = _line_within(server, 30.0)
        assert url.startswith("http://"), url

        result, prompts = asyncio.run(
            asyncio.wait_for(
                _correct_over_the_wire(url, memory_id, "corrected over the wire", approve=approve),
                timeout=60,
            )
        )
    finally:
        for proc in (server, daemon):
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)

    assert prompts, "the client was never asked to sign off"
    assert "the observer's original reading" in prompts[0], "the prompt hid what changes"

    if approve:
        assert not result.is_error, result.content[0].text
        assert _row_count(db_path) == 2
        successor = json.loads(result.content[0].text)
        assert _raw_row(db_path, successor["memory_id"])["statement"] == "corrected over the wire"
    else:
        assert result.is_error, "a refused sign-off returned success"
        assert _row_count(db_path) == 1, "the store was written despite a refusal"


def test_a_client_that_cannot_elicit_is_refused_rather_than_written_for(short_tmp):
    """No back-channel means no sign-off, and no sign-off means no write.

    The SDK reports this as `MCPError: Elicitation not supported` -- named
    and immediate, not a hang. Failing closed is the only safe direction: a
    memory rewritten at tier 1 because nobody could be asked is exactly the
    confidently-wrong state the tier system exists to prevent.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)
    port = _free_port()

    daemon = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(db_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = None
    try:
        assert _line_within(daemon, 30.0) == "HELD"
        server = subprocess.Popen(
            [sys.executable, "-c", _MCP_SERVE, str(db_path), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        url = _line_within(server, 30.0)

        async def call_without_elicitation():
            # No `elicitation_callback`, so the client declares no such
            # capability -- which is what an older or simpler client is.
            async with streamable_http_client(url) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool(
                        "palaver_correct",
                        {"memory_id": memory_id, "statement": "written without consent"},
                    )

        result = asyncio.run(asyncio.wait_for(call_without_elicitation(), timeout=60))
        message = result.content[0].text
    finally:
        for proc in (server, daemon):
            if proc is not None:
                proc.kill()
                proc.wait(timeout=10)

    assert result.is_error, "a client that cannot ask its user got a write anyway"
    # The SDK's own wording is "Elicitation not supported", which names
    # neither the correction nor whether it landed. The refusal has to say
    # both, or a caller is left guessing about the state of their store.
    assert "was not corrected" in message, message
    assert "nothing was written" in message, message
    assert _row_count(db_path) == 1


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


#: Takes the writer lock and nothing else -- no socket, no listener. The only
#: thing that can refuse a daemon against this is the `flock` itself.
_LOCK_ONLY = """
import fcntl, os, sys, time
from pathlib import Path
from palaver.observer.socket import lock_path_for

db_path = Path(sys.argv[1])
db_path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(lock_path_for(db_path), os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("LOCKED", flush=True)
time.sleep(float(sys.argv[2]))
"""


def test_the_lock_alone_refuses_a_second_daemon_when_no_socket_exists_yet(short_tmp):
    """The case the lock exists for, and the one the other tests never reach.

    `test_a_second_daemon_refuses_to_start_while_the_first_still_serves` was
    passing on the *connect probe*: the first daemon was already listening,
    so the intruder was turned away by a successful connect and the `flock`
    was never the reason. Deleting the `flock` outright left that test green
    -- which mutation testing found and no amount of reading would have.

    Here there is no socket at all, which is the state two daemons racing
    from cold start are both in. The lock is the only thing that can refuse,
    so if it is not taken, the second daemon binds and there are two writers.
    """
    db_path = short_tmp / "palaver.db"
    socket_path = writer_socket.socket_path_for(db_path)

    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_ONLY, str(db_path), "30"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _line_within(holder, 30.0) == "LOCKED"
        assert not socket_path.exists(), "the fixture must leave no socket to probe"

        second = subprocess.run(
            [sys.executable, "-c", _HOLDER, str(db_path), "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert second.returncode != 0, "a second daemon started with the lock already held"
        assert "DaemonAlreadyRunningError" in second.stdout, second.stdout + second.stderr
        assert not socket_path.exists(), "the refused daemon bound a socket anyway"
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_daemon_starts_once_that_lock_is_released(short_tmp):
    """The positive control: the refusal above is the lock, not the path.

    Without this, a `single_writer` that refused every start for an
    unrelated reason -- a permissions problem on the lock file, say --
    would satisfy the test above perfectly.
    """
    db_path = short_tmp / "palaver.db"
    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_ONLY, str(db_path), "0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert _line_within(holder, 30.0) == "LOCKED"
    holder.wait(timeout=30)

    started = subprocess.run(
        [sys.executable, "-c", _HOLDER, str(db_path), "0.1"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert started.returncode == 0, started.stdout + started.stderr
    assert "HELD" in started.stdout


def test_correcting_a_memory_with_no_evidence_names_the_evidence_as_the_problem(short_tmp):
    """INV-6 is refused either way; what is under test is the diagnosis.

    Without the explicit check the write still fails -- `write_memory`
    raises on empty evidence -- but it reports "evidence must not be empty"
    about a memory the caller never mentioned, which sends a reader looking
    at the wrong row. The store is equally safe and the message is useless.
    """
    from palaver.store.migrate import connect

    db_path = short_tmp / "palaver.db"
    memory_id = _seed_memory(db_path)

    # INV-6 is enforced in Python, not by the schema (TASKS.md Task 13's
    # sibling), so a row with no evidence is reachable on a raw connection --
    # which is exactly the inconsistent store this branch reports on.
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DELETE FROM memory_evidence WHERE memory_id = ?", (memory_id,))
        raw.commit()
    finally:
        raw.close()

    conn = connect(db_path)
    try:
        reply = writer_socket.apply_request(
            conn, {"op": "correct", "memory_id": memory_id, "statement": "no evidence to inherit"}
        )
    finally:
        conn.close()

    assert reply["ok"] is False
    assert "evidence to inherit" in reply["detail"], reply
    assert str(memory_id) in reply["detail"], "the message names no memory"
    assert _row_count(db_path) == 1
