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

import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    DEFAULT_LABEL,
    RESTART_WINDOW_SECONDS,
    THROTTLE_INTERVAL_SECONDS,
    InstallAgentError,
    bootout,
    bootstrap,
    domain_target,
    observe_program_arguments,
    pid_is_ours,
    print_service,
    render_plist,
    service_pid,
    service_target,
    wait_for_new_pid,
    wait_for_running_pid,
)

#: Never `DEFAULT_LABEL`. See the module docstring.
SELFTEST_LABEL = "com.zerodelta.palaver.observe.selftest"

#: Tick fast enough that a restarted job has visibly done something, slow
#: enough that a two-second test window does not accumulate ticks.
SELFTEST_INTERVAL = "5"

#: Seconds to wait for a freshly bootstrapped job to have a pid.
STARTUP_WINDOW_SECONDS = 10.0

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
            "label": DEFAULT_LABEL,
            "db": None,
            "cursors": None,
            "interval": None,
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
    assert original is not None, "the job never started"

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
        assert original is not None, "the job never started"
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
