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
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
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
from palaver.observer import socket as writer_socket
from palaver.observer.socket import (
    DaemonAlreadyRunningError,
    NonLocalFilesystemError,
    single_writer,
)

#: Never `DEFAULT_LABEL`. See the module docstring.
SELFTEST_LABEL = "com.zerodelta.palaver.observe.selftest"

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
        assert writer_socket.daemon_alive(db_path) is True

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
    assert writer_socket.daemon_alive(db_path) is False

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
