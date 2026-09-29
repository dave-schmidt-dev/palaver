"""
Reading a pid out of launchctl output, and the restart-wait's tolerance for the root
stub and a mid-restart pid.
"""

from __future__ import annotations

import os
import subprocess

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    RESTART_WINDOW_SECONDS,
    THROTTLE_INTERVAL_SECONDS,
    pid_is_ours,
    service_pid,
    wait_for_new_pid,
    wait_for_running_pid,
)
from tests._supervision_support import _install_args

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


def test_waiting_for_an_unload_returns_once_launchctl_stops_knowing_the_label(monkeypatch):
    """The second call is what the caller is waiting for, not the first."""
    calls = []

    def fake_print(label, uid=None):
        calls.append(label)
        return subprocess.CompletedProcess([], 0 if len(calls) < 2 else 1, "", "")

    monkeypatch.setattr(install_agent, "print_service", fake_print)
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert install_agent.wait_for_unloaded("whatever", window=5.0, poll_interval=0.01) is True


def test_waiting_for_an_unload_reports_the_timeout_rather_than_claiming_success(monkeypatch):
    """A label that never goes away must return False.

    Returning True on timeout would put the bootstrap right back into the
    race the wait exists to close, while making the code read as if it had
    been closed.
    """
    monkeypatch.setattr(
        install_agent,
        "print_service",
        lambda label, uid=None: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    assert install_agent.wait_for_unloaded("whatever", window=0.2, poll_interval=0.01) is False


def test_waiting_for_an_unload_reports_progress(monkeypatch):
    """INV-1: this blocks for seconds, and a silent block reads as a hang."""
    monkeypatch.setattr(
        install_agent,
        "print_service",
        lambda label, uid=None: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(install_agent.time, "sleep", lambda _seconds: None)
    messages = []
    install_agent.wait_for_unloaded(
        "whatever", window=0.2, poll_interval=0.01, on_status=messages.append
    )
    assert messages, "the wait emitted nothing on the status channel"
    assert all("whatever" in message for message in messages)


def test_reloading_waits_for_the_unload_before_bootstrapping(tmp_path, monkeypatch):
    """The race the tests fixed for themselves was still open in the command.

    `bootout` returns on acceptance, not completion, so `--reload` could
    unload a job and then have its own bootstrap refused with `Bootstrap
    failed: 5`. For the observer that leaves the machine with no database
    writer at all — the unload half succeeds and the load half does not.

    Ordering is what this asserts, because both calls happen either way and a
    version that waited *after* bootstrapping would look identical in every
    other respect.
    """
    import io

    order = []
    monkeypatch.setattr(install_agent, "bootout", lambda label, uid=None: order.append("bootout"))
    monkeypatch.setattr(
        install_agent,
        "wait_for_unloaded",
        lambda label, **kwargs: (order.append("wait"), True)[1],
    )

    def fake_bootstrap(plist_path, uid=None):
        order.append("bootstrap")
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(install_agent, "bootstrap", fake_bootstrap)
    monkeypatch.setattr(install_agent, "service_pid", lambda label, uid=None: 4242)

    args = _install_args(
        reload=True, plist_path=tmp_path / "reload.plist", log_dir=tmp_path / "logs"
    )
    status = install_agent.run(args, out=io.StringIO(), on_status=lambda _message: None)

    assert status == 0
    assert order == ["bootout", "wait", "bootstrap"]
