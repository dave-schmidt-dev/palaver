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

import concurrent.futures
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    THROTTLE_INTERVAL_SECONDS,
    bootout,
    print_service,
    wait_for_running_pid,
)

#: Never `DEFAULT_LABEL`. See the module docstring.
SELFTEST_LABEL = "com.zerodelta.palaver.observe.selftest"

#: The same isolation rule for task 6.5's MCP agent: never `MCP_LABEL`, so a
#: live test cannot bootout the server an agent is currently registered
#: against. Its port is allocated per run rather than fixed, because unlike
#: the observer this job binds one — and 8787 is the port real clients hold.
MCP_SELFTEST_LABEL = "com.zerodelta.palaver.mcp.selftest"


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


live = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("launchctl") is None,
    reason="launchd user agents exist only on macOS with launchctl on PATH",
)


def _await_pid(label: str, *, window: float = STARTUP_WINDOW_SECONDS) -> int | None:
    """Wait for a job to reach a pid this test can actually signal.

    Not `service_pid`. launchd publishes the pid while it is still the root
    `xpcproxy` stub, and killing that raises `EPERM` — which is exactly how
    this test failed the first time it ran.
    """
    return wait_for_running_pid(label, window=window)


def _await_unloaded(label: str, *, window: float = STARTUP_WINDOW_SECONDS) -> bool:
    """Wait until launchd no longer knows the label at all.

    `bootout` returns as soon as launchd has *accepted* the request, not once
    the job is gone. Bootstrapping a label still being torn down fails with
    `Bootstrap failed: 5: Input/output error`, and because every live test in
    this file reuses one label per service, that lands on whichever test runs
    next rather than on the one that left the job behind.

    This was a real intermittent failure, not a precaution: a whole-suite run
    lost `test_an_agent_without_keepalive_stays_dead_when_killed` to exactly
    that error while the same test passed in isolation minutes earlier. It is
    worse for jobs under `KeepAlive` that these tests SIGKILL, because
    teardown can then race a restart that launchd has already scheduled.
    """
    deadline = time.monotonic() + window
    while time.monotonic() < deadline:
        if print_service(label).returncode != 0:
            return True
        time.sleep(0.2)
    return False


def _bootout_and_settle(label: str, *, strict: bool = True) -> None:
    """Unload a label and do not return until launchd has finished with it.

    The docstring above is a promise this function cannot always keep, so the
    failure to keep it is reported rather than discarded. An earlier version
    called `_await_unloaded` and dropped its return value, which meant a
    teardown that timed out looked identical to one that succeeded — and the
    symptom surfaced as `Bootstrap failed: 5` in an unrelated test, which is
    precisely the confusion `_await_unloaded` was added to end.

    Args:
        label: launchd label to unload.
        strict: Raise on timeout rather than warn. True at setup, where a
            label that is still going away invalidates the test about to run.
            False at teardown, where raising would replace the test's own
            result — a passing test would be reported as an error, and a
            failing one would lose its actual assertion message.

    Raises:
        AssertionError: `strict` and the label outlived the window.
    """
    bootout(label)
    if _await_unloaded(label):
        return
    message = (
        f"{label}: still known to launchd {STARTUP_WINDOW_SECONDS:.0f}s after bootout. "
        f"The next bootstrap of this label will probably fail with "
        f"'Bootstrap failed: 5: Input/output error'."
    )
    if strict:
        raise AssertionError(message)
    # `warnings.warn`, not `print`: pytest captures and discards stdout from a
    # passing test, so a printed warning here would be invisible in exactly
    # the case it needs to be seen — a green run that left a job behind.
    warnings.warn(message, stacklevel=2)


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


def _row_count(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    finally:
        conn.close()
