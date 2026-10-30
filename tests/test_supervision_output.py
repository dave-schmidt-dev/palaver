"""Headless integration of both launchd templates and bounded stream capture."""

import os
import plistlib
import signal
import subprocess
import sys
import time

import pytest

from palaver import logging_setup
from palaver.cli.install_agent import SERVICES, render_plist
from palaver.supervision.output import capture


@pytest.mark.parametrize("service", SERVICES.values(), ids=SERVICES)
def test_rendered_job_captures_startup_crashes_before_app_imports(tmp_path, service):
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    child = [sys.executable, "-c", "print('result', flush=True); raise RuntimeError('startup')"]
    plist = plistlib.loads(render_plist(
        program_arguments=child, stdout_path=out, stderr_path=err,
        template_path=service.template_path,
    ).encode())
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == os.devnull
    assert plist["ProgramArguments"][5:] == child
    result = subprocess.run(plist["ProgramArguments"], capture_output=True, timeout=15)
    assert result.returncode == 1
    assert result.stdout == result.stderr == b""
    assert out.read_text() == "result\n"
    assert "RuntimeError: startup" in err.read_text()
    assert "service exited 1" in err.read_text()


@pytest.mark.parametrize("payload", ["b'x'", "'€'.encode()", "b'\\xff'"])
def test_repeated_crash_floods_remain_bounded_without_newlines(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(logging_setup, "MAX_LOG_BYTES", 16384)
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    code = f"import os; os.write(1,{payload}*200000); os.write(2,{payload}*200000); exit(9)"
    for _ in range(8):
        assert capture([sys.executable, "-c", code], out, err) == 9
    for stream in ("out", "err"):
        files = list(tmp_path.glob(f"{stream}.log*"))
        assert len(files) == logging_setup.LOG_BACKUPS + 1
        assert all(p.stat().st_size <= 16384 for p in files)
    assert "exited 9" in err.read_text()


def test_partial_utf8_and_invalid_bytes_are_drained(tmp_path):
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    code = "import os; os.write(1,b'a'*4095+b'\\xe2\\x82\\xac'); os.write(2,b'partial\\xff')"
    assert capture([sys.executable, "-c", code], out, err) == 0
    assert out.read_text() == "a" * 4095 + "€"
    assert "partial�" in err.read_text()


def test_missing_executable_is_visible_and_fails(tmp_path):
    err = tmp_path / "err.log"
    assert capture([str(tmp_path / "missing")], tmp_path / "out.log", err) == 127
    assert "cannot start service" in err.read_text()


def test_signal_reaches_child_and_final_output_is_drained(tmp_path):
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    code = (
        "import signal,time,sys; "
        "signal.signal(signal.SIGTERM,lambda *_: (print('stopped',flush=True),sys.exit(0))); "
        "print('ready',flush=True); time.sleep(30)"
    )
    process = subprocess.Popen([
        sys.executable, "-m", "palaver.supervision.output", str(out), str(err),
        sys.executable, "-c", code,
    ])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if out.exists() and "ready" in out.read_text():
                break
            time.sleep(0.02)
        else:
            pytest.fail("child never became ready")
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        assert out.read_text() == "ready\nstopped\n"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_failed_output_sink_prevents_child_start(tmp_path):
    marker = tmp_path / "started"
    with pytest.raises(OSError):
        capture([sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
                tmp_path, tmp_path / "err.log")
    assert not marker.exists()


def test_log_write_failure_is_not_swallowed_and_reaps_child(tmp_path, monkeypatch):
    from palaver.supervision import output

    original_factory = output.rotating_handler
    original_popen = subprocess.Popen
    children = []

    class FailedStream:
        def tell(self):
            return 0

        def write(self, text):
            raise OSError("disk failure")

        def flush(self):
            pass

        def close(self):
            pass

    def factory(path):
        handler = original_factory(path)
        if path.name == "out.log":
            handler.stream.close()
            handler.stream = FailedStream()
        return handler

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(output, "rotating_handler", factory)
    monkeypatch.setattr(output.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="log write failed"):
        capture([sys.executable, "-c", "import time; print('hello',flush=True); time.sleep(30)"],
                tmp_path / "out.log", tmp_path / "err.log")
    assert len(children) == 1
    assert children[0].poll() is not None
