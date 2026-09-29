"""
Leg identity and pinning, assert_legs_distinct, resolve_llama_server_binary,
managed_e2b_server's process teardown across every exit path, and the anti-vacuity E4B
inference check.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from palaver.eval.harness import (
    E2B_LEG,
    E4B_LEG,
    FixtureLabel,
    LegConfig,
    assert_legs_distinct,
    managed_e2b_server,
    resolve_llama_server_binary,
    run_eval,
)
from palaver.extract.client import ModelClient
from tests._eval_support import _CONFORMING_EXTRACTION, FIXTURES_DIR, _wrap
from tests._eval_support import store_conn as store_conn
from tests._eval_support import stub_server as stub_server

# =============================================================================
# Leg identity, structural pins (advisor point 5, half 2: leg identity, not port)
# =============================================================================


def test_e4b_leg_is_pinned_to_8090_and_unmanaged():
    """E4B is the pre-existing server: fixed port, and this module never holds a Popen for it."""
    assert E4B_LEG.port == 8090
    assert E4B_LEG.host == "127.0.0.1"
    assert E4B_LEG.managed is False


def test_e2b_leg_is_pinned_to_8091_and_managed():
    """E2B is the leg this harness starts and stops itself."""
    assert E2B_LEG.port == 8091
    assert E2B_LEG.host == "127.0.0.1"
    assert E2B_LEG.managed is True


# =============================================================================
# assert_legs_distinct: mutation guard for "point both legs at the same GGUF"
# =============================================================================


def test_assert_legs_distinct_accepts_the_real_pinned_pair():
    """Positive control: the actual E4B/E2B pair task 3.5 pins must pass this guard."""
    assert_legs_distinct(E4B_LEG, E2B_LEG)  # must not raise


def test_assert_legs_distinct_rejects_same_model_path():
    same_path = E4B_LEG.model_path
    e2b_pointed_at_e4b = LegConfig(
        name="E2B", host="127.0.0.1", port=8091, model_path=same_path, model_name="x", managed=True
    )
    with pytest.raises(ValueError, match="different GGUFs"):
        assert_legs_distinct(E4B_LEG, e2b_pointed_at_e4b)


def test_assert_legs_distinct_rejects_same_port():
    e2b_on_e4b_port = LegConfig(
        name="E2B",
        host="127.0.0.1",
        port=8090,
        model_path=E2B_LEG.model_path,
        model_name="x",
        managed=True,
    )
    with pytest.raises(ValueError, match="different ports"):
        assert_legs_distinct(E4B_LEG, e2b_on_e4b_port)


def test_assert_legs_distinct_rejects_non_eval_gguf():
    bogus = LegConfig(
        name="E2B",
        host="127.0.0.1",
        port=8091,
        model_path=Path("/Users/dave/models/gemma-4-repaired/gemma-4-26B_q4_0-it.gguf"),
        model_name="x",
        managed=True,
    )
    with pytest.raises(ValueError, match="not an eval leg"):
        assert_legs_distinct(E4B_LEG, bogus)


# =============================================================================
# resolve_llama_server_binary
# =============================================================================


def test_resolve_llama_server_binary_returns_which_result(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/fake/bin/llama-server")
    assert resolve_llama_server_binary() == "/fake/bin/llama-server"


def test_resolve_llama_server_binary_raises_when_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        resolve_llama_server_binary()


# =============================================================================
# managed_e2b_server: guaranteed teardown (task 3.5's Done-when bullet)
# =============================================================================


class _FakeProcess:
    def __init__(self):
        self.terminate_calls = 0
        self.wait_calls = 0
        self.kill_calls = 0

    def terminate(self):
        self.terminate_calls += 1

    def wait(self, timeout=None):
        self.wait_calls += 1

    def kill(self):
        self.kill_calls += 1


def test_managed_e2b_server_terminates_process_when_body_raises():
    """The Done-when bullet: teardown happens even when the E2B leg raises."""
    fake_process = _FakeProcess()

    with pytest.raises(RuntimeError, match="boom"):
        with managed_e2b_server(
            E2B_LEG,
            binary="fake-llama-server",
            popen=lambda *a, **k: fake_process,
            check_health=lambda host, port: True,
        ):
            raise RuntimeError("boom")

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1


def test_managed_e2b_server_terminates_process_on_clean_exit():
    """Positive control for the above: teardown is not an artifact of the exception path."""
    fake_process = _FakeProcess()

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=lambda host, port: True,
    ) as process:
        assert process is fake_process

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1


def test_managed_e2b_server_raises_timeout_and_still_tears_down():
    fake_process = _FakeProcess()

    with pytest.raises(TimeoutError):
        with managed_e2b_server(
            E2B_LEG,
            binary="fake-llama-server",
            popen=lambda *a, **k: fake_process,
            check_health=lambda host, port: False,
            health_timeout=0.05,
            poll_interval=0.01,
        ):
            raise AssertionError("body must never run when health never reports ok")

    assert fake_process.terminate_calls == 1


def test_managed_e2b_server_polls_health_until_ready():
    fake_process = _FakeProcess()
    calls = {"count": 0}

    def flaky_health(host, port):
        calls["count"] += 1
        return calls["count"] >= 3

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=flaky_health,
        poll_interval=0.01,
    ):
        pass

    assert calls["count"] >= 3
    assert fake_process.terminate_calls == 1


def test_managed_e2b_server_never_calls_the_real_subprocess_module(monkeypatch):
    """Positive control that the injected `popen` is actually what gets called.

    Without this, a `managed_e2b_server` that silently ignored the `popen`
    argument and shelled out to the real `subprocess.Popen` would still pass
    every test above, since none of them inspect what started the process.
    """

    def _real_popen_must_not_be_called(*args, **kwargs):
        raise AssertionError("the real subprocess.Popen must not be called in this test")

    monkeypatch.setattr(subprocess, "Popen", _real_popen_must_not_be_called)
    fake_process = _FakeProcess()

    with managed_e2b_server(
        E2B_LEG,
        binary="fake-llama-server",
        popen=lambda *a, **k: fake_process,
        check_health=lambda host, port: True,
    ):
        pass

    assert fake_process.terminate_calls == 1


# =============================================================================
# Anti-vacuity: E4B leg gets ordinary inference, never a shutdown-shaped call
# =============================================================================


def test_e4b_leg_receives_inference_but_no_shutdown_call(stub_server, store_conn):
    """Task 3.5's named vacuity trap, addressed directly.

    Checking only "no shutdown call was recorded" would pass trivially if
    `run_eval`'s client wiring never reached the E4B leg's server at all.
    This test also asserts the positive half in the same body: the E4B
    leg's recorded request set contains at least one
    `/v1/chat/completions` POST, so the negative half is meaningful.

    The E4B leg's stub stands in for port 8090 by *role* (it is passed as
    `e4b_client` to `run_eval`, the same parameter production code points at
    `E4B_LEG`), not by literal port number -- `test_e4b_leg_is_pinned_to_8090_and_unmanaged`
    above pins the real port/managed-ness separately, since binding an
    ephemeral test port to 8090 itself is not possible.
    """
    e4b_requests: list[tuple[str, str]] = []
    e2b_requests: list[tuple[str, str]] = []
    e4b_port = stub_server(e4b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))
    e2b_port = stub_server(e2b_requests, lambda: _wrap(_CONFORMING_EXTRACTION))

    labels = (
        FixtureLabel(
            id="bookkeeping-only",
            path="bookkeeping-only.jsonl",
            expect_question=False,
            expect_blocker=False,
            expect_current_task=False,
            expect_decision=False,
            expect_completion=False,
        ),
    )
    e4b_client = ModelClient(store_conn, port=e4b_port, timeout=5.0)
    e2b_client = ModelClient(store_conn, port=e2b_port, timeout=5.0)

    run_eval(labels, FIXTURES_DIR, e4b_client=e4b_client, e2b_client=e2b_client)

    e4b_paths = {path for _method, path in e4b_requests}
    # Positive control: the wiring actually reached the E4B leg's server.
    assert len(e4b_requests) >= 1
    # Negative: an allowlist of paths actually seen, not a blacklist of
    # shutdown-ish names -- the only path a `ModelClient` ever POSTs to.
    assert e4b_paths == {"/v1/chat/completions"}
    assert all(method == "POST" for method, _path in e4b_requests)
