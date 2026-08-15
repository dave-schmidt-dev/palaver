"""Task 4.2: KV slot management and `palaver doctor --server-cmdline`.

Every test here runs against an in-process stub bound to an ephemeral
`127.0.0.1` port, which is what the plan's amendment asked for: the machine
Palaver runs on may or may not have `llama-server` up, and if it is up it is
almost certainly *not* running with `--slot-save-path` (this one is not, as of
2026-08-15), so a suite that needed a live server would be a suite that skipped.

The stub speaks the real protocol, taken from `tools/server/server-context.cpp`
and `tools/server/server-task.cpp` rather than from memory: `/props` with
`total_slots` and a nested `default_generation_settings.n_ctx`, a save response
of `{id_slot, filename, n_saved, n_written, timings.save_ms}`, a restore
response of `{id_slot, filename, n_restored, n_read, timings.restore_ms}`, and
the exact 501 body a server started without a slot-save path returns.

Two pairings run through the file and are the reason it is longer than the
done-when list:

* **Every negative assertion has a positive control.** "The save path raises a
  named precondition error" passes trivially on a client that names every
  non-200, so the generic-500 control sits next to it. "A failed `/slots` probe
  falls back to one slot" passes trivially on a client that always reports one,
  so the four-slot success case sits next to it.
* **The capability probe is asserted to be a probe.** Not just that it returns
  the right answer, but that the server never saw `action=save` while it was
  answering — a diagnostic that writes a KV dump to disk to find out whether it
  can write a KV dump to disk is not a diagnostic.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from palaver.cli import SUBCOMMANDS, build_parser, doctor
from palaver.extract.client import (
    ModelClientError,
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)
from palaver.extract.slots import (
    CAPABILITY_PROBE_ACTION,
    DEFAULT_SLOT_COUNT,
    PROPS_PATH,
    SLOTS_PATH,
    SlotClient,
    SlotSaveUnsupportedError,
)

#: The server's own words when it was started without `--slot-save-path`,
#: copied from `tools/server/server-context.cpp` and confirmed against the live
#: server on this machine. Reproduced verbatim because `palaver doctor` prints
#: the message it receives rather than a paraphrase, and a test that asserted a
#: paraphrase would not notice the paraphrase drifting from the instruction the
#: operator actually has to follow.
UNSUPPORTED_MESSAGE = "This server does not support slots action. Start it with `--slot-save-path`"

#: Token count the stub reports for a save, and therefore the count a restore of
#: the same file must return.
SAVED_TOKENS = 512


# --- stub server -------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, handle, seen, *args, **kwargs):
        self._handle = handle
        self._seen = seen
        super().__init__(*args, **kwargs)

    def log_message(self, format_string, *args) -> None:
        pass  # BaseHTTPRequestHandler logs to stderr by default; keep the suite quiet

    def _dispatch(self, method: str) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self._seen.append((method, self.path, body))
        status, payload = self._handle(method, self.path, body)
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")


class _StubServer:
    """An in-process HTTP server bound to an ephemeral 127.0.0.1 port (INV-9)."""

    def __init__(self, handle):
        self.seen: list[tuple[str, str, bytes]] = []

        def _factory(*args, **kwargs):
            return _Handler(handle, self.seen, *args, **kwargs)

        self._server = HTTPServer(("127.0.0.1", 0), _factory)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def stub_server():
    """Yield `start(handle) -> _StubServer`; every server is closed after the test."""
    servers: list[_StubServer] = []

    def _start(handle) -> _StubServer:
        server = _StubServer(handle)
        servers.append(server)
        return server

    yield _start

    for server in servers:
        server.close()


# --- protocol fixtures --------------------------------------------------------


def _props(**overrides) -> dict:
    """A `/props` body in the shape the real endpoint returns."""
    payload = {
        "build_info": "10360 (48d22e295)",
        "model_path": "/models/fixture-model.gguf",
        "model_alias": "fixture-model",
        "model_ftype": 2,
        "total_slots": 4,
        "endpoint_slots": True,
        "endpoint_props": True,
        "is_sleeping": False,
        "default_generation_settings": {"n_ctx": 32768},
    }
    payload.update(overrides)
    return payload


def _error(code: int, message: str, type_: str) -> dict:
    """The `{"error": {...}}` envelope `format_error_response` produces."""
    return {"error": {"code": code, "message": message, "type": type_}}


def _slot_entries(count: int, *, processing: int = 0) -> list[dict]:
    return [
        {"id": index, "is_processing": index < processing, "n_ctx": 8192} for index in range(count)
    ]


def _action_of(path: str) -> str:
    return path.split("action=", 1)[1] if "action=" in path else ""


def _slot_of(path: str) -> int:
    return int(path.split("/")[2].split("?")[0])


def _supporting_server(*, props: dict | None = None, slots: list[dict] | None = None):
    """A stub that supports slot save and restore, backed by an in-memory store.

    Mirrors the real handler's order of business: the `--slot-save-path`
    precondition would come first (it passes here), then the slot id, then the
    action — so an action it cannot dispatch is a 400, exactly as upstream.
    """
    saved: dict[str, int] = {}

    def handle(method: str, path: str, body: bytes):
        if method == "GET" and path == PROPS_PATH:
            return 200, _props() if props is None else props
        if method == "GET" and path == SLOTS_PATH:
            return 200, _slot_entries(4) if slots is None else slots
        if method == "POST" and path.startswith(f"{SLOTS_PATH}/"):
            action = _action_of(path)
            id_slot = _slot_of(path)
            if action == "save":
                filename = json.loads(body)["filename"]
                saved[filename] = SAVED_TOKENS
                return 200, {
                    "id_slot": id_slot,
                    "filename": filename,
                    "n_saved": SAVED_TOKENS,
                    "n_written": 4096,
                    "timings": {"save_ms": 33.0},
                }
            if action == "restore":
                filename = json.loads(body)["filename"]
                if filename not in saved:
                    return 500, _error(500, "Unable to restore slot", "server_error")
                return 200, {
                    "id_slot": id_slot,
                    "filename": filename,
                    "n_restored": saved[filename],
                    "n_read": 4096,
                    "timings": {"restore_ms": 21.0},
                }
            return 400, _error(400, "Invalid action", "invalid_request_error")
        return 404, _error(404, "File Not Found", "not_found_error")

    return handle, saved


def _unsupporting_server(*, props: dict | None = None):
    """A stub started without `--slot-save-path`: every slot action is a 501."""

    def handle(method: str, path: str, body: bytes):
        if method == "GET" and path == PROPS_PATH:
            return 200, _props() if props is None else props
        if method == "GET" and path == SLOTS_PATH:
            return 200, _slot_entries(4)
        if method == "POST" and path.startswith(f"{SLOTS_PATH}/"):
            return 501, _error(501, UNSUPPORTED_MESSAGE, "not_supported_error")
        return 404, _error(404, "File Not Found", "not_found_error")

    return handle


def _client(server: _StubServer) -> SlotClient:
    return SlotClient(host="127.0.0.1", port=server.port, timeout=5.0)


# --- /props ------------------------------------------------------------------


def test_a_probe_of_props_parses_the_slot_count_the_stub_reports(stub_server):
    """The done-when bullet, stated exactly: the parsed count equals the stub's value.

    Asserted against two different stub values, not one. A client that hardcoded
    4 — or that returned `DEFAULT_SLOT_COUNT` — would satisfy a single-value
    assertion, and both are failure modes this module could plausibly have.
    """
    for reported in (4, 7):
        handle, _ = _supporting_server(props=_props(total_slots=reported))
        server = stub_server(handle)
        assert _client(server).slot_count() == reported


def test_a_probe_of_props_parses_the_rest_of_the_effective_configuration(stub_server):
    """Every field `palaver doctor` prints comes from `/props`, parsed not guessed."""
    handle, _ = _supporting_server()
    server = stub_server(handle)

    properties = _client(server).properties()

    assert properties.total_slots == 4
    assert properties.build_info == "10360 (48d22e295)"
    assert properties.model_path == "/models/fixture-model.gguf"
    assert properties.model_alias == "fixture-model"
    assert properties.n_ctx == 32768
    assert properties.endpoint_slots is True


def test_a_probe_of_props_records_an_unreported_context_window_as_none(stub_server):
    """An absent `n_ctx` is `None`, never a plausible default.

    The positive control is the test above: with the field present it parses to
    32768, so `None` here is the absence being recorded and not the parse being
    broken in both directions.
    """
    payload = _props()
    del payload["default_generation_settings"]
    handle, _ = _supporting_server(props=payload)
    server = stub_server(handle)

    assert _client(server).properties().n_ctx is None


def test_a_probe_of_props_without_a_slot_count_is_a_response_error(stub_server):
    """`total_slots` is the one field with no safe fallback, so its absence raises."""
    payload = _props()
    del payload["total_slots"]
    handle, _ = _supporting_server(props=payload)
    server = stub_server(handle)

    with pytest.raises(ModelResponseError, match="total_slots"):
        _client(server).properties()


def test_slot_count_refuses_to_invent_a_number_when_props_is_unreachable():
    """`slot_count` raises where `probe_slots` falls back, and that asymmetry is deliberate.

    Live slot state is advisory — not having it makes a scheduler uninformed.
    The configured slot count is not advisory: a wrong one silently over-commits
    the server, which is the exact "unexplained latency" this task exists to turn
    into a visible diagnostic.
    """
    client = SlotClient(host="127.0.0.1", port=1, timeout=1.0)
    with pytest.raises(ModelConnectionError):
        client.slot_count()


# --- /slots ------------------------------------------------------------------


def test_a_failed_slots_probe_warns_and_falls_back_to_a_single_slot(stub_server, caplog):
    """The done-when bullet: a `/slots` failure warns and degrades, it does not raise."""
    handle, _ = _supporting_server()

    # `--no-slots` makes the endpoint answer 501; any failure takes this path.
    def refusing(method: str, path: str, body: bytes):
        if path == SLOTS_PATH:
            return 501, _error(
                501, "This server does not support slots endpoint.", "not_supported_error"
            )
        return handle(method, path, body)

    server = stub_server(refusing)

    with caplog.at_level(logging.WARNING, logger="palaver.extract.slots"):
        slots = _client(server).probe_slots()

    assert len(slots) == 1
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert SLOTS_PATH in caplog.records[0].getMessage()


def test_a_successful_slots_probe_does_not_fall_back(stub_server, caplog):
    """The positive control for the fallback: four slots stay four, with no warning."""
    handle, _ = _supporting_server(slots=_slot_entries(4, processing=2))
    server = stub_server(handle)

    with caplog.at_level(logging.WARNING, logger="palaver.extract.slots"):
        slots = _client(server).probe_slots()

    assert [slot.id for slot in slots] == [0, 1, 2, 3]
    assert [slot.is_processing for slot in slots] == [True, True, False, False]
    assert caplog.records == []


def test_an_empty_slots_probe_falls_back_rather_than_reporting_zero_slots(stub_server, caplog):
    """Zero slots is not an answer a scheduler can use, so it degrades like a failure."""
    handle, _ = _supporting_server(slots=[])
    server = stub_server(handle)

    with caplog.at_level(logging.WARNING, logger="palaver.extract.slots"):
        slots = _client(server).probe_slots()

    assert len(slots) == 1
    assert caplog.records != []


def test_the_slots_fallback_is_exactly_one_slot():
    """The done-when bullet says a *single* slot, so the constant is pinned here.

    The two tests above deliberately assert the literal `1` rather than
    `DEFAULT_SLOT_COUNT`. Asserting against the constant is self-referential: it
    holds for whatever value the module happens to carry, and mutating that
    constant to 4 survived exactly that assertion. The literal catches the
    fallback drifting; this catches the constant drifting.
    """
    assert DEFAULT_SLOT_COUNT == 1


# --- save and restore ---------------------------------------------------------


def test_a_save_then_a_restore_returns_the_token_count_the_save_reported(stub_server):
    """The done-when bullet: the round trip conserves the token count.

    Written as a real round trip through the stub's store, not two independent
    reads of one constant: the restore only answers because the save wrote, and
    the count it answers with is the one the save recorded.
    """
    handle, saved = _supporting_server()
    server = stub_server(handle)
    client = _client(server)

    save = client.save(2, "session-7.bin")
    restore = client.restore(2, "session-7.bin")

    assert save.n_saved == SAVED_TOKENS
    assert restore.n_restored == save.n_saved
    assert saved == {"session-7.bin": SAVED_TOKENS}
    assert save.id_slot == 2
    assert save.filename == "session-7.bin"
    assert save.save_ms == 33.0
    assert restore.restore_ms == 21.0
    assert restore.n_read == 4096


def test_a_restore_of_a_file_no_save_wrote_is_a_response_error(stub_server):
    """The negative half of the round trip: a 500 from the server stays a 500-shaped error."""
    handle, _ = _supporting_server()
    server = stub_server(handle)

    with pytest.raises(ModelResponseError, match="HTTP 500"):
        _client(server).restore(0, "never-saved.bin")


def test_save_raises_a_named_precondition_error_when_the_capability_is_absent(stub_server):
    """The done-when bullet: a named precondition error, and *not* a transport error.

    The three exclusions are the assertion, not decoration. `SlotSaveUnsupported
    Error` is a sibling of the transport classes rather than a subclass, so a
    caller catching `ModelConnectionError` to retry does not catch a condition
    that will never change on a retry.
    """
    server = stub_server(_unsupporting_server())

    with pytest.raises(SlotSaveUnsupportedError) as caught:
        _client(server).save(0, "session-7.bin")

    assert isinstance(caught.value, ModelClientError)
    assert not isinstance(caught.value, ModelConnectionError)
    assert not isinstance(caught.value, ModelTimeoutError)
    assert not isinstance(caught.value, ModelResponseError)
    assert "--slot-save-path" in str(caught.value)
    assert "not_supported_error" in str(caught.value)


def test_restore_raises_the_same_named_precondition_error(stub_server):
    """Save and restore share one classification, so they cannot drift apart."""
    server = stub_server(_unsupporting_server())

    with pytest.raises(SlotSaveUnsupportedError, match="--slot-save-path"):
        _client(server).restore(0, "session-7.bin")


def test_a_generic_server_error_is_a_response_error_not_a_precondition_error(stub_server):
    """The control that makes the test above mean something.

    Without this, "save raises the named precondition error" also passes on a
    client that classified every non-200 as unsupported — which would report a
    crashed server as a configuration choice.
    """

    def failing(method: str, path: str, body: bytes):
        if method == "POST" and path.startswith(f"{SLOTS_PATH}/"):
            return 500, _error(500, "Unable to save slot", "server_error")
        return 200, _props()

    server = stub_server(failing)

    with pytest.raises(ModelResponseError) as caught:
        _client(server).save(0, "session-7.bin")
    assert not isinstance(caught.value, SlotSaveUnsupportedError)


def test_a_precondition_error_survives_an_unparseable_error_body(stub_server):
    """The status decides; the body is reported detail, not the discriminator.

    llama.cpp sets HTTP 501 and `type: "not_supported_error"` in the same switch
    arm, so keying on the body would buy no independent evidence and would break
    on any build that changed the wording. This pins the choice: garbage body,
    same named error.
    """

    def garbled(method: str, path: str, body: bytes):
        if method == "POST" and path.startswith(f"{SLOTS_PATH}/"):
            return 501, b"<html>not json</html>"
        return 200, _props()

    server = stub_server(garbled)

    with pytest.raises(SlotSaveUnsupportedError, match="HTTP 501"):
        _client(server).save(0, "session-7.bin")


# --- the capability probe ------------------------------------------------------


def test_the_capability_probe_reports_absence_in_the_servers_own_words(stub_server):
    """An unsupporting server yields `supported=False` carrying its own instruction."""
    server = stub_server(_unsupporting_server())

    support = _client(server).slot_save_support()

    assert support.supported is False
    assert support.detail == UNSUPPORTED_MESSAGE


def test_the_capability_probe_reports_support_when_the_sentinel_action_is_rejected(stub_server):
    """A supporting server rejects the sentinel as an invalid action, which is a yes."""
    handle, _ = _supporting_server()
    server = stub_server(handle)

    support = _client(server).slot_save_support()

    assert support.supported is True
    assert CAPABILITY_PROBE_ACTION in support.detail


def test_the_capability_probe_never_performs_a_save(stub_server):
    """The probe must not write a KV dump to disk to discover that it could.

    Asserted from the server's side: across both a supporting and an
    unsupporting server, the only slot action it ever saw was the sentinel.
    """
    handle, saved = _supporting_server()
    supporting = stub_server(handle)
    unsupporting = stub_server(_unsupporting_server())

    _client(supporting).slot_save_support()
    _client(unsupporting).slot_save_support()

    for server in (supporting, unsupporting):
        actions = [
            _action_of(path) for _, path, _ in server.seen if path.startswith(f"{SLOTS_PATH}/")
        ]
        assert actions == [CAPABILITY_PROBE_ACTION]
    assert saved == {}


def test_an_uninterpretable_capability_probe_status_is_a_response_error(stub_server):
    """Neither 501 nor 400 means the probe failed, which is not the same as a no.

    Collapsing this into `supported=False` would report a broken or proxied
    server as a deliberate configuration choice.
    """

    def odd(method: str, path: str, body: bytes):
        if method == "POST" and path.startswith(f"{SLOTS_PATH}/"):
            return 403, _error(403, "forbidden", "permission_error")
        return 200, _props()

    server = stub_server(odd)

    with pytest.raises(ModelResponseError, match="HTTP 403"):
        _client(server).slot_save_support()


# --- palaver doctor -------------------------------------------------------------


def _doctor(server: _StubServer, *, extra: list[str] | None = None):
    """Run `palaver doctor --server-cmdline` through the real parser, capturing both streams."""
    parser = build_parser()
    args = parser.parse_args(
        ["doctor", "--server-cmdline", "--port", str(server.port), "--timeout", "5", *(extra or [])]
    )
    out = io.StringIO()
    status: list[str] = []
    code = doctor.run(args, out=out, on_status=status.append)
    return code, out.getvalue(), status


def test_doctor_is_registered_as_a_subcommand():
    """The registration in `palaver/cli/__init__.py` is the last edit of this task."""
    assert doctor in SUBCOMMANDS
    assert doctor.NAME == "doctor"


def test_doctor_reports_the_slot_count_and_an_absent_save_capability(stub_server):
    """The two done-when bullets that name `doctor`'s output, on the common case.

    An unsupporting server is the common case — it is what this machine runs
    today — and the report has to name the missing capability rather than
    surface it as an opaque transport error or omit it.
    """
    server = stub_server(_unsupporting_server())

    code, stdout, _ = _doctor(server)

    assert code == 0
    assert "slots: 4" in stdout
    assert "slot save/restore: unavailable" in stdout
    assert UNSUPPORTED_MESSAGE in stdout
    assert "slots observed: 4 (0 processing)" in stdout
    assert "context: 32768 tokens" in stdout
    assert "build: 10360 (48d22e295)" in stdout


def test_doctor_reports_an_available_save_capability(stub_server):
    """The positive control: the capability line tracks the server, not a constant."""
    handle, _ = _supporting_server()
    server = stub_server(handle)

    code, stdout, _ = _doctor(server)

    assert code == 0
    assert "slot save/restore: available" in stdout
    assert "unavailable" not in stdout


def test_doctor_reports_the_slot_count_the_server_reports_not_a_default(stub_server):
    """A server with seven slots prints seven, so the line is a measurement."""
    handle, _ = _supporting_server(props=_props(total_slots=7), slots=_slot_entries(7))
    server = stub_server(handle)

    _, stdout, _ = _doctor(server)

    assert "slots: 7" in stdout
    assert "slots observed: 7 (0 processing)" in stdout


def test_doctor_never_claims_to_be_showing_a_command_line(stub_server):
    """`--server-cmdline` cannot show an invocation, and the report says so.

    `/props` carries no argv, cmdline, or params object. A command that took the
    flag's name literally would have to invent the answer, so the note is part
    of the contract rather than a comment.
    """
    server = stub_server(_unsupporting_server())

    _, stdout, _ = _doctor(server)

    assert doctor.NO_CMDLINE_NOTE in stdout
    assert "observed property" in stdout


def test_doctor_exits_nonzero_when_the_server_cannot_be_reached():
    """An unreachable server is a failed run, not a report full of blanks."""
    parser = build_parser()
    args = parser.parse_args(["doctor", "--server-cmdline", "--port", "1", "--timeout", "1"])
    out = io.StringIO()

    code = doctor.run(args, out=out, on_status=lambda _message: None)

    assert code == 1
    assert out.getvalue() == ""


def test_doctor_emits_progress_on_the_status_channel_and_not_on_stdout(stub_server):
    """INV-1: every request announces itself, and never through the result stream."""
    server = stub_server(_unsupporting_server())

    _, stdout, status = _doctor(server)

    assert status, "doctor made three HTTP requests and announced none of them"
    assert any(PROPS_PATH in message for message in status)
    for message in status:
        assert message not in stdout


def test_doctor_survives_a_capability_probe_it_cannot_interpret(stub_server):
    """A probe failure is reported as unknown, distinct from a reported absence."""

    def odd(method: str, path: str, body: bytes):
        if method == "GET" and path == PROPS_PATH:
            return 200, _props()
        if method == "GET" and path == SLOTS_PATH:
            return 200, _slot_entries(4)
        return 403, _error(403, "forbidden", "permission_error")

    server = stub_server(odd)

    code, stdout, _ = _doctor(server)

    assert code == 0
    assert "slot save/restore: unknown" in stdout
    assert "unavailable" not in stdout


def test_the_probe_quick_check_selects_these_tests():
    """`uv run pytest -q tests/test_slots.py -k probe` must select a real subset.

    The quick check is only worth running if it runs something. This asserts the
    selector's population directly rather than trusting the naming to have held,
    which is the same guard task 7.3's `-k coverage_gate` needed after that
    selector was found to collect zero tests.
    """
    import tests.test_slots as module

    selected = [name for name in vars(module) if name.startswith("test_") and "probe" in name]
    assert len(selected) >= 10, selected
