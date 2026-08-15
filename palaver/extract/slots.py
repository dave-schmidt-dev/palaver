"""KV slot management against the pre-existing `llama-server` (Task 4.2).

Palaver does not manage the server process (plan section 5). It does, however,
have to share that server's KV slots with whatever else is talking to it, and a
slot count that silently differs from the assumed one shows up as unexplained
latency rather than as an error. This module is the read side of that problem —
`/props` for what the server actually is, `/slots` for what it is doing right
now — plus the save/restore pair that pages one session's KV cache out and back
in between ticks.

**Everything here was verified against the llama.cpp server source on
2026-08-15**, at `tools/server/server-context.cpp` and
`tools/server/server-common.cpp`, and against the live server on this machine.
Three findings shaped the code:

1. **`/props` carries no command line.** Its keys are exactly `bos_token`,
   `build_info`, `chat_template`, `chat_template_caps`, `cors_proxy_enabled`,
   `default_generation_settings`, `endpoint_metrics`, `endpoint_props`,
   `endpoint_slots`, `eos_token`, `is_sleeping`, `media_marker`, `modalities`,
   `model_alias`, `model_ftype`, `model_path`, `total_slots`, `ui`,
   `ui_settings`. There is no `argv`, no `cmdline`, and no `params` object. So
   `palaver doctor --server-cmdline` cannot report the invocation and does not
   pretend to; it reports the *effective configuration* the server exposes,
   which is what the diagnostic was actually for. See `palaver/cli/doctor.py`.

2. **Slot save/restore is off unless the server was started with
   `--slot-save-path`,** and the check for it sits at the very top of the
   `post_slots` handler, ahead of both the slot-id parse and the action
   dispatch. When it fails the server answers HTTP 501 with
   `{"error": {"code": 501, "message": "This server does not support slots
   action. Start it with \\`--slot-save-path\\`", "type":
   "not_supported_error"}}`. That is a *precondition*, not a transport failure
   and not an untrustworthy response, so it raises `SlotSaveUnsupportedError`
   rather than any of `palaver/extract/client.py`'s three error classes. The
   server on this machine today returns exactly that.

3. **501 and `not_supported_error` are the same fact.** `format_error_response`
   sets both in a single `case ERROR_TYPE_NOT_SUPPORTED` arm, so neither is
   independent corroboration of the other. The classification below keys on the
   status code, which is the coarser and more stable of the two, and carries the
   server's `type` and `message` into the exception text as reported detail
   rather than branching on them. A 500 with a generic body stays a
   `ModelResponseError`; `tests/test_slots.py` pins that as the control, since
   "raises the named error" also passes on a client that names every non-200.

**Capability probing must not write anything.** The obvious way to ask "does
save work" is to save, but on a server where it *does* work that writes a KV
dump to disk as a side effect of a diagnostic. Because the precondition check
runs before the action is parsed, a request carrying an action the server cannot
dispatch reaches one of exactly two outcomes and mutates nothing either way:
501 when the capability is absent, 400 `invalid_request_error` ("Invalid
action") when it is present. `CAPABILITY_PROBE_ACTION` is that sentinel.

Transport is `http.client`, matching `palaver/extract/client.py` for the reasons
its module docstring gives: no proxy-environment consultation, no runtime
dependency, and `tests/test_invariants.py`'s `FORBIDDEN_HTTP_MODULES` names
`urllib.request`, `httpx`, `requests`, and `openai` but not `http.client`.

Progress (INV-1) is the same surface-agnostic `on_status` callback the rest of
the project uses: a plain callable, defaulting to doing nothing, never a write
to stdout or stderr. This module imports neither stream.
"""

from __future__ import annotations

import http.client
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from palaver.extract.client import (
    ModelClientError,
    ModelConnectionError,
    ModelResponseError,
    ModelTimeoutError,
)

logger = logging.getLogger(__name__)

#: Server introspection. Returns the effective configuration, never the
#: invocation — see finding 1 in the module docstring.
PROPS_PATH = "/props"

#: Live per-slot state. Disabled by `--no-slots`, in which case reading it is a
#: failure rather than an empty list, which is why `probe_slots` falls back.
SLOTS_PATH = "/slots"

#: The count assumed when `/slots` cannot be read at all. One, not zero and not
#: `total_slots`: a caller that got no answer should behave as though it is the
#: only tenant of a single slot, which is the conservative scheduling choice.
#: Reporting a larger number on no evidence would invent parallelism.
DEFAULT_SLOT_COUNT = 1

#: An action `post_slots` cannot dispatch, used to ask whether slot save and
#: restore are available without performing either. Safe because llama.cpp
#: checks `params.slot_save_path` before it parses the action (verified in
#: `tools/server/server-context.cpp`, 2026-08-15): the request is refused at
#: the precondition when the capability is absent, and refused as an unknown
#: action when it is present. Neither branch touches a slot or the filesystem.
CAPABILITY_PROBE_ACTION = "palaver-capability-probe"

#: HTTP status llama.cpp uses for `ERROR_TYPE_NOT_SUPPORTED`, the only error
#: type the slot precondition raises.
_NOT_SUPPORTED_STATUS = 501

#: HTTP status llama.cpp uses for `ERROR_TYPE_INVALID_REQUEST`, which is what a
#: server that *does* support slot actions answers `CAPABILITY_PROBE_ACTION`
#: with ("Invalid action").
_INVALID_REQUEST_STATUS = 400


class SlotSaveUnsupportedError(ModelClientError):
    """Raised when the server refuses a slot action for want of `--slot-save-path`.

    Deliberately a sibling of `ModelConnectionError`, `ModelTimeoutError`, and
    `ModelResponseError` rather than a subclass of any of them. The server was
    reached, answered promptly, and answered *truthfully*: nothing about the
    exchange was untrustworthy, and nothing about it will change on a retry.
    A caller wanting to degrade from "page the KV cache" to "re-prefill every
    tick" catches this specifically; a caller catching a transport error would
    otherwise retry a precondition forever.
    """


@dataclass(frozen=True)
class ServerProperties:
    """The effective configuration `/props` reports for the running server.

    Attributes:
        total_slots: The server's `--parallel` value — how many requests it can
            hold KV state for at once. The number a scheduler has to respect.
        model_path: Filesystem path of the loaded GGUF.
        model_alias: The `--alias` the server answers to, or the empty string.
        build_info: Build number and commit, e.g. `"10360 (48d22e295)"`.
        n_ctx: Context window in tokens, read from
            `default_generation_settings.n_ctx`. `None` when the server did not
            report it — recorded as absent rather than guessed, because a wrong
            context size is worse than an unknown one.
        endpoint_slots: Whether `GET /slots` is enabled (`--slots`, on by
            default; `--no-slots` turns it off).
    """

    total_slots: int
    model_path: str
    model_alias: str
    build_info: str
    n_ctx: int | None
    endpoint_slots: bool


@dataclass(frozen=True)
class SlotState:
    """One entry of `GET /slots`.

    Only the two fields Palaver schedules against are modelled. The endpoint
    returns considerably more per slot, and deliberately none of it is carried
    here: this module has no business holding another tenant's prompt.

    Attributes:
        id: The slot index, as passed to `save` and `restore`.
        is_processing: Whether the server considers the slot busy right now.
    """

    id: int
    is_processing: bool


@dataclass(frozen=True)
class SlotSaveResult:
    """What `POST /slots/{id}?action=save` reported.

    Field names match the server's JSON exactly (`tools/server/server-task.cpp`),
    so a reader can line this up against a captured response without a mapping
    table.

    Attributes:
        id_slot: The slot that was saved.
        filename: Name written under the server's `--slot-save-path`.
        n_saved: Tokens of KV state written. The number `restore` must return.
        n_written: Bytes written.
        save_ms: Server-measured duration.
    """

    id_slot: int
    filename: str
    n_saved: int
    n_written: int
    save_ms: float


@dataclass(frozen=True)
class SlotRestoreResult:
    """What `POST /slots/{id}?action=restore` reported.

    Attributes:
        id_slot: The slot that was restored into.
        filename: Name read from under the server's `--slot-save-path`.
        n_restored: Tokens of KV state read back. Equals the `n_saved` of the
            save that wrote this file.
        n_read: Bytes read.
        restore_ms: Server-measured duration.
    """

    id_slot: int
    filename: str
    n_restored: int
    n_read: int
    restore_ms: float


@dataclass(frozen=True)
class SlotSaveSupport:
    """Whether the running server can save and restore slots, and how that was learned.

    Attributes:
        supported: True when the server dispatched the probe as an unknown
            action, meaning the `--slot-save-path` precondition passed.
        detail: The server's own words when it refused, or a short note on what
            the probe observed. Rendered verbatim by `palaver doctor`, so the
            operator reads llama.cpp's instruction ("Start it with
            `--slot-save-path`") rather than a paraphrase of it.
    """

    supported: bool
    detail: str


class SlotClient:
    """Read-only introspection plus save/restore against one `llama-server`.

    Every method opens exactly one connection to the single `host`/`port` the
    client was constructed with, which must stay `127.0.0.1` in any real
    deployment (INV-9). Nothing here starts, stops, or reconfigures the server.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        timeout: float = 10.0,
    ) -> None:
        """Configure a client against one llama-server instance.

        Args:
            host: Must stay `127.0.0.1` in any real deployment. Overridable for
                tests, which bind their stub server to `127.0.0.1` anyway.
            port: 8090, the plan's documented port for the pre-existing
                inference server.
            timeout: Seconds allowed for the socket to connect and for each
                read. Lower than `ModelClient`'s default because every request
                here is a small local introspection or a KV page measured in
                tens of milliseconds, not a generation.
        """
        self._host = host
        self._port = port
        self.timeout = timeout

    # --- introspection ---------------------------------------------------

    def properties(self, *, on_status: Callable[[str], None] | None = None) -> ServerProperties:
        """Read `/props` and return the server's effective configuration.

        Args:
            on_status: Progress channel (INV-1), called once before the request.

        Returns:
            The parsed `ServerProperties`.

        Raises:
            ModelConnectionError: The connection could not be established.
            ModelTimeoutError: Connecting or reading exceeded `timeout`.
            ModelResponseError: A non-200 status, a body that is not JSON, or a
                body missing `total_slots` — without which there is no answer to
                the one question this call exists to ask.
        """
        payload = self._get_json(PROPS_PATH, on_status=on_status)

        total_slots = payload.get("total_slots")
        if not isinstance(total_slots, int) or isinstance(total_slots, bool):
            raise ModelResponseError(
                f"{PROPS_PATH} did not report an integer total_slots: {total_slots!r}"
            )

        settings = payload.get("default_generation_settings")
        n_ctx = None
        if isinstance(settings, dict):
            candidate = settings.get("n_ctx")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                n_ctx = candidate

        return ServerProperties(
            total_slots=total_slots,
            model_path=_as_str(payload.get("model_path")),
            model_alias=_as_str(payload.get("model_alias")),
            build_info=_as_str(payload.get("build_info")),
            n_ctx=n_ctx,
            endpoint_slots=payload.get("endpoint_slots") is True,
        )

    def slot_count(self, *, on_status: Callable[[str], None] | None = None) -> int:
        """Return the number of KV slots the server reports.

        This is `/props.total_slots`, the server's `--parallel` value, and it is
        the number a scheduler must not exceed.

        Args:
            on_status: Progress channel (INV-1).

        Returns:
            The slot count.

        Raises:
            ModelClientError: Any failure reaching or trusting `/props`. Unlike
                `probe_slots`, this one does not fall back — a caller asking for
                the configured slot count has asked a question with no safe
                default, and inventing one would be the silent-wrong-answer
                failure this module exists to surface.
        """
        return self.properties(on_status=on_status).total_slots

    def probe_slots(
        self, *, on_status: Callable[[str], None] | None = None
    ) -> tuple[SlotState, ...]:
        """Read live per-slot state, falling back to a single idle slot.

        `GET /slots` is optional (`--no-slots` disables it) and reports
        transient state, so a caller that cannot read it is not blocked — it is
        merely uninformed. This logs a warning and returns one synthetic idle
        slot rather than raising, because the alternative is that an observer
        tick dies on a diagnostic endpoint being switched off.

        Args:
            on_status: Progress channel (INV-1).

        Returns:
            One `SlotState` per slot the server reported, or a single synthetic
            idle slot when `/slots` could not be read or was not a list.
        """
        try:
            payload = self._get_json(SLOTS_PATH, on_status=on_status)
        except ModelClientError as exc:
            logger.warning(
                "could not read %s from %s:%s (%s); assuming %d slot",
                SLOTS_PATH,
                self._host,
                self._port,
                exc,
                DEFAULT_SLOT_COUNT,
            )
            return _fallback_slots()

        if not isinstance(payload, list):
            logger.warning(
                "%s returned %s, not a list of slots; assuming %d slot",
                SLOTS_PATH,
                type(payload).__name__,
                DEFAULT_SLOT_COUNT,
            )
            return _fallback_slots()

        states: list[SlotState] = []
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                logger.warning("%s entry %d is not an object; skipping it", SLOTS_PATH, index)
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, int) or isinstance(identifier, bool):
                identifier = index
            states.append(
                SlotState(id=identifier, is_processing=entry.get("is_processing") is True)
            )
        if not states:
            logger.warning(
                "%s reported no usable slots; assuming %d", SLOTS_PATH, DEFAULT_SLOT_COUNT
            )
            return _fallback_slots()
        return tuple(states)

    def slot_save_support(
        self, *, id_slot: int = 0, on_status: Callable[[str], None] | None = None
    ) -> SlotSaveSupport:
        """Ask whether slot save and restore are available, without using them.

        Sends `CAPABILITY_PROBE_ACTION`, which the server cannot dispatch. See
        the module docstring: the `--slot-save-path` precondition is checked
        before the action is parsed, so this mutates nothing on either branch.

        Args:
            id_slot: Slot named in the probe. Never touched — the request is
                refused before the slot is used either way.
            on_status: Progress channel (INV-1).

        Returns:
            A `SlotSaveSupport` naming the answer and how it was reached.

        Raises:
            ModelConnectionError: The connection could not be established.
            ModelTimeoutError: Connecting or reading exceeded `timeout`.
            ModelResponseError: The server answered with something other than
                the two statuses this probe can interpret. Reported rather than
                collapsed into `supported=False`, because "the probe did not
                work" and "the capability is absent" are different facts.
        """
        status, raw = self._request(
            "POST",
            _action_path(id_slot, CAPABILITY_PROBE_ACTION),
            body=b"{}",
            on_status=on_status,
        )
        error = _parse_error(raw)
        if status == _NOT_SUPPORTED_STATUS:
            return SlotSaveSupport(supported=False, detail=error.message or "not supported")
        if status == _INVALID_REQUEST_STATUS:
            return SlotSaveSupport(
                supported=True,
                detail=(
                    f"the server rejected the {CAPABILITY_PROBE_ACTION!r} probe as an invalid "
                    "action, which means it passed the --slot-save-path precondition"
                ),
            )
        raise ModelResponseError(
            f"slot capability probe returned HTTP {status}, which is neither the "
            f"{_NOT_SUPPORTED_STATUS} the precondition raises nor the "
            f"{_INVALID_REQUEST_STATUS} an unknown action raises: {raw[:200]!r}"
        )

    # --- save and restore -------------------------------------------------

    def save(
        self, id_slot: int, filename: str, *, on_status: Callable[[str], None] | None = None
    ) -> SlotSaveResult:
        """Write one slot's KV cache to `filename` under the server's slot-save path.

        Args:
            id_slot: Slot to page out.
            filename: Bare filename, not a path. The server joins it onto its
                own `--slot-save-path` and validates it, so a traversal attempt
                is refused server-side as `Invalid filename`.
            on_status: Progress channel (INV-1).

        Returns:
            The parsed `SlotSaveResult`, whose `n_saved` is the token count a
            later `restore` of the same file must return.

        Raises:
            SlotSaveUnsupportedError: The server was started without
                `--slot-save-path`. A named precondition, not a transport
                failure — see the class docstring.
            ModelConnectionError: The connection could not be established.
            ModelTimeoutError: Connecting or reading exceeded `timeout`.
            ModelResponseError: Any other non-200 status, or a 200 whose body
                is not the documented save response.
        """
        payload = self._slot_action(id_slot, "save", filename, on_status=on_status)
        timings = payload.get("timings")
        return SlotSaveResult(
            id_slot=_require_int(payload, "id_slot", "save"),
            filename=_as_str(payload.get("filename")),
            n_saved=_require_int(payload, "n_saved", "save"),
            n_written=_require_int(payload, "n_written", "save"),
            save_ms=_as_float(timings.get("save_ms") if isinstance(timings, dict) else None),
        )

    def restore(
        self, id_slot: int, filename: str, *, on_status: Callable[[str], None] | None = None
    ) -> SlotRestoreResult:
        """Read `filename` back into one slot's KV cache.

        Args:
            id_slot: Slot to page in. Need not be the slot the file was saved
                from.
            filename: The bare filename a previous `save` wrote.
            on_status: Progress channel (INV-1).

        Returns:
            The parsed `SlotRestoreResult`, whose `n_restored` equals the
            `n_saved` of the save that wrote the file.

        Raises:
            SlotSaveUnsupportedError: The server was started without
                `--slot-save-path`.
            ModelConnectionError: The connection could not be established.
            ModelTimeoutError: Connecting or reading exceeded `timeout`.
            ModelResponseError: Any other non-200 status, or a 200 whose body is
                not the documented restore response.
        """
        payload = self._slot_action(id_slot, "restore", filename, on_status=on_status)
        timings = payload.get("timings")
        return SlotRestoreResult(
            id_slot=_require_int(payload, "id_slot", "restore"),
            filename=_as_str(payload.get("filename")),
            n_restored=_require_int(payload, "n_restored", "restore"),
            n_read=_require_int(payload, "n_read", "restore"),
            restore_ms=_as_float(timings.get("restore_ms") if isinstance(timings, dict) else None),
        )

    # --- transport --------------------------------------------------------

    def _slot_action(
        self,
        id_slot: int,
        action: str,
        filename: str,
        *,
        on_status: Callable[[str], None] | None,
    ) -> dict:
        """Post one slot action and return its parsed 200 body.

        Holds the single copy of the precondition classification, so `save` and
        `restore` cannot drift apart on which failures are named.
        """
        body = json.dumps({"filename": filename}).encode("utf-8")
        status, raw = self._request(
            "POST", _action_path(id_slot, action), body=body, on_status=on_status
        )
        if status == _NOT_SUPPORTED_STATUS:
            error = _parse_error(raw)
            raise SlotSaveUnsupportedError(
                f"llama-server at {self._host}:{self._port} cannot {action} slot {id_slot}: "
                f"{error.message or 'slot actions are not supported'} "
                f"(HTTP {status}, type {error.type or 'unset'})"
            )
        if status != 200:
            raise ModelResponseError(
                f"slot {action} returned HTTP {status} from {_action_path(id_slot, action)}: "
                f"{raw[:300]!r}"
            )
        payload = _decode_json(raw, f"slot {action} response")
        if not isinstance(payload, dict):
            raise ModelResponseError(
                f"slot {action} response is a {type(payload).__name__}, not a JSON object"
            )
        return payload

    def _get_json(self, path: str, *, on_status: Callable[[str], None] | None) -> object:
        """GET `path`, require a 200, and return the decoded body."""
        status, raw = self._request("GET", path, body=None, on_status=on_status)
        if status != 200:
            raise ModelResponseError(f"{path} returned HTTP {status}: {raw[:300]!r}")
        return _decode_json(raw, path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None,
        on_status: Callable[[str], None] | None,
    ) -> tuple[int, bytes]:
        """Perform one request and return `(status, body)` without judging the status.

        Status classification lives in the callers, because 501 means something
        specific on a slot action and nothing in particular on `/props`.
        """
        if on_status is not None:
            on_status(f"{method} {path} on {self._host}:{self._port}")
        connection = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
        try:
            connection.request(
                method,
                path,
                body=body,
                headers={"Content-Type": "application/json"} if body is not None else {},
            )
            response = connection.getresponse()
            return response.status, response.read()
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"{method} {path} on {self._host}:{self._port} timed out after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise ModelConnectionError(
                f"could not reach llama-server at {self._host}:{self._port}: {exc}"
            ) from exc
        finally:
            connection.close()


@dataclass(frozen=True)
class _ServerError:
    """The `{"error": {...}}` envelope llama.cpp returns on a non-200."""

    message: str
    type: str


def _parse_error(raw: bytes) -> _ServerError:
    """Pull `message` and `type` out of an error body, tolerating any shape.

    Never raises: this runs on a path that is already reporting a failure, and
    an unparseable error body must not replace the status code the caller is
    about to classify on.
    """
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError, UnicodeDecodeError:
        return _ServerError(message="", type="")
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return _ServerError(message="", type="")
    return _ServerError(message=_as_str(error.get("message")), type=_as_str(error.get("type")))


def _action_path(id_slot: int, action: str) -> str:
    return f"{SLOTS_PATH}/{id_slot}?action={action}"


def _fallback_slots() -> tuple[SlotState, ...]:
    return tuple(SlotState(id=index, is_processing=False) for index in range(DEFAULT_SLOT_COUNT))


def _decode_json(raw: bytes, where: str) -> object:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelResponseError(f"{where} is not valid JSON: {exc}") from exc


def _require_int(payload: dict, key: str, where: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ModelResponseError(f"{where} response field {key!r} is not an integer: {value!r}")
    return value


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
