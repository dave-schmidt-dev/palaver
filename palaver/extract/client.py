"""HTTP client for the pre-existing `llama-server`, schema-constrained per request (Task 3.2).

Palaver does not manage the `llama-server` process it talks to (plan section
5): it is started, stopped, and configured outside this project, and this
module's only job is to send one JSON request per call and come back with
either a parsed, schema-conforming object or a typed error a caller can
degrade on. It never returns a partial state object — a response this module
cannot fully trust is an exception, never a best-effort dict.

**Request shape — verified against the llama.cpp server parser source
(`tools/server/server-common.cpp`), not the README (orchestrator research,
2026-08-14).** The OpenAI-compatible `/v1/chat/completions` route reads
`response_format.json_schema.schema`, one level deeper than a flatter
`response_format.schema` shape that appears in some secondary summaries of
the README. Sending both `grammar` and `json_schema` is a hard server error
("Cannot use both json_schema and grammar"), not a precedence rule, so this
module only ever sends `json_schema`. Schemas carrying external `$ref` are
not supported by the server's built-in converter, so callers must pass
self-contained schemas; this module does not attempt to inline `$ref`s
itself. The locally installed build (`llama-server --version` reports
`10360 (48d22e295)`, matching `build_info` served by the running router
process on this machine's port 8080) matches the version the orchestrator's
source read was against, which is the one live check available without a
model actually loaded on 8090 — see this task's report for what that does
and doesn't verify.

**Why `http.client`, not `urllib.request` or a third-party library.**
`pyproject.toml` declares zero runtime dependencies, and this module keeps
it that way: nothing here needs retries, connection pooling, or streaming,
so a dependency would buy nothing a ~120-line stdlib client doesn't already
cover. Between the two stdlib options, `http.client.HTTPConnection` is
deliberately preferred over `urllib.request.urlopen`: `urlopen` consults
`http_proxy`/`HTTP_PROXY`-style environment variables by default (via
`urllib.request.getproxies()`) and can route a "localhost" request through
whatever proxy happens to be configured in the shell Palaver runs under.
`HTTPConnection` never looks at proxy environment variables — it connects to
exactly the host and port given, which is the stronger guarantee for INV-9's
"the only sockets opened are 127.0.0.1 inference and the local MCP
listener." This also happens to be the choice `tests/test_invariants.py`'s
`test_no_outbound_http_clients` static-import scan already permits, since
`FORBIDDEN_HTTP_MODULES` names `urllib.request` but not `http.client` — see
this task's report for why that is read as confirmation of an independently
correct choice, not the reason for it.

**Progress and blocking (INV-1).** `complete()` runs the request on a
background thread and polls it from the caller's thread with
`Thread.join(progress_interval)`, calling `on_status` once per interval the
request is still in flight. `on_status` defaults to doing nothing, matching
`palaver/replay.py` and every `palaver/cli/*.py` entry point: the channel is
a plain callback, never a stdout write, so a caller that wants console
output wires its own stderr writer in exactly the way each CLI module's
`_stderr_status` does. This module never imports `sys` or writes anything to
either stream itself.

**`model_runs` (task 3.2's schema amendment).** Every call to `complete()`
inserts a `model_runs` row before the request starts (`status='running'`)
and updates that same row once the request settles, whether it succeeds or
raises (`status='done'` or `'error'`), so "every request and its latency
lands in `model_runs`" holds for failed requests too. `latency_ms` is always
set on the settled row; `prompt_tokens` is only ever known from a
successful response's `usage.prompt_tokens` field, so it stays NULL on an
`'error'` row — migration 6 in `palaver/store/schema.py` makes both columns
nullable for exactly that reason.
"""

from __future__ import annotations

import http.client
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Route verified against the llama.cpp server parser source, not the
#: README (see module docstring). The native `/completion` route takes
#: different top-level fields and is not used here.
_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

_CREATED_AT_UPDATE = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class ModelClientError(Exception):
    """Base class for every typed error `ModelClient` raises.

    Callers catch this (or a specific subclass) to degrade gracefully —
    per this task's contract, `ModelClient` never returns a partial state
    object, so "the request failed" is always this exception, never a dict
    with missing or null fields standing in for failure.
    """


class ModelConnectionError(ModelClientError):
    """Raised when the client cannot open or complete a TCP connection.

    Covers connection refused (nothing listening on the configured port)
    and any other `OSError` the socket layer raises that is not a timeout.
    """


class ModelTimeoutError(ModelClientError):
    """Raised when connecting or reading a response exceeds `timeout` seconds."""


class ModelResponseError(ModelClientError):
    """Raised when llama-server responds but the response cannot be trusted.

    Covers a non-200 HTTP status, a response body that is not valid JSON, a
    `choices[0].message.content` that is not itself valid JSON (the
    unconstrained-text failure this task exists to prevent — schema-
    constrained decoding is supposed to make this impossible, so seeing it
    means the build or the request shape does not match what was verified),
    a parsed content value that is not a JSON object, or an object missing a
    field the requested schema names as `required`.
    """


class ModelClient:
    """HTTP client for a pre-existing `llama-server`, one schema-constrained request at a time.

    Palaver does not start, stop, or otherwise manage the server process
    (plan section 5); this class only ever opens outbound connections to the
    single `host`/`port` it is constructed with, which must stay
    `127.0.0.1`-only per INV-9 in any real deployment of this project.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        host: str = "127.0.0.1",
        port: int = 8090,
        timeout: float = 30.0,
        progress_interval: float = 5.0,
    ) -> None:
        """Configure a client against one llama-server instance.

        Args:
            conn: Open connection to Palaver's store. Used only from the
                calling thread, before the request starts and after it
                settles, to write the `model_runs` row (task 3.2) — never
                from the background thread that performs the request, so no
                sqlite connection ever crosses a thread boundary here.
            host: Must stay `127.0.0.1` in any real deployment; INV-9
                permits no other outbound socket destination for this
                client. Overridable for tests, which bind their stub server
                to `127.0.0.1` anyway.
            port: 8090 is the plan's documented port for the pre-existing
                inference server (distinct from 8091, which task 3.5's eval
                harness starts and stops itself).
            timeout: Seconds allowed for the underlying socket to connect
                and for each read; bounds both a hung connect and a hung
                response, per Python's `socket` semantics for a connection
                timeout applied to an `http.client.HTTPConnection`.
            progress_interval: Seconds between `on_status` calls while a
                request is still in flight (INV-1). Kept small in tests so
                the assertion that at least one call happens does not
                require waiting on wall-clock seconds.
        """
        self._conn = conn
        self._host = host
        self._port = port
        self.timeout = timeout
        self.progress_interval = progress_interval

    def complete(
        self,
        *,
        model: str,
        purpose: str,
        prompt: str,
        schema: dict,
        session_id: int | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> dict:
        """Send one schema-constrained request and return the parsed object.

        Args:
            model: Recorded in `model_runs.model`; llama-server ignores this
                field when it has exactly one model loaded, but the column
                is `NOT NULL` so a value is always required here.
            purpose: Free-text label recorded in `model_runs.purpose`
                (e.g. `"extraction"`), for later query by call site.
            prompt: The full prompt text, already assembled by the caller.
                This module does no prompt construction of its own.
            schema: A self-contained JSON Schema object (no external
                `$ref`) describing the object the model must return. Sent
                nested under `response_format.json_schema.schema`, per the
                verified request shape in the module docstring.
            session_id: Recorded in `model_runs.session_id`. `None` is
                valid — the column has no `NOT NULL` constraint — for calls
                not tied to one observed session.
            on_status: Progress channel (INV-1). Defaults to doing nothing;
                called with a short string at most once per
                `progress_interval` seconds while the request is in flight.
                Never writes to stdout or stderr itself.

        Returns:
            The parsed JSON object from the model's response.

        Raises:
            ModelConnectionError: The connection could not be established.
            ModelTimeoutError: Connecting or reading exceeded `timeout`.
            ModelResponseError: A response arrived but was not a trustworthy,
                schema-conforming JSON object.
        """
        run_id = self._start_run(session_id=session_id, model=model, purpose=purpose)
        started = time.monotonic()
        try:
            parsed, prompt_tokens = self._run_with_progress(
                lambda: self._request(model=model, prompt=prompt, schema=schema),
                on_status,
            )
        except ModelClientError:
            latency_ms = int((time.monotonic() - started) * 1000)
            self._finish_run(run_id, status="error", latency_ms=latency_ms, prompt_tokens=None)
            raise
        else:
            latency_ms = int((time.monotonic() - started) * 1000)
            self._finish_run(
                run_id, status="done", latency_ms=latency_ms, prompt_tokens=prompt_tokens
            )
            return parsed

    def _run_with_progress(
        self,
        request_fn: Callable[[], tuple[dict, int | None]],
        on_status: Callable[[str], None] | None,
    ) -> tuple[dict, int | None]:
        """Run `request_fn` on a background thread, polling it for `on_status`.

        `Thread.join(progress_interval)` blocks for up to `progress_interval`
        seconds or until the worker finishes, whichever is first. Each time
        it returns with the worker still alive, that is one full interval
        elapsed with no response yet, so `on_status` fires once per lap —
        never from the worker thread itself, so a slow or reentrant
        callback cannot delay the request.
        """
        outcome: dict[str, object] = {}

        def worker() -> None:
            try:
                outcome["value"] = request_fn()
            except Exception as exc:  # noqa: BLE001 - re-raised as-is on the caller's thread
                outcome["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        elapsed = 0.0
        while True:
            thread.join(self.progress_interval)
            if not thread.is_alive():
                break
            elapsed += self.progress_interval
            if on_status is not None:
                on_status(f"waiting on model response ({elapsed:.1f}s elapsed)")

        if "error" in outcome:
            raise outcome["error"]  # type: ignore[misc]
        return outcome["value"]  # type: ignore[return-value]

    def _request(self, *, model: str, prompt: str, schema: dict) -> tuple[dict, int | None]:
        """Send the request and return (parsed object, prompt token count or None).

        Runs on the background thread started by `_run_with_progress`; talks
        to the socket only, never to `self._conn`.
        """
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": schema},
                },
            }
        ).encode("utf-8")

        connection = http.client.HTTPConnection(self._host, self._port, timeout=self.timeout)
        try:
            connection.request(
                "POST",
                _CHAT_COMPLETIONS_PATH,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw_body = response.read()
            status = response.status
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"request to {self._host}:{self._port} timed out after {self.timeout}s"
            ) from exc
        except OSError as exc:
            raise ModelConnectionError(
                f"could not reach llama-server at {self._host}:{self._port}: {exc}"
            ) from exc
        finally:
            connection.close()

        if status != 200:
            raise ModelResponseError(
                f"llama-server returned HTTP {status} from {_CHAT_COMPLETIONS_PATH}: "
                f"{raw_body[:500]!r}"
            )

        return self._parse_response(raw_body, schema)

    @staticmethod
    def _parse_response(raw_body: bytes, schema: dict) -> tuple[dict, int | None]:
        """Parse the OpenAI-shaped envelope, then the schema-constrained content within it.

        Two JSON parses happen here, not one: the HTTP response body is a
        JSON envelope (`choices`, `usage`, ...), and `choices[0].message.
        content` is itself a JSON-encoded string carrying the actual
        schema-constrained object. A 200 response carrying unconstrained
        text — the failure this task exists to prevent, per the
        orchestrator's research note — fails the second parse, the
        `isinstance(parsed, dict)` check, or the `required`-field check
        below, so all three are treated as the same `ModelResponseError`
        rather than three different classes of trust failure.
        """
        try:
            envelope = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"response body is not valid JSON: {exc}") from exc

        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(f"response missing choices[0].message.content: {exc}") from exc

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                f"message content is not schema-conforming JSON: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ModelResponseError(
                f"message content parsed to {type(parsed).__name__}, not a JSON object"
            )

        required = schema.get("required")
        if isinstance(required, list):
            missing = [key for key in required if key not in parsed]
            if missing:
                raise ModelResponseError(
                    f"response object missing field(s) {missing} required by the requested schema"
                )

        prompt_tokens = None
        usage = envelope.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("prompt_tokens")
            if isinstance(tokens, int):
                prompt_tokens = tokens

        return parsed, prompt_tokens

    def _start_run(self, *, session_id: int | None, model: str, purpose: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO model_runs(session_id, model, purpose, status) "
            "VALUES (?, ?, ?, 'running')",
            (session_id, model, purpose),
        )
        self._conn.commit()
        run_id = cursor.lastrowid
        assert run_id is not None
        return run_id

    def _finish_run(
        self,
        run_id: int,
        *,
        status: str,
        latency_ms: int,
        prompt_tokens: int | None,
    ) -> None:
        self._conn.execute(
            f"UPDATE model_runs SET status = ?, finished_at = {_CREATED_AT_UPDATE}, "
            "latency_ms = ?, prompt_tokens = ? WHERE id = ?",
            (status, latency_ms, prompt_tokens, run_id),
        )
        self._conn.commit()
