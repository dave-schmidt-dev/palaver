"""The long-running observer: one writer, one tick loop (Task 4.1).

`ObserverDaemon` is the process that watches every configured source and
turns "this session changed" into "extract this session". Two properties
make it the daemon rather than a loop somebody could have written inline:

**It is the single SQLite writer.** The connection is opened once, in
`start()`, and lives for the daemon's whole run — `tick()` never opens one,
and `start()` called twice does not open a second. Task 6.3 makes the socket
the enforcement boundary for *other* processes; this class is what makes the
guarantee true within this one. `migrate()` runs once at `start()` too,
before that connection exists, so the schema upgrade's own short-lived
connections are never concurrent with the daemon's.

**It costs nothing for an idle session.** Scheduling is delegated whole to
`palaver.observer.scheduler.plan_tick`, which schedules only sessions whose
cursor moved. Ten ticks over a store nobody typed into issue zero inference
requests, which is the property `tests/test_scheduler.py` pins.

**Cursors are saved after extraction, never before.** A cursor persisted at
tail time would mean an extraction that failed — model server down, request
timed out — silently consumed the only record that those bytes had not been
looked at yet, and the session would never be re-examined. Saving after
success makes the pipeline at-least-once: a failed extraction leaves the
cursor where it was and the next tick re-schedules the same session. The
cost is that a *persistently* failing extractor re-requests every tick
rather than backing off; slot management and backoff are task 4.2's, and
this module deliberately does not pre-empt them.

**One session's failure is not the daemon's failure.** An extractor raising
for one session is recorded in that tick's `TickResult.failed` and the tick
continues to the next session. A daemon that exits because one model
request timed out would stop observing five healthy sessions to report one
sick one.

INV-1: every tick emits progress on `on_status` — at tick start, once per
session tailed, and again per session extracted — so a 17-second inference
request is never a silent wait. The channel defaults to stderr in the CLI
and writes nothing to stdout.

INV-2: nothing here opens an observed session store. Reads go through the
adapter's `tail` and through `normalize_path`, both of which reach the file
via `open_source_readonly`.

INV-7: `ModelExtractor` builds its request schema from
`REFINEMENT_PAYLOAD_KEYS` with `additionalProperties: false`, so a
schema-constrained server structurally cannot return a status field, and the
response still goes through `extraction_from_model_payload`, which rejects
one anyway. The instruction does not mention status at all — not even to
forbid it. Naming the concept in a prompt is how it ends up in the answer,
and the two enforcement layers above make the disclaimer redundant. Status
stays derived from deterministic signals.

This repository is public. Nothing in this module's docstrings, prompt text,
or examples is derived from a real observed session (INV-9).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from palaver.extract.client import ModelClient
from palaver.extract.normalize import normalize_path
from palaver.extract.persist import persist_extraction
from palaver.ingest.adapters.base import DEFAULT_SINCE, Adapter, SessionRef
from palaver.ingest.cursors import CursorStore
from palaver.observer.scheduler import SessionWork, TickPlan, plan_tick
from palaver.observer.signals import REFINEMENT_PAYLOAD_KEYS, extraction_from_model_payload
from palaver.project_identity import (
    ProjectIdentity,
    canonical_project_path,
    project_identity_for_cwd,
)
from palaver.store.migrate import connect, migrate

#: Recorded in `model_runs.model`. llama-server ignores the field when it
#: has one model loaded (see `ModelClient.complete`), but the column is
#: NOT NULL, so the daemon names what it believes it is talking to.
DEFAULT_MODEL = "local-llama-server"

#: Recorded in `model_runs.purpose`, so a daemon request is separable from
#: task 3.5's eval requests in the same table.
EXTRACTION_PURPOSE = "observer-extraction"

#: Default seconds between ticks. The plan's measured slot save/restore
#: (33 ms / 21 ms) is affordable at a 30-60 s tick; 30 s is the responsive
#: end of that range.
DEFAULT_INTERVAL = 30.0


def extraction_schema() -> dict:
    """Build the JSON Schema for one extraction request.

    Derived from `REFINEMENT_PAYLOAD_KEYS` rather than written out, so a
    field added to the status path's payload contract appears in the request
    schema without editing this function — and a field removed there stops
    being asked for, instead of being asked for and silently dropped by
    `extraction_from_model_payload`.

    Every property is nullable on purpose: `null` is how the model says
    "this pass has no opinion on that field", which `Extraction` preserves
    and `derive_status` depends on. Requiring a string would force the model
    to invent one, and an invented `remaining_work` is indistinguishable
    from a real one by the time it reaches the status rules.

    Returns:
        A self-contained JSON Schema object (no external `$ref`), shaped for
        `ModelClient.complete`'s `schema` argument.
    """
    return {
        "type": "object",
        "properties": {key: {"type": ["string", "null"]} for key in REFINEMENT_PAYLOAD_KEYS},
        "required": list(REFINEMENT_PAYLOAD_KEYS),
        "additionalProperties": False,
    }


#: The extraction instruction. Names the four ephemeral fields and nothing
#: else — see the module docstring's INV-7 note for why it does not mention
#: status even to forbid it. `tests/test_scheduler.py` asserts that against
#: `FORBIDDEN_PAYLOAD_KEYS`, so the prompt and the response-side rejection
#: cannot drift apart.
EXTRACTION_INSTRUCTION = """\
You are reading a transcript of one coding-agent session. Report only what \
the transcript shows, in the session's own words where possible.

Return a JSON object with exactly these fields, each a string or null:

- current_task: what the session is working on right now.
- remaining_work: what is left to do.
- blockers_now: what is preventing progress right now.
- open_questions: questions the session has asked and not yet had answered.

Use null for a field the transcript does not show. Use an empty string only \
when the transcript shows there is affirmatively nothing — for example, no \
remaining work because the task finished. Do not guess: a field you cannot \
support from the transcript is null.

Transcript:
"""


class DaemonNotStartedError(RuntimeError):
    """Raised when a tick is attempted before `start()` opened the writer."""


class Extractor(Protocol):
    """What the daemon calls for one scheduled session.

    Implementations receive the daemon's single write connection and must
    not open their own — that is the guarantee this class exists to keep.
    Raising is a per-session failure, recorded in `TickResult.failed`; the
    daemon catches it and moves on.
    """

    def __call__(
        self,
        work: SessionWork,
        *,
        conn: sqlite3.Connection,
        on_status: Callable[[str], None],
    ) -> None:
        """Extract one session, writing through `conn`."""


@dataclass(frozen=True)
class TickResult:
    """What one tick discovered, skipped, extracted, and failed on.

    Attributes:
        tick: 1-based tick number within this daemon's run.
        plan: The scheduling decision this tick made — see `TickPlan`.
        extracted: `session_key` for each session the extractor completed,
            in scheduled order. These are the sessions whose cursors were
            advanced and persisted.
        failed: `(session_key, error text)` for each session whose extractor
            raised. Their cursors are deliberately left where they were, so
            the next tick re-schedules them.
    """

    tick: int
    plan: TickPlan
    extracted: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def _get_or_create_project(conn: sqlite3.Connection, name: str, path: str) -> int:
    """Return the project at ``path``, never merging distinct paths by name."""
    path = str(canonical_project_path(path))
    row = conn.execute("SELECT id FROM projects WHERE path = ?", (path,)).fetchone()
    if row is not None:
        return row[0]
    existing = conn.execute("SELECT path FROM projects WHERE name = ?", (name,)).fetchone()
    if existing is not None and existing[0] != path:
        # The identity normally already carries a path digest. This fallback
        # also protects callers supplying a legacy/readable name directly.
        name = project_identity_for_cwd(path).name
        suffix = 2
        while conn.execute("SELECT 1 FROM projects WHERE name = ?", (name,)).fetchone():
            name = f"{project_identity_for_cwd(path).name}-{suffix}"
            suffix += 1
    cursor = conn.execute("INSERT INTO projects(name, path) VALUES (?, ?)", (name, path))
    return cursor.lastrowid


def _get_or_create_session(
    conn: sqlite3.Connection, project_id: int, source: str, external_id: str
) -> int:
    """Return `sessions.id` for `(source, external_id)`, inserting if absent."""
    row = conn.execute(
        "SELECT id FROM sessions WHERE source = ? AND external_id = ?",
        (source, external_id),
    ).fetchone()
    if row is not None:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO sessions(project_id, source, external_id) VALUES (?, ?, ?)",
        (project_id, source, external_id),
    )
    return cursor.lastrowid


def ensure_scope(conn: sqlite3.Connection, ref: SessionRef) -> tuple[int, int]:
    """Resolve one discovered session to `(project_id, session_id)`, creating rows.

    The project is identified by everything in the `session_key` before the
    final `/` — the adapter's own encoding of the project a session belongs
    to — and recorded with the store directory as its path, which is unique
    per project because that is the directory the adapter enumerates. The
    session's `external_id` is the whole `session_key`, which is unique
    within a source by construction, matching the schema's
    `UNIQUE (source, external_id)`.

    Args:
        conn: The daemon's write connection.
        ref: A discovered session.

    Returns:
        `(project_id, session_id)`, both existing rows after this call.
    """
    identity = ref.project or ProjectIdentity(
        name=ref.path.parent.name,
        path=canonical_project_path(ref.path.parent),
    )
    project_id = _get_or_create_project(conn, identity.name, str(identity.path))
    session_id = _get_or_create_session(conn, project_id, ref.source, ref.session_key)
    return project_id, session_id


class ModelExtractor:
    """The daemon's production extractor: one schema-constrained request per session.

    Reads the session's transcript through `normalize_path` (INV-2), asks
    the local llama-server for the four ephemeral fields, and routes the
    answer through `persist_extraction` (task 3.4), which upserts them into
    `current_state` and never into `memories`.

    Durable claims — decisions and resolved questions — are deliberately not
    produced here. They must be quote-grounded against a stored
    `transcript_chunks` row (INV-6), which is the quote gate's write
    boundary, not the scheduler's; `extraction_from_model_payload`'s own
    docstring draws the same line. This extractor therefore writes exactly
    the ephemeral half of a pass, and writes it through the same function a
    complete pass would use.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 8090,
        timeout: float = 60.0,
    ) -> None:
        """Configure the extractor against one local inference server.

        Args:
            model: Recorded in `model_runs.model`.
            host: Must stay `127.0.0.1` in any real deployment (INV-9).
            port: The plan's documented port for the pre-existing server.
            timeout: Seconds allowed per request. Higher than
                `ModelClient`'s own default because the plan's measured
                single-request latency is 17.1 s at 16.7K prompt tokens, and
                a timeout below that would fail every real extraction.
        """
        self.model = model
        self.host = host
        self.port = port
        self.timeout = timeout

    def __call__(
        self,
        work: SessionWork,
        *,
        conn: sqlite3.Connection,
        on_status: Callable[[str], None],
    ) -> None:
        """Extract one session and persist its ephemeral state."""
        if work.ref.source not in {"claude-code", "codex"}:
            raise ValueError(f"unsupported extraction source: {work.ref.source}")
        project_id, session_id = ensure_scope(conn, work.ref)
        transcript = normalize_path(work.ref.path, source=work.ref.source)
        client = ModelClient(conn, host=self.host, port=self.port, timeout=self.timeout)
        payload = client.complete(
            model=self.model,
            purpose=EXTRACTION_PURPOSE,
            prompt=EXTRACTION_INSTRUCTION + transcript,
            schema=extraction_schema(),
            session_id=session_id,
            on_status=on_status,
        )
        # `extraction_from_model_payload` returns an `Extraction` whose
        # `decisions` and `resolved_questions` are always empty — it reads
        # only `REFINEMENT_PAYLOAD_KEYS` — so handing it straight to
        # `persist_extraction` writes four `current_state` upserts and,
        # structurally rather than by convention, zero `memories` rows.
        extraction = extraction_from_model_payload(payload)
        persist_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            extraction=extraction,
            source=work.ref.source,
        )
        conn.commit()


def _null_status(message: str) -> None:
    """Default progress channel: drop the message rather than pick a stream."""


class ObserverDaemon:
    """The observer process: single writer, tick loop, per-session isolation.

    Construct, `start()`, then either `tick()` for one pass or `run()` for
    the loop. `close()` releases the writer; the class is also a context
    manager, which is the shape the CLI uses so an interrupted daemon still
    closes its connection.
    """

    def __init__(
        self,
        *,
        db_path: str | Path,
        adapters: Sequence[Adapter],
        cursors: CursorStore,
        extractor: Extractor | None = None,
        on_status: Callable[[str], None] | None = None,
        since: timedelta | None = DEFAULT_SINCE,
        all: bool = False,
        connect_fn: Callable[[str | Path], sqlite3.Connection] = connect,
        migrate_fn: Callable[[str | Path], int] = migrate,
    ) -> None:
        """Configure the daemon. No file is opened until `start()`.

        Args:
            db_path: Palaver's store. Created and migrated by `start()`.
            adapters: Sources to sweep each tick, in order.
            cursors: Durable cursor store. The daemon writes a session's
                cursor only after that session's extraction succeeds.
            extractor: Called once per scheduled session. Defaults to
                `ModelExtractor`, which issues one inference request.
            on_status: Progress channel (INV-1). Defaults to discarding,
                so a library caller that wants silence gets it without the
                daemon picking a stream; the CLI supplies a stderr writer.
            since: Recency floor for discovery, passed to `plan_tick`.
            all: Skip discovery windowing entirely.
            connect_fn: Opens the single write connection. Injected so a
                test can observe exactly how many connections exist.
            migrate_fn: Brings the store to the current schema version. Runs
                once, in `start()`, before the write connection is opened.
        """
        self.db_path = Path(db_path)
        self.adapters = tuple(adapters)
        self.cursors = cursors
        self.extractor: Extractor = ModelExtractor() if extractor is None else extractor
        self.on_status = _null_status if on_status is None else on_status
        self.since = since
        self.all = all
        self._connect_fn = connect_fn
        self._migrate_fn = migrate_fn
        self._conn: sqlite3.Connection | None = None
        self._ticks = 0

    @property
    def conn(self) -> sqlite3.Connection:
        """The single write connection.

        Raises:
            DaemonNotStartedError: `start()` has not run, or `close()` has.
        """
        if self._conn is None:
            raise DaemonNotStartedError("observer daemon has no write connection; call start()")
        return self._conn

    def start(self) -> ObserverDaemon:
        """Migrate the store and open the one write connection. Idempotent.

        Calling this twice does not open a second connection — that is the
        single-writer guarantee stated as code rather than as a comment.

        Returns:
            `self`, so `ObserverDaemon(...).start()` reads as one expression.
        """
        if self._conn is not None:
            return self
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_status(f"migrating store at {self.db_path}")
        self._migrate_fn(self.db_path)
        self._conn = self._connect_fn(self.db_path)
        return self

    def close(self) -> None:
        """Close the write connection, if one is open. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ObserverDaemon:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def tick(self, *, now: datetime | None = None) -> TickResult:
        """Run one pass: discover, tail, extract what changed, save cursors.

        Args:
            now: Reference time for the discovery window. Defaults to the
                current UTC time; tests pass a fixed value.

        Returns:
            A `TickResult` for this pass.

        Raises:
            DaemonNotStartedError: `start()` has not run.
        """
        conn = self.conn
        self._ticks += 1
        tick = self._ticks
        self.on_status(f"tick {tick}: discovering sessions")

        plan = plan_tick(
            self.adapters,
            self.cursors,
            since=self.since,
            all=self.all,
            now=now,
            on_status=self.on_status,
        )
        self.on_status(f"tick {tick}: {len(plan.scheduled)} changed, {len(plan.skipped)} unchanged")

        extracted: list[str] = []
        failed: list[tuple[str, str]] = []
        for work in plan.scheduled:
            key = work.ref.session_key
            self.on_status(f"tick {tick}: extracting {key}")
            try:
                self.extractor(work, conn=conn, on_status=self.on_status)
            except Exception as exc:  # noqa: BLE001 - one session must not stop the loop
                failed.append((key, f"{type(exc).__name__}: {exc}"))
                self.on_status(f"tick {tick}: extraction failed for {key}: {exc}")
                continue
            self.cursors.save(key, work.cursor_after, source=work.ref.source)
            extracted.append(key)

        return TickResult(
            tick=tick,
            plan=plan,
            extracted=tuple(extracted),
            failed=tuple(failed),
        )

    def run(
        self,
        *,
        interval: float = DEFAULT_INTERVAL,
        max_ticks: int | None = None,
        stop: Callable[[], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_tick: Callable[[TickResult], None] | None = None,
        now: datetime | None = None,
    ) -> list[TickResult]:
        """Tick until `max_ticks` is reached or `stop()` returns True.

        Sleeps *between* ticks, never after the last one, so a bounded run
        returns as soon as its work is done rather than idling out a final
        interval.

        Args:
            interval: Seconds between ticks.
            max_ticks: Stop after this many ticks. `None` runs until
                `stop()` says otherwise — the real daemon's mode.
            stop: Checked before each tick and before each sleep. `None`
                means never stop on request.
            sleep: Injected so a test can run a loop without wall time.
            on_tick: Called with each `TickResult` as soon as that tick
                finishes. The unbounded mode never returns, so a caller that
                only read the return value would print nothing for the
                daemon's entire life — this is how the CLI streams a result
                line per tick.
            now: Reference time forwarded to each tick.

        Returns:
            One `TickResult` per completed tick, in order. Unreachable in
            the unbounded mode, by design.
        """
        results: list[TickResult] = []
        while max_ticks is None or len(results) < max_ticks:
            if stop is not None and stop():
                break
            result = self.tick(now=now)
            results.append(result)
            if on_tick is not None:
                on_tick(result)
            if max_ticks is not None and len(results) >= max_ticks:
                break
            if stop is not None and stop():
                break
            sleep(interval)
        return results


__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_MODEL",
    "DaemonNotStartedError",
    "EXTRACTION_INSTRUCTION",
    "EXTRACTION_PURPOSE",
    "Extractor",
    "ModelExtractor",
    "ObserverDaemon",
    "TickResult",
    "ensure_scope",
    "extraction_schema",
]
