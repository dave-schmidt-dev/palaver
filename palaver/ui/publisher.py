"""Write each pane's status into iTerm2, on a heartbeat.

This module is the producer the rest of Phase 5 assumed. Tasks 5.0 through
5.5 built a reader (`component.build_component`), a registrar
(`component.register`), a pane-to-session join (`pane_join.join_pane`), a
renderer (`render.render`), and a staleness horizon (`component.STALE_AFTER`)
-- and nothing that writes `user.palaver_status` outside a test or the
selftest probe. Each layer proved itself and none proved a producer existed,
so every check in the phase passed against a surface that would have rendered
`unknown` forever on a real machine.

**The push is a heartbeat, not an edge.** Task 5.5 made an unrefreshed status
expire: a payload older than `STALE_AFTER` decodes to `UNKNOWN` with its task
text dropped. That makes "push only when the state changed" actively wrong --
an agent working steadily for ten minutes changes nothing, gets no push, and
degrades to `unknown` ninety seconds in. The most common case would fail the
loudest. So every tick pushes every attached pane, unchanged or not, and
`PUSH_CADENCE` lives next to `STALE_AFTER` in `component` with a test pinning
the two together. The observer daemon's own `DEFAULT_INTERVAL` is deliberately
not that clock: it governs how fresh the *content* is, not how fresh the
*stamp* is, and the two are free to differ.

**Where it runs.** Inside the AutoLaunch process (task 5.1), which already
owns the iTerm2 connection and the attached-pane registry. Not inside the
observer daemon: that runs under its own launchd job, in an install where the
`ui` extra need not be present, with no iTerm2 connection to write through.
The two processes meet at the database and at the session stores, both of
which the publisher opens read-only -- it is not the single writer (INV-2 for
the stores, task 6.3's socket for the database), and it creates no `projects`
or `sessions` row even when a pane names one the daemon has not seen.

**Refusals are pushed, not skipped.** A pane whose join yields no single
session gets `UNKNOWN`, because leaving it alone leaves whatever was there --
which is the stale-value failure INV-7 exists to prevent, arriving by way of
omission rather than by way of a wrong write. The same goes for a pane that
is not running an agent at all: it is attached, so it is answered.

**One pane's failure is one pane's failure.** A pane can close between the
registry read and the write, and iTerm2 raises on the write to a session that
no longer exists. An unguarded loop would abandon every pane after it in the
same tick, so each pane is pushed under its own guard and a failure is logged
and stepped over.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.extract.persist import CURRENT_TASK
from palaver.ingest.adapters.codex import CodexAdapter
from palaver.ingest.cursors import Cursor
from palaver.observer.signals import (
    SIGNAL_NAMES,
    Status,
    Tri,
    apply_liveness,
    derive_status,
    derive_status_for_source,
)
from palaver.observer.turn_boundary import derive_signals_from_events, observe_session
from palaver.ui import component
from palaver.ui.pane_join import (
    PIN_VARIABLE,
    PaneVariables,
    ProcessTable,
    join_pane,
    observe_liveness,
    process_is_alive,
    read_process_table,
    working_directory,
)

log = logging.getLogger(__name__)
SESSION_PIN_VARIABLE = PIN_VARIABLE

#: Compatibility name retained for callers that used the original Claude-only
#: publisher. Source-aware joins now publish both Claude Code and Codex; an
#: unsupported source still degrades to `UNKNOWN` in the join layer.
PUBLISHABLE_SOURCE = "claude-code"

#: How long a `current_task` row stays worth attaching to a fresh status.
#: The status and the task text come from different clocks -- the status is
#: derived from the store on this tick, the task text was written by the
#: daemon's last extraction -- and pairing a live status with an hour-old
#: task would put a stale claim inside the freshest possible wrapper. Wider
#: than `STALE_AFTER` because an extraction pass is expensive and gated on
#: cursor advance, so a task text that outlives one tick is normal.
TASK_HORIZON = timedelta(minutes=10)

#: `updated_at`'s storage format, from `palaver.store.schema`. Zero-padded
#: and UTC, so a cutoff can be compared as a string in SQL rather than
#: parsed row by row.
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass(frozen=True)
class PanePush:
    """What one pane got on one tick.

    Attributes:
        pane_id: The pane.
        status: The status that was published, or would have been.
        task: The task text published with it, or `None`.
        payload: The encoded payload iTerm2 accepted, or `None` if the write
            failed. `None` is how a caller tells a refusal (`status` is
            `UNKNOWN`, `payload` is set) from a failure (`payload` is
            `None`), which read the same from outside.
    """

    pane_id: str
    status: Status
    task: str | None
    payload: str | None

    @property
    def published(self) -> bool:
        """Whether iTerm2 accepted the write."""
        return self.payload is not None


def _no_status(_message: str) -> None:
    """Default progress sink."""


def read_current_task(
    db_path: Path,
    *,
    source: str,
    session_key: str,
    now: datetime,
    horizon: timedelta = TASK_HORIZON,
) -> str | None:
    """Read one session's `current_task`, read-only, or `None`.

    Opens the database in SQLite's `mode=ro` rather than trusting this
    function to contain only `SELECT`s: the guarantee then holds against a
    later edit here, and against the library, rather than against a reading
    of the source. An absent database and an unmigrated one both return
    `None` -- a machine running the terminal surface without the observer
    daemon is a supported state, not an error, and making the pane depend on
    the model layer being up would surface a `🐞` on every pane the moment
    the daemon stopped.

    Args:
        db_path: The observer's database.
        source: The adapter source name, as `sessions.source` stores it.
        session_key: The session's external id.
        now: Reference time the horizon is measured back from.
        horizon: How old a row may be and still be attached to this tick's
            status.

    Returns:
        The task text, or `None` when there is no row, the row is older than
        `horizon`, or the database cannot be read.
    """
    if not db_path.exists():
        return None
    cutoff = (now - horizon).strftime(_TIMESTAMP_FORMAT)
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        log.warning("could not open %s read-only for task text", db_path, exc_info=True)
        return None
    try:
        row = conn.execute(
            "SELECT cs.value FROM current_state cs "
            "JOIN sessions s ON s.id = cs.session_id "
            "WHERE s.source = ? AND s.external_id = ? AND cs.key = ? AND cs.updated_at >= ?",
            (source, session_key, CURRENT_TASK, cutoff),
        ).fetchone()
    except sqlite3.Error:
        log.warning("could not read current_task from %s", db_path, exc_info=True)
        return None
    finally:
        conn.close()
    if row is None:
        return None
    value = row[0]
    return value if isinstance(value, str) and value else None


def status_for_pane(
    variables: PaneVariables,
    *,
    sessions_root: Path | None = None,
    store_roots: Mapping[str, Path] | None = None,
    db_path: Path | None = None,
    now: datetime,
    table: ProcessTable | None = None,
    cwd_reader: Callable[[int], Path | None] = working_directory,
    alive_probe: Callable[[int], bool] = process_is_alive,
    candidate_cache: MutableMapping[tuple[str, str, str], tuple[Path, ...]] | None = None,
) -> tuple[Status, str | None]:
    """Derive what one pane should be shown as.

    Every refusal yields `(Status.UNKNOWN, None)`. A refusal is the ordinary
    outcome for most panes on a machine -- a shell, an editor, an agent over
    ssh, a project with two panes open -- and none of them is a failure.

    The task text is dropped whenever the status is `UNKNOWN`, matching
    `component.decode_status`'s rule for a stale payload: `unknown: reading a
    file` is still a confident claim about the pane.

    Args:
        variables: The pane's iTerm2 variables.
        sessions_root: The session store root; `join_pane`'s default when
            `None`.
        store_roots: Independent roots keyed by source.
        db_path: The observer database to read task text from. `None` reads
            no task text at all, which is what a caller with no database
            configured wants.
        now: Reference time, aware UTC.
        table: Process table snapshot; read fresh when `None`.
        cwd_reader: Working-directory reader, injected for tests.
        alive_probe: Liveness probe for the agent's pid, injected for the
            same reason -- a test that used the real one would assert
            different things depending on which pids happened to exist.
        candidate_cache: Per-tick Codex candidate cache.

    Returns:
        The status and task text to push.
    """
    join = join_pane(
        variables,
        table=table,
        cwd_reader=cwd_reader,
        sessions_root=sessions_root,
        store_roots=store_roots,
        now=now,
        pin=variables.pin,
        candidate_cache=candidate_cache,
    )
    if join is None or join.session_key is None:
        return Status.UNKNOWN, None
    store = join.store_path
    if store is None:
        return Status.UNKNOWN, None
    try:
        if join.source == "codex":
            # `tail` reads the validated path directly; it does not infer a
            # project from the date-partitioned parent directories.
            tail = CodexAdapter().tail(store, Cursor(0))
            observation = derive_signals_from_events(tail.events, parsed=Tri.TRUE)
            derived = derive_status_for_source(
                observation.signals,
                {name: 100.0 for name in SIGNAL_NAMES},
            )
        else:
            observation = observe_session(store, now=now)
            derived = derive_status(observation.signals)
        last_advance = datetime.fromtimestamp(store.stat().st_mtime, tz=timezone.utc)
    except OSError:
        log.warning("could not read %s for pane %s", store, join.pane_id, exc_info=True)
        return Status.UNKNOWN, None

    liveness = observe_liveness(
        join.pid, last_advance=last_advance, now=now, alive_probe=alive_probe
    )
    status = apply_liveness(derived, liveness)
    if status is Status.UNKNOWN:
        return Status.UNKNOWN, None

    task = None
    if db_path is not None:
        task = read_current_task(db_path, source=join.source, session_key=join.session_key, now=now)
    return status, task


async def publish_once(
    pane_ids: Iterable[str],
    *,
    read_variables: Callable[[str], object],
    set_variable: component.SetVariable,
    sessions_root: Path | None = None,
    store_roots: Mapping[str, Path] | None = None,
    db_path: Path | None = None,
    now: datetime | None = None,
    table: ProcessTable | None = None,
    cwd_reader: Callable[[int], Path | None] = working_directory,
    alive_probe: Callable[[int], bool] = process_is_alive,
    on_status: Callable[[str], None] = _no_status,
) -> tuple[PanePush, ...]:
    """Push one status into every pane named, and report what each got.

    The process table is read once for the whole tick rather than once per
    pane: it is a `ps` subprocess, the panes are being resolved against the
    same instant, and reading it per pane would let two panes disagree about
    which processes exist.

    Args:
        pane_ids: The panes to push to, read from the registry by the
            caller on this tick.
        read_variables: Async callable taking a pane id and returning its
            `PaneVariables`, or `None` if the pane could not be read.
        set_variable: The variable writer.
        sessions_root: The session store root.
        db_path: The observer database, or `None` for no task text.
        now: Reference time, aware UTC. Defaults to the current time.
        table: Process table snapshot; read fresh when `None`.
        cwd_reader: Working-directory reader, injected for tests.
        alive_probe: Liveness probe, injected for tests.
        on_status: Progress channel (INV-1).

    Returns:
        One `PanePush` per pane id, in order. A pane whose write failed is
        present with `payload=None` rather than absent, so a caller can tell
        "not pushed" from "not attached".
    """
    when = datetime.now(timezone.utc) if now is None else now
    stamp = when.timestamp()
    panes = tuple(pane_ids)
    if table is None and panes:
        table = read_process_table()

    # Candidate metadata is expensive for Codex because identity lives in
    # the file. Share it across panes in this one process-table tick.
    candidate_cache: dict[tuple[str, str, str], tuple[Path, ...]] = {}

    pushes = []
    for pane_id in panes:
        status, task, payload = Status.UNKNOWN, None, None
        try:
            variables = await read_variables(pane_id)
            if isinstance(variables, PaneVariables):
                status, task = status_for_pane(
                    variables,
                    sessions_root=sessions_root,
                    store_roots=store_roots,
                    db_path=db_path,
                    now=when,
                    table=table,
                    cwd_reader=cwd_reader,
                    alive_probe=alive_probe,
                    candidate_cache=candidate_cache,
                )
            payload = await component.push_status(set_variable, pane_id, status, task, now=stamp)
        except Exception:
            log.warning("could not publish status to pane %s", pane_id, exc_info=True)
        pushes.append(PanePush(pane_id=pane_id, status=status, task=task, payload=payload))

    published = sum(1 for push in pushes if push.published)
    on_status(f"published {published}/{len(pushes)} pane(s)")
    return tuple(pushes)


#: The pane variables `join_pane` reads, in the order `make_variables_reader`
#: asks for them. `session.id` is not among them: the reader already knows
#: which pane it asked about, and iTerm2 answers a variable request for a
#: closed pane with an error rather than with a corrected id.
PANE_VARIABLE_NAMES = ("jobPid", "jobName", "path", "user.palaver_session_pin")


def make_variables_reader(connection: object) -> Callable[[str], object]:
    """Return an async reader of one pane's `PaneVariables`.

    Goes through `iterm2.rpc.async_variable` for the same reason
    `component.make_variable_writer` does: the publisher holds session *ids*,
    and the library's `Session.__init__` says not to construct one directly.
    All three variables are fetched in a single request, so they describe one
    instant -- fetched separately, `jobPid` and `path` could straddle a `cd`
    and produce a join that corroborates nothing.

    Args:
        connection: A live `iterm2.Connection`.

    Returns:
        An async `(pane_id) -> PaneVariables | None` reader. `None` means
        iTerm2 refused the read, which is the ordinary answer for a pane that
        closed between the registry read and this call.
    """
    from palaver.ui.connection import import_iterm2  # noqa: PLC0415 - keeps the extra optional

    import_iterm2()
    import iterm2.api_pb2  # noqa: PLC0415 - optional extra, checked immediately above
    import iterm2.rpc  # noqa: PLC0415

    ok = iterm2.api_pb2.VariableResponse.Status.Value("OK")

    async def read_variables(pane_id: str) -> PaneVariables | None:
        result = await iterm2.rpc.async_variable(connection, pane_id, [], list(PANE_VARIABLE_NAMES))
        response = result.variable_response
        # Older iTerm2/API test doubles may omit the optional pin variable;
        # the three process variables remain sufficient for automatic join.
        if response.status != ok or len(response.values) < 3:
            return None
        decoded = [_decode_variable(value) for value in response.values]
        job_pid, job_name, path = decoded[:3]
        pin = decoded[3] if len(decoded) > 3 else None
        return PaneVariables(
            pane_id=pane_id,
            job_pid=int(job_pid)
            if isinstance(job_pid, (int, float, str)) and _is_pid(job_pid)
            else None,
            job_name=job_name if isinstance(job_name, str) else None,
            path=path if isinstance(path, str) else None,
            pin=pin if isinstance(pin, str) else None,
        )

    return read_variables


def _decode_variable(raw: str) -> object:
    """Decode one JSON-encoded variable value, or `None` if it is not JSON.

    iTerm2 returns `"null"` for a variable a pane does not have, which
    decodes to `None` and is exactly the absent case `join_pane` refuses on.
    """
    try:
        return json.loads(raw)
    except TypeError, ValueError:
        return None


def _is_pid(value: object) -> bool:
    """Whether a decoded `jobPid` can be read as a positive integer."""
    try:
        return int(value) > 0  # type: ignore[arg-type]
    except TypeError, ValueError:
        return False


async def publish_forever(
    registry,
    *,
    read_variables: Callable[[str], object],
    set_variable: component.SetVariable,
    sessions_root: Path | None = None,
    store_roots: Mapping[str, Path] | None = None,
    db_path: Path | None = None,
    cadence: float = component.PUSH_CADENCE,
    limit: int | None = None,
    on_status: Callable[[str], None] = _no_status,
    sleep=asyncio.sleep,
) -> int:
    """Heartbeat every attached pane's status until cancelled.

    The registry is read fresh on every tick rather than snapshotted at the
    start: `watch_new_sessions` and `watch_terminations` are mutating it
    concurrently in the same event loop, and a captured list would push to
    panes that had closed and miss panes that had opened.

    Nothing here catches a connection-level failure. `publish_once` guards
    each pane against its own error, but a dropped iTerm2 connection fails
    every pane at once and should propagate out of the `asyncio.gather` in
    `autolaunch.main`, exactly as the two monitors' failures do -- the shim
    restarts the process, which is the only thing that can re-establish the
    connection.

    Args:
        registry: The `SessionRegistry`; its `attached` property is read once
            per tick.
        read_variables: Async callable taking a pane id and returning its
            `PaneVariables`, or `None`.
        set_variable: The variable writer.
        sessions_root: The session store root.
        db_path: The observer database, or `None` for no task text.
        cadence: Seconds between ticks.
        limit: Stop after this many ticks. `None` means never, which is the
            daemon's case; a number is what makes this testable.
        on_status: Progress channel (INV-1).
        sleep: The sleep callable, injected so a test does not wait.

    Returns:
        How many panes were successfully pushed to, summed over every tick.
    """
    published = 0
    ticks = 0
    while limit is None or ticks < limit:
        ticks += 1
        pushes = await publish_once(
            sorted(registry.attached),
            read_variables=read_variables,
            set_variable=set_variable,
            sessions_root=sessions_root,
            store_roots=store_roots,
            db_path=db_path,
            on_status=on_status,
        )
        published += sum(1 for push in pushes if push.published)
        if limit is None or ticks < limit:
            await sleep(cadence)
    return published
