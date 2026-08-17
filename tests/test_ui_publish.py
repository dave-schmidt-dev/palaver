"""Task 5.6: the producer, and the heartbeat the horizon is measured against.

Phase 5 shipped a reader, a registrar, a join, a renderer, and a staleness
horizon, and no production writer. Every layer's tests passed; none of them
could fail on the absence of a caller. So the first thing this module asserts
is that a caller exists, and the rest of it attacks the two ways a caller can
be built wrong.

**The heartbeat.** A publisher that pushed only on change would be
indistinguishable from this one under any single-tick test: same panes, same
payloads, same statuses. It only diverges over time, and only against task
5.5's expiry -- so the heartbeat is asserted by pushing an unchanged state
twice and by then feeding the *first* tick's payload to `decode_status` at a
later clock, which is the pane the user would actually be looking at. The
constant pair behind it (`STALE_AFTER` against `PUSH_CADENCE`) is pinned
directly, because two constants drifting apart is invisible: every pane
quietly reads `unknown`, and every test that injects its own `now` still
passes.

**The isolation.** One pane closing mid-tick must cost that pane and nothing
else. Asserted with a writer that raises for exactly one pane and a positive
control that raises for none, because "every pane was pushed" and "the loop
never reached the failure" produce the same count when the failing pane is
last.

Nothing here reads a real transcript. The stores are records written for this
module (INV-9), and the process trees are described rather than spawned.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from palaver.observer.signals import Status
from palaver.store.migrate import migrate
from palaver.ui import autolaunch, component, publisher
from palaver.ui.pane_join import (
    CODEX_SOURCE,
    PaneJoin,
    PaneVariables,
    ProcessInfo,
    process_name,
    project_key_for_cwd,
)
from palaver.ui.publisher import (
    PUBLISHABLE_SOURCE,
    TASK_HORIZON,
    PanePush,
    publish_forever,
    publish_once,
    read_current_task,
    status_for_pane,
)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)

#: The measured Claude Code pane shape from task 5.2: the foreground job is an
#: MCP server the agent spawned, and `claude` itself is three hops up.
CLAUDE_TREE = (
    (63488, 63369, "node /Users/dave/.npm/_npx/abc/node_modules/.bin/playwright-mcp"),
    (63369, 63354, "npm exec @playwright/mcp@latest"),
    (63354, 62921, "claude"),
    (62921, 62920, "-zsh"),
    (62920, 83829, "/usr/bin/login -fpl dave /Applications/iTerm.app/Contents/MacOS/ShellLauncher"),
    (83829, 1, "/Users/dave/Library/Application Support/iTerm2/iTermServer-3.6.11"),
)
JOB_PID = 63488
AGENT_PID = 63354
PANE = "pane-1"
SESSION_KEY = "session-1"


def _table(rows=CLAUDE_TREE):
    return {
        pid: ProcessInfo(pid=pid, ppid=ppid, name=process_name(command), command=command)
        for pid, ppid, command in rows
    }


def _line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _tool_use(name: str = "Bash") -> dict:
    """An unresolved tool call -- the record shape that derives `WORKING`."""
    return {
        "type": "assistant",
        "sessionId": SESSION_KEY,
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu-1", "name": name, "input": {}}],
        },
    }


def _human(text: str = "please check the deploy") -> dict:
    return {
        "type": "user",
        "sessionId": SESSION_KEY,
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


@pytest.fixture
def pane(tmp_path):
    """A cwd, a store root, and one session store, wired so the join succeeds.

    Returns a `(cwd, sessions_root)` pair. Every test builds from this and
    breaks exactly one thing about it.
    """
    cwd = tmp_path / "Projects" / "palaver"
    cwd.mkdir(parents=True)
    sessions_root = tmp_path / "store"
    project = sessions_root / project_key_for_cwd(cwd)
    project.mkdir(parents=True)
    store = project / f"{SESSION_KEY}.jsonl"
    store.write_bytes(_line(_human()) + _line(_tool_use()))
    stamp = (NOW - timedelta(seconds=5)).timestamp()
    os.utime(store, (stamp, stamp))
    return cwd, sessions_root


def _variables(cwd, *, pane_id=PANE, job_pid=JOB_PID, job_name="node", path=None):
    return PaneVariables(
        pane_id=pane_id,
        job_pid=job_pid,
        job_name=job_name,
        path=str(cwd) if path is None else path,
    )


class _Writer:
    """Records variable writes, optionally refusing some panes."""

    def __init__(self, *, fail_for: frozenset[str] = frozenset()):
        self.writes: list[tuple[str, str, object]] = []
        self.fail_for = fail_for

    async def __call__(self, session_id: str, name: str, value: object) -> None:
        if session_id in self.fail_for:
            raise RuntimeError(f"iTerm2 refused a write to {session_id}")
        self.writes.append((session_id, name, value))

    def payload_for(self, session_id: str) -> str | None:
        for written_id, name, value in self.writes:
            if written_id == session_id and name == component.STATUS_VARIABLE:
                return value
        return None

    @property
    def panes(self) -> list[str]:
        return [written_id for written_id, _, _ in self.writes]


def _reader(mapping):
    """Build an async pane-variables reader over a `{pane_id: PaneVariables}`."""

    async def read_variables(pane_id):
        return mapping.get(pane_id)

    return read_variables


def _cwd_reader(cwd):
    def reader(pid):
        return cwd if pid == AGENT_PID else None

    return reader


def _alive(is_alive=True):
    """A liveness probe with a fixed answer.

    Pinned rather than left to the default, which asks this machine whether
    pid 63354 exists -- an answer that changes between runs and would make
    `WORKING` and `UNKNOWN` both plausible outcomes of the same test.
    """

    def probe(_pid):
        return is_alive

    return probe


def _push(pane_ids, *, cwd, sessions_root, writer=None, variables=None, **kwargs):
    """Run one publish tick against the fixture's pane, synchronously."""
    writer = _Writer() if writer is None else writer
    if variables is None:
        variables = {pane_id: _variables(cwd, pane_id=pane_id) for pane_id in pane_ids}
    pushes = asyncio.run(
        publish_once(
            pane_ids,
            read_variables=_reader(variables),
            set_variable=writer,
            sessions_root=sessions_root,
            now=kwargs.pop("now", NOW),
            table=_table(),
            cwd_reader=_cwd_reader(cwd),
            alive_probe=kwargs.pop("alive_probe", _alive()),
            **kwargs,
        )
    )
    return pushes, writer


def _store_with_task(tmp_path, task: str, *, age: timedelta = timedelta(seconds=30)) -> Path:
    """A migrated database carrying one `current_task` row of the given age."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "observe.db"
    migrate(db_path)
    written = (NOW - age).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO projects(name, path) VALUES ('palaver', '/tmp/palaver')")
        conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (1, ?, ?)",
            (PUBLISHABLE_SOURCE, SESSION_KEY),
        )
        conn.execute(
            "INSERT INTO current_state(project_id, session_id, key, value, updated_at) "
            "VALUES (1, 1, 'current_task', ?, ?)",
            (task, written),
        )
    return db_path


def _scope_counts(db_path: Path) -> tuple[int, int]:
    with sqlite3.connect(db_path) as conn:
        projects = conn.execute("SELECT count(*) FROM projects").fetchone()[0]
        sessions = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
    return projects, sessions


# --- a producer exists, and it says something true ---------------------------


def test_a_working_pane_is_published_with_its_status(pane):
    cwd, sessions_root = pane
    pushes, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root)
    assert [push.status for push in pushes] == [Status.WORKING]
    assert writer.panes == [PANE]
    assert component.decode_status(writer.payload_for(PANE), now=NOW.timestamp()) == (
        Status.WORKING,
        None,
    )


def test_codex_status_reads_the_validated_store_path_and_events(tmp_path, monkeypatch):
    """Codex status uses its date-partitioned join path, not Claude layout."""
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    store = tmp_path / "codex" / "2026" / "08" / "15" / "rollout-root.jsonl"
    store.parent.mkdir(parents=True)
    records = [
        {"type": "session_meta", "payload": {"id": "root", "session_id": "root", "cwd": str(cwd)}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": []},
        },
    ]
    store.write_text("".join(json.dumps(record) + "\n" for record in records))
    os.utime(store, (NOW.timestamp(), NOW.timestamp()))
    joined = PaneJoin(
        pane_id=PANE,
        pid=AGENT_PID,
        source=CODEX_SOURCE,
        cwd=cwd,
        project_key="unused",
        session_candidates=(store.stem,),
        session_key=store.stem,
        store_path=store.resolve(),
    )
    monkeypatch.setattr(publisher, "join_pane", lambda *args, **kwargs: joined)
    status, task = status_for_pane(
        PaneVariables(PANE, AGENT_PID, "codex", str(cwd)),
        now=NOW,
        table={},
        cwd_reader=lambda _pid: cwd,
        alive_probe=lambda _pid: True,
    )
    assert status is Status.WORKING
    assert task is None


def test_codex_turn_boundary_is_derived_without_reconstructing_a_claude_path(tmp_path, monkeypatch):
    cwd = tmp_path / "codex-project"
    cwd.mkdir()
    store = tmp_path / "codex" / "2026" / "08" / "15" / "rollout-ended.jsonl"
    store.parent.mkdir(parents=True)
    records = [
        {
            "type": "session_meta",
            "payload": {"id": "ended", "session_id": "ended", "cwd": str(cwd)},
        },
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    store.write_text("".join(json.dumps(record) + "\n" for record in records))
    os.utime(store, (NOW.timestamp(), NOW.timestamp()))
    joined = PaneJoin(
        pane_id=PANE,
        pid=AGENT_PID,
        source=CODEX_SOURCE,
        cwd=cwd,
        project_key="unused",
        session_candidates=(store.stem,),
        session_key=store.stem,
        store_path=store.resolve(),
    )
    monkeypatch.setattr(publisher, "join_pane", lambda *args, **kwargs: joined)
    status, _ = status_for_pane(
        PaneVariables(PANE, AGENT_PID, "codex", str(cwd)),
        now=NOW,
        table={},
        cwd_reader=lambda _pid: cwd,
        alive_probe=lambda _pid: True,
    )
    assert status is Status.AWAITING_HUMAN


def test_the_published_payload_is_what_the_component_would_render(pane):
    cwd, sessions_root = pane
    _, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root)
    line = component.line_for(writer.payload_for(PANE), 40, now=NOW.timestamp())
    assert "working" in line
    assert component.LADYBUG not in line


def test_the_task_text_is_attached_from_the_database(pane, tmp_path):
    cwd, sessions_root = pane
    db_path = _store_with_task(tmp_path, "reading the deploy log")
    pushes, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=db_path)
    assert pushes[0].task == "reading the deploy log"
    assert component.decode_status(writer.payload_for(PANE), now=NOW.timestamp()) == (
        Status.WORKING,
        "reading the deploy log",
    )


def test_the_task_horizon_is_bounded_in_absolute_terms():
    """Expressed as a duration, not as a multiple of itself.

    Every staleness test below writes a row at `TASK_HORIZON + 1s`, which
    scales with the constant: a horizon of ten years would pass all of them.
    """
    assert timedelta(minutes=1) <= TASK_HORIZON <= timedelta(hours=1)


def test_a_task_text_older_than_its_horizon_is_not_attached(pane, tmp_path):
    cwd, sessions_root = pane
    db_path = _store_with_task(tmp_path, "stale work", age=TASK_HORIZON + timedelta(seconds=1))
    pushes, _ = _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=db_path)
    assert pushes[0].task is None
    # Positive control: the same row one second inside the horizon is used.
    fresh = _store_with_task(
        tmp_path / "fresh", "current work", age=TASK_HORIZON - timedelta(seconds=1)
    )
    pushes, _ = _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=fresh)
    assert pushes[0].task == "current work"


def test_a_missing_database_yields_a_status_only_push(pane, tmp_path, caplog):
    """And says nothing about it.

    A machine running the terminal surface without the observer daemon is a
    supported state, not a fault, so it must not log a warning every tick --
    which is what letting SQLite raise and catching it would do.
    """
    cwd, sessions_root = pane
    with caplog.at_level("WARNING", logger="palaver.ui.publisher"):
        pushes, _ = _push(
            [PANE], cwd=cwd, sessions_root=sessions_root, db_path=tmp_path / "nothing-here.db"
        )
    assert pushes[0].task is None
    assert pushes[0].status is Status.WORKING
    assert pushes[0].published
    assert caplog.records == []


# --- the heartbeat -----------------------------------------------------------


def test_an_unchanged_status_is_pushed_again_on_the_next_tick(pane):
    cwd, sessions_root = pane
    writer = _Writer()
    _push([PANE], cwd=cwd, sessions_root=sessions_root, writer=writer)
    _push(
        [PANE], cwd=cwd, sessions_root=sessions_root, writer=writer, now=NOW + timedelta(seconds=30)
    )
    statuses = [name for _, name, _ in writer.writes]
    assert statuses == [component.STATUS_VARIABLE, component.STATUS_VARIABLE]
    first, second = (value for _, _, value in writer.writes)
    assert json.loads(first)["status"] == json.loads(second)["status"]
    assert json.loads(first)[component.PUSHED_AT_KEY] < json.loads(second)[component.PUSHED_AT_KEY]


def test_a_pane_left_unpushed_past_the_horizon_stops_being_believed(pane):
    cwd, sessions_root = pane
    _, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root)
    payload = writer.payload_for(PANE)
    later = NOW.timestamp() + component.STALE_AFTER + 1
    assert component.decode_status(payload, now=later) == (Status.UNKNOWN, None)
    # Positive control: one second inside the horizon it is still believed.
    inside = NOW.timestamp() + component.STALE_AFTER - 1
    assert component.decode_status(payload, now=inside) == (Status.WORKING, None)


def test_the_horizon_moves_with_the_heartbeat():
    """The rule, not the value.

    `STALE_AFTER = 3.0 * PUSH_CADENCE` and `STALE_AFTER = 90.0` are the same
    number today, so no assertion about the value can tell them apart -- the
    difference only appears the day the cadence changes. What is testable is
    the rule the derivation applies, so that is what is asserted, at a
    cadence deliberately not the configured one.
    """
    assert component.stale_horizon(60.0) == 180.0
    assert component.stale_horizon(10.0) == 30.0
    assert component.STALE_AFTER == component.stale_horizon(component.PUSH_CADENCE)
    # Positive control: the horizon is not a constant that ignores its input.
    assert component.stale_horizon(60.0) != component.stale_horizon(10.0)


def test_the_horizon_is_at_least_three_heartbeats():
    assert component.PUSH_CADENCE > 0
    assert component.STALE_AFTER >= 3 * component.PUSH_CADENCE
    # Positive control: the assertion above is capable of failing, so a
    # horizon that had been detached from the cadence would be caught.
    assert not component.STALE_AFTER >= 3 * (component.PUSH_CADENCE + 1)


def test_the_publisher_heartbeats_on_its_own_cadence(pane):
    cwd, sessions_root = pane
    writer = _Writer()
    slept: list[float] = []

    async def sleep(seconds):
        slept.append(seconds)

    registry = autolaunch.SessionRegistry([PANE])
    published = asyncio.run(
        publish_forever(
            registry,
            read_variables=_reader({PANE: _variables(cwd)}),
            set_variable=writer,
            sessions_root=sessions_root,
            limit=3,
            sleep=sleep,
        )
    )
    assert published == 3
    assert writer.panes == [PANE, PANE, PANE]
    # Two sleeps for three ticks: the loop does not sleep after its last one,
    # so a bounded run returns instead of waiting out a cadence nobody needs.
    assert slept == [component.PUSH_CADENCE, component.PUSH_CADENCE]


# --- refusals are pushed, not skipped ----------------------------------------


def test_a_pane_with_no_single_session_is_pushed_unknown(pane):
    cwd, sessions_root = pane
    second = sessions_root / project_key_for_cwd(cwd) / "session-2.jsonl"
    second.write_bytes(_line(_human()))
    stamp = (NOW - timedelta(seconds=5)).timestamp()
    os.utime(second, (stamp, stamp))
    pushes, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root)
    assert [push.status for push in pushes] == [Status.UNKNOWN]
    assert len(writer.writes) == 1
    assert component.decode_status(writer.payload_for(PANE), now=NOW.timestamp()) == (
        Status.UNKNOWN,
        None,
    )


def test_a_pane_with_two_candidates_is_refused_before_any_store_is_read(pane, monkeypatch):
    """The refusal and the failure look identical from outside.

    Dropping the `session_key is None` check yields a store path of
    `None.jsonl`, which does not exist, which is caught, which returns
    `UNKNOWN` -- the same answer by an entirely different route, one of them
    reading a file it had no business naming. So what is asserted is that no
    read was attempted.
    """
    cwd, sessions_root = pane
    second = sessions_root / project_key_for_cwd(cwd) / "session-2.jsonl"
    second.write_bytes(_line(_human()))
    stamp = (NOW - timedelta(seconds=5)).timestamp()
    os.utime(second, (stamp, stamp))

    read: list[object] = []
    real = publisher.observe_session

    def spy(path, **kwargs):
        read.append(path)
        return real(path, **kwargs)

    monkeypatch.setattr(publisher, "observe_session", spy)
    pushes, _ = _push([PANE], cwd=cwd, sessions_root=sessions_root)
    assert pushes[0].status is Status.UNKNOWN
    assert read == []

    # Positive control: with one candidate the same call does read a store.
    second.unlink()
    _push([PANE], cwd=cwd, sessions_root=sessions_root)
    assert len(read) == 1


def test_every_attached_pane_is_answered(pane):
    cwd, sessions_root = pane
    panes = [PANE, "pane-2", "pane-3"]
    variables = {
        PANE: _variables(cwd),
        "pane-2": _variables(cwd, pane_id="pane-2", job_pid=None),
        "pane-3": _variables(cwd, pane_id="pane-3", path="/nowhere/that/exists"),
    }
    pushes, writer = _push(panes, cwd=cwd, sessions_root=sessions_root, variables=variables)
    assert len(pushes) == len(panes)
    assert writer.panes == panes
    assert [push.status for push in pushes] == [Status.WORKING, Status.UNKNOWN, Status.UNKNOWN]


def test_a_pane_iterm2_will_not_describe_is_pushed_unknown(pane):
    cwd, sessions_root = pane
    pushes, writer = _push([PANE], cwd=cwd, sessions_root=sessions_root, variables={})
    assert pushes[0].status is Status.UNKNOWN
    assert pushes[0].published
    assert writer.payload_for(PANE) is not None


def test_a_pane_running_another_agent_is_pushed_unknown(pane):
    cwd, sessions_root = pane
    tree = tuple(
        (pid, ppid, "codex" if command == "claude" else command)
        for pid, ppid, command in CLAUDE_TREE
    )
    status, task = status_for_pane(
        _variables(cwd),
        sessions_root=sessions_root,
        now=NOW,
        table=_table(tree),
        cwd_reader=_cwd_reader(cwd),
        alive_probe=_alive(True),
    )
    assert (status, task) == (Status.UNKNOWN, None)
    # Positive control: the same tree with `claude` in it does resolve.
    status, _ = status_for_pane(
        _variables(cwd),
        sessions_root=sessions_root,
        now=NOW,
        table=_table(),
        cwd_reader=_cwd_reader(cwd),
        alive_probe=_alive(True),
    )
    assert status is Status.WORKING


def test_an_unknown_status_never_carries_task_text(pane, tmp_path):
    """The join succeeds, the store reads, and the *status* is what withdraws.

    A dead agent whose pid is gone from the process table refuses at the join
    and never reaches the database, so it proves nothing about this rule. The
    case that does is a pane that joins and reads normally and is then
    demoted by `apply_liveness` -- there, a `current_task` row exists, is
    fresh, and must still not be shown.
    """
    cwd, sessions_root = pane
    db_path = _store_with_task(tmp_path, "work that is no longer claimable")
    status, task = status_for_pane(
        _variables(cwd),
        sessions_root=sessions_root,
        db_path=db_path,
        now=NOW,
        table=_table(),
        cwd_reader=_cwd_reader(cwd),
        alive_probe=_alive(False),
    )
    assert (status, task) == (Status.UNKNOWN, None)

    # Positive control: the same store, the same row, a live agent -- the task
    # text is there to be read, so its absence above is the status doing it.
    status, task = status_for_pane(
        _variables(cwd),
        sessions_root=sessions_root,
        db_path=db_path,
        now=NOW,
        table=_table(),
        cwd_reader=_cwd_reader(cwd),
        alive_probe=_alive(True),
    )
    assert (status, task) == (Status.WORKING, "work that is no longer claimable")


# --- one pane's failure is one pane's failure --------------------------------


def test_one_panes_failed_write_does_not_stop_the_others(pane):
    cwd, sessions_root = pane
    panes = [PANE, "pane-2", "pane-3"]
    writer = _Writer(fail_for=frozenset({"pane-2"}))
    pushes, _ = _push(panes, cwd=cwd, sessions_root=sessions_root, writer=writer)
    assert len(writer.writes) == len(panes) - 1
    assert writer.panes == [PANE, "pane-3"]
    assert [push.published for push in pushes] == [True, False, True]
    # Positive control: with no refusal every pane is written.
    ok = _Writer()
    _push(panes, cwd=cwd, sessions_root=sessions_root, writer=ok)
    assert len(ok.writes) == len(panes)


def test_the_progress_line_counts_what_was_written(pane):
    """INV-1's channel must not report a refused write as a published one."""
    cwd, sessions_root = pane
    panes = [PANE, "pane-2", "pane-3"]
    messages: list[str] = []
    _push(
        panes,
        cwd=cwd,
        sessions_root=sessions_root,
        writer=_Writer(fail_for=frozenset({"pane-2"})),
        on_status=messages.append,
    )
    assert messages[-1] == "published 2/3 pane(s)"
    # Positive control: with nothing refused the same line reads 3/3.
    messages.clear()
    _push(panes, cwd=cwd, sessions_root=sessions_root, on_status=messages.append)
    assert messages[-1] == "published 3/3 pane(s)"


def test_a_failed_push_is_reported_rather_than_dropped(pane):
    cwd, sessions_root = pane
    writer = _Writer(fail_for=frozenset({PANE}))
    pushes, _ = _push([PANE], cwd=cwd, sessions_root=sessions_root, writer=writer)
    # The status it *would* have shown is kept, so a log reader can tell a
    # pane Palaver could not describe from a pane it could not write to.
    assert pushes == (PanePush(pane_id=PANE, status=Status.WORKING, task=None, payload=None),)
    assert not pushes[0].published


def test_a_failed_push_is_logged_with_its_traceback(pane, caplog):
    cwd, sessions_root = pane
    writer = _Writer(fail_for=frozenset({PANE}))
    with caplog.at_level("WARNING", logger="palaver.ui.publisher"):
        _push([PANE], cwd=cwd, sessions_root=sessions_root, writer=writer)
    assert any(record.exc_info for record in caplog.records)
    assert any(PANE in record.getMessage() for record in caplog.records)


# --- the registry is read live -----------------------------------------------


def test_each_tick_pushes_the_registry_as_it_then_stands(pane):
    cwd, sessions_root = pane
    registry = autolaunch.SessionRegistry([PANE, "pane-2"])
    writer = _Writer()
    ticks = {"n": 0}

    async def sleep(_seconds):
        # Stands in for the monitors, which mutate the registry between ticks.
        ticks["n"] += 1
        if ticks["n"] == 1:
            registry.detach("pane-2")
            registry.attach("pane-3")

    variables = {
        pane_id: _variables(cwd, pane_id=pane_id) for pane_id in (PANE, "pane-2", "pane-3")
    }
    asyncio.run(
        publish_forever(
            registry,
            read_variables=_reader(variables),
            set_variable=writer,
            sessions_root=sessions_root,
            limit=2,
            sleep=sleep,
        )
    )
    assert writer.panes[:2] == [PANE, "pane-2"]
    assert writer.panes[2:] == [PANE, "pane-3"]


# --- the publisher is not the writer -----------------------------------------


def test_publishing_creates_no_scope_rows(pane, tmp_path):
    cwd, sessions_root = pane
    db_path = tmp_path / "empty.db"
    migrate(db_path)
    before = _scope_counts(db_path)
    _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=db_path)
    assert _scope_counts(db_path) == before == (0, 0)


def test_the_database_is_opened_read_only(pane, tmp_path):
    cwd, sessions_root = pane
    db_path = _store_with_task(tmp_path, "reading the deploy log")
    opened: list[str] = []
    real_connect = sqlite3.connect

    def spy(target, *args, **kwargs):
        opened.append(str(target))
        return real_connect(target, *args, **kwargs)

    sqlite3.connect = spy
    try:
        _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=db_path)
    finally:
        sqlite3.connect = real_connect
    assert opened
    assert all("mode=ro" in target for target in opened)


def test_an_unreadable_database_degrades_to_status_only(pane, tmp_path):
    cwd, sessions_root = pane
    db_path = tmp_path / "not-a-database.db"
    db_path.write_bytes(b"this is not a SQLite file")
    pushes, _ = _push([PANE], cwd=cwd, sessions_root=sessions_root, db_path=db_path)
    assert pushes[0].status is Status.WORKING
    assert pushes[0].task is None


def test_a_database_with_no_row_for_the_session_reads_none(tmp_path):
    db_path = _store_with_task(tmp_path, "someone else's work")
    assert (
        read_current_task(
            db_path, source=PUBLISHABLE_SOURCE, session_key="session-elsewhere", now=NOW
        )
        is None
    )
    # Positive control: the session that does have a row reads it.
    assert (
        read_current_task(db_path, source=PUBLISHABLE_SOURCE, session_key=SESSION_KEY, now=NOW)
        == "someone else's work"
    )


def test_two_agents_sharing_a_session_id_do_not_share_a_task(tmp_path):
    """`sessions` is unique on `(source, external_id)`, not on `external_id`.

    Session ids are each agent's own namespace and nothing stops two of them
    minting the same one, so a lookup that matched on the id alone would show
    one agent's work in another agent's pane.
    """
    db_path = _store_with_task(tmp_path, "the claude session's work")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions(project_id, source, external_id) VALUES (1, 'codex', ?)",
            (SESSION_KEY,),
        )
        conn.execute(
            "INSERT INTO current_state(project_id, session_id, key, value, updated_at) "
            "VALUES (1, 2, 'current_task', ?, ?)",
            ("the codex session's work", NOW.strftime("%Y-%m-%dT%H:%M:%S.%fZ")),
        )
    assert (
        read_current_task(db_path, source=PUBLISHABLE_SOURCE, session_key=SESSION_KEY, now=NOW)
        == "the claude session's work"
    )
    assert (
        read_current_task(db_path, source="codex", session_key=SESSION_KEY, now=NOW)
        == "the codex session's work"
    )


def test_task_lookup_is_source_scoped_even_when_only_one_source_has_a_row(tmp_path):
    db_path = _store_with_task(tmp_path, "claude-only work")
    assert read_current_task(db_path, source="codex", session_key=SESSION_KEY, now=NOW) is None
    assert (
        read_current_task(db_path, source=PUBLISHABLE_SOURCE, session_key=SESSION_KEY, now=NOW)
        == "claude-only work"
    )


# --- reading the pane's variables back out of iTerm2 -------------------------


def _stub_rpc(monkeypatch, values, *, ok=True):
    """Stub `iterm2.rpc.async_variable` with one canned response."""
    import iterm2.api_pb2
    import iterm2.rpc

    ok_status = iterm2.api_pb2.VariableResponse.Status.Value("OK")
    other = ok_status + 1

    async def async_variable(_connection, _session_id, _sets, _gets):
        response = iterm2.api_pb2.VariableResponse()
        response.status = ok_status if ok else other
        response.values.extend(values)
        return type("Result", (), {"variable_response": response})()

    monkeypatch.setattr(iterm2.rpc, "async_variable", async_variable)


def test_the_reader_decodes_a_pane_iterm2_describes(monkeypatch):
    _stub_rpc(monkeypatch, [json.dumps(JOB_PID), json.dumps("node"), json.dumps("/tmp/project")])
    read = publisher.make_variables_reader(object())
    assert asyncio.run(read(PANE)) == PaneVariables(
        pane_id=PANE, job_pid=JOB_PID, job_name="node", path="/tmp/project"
    )


def test_the_reader_reports_a_pane_with_no_job_rather_than_guessing(monkeypatch):
    _stub_rpc(monkeypatch, ["null", "null", json.dumps("/tmp/project")])
    read = publisher.make_variables_reader(object())
    variables = asyncio.run(read(PANE))
    assert (variables.job_pid, variables.job_name) == (None, None)
    assert variables.path == "/tmp/project"


def test_the_reader_refuses_a_pane_iterm2_would_not_answer_for(monkeypatch):
    _stub_rpc(monkeypatch, ["null", "null", "null"], ok=False)
    read = publisher.make_variables_reader(object())
    assert asyncio.run(read(PANE)) is None


def test_a_job_pid_that_is_not_a_number_is_not_a_pid(monkeypatch):
    _stub_rpc(monkeypatch, [json.dumps("not a pid"), json.dumps("node"), json.dumps("/tmp")])
    read = publisher.make_variables_reader(object())
    assert asyncio.run(read(PANE)).job_pid is None
    # Positive control: zero and negatives are refused too, and a real pid is not.
    _stub_rpc(monkeypatch, [json.dumps(0), json.dumps("node"), json.dumps("/tmp")])
    assert asyncio.run(publisher.make_variables_reader(object())(PANE)).job_pid is None
    _stub_rpc(monkeypatch, [json.dumps(JOB_PID), json.dumps("node"), json.dumps("/tmp")])
    assert asyncio.run(publisher.make_variables_reader(object())(PANE)).job_pid == JOB_PID


def test_a_variable_that_is_not_json_is_read_as_absent(monkeypatch):
    _stub_rpc(monkeypatch, ["<not json>", "<not json>", "<not json>"])
    read = publisher.make_variables_reader(object())
    assert asyncio.run(read(PANE)) == PaneVariables(
        pane_id=PANE, job_pid=None, job_name=None, path=None
    )


# --- the wiring --------------------------------------------------------------


def test_the_autolaunch_daemon_runs_the_publisher(pane, monkeypatch):
    cwd, sessions_root = pane
    ran: list[object] = []

    async def fake_register(_connection):
        return None

    async def fake_publish_forever(registry, **kwargs):
        ran.append(registry)
        return 0

    monkeypatch.setattr(component, "register", fake_register)
    monkeypatch.setattr(publisher, "publish_forever", fake_publish_forever)
    monkeypatch.setattr(publisher, "make_variables_reader", lambda _connection: _reader({}))
    monkeypatch.setattr(component, "make_variable_writer", lambda _connection: _Writer())
    _run_main(monkeypatch, publish=True)
    assert len(ran) == 1


def test_autolaunch_registers_once_before_attaching_or_publishing(monkeypatch):
    """The visible component must exist before its variables can be rendered."""
    events: list[str] = []
    statuses: list[str] = []

    async def fake_register(_connection):
        events.append("register")

    async def fake_attach(app, _registry, **_kwargs):
        sessions = app.terminal_windows[0].tabs[0].sessions
        assert len(sessions) == 2, "the count must not depend on how many panes attach"
        events.append("attach")
        return len(sessions)

    async def fake_publish_forever(_registry, **_kwargs):
        events.append("publish")
        return 0

    monkeypatch.setattr(component, "register", fake_register)
    monkeypatch.setattr(autolaunch, "attach_existing", fake_attach)
    monkeypatch.setattr(publisher, "publish_forever", fake_publish_forever)
    _run_main(
        monkeypatch,
        publish=True,
        session_ids=("one", "two"),
        on_status=statuses.append,
    )

    assert events == ["register", "attach", "publish"]
    assert statuses[0] == "registering Palaver status component"


def test_the_publisher_can_be_left_out(pane, monkeypatch):
    ran: list[object] = []

    async def fake_register(_connection):
        return None

    async def fake_publish_forever(registry, **kwargs):
        ran.append(registry)
        return 0

    monkeypatch.setattr(component, "register", fake_register)
    monkeypatch.setattr(publisher, "publish_forever", fake_publish_forever)
    _run_main(monkeypatch, publish=False)
    assert ran == []


def _run_main(
    monkeypatch,
    *,
    publish: bool,
    session_ids=(),
    on_status=lambda _message: None,
):
    """Drive `autolaunch.main` over a stub library, one event per monitor."""

    class _Monitor:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def async_get(self):
            return None

    sessions = tuple(types.SimpleNamespace(session_id=session_id) for session_id in session_ids)
    app = types.SimpleNamespace(
        terminal_windows=(types.SimpleNamespace(tabs=(types.SimpleNamespace(sessions=sessions),)),)
        if sessions
        else ()
    )

    async def async_get_app(_connection):
        return app

    stub = type(
        "StubIterm2",
        (),
        {
            "NewSessionMonitor": _Monitor,
            "SessionTerminationMonitor": _Monitor,
            "async_get_app": staticmethod(async_get_app),
        },
    )()
    monkeypatch.setattr(autolaunch, "import_iterm2", lambda: stub)
    return asyncio.run(
        autolaunch.main(object(), limit=1, publish=publish, on_status=on_status)
    )
