"""Tests for the adapter interface, session discovery windowing, and cursors.

Every fixture is a JSONL file this test suite writes itself under pytest's
`tmp_path`. No test in this module opens, globs, or otherwise references a
real observed session store (`~/.claude/`, `~/.codex/`,
`~/.local/share/opencode/`) — INV-3 forbids it, and these tests exist to
exercise the adapter interface against fixtures, not real data.

`FixtureAdapter` is a minimal `Adapter` over a directory of JSONL files,
built only for these tests. Each record is a JSON object with a "kind"
field; a session's last record determines whether it has an unresolved
trailing `tool_use`.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from palaver.ingest.adapters.base import (
    Adapter,
    Event,
    TailResult,
    read_complete_records,
)
from palaver.ingest.cursors import Cursor, CursorStore

NOW = datetime(2026, 8, 14, tzinfo=timezone.utc)


class FixtureAdapter(Adapter):
    """A JSONL-backed Adapter over a fixture directory, for this test module only."""

    source = "fixture"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_store_paths(self):
        return sorted(self.root.glob("*.jsonl"))

    def session_key_for(self, path: Path) -> str:
        return path.stem

    def has_unresolved_trailing_tool_use(self, path: Path) -> bool:
        records, _ = read_complete_records(path, 0)
        if not records:
            return False
        return json.loads(records[-1])["kind"] == "tool_use"

    def tail(self, path: Path, cursor: Cursor) -> TailResult:
        records, new_offset = read_complete_records(path, cursor.offset)
        session_key = self.session_key_for(path)
        events = []
        for record in records:
            payload = json.loads(record)
            events.append(Event(session_key=session_key, kind=payload["kind"], payload=payload))
        return TailResult(events=tuple(events), cursor=Cursor(offset=new_offset))


def _jsonl_line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_bytes(b"".join(_jsonl_line(r) for r in records))


def _append_records(path: Path, records: list[dict]) -> None:
    with path.open("ab") as f:
        f.write(b"".join(_jsonl_line(r) for r in records))


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def test_discover_sessions_default_window_includes_recent_session(tmp_path):
    """A session modified within the default 24h window is returned."""
    store = tmp_path / "store"
    store.mkdir()
    path = store / "recent.jsonl"
    _write_records(path, [{"kind": "message", "seq": 1}])
    _set_mtime(path, timedelta(hours=1))

    sessions = FixtureAdapter(store).discover_sessions(now=NOW)

    assert {s.path for s in sessions} == {path}


def test_discover_sessions_includes_old_session_with_unresolved_tool_use(tmp_path):
    """The window is a floor, not a filter: an old session with an unresolved trailing
    tool_use is included even though it is well outside the default 24h window."""
    store = tmp_path / "store"
    store.mkdir()
    path = store / "stale-but-working.jsonl"
    _write_records(path, [{"kind": "message", "seq": 1}, {"kind": "tool_use", "seq": 2}])
    _set_mtime(path, timedelta(days=3))  # older than 24h, inside the 7d outer window

    sessions = FixtureAdapter(store).discover_sessions(now=NOW)

    assert {s.path for s in sessions} == {path}


def test_discover_sessions_excludes_old_session_with_resolved_trailing_record(tmp_path):
    """An old session whose last record is not a tool_use stays excluded."""
    store = tmp_path / "store"
    store.mkdir()
    path = store / "stale-and-done.jsonl"
    _write_records(path, [{"kind": "message", "seq": 1}, {"kind": "tool_result", "seq": 2}])
    _set_mtime(path, timedelta(days=3))

    sessions = FixtureAdapter(store).discover_sessions(now=NOW)

    assert sessions == []


def test_discover_sessions_excludes_and_never_opens_beyond_outer_window(tmp_path, monkeypatch):
    """A session past the outer 7d window is excluded on mtime alone and never opened.

    The fixture's last record is an unresolved tool_use, which would flip the
    result to "included" if the always-include rule were unbounded — proving
    the exclusion here depends on the outer-window bound, not on the content.
    """
    store = tmp_path / "store"
    store.mkdir()
    path = store / "ancient.jsonl"
    _write_records(path, [{"kind": "tool_use", "seq": 1}])
    _set_mtime(path, timedelta(days=10))

    opened = []
    real_open = os.open

    def _counting_open(p, *args, **kwargs):
        opened.append(p)
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(os, "open", _counting_open)
    adapter = FixtureAdapter(store)

    sessions = adapter.discover_sessions(now=NOW)

    assert sessions == []
    assert opened == []

    # Positive control: the same fixture, same spy, but a wide enough
    # outer_window to bring it into the always-include check. This proves
    # the spy is actually live and wired to the open path discover_sessions
    # uses — the exclusion above came from the outer-window bound, not from
    # an inert counter or a spy on the wrong function.
    sessions = adapter.discover_sessions(now=NOW, outer_window=timedelta(days=30))

    assert {s.path for s in sessions} == {path}
    assert opened == [path]


def test_discover_sessions_all_returns_every_session(tmp_path):
    """--all (all=True) returns every session regardless of age or trailing record."""
    store = tmp_path / "store"
    store.mkdir()
    recent = store / "recent.jsonl"
    ancient = store / "ancient.jsonl"
    _write_records(recent, [{"kind": "message", "seq": 1}])
    _write_records(ancient, [{"kind": "tool_result", "seq": 1}])
    _set_mtime(recent, timedelta(hours=1))
    _set_mtime(ancient, timedelta(days=30))

    sessions = FixtureAdapter(store).discover_sessions(all=True, now=NOW)

    assert {s.path for s in sessions} == {recent, ancient}


def test_tail_mid_line_split_yields_nothing_until_completed(tmp_path):
    """A single record flushed as two chunks, split mid-line, is not yielded until complete.

    The writer flushes only the first half of one record's line — no
    terminating newline yet — simulating a torn write caught mid-flush. The
    partial read must yield nothing at all (the record is not yet complete)
    and must not advance the cursor past it. Once the rest of the line
    arrives, tailing from that same cursor yields the record exactly once.
    """
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"

    full_line = json.dumps({"kind": "message", "seq": 1})
    midpoint = len(full_line) // 2
    first_half = full_line[:midpoint].encode("utf-8")
    second_half = full_line[midpoint:].encode("utf-8") + b"\n"

    path.write_bytes(first_half)

    adapter = FixtureAdapter(store)
    partial_result = adapter.tail(path, Cursor())

    assert partial_result.events == ()
    assert partial_result.cursor.offset == 0

    with path.open("ab") as f:
        f.write(second_half)

    completed_result = adapter.tail(path, partial_result.cursor)

    assert len(completed_result.events) == 1
    assert completed_result.events[0].payload == {"kind": "message", "seq": 1}
    assert completed_result.cursor.offset == len(first_half) + len(second_half)


def test_tail_cursor_advances_only_past_complete_record_with_partial_trailing(tmp_path):
    """With a complete record AND a trailing partial, the cursor stops exactly at the
    newline — not at len(raw) — so the partial is re-read whole, not skipped or lost.

    This is the production shape: a live writer's buffer holds one whole
    record plus a torn start of the next. An off-by-one that used the raw
    read length instead of the last newline's position would either drop
    the second record's bytes or hand a truncated string to the JSON
    decoder on the next tick.
    """
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"

    line1 = _jsonl_line({"kind": "message", "seq": 1})
    line2_full = json.dumps({"kind": "message", "seq": 2})
    midpoint = len(line2_full) // 2
    line2_first_half = line2_full[:midpoint].encode("utf-8")
    line2_second_half = line2_full[midpoint:].encode("utf-8") + b"\n"

    path.write_bytes(line1 + line2_first_half)

    adapter = FixtureAdapter(store)
    first_result = adapter.tail(path, Cursor())

    assert len(first_result.events) == 1
    assert first_result.events[0].payload == {"kind": "message", "seq": 1}
    assert first_result.cursor.offset == len(line1)  # stops at the newline, not len(raw)

    with path.open("ab") as f:
        f.write(line2_second_half)

    second_result = adapter.tail(path, first_result.cursor)

    assert len(second_result.events) == 1
    assert second_result.events[0].payload == {"kind": "message", "seq": 2}  # not re-seen


def test_tail_recovers_when_source_shrinks_below_stored_cursor(tmp_path, caplog):
    """A source shorter than the stored cursor is detected and recovered, not
    read as silence forever (Task 3, TASKS.md).

    Before this fix, seeking to an offset past a shrunk file's new EOF read
    as an empty tail with the cursor pinned at the stale offset: the session
    would go permanently silent with no error on every future tail, since
    `new_offset` never advances past a past-EOF seek. This truncates the
    fixture in place to simulate the shrink (rotation or replacement takes
    the same path, since the mechanism keys on size, not the file's inode
    history) and asserts the very next tail both surfaces the replacement's
    content and repairs the cursor into the new file, rather than repeating
    the silent-empty result the old code would have produced forever after.
    """
    caplog.set_level(logging.WARNING)
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"

    _write_records(path, [{"kind": "message", "seq": n} for n in range(1, 20)])
    adapter = FixtureAdapter(store)
    grown_result = adapter.tail(path, Cursor())
    stale_cursor = grown_result.cursor
    assert stale_cursor.offset == path.stat().st_size  # sanity: cursor caught all the way up

    # Shrink the file far below the stale cursor, as if a new, shorter
    # session's content had landed under the same path/key.
    _write_records(path, [{"kind": "message", "seq": 1}])
    assert path.stat().st_size < stale_cursor.offset  # the shrink this test targets

    recovered = adapter.tail(path, stale_cursor)

    assert len(recovered.events) == 1  # not silently empty forever
    assert recovered.events[0].payload == {"kind": "message", "seq": 1}
    assert recovered.cursor.offset <= path.stat().st_size  # repaired within the file's new length
    assert any("shrank below its cursor" in r.message for r in caplog.records)


def test_tail_shrink_recovery_does_not_fire_on_an_ordinary_caught_up_retail(tmp_path, caplog):
    """Positive control for the shrink check: keying on `offset > size`, not on
    every re-tail with a nonzero cursor.

    Without this control, a shrink-recovery implementation that (incorrectly)
    triggered on any cursor offset other than 0 — rather than specifically
    on the cursor being past the file's current size — would still pass the
    test above, since that scenario also has a nonzero cursor. Re-tailing a
    file that has not grown since the cursor was last saved must stay a
    plain, silent no-op: no new events, cursor unchanged, and critically no
    shrink warning logged, proving the branch above did not run.
    """
    caplog.set_level(logging.WARNING)
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"
    _write_records(path, [{"kind": "message", "seq": 1}])
    adapter = FixtureAdapter(store)
    caught_up = adapter.tail(path, Cursor())
    assert caught_up.cursor.offset == path.stat().st_size

    steady = adapter.tail(path, caught_up.cursor)

    assert steady.events == ()
    assert steady.cursor.offset == caught_up.cursor.offset
    assert caplog.records == []


def test_cursor_resumes_after_restart_without_reingest_or_skip(tmp_path):
    """A durable cursor reloaded after a simulated restart neither re-ingests nor skips."""
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"
    _write_records(path, [{"kind": "message", "seq": 1}, {"kind": "message", "seq": 2}])

    adapter = FixtureAdapter(store)
    session_key = adapter.session_key_for(path)
    cursor_dir = tmp_path / "cursors"

    pre_restart_store = CursorStore(cursor_dir)
    first_pass = adapter.tail(path, pre_restart_store.load(session_key))
    pre_restart_store.save(session_key, first_pass.cursor)

    assert [e.payload["seq"] for e in first_pass.events] == [1, 2]

    # New records arrive, then the process "restarts": a fresh CursorStore
    # instance is opened over the same directory rather than reusing the one
    # already in memory.
    _append_records(path, [{"kind": "message", "seq": 3}])
    post_restart_store = CursorStore(cursor_dir)
    resumed_cursor = post_restart_store.load(session_key)

    assert resumed_cursor == first_pass.cursor

    second_pass = adapter.tail(path, resumed_cursor)

    assert [e.payload["seq"] for e in second_pass.events] == [3]


def test_cursor_store_load_before_any_save_starts_at_zero(tmp_path):
    """A session never tailed before yields a fresh Cursor(offset=0), not an error."""
    store = CursorStore(tmp_path / "cursors")

    assert store.load("never-seen-session") == Cursor(offset=0)


def test_adapters_never_open_source_writable(tmp_path, monkeypatch):
    """No operation ever requests write access to an observed session store (INV-2).

    The fixture file is chmod'd read-only before any adapter call runs, so a
    code path that requested a write-capable open would raise a real
    PermissionError rather than merely being unobserved. On top of that, the
    exact `os.open` flags used on every call are captured and asserted to
    carry only the read-only access mode — proving the mode requested, not
    just that reading happened to succeed.
    """
    store = tmp_path / "store"
    store.mkdir()
    path = store / "session-1.jsonl"
    _write_records(path, [{"kind": "tool_use", "seq": 1}])
    # Old enough to need the has_unresolved_trailing_tool_use check (which
    # opens the file) rather than being included on recency alone — this
    # exercises the discovery-side open path, not just tail()'s.
    _set_mtime(path, timedelta(days=3))
    path.chmod(0o444)

    observed_flags = []
    real_open = os.open

    def _spy_open(p, flags, *args, **kwargs):
        observed_flags.append(flags)
        return real_open(p, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", _spy_open)

    adapter = FixtureAdapter(store)
    tail_result = adapter.tail(path, Cursor())
    sessions = adapter.discover_sessions(now=NOW)

    assert tail_result.events  # the chmod'd-read-only file was still readable
    assert sessions  # exercised has_unresolved_trailing_tool_use's open path too
    assert observed_flags, "expected at least one os.open call to inspect"
    assert all(flags & os.O_ACCMODE == os.O_RDONLY for flags in observed_flags)
