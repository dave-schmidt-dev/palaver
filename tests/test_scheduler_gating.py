"""
Cursor-gated scheduling: an idle store costs nothing, a changed session gets exactly one
request, and a shrunken store is scheduled, not read as idle.
"""

import json
from datetime import timedelta
from pathlib import Path

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.ingest.cursors import Cursor, CursorStore
from palaver.observer.scheduler import plan_tick
from tests._test_scheduler_support import (
    NOW,
    RecordingExtractor,
    _assistant,
    _daemon,
    _human,
    _set_mtime,
    _write_store,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _append(path: Path, record: dict) -> None:
    """Append one complete record, the way an agent writing this store would."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _frozen_sample(tmp_path: Path) -> Path:
    """Copy the committed flat `tests/fixtures/` corpus into adapter shape.

    `glob("*.jsonl")` is deliberately non-recursive: the flat root is the
    Claude Code transcript namespace, and `tests/fixtures/labels/` holds
    task 7.1's label artifacts, which are not session stores.
    """
    project_dir = tmp_path / "projects" / "fixture-corpus"
    project_dir.mkdir(parents=True)
    for fixture in sorted(FIXTURES_DIR.glob("*.jsonl")):
        copy = project_dir / fixture.name
        copy.write_bytes(fixture.read_bytes())
        _set_mtime(copy, timedelta(minutes=5))
    return tmp_path / "projects"


def _warm_cursors(adapter: ClaudeCodeAdapter, cursors: CursorStore) -> list[str]:
    """Seed every session's cursor to where a tail leaves it, as if already read.

    This is what makes a store *static* rather than merely unchanging: a
    cold `CursorStore` hands back `Cursor(offset=0)` for an unknown session,
    which the scheduler correctly reads as "everything after byte zero is
    new". Seeding uses the adapter itself, not the scheduler, so the fixture
    is not defined in terms of the thing under test.

    Returns:
        Every seeded `session_key`.
    """
    keys = []
    for ref in adapter.discover_sessions(all=True):
        cursor = adapter.tail(ref.path, Cursor()).cursor
        assert cursor.offset > 0, f"{ref.session_key} tailed to offset 0; nothing was seeded"
        cursors.save(ref.session_key, cursor)
        keys.append(ref.session_key)
    return keys


# --- the idle case: zero inference requests ----------------------------------


def test_ten_ticks_over_an_idle_store_record_zero_inference_requests(tmp_path):
    """Ten ticks, nothing changed, zero requests — then one append proves the counter counts.

    The zero is only meaningful alongside the two facts asserted with it:
    discovery found sessions, and every one of them was *skipped* rather
    than never looked at. The eleventh tick is the positive control — the
    same daemon, the same counter, one appended record, exactly one request.
    """
    sample_root = _frozen_sample(tmp_path)
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    seeded = _warm_cursors(adapter, cursors)
    assert seeded, "the frozen fixture corpus produced no sessions to seed"

    extractor = RecordingExtractor()
    with _daemon(tmp_path, sample_root, extractor) as daemon:
        for _ in range(10):
            result = daemon.tick(now=NOW)
            assert result.plan.discovered == len(seeded)
            assert len(result.plan.skipped) == len(seeded)
            assert result.plan.scheduled == ()

        assert extractor.calls == []

        # Positive control: the counter above can count.
        changed = sorted((sample_root / "fixture-corpus").glob("*.jsonl"))[0]
        _append(changed, _human("one more turn"))
        eleventh = daemon.tick(now=NOW)

    assert len(eleventh.plan.scheduled) == 1
    assert len(extractor.calls) == 1


def test_idle_ticks_do_not_rewrite_cursors(tmp_path):
    """A skipped session's cursor file is not touched — idle really is a no-op."""
    sample_root = _frozen_sample(tmp_path)
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    seeded = _warm_cursors(adapter, cursors)

    before = {key: cursors.load(key).offset for key in seeded}
    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        daemon.tick(now=NOW)
        daemon.tick(now=NOW)

    assert {key: cursors.load(key).offset for key in seeded} == before


# --- the changed case: exactly one request per tick --------------------------


def test_advanced_cursor_emits_exactly_one_extraction_request_per_tick(tmp_path):
    """One append, one request. Two appends over two ticks, two requests. Then nothing."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human(), _assistant()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    cursors.save(key, ClaudeCodeAdapter(root=sample_root).tail(path, Cursor()).cursor)

    extractor = RecordingExtractor()
    with _daemon(tmp_path, sample_root, extractor) as daemon:
        _append(path, _human("and now the other thing"))
        first = daemon.tick(now=NOW)
        assert len(first.plan.scheduled) == 1
        assert extractor.calls == [key]

        _append(path, _assistant("done"))
        second = daemon.tick(now=NOW)
        assert len(second.plan.scheduled) == 1
        assert extractor.calls == [key, key]

        # Nothing appended: the gate is change, not existence.
        third = daemon.tick(now=NOW)

    assert third.plan.scheduled == ()
    assert len(third.plan.skipped) == 1
    assert extractor.calls == [key, key]


def test_a_successful_extraction_advances_the_persisted_cursor(tmp_path):
    """The cursor the tick reported is the cursor the store holds afterwards."""
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human()])
    cursors = CursorStore(tmp_path / "cursors")
    key = ClaudeCodeAdapter(root=sample_root).session_key_for(path)
    assert cursors.load(key).offset == 0

    with _daemon(tmp_path, sample_root, RecordingExtractor()) as daemon:
        result = daemon.tick(now=NOW)

    assert result.extracted == (key,)
    assert cursors.load(key).offset == result.plan.scheduled[0].cursor_after.offset
    assert cursors.load(key).offset > 0


# --- shrink recovery is change, not idleness ---------------------------------


def test_a_shrunken_store_is_scheduled_rather_than_read_as_idle(tmp_path):
    """A truncated or swapped store returns a lower cursor and must still be scheduled.

    `read_complete_records` repairs a past-EOF cursor by re-reading from 0
    (task 1.3), which returns an offset *below* the one it was handed. A
    scheduler gating on `cursor > before` would classify that as idle and
    never persist the repair, leaving the session silently unread forever.
    """
    sample_root = tmp_path / "projects"
    path = _write_store(sample_root, "proj", "session-1", [_human(), _assistant(), _human("more")])
    adapter = ClaudeCodeAdapter(root=sample_root)
    cursors = CursorStore(tmp_path / "cursors")
    key = adapter.session_key_for(path)
    cursors.save(key, adapter.tail(path, Cursor()).cursor)

    assert plan_tick((adapter,), cursors, all=True, now=NOW).scheduled == ()

    _write_store(sample_root, "proj", "session-1", [_human("a fresh, shorter file")])
    plan = plan_tick((adapter,), cursors, all=True, now=NOW)

    assert len(plan.scheduled) == 1
    work = plan.scheduled[0]
    assert work.cursor_after.offset < work.cursor_before.offset
    assert work.cursor_after.offset > 0
