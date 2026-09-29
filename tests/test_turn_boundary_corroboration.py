"""
Corroboration (stop-hook and mtime) is reported but never applied, the remaining
signals, and the injected-record adapter fix.
"""

from datetime import timedelta

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import (
    SIGNAL_NAMES,
    Status,
    Tri,
    derive_status,
)
from palaver.observer.turn_boundary import (
    BASIS_SOURCE_UNREADABLE,
    BASIS_UNDECODABLE_RECORD,
    derive_turn_boundary,
)
from tests._turn_boundary_support import (
    NOW,
    _assistant,
    _human,
    _injected,
    _observe,
    _session,
    _set_mtime,
    _status,
    _stop_hook,
    _tool_result,
    _tool_use,
)

# --- corroboration is never a dependency -------------------------------------


def test_stop_hook_record_cannot_override_the_structural_boundary(tmp_path):
    """A stop-hook record positioned after an unresolved `tool_use` claims the
    turn ended. The structural boundary still answers WORKING and reports the
    disagreement as `corroboration is FALSE` — corroboration is reported, never
    applied. The control shows the same hook record agreeing with a boundary
    that really did end, so `FALSE` above is a disagreement rather than a
    corroboration path that always fails."""
    disagreeing = _session(tmp_path, "hook-disagrees", [_human(), _tool_use(), _stop_hook()])
    agreeing = _session(tmp_path, "hook-agrees", [_human(), _assistant(), _stop_hook()])

    disagreed = _observe(disagreeing)
    agreed = _observe(agreeing)

    assert disagreed.signals.agent_turn_ended is Tri.FALSE
    assert derive_status(disagreed.signals) is Status.WORKING
    assert disagreed.boundary.corroboration is Tri.FALSE

    assert agreed.signals.agent_turn_ended is Tri.TRUE
    assert agreed.boundary.corroboration is Tri.TRUE


def test_stop_hook_from_an_earlier_turn_is_not_evidence_about_this_one():
    """A hook record positioned *before* the boundary belongs to a turn that is
    already over and makes no claim about the current one. Both shapes below
    are correct boundaries, so neither may be reported as a disagreement:
    counting a stale hook's silence as a denial would flag every session that
    ran a second turn after its last hook fired, and would fire on the live
    race where the assistant's final message lands a moment before its hook
    record does. The control keeps a hook after the boundary registering, so
    this is a scoping rule rather than corroboration switched off."""
    stale_then_working = [_human(), _assistant(), _stop_hook(), _human(), _tool_use()]
    stale_then_ended = [_human(), _assistant(), _stop_hook(), _human(), _assistant()]

    working = derive_turn_boundary(stale_then_working)
    ended = derive_turn_boundary(stale_then_ended)

    assert working.ended is Tri.FALSE
    assert working.corroboration is not Tri.FALSE
    assert working.corroboration is Tri.UNKNOWN

    assert ended.ended is Tri.TRUE
    assert ended.corroboration is not Tri.FALSE
    assert ended.corroboration is Tri.UNKNOWN

    # Control: the same records with the hook moved after the boundary still
    # corroborate, so the UNKNOWNs above come from position, not from a
    # corroboration branch that never fires.
    assert derive_turn_boundary([*stale_then_ended, _stop_hook()]).corroboration is Tri.TRUE


def test_boundary_is_identical_without_any_stop_hook_record(tmp_path):
    """The 83% case: the same two transcripts with every stop-hook record
    removed produce the same boundaries, only uncorroborated. A boundary that
    depended on those records would go UNKNOWN for five sessions in six."""
    ended = _session(tmp_path, "no-hook-ended", [_human(), _assistant()])
    working = _session(tmp_path, "no-hook-working", [_human(), _tool_use()])

    assert _observe(ended).signals.agent_turn_ended is Tri.TRUE
    assert _observe(working).signals.agent_turn_ended is Tri.FALSE

    # With no mtime and no hook records there is nothing left to corroborate.
    records = [_human(), _assistant()]
    assert derive_turn_boundary(records).ended is Tri.TRUE
    assert derive_turn_boundary(records).corroboration is Tri.UNKNOWN


def test_file_mtime_corroborates_but_never_contradicts(tmp_path):
    """An agent blocked on a long tool call writes nothing for minutes, so a
    quiet file must not contradict "the agent holds the turn". A three-day-old
    store with an unresolved `tool_use` still reads WORKING and is merely
    uncorroborated; the control, written seconds ago, is corroborated."""
    stale = _session(tmp_path, "stale", [_human(), _tool_use()])
    fresh = _session(tmp_path, "fresh", [_human(), _tool_use()])
    _set_mtime(stale, timedelta(days=3))
    _set_mtime(fresh, timedelta(seconds=5))

    stale_observation = _observe(stale)

    assert stale_observation.signals.agent_turn_ended is Tri.FALSE
    assert derive_status(stale_observation.signals) is Status.WORKING
    assert stale_observation.boundary.corroboration is not Tri.FALSE
    assert stale_observation.boundary.corroboration is Tri.UNKNOWN

    assert _observe(fresh).boundary.corroboration is Tri.TRUE


# --- the other signals in the set --------------------------------------------


def test_latest_tool_outcome_decides_the_error_signal(tmp_path):
    """`unresolved_tool_error` is a claim about the session's latest outcome:
    a trailing error sets it, and a later successful outcome in the same turn
    clears it — asserted through `derive_status`, where rule 3 outranks the
    boundary in both directions."""
    errored = _session(tmp_path, "errored", [_human(), _tool_use(), _tool_result(is_error=True)])
    recovered = _session(
        tmp_path,
        "recovered",
        [
            _human(),
            _tool_use(),
            _tool_result(is_error=True),
            _tool_use(),
            _tool_result(is_error=False),
            _assistant(),
        ],
    )

    assert _observe(errored).signals.unresolved_tool_error is Tri.TRUE
    assert _status(errored) is Status.ERROR

    assert _observe(recovered).signals.unresolved_tool_error is Tri.FALSE
    assert _status(recovered) is Status.AWAITING_HUMAN


def test_a_corrupt_line_early_in_history_does_not_pin_the_session_to_unknown(tmp_path):
    """The signal window is the current turn, so one unparseable line from an
    old turn cannot make a session permanently unreportable. The control puts
    the same corrupt line inside the window, where it does force UNKNOWN — the
    signals were computed over a view with a hole in it."""
    early = _session(
        tmp_path,
        "corrupt-early",
        [_human("first task"), b"{not json\n", _human("second task"), _assistant()],
    )
    late = _session(tmp_path, "corrupt-late", [_human(), _assistant(), b"{not json\n"])

    early_observation = _observe(early)

    assert early_observation.signals.signal_records_parsed is Tri.TRUE
    assert early_observation.signals.agent_turn_ended is Tri.TRUE
    assert derive_status(early_observation.signals) is Status.AWAITING_HUMAN

    late_observation = _observe(late)

    assert late_observation.signals.signal_records_parsed is Tri.FALSE
    assert late_observation.boundary.basis == BASIS_UNDECODABLE_RECORD
    assert derive_status(late_observation.signals) is Status.UNKNOWN


def test_unreadable_store_reports_nothing_about_the_session(tmp_path):
    """A store that could not be read yields `source_readable is FALSE` and
    `UNKNOWN` for every derived signal — a component that cannot read a
    session must not report on it. The readable control proves the same code
    path produces real signals when the file exists."""
    missing = tmp_path / "projects" / "-Users-test-project" / "gone.jsonl"
    present = _session(tmp_path, "present", [_human(), _assistant()])

    observation = _observe(missing)

    assert observation.signals.source_readable is Tri.FALSE
    assert observation.signals.signal_records_parsed is Tri.UNKNOWN
    assert observation.signals.unresolved_tool_error is Tri.UNKNOWN
    assert observation.signals.agent_turn_ended is Tri.UNKNOWN
    assert observation.boundary.basis == BASIS_SOURCE_UNREADABLE
    assert derive_status(observation.signals) is Status.UNKNOWN

    assert _observe(present).signals.source_readable is Tri.TRUE


def test_every_declared_signal_is_produced_as_a_tri(tmp_path):
    """`observe_session` fills every name in `SIGNAL_NAMES` with a real `Tri`,
    so a signal added to `Signals` cannot be left unproduced (and silently
    default to nothing) by this module."""
    path = _session(tmp_path, "complete", [_human(), _assistant()])

    signals = _observe(path).signals

    assert len(SIGNAL_NAMES) == 4
    assert all(isinstance(getattr(signals, name), Tri) for name in SIGNAL_NAMES)


# --- the adapter fix: an injected record does not resolve a tool call ---------


def test_injected_record_keeps_an_unresolved_call_discoverable_past_the_floor(tmp_path):
    """A hook firing after the agent's last tool call must not make the session
    look resolved: `has_unresolved_trailing_tool_use` reads through the
    injected record, so `discover_sessions` still returns the session on the
    always-include rule even though its mtime is three days past the 24h
    recency floor.

    The control is the same fixture with a real `tool_result` in place of the
    injection: that one *is* resolved, and the same call drops it — proving
    the always-include rule is conditioned on the outstanding call rather than
    on the age of the file.
    """
    root = tmp_path / "projects"
    hooked = _session(tmp_path, "hooked", [_human(), _tool_use(), _injected()])
    _set_mtime(hooked, timedelta(days=3))
    adapter = ClaudeCodeAdapter(root=root)

    assert adapter.has_unresolved_trailing_tool_use(hooked) is True

    keys = {ref.session_key for ref in adapter.discover_sessions(now=NOW)}
    assert adapter.session_key_for(hooked) in keys

    # Positive control: resolved by a real tool_result, and out it goes.
    resolved = _session(tmp_path, "resolved-old", [_human(), _tool_use(), _tool_result()])
    _set_mtime(resolved, timedelta(days=3))

    assert adapter.has_unresolved_trailing_tool_use(resolved) is False
    refreshed = {ref.session_key for ref in adapter.discover_sessions(now=NOW)}
    assert adapter.session_key_for(resolved) not in refreshed
    assert adapter.session_key_for(hooked) in refreshed
