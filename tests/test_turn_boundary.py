"""Tests for the structural turn boundary and `palaver diagnose --coverage`.

Every fixture here is a JSONL file this module writes under pytest's
`tmp_path`, with prose invented for the test. No real session store
(`~/.claude/`, `~/.codex/`, `~/.local/share/opencode/`) is opened, globbed, or
read — INV-3 forbids it, and the CLI is always given an explicit `--sample`
pointing into `tmp_path`.

The module's sharpest tests are the pair that flip a single `isMeta` byte:
a trailing harness-injected `user` record must not be read as a human turn,
and reading it as one inverts the status of exactly the sessions that use
hooks and skills most.
"""

import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

from palaver.cli import main
from palaver.cli.diagnose import collect_coverage
from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import SIGNAL_NAMES, Status, Tri, derive_status
from palaver.observer.turn_boundary import (
    BASIS_ASSISTANT_FINAL,
    BASIS_HUMAN_MESSAGE_PENDING,
    BASIS_NO_CONVERSATIONAL_RECORD,
    BASIS_SOURCE_UNREADABLE,
    BASIS_TOOL_RESULT_PENDING,
    BASIS_UNDECODABLE_RECORD,
    BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE,
    BASIS_UNRESOLVED_TOOL_USE,
    derive_turn_boundary,
    observe_session,
)

#: Fixed reference time; every fixture's mtime is set relative to it, so no
#: assertion in this module depends on when the suite runs.
NOW = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- fixture builders --------------------------------------------------------


def _line(record: dict) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write(path: Path, items: list[dict | bytes]) -> Path:
    """Write a JSONL fixture. A `bytes` item is written verbatim (corrupt lines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(item if isinstance(item, bytes) else _line(item) for item in items))
    return path


def _session(tmp_path: Path, name: str, items: list[dict | bytes]) -> Path:
    """Write one session store into a Claude-Code-shaped sample directory."""
    return _write(tmp_path / "projects" / "-Users-test-project" / f"{name}.jsonl", items)


def _set_mtime(path: Path, age: timedelta, now: datetime = NOW) -> None:
    ts = (now - age).timestamp()
    os.utime(path, (ts, ts))


def _human(text: str = "please run the build") -> dict:
    return {
        "type": "user",
        "sessionId": "session-1",
        "isMeta": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _injected(text: str = "<system-reminder>A hook fired.</system-reminder>") -> dict:
    """A harness-injected `type: "user"` record — the same role, nothing said."""
    record = _human(text)
    record["isMeta"] = True
    return record


def _assistant(text: str = "the build is green") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _tool_use(name: str = "Bash") -> dict:
    return {
        "type": "assistant",
        "sessionId": "session-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "tu-1", "name": name, "input": {}}],
        },
    }


def _tool_result(*, is_error: bool = False, content: str = "ok") -> dict:
    return {
        "type": "user",
        "sessionId": "session-1",
        "isMeta": False,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu-1",
                    "is_error": is_error,
                    "content": content,
                }
            ],
        },
    }


def _stop_hook(subtype: str = "stop_hook_summary") -> dict:
    return {"type": "system", "subtype": subtype, "sessionId": "session-1", "summary": "hook ran"}


def _bookkeeping() -> dict:
    return {"type": "mode", "sessionId": "session-1", "mode": "default"}


def _observe(path: Path):
    return observe_session(path, now=NOW)


def _status(path: Path) -> Status:
    return derive_status(_observe(path).signals)


# --- the inversion: a trailing injected record is not a human turn -----------


def test_trailing_hook_injected_user_record_is_working_not_a_pending_human_turn(tmp_path):
    """A session mid-tool-call whose transcript ends on a hook-injected `user`
    record is still WORKING: the injection is not a turn, and it does not
    resolve the outstanding call.

    The `last_message_bearing_record` assertion is the non-vacuity control —
    it proves the fixture really does end on a `type: "user"` record, so a
    boundary that read role raw would have had to answer from that record.
    `unresolved_tool_error` is asserted directly because rule 3 outranks the
    boundary in `derive_status()`: without it, an ERROR could mask which
    signal produced the result.
    """
    path = _session(tmp_path, "injected-mid-call", [_human(), _tool_use(), _injected()])

    observation = _observe(path)

    assert observation.signals.agent_turn_ended is Tri.FALSE
    assert observation.boundary.basis == BASIS_UNRESOLVED_TOOL_USE
    assert observation.signals.unresolved_tool_error is Tri.FALSE
    assert derive_status(observation.signals) is Status.WORKING
    assert derive_status(observation.signals) is not Status.AWAITING_HUMAN

    # Non-vacuity: the last message-bearing record really is a `user` record.
    last = ClaudeCodeAdapter(root=tmp_path / "projects").last_message_bearing_record(path)
    assert last is not None and last["type"] == "user" and last["isMeta"] is True


def test_injected_record_after_a_final_reply_does_not_invert_status_to_working(tmp_path):
    """The costly inversion, in both directions from one flipped `isMeta` byte.

    Same three records, same roles, same prose: with `isMeta: true` the
    trailing record is harness-injected, the boundary looks through it to the
    assistant's final reply, and the session is AWAITING_HUMAN. With
    `isMeta: false` it is a human message the agent has not answered, and the
    session is WORKING. A boundary reading role raw returns WORKING for both
    — telling the human that a session waiting on them needs nothing.
    """
    injected = _session(tmp_path, "hook-after-reply", [_human(), _assistant(), _injected()])
    typed = _session(tmp_path, "human-after-reply", [_human(), _assistant(), _human("and now?")])

    injected_observation = _observe(injected)
    typed_observation = _observe(typed)

    assert injected_observation.signals.agent_turn_ended is Tri.TRUE
    assert injected_observation.boundary.basis == BASIS_ASSISTANT_FINAL
    assert derive_status(injected_observation.signals) is Status.AWAITING_HUMAN

    # Positive control: the only difference is the channel of the last record.
    assert typed_observation.signals.agent_turn_ended is Tri.FALSE
    assert typed_observation.boundary.basis == BASIS_HUMAN_MESSAGE_PENDING
    assert derive_status(typed_observation.signals) is Status.WORKING


def test_injected_records_alone_leave_the_boundary_unknown(tmp_path):
    """A transcript containing nothing but harness-injected records supports no
    boundary claim at all — `UNKNOWN`, not a guess in either direction. The
    control adds one assistant reply to the same file and gets a real answer,
    so the UNKNOWN above is not an inert fixture."""
    only_injected = _session(
        tmp_path, "only-injected", [_injected(), _injected("<command-name>/x")]
    )
    with_reply = _session(tmp_path, "with-reply", [_injected(), _assistant()])

    assert _observe(only_injected).signals.agent_turn_ended is Tri.UNKNOWN
    assert _observe(only_injected).boundary.basis == BASIS_NO_CONVERSATIONAL_RECORD
    assert _status(only_injected) is Status.UNKNOWN

    assert _observe(with_reply).signals.agent_turn_ended is Tri.TRUE
    assert _status(with_reply) is Status.AWAITING_HUMAN


# --- tool_use / tool_result pairing ------------------------------------------


def test_unresolved_trailing_tool_use_is_working(tmp_path):
    """An outstanding tool call with nothing after it means the agent holds the
    turn. The control resolves that same call and lets the agent reply, which
    flips the boundary — so WORKING here comes from the pairing, not from the
    presence of a `tool_use` block anywhere in the file."""
    unresolved = _session(tmp_path, "unresolved", [_human(), _tool_use()])
    resolved = _session(
        tmp_path,
        "resolved",
        [_human(), _tool_use(), _tool_result(), _assistant()],
    )

    observation = _observe(unresolved)

    assert observation.signals.agent_turn_ended is Tri.FALSE
    assert observation.boundary.basis == BASIS_UNRESOLVED_TOOL_USE
    assert observation.signals.unresolved_tool_error is Tri.FALSE
    assert derive_status(observation.signals) is Status.WORKING

    assert _observe(resolved).signals.agent_turn_ended is Tri.TRUE
    assert _status(resolved) is Status.AWAITING_HUMAN


def test_trailing_tool_result_is_a_tool_outcome_not_a_human_turn(tmp_path):
    """A `tool_result` arrives as a `type: "user"` record carrying no text, so
    it must be recognized structurally: the agent consumed an outcome and
    continues (WORKING, basis `tool_result_pending`), rather than being
    credited to the human channel because no injected prefix matched."""
    path = _session(tmp_path, "tool-result", [_human(), _tool_use(), _tool_result()])

    observation = _observe(path)

    assert observation.boundary.basis == BASIS_TOOL_RESULT_PENDING
    assert observation.boundary.basis != BASIS_HUMAN_MESSAGE_PENDING
    assert observation.signals.agent_turn_ended is Tri.FALSE
    assert derive_status(observation.signals) is Status.WORKING


def test_assistant_final_reply_is_awaiting_human_never_done(tmp_path):
    """An ended turn is AWAITING_HUMAN. Structure proves control came back; it
    proves nothing about the work being finished, which is the brief's single
    named prohibition."""
    path = _session(tmp_path, "final", [_human(), _tool_use(), _tool_result(), _assistant()])

    status = _status(path)

    assert status is Status.AWAITING_HUMAN
    assert status is not Status.DONE


def test_no_message_bearing_record_is_unknown(tmp_path):
    """A file holding only bookkeeping records supports no boundary claim.
    The control writes one conversational record into the same shape and gets
    a determinate answer."""
    bookkeeping = _session(tmp_path, "bookkeeping", [_bookkeeping()])
    conversational = _session(tmp_path, "conversational", [_bookkeeping(), _human()])

    assert _observe(bookkeeping).signals.agent_turn_ended is Tri.UNKNOWN
    assert _observe(bookkeeping).boundary.basis == BASIS_NO_CONVERSATIONAL_RECORD
    assert _status(bookkeeping) is Status.UNKNOWN

    assert _observe(conversational).signals.agent_turn_ended is Tri.FALSE


# --- an unresolved human-blocking tool_use ends the turn, by name only ------


def test_unresolved_human_blocking_tool_use_ends_the_turn_by_name(tmp_path):
    """The tool's name, not merely a `tool_use` block's presence, decides.

    An unresolved `AskUserQuestion` is an agent that has stopped and put a
    prompt in front of its human — the turn already ended even though the
    call itself never got a `tool_result`. The control is the identical
    shape with `Bash` in place of `AskUserQuestion`: it must stay WORKING,
    which is what proves the fix keys on the tool name rather than simply
    inverting the unresolved-`tool_use` rule (that would flip both cases).
    """
    question = _session(tmp_path, "question", [_human(), _tool_use("AskUserQuestion")])
    bash = _session(tmp_path, "bash", [_human(), _tool_use("Bash")])

    question_observation = _observe(question)
    assert question_observation.signals.agent_turn_ended is Tri.TRUE
    assert question_observation.boundary.basis == BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE
    assert derive_status(question_observation.signals) is Status.AWAITING_HUMAN

    # Positive control: the same shape with an ordinary tool stays WORKING.
    bash_observation = _observe(bash)
    assert bash_observation.signals.agent_turn_ended is Tri.FALSE
    assert bash_observation.boundary.basis == BASIS_UNRESOLVED_TOOL_USE
    assert derive_status(bash_observation.signals) is Status.WORKING


def test_resolved_askuserquestion_is_awaiting_human_from_the_ordinary_rule(tmp_path):
    """Over-trigger control: a resolved `AskUserQuestion` must not take the new path.

    A fix that matched the tool name anywhere in the record, rather than only
    on an *unresolved* `tool_use` block, would still land here by accident
    once the question is answered and the agent replies. This fixture answers
    it and lets the agent reply, so it must derive AWAITING_HUMAN through the
    ordinary `assistant_final` basis, not the human-blocking one.
    """
    path = _session(
        tmp_path,
        "answered",
        [_human(), _tool_use("AskUserQuestion"), _tool_result(), _assistant()],
    )

    observation = _observe(path)

    assert observation.signals.agent_turn_ended is Tri.TRUE
    assert observation.boundary.basis == BASIS_ASSISTANT_FINAL
    assert observation.boundary.basis != BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE
    assert derive_status(observation.signals) is Status.AWAITING_HUMAN


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


# --- palaver diagnose --coverage ---------------------------------------------


def _coverage_sample(tmp_path: Path) -> Path:
    """Write a five-session sample with one deliberately undeterminable session.

    The bookkeeping-only session is what keeps the coverage assertions from
    being satisfiable by a command that prints 100% unconditionally.
    """
    _session(tmp_path, "a-final", [_human(), _assistant()])
    _session(tmp_path, "b-mid-call", [_human(), _tool_use()])
    _session(tmp_path, "c-hooked", [_human(), _assistant(), _injected()])
    _session(tmp_path, "d-error", [_human(), _tool_use(), _tool_result(is_error=True)])
    _session(tmp_path, "e-bookkeeping", [_bookkeeping()])
    return tmp_path / "projects"


def _coverage_rows(stdout: str) -> dict[str, tuple[str, str]]:
    """Parse the report's per-signal rows into {name: (counted, percentage)}."""
    rows = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in SIGNAL_NAMES:
            rows[parts[0]] = (parts[1], parts[2])
    return rows


def test_coverage_reports_a_percentage_for_every_signal(tmp_path, capsys):
    """The report emits one row per entry in `SIGNAL_NAMES` with the measured
    percentage, and the percentages are the real counts: four of the five
    sample sessions support a turn boundary, so `agent_turn_ended` is 80.0% —
    not the 100% a stub would print."""
    sample = _coverage_sample(tmp_path)

    exit_code = main(["diagnose", "--coverage", "--sample", str(sample)])
    stdout = capsys.readouterr().out
    rows = _coverage_rows(stdout)

    assert exit_code == 0
    assert set(rows) == set(SIGNAL_NAMES)
    assert len(rows) == len(SIGNAL_NAMES) == 4
    assert rows["source_readable"] == ("5/5", "100.0%")
    assert rows["signal_records_parsed"] == ("5/5", "100.0%")
    assert rows["unresolved_tool_error"] == ("5/5", "100.0%")
    assert rows["agent_turn_ended"] == ("4/5", "80.0%")
    assert "status: WORKING 1, AWAITING_HUMAN 2, ERROR 1, UNKNOWN 1" in stdout
    assert "coverage counts sessions a signal was determinable for" in stdout


def test_coverage_counts_match_an_independent_reading_of_the_same_sample(tmp_path):
    """The reported counts are the same ones `observe_session` produces
    session by session — computed here independently of the command, so a
    report that hardcoded its numbers fails."""
    sample = _coverage_sample(tmp_path)

    report = collect_coverage(sample, now=NOW)
    expected = dict.fromkeys(SIGNAL_NAMES, 0)
    paths = sorted(sample.glob("*/*.jsonl"))
    for path in paths:
        signals = observe_session(path, now=NOW).signals
        for name in SIGNAL_NAMES:
            if getattr(signals, name) is not Tri.UNKNOWN:
                expected[name] += 1

    assert len(paths) == 5
    assert report.sessions == 5
    assert report.determinable == expected
    assert report.percentage("agent_turn_ended") == 80.0


def test_coverage_over_an_empty_sample_is_not_reported_as_success(tmp_path, capsys):
    """A sample with no sessions exits non-zero instead of printing a vacuous
    100%. The populated control exits 0, so this is an empty-sample rule and
    not a command that always fails."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert main(["diagnose", "--coverage", "--sample", str(empty)]) == 1
    assert "no sessions found" in capsys.readouterr().err

    assert main(["diagnose", "--coverage", "--sample", str(_coverage_sample(tmp_path))]) == 0


def test_console_script_runs_the_coverage_report_with_progress_on_stderr(tmp_path):
    """End-to-end through the installed `palaver` console script: it exits 0,
    the report goes to stdout, and per-session progress goes to stderr (INV-1
    — a scan over many stores is never a silent wait, and stdout stays the
    result channel so the report can be piped)."""
    sample = _coverage_sample(tmp_path)
    script = Path(sys.executable).parent / "palaver"

    assert script.exists(), f"console script not installed at {script}"

    result = subprocess.run(
        [str(script), "diagnose", "--coverage", "--sample", str(sample)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert _coverage_rows(result.stdout)["agent_turn_ended"] == ("4/5", "80.0%")
    assert "observing 5/5" in result.stderr
    assert "observing" not in result.stdout


def test_console_script_help_exits_zero_and_lists_diagnose(tmp_path):
    """`palaver --help` works, and the subcommand table is the CLI's extension
    point task 1.9 adds `status` and `inspect` to."""
    script = Path(sys.executable).parent / "palaver"

    result = subprocess.run([str(script), "--help"], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0
    assert "diagnose" in result.stdout


def test_pyproject_declares_the_console_script_at_an_importable_target():
    """The `palaver` script is declared in `pyproject.toml` and its target
    really resolves — a declaration pointing at a missing callable fails at
    install time for the user, not here."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    target = pyproject["project"]["scripts"]["palaver"]
    module_name, _, attribute = target.partition(":")

    assert target == "palaver.cli:main"
    assert callable(getattr(import_module(module_name), attribute))
