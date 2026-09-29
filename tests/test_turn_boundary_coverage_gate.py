"""
Task 7.3: the per-source coverage gate, and the event vocabulary/three-source report
(Claude Code, Codex, OpenCode) it feeds.
"""

import json
import sqlite3
from pathlib import Path

from palaver.cli import main
from palaver.cli.diagnose import (
    collect_all_coverage,
    collect_codex_coverage,
    collect_coverage,
    collect_opencode_coverage,
)
from palaver.ingest.adapters import codex, opencode
from palaver.ingest.adapters.base import Event
from palaver.observer import turn_boundary
from palaver.observer.signals import (
    SIGNAL_NAMES,
    Signals,
    Status,
    Tri,
    derive_status,
    derive_status_for_source,
    derive_status_with_provenance,
    under_covered,
)
from palaver.observer.turn_boundary import (
    BASIS_EVENT_MESSAGE_PENDING,
    BASIS_EVENT_TURN_BOUNDARY,
    BASIS_NO_CONVERSATIONAL_RECORD,
    derive_signals_from_events,
)
from tests._turn_boundary_support import NOW, REPO_ROOT, _coverage_sample, _write

# --- task 7.3: the per-source coverage gate ----------------------------------
#
# Every test below is named `..._coverage_gate_...` so the plan's quick check
# (`-k coverage_gate`) selects the whole group. A selector that matches zero
# tests reads as coverage and is worse than no quick check, so
# `test_the_coverage_gate_quick_check_selects_these_tests` asserts the count.


def _signals(
    *,
    readable: Tri = Tri.TRUE,
    parsed: Tri = Tri.TRUE,
    error: Tri = Tri.FALSE,
    ended: Tri = Tri.TRUE,
) -> Signals:
    """Build a `Signals` set; the defaults derive `AWAITING_HUMAN`."""
    return Signals(
        source_readable=readable,
        signal_records_parsed=parsed,
        unresolved_tool_error=error,
        agent_turn_ended=ended,
    )


def _coverage(**overrides: float) -> dict[str, float]:
    """Full coverage for every signal, minus whatever the caller thins out."""
    return {**dict.fromkeys(SIGNAL_NAMES, 100.0), **overrides}


def test_a_source_with_ten_percent_boundary_coverage_gets_no_status_coverage_gate():
    """A source whose turn boundary is determinable for a tenth of its sessions
    has not been shown to fit that source's format, so a status derived from
    the boundary is withdrawn to `UNKNOWN` — even though this session's own
    boundary signal was determinable and the rule list reached a confident
    answer from it."""
    signals = _signals()

    assert derive_status(signals) is Status.AWAITING_HUMAN
    assert derive_status_for_source(signals, _coverage(agent_turn_ended=10.0)) is Status.UNKNOWN


def test_the_same_fixture_above_the_threshold_keeps_its_status_coverage_gate():
    """The positive control for the test above, on the identical signal set: at
    90% boundary coverage the gate withdraws nothing and the status is the one
    the rules derived. Without this, "returns UNKNOWN" would also pass against
    a gate that returned `UNKNOWN` unconditionally."""
    signals = _signals()

    assert derive_status_for_source(signals, _coverage(agent_turn_ended=90.0)) is (
        Status.AWAITING_HUMAN
    )
    assert derive_status_for_source(signals, _coverage()) is Status.AWAITING_HUMAN


def test_the_threshold_is_a_parameter_not_a_constant_coverage_gate():
    """Both sides of one coverage number, driven by the threshold alone: 40%
    coverage passes a 30% bar and fails a 50% one. The gate is therefore a
    comparison against a caller-supplied value, not a hardcoded verdict about
    a particular percentage."""
    signals = _signals()
    coverage = _coverage(agent_turn_ended=40.0)

    assert derive_status_for_source(signals, coverage, threshold=30.0) is Status.AWAITING_HUMAN
    assert derive_status_for_source(signals, coverage, threshold=50.0) is Status.UNKNOWN


def test_only_the_signals_a_status_consulted_can_withdraw_it_coverage_gate():
    """`ERROR` is decided at rule 3, before the turn boundary is ever read, so
    a source that cannot read boundaries at all still reports `ERROR` — while
    the same source's `AWAITING_HUMAN` is withdrawn. The gate follows the rule
    list's actual reads rather than blanking every status whenever any signal
    is thin, which is what makes it a refinement and not a mute button."""
    coverage = _coverage(agent_turn_ended=0.0)

    assert derive_status_for_source(_signals(error=Tri.TRUE), coverage) is Status.ERROR
    assert derive_status_for_source(_signals(), coverage) is Status.UNKNOWN


def test_an_unmeasured_signal_counts_as_uncovered_coverage_gate():
    """A signal missing from the coverage mapping is 0%, not "assume fine" —
    the same rule that makes `percentage()` return 0.0 over an empty sample
    rather than 100% by vacuity. Unmeasured and badly covered are both
    "nothing here supports a status"."""
    signals = _signals()

    assert derive_status_for_source(signals, {}) is Status.UNKNOWN
    assert under_covered(SIGNAL_NAMES, {}) == SIGNAL_NAMES


def test_the_gate_can_only_ever_weaken_a_status_coverage_gate():
    """Across every reachable status and every one-signal-thin coverage map,
    the gated answer is either the ungated one or `UNKNOWN`. A coverage
    number is a property of a whole sample, so it may withdraw a status but
    must never manufacture a different confident one."""
    cases = [
        _signals(readable=Tri.FALSE),
        _signals(parsed=Tri.FALSE),
        _signals(error=Tri.TRUE),
        _signals(ended=Tri.FALSE),
        _signals(ended=Tri.UNKNOWN),
        _signals(),
    ]
    ungated = {derive_status(signals) for signals in cases}

    for signals in cases:
        for thin in SIGNAL_NAMES:
            gated = derive_status_for_source(signals, _coverage(**{thin: 0.0}))
            assert gated in (derive_status(signals), Status.UNKNOWN)

    assert ungated == {Status.UNKNOWN, Status.ERROR, Status.WORKING, Status.AWAITING_HUMAN}


def test_the_consulted_signals_are_the_ones_the_rules_read_coverage_gate():
    """`StatusDerivation.consulted` is the rule list's own reads, in order —
    the thing the gate is computed from. A hand-maintained status-to-signals
    table would be a second copy of the rule order and would drift from it."""
    assert derive_status_with_provenance(_signals(readable=Tri.FALSE)).consulted == (
        "source_readable",
    )
    assert derive_status_with_provenance(_signals(parsed=Tri.FALSE)).consulted == (
        "source_readable",
        "signal_records_parsed",
    )
    assert derive_status_with_provenance(_signals(error=Tri.TRUE)).consulted == (
        "source_readable",
        "signal_records_parsed",
        "unresolved_tool_error",
    )
    assert derive_status_with_provenance(_signals()).consulted == SIGNAL_NAMES


# --- task 7.3: three sources, one report -------------------------------------


def _event(kind: str) -> Event:
    """One canonical event of `kind`; the derivation reads nothing else."""
    return Event(session_key="fixture-session", kind=kind, payload={})


def _codex_rollout(root: Path, name: str, records: list[dict]) -> Path:
    """Write one Codex rollout into a date-partitioned sample root."""
    return _write(root / "2026" / "08" / "14" / f"rollout-{name}.jsonl", records)


def _codex_meta(session: str) -> dict:
    return {
        "type": "session_meta",
        "payload": {"id": session, "session_id": session, "cwd": "/tmp/fixture-codex-project"},
    }


def _codex_message(role: str, text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _codex_boundary() -> dict:
    return {"type": "event_msg", "payload": {"type": "task_complete"}}


def _codex_sample(tmp_path: Path) -> Path:
    """A three-rollout Codex sample: one finished, one mid-turn, one meta-only."""
    root = tmp_path / "codex"
    _codex_rollout(
        root,
        "finished",
        [
            _codex_meta("fixture-a"),
            _codex_message("user", "check the staging deploy"),
            _codex_message("assistant", "staging is green"),
            _codex_boundary(),
        ],
    )
    _codex_rollout(
        root,
        "mid-turn",
        [_codex_meta("fixture-b"), _codex_message("user", "rebuild the index")],
    )
    _codex_rollout(root, "meta-only", [_codex_meta("fixture-c")])
    return root


def _opencode_sample(tmp_path: Path, records_by_file: dict[str, list[dict]] | None = None) -> Path:
    """Build an OpenCode-shaped SQLite store from the task 7.0 fixture corpus.

    Writes through a plain `sqlite3.connect` — seeding a fixture is the one
    thing that needs write access. Every read the command performs afterwards
    goes through `opencode.open_store_readonly`, which installs both INV-3
    defenses.
    """
    if records_by_file is None:
        records_by_file = {
            path.name: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in sorted((REPO_ROOT / "tests" / "fixtures" / "opencode").glob("*.jsonl"))
        }

    db_path = tmp_path / "opencode.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT);
            """
        )
        sessions = set()
        for records in records_by_file.values():
            for record in records:
                if record["type"] == "opencode_message":
                    sessions.add(record["session_id"])
                    conn.execute(
                        "INSERT INTO message VALUES (?, ?, ?)",
                        (record["id"], record["session_id"], json.dumps(record["data"])),
                    )
                elif record["type"] == "opencode_part":
                    conn.execute(
                        "INSERT INTO part VALUES (?, ?, ?)",
                        (record["id"], record["message_id"], json.dumps(record["data"])),
                    )
        for session_id in sorted(sessions):
            conn.execute("INSERT INTO session VALUES (?, ?)", (session_id, "/tmp/fixture-oc"))
        conn.commit()
    finally:
        conn.close()
    return db_path


def _coverage_blocks(stdout: str) -> dict[str, dict[str, tuple[str, str]]]:
    """Parse the multi-source report into {source: {signal: (counted, pct)}}."""
    blocks: dict[str, dict[str, tuple[str, str]]] = {}
    current: dict[str, tuple[str, str]] | None = None
    for line in stdout.splitlines():
        if line.startswith("source: "):
            current = blocks.setdefault(line.removeprefix("source: ").strip(), {})
            continue
        parts = line.split()
        if current is not None and len(parts) == 3 and parts[0] in SIGNAL_NAMES:
            current[parts[0]] = (parts[1], parts[2])
    return blocks


def test_the_event_vocabulary_is_shared_by_both_adapters_coverage_gate():
    """`derive_signals_from_events` names the kinds itself rather than importing
    either adapter, so this is what keeps the three spellings one vocabulary. A
    rename in Codex or OpenCode fails here instead of silently making a branch
    of the derivation unreachable — which would show up as that source's
    coverage quietly collapsing to zero."""
    assert turn_boundary.KIND_MESSAGE == codex.KIND_MESSAGE == opencode.KIND_MESSAGE
    assert (
        turn_boundary.KIND_TURN_BOUNDARY == codex.KIND_TURN_BOUNDARY == opencode.KIND_TURN_BOUNDARY
    )
    assert turn_boundary.KIND_ERROR == codex.KIND_ERROR == opencode.KIND_ERROR


def test_event_derivation_reads_the_last_message_bearing_kind_coverage_gate():
    """The boundary rule over an `Event` stream: a trailing `turn_boundary` ends
    the turn, a trailing `message` does not, and a stream with neither settles
    nothing. The third case is what stops this derivation from scoring 100%
    coverage on a store it cannot actually read."""
    ended = derive_signals_from_events([_event("message"), _event("turn_boundary")])
    holding = derive_signals_from_events([_event("turn_boundary"), _event("message")])
    silent = derive_signals_from_events([_event("session_meta")])

    assert ended.signals.agent_turn_ended is Tri.TRUE
    assert ended.boundary.basis == BASIS_EVENT_TURN_BOUNDARY
    assert holding.signals.agent_turn_ended is Tri.FALSE
    assert holding.boundary.basis == BASIS_EVENT_MESSAGE_PENDING
    assert silent.signals.agent_turn_ended is Tri.UNKNOWN
    assert silent.boundary.basis == BASIS_NO_CONVERSATIONAL_RECORD


def test_a_closed_turn_resolves_the_errors_inside_it_coverage_gate():
    """An `error` after the last `turn_boundary` is unresolved; one before it is
    not. Same reading `CodexAdapter.has_unresolved_trailing_tool_use` applies to
    pending calls — a turn that closed, closed over whatever went wrong in it.
    An error-free stream is `FALSE` (observed absence); an empty one is
    `UNKNOWN` (nothing was observed at all)."""
    unresolved = derive_signals_from_events([_event("turn_boundary"), _event("error")])
    resolved = derive_signals_from_events([_event("error"), _event("turn_boundary")])
    clean = derive_signals_from_events([_event("message")])
    empty = derive_signals_from_events([])

    assert unresolved.signals.unresolved_tool_error is Tri.TRUE
    assert resolved.signals.unresolved_tool_error is Tri.FALSE
    assert clean.signals.unresolved_tool_error is Tri.FALSE
    assert empty.signals.unresolved_tool_error is Tri.UNKNOWN


def test_event_derivation_will_not_claim_records_parsed_by_itself_coverage_gate():
    """`parsed` defaults to `UNKNOWN`, which rule 2 turns into `UNKNOWN`. Both
    event-sourced adapters *drop* an undecodable record rather than marking it,
    so a hole is invisible downstream and a derivation that assumed `TRUE`
    would be asserting something nobody checked."""
    events = [_event("message"), _event("turn_boundary")]

    assert derive_signals_from_events(events).signals.signal_records_parsed is Tri.UNKNOWN
    assert derive_status(derive_signals_from_events(events).signals) is Status.UNKNOWN
    assert (
        derive_status(derive_signals_from_events(events, parsed=Tri.TRUE).signals)
        is Status.AWAITING_HUMAN
    )


def test_coverage_reports_every_signal_for_all_three_sources_coverage_gate(tmp_path, capsys):
    """The plan's second done-when: one per-signal percentage per source, for
    all three. The counts differ per source, so this cannot be satisfied by a
    command that prints one block three times."""
    exit_code = main(
        [
            "diagnose",
            "--coverage",
            "--sample",
            str(_coverage_sample(tmp_path)),
            "--codex-sample",
            str(_codex_sample(tmp_path)),
            "--opencode-db",
            str(_opencode_sample(tmp_path)),
        ]
    )
    stdout = capsys.readouterr().out
    blocks = _coverage_blocks(stdout)

    assert exit_code == 0
    assert set(blocks) == {"claude-code", "codex", "opencode"}
    for source, rows in blocks.items():
        assert set(rows) == set(SIGNAL_NAMES), source
        for name in SIGNAL_NAMES:
            assert rows[name][1].endswith("%"), (source, name)
    assert blocks["claude-code"]["agent_turn_ended"] == ("4/5", "80.0%")
    assert blocks["codex"]["agent_turn_ended"] == ("2/3", "66.7%")
    assert blocks["opencode"]["agent_turn_ended"][1] == "100.0%"


def test_naming_one_sample_never_reaches_another_sources_store_coverage_gate(tmp_path):
    """Naming a sample scopes the run to that source. This is what lets every
    test in this module point at `tmp_path` without the command falling back to
    `~/.codex/sessions` or `~/.local/share/opencode/opencode.db` for the sources
    the test did not name — INV-2 and INV-3, not a convenience."""
    claude_only = collect_all_coverage(sample_root=_coverage_sample(tmp_path))
    codex_only = collect_codex_coverage(_codex_sample(tmp_path))

    assert [report.source for report in claude_only] == ["claude-code"]
    assert codex_only.source == "codex"
    assert codex_only.sessions == 3


def test_a_thin_source_reports_unknown_end_to_end_coverage_gate(tmp_path, capsys):
    """The gate firing through the real command, not through a synthetic
    coverage mapping: a Codex sample of ten rollouts where one closed a turn is
    10% boundary coverage, so every status in that block is withdrawn to
    `UNKNOWN` and the report says which signal did it."""
    root = tmp_path / "thin"
    _codex_rollout(
        root,
        "closed",
        [_codex_meta("thin-0"), _codex_message("assistant", "done"), _codex_boundary()],
    )
    for index in range(1, 10):
        _codex_rollout(root, f"silent-{index}", [_codex_meta(f"thin-{index}")])

    assert main(["diagnose", "--coverage", "--codex-sample", str(root)]) == 0
    stdout = capsys.readouterr().out
    blocks = _coverage_blocks(stdout)

    assert blocks["codex"]["agent_turn_ended"] == ("1/10", "10.0%")
    assert "gate: agent_turn_ended below 50.0%" in stdout
    assert "status: UNKNOWN 10" in stdout
    assert "AWAITING_HUMAN" not in stdout


def test_raising_the_threshold_withdraws_a_status_the_default_kept_coverage_gate(tmp_path):
    """The end-to-end positive control for the test above, on the same sample:
    at the default threshold the Claude Code block keeps its statuses, and only
    a threshold above its 80% boundary coverage withdraws them. So the gate line
    tracks the measurement rather than always firing.

    The one session that survives the raised threshold is the `ERROR` one, and
    that is the consulted-signals rule showing up in a real sweep rather than
    in a synthetic coverage mapping: rule 3 settled that session before the
    boundary was ever read, so a thin boundary signal has nothing to say about
    it."""
    sample = _coverage_sample(tmp_path)

    kept = collect_coverage(sample, now=NOW)
    withdrawn = collect_coverage(sample, now=NOW, threshold=90.0)

    assert kept.under_covered_signals() == ()
    assert kept.statuses == kept.ungated_statuses
    assert withdrawn.under_covered_signals() == ("agent_turn_ended",)
    assert withdrawn.statuses[Status.UNKNOWN] == 4
    assert withdrawn.statuses[Status.ERROR] == 1
    assert withdrawn.ungated_statuses[Status.UNKNOWN] == 1


def test_the_report_feeds_the_gate_it_describes_coverage_gate(tmp_path):
    """`CoverageReport.as_coverage()` is the mapping `derive_status_for_source`
    takes, so the number printed in the report is the number the gate applies —
    not a parallel calculation that could drift from it."""
    report = collect_coverage(_coverage_sample(tmp_path), now=NOW)

    assert report.as_coverage() == {
        "source_readable": 100.0,
        "signal_records_parsed": 100.0,
        "unresolved_tool_error": 100.0,
        "agent_turn_ended": 80.0,
    }
    assert (
        derive_status_for_source(_signals(), report.as_coverage(), threshold=90.0) is Status.UNKNOWN
    )


def test_the_coverage_gate_quick_check_selects_these_tests():
    """The plan's quick check is `-k coverage_gate`. A selector that silently
    matches nothing reads as coverage, so the count is asserted here rather
    than trusted — the same failure task 3.6 hit with `-k refine`."""
    selected = [
        name
        for name in globals()
        if name.startswith("test_")
        and "coverage_gate" in name
        and name != "test_the_coverage_gate_quick_check_selects_these_tests"
    ]

    assert len(selected) >= 14


def test_the_limit_bounds_every_source_coverage_gate(tmp_path):
    """`--limit` is per source, not a global budget spent on whichever source
    ran first. OpenCode's slice comes off the newest end (`fetch_sessions` is
    descending), because sweeping the oldest sessions of a long-lived store
    would measure a format the adapter has already moved past."""
    db_path = _opencode_sample(tmp_path)

    all_sessions = collect_opencode_coverage(db_path)
    limited = collect_opencode_coverage(db_path, limit=1)

    assert all_sessions.sessions >= 3
    assert limited.sessions == 1
    assert collect_codex_coverage(_codex_sample(tmp_path), limit=2).sessions == 2
