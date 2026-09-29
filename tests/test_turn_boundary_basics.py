"""
The core turn boundary: the hook-injected-record inversion, tool_use/tool_result
pairing, and named human-blocking tools.
"""

from palaver.ingest.adapters.claude_code import ClaudeCodeAdapter
from palaver.observer.signals import (
    Status,
    Tri,
    derive_status,
)
from palaver.observer.turn_boundary import (
    BASIS_ASSISTANT_FINAL,
    BASIS_BACKGROUND_TASK_PENDING,
    BASIS_HUMAN_MESSAGE_PENDING,
    BASIS_NO_CONVERSATIONAL_RECORD,
    BASIS_TOOL_RESULT_PENDING,
    BASIS_UNRESOLVED_HUMAN_BLOCKING_TOOL_USE,
    BASIS_UNRESOLVED_TOOL_USE,
)
from tests._turn_boundary_support import (
    _assistant,
    _background_notification,
    _background_tool_result,
    _bookkeeping,
    _human,
    _injected,
    _observe,
    _session,
    _status,
    _tool_result,
    _tool_use,
)

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


def test_active_background_task_keeps_followup_prose_working_until_notification(tmp_path):
    pending = _session(
        tmp_path,
        "background-pending",
        [_human(), _tool_use(), _background_tool_result(), _assistant()],
    )
    finished = _session(
        tmp_path,
        "background-finished",
        [
            _human(),
            _tool_use(),
            _background_tool_result(),
            _assistant(),
            _background_notification(),
        ],
    )

    pending_observation = _observe(pending)
    assert pending_observation.signals.agent_turn_ended is Tri.FALSE
    assert pending_observation.boundary.basis == BASIS_BACKGROUND_TASK_PENDING
    assert _status(finished) is Status.AWAITING_HUMAN


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
