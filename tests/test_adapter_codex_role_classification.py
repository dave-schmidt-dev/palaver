"""
codex_role_class: developer/user/harness-injected channel classification, and
message_text extraction.
"""

import pytest

from palaver.ingest.adapters.codex import (
    CHANNEL_HUMAN,
    CHANNEL_INJECTED,
    INJECTED_TEXT_PREFIXES,
    codex_role_class,
    message_role,
    message_text,
    strip_codex_image_attachment_markers,
)
from tests._adapter_codex_support import _event, _function_call, _message, _session_meta

# --- codex_role_class: channel classification (INV-8) -----------------------


@pytest.mark.inv8
def test_developer_role_record_is_the_harness_channel():
    """Done-when: a `role == "developer"` record returns the harness channel.

    Codex has no `isMeta`, and `developer` is the one role research §2 found
    reliably harness across 817 sampled records. It is the strongest
    structural signal the source offers.
    """
    record = _message("developer", "you are operating inside a sandboxed fixture container")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_bare_user_prose_is_the_human_channel():
    """Positive control for every harness assertion in this file.

    Without this, `codex_role_class` could be `return CHANNEL_INJECTED` and
    the developer test, the prefix tests, and the role-less tests would all
    still pass.
    """
    record = _message("user", "check the staging deploy status")
    assert codex_role_class(record) == CHANNEL_HUMAN


@pytest.mark.inv8
@pytest.mark.parametrize("prefix", INJECTED_TEXT_PREFIXES)
def test_every_injected_prefix_is_classified_harness(prefix):
    """Each entry in the prefix table actually fires.

    Parametrized over the table itself rather than a hand-copied list, so a
    prefix added to the table without being reachable — a typo, a stray
    leading space — fails here instead of silently classifying real harness
    content as human.
    """
    record = _message("user", f"{prefix} trailing fixture text")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_leading_whitespace_cannot_launder_injected_content():
    """A harness block behind a newline is still harness.

    `str.startswith` on the raw text would classify this as human, which is
    the exact INV-8 failure: injected content wearing the human channel, and
    therefore quotable as a tier-1 instruction.
    """
    record = _message("user", "\n\n  <environment_context>fixture sandbox</environment_context>")
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
@pytest.mark.parametrize(
    "record",
    [
        _session_meta(),
        _event("task_complete", last_agent_message=None),
        _event("context_compacted"),
        {"type": "compacted", "payload": {"replacement_history": []}},
        _function_call(),
        {"type": "some_future_record_type", "payload": {}},
        {},
    ],
    ids=[
        "session_meta",
        "task_complete",
        "context_compacted",
        "compacted",
        "function_call",
        "unknown_type",
        "empty",
    ],
)
def test_records_without_a_message_role_are_never_human(record):
    """Rule 1 is structural, and the narrowed measurement denominator rests on it.

    `measure_role_class` counts only role-bearing records toward the
    200-record threshold, on the grounds that a role-less record cannot be
    misclassified. That is only sound if it is actually impossible, so it is
    asserted rather than assumed.
    """
    assert message_role(record) is None
    assert codex_role_class(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_assistant_output_is_not_the_human_channel():
    record = _message("assistant", "the staging deploy is healthy", block_type="output_text")
    assert message_role(record) == "assistant"
    assert codex_role_class(record) == CHANNEL_INJECTED


def test_message_text_concatenates_only_recognized_text_blocks():
    record = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_image", "image_url": "ignored"},
                {"type": "input_text", "text": " second"},
            ],
        },
    }
    assert message_text(record) == "first second"


def test_message_text_of_a_non_message_record_is_empty():
    assert message_text(_event("task_complete", last_agent_message=None)) == ""
    assert message_text(_function_call()) == ""


def test_strip_codex_image_attachment_marker_leaves_only_genuine_prose():
    marker = '<image name=[Image #1] path="/var/folders/fixture/image.png">'
    assert strip_codex_image_attachment_markers(f"inspect this {marker} carefully") == (
        "inspect this  carefully"
    )
    # The double-quoted image-number shape is intentional. Do not turn an
    # arbitrary user-authored XML-looking string into transport metadata.
    similar = "<image name=[Image #one] path='/tmp/not-a-codex-marker.png'>"
    assert strip_codex_image_attachment_markers(similar) == similar
