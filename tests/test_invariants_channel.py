"""
INV-8: an isMeta record classifies to the injected channel, and an ordinary user record
to the human one.
"""

from __future__ import annotations

import pytest

from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED, classify_channel

# =============================================================================
# INV-8 — human-typed input and harness-injected content classified distinctly
# =============================================================================


@pytest.mark.inv8
def test_isMeta_record_classified_as_injected_channel():
    """An `isMeta: true` record classifies to the injected channel, not the human one."""
    record = {
        "type": "user",
        "isMeta": True,
        "message": {
            "role": "user",
            "content": "<system-reminder>fixture text invented for this test</system-reminder>",
        },
    }
    assert classify_channel(record) == CHANNEL_INJECTED


@pytest.mark.inv8
def test_user_authored_record_classified_as_human_channel():
    """Positive control: an ordinary, non-meta user record classifies to the human channel.

    Without this, `classify_channel` returning `CHANNEL_INJECTED`
    unconditionally would also satisfy the assertion above.
    """
    record = {
        "type": "user",
        "isMeta": False,
        "message": {"role": "user", "content": "what's the status of the deploy?"},
    }
    assert classify_channel(record) == CHANNEL_HUMAN
