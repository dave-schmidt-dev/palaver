"""
Codex and OpenCode shape-table negative suites, their pairwise disjointness pin, and the
CODEX_RECORD_SHAPES/OPENCODE_RECORD_SHAPES patch pin.
"""

from palaver.cli import fixture_lint
from palaver.cli.fixture_lint import (
    RULE_BAD_IDENTIFIER,
    RULE_UNALLOWLISTED_TEXT,
    RULE_UNKNOWN_SUBTYPE,
)
from tests._fixture_lint_support import FIXTURES, _corpus, _lint


def _valid_codex_session_meta(**overrides) -> dict:
    """A Codex `session_meta` record the allowlist accepts."""
    record = {
        "type": "session_meta",
        "payload": {
            "id": "fixture-poison-control",
            "session_id": "fixture-poison-control",
            "cwd": "/tmp/fixture-poison-control",
        },
    }
    record.update(overrides)
    return record


def _valid_codex_response_item(**overrides) -> dict:
    """A Codex `response_item` record the allowlist accepts."""
    record = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "check the staging deploy status"}],
        },
    }
    record.update(overrides)
    return record


def _valid_opencode_message(**overrides) -> dict:
    """An OpenCode `opencode_message` record the allowlist accepts."""
    record = {
        "type": "opencode_message",
        "id": "fixture-poison-control",
        "session_id": "fixture-poison-control",
        "data": {"role": "user"},
    }
    record.update(overrides)
    return record


def _valid_opencode_part(**overrides) -> dict:
    """An OpenCode `opencode_part` record the allowlist accepts."""
    record = {
        "type": "opencode_part",
        "id": "fixture-poison-control-part",
        "message_id": "fixture-poison-control",
        "session_id": "fixture-poison-control",
        "data": {"type": "text", "text": "restart the worker queue"},
    }
    record.update(overrides)
    return record


# --- the linter: Codex and OpenCode shape tables ------------------------


def test_codex_source_unclassified_record_fails(tmp_path, capsys):
    """Catches an `event_msg.payload.type` nobody has classified.

    Mirrors `test_unclassified_record_fails` for Claude Code's `system.
    subtype`: the envelope, and everything but the discriminator, is
    allowlisted, so `RULE_UNKNOWN_SUBTYPE` is the only rule that can fire.
    """
    poisoned = {"type": "event_msg", "payload": {"type": "codex_unclassified_event"}}
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNKNOWN_SUBTYPE in capsys.readouterr().out

    # Positive control: same envelope, an event type the corpus classifies.
    accepted = {"type": "event_msg", "payload": {"type": "context_compacted"}}
    assert _lint(_corpus(tmp_path, [accepted])) == 0


def test_codex_source_unallowlisted_prose_fails(tmp_path, capsys):
    """Catches real prose smuggled into a Codex `response_item` text block.

    Done-when (task 7.0): "A test feeding a Codex rollout record with
    unrecognized free text asserts fixture-lint exits non-zero." Shape, keys,
    and role are all allowlisted; only the sentence is unreviewed.
    """
    poisoned = _valid_codex_response_item(
        payload={
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "the northwind reconciliation job is still failing on row 40191",
                }
            ],
        }
    )
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNALLOWLISTED_TEXT in capsys.readouterr().out

    # Positive control: same shape, phrasebook sentence.
    assert _lint(_corpus(tmp_path, [_valid_codex_response_item()])) == 0


def test_codex_source_real_shaped_session_id_fails(tmp_path, capsys):
    """Catches a `session_meta` record keeping a real Codex UUID.

    A real Codex file's `id` matches its rollout filename UUID. Requiring
    `fixture-*` makes provenance structural, the same rule a Claude Code
    `sessionId` copy-paste dies on (`test_a_real_shaped_session_id_fails`).
    """
    poisoned = _valid_codex_session_meta(
        payload={
            "id": "9f1c2d34-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
            "session_id": "fixture-poison-control",
            "cwd": "/tmp/fixture-poison-control",
        }
    )
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_BAD_IDENTIFIER in capsys.readouterr().out

    # Positive control: identical record, fixture-shaped id.
    assert _lint(_corpus(tmp_path, [_valid_codex_session_meta()])) == 0


def test_opencode_source_unclassified_record_fails(tmp_path, capsys):
    """Catches a `part.data.type` nobody has classified."""
    poisoned = _valid_opencode_part(data={"type": "opencode_unclassified_part"})
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNKNOWN_SUBTYPE in capsys.readouterr().out

    # Positive control: same record, a part type the corpus classifies.
    assert _lint(_corpus(tmp_path, [_valid_opencode_part()])) == 0


def test_opencode_source_part_with_real_prose_fails(tmp_path, capsys):
    """Catches real prose smuggled into an OpenCode `part` row's text.

    Done-when (task 7.0): "A test feeding an OpenCode part row carrying real
    prose asserts fixture-lint exits non-zero."
    """
    poisoned = _valid_opencode_part(
        data={
            "type": "text",
            "text": "the northwind reconciliation job is still failing on row 40191",
        }
    )
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_UNALLOWLISTED_TEXT in capsys.readouterr().out

    # Positive control: same shape, phrasebook sentence.
    assert _lint(_corpus(tmp_path, [_valid_opencode_part()])) == 0


def test_opencode_source_real_shaped_identifier_fails(tmp_path, capsys):
    """Catches an OpenCode row keeping a real KSUID-shaped id.

    Real `message`/`part` ids are KSUID-style and lexicographically monotonic
    (`docs/research.md` §3); `fixture-*` makes provenance structural here too.
    """
    poisoned = _valid_opencode_message(id="01H8XGJTF3ZQVN9K2M5R7S8T4W")
    assert _lint(_corpus(tmp_path, [poisoned])) == 1
    assert RULE_BAD_IDENTIFIER in capsys.readouterr().out

    # Positive control: identical record, fixture-shaped id.
    assert _lint(_corpus(tmp_path, [_valid_opencode_message()])) == 0


def test_source_shape_tables_are_pairwise_disjoint():
    """Catches a `type` value silently routing to the wrong source's shape.

    `classify_record` dispatches by trying `RECORD_SHAPES`, then
    `CODEX_RECORD_SHAPES`, then `OPENCODE_RECORD_SHAPES` for the record's
    `type` — safe only because no two tables define the same key. A fourth
    source (or a careless rename) could break that silently; this pins it as
    an assertion rather than an assumption.
    """
    claude_code = set(fixture_lint.RECORD_SHAPES)
    codex = set(fixture_lint.CODEX_RECORD_SHAPES)
    opencode = set(fixture_lint.OPENCODE_RECORD_SHAPES)
    assert claude_code & codex == set()
    assert claude_code & opencode == set()
    assert codex & opencode == set()


def test_codex_source_and_opencode_source_corpora_require_their_shape_tables(monkeypatch):
    """Catches a shape table nothing actually depends on.

    The anti-vacuity check specified for this task: delete a source's shape
    table and re-run. Emptying `CODEX_RECORD_SHAPES` must make the committed
    Codex corpus fail to classify, and likewise for `OPENCODE_RECORD_SHAPES` —
    proving each corpus's acceptance rests on its own table, not on generic
    path/argument handling that would accept anything it was given.
    """
    monkeypatch.setattr(fixture_lint.codex, "CODEX_RECORD_SHAPES", {})
    assert _lint(FIXTURES / "codex") == 1

    monkeypatch.setattr(fixture_lint.opencode, "OPENCODE_RECORD_SHAPES", {})
    assert _lint(FIXTURES / "opencode") == 1
