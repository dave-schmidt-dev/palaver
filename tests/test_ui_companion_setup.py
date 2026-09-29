"""
Profile construction, initial-state seeding, and the source-level agent-mutation
contract.
"""

from __future__ import annotations

import types
from pathlib import Path

from palaver.ui import companion
from palaver.ui.companion_state import JoinState, read_state
from palaver.ui.pane_join import SupportedPaneProcess
from tests import python_source


def test_companion_profile_is_quiet_bounded_and_persistent(monkeypatch):
    class Profile:
        def __init__(self):
            self.calls = {}

        def __getattr__(self, name):
            if not name.startswith("set_"):
                raise AttributeError(name)

            def setter(value):
                self.calls[name] = value

            return setter

    profile = Profile()
    monkeypatch.setattr(
        companion.creation,
        "import_iterm2",
        lambda: types.SimpleNamespace(LocalWriteOnlyProfile=lambda: profile),
    )
    assert companion.build_companion_profile("render") is profile
    assert profile.calls == {
        "set_use_custom_command": "Yes",
        "set_command": "render",
        "set_close_sessions_on_end": False,
        "set_prompt_before_closing": False,
        "set_unlimited_scrollback": False,
        "set_scrollback_lines": companion.SCROLLBACK_LINES,
        "set_silence_bell": True,
        "set_send_bell_alert": False,
        "set_flashing_bell": False,
        "set_visual_bell": False,
        "set_use_custom_window_title": True,
        "set_custom_window_title": "Palaver",
    }


def test_default_state_transport_marks_an_unresolved_join_unjoined(tmp_path):
    path = tmp_path / "opaque.json"
    process = SupportedPaneProcess("agent", 10, "codex", Path("/tmp/example"))
    companion.write_initial_state(path, process, None)
    state = read_state(path)
    assert state.join_state is JoinState.UNJOINED
    assert state.project == "example"
    assert state.source == "codex"


def test_agent_mutations_are_bounded_to_split_link_and_conditional_focus():
    source = python_source.module_text("palaver/ui/companion")
    assert "agent.session.async_send_text" not in source
    assert "agent.session.async_close" not in source
    assert "agent.session.async_set_profile_properties" not in source
    assert source.count(".async_activate(") == 1
    assert "select_tab=False" in source and "order_window_front=False" in source
