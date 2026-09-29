"""
The mcp agent's own plist and options: not throttled like a background job, its label
never collides with the observer's, and no option is claimed by both services.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

import pytest

from palaver.cli import install_agent
from palaver.cli.install_agent import (
    MCP_LABEL,
    MCP_TEMPLATE_PATH,
    OBSERVE_LABEL,
    OBSERVE_TEMPLATE_PATH,
    mcp_program_arguments,
    render_plist,
)
from tests._supervision_support import (
    MCP_SELFTEST_LABEL,
    SELFTEST_LABEL,
    _install_args,
    _run_capture,
)

# --- the mcp agent (task 6.5) ----------------------------------------------
#
# Every name here carries `mcp_agent`, which is the selection the plan's quick
# check runs: `pytest -q tests/test_supervision.py -k mcp_agent`. A test whose
# name misses that substring is a test the gate does not run.


def _render_keys(template_path: Path, tmp_path: Path) -> dict:
    """Render a template with throwaway values and parse what launchd would see.

    The templates cannot be parsed directly — they are `string.Template`
    sources, and the unsubstituted placeholders are not valid inside the
    elements they sit in. Rendering first is the only way to compare them as
    plists rather than as text.
    """
    rendered = render_plist(
        label="com.zerodelta.palaver.render-probe",
        program_arguments=["/usr/bin/true"],
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        working_directory=tmp_path,
        template_path=template_path,
    )
    return plistlib.loads(rendered.encode("utf-8"))


def test_the_rendered_mcp_agent_plist_is_something_launchd_can_parse(tmp_path):
    plist = tmp_path / "mcp-agent.plist"
    plist.write_text(
        render_plist(
            label=MCP_SELFTEST_LABEL,
            program_arguments=mcp_program_arguments(
                executable=Path("/bin/palaver"), db_path=tmp_path / "m.db", port=9999
            ),
            stdout_path=tmp_path / "out.log",
            stderr_path=tmp_path / "err.log",
            template_path=MCP_TEMPLATE_PATH,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["plutil", "-lint", str(plist)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # `plutil` alone is not enough, and this is not a hypothetical. The first
    # draft of this template wrote a command-line flag inside an XML comment.
    # A double hyphen is illegal there, `plutil -lint` accepted the file
    # regardless, and plistlib refused it. Parsing with a second, stricter
    # implementation is what turned that into a failure instead of a latent
    # portability bug in a file only launchd normally reads.
    parsed = plistlib.loads(plist.read_bytes())
    assert parsed["Label"] == MCP_SELFTEST_LABEL


def test_the_mcp_agent_is_not_throttled_the_way_a_background_job_is(tmp_path):
    """The finding that made task 6.5 more than a copy of task 5.0.

    `man launchd.plist` describes Background's resource limits as existing
    "to prevent them from disrupting the user experience". This server is
    *inside* the user experience: every cycle it spends is an agent's blocking
    tool call. So it is Standard, and it omits LowPriorityIO and Nice rather
    than setting them low.

    The observer assertions are the positive control. Without them, this test
    would pass just as happily against a template that had lost its scheduling
    keys entirely, which is a different bug wearing the same result.
    """
    mcp = _render_keys(MCP_TEMPLATE_PATH, tmp_path)
    assert mcp["ProcessType"] == "Standard"
    assert "LowPriorityIO" not in mcp
    assert "Nice" not in mcp

    observe = _render_keys(OBSERVE_TEMPLATE_PATH, tmp_path)
    assert observe["ProcessType"] == "Background"
    assert observe["LowPriorityIO"] is True
    assert observe["Nice"] == 5


def test_the_two_mcp_agent_templates_agree_on_everything_but_scheduling(tmp_path):
    """The guard that lets these be two files instead of one.

    Two static templates were chosen over one parameterized template because
    the MCP job *omits* keys the observer sets, and a placeholder can supply a
    value but cannot remove an element. The cost of that choice is ~60 lines
    of duplicated XML that could drift. This is what makes drift fail loudly:
    every key outside the scheduling set must be present in both and identical
    in both.
    """
    scheduling = {"ProcessType", "LowPriorityIO", "Nice"}
    observe = _render_keys(OBSERVE_TEMPLATE_PATH, tmp_path)
    mcp = _render_keys(MCP_TEMPLATE_PATH, tmp_path)

    assert set(observe) - scheduling == set(mcp) - scheduling
    for key in sorted(set(observe) - scheduling):
        assert observe[key] == mcp[key], f"{key} drifted between the two templates"

    # And the divergence is exactly the one that was intended, not merely
    # non-empty: naming both sides keeps this from passing if the MCP job
    # quietly grew a Nice key back.
    assert set(observe) & scheduling == scheduling
    assert set(mcp) & scheduling == {"ProcessType"}

    # And the exemption cannot be widened to cover a drift. Every key named in
    # `scheduling` has to be a real point of divergence, so adding a newly
    # drifted key to the set in the same commit as the drift does not buy a
    # green run — which is the one way the check above could be defeated by
    # editing only this file.
    for key in sorted(scheduling):
        diverges = key not in observe or key not in mcp or observe[key] != mcp[key]
        assert diverges, f"{key} is exempted from the drift check but does not actually differ"


def test_unset_mcp_agent_options_are_omitted_rather_than_duplicated():
    """The server owns its own defaults; the plist must not restate them."""
    bare = mcp_program_arguments(executable=Path("/bin/palaver"))
    assert bare == ["/bin/palaver", "mcp"]

    full = mcp_program_arguments(
        executable=Path("/bin/palaver"),
        db_path=Path("/tmp/m.db"),
        host="127.0.0.1",
        port=8787,
    )
    assert full == [
        "/bin/palaver",
        "mcp",
        "--db",
        "/tmp/m.db",
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
    ]


def test_the_mcp_agent_and_the_observer_never_share_a_label():
    """Two jobs under one label is one job; launchd keys everything on it."""
    assert MCP_LABEL != OBSERVE_LABEL
    assert MCP_SELFTEST_LABEL not in {MCP_LABEL, OBSERVE_LABEL, SELFTEST_LABEL}
    assert install_agent.SERVICES["mcp"].label == MCP_LABEL
    assert install_agent.SERVICES["observe"].label == OBSERVE_LABEL
    assert install_agent.SERVICES["mcp"].template_path != (
        install_agent.SERVICES["observe"].template_path
    )


@pytest.mark.parametrize(
    ("service", "option", "value"),
    [
        ("mcp", "interval", 45.0),
        ("mcp", "cursors", Path("/tmp/cursors")),
        ("observe", "host", "127.0.0.1"),
        ("observe", "port", 9999),
    ],
)
def test_an_mcp_agent_option_meant_for_the_other_service_is_refused(
    tmp_path, capsys, service, option, value
):
    """Refused, not ignored.

    `--service mcp --interval 45` is a coherent sentence that means nothing.
    Dropping the flag silently would write a plist that loads, runs, and
    supervises something other than what was asked, with nothing anywhere
    recording that a flag was discarded.
    """
    import io

    target = tmp_path / "never-written.plist"
    args = _install_args(service=service, plist_path=target, **{option: value})
    status = install_agent.run(args, out=io.StringIO(), on_status=lambda _message: None)

    assert status == 2
    assert f"--{option}" in capsys.readouterr().err
    assert not target.exists(), "a refused invocation still wrote a plist"


def test_no_option_is_claimed_by_more_than_one_mcp_agent_service():
    """The cross-service refusal assumes the option sets are disjoint.

    It refuses any option declared by a service other than the selected one,
    without first checking whether the selected service declares it too. That
    is correct only while no two services share an option name; the moment
    one does, the shared flag would be refused for both services and neither
    could use it. Pinning the assumption here is cheaper than the defensive
    exclusion, and it fails at the commit that breaks it rather than at the
    invocation.
    """
    seen: dict[str, str] = {}
    for service in install_agent.SERVICES.values():
        for option in service.options:
            assert option not in seen, (
                f"--{option} is declared by both {seen[option]} and {service.name}; "
                f"the refusal loop in `run` would now reject it for both"
            )
            seen[option] = service.name


@pytest.mark.inv9
@pytest.mark.parametrize("host", ["0.0.0.0", "", "192.168.1.10", "::1", "localhost"])
def test_an_mcp_agent_asked_to_bind_off_loopback_is_refused(tmp_path, capsys, host):
    """INV-9, at the layer that makes the mistake durable.

    `palaver mcp` refuses the address at startup, so the server never serves
    it either way. What this stops is the plist: without the check the
    command writes a `RunAtLoad`+`KeepAlive` job on disk asking for a bind
    that INV-9 forbids, and launchd then retries it every ten seconds
    forever. The file is the artifact that outlives the typo, so the refusal
    has to happen before the file exists — which is what the last assertion
    pins.
    """
    import io

    target = tmp_path / "never-written.plist"
    args = _install_args(service="mcp", host=host, plist_path=target)
    status = install_agent.run(args, out=io.StringIO(), on_status=lambda _message: None)

    assert status == 2
    assert "127.0.0.1" in capsys.readouterr().err
    assert not target.exists(), "a plist asking for a forbidden bind was written to disk"


@pytest.mark.inv9
def test_an_mcp_agent_on_loopback_is_still_installable(tmp_path):
    """Positive control. A refusal that rejected every `--host` would satisfy
    the test above while making `--service mcp` unusable."""
    target = tmp_path / "loopback.plist"
    rendered = _run_capture(
        _install_args(
            service="mcp",
            host="127.0.0.2",
            print_only=True,
            plist_path=target,
            log_dir=tmp_path / "logs",
        )
    )
    assert "127.0.0.2" in rendered


def test_the_mcp_agent_service_is_reachable_from_the_real_parser(tmp_path):
    """The wiring check, and the reason it goes through `run` and not `render_plist`.

    Tasks 6.3 and 6.4 both produced a fully-tested unit that nothing called.
    Asserting on `render_plist(template_path=MCP_TEMPLATE_PATH)` would pass
    identically if `--service` had never been registered on the parser. So
    this drives the argv a user would actually type.
    """
    import io

    from palaver.cli import build_parser

    target = tmp_path / "from-parser.plist"
    parsed = build_parser().parse_args(
        [
            "install-agent",
            "--service",
            "mcp",
            "--port",
            "9999",
            "--plist-path",
            str(target),
            # Without this, `run()` mkdirs `.logs/` inside the real
            # repository. The plist only needs the paths to render; it is
            # never loaded here, so nothing has to write to them.
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    assert parsed.service == "mcp"
    assert parsed.handler is install_agent.run

    status = parsed.handler(parsed, out=io.StringIO(), on_status=lambda _message: None)
    assert status == 0

    written = plistlib.loads(target.read_bytes())
    assert written["Label"] == MCP_LABEL
    assert written["ProcessType"] == "Standard"
    assert written["ProgramArguments"][1] == "mcp"
    assert written["ProgramArguments"][-2:] == ["--port", "9999"]


def test_the_default_mcp_agent_service_is_still_the_observer(tmp_path):
    """Task 5.0's invocation must keep meaning what it meant.

    `palaver install-agent` shipped before `--service` existed. If the default
    moved, an existing habit would silently install a different job.
    """
    import io

    from palaver.cli import build_parser

    target = tmp_path / "default.plist"
    parsed = build_parser().parse_args(
        # `--log-dir` for the same reason as the test above: keep `run()`'s
        # mkdir inside tmp_path rather than in the checkout.
        ["install-agent", "--plist-path", str(target), "--log-dir", str(tmp_path / "logs")]
    )
    assert parsed.service == "observe"

    status = install_agent.run(parsed, out=io.StringIO(), on_status=lambda _message: None)
    assert status == 0

    written = plistlib.loads(target.read_bytes())
    assert written["Label"] == OBSERVE_LABEL
    assert written["ProgramArguments"][1] == "observe"
