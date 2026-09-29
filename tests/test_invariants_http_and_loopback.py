"""
INV-9: no first-party module constructs an outbound HTTP client, the runtime dependency
set is a reviewed allowlist, and the MCP listener binds only to loopback.
"""

from __future__ import annotations

import ast
import io
import re
import socket
import tomllib
from argparse import Namespace
from pathlib import Path

import pytest

from palaver.cli import mcp as mcp_cli
from palaver.mcp import server as mcp_server
from tests import python_source
from tests._test_invariants_support import PALAVER_ROOT

# =============================================================================
# INV-9 — no Phase 1 module constructs an outbound HTTP client
# =============================================================================

#: Modules whose presence anywhere in first-party source is a violation.
#: `urllib.request` is stdlib and always "installed"; the rest are
#: third-party. `httpx2` is listed separately from `httpx` on purpose: the
#: matcher below keys on the exact dotted name, so `httpx` does not cover
#: `httpx2`, and task 6.1's `mcp` dependency put a real `httpx2` in this
#: environment for the first time. A list that had silently stopped covering
#: the one HTTP client actually installed would be worse than no list.
FORBIDDEN_HTTP_MODULES = ("httpx", "httpx2", "requests", "urllib.request", "openai")


def _phase1_source_paths() -> list[Path]:
    """Every first-party `.py` file.

    Written for Phase 1, when the whole package was the Phase 1 import
    graph. It still sweeps the entire package, so the later phases that have
    since landed — `palaver/memory/`, the inference client, `palaver/mcp/` —
    are covered without the sweep needing to enumerate them. What it does
    *not* cover is anything outside `palaver/`; see
    `test_the_http_client_gate_does_not_see_dependencies`.
    """
    return sorted(PALAVER_ROOT.rglob("*.py"))


def count_http_client_references(paths: list[Path]) -> dict[str, int]:
    """Static-AST count of import references to each forbidden HTTP-client module.

    Counts import statements (`import httpx`, `import httpx.something`,
    `from httpx import Client`, `from httpx.x import y`) rather than
    instrumenting the libraries at runtime — deliberately, since none of
    `httpx`, `requests`, or `openai` is installed here, and any construction
    of a client (`httpx.Client()`) is necessarily preceded by one of these
    import forms, so counting imports is a strict superset check that needs
    no runtime access to the libraries at all. Parsing source text gives the
    same answer whether or not the target package is actually installed,
    which is the property a runtime-instrumentation check cannot offer on
    this machine.

    Args:
        paths: Source files to scan.

    Returns:
        A mapping from each name in `FORBIDDEN_HTTP_MODULES` to the number
        of import references to it found across `paths`.
    """
    counts = {name: 0 for name in FORBIDDEN_HTTP_MODULES}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for target in counts:
                        if alias.name == target or alias.name.startswith(f"{target}."):
                            counts[target] += 1
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for target in counts:
                    if module == target or module.startswith(f"{target}."):
                        counts[target] += 1
    return counts


@pytest.mark.inv9
def test_no_outbound_http_clients():
    """INV-9 gate: no Phase 1 module imports httpx, requests, urllib.request, or openai.

    See the module docstring and `count_http_client_references` for why this
    is a static source scan rather than runtime instrumentation of the
    (absent) libraries themselves.
    """
    paths = _phase1_source_paths()
    # Enumeration is part of the contract: an empty sweep would make the
    # all-zero assertion below pass vacuously regardless of what the source
    # tree contains. `palaver/observer/signals` is a real, already-landed Phase 1
    # module; every one of its files must be swept, whether it is one file or a package.
    assert set(python_source.module_files("palaver/observer/signals")) <= set(paths)

    counts = count_http_client_references(paths)
    assert counts == {name: 0 for name in FORBIDDEN_HTTP_MODULES}


@pytest.mark.inv9
def test_no_outbound_http_clients_check_is_not_vacuous(tmp_path):
    """Positive control: the same detector fails against a module that constructs a client.

    Without this, a detector that always reports zero (e.g. one that
    silently swallowed `ImportError` while trying to instrument the
    libraries at runtime) would make the assertion above pass forever,
    regardless of what Phase 1 code actually does.
    """
    poisoned_httpx = tmp_path / "poisoned_httpx.py"
    poisoned_httpx.write_text(
        "import httpx\n\n"
        "def call_out():\n"
        "    client = httpx.Client()\n"
        "    return client.get('http://example.invalid')\n"
    )
    poisoned_openai = tmp_path / "poisoned_openai.py"
    poisoned_openai.write_text("from openai import OpenAI\n\nclient = OpenAI()\n")
    poisoned_urllib = tmp_path / "poisoned_urllib.py"
    poisoned_urllib.write_text(
        "import urllib.request\n\ndef call_out():\n    return urllib.request.urlopen('x')\n"
    )
    poisoned_requests = tmp_path / "poisoned_requests.py"
    poisoned_requests.write_text("import requests.sessions\n\ns = requests.sessions.Session()\n")

    counts = count_http_client_references(
        [poisoned_httpx, poisoned_openai, poisoned_urllib, poisoned_requests]
    )

    assert counts["httpx"] == 1
    assert counts["openai"] == 1
    assert counts["urllib.request"] == 1
    assert counts["requests"] == 1

    # Positive control on the clean side too: an unrelated stdlib import in
    # the same file does not get miscounted as a forbidden reference.
    clean = tmp_path / "clean.py"
    clean.write_text("import json\nimport os\n\ndef f():\n    return json.dumps({})\n")
    assert count_http_client_references([clean]) == {name: 0 for name in FORBIDDEN_HTTP_MODULES}


@pytest.mark.inv9
def test_the_http_client_gate_does_not_see_dependencies():
    """States the layer this gate covers, so the limit is known rather than assumed.

    Task 6.1 added `mcp`, which pulls `httpx2` transitively — the first
    third-party code in this environment that can open an outbound socket.
    The gate above is a static scan of `palaver/**` and therefore cannot say
    anything about it. That is a real limit, and the honest response is to
    pin it with a test rather than to let the passing gate read as a
    guarantee it never made.

    What INV-9 actually rests on for dependencies is different and stronger:
    the dependency list is one line of `pyproject.toml`, itself inside
    INV-9's declared area, so adding a package is a reviewable event. This
    test asserts the scan's blind spot exists exactly where that review
    takes over.
    """
    import httpx2  # noqa: PLC0415 - imported to prove it is installed and reachable

    assert httpx2.AsyncClient is not None

    # Installed and importable, yet the first-party sweep reports zero —
    # because no file under `palaver/` imports it.
    counts = count_http_client_references(_phase1_source_paths())
    assert counts["httpx2"] == 0

    # And the gate would not have caught it had the reference been in a
    # dependency: the sweep never visits a path outside `palaver/`.
    dependency_path = Path(httpx2.__file__)
    assert PALAVER_ROOT not in dependency_path.parents
    assert dependency_path not in _phase1_source_paths()


#: Every runtime dependency Palaver is allowed to declare, and why it is
#: allowed. An entry here is a decision that this package may open sockets on
#: Palaver's behalf; `mcp` may, because INV-9 permits exactly one local MCP
#: listener and that listener is what this package is.
RUNTIME_DEPENDENCY_ALLOWLIST = {"mcp": "the local MCP listener INV-9 permits, task 6.1"}


@pytest.mark.inv9
def test_the_runtime_dependency_set_is_an_allowlist():
    """INV-9 at the layer the source scan cannot reach.

    The scan above proves Palaver's own code opens nothing. Nothing proves
    the same of a dependency, and no test can without vendoring an opinion
    about every transitive package. What *is* checkable, and what actually
    controls the risk, is the declared set: a new runtime dependency is one
    line of `pyproject.toml`, and this fails until that line is added here
    too, with a reason.

    So the gate is not "dependencies are safe" — it is "no dependency
    arrives unreviewed". That is a claim this test can actually keep.
    """
    pyproject = tomllib.loads((PALAVER_ROOT.parent / "pyproject.toml").read_text())
    declared = pyproject["project"].get("dependencies", [])

    # Names only; the version pin is 6.1's business, not this invariant's.
    names = {re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0].strip() for spec in declared}
    assert names == set(RUNTIME_DEPENDENCY_ALLOWLIST), (
        f"undeclared runtime dependency change: {names ^ set(RUNTIME_DEPENDENCY_ALLOWLIST)}. "
        "Add it to RUNTIME_DEPENDENCY_ALLOWLIST with the reason it may open sockets."
    )

    # Positive control: the parser really does extract a name from a pin,
    # so an allowlist that matched by accident would be visible here.
    assert re.split(r"[<>=!~\[ ]", "mcp>=2.0.0,<3", maxsplit=1)[0] == "mcp"


# INV-9 — the MCP listener cannot be opened anywhere but loopback
#
# The charter names three source-scanning gate tests for INV-9, and they all
# answer the *outbound* half of it: no Palaver module constructs an HTTP
# client. Task 6.1 opened an inbound socket, and `--host` reached
# `socket.bind` with nothing between them, so the store could be served to
# the network by a flag rather than by a code change no scan would see. The
# checks below go after the bind itself and after the two commands that can
# reach it, because a wrong value here is not a crash — it is a listener
# quietly answering the LAN.

#: Addresses that must be refused, each for a different reason. `0.0.0.0` and
#: the empty string are `INADDR_ANY` under two spellings; a LAN literal is
#: the plausible-looking mistake; `::1` *is* loopback but not in the family
#: `bind_listener` creates, and accepting it would surface as a misleading
#: "already in use" error; `localhost` resolves through state the check
#: cannot see.
NON_LOOPBACK_HOSTS = ("0.0.0.0", "", "192.168.1.10", "10.0.0.5", "::1", "localhost", "127.1")


@pytest.mark.inv9
@pytest.mark.parametrize("host", NON_LOOPBACK_HOSTS)
def test_a_non_loopback_bind_address_is_refused(host):
    """INV-9 gate: the validator refuses every address that is not IPv4 loopback."""
    with pytest.raises(mcp_server.NonLoopbackHost) as excinfo:
        mcp_server.ensure_loopback(host)
    # The message has to be actionable, not merely present: whoever typed the
    # flag needs to be told what to type instead.
    assert mcp_server.DEFAULT_HOST in str(excinfo.value)


@pytest.mark.inv9
@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "127.1.2.3"])
def test_the_loopback_check_is_not_a_blanket_refusal(host):
    """Positive control. A validator that refused everything would pass the
    test above while making the server unrunnable, so the accepted set is
    asserted too — and it is the whole of 127.0.0.0/8, not just the default."""
    assert mcp_server.ensure_loopback(host) == host


@pytest.mark.inv9
@pytest.mark.parametrize("host", ["0.0.0.0", "", "192.168.1.10"])
def test_the_listener_itself_cannot_be_opened_off_loopback(host):
    """After the bind, not merely after the CLI.

    `run()` refusing first is what produces a clean exit code, but it is the
    friendly wrapper. This attacks the one function in the codebase that
    creates a listening socket, so a future caller reaching it directly
    cannot open the store to the network by skipping the front door.
    """
    with pytest.raises(mcp_server.NonLoopbackHost):
        mcp_cli.bind_listener(host, 0)


@pytest.mark.inv9
def test_the_listener_still_binds_on_loopback():
    """Positive control for the test above: the refusal did not break binding."""
    sock = mcp_cli.bind_listener(mcp_server.DEFAULT_HOST, 0)
    try:
        assert sock.getsockname()[0] == mcp_server.DEFAULT_HOST
        assert sock.family == socket.AF_INET
    finally:
        sock.close()


@pytest.mark.inv9
@pytest.mark.parametrize("selftest", [False, True])
def test_both_mcp_command_paths_refuse_a_non_loopback_host(tmp_path, selftest):
    """`--selftest` reaches uvicorn through `_build_server`, not `bind_listener`.

    Parametrized over both paths because the check's placement is the whole
    point: it sits ahead of the selftest branch, and moving it below would
    leave `--selftest --host 0.0.0.0` binding `INADDR_ANY` with every other
    test in this file still green.
    """
    out = io.StringIO()
    args = Namespace(
        selftest=selftest,
        clients=1,
        db=tmp_path / "absent.db",
        host="0.0.0.0",
        port=0,
    )
    assert mcp_cli.run(args, out=out, on_status=lambda _: None) == 2
    assert "127.0.0.1" in out.getvalue()
