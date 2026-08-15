"""Invariant gate tests attacking each enforcement layer below the Python API.

Per the plan's standing rule, every test here goes after the mechanism that
actually enforces its invariant, not the friendliest wrapper around it — a
test that only proves a Python-level convenience function refuses is not
attacking the invariant, it is attacking the wrapper. Every negative
assertion is paired with a positive control proving the same mechanism is
live and would catch a real violation, not merely agree with whatever the
codebase already looks like.

**INV-3.** `test_opencode_credential_tables_unreachable` and its neighbors
build a fixture SQLite database under `tmp_path` that mirrors OpenCode's real
schema (`docs/research.md` section 3: `session`, `project`, `message`,
`part`, `account`, `credential`), populated with obviously invented token
values. **This module never opens, reads, connects to, queries, copies, or
stats the real `~/.local/share/opencode/opencode.db`.** That file is 2.2 GB
and its `account`/`credential` tables hold live plaintext `access_token`/
`refresh_token` values for real accounts on this machine — exactly what
INV-3 exists to keep unreachable. A fixture proves the identical proposition
(the table is genuinely reachable without the guard, and the allowlist —
not the read-only flag — is what blocks it through the guard) at zero
exposure.

**INV-9.** `test_no_outbound_http_clients` is the charter-named gate test.
`httpx`, `requests`, and `openai` are all absent from this environment, so a
check that imports them to instrument their constructors would either crash
or have to swallow `ImportError` and silently no-op — passing vacuously
forever regardless of what Phase 1 code does. The detector here is a static
AST scan of every Phase 1 source file instead, which needs no such guard and
answers identically whether or not the packages are installed. Its positive
control runs the same detector against a synthetic module that does
construct a client.

**INV-8.** Classification only: an `isMeta` record classifies to the
injected channel, paired with a user-authored record classifying to the
human channel. The write-rejection half of INV-8 (that injected content
cannot be written as tier-1 evidence) has no code path to attack yet —
task 1.2's schema constrains `tier` to 1-5 and nothing more, and the writer
that will enforce provenance is task 2.1 — so it is not attempted here; it
is pinned at `tests/test_normalize.py::test_injected_content_is_not_tier_one`
by task 3.1.

**INV-2 (chokepoint).** `test_adapters_route_every_read_through_the_chokepoint`
asserts no module under `palaver/ingest/adapters/` other than `base.py`
calls `open(`/`os.open` directly — every read must go through
`open_source_readonly`, the one place a source file is ever opened.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import tomllib
from pathlib import Path

import pytest

import palaver
from palaver.ingest.adapters import opencode_guard
from palaver.ingest.adapters.claude_code import CHANNEL_HUMAN, CHANNEL_INJECTED, classify_channel

PALAVER_ROOT = Path(palaver.__file__).resolve().parent
ADAPTERS_DIR = PALAVER_ROOT / "ingest" / "adapters"

# Invented, obviously-fake token values for the fixture db. Never real.
FIXTURE_ACCESS_TOKEN = "invented-access-token-not-real-c0ffee"
FIXTURE_REFRESH_TOKEN = "invented-refresh-token-not-real-decaf0"


# =============================================================================
# INV-3 — OpenCode `account`/`credential` unreachable through the guard
# =============================================================================


def _build_fixture_opencode_db(path: Path) -> None:
    """Create a fixture SQLite db mirroring OpenCode's real schema.

    Table shapes follow `docs/research.md` section 3 (verified against the
    real store): `session` + `project` for identity, `message` + `part` for
    turn content, `account` + `credential` for OAuth material. Every value
    inserted here is invented for this test; none of it is real.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE session (id TEXT PRIMARY KEY, directory TEXT);
        CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
        CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT);
        CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT);
        CREATE TABLE account (id TEXT PRIMARY KEY, access_token TEXT, refresh_token TEXT);
        CREATE TABLE credential (id TEXT PRIMARY KEY, access_token TEXT, refresh_token TEXT);
        """
    )
    conn.execute(
        "INSERT INTO session VALUES (?, ?)",
        ("fixture-session-1", "/tmp/fixture-project"),
    )
    conn.execute(
        "INSERT INTO account VALUES (?, ?, ?)",
        ("fixture-account-1", FIXTURE_ACCESS_TOKEN, FIXTURE_REFRESH_TOKEN),
    )
    conn.execute(
        "INSERT INTO credential VALUES (?, ?, ?)",
        ("fixture-credential-1", FIXTURE_ACCESS_TOKEN, FIXTURE_REFRESH_TOKEN),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_opencode_db(tmp_path: Path) -> Path:
    path = tmp_path / "opencode-fixture.db"
    _build_fixture_opencode_db(path)
    return path


@pytest.mark.inv3
def test_allowed_tables_excludes_account_and_credential():
    """Catches `account`/`credential` being added to the allowlist by mistake."""
    assert "account" not in opencode_guard.ALLOWED_TABLES
    assert "credential" not in opencode_guard.ALLOWED_TABLES
    assert opencode_guard.ALLOWED_TABLES == {"session", "project", "message", "part"}


@pytest.mark.inv3
def test_opencode_credential_tables_unreachable(fixture_opencode_db):
    """A query naming `credential` (or `account`) raises before any SQL executes,
    and the allowlist — not the read-only flag — is what raises it.

    The bullet's own positive control comes first: a raw, unguarded
    read-only `sqlite3` connection against the identical fixture *succeeds*
    in reaching the credential table, proving the table is genuinely
    reachable and the fixture is not accidentally empty or malformed. Only
    then does the guarded connection's failure mean anything.

    "Before any SQL executes" is pinned to the authorizer specifically, not
    just to "the call raised": a guard that instead ran the SELECT, fetched
    rows, and raised only after inspecting the table name would make an
    unqualified `pytest.raises(sqlite3.DatabaseError)` pass identically. The
    message SQLite's own authorizer produces on denial —
    `"access to <table>.<column> is prohibited"` — only exists on the
    compile-time rejection path, so asserting it is present rules out a
    post-hoc check that happened to also raise `DatabaseError`.

    LAYER PROOF follows on the same connection object: `mode=ro` never
    changes; only the authorizer is stripped (`set_authorizer(None)`), and
    the identical query that just raised now succeeds. If `mode=ro` were
    what had blocked it, stripping the allowlist could not have changed the
    outcome. A write attempt on that same de-allowlisted connection still
    fails, proving `mode=ro` was independently in force the whole time
    rather than one layer silently subsuming the other.
    """
    raw_conn = sqlite3.connect(opencode_guard.readonly_uri(fixture_opencode_db), uri=True)
    try:
        rows = raw_conn.execute("SELECT access_token FROM credential").fetchall()
        assert rows == [(FIXTURE_ACCESS_TOKEN,)]
    finally:
        raw_conn.close()

    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="prohibited") as excinfo:
            conn.execute("SELECT access_token FROM credential")
        assert "credential" in str(excinfo.value)

        with pytest.raises(sqlite3.DatabaseError, match="prohibited") as excinfo:
            conn.execute("SELECT access_token, refresh_token FROM account")
        assert "account" in str(excinfo.value)

        # LAYER PROOF: strip only the allowlist; mode=ro is untouched.
        conn.set_authorizer(None)
        rows = conn.execute("SELECT access_token FROM credential").fetchall()
        assert rows == [(FIXTURE_ACCESS_TOKEN,)]

        # mode=ro is still independently in force on this same connection.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO session VALUES ('s2', '/tmp/y')")
    finally:
        conn.close()


@pytest.mark.inv3
def test_opencode_allowlist_permits_allowed_tables(fixture_opencode_db):
    """Positive control: the allowlist blocks `credential`/`account` specifically.

    An authorizer that denied every query would also "block" credential —
    for the wrong reason. Reading an allowlisted table through the same
    guarded connection must still work.
    """
    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        rows = conn.execute("SELECT directory FROM session").fetchall()
        assert rows == [("/tmp/fixture-project",)]
    finally:
        conn.close()


@pytest.mark.inv3
def test_opencode_allowlist_blocks_credential_regardless_of_query_shape(fixture_opencode_db):
    """The allowlist is a structural SQLite check, not a text match on the SQL string.

    Varying case and wrapping the reference in a subquery would both slip
    past a naive `"credential" in sql.lower()` substring check; the real
    guard uses SQLite's own authorizer, which resolves the referenced table
    regardless of surface syntax.
    """
    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT * FROM CREDENTIAL")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT * FROM (SELECT access_token FROM credential)")
    finally:
        conn.close()


@pytest.mark.inv3
def test_open_guarded_readonly_uses_mode_ro_uri(fixture_opencode_db, monkeypatch):
    """Asserts the connection string `open_guarded_readonly` actually issues carries `mode=ro`.

    Spies on `sqlite3.connect` itself rather than inspecting a helper in
    isolation, so this proves what is actually passed to the database
    driver, not just that some string somewhere contains the substring.
    """
    captured = {}
    real_connect = sqlite3.connect

    def _spy_connect(database, *args, **kwargs):
        captured["database"] = database
        captured["kwargs"] = kwargs
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _spy_connect)

    conn = opencode_guard.open_guarded_readonly(fixture_opencode_db)
    conn.close()

    assert "mode=ro" in captured["database"]
    assert captured["kwargs"].get("uri") is True


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
    # tree contains. `signals.py` is a real, already-landed Phase 1 module.
    assert any(path.name == "signals.py" for path in paths)

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


# =============================================================================
# INV-2 (chokepoint) — every adapter read goes through open_source_readonly
# =============================================================================


def _direct_open_call_lines(path: Path) -> list[int]:
    """Line numbers of direct `open(...)` or `os.open(...)` calls in `path`.

    A structural AST-`Call`-node check, not a text grep — a call spread
    across lines or preceded by unrelated text is still found, and a
    substring `"open"` appearing inside a string literal or a comment is
    not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            hits.append(node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and func.attr == "open"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.inv2
def test_adapters_route_every_read_through_the_chokepoint(tmp_path):
    """No module under `palaver/ingest/adapters/` other than `base.py` opens a file directly.

    `base.py` owns `open_source_readonly`, the one chokepoint every adapter
    read must go through (INV-2's docstring, `palaver/ingest/adapters/base.py`).
    A sibling module opening a file itself could quietly request write
    access from a path `open_source_readonly`'s own tests never see.
    """
    scanned = sorted(ADAPTERS_DIR.rglob("*.py"))
    # Enumeration itself is part of the contract: an empty or wrong-directory
    # scan would make `violations == {}` pass vacuously. `rglob`, not `glob`,
    # so a future subpackage under `adapters/` (e.g. task 7.2's OpenCode
    # adapter) is not silently skipped the way a flat `glob` would skip it.
    assert any(path.name == "claude_code.py" for path in scanned)

    violations = {}
    for path in scanned:
        if path.name == "base.py":
            continue
        hits = _direct_open_call_lines(path)
        if hits:
            violations[path.name] = hits
    assert violations == {}

    # Positive control: prove the detector is live by pointing it at modules
    # that do call open()/os.open() directly.
    poisoned_open = tmp_path / "poisoned_open_adapter.py"
    poisoned_open.write_text(
        "def read_it(path):\n    with open(path) as f:\n        return f.read()\n"
    )
    assert _direct_open_call_lines(poisoned_open) == [2]

    poisoned_os_open = tmp_path / "poisoned_os_open_adapter.py"
    poisoned_os_open.write_text(
        "import os\n\ndef read_it(path):\n    return os.open(path, os.O_RDONLY)\n"
    )
    assert _direct_open_call_lines(poisoned_os_open) == [4]
