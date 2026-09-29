"""INV-7/INV-9: nothing in the status path can reach a model client or a socket."""

import ast

from tests import python_source

#: Import roots that would put a model call or a socket inside the status
#: path. `derive_status()` reaching any of these would breach INV-7 (the
#: model never sets status) or INV-9 (content never leaves this machine).
BANNED_IMPORT_ROOTS = frozenset(
    {"httpx", "requests", "urllib", "openai", "aiohttp", "socket", "http", "llama_cpp"}
)


def _imported_modules(source: str) -> set[str]:
    """Return the full dotted module name of every import in `source`."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _imported_roots(source: str) -> set[str]:
    """Return the top-level module name of every import in `source`."""
    return {module.split(".")[0] for module in _imported_modules(source)}


# --- INV-7 / INV-9: nothing in the status path can reach a model or a socket --


def test_signals_module_imports_no_network_or_model_client():
    """The status path imports nothing that could reach a model or a socket.

    INV-7 (the model never sets status) and INV-9 (content never leaves this
    machine) both fail the moment a client library appears in this import
    list. The exact-set assertion is the tripwire: any new import here has to
    be a deliberate decision reviewed against both invariants, not a quiet
    addition.

    Asserted over full dotted module paths, not top-level roots. Task 3.6
    imports `Extraction` from `palaver.extract.persist`, and a root-set
    assertion would from then on admit the whole of `palaver` — including
    `palaver.extract.client`, the one module in this tree that opens a
    socket. Naming the exact module keeps the tripwire as tight as it was
    before the import existed.
    """
    modules = python_source.imported_modules("palaver/observer/signals")

    assert modules == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "enum",
        "palaver.extract.persist",
    }
    assert "palaver.extract.client" not in modules
    assert not {module.split(".")[0] for module in modules} & BANNED_IMPORT_ROOTS


def test_banned_import_detector_is_not_inert():
    """Positive control for the test above: the same extraction, run over source
    that really does import a network client, flags it.

    Without this, an `_imported_roots` that returned an empty set for every
    input would make the INV-7/INV-9 check pass unconditionally. The
    module-path extraction is controlled the same way, since the tripwire
    above now depends on it distinguishing `palaver.extract.persist` from
    `palaver.extract.client`.
    """
    source = "import httpx\nfrom urllib import request\nimport json\n"
    roots = _imported_roots(source)

    assert roots == {"httpx", "urllib", "json"}
    assert roots & BANNED_IMPORT_ROOTS == {"httpx", "urllib"}

    assert _imported_modules("from palaver.extract.client import ModelClient\n") == {
        "palaver.extract.client"
    }
