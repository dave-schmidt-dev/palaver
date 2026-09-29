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

from pathlib import Path

import palaver

PALAVER_ROOT = Path(palaver.__file__).resolve().parent
