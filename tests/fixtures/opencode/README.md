# OpenCode fixture corpus

Three hand-authored OpenCode turn excerpts, sanitized under the same
allowlist as `../README.md`'s Claude Code corpus.

## Provenance: nothing here was observed

**Every record in this directory was written by hand.** None of it was
copied, sampled, or paraphrased from a row in
`~/.local/share/opencode/opencode.db`. All prose is invented, about invented
work. No query against that database — read-only or otherwise — ever touched
its `account` or `credential` tables (INV-3); only `message` and `part`'s
schema and row counts were consulted, never their content, before these
fixtures were written.

Shapes are authored from `docs/research.md` §3 (measured against the real
store: 2,156 sessions, 12,915 messages, 53,378 parts), not from an OpenCode
adapter: task 7.2 has not landed yet. `palaver/cli/fixture_lint.py`'s
`OPENCODE_RECORD_SHAPES` is the allowlist these files must satisfy:

```
uv run palaver fixture-lint tests/fixtures
```

## `opencode_message` / `opencode_part` are this corpus's own invention

OpenCode's real store is SQLite rows, not a JSONL transcript — there is no
line-delimited file to sanitize a copy of. Each line here is one JSON object
standing in for one `message` or `part` row: `type` (`"opencode_message"` /
`"opencode_part"`) is a discriminator this fixture format invented, not a
real column; `id`, `session_id`, `message_id`, and `data` mirror the real
schema (`docs/research.md` §3). Every identifier is `fixture-*`, never the
KSUID shape a real OpenCode id takes.

## What this corpus does not attempt

There is no OpenCode `derive_status` yet, so nothing here carries a
documented ground-truth label the way `../README.md`'s Claude Code entries
do, and `tests/test_fixture_lint.py`'s metadata/ground-truth tests do not
read this directory (they glob `tests/fixtures/*.jsonl` non-recursively).
These files exist to prove the allowlist gate holds for OpenCode's shapes and
to give task 7.2 real structural material to build an adapter against.

| File | What it covers |
|---|---|
| `turn-finished.jsonl` | A finished turn, doubly confirmed: `message.data.finish == "stop"` and a terminal `step-finish` part with `reason == "stop"` |
| `tool-call-error.jsonl` | A `type: "tool"` part with `state.status == "error"` and `state.error` |
| `compaction.jsonl` | The rare `part.data.type == "compaction"` marker, followed by a `synthetic: true` text part attached to a `role: "user"` message — provenance is per-*part*, not per-message, which is the same channel-ambiguity lesson INV-8 names for Claude Code, reproduced in a second store |
