# Codex fixture corpus

Five hand-authored Codex rollout excerpts, sanitized under the same allowlist
as `../README.md`'s Claude Code corpus.

## Provenance: nothing here was observed

**Every record in this directory was written by hand.** None of it was
copied, sampled, or paraphrased from a real file under `~/.codex/sessions/`.
All prose is invented, about invented work.

Shapes are authored from `docs/research.md` §2 (4,884 real rollout files,
measured structurally — no free text was ever read out of them), not from a
Codex adapter: task 7.1 (the adapter) has not landed yet. `palaver/cli/
fixture_lint.py`'s `CODEX_RECORD_SHAPES` is the allowlist these files must
satisfy:

```
uv run palaver fixture-lint tests/fixtures
```

Every record's envelope is exactly `{"type", "payload"}` — no `timestamp`, no
`ordinal` — even though every real rollout record carries a `timestamp`. That
omission is deliberate, mirrors the Claude Code corpus's missing `uuid`/`cwd`/
`version` keys, and means a record pasted from a real session is rejected on
its key set before its prose is ever considered. `session_meta.payload.{id,
session_id}` and every other identifier are `fixture-*`, never the UUID shape
a real Codex session id takes.

## What this corpus does not attempt

This is not the ground-truth accuracy corpus `../README.md` documents for
Claude Code — there is no Codex `derive_status` yet, so nothing here carries
a documented "derived today" label, and `tests/test_fixture_lint.py`'s
metadata/ground-truth tests do not read this directory (they glob
`tests/fixtures/*.jsonl` non-recursively). These files exist to prove the
allowlist gate holds for Codex's shapes and to give task 7.1 real structural
material to build an adapter against.

| File | What it covers |
|---|---|
| `session-finished.jsonl` | `session_meta` → human turn → assistant reply → `task_complete` |
| `session-turn-aborted.jsonl` | Turn boundary via `turn_aborted` instead of `task_complete` |
| `session-error.jsonl` | `event_msg` `type: "error"` with `message` and `codex_error_info` |
| `session-compaction.jsonl` | The paired `compacted` + `context_compacted` marker |
| `session-developer-and-injected-channel.jsonl` | `role: "developer"` (always harness) and a `role: "user"` record wearing an injected prefix — Codex has no `isMeta` equivalent, so this is the shape a channel heuristic has to see |
