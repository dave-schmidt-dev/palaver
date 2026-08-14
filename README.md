# palaver

A local-first observer, memory, and situational-awareness system for people running several AI coding agents in terminal sessions at once.

**Status:** pre-implementation. Research and planning are done; Phase 1 of the implementation is starting now. There is no working system yet.

## Problem

Running 4–6 coding agents simultaneously means losing track of which one is working, which is blocked, which has asked a question, what each has accomplished, and what was decided hours ago. Worse, when a coding agent compacts its own context, the reasoning behind earlier decisions disappears from the only place it lived.

Palaver watches those sessions independently, keeps evidence-backed structured memory that outlives any agent's context window, and reports status in one place. Eventually the agents themselves can query it.

The mental model:

```text
Raw transcript    = evidence
Structured memory = knowledge
Local LLM         = interpretation
Status bar / API  = interfaces
```

## Priorities (in order)

1. **Local-first.** Everything runs on one machine. No cloud dependencies, no sync, no accounts.
2. **Simplicity.** Straightforward components until complexity is demonstrably justified. This is a personal developer tool, not a platform.
3. **Observability and replayability.** Any observer decision must be reproducible by replaying a recorded transcript. Debugging a small model's mistakes depends on it.
4. **Evidence-backed memory, unattended.** Memories are structured records with provenance and supersession, never a prose blob the model periodically rewrites. Assume nobody will ever hand-curate the store.

## How it works

Palaver reads the session state that coding agents already persist to disk, rather than scraping their terminal output:

| Agent | Source |
|---|---|
| Claude Code | `~/.claude/projects/<cwd-key>/<sessionId>.jsonl` |
| Codex | `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` |
| OpenCode | `~/.local/share/opencode/opencode.db` |

Each source sits behind an adapter that yields canonical events and keeps a durable per-session cursor, so a restart neither re-ingests nor skips. Terminal capture is a planned **fallback** adapter for panes with no structured feed.

Status is computed in Python from deterministic signals — turn boundaries, unresolved tool calls, error results, compaction markers. The local model supplies the semantic layer only: current task, decisions, remaining work, rollups. It never sets status. When no signal supports an answer, the status is `UNKNOWN`, which is a first-class value rather than a guess.

## Layout

| Path | Purpose |
|---|---|
| `palaver/` | The package. |
| `palaver/ingest/adapters/` | One adapter per agent session store. |
| `palaver/observer/` | Deterministic signals and status derivation. |
| `palaver/memory/` | Append-only memory with provenance tiers. |
| `palaver/cli/` | `palaver status`, `inspect`, and friends. |
| `tests/` | Test suite, including a sanitized transcript fixture corpus. |
| `INVARIANTS.md` | The system contract. Read this before changing behavior. |
| `LICENSE` | MIT. |

## Design decisions

- **Ingest is adapter-first, not terminal-first.** Every agent in use already persists structured session state on disk. Reading that beats reconstructing a TUI from ANSI escape sequences.
- **Deterministic signals first, model second.** Status, session identity, tool results, and compaction come from the adapter. The model handles interpretation only.
- **Observer cadence:** a 30–60s tick, gated on cursor advance. Idle sessions cost zero inference.
- **Runtime:** `llama-server` (llama.cpp) with a small local model (Gemma-4 E4B, QAT-Q4_0) on a dedicated localhost port. Multi-session hot memory uses server slots (`-np`) with `--slot-save-path` for persistence; save/restore is measured at 16,732 tokens saved in 33 ms and restored in 21 ms.
- **Identity:** project name plus timestamp, derived from the session's working directory or workspace name, with a manual pin available as an override. Two panes on the same project share **project-level** memory but keep **separate session-level** state.
- **Memory is append-only.** Correction creates a new superseding row; nothing is deleted or mutated in place. Provenance ordering is enforced by database constraint, not by prompt text — an observer inference cannot supersede an explicit user instruction.
- **UI:** status should live as close to its pane as possible. The leading candidate is an iTerm2 per-session status bar component rather than a window-level panel.
- **Self-observation:** Palaver records the *fact* of a query from the server side and does not feed its own output back through the observer.

## Sensitivity

Palaver's database aggregates the full, unredacted content of every observed session across every project on the machine. That is a strictly higher-value target than any single agent's transcript, because it is the join of all of them. Consequently: no egress beyond localhost, no telemetry, no crash reporting, no non-local model API. The database is gitignored and must never sit in a cloud-synced path.

Test fixtures are transcripts, so the corpus is sanitized under an **allowlist**: a record ships only if it matches a structural shape carrying no free text, or its free-text payload was replaced with prose written for the fixture. A linter enforces this and fails on any record it cannot classify.

## Open questions

- The iTerm2 status bar rendering budget — iTerm2 documents no character limit, only a user-configurable width in points.
- Whether the iTerm2 API can prove a status bar component is *registered and attached*, as opposed to merely round-tripping a variable.
- Whether a 30–60s observer tick feels live in actual use.
- Whether one MCP server over Streamable HTTP sustains 6 concurrent local clients.
- Whether the structural turn boundary is *correct*, not merely computable on 100% of transcripts.

## Development

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # install, including dev dependencies
uv run pytest -q        # test suite
uv run ruff check .     # lint and dead-code check
uv run pre-commit install   # fast lint hook on commit
```

## Conventions

- Components that touch external surfaces (capture, inference runtime, UI) sit behind interfaces so they can be swapped without rewriting the memory layer.
- Tests verify real behavior. A gate asserts what a measurement *says*, never merely that the measurement ran.
- Every invariant in `INVARIANTS.md` gets a negative test that attacks its enforcement layer, not the Python API above it.

## License

MIT. See `LICENSE`.
