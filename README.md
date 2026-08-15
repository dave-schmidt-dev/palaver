# palaver

A local-first observer, memory, and situational-awareness system for people running several AI coding agents in terminal sessions at once.

**Status:** built and running. Ingest, memory, extraction, the observer daemon, the iTerm2 pane surface, the Codex/OpenCode adapters, and the MCP surface — reads, pagination, the single-writer socket, `palaver_correct`, and query events — are all in place, each supervised by its own launchd user agent.

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
| `palaver/mcp/` | The MCP surface other agents query. Reads go through a `mode=ro` connection; the two writes it can make (a correction, a query event) are posted to the observer daemon rather than opened locally. |
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
- Whether the structural turn boundary is *correct*, not merely computable on 100% of transcripts.

## Development

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # install, including dev dependencies
uv run pytest -q        # test suite
uv run ruff check .     # lint and dead-code check
uv run pre-commit install   # fast lint hook on commit
```

## Setup: the iTerm2 pane surface

The status Palaver shows inside each pane needs iTerm2's Python API, and **that
preference is off by default**. Until it is on, iTerm2 creates no API socket, no
script can connect, and nothing anywhere reports why — the surface is simply
absent. Turn it on first:

1. **iTerm2 > Settings > General > Magic > Enable Python API.**
2. Install the extra and the AutoLaunch script:

   ```sh
   uv sync --extra ui
   uv run python -m palaver.ui.autolaunch --install
   ```

   That writes a small shim to `~/Library/Application Support/iTerm2/Scripts/AutoLaunch/palaver.py`.
   iTerm2 runs everything in `AutoLaunch` at launch, so the surface comes back
   on its own after a restart; the shim also restarts the attachment with a
   backoff if it exits.

3. Restart iTerm2, or run the attachment by hand for a session:

   ```sh
   uv run python -m palaver.ui.autolaunch
   ```

Palaver connects over iTerm2's Unix domain socket and **refuses to run** if that
socket is absent, rather than falling back to the library's loopback TCP
listener. Authentication uses the `ITERM2_COOKIE` iTerm2 issues; the cookie is a
credential and is passed only through the environment, never on a command line.

To check whether the pane surface actually works on this machine:

```sh
uv run palaver ui --selftest             # registers, round-trips a variable, renders, publishes
uv run palaver ui --enable-status-bar    # turns the bar on; changes how every pane looks
uv run palaver ui --disable-status-bar   # and turns it back off
```

The selftest reports rather than judges. A profile with the component not yet
added to its status bar layout is normal on a fresh machine, so that exits 0
with the remedy printed; it fails only for things Palaver owns — registration
refused, a variable that will not round-trip, a render tick that does not
advance. Adding the component to the layout is a one-time manual step in
iTerm2 > Settings > Profiles > Session > Configure Status Bar, and turning the
bar on is a separate flag because it changes what every pane looks like.

What actually writes each pane's status is the AutoLaunch process, not the
observer daemon: it holds the iTerm2 connection, joins each pane to a session
on disk, and pushes on a heartbeat. The heartbeat is the point rather than an
implementation detail — a status carries the time it was pushed, and one that
stops being refreshed stops being shown, so a publisher that skipped an
unchanged status would blank the pane of an agent working steadily.

## Serving memory to other agents

Palaver exposes its memory to Claude Code, Codex, and anything else that
speaks MCP:

```sh
uv run palaver mcp                       # serve on http://127.0.0.1:8787/mcp
uv run palaver mcp --selftest --clients 6   # prove it serves six at once
```

**Streamable HTTP, not stdio.** stdio is subprocess-per-client, so six
attached agents would be six processes writing one SQLite file — the opposite
of the single-writer property the memory layer rests on. One HTTP server at a
fixed localhost endpoint keeps the process count at one however many agents
connect, and a Palaver restart costs each attached agent one reconnect
rather than a dead tool.

That last part is narrower than it first looks, and it was measured rather
than assumed. The HTTP transport does reconnect, but the MCP **session** does
not survive: the restarted server has never seen the `Mcp-Session-Id` the
client is holding, so it answers 404 and the SDK reports `Session terminated`
instead of re-initializing. Re-establishing the session is the host
application's job. A client that does so reads exactly what it read before;
one that holds a session across a restart gets an error, not stale data.

Every read tool requires an explicit scope of exactly one of
`{project: <name>}` or `{session: <session_key>}`. There is no default, on
purpose: project memory returned where session memory was meant reads exactly
as authoritative as the right answer, and nothing downstream can tell the
difference. Session keys are the `<project>/<session-id>` form `palaver
status` prints; `palaver_sessions` lists them, and an internal rowid is
refused rather than resolved.

Register it once:

```sh
claude mcp add --transport http palaver http://127.0.0.1:8787/mcp
```

Supervising Palaver's two long-lived processes is separate and needs no
iTerm2. Each gets its own launchd user agent:

```sh
uv run palaver install-agent                        # the observer daemon
uv run palaver install-agent --load                 # and load it

uv run palaver install-agent --service mcp --load   # the MCP server
```

The two plists differ in more than their argv. The observer is `Background`
with `LowPriorityIO` and a positive `Nice`, because its work is nobody's
request and must never contend with the sessions it watches. The MCP server
is `Standard` with neither, because every cycle it spends is inside an
agent's blocking tool call. `Adaptive` is not an option for it: launchd
promotes an Adaptive job out of Background on *XPC* activity, and this one
speaks HTTP, so it would stay throttled forever.

## Conventions

- Components that touch external surfaces (capture, inference runtime, UI) sit behind interfaces so they can be swapped without rewriting the memory layer.
- Tests verify real behavior. A gate asserts what a measurement *says*, never merely that the measurement ran.
- Every invariant in `INVARIANTS.md` gets a negative test that attacks its enforcement layer, not the Python API above it.

## License

MIT. See `LICENSE`.
