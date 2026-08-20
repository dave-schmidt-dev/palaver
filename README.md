# palaver

A local-first observer, memory, and situational-awareness system for people running several AI coding agents in terminal sessions at once.

**Status:** iTerm2 shows a deterministic, session-owned companion pane above each supported agent pane. That surface is fully deterministic and depends on no model.

The observer -- and with it extraction and memory -- is **dormant as of 2026-08-18**. It requires a local llama-server on `127.0.0.1:8090`; its launch agent is stopped and disabled rather than removed until that model path is deliberately re-enabled:

```sh
launchctl enable gui/$(id -u)/com.zerodelta.palaver.observe    # revisit later
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zerodelta.palaver.observe.plist
```

Codex and Claude Code have supervised adapters; OpenCode has an adapter but is not scheduled yet.

## Problem

Running 4–6 coding agents simultaneously means losing track of which one is working, which is blocked, which has asked a question, what each has accomplished, and what was decided hours ago. Worse, when a coding agent compacts its own context, the reasoning behind earlier decisions disappears from the only place it lived.

Palaver watches those sessions independently, keeps evidence-backed structured memory that outlives any agent's context window, and reports status in one place. Eventually the agents themselves can query it.

The mental model:

```text
Raw transcript    = evidence
Structured memory = knowledge
Local LLM         = interpretation
Companion panes / API = interfaces
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

Claude Code and Codex are supervised through adapters that yield canonical events and keep source-namespaced durable cursors, so a restart neither re-ingests nor skips. OpenCode remains excluded from runtime scheduling until its SQLite multi-session contract is designed. Terminal capture is a planned **fallback** adapter for panes with no structured feed.

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
| `ledger.yaml` | Per-invariant state for the charter above: which are gate-covered, and the project's maturity. |
| `LICENSE` | MIT. |

## Design decisions

- **Ingest is adapter-first, not terminal-first.** Every agent in use already persists structured session state on disk. Reading that beats reconstructing a TUI from ANSI escape sequences.
- **Deterministic signals first, model second.** Status, session identity, tool results, and compaction come from the adapter. The model handles interpretation only.
- **Observer cadence:** a 30–60s tick, gated on cursor advance. Idle sessions cost zero inference.
- **Runtime:** `llama-server` (llama.cpp) with a small local model (Gemma-4 E4B, QAT-Q4_0) on a dedicated localhost port. Multi-session hot memory uses server slots (`-np`) with `--slot-save-path` for persistence; save/restore is measured at 16,732 tokens saved in 33 ms and restored in 21 ms.
- **Identity:** Codex projects use the canonical working directory plus a stable collision-resistant suffix; Claude Code preserves its existing cwd-key identity. A pane-local session pin is available for deliberate rename/move recovery. Two panes on the same project share **project-level** memory but keep **separate session-level** state.
- **Memory is append-only.** Correction creates a new superseding row; nothing is deleted or mutated in place. Provenance ordering is enforced by database constraint, not by prompt text — an observer inference cannot supersede an explicit user instruction.
- **UI:** each agent pane gets its own shallow companion pane above it, showing that session's deterministic activity summary, goal, open questions, and recent activity. A local LLM is optional compression, not the source of status.
- **Pane layout:** labeled sections in a fixed gutter — `REQUEST`, `NOW`, `ASK`, `COMMAND`, `DETAIL`. Every section with content earns one row before any earns a second, then spare rows go to the lists and `NOW` absorbs the remainder, so a two-row pane still shows the request and a ten-row pane fills. Activity rows fold each completed tool call and result into one row, and are colored by the producer's evidence kind (red for a failure, dim for tool traffic, default weight for the agent's own prose); the renderer never reads display text to pick a color.
- **Self-observation:** Palaver records the *fact* of a query from the server side and does not feed its own output back through the observer.

## Sensitivity

Palaver's database aggregates the full, unredacted content of every observed session across every project on the machine. That is a strictly higher-value target than any single agent's transcript, because it is the join of all of them. Consequently: no egress beyond localhost, no telemetry, no crash reporting, no non-local model API. The database is gitignored and must never sit in a cloud-synced path.

Test fixtures are transcripts, so the corpus is sanitized under an **allowlist**: a record ships only if it matches a structural shape carrying no free text, or its free-text payload was replaced with prose written for the fixture. `palaver fixture-lint` enforces this and fails on any record it cannot classify.

It checks *every* file under `tests/fixtures/`, not every `.jsonl`. Discovery was a `*.jsonl` glob until 2026-08-15, which left seven committed files unopened — among them a golden output holding verbatim `HUMAN:`/`AGENT:` lines. An extension no checker claims is now a rejection rather than a skip, so a fixture cannot arrive in a new format and be counted as passing.

Before changing fixtures, run provenance lint against every local source corpus that could have supplied their prose: `uv run palaver fixture-lint --provenance-source <source-root>`. This is a manual release gate because the real source corpora are private and absent from CI; an absent requested source is a non-zero error, never a pass.

## Open questions

- Whether a 30–60s observer tick feels live in actual use.
- Whether the structural turn boundary is *correct*, not merely computable on 100% of transcripts.

## Development

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # install, including dev dependencies
uv run pytest -q        # test suite
uv run ruff check .     # lint and dead-code check
uv run pre-commit install   # installs both hooks (see below)
```

`pre-commit install` wires two stages. On **commit**, ruff runs over the files
that commit touches — fast, and silent about the rest of the tree. On **push**,
ruff runs over the whole tree and then the full suite runs, which takes about
two minutes. The push hook is where the fixture-corpus gate lives:
`palaver fixture-lint` is not a hook of its own, it is a test, so a fixture
carrying real session prose fails the suite and the push stops. That gate is
still skippable with `git push --no-verify`, which is a deliberate escape hatch
and not a check.

## Setup: iTerm2 pane discovery

Palaver's pane discovery needs iTerm2's Python API, and **that
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
   iTerm2 runs everything in `AutoLaunch` at launch, so discovery comes back
   on its own after a restart; the shim also restarts the attachment with a
   backoff if it exits.

3. Restart iTerm2, or launch `AutoLaunch/palaver.py` from iTerm2's **Scripts** menu.

Palaver connects over iTerm2's Unix domain socket and **refuses to run** if that
socket is absent, rather than falling back to the library's loopback TCP
listener. Authentication uses the `ITERM2_COOKIE` iTerm2 issues; the cookie is a
credential and is passed only through the environment, never on a command line.

The AutoLaunch process creates and maintains one ten-row companion above each
supported agent pane. The height is set once, when the companion is split, and
is never changed afterwards: an existing pane is reused as-is. **Palaver never
resizes your window.** iTerm2's only pane-sizing call rewrites the whole tab
from the library's cached per-pane sizes, which are captured once and never
refreshed, so that one write resyncs every cached size from live geometry
first, leaving the agent and its companion to redivide only the rows they
already occupy. The call still shrinks the window on its own — it does so even
when it asks for the sizes already on screen, because the layout it sends does
not describe the pane title bars and dividers iTerm draws — so the window frame
is captured beforehand and put back afterwards every time, which also returns
the rows the shrink took. If the frame cannot be read, the sizing write is
skipped and iTerm's own even split stands. Verifying this needs a real
terminal: `PALAVER_RUN_LIVE_COMPANION_TEST=1 pytest tests/test_companion_live.py`
creates a disposable window and asserts the frame never moves. It pairs panes
with reciprocal iTerm variables, restores them after renderer restarts, and
writes private atomic state files that the terminal renderer displays. Long
values wrap at terminal-cell boundaries, including wide characters and
overlong words; only Palaver-owned headers, labels, and statuses receive ANSI
color. User-provided values are rendered as plain text. Input typed into a
companion is discarded and never forwarded.

Claude Code panes join through `~/.claude/sessions/<pid>.json`, the registry
the CLI keeps for each of its own live processes. That record names the pid's
session directly, which the alternative -- narrowing a project directory by
file mtime -- cannot do once the same project has been run more than once
within the hour. The record is accepted only when it claims the pid that was
asked for and the directory the pane and the agent process already agree on;
anything else falls back to the mtime scan, so an older CLI that does not
write the registry behaves exactly as before. Project directories are looked
up under both spellings Claude Code has used for `_`, since both are on disk.

Automatic Codex joining finds recent root rollouts whose recorded cwd exactly
matches the pane. When multiple recent rollouts match, the join narrows them to
the one the live agent process still holds open. If several remain open, it
waits for a second metadata-only observation and joins only when exactly one
stable candidate advances; zero, multiple, or regressing candidates remain
unjoined. For an intentional directory rename or move, pin the known rollout
to the pane without focusing it, and clear the override later:

```sh
uv run palaver ui --session PANE_ID --pin codex SESSION_KEY
uv run palaver ui --session PANE_ID --clear-pin
uv run palaver ui --session PANE_ID --enable-companion
uv run palaver ui --session PANE_ID --disable-companion
```

The observer's semantic extraction requires the configured local
`llama-server` to be running on loopback. Companion panes remain deterministic;
local inference is optional compression rather than the source of status.

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

`--host` accepts `127.0.0.0/8` and nothing else — not `0.0.0.0`, not a LAN
address, not `localhost`, and not `::1`. Both `palaver mcp` and
`install-agent --service mcp` refuse anything else rather than binding it,
because INV-9 keeps the aggregated store of every observed session on this
machine. The refusal happens before the plist is written, so a typo cannot
leave a `KeepAlive` job on disk retrying a forbidden bind every ten seconds.

## Conventions

- Components that touch external surfaces (capture, inference runtime, UI) sit behind interfaces so they can be swapped without rewriting the memory layer.
- Tests verify real behavior. A gate asserts what a measurement *says*, never merely that the measurement ran.
- Every invariant in `INVARIANTS.md` gets a negative test that attacks its enforcement layer, not the Python API above it.
- The charter is itself under test. `tests/test_invariants.py` parses `INVARIANTS.md` and asserts every `gate_test:` resolves to a real function and every `area:` glob matches at least one file on disk. Those fields are read by `harvest`, which maps bug entries to invariants through `area:` — a glob matching nothing produces an invariant that reads as clean because nothing can reach it.
- `ledger.yaml` records per-invariant state beside the charter, and the same suite asserts the two name the same set of invariants. The project is `maturity: pre-mvp`, which means recurrence counts are tracked and reported but never gate work; recurrences accrued before 2026-08-15 are baselined out, because the repo is a day old and those entries are development findings against a contract that was never settled rather than regressions against one that was.

## License

MIT. See `LICENSE`.
