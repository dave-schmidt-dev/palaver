# Invariants — palaver

> System contract. Each entry pairs an `area:` glob with the `gate_test:` that
> enforces it, so a breach maps to a specific failing test rather than to a
> discussion. Written 2026-08-14 before any implementation, so several globs and
> gate tests name files that do not exist yet; they go live as each phase lands.
> The ordering is deliberate — a charter written after the code would have been
> written to match the code.

### INV-1 — Every network call, subprocess, model inference, and stall-prone IO surfaces live progress
area: ["palaver/**/*.py"]
gate_test: tests/test_scheduler.py::test_observer_tick_emits_status
threshold: 3
rationale: Palaver is a
  watcher whose whole value is telling a human what is happening; a Palaver operation that itself
  blocks silently is the defect it exists to prevent. Local inference against `llama-server` takes
  3–17 seconds per session at realistic transcript sizes (measured, spike run 1), and six sessions
  tick on a 30–60s cadence, so a silent inference stall is indistinguishable from a hung observer.
  Every such call takes an `on_status` channel; the channel never writes to stdout, which belongs to
  the CLI's own output contract.

### INV-2 — Palaver never writes to, controls, or interrupts an observed agent session
area: ["palaver/ingest/**/*.py", "palaver/observer/**/*.py"]
gate_test: tests/test_adapters.py::test_adapters_never_open_source_writable
threshold: 3
rationale: The brief's first non-goal is autonomous control of coding agents, and the second is
  automatic injection of warnings into agent prompts. An observer that can write to the thing it
  observes stops being evidence and becomes a participant — its own writes appear in the next tick's
  transcript and it begins reasoning about its own output. Warnings surface in Palaver's UI only.
  This holds until Palaver has demonstrated reliability the user explicitly signs off on; it is not a
  temporary scaffold.

### INV-3 — Third-party session stores are opened read-only through an explicit allowlist
area: ["palaver/ingest/adapters/**/*.py"]
gate_test: tests/test_invariants.py::test_opencode_credential_tables_unreachable
threshold: 3
rationale: `~/.local/share/opencode/opencode.db` contains `account` and `credential` tables holding
  plaintext `access_token` and `refresh_token`. Palaver has no use for either. The OpenCode adapter
  opens the database with a read-only URI (`file:...?mode=ro`) AND enforces a table allowlist in code,
  because read-only alone still permits reading a token into memory, a log, a prompt, or a model
  request — and a model request is the one place a leaked token could leave the process. The gate
  asserts the credential tables are unreachable through the adapter, not merely unread by convention.
  The same allowlist discipline applies to any future adapter over a store the project does not own.

### INV-4 — Memory is append-only: supersession, never deletion or in-place mutation
area: ["palaver/memory/**/*.py"]
gate_test: tests/test_memory.py::test_supersede_preserves_original_row
threshold: 3
rationale: The brief is explicit — "Do not destroy old memories" — and the reason is recoverability:
  the observer is a 4B model that will be wrong, and an audit trail is the only way to recover from
  its mistakes without a human curating the store. Correction creates a new row with
  `supersedes: <id>` and marks the original `superseded`; no code path issues `DELETE` or overwrites a
  memory's `statement`, `origin`, or `created_at`. Regeneratable current-state summaries are exempt
  and are stored separately from durable memories precisely so this invariant can be absolute.

### INV-5 — Provenance ordering is enforced in the database, not in prompt text
area: ["palaver/store/schema.py", "palaver/memory/supersede.py"]
gate_test: tests/test_memory.py::test_lower_tier_cannot_supersede_higher_tier
threshold: 3
rationale: Tiers, highest first: (1) explicit user instruction or correction, (2) explicit main-agent
  conclusion, (3) observed tool or command result, (4) observer inference, (5) observer speculation.
  An observer inference must never supersede an explicit user instruction. Enforcing this in the
  prompt would make the guarantee only as strong as a 4B model's instruction-following, which spike
  run 2 measured directly: the model ignored an ordered rule list whose predicates were its own output
  fields. The constraint therefore lives in a CHECK/trigger at the write boundary, where it holds
  regardless of which model wrote the row or how it was prompted.

### INV-6 — Every durable memory carries at least one evidence link to stored transcript
area: ["palaver/memory/**/*.py"]
gate_test: tests/test_memory.py::test_memory_without_evidence_is_rejected
threshold: 3
rationale: **Amended 2026-08-14 (task 3.3), and the word "raw" was dropped from this entry's title
  deliberately.** Two artifacts are stored, and an anchor may name either: `events.payload` holds the
  raw record byte-for-byte, and `transcript_chunks.content` holds the normalized semantic text. Until
  task 3.3 both were raw. The change is not a weakening of the evidence requirement — nothing is
  discarded — but it matters to what an anchor means, so it is written here rather than left implied.
  A chunk anchor now indexes exactly the text the model read, which is what makes the quote-grounding
  gate meaningful: checking a model's quote against raw JSON made the check depend on JSON escaping,
  so it admitted plain one-line quotes and rejected any quote containing a newline, a double quote, or
  a backslash — passing the trivial cases and failing the substantive ones, at a rate that would have
  read as a quality measurement. An anchor into raw bytes is still available through `events.payload`
  for anything needing the original record.
  "Raw transcript = evidence, structured memory = knowledge." A memory with no evidence link
  is indistinguishable from a fabrication and cannot be audited, replayed, or refuted. Spike run 1
  proved the cheap enforcement mechanism: require a verbatim `quote` on every extracted memory and
  substring-check it against the source — 17/17 quotes verified, zero fabrications. A quote that does
  not appear in its cited evidence span is rejected at write time, not flagged for later review.
  Spike run 2 added the second half: the quote must come from the channel the memory claims, or a real
  quote from injected content passes a check it should fail.

### INV-7 — Status is computed from deterministic signals; the model never sets it
area: ["palaver/observer/signals.py"]
gate_test: tests/test_signals.py::test_status_is_never_model_supplied
threshold: 3
rationale: Measured, not assumed. Spike run 1: E4B extracted 17/17 user decisions correctly but got
  `status` wrong on the sessions that mattered, and an explicit ordered rule list in the prompt changed
  nothing. Spike run 2: the model honours rules whose predicates are computed signals and ignores rules
  whose predicates are its own generated fields. So `derive_status()` owns the ordered list in Python
  and the model supplies extraction only. Corollary, from the brief's "do not equate lack of terminal
  output with DONE": when no signal supports any status, the answer is `UNKNOWN`, which is a
  first-class enum value. A confident wrong `DONE` tells the human a session needs nothing when it may
  be waiting on them — the single most costly error this system can make.

### INV-8 — Human-typed input and harness-injected content are distinguished at ingest
area: ["palaver/extract/normalize.py", "palaver/extract/quote_gate.py", "palaver/ingest/adapters/**/*.py"]
gate_test: tests/test_extraction.py::test_injected_content_is_not_tier_one
threshold: 3
rationale: Tier-1 provenance means "the user said this", and it is the tier every other tier defers
  to under INV-5, so a mis-tagged channel corrupts the memory store by construction. This is not
  hypothetical: spike run 1's normalizer rendered every `type: "user"` record as `USER:`, and the model
  extracted a `frontend-design` skill preamble as an explicit user decision with a quote that passed
  the fabrication check, because the quote was real and only the attribution was wrong. Claude Code's
  `isMeta` flag plus a prefix table fixed it (all 11 injected lines correctly classified across three
  fixtures). Harness records OF human actions — `[Request interrupted by user]` — are retained as
  tier-3 observed events; they are evidence, they are just not quotable as instructions.

### INV-9 — Observed-session content never leaves this machine
area: ["palaver/**/*.py", "pyproject.toml", "tests/fixtures/**"]
gate_test: tests/test_invariants.py::test_no_outbound_http_clients
gate_test: tests/test_invariants.py::test_the_http_client_gate_does_not_see_dependencies
gate_test: tests/test_invariants.py::test_the_runtime_dependency_set_is_an_allowlist
gate_test: tests/test_fixture_lint.py::test_unclassified_record_fails
gate_test: tests/test_fixture_lint.py::test_every_file_under_the_committed_corpus_is_read
threshold: 3
rationale: The brief's first engineering preference and the reason the whole design tolerates a 4B
  model instead of a frontier one. Palaver's database aggregates the full unredacted content of every
  observed session across every project on the machine — a strictly higher-value target than any
  single agent's transcript, because it is the join of all of them.
  There are two ways that content can leave, and this invariant covers both, because a rule that
  closed only one of them would read as satisfied while the other stayed open:
  (1) **Sockets.** The only ones opened are `127.0.0.1` inference and the local MCP listener. No
  telemetry, no crash reporting, no model API that is not local. The database is gitignored and must
  never sit in a cloud-synced path.
  **Which layer the socket gate covers, stated plainly:** `test_no_outbound_http_clients` is a static
  AST scan of first-party source (`palaver/**`) and nothing else. It proves Palaver's own code
  constructs no outbound HTTP client. It does **not**, and cannot, constrain what a dependency does —
  task 6.1's `mcp` pulls `httpx2` transitively, so an outbound client is now installed and reachable
  in this environment for the first time. The scan's blind spot is pinned by
  `test_the_http_client_gate_does_not_see_dependencies` rather than left to be discovered. What covers
  the dependency layer instead is an allowlist: `test_the_runtime_dependency_set_is_an_allowlist`
  compares `pyproject.toml`'s declared runtime dependencies against a named set, each entry carrying
  the reason that package may open sockets on Palaver's behalf. Adding a dependency therefore fails
  the gate until it is declared and justified. The claim that buys is not "dependencies are safe" —
  no test can vendor an opinion about every transitive package — but "no dependency arrives
  unreviewed", which is checkable and is what actually controls the risk. A passing source scan
  means "Palaver does not phone home", never "nothing in this process can".
  (2) **Git.** Test fixtures are transcripts, so a committed fixture is an export. A record ships
  only if it matches a structural shape carrying no free text, or its free-text payload was replaced
  with prose written for the fixture; `palaver fixture-lint` enforces that as an allowlist and fails
  on any record it cannot classify. A fixture pushed to a public remote leaves the machine as surely
  as an HTTP POST does, and unlike a POST it cannot be recalled.
  **Two gate tests, because the fixture surface has two ways to fail.**
  `test_unclassified_record_fails` proves a record the allowlist does not cover is rejected. It says
  nothing about whether that record was ever handed to the allowlist, and until 2026-08-15 seven of
  the twenty-eight committed files were not: discovery was a `*.jsonl` glob, so the gate reported
  clean over a subset and read as clean over the corpus.
  `test_every_file_under_the_committed_corpus_is_read` closes the other half — the set of files the
  linter opens equals the set present, and an extension no checker claims is a rejection rather than
  a skip. Caught and read are separate claims; one gate test cannot carry both.
