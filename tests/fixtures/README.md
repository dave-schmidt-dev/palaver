# Fixture corpus

Eleven hand-authored Claude Code session stores, each labelled with a ground
truth that a reader can check against the file itself. This is the *accuracy*
counterpart to `palaver diagnose --coverage`: coverage counts the sessions a
signal was determinable for, and a uniformly wrong classifier scores 100% at
it. Only labelled fixtures can say whether an answer was right.

## Provenance: nothing here was observed

**Every record in this directory was written by hand. None of it was copied,
sampled, or paraphrased from a real agent session store.** All prose is
invented, about invented work, and deliberately too thin to be anything a
person actually typed at an agent.

This repository is public, so a fixture pushed here leaves the machine as
surely as an HTTP POST does, and unlike a POST it cannot be recalled. That is
INV-9's git clause, and `palaver fixture-lint` enforces it as an allowlist:

```
uv run palaver fixture-lint tests/fixtures
```

A record ships only if it matches a structural shape the linter declares *and*
every free-text payload it carries is an exact member of the linter's
phrasebook. A record it cannot classify is a failure, not a warning. Adding
prose to this corpus therefore requires editing `palaver/cli/fixture_lint.py`,
which is the point — the gate is that edit, and the edit shows up in review.

The fixtures also carry no `uuid`, `parentUuid`, `timestamp`, `cwd`,
`gitBranch`, or `version` keys, and their `sessionId` values are all
`fixture-*` rather than UUIDs. Nothing in the observer reads those fields, and
leaving them out means a record pasted here from a real transcript is rejected
on its shape before its prose is ever considered.

## Shape fidelity is not this corpus's job

These files are minimal: they carry the fields the Claude Code adapter and the
turn-boundary walk actually read, and nothing else. Whether that shape still
matches what Claude Code writes today is measured locally by
`palaver diagnose --coverage`, which reads real stores, commits nothing, and
exists precisely so this corpus does not have to be a transcript to be useful.

## How to read an entry

Each entry states the structural facts that fix its status — which record is
the last message-bearing one, whether any `tool_use` is unresolved, what the
latest tool outcome was — so the label can be checked by reading the fixture,
without trusting this file. "Constructed to be `WORKING`" is authorial intent,
is circular, and is rejected by
`tests/test_fixture_lint.py::test_metadata_derivation_rejects_authorial_intent`.

Field meanings:

* **ground truth** — what is actually true of the session, expressed in Phase
  1's vocabulary (`PHASE1_STATUS_RANGE`).
* **derived today** — what `derive_status(derive_signals(records).signals)`
  returns right now. Measured, not asserted. Where it differs from ground
  truth there is a **divergence** field saying why, and that difference is a
  known defect rather than a licence to relabel the fixture.
* **phase 3 target** — the finer label §4.2 stages for Phase 3.6, once
  semantic extraction exists. Not reachable today; recorded so the corpus is
  ready when it is.
* **boundary basis** — which of `BASIS_NAMES` carried the decision.
* **latest tool outcome** — what `_unresolved_tool_error` finds scanning back
  through the window. Only this decides rule 3.

Statuses were measured with `store_mtime=None`. mtime never moves a status —
it is corroboration only — but a checked-out file's mtime is its checkout
time, so withholding it keeps the corroboration column reproducible.

---

### `working-mid-tool-use.jsonl`

- **case:** MID_TOOL_USE
- **ground truth:** WORKING
- **derived today:** WORKING
- **phase 3 target:** WORKING
- **boundary basis:** unresolved_tool_use
- **last message-bearing record:** line 2, an `assistant` record
- **unresolved tool_use:** yes — `tu-1`, never answered
- **latest tool outcome:** none in the window
- **channel:** line 1 is human-channel; `isMeta` is false and its text matches no injected prefix
- **derivation:** The last message-bearing record is line 2, an `assistant`
  record whose only content block is a `tool_use` (`Bash`). No `tool_result`
  block for `tu-1` appears after it anywhere in the file, so the call is
  unresolved and the backwards walk stops there with `ended = FALSE`. The
  window holds no tool outcome at all, so `unresolved_tool_error` is `FALSE`
  by observed absence and rule 3 does not fire; rule 4 returns WORKING.

### `working-tool-result-pending.jsonl`

- **case:** WORKING
- **ground truth:** WORKING
- **derived today:** WORKING
- **phase 3 target:** WORKING
- **boundary basis:** tool_result_pending
- **last message-bearing record:** line 3, a `user` record carrying a `tool_result`
- **unresolved tool_use:** no — `tu-1` on line 2 is answered on line 3
- **latest tool outcome:** success (`is_error: false`)
- **channel:** line 3 is a tool outcome, not a turn; it is excluded structurally, before channel classification
- **derivation:** The last message-bearing record is line 3, a `user` record
  whose only content block is a `tool_result`. The walk tests for
  `tool_result` blocks structurally and before consulting `classify_channel`,
  which matters because a tool result carries no text and would otherwise fall
  through the prefix table and be classified as a human turn by accident.
  Having found one, the walk stops with `ended = FALSE`: an outcome came back
  and the agent consumes it. The `tool_use` on line 2 is resolved by that same
  result and it carries `is_error: false`, so rule 3 does not fire and rule 4
  returns WORKING.

### `waiting-for-user-reply.jsonl`

- **case:** WAITING_FOR_USER
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** QUESTION
- **phase 3 target corrected 2026-08-14:** was `WAITING_FOR_USER`. Task 3.6 surfaced that this row
  disagreed with `tests/fixtures/eval/labels.json`, which marks the same fixture `expect_question: true`.
  The fixture decides it: line 4, the last message-bearing record, is `should i also rename the helper?`
  — a direct question to the user, not merely an ended turn with work outstanding. Task 3.6's ordered
  refinement puts `QUESTION` ahead of `WAITING_FOR_USER`, so this row was the stale one. The file name
  predates the three-way split and is left alone rather than churning a path two test modules reference.
- **boundary basis:** assistant_final
- **last message-bearing record:** line 4, an `assistant` record
- **unresolved tool_use:** no — `tu-1` on line 2 is answered on line 3
- **latest tool outcome:** success (`is_error: false`)
- **channel:** lines 1 and 3 are `user` records; line 1 is human-channel, line 3 is a tool outcome
- **derivation:** The last message-bearing record is line 4, an `assistant`
  record carrying one `text` block and no `tool_use` block. Everything before
  it is resolved, and nothing message-bearing follows, so `ended = TRUE` and
  rule 5 returns AWAITING_HUMAN. Note that this file is structurally identical
  to `finished-session.jsonl` minus its bookkeeping tail: the difference
  between "the agent asked something" and "the agent finished" lives entirely
  in prose, which Phase 1 does not read. That is exactly why AWAITING_HUMAN is
  the union label and why WAITING_FOR_USER waits for extraction.

### `question-askuserquestion-unresolved.jsonl`

- **case:** QUESTION
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** QUESTION
- **boundary basis:** unresolved_human_blocking_tool_use
- **last message-bearing record:** line 2, an `assistant` record
- **unresolved tool_use:** yes — `tu-1`, an `AskUserQuestion` call, never answered
- **latest tool outcome:** none in the window
- **channel:** line 1 is human-channel
- **derivation:** The last message-bearing record is line 2, an `assistant`
  record whose only content block is a `tool_use` named `AskUserQuestion`,
  with no `tool_result` for `tu-1` after it. `AskUserQuestion` is in
  `HUMAN_BLOCKING_TOOL_NAMES`, read from that block's `name`, so the walk
  treats the call as resolved by the human rather than by a `tool_result`
  and stops with `ended = TRUE`, returning AWAITING_HUMAN. An unresolved
  `AskUserQuestion` is not an agent that is busy — it is an agent that has
  stopped and put a prompt in front of its human, and the fix keys on the
  tool name rather than merely inverting the unresolved-`tool_use` rule: an
  unresolved `Bash` call in the same shape still returns WORKING.

### `finished-session.jsonl`

- **case:** FINISHED
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** DONE
- **boundary basis:** assistant_final
- **last message-bearing record:** line 4, an `assistant` record
- **unresolved tool_use:** no — `tu-1` on line 2 is answered on line 3
- **latest tool outcome:** success (`is_error: false`)
- **channel:** line 1 is human-channel; line 5 is bookkeeping, not a turn
- **derivation:** The last message-bearing record is line 4, an `assistant`
  record carrying one `text` block and no `tool_use` block. Line 5 is an
  `ai-title` bookkeeping record whose type is not in `MESSAGE_RECORD_TYPES`,
  so the backwards walk steps over it — real transcripts routinely close out
  on records like it. The `tool_use` on line 2 was answered on line 3, so
  nothing is outstanding: `ended = TRUE`, and rule 5 returns AWAITING_HUMAN.
- **note:** This is the fixture that proves silence is not read as completion.
  The work in it is finished by any human reading, the file has stopped
  growing, and the status is still AWAITING_HUMAN, not DONE. Structure can
  show that control returned to the human; it cannot show that the work is
  done. A confident wrong DONE tells the human a session needs nothing when it
  may be waiting on them, which is the most costly error this system can make,
  so DONE is outside `PHASE1_STATUS_RANGE` entirely.

### `error-tool-result.jsonl`

- **case:** ERROR
- **ground truth:** ERROR
- **derived today:** ERROR
- **phase 3 target:** ERROR
- **boundary basis:** tool_result_pending
- **last message-bearing record:** line 3, a `user` record carrying a `tool_result`
- **unresolved tool_use:** no — `tu-1` on line 2 is answered on line 3
- **latest tool outcome:** error (`is_error: true`)
- **channel:** line 1 is human-channel; line 3 is a tool outcome
- **derivation:** The latest tool outcome in the window is the `tool_result`
  on line 3 and it carries `is_error: true`, so `unresolved_tool_error` is
  `TRUE` and rule 3 returns ERROR. Rule ordering is what fixes this label, not
  the boundary: the boundary here is `ended = FALSE` from that same
  `tool_result` record, which would have produced WORKING had it been
  consulted first. Rule 3 sits ahead of rules 4 and 5 precisely so ERROR stays
  reachable for the sessions whose boundary *is* determinable — which is
  nearly all of them.

### `compaction-boundary.jsonl`

- **case:** COMPACT_BOUNDARY
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** DONE
- **boundary basis:** assistant_final
- **last message-bearing record:** line 5, an `assistant` record
- **unresolved tool_use:** no — this session makes no tool calls
- **latest tool outcome:** none in the window
- **channel:** lines 1 and 4 are human-channel; line 3 is a `system` record, not a turn
- **derivation:** The last message-bearing record is line 5, an `assistant`
  record with one `text` block and no `tool_use` block, so `ended = TRUE` and
  rule 5 returns AWAITING_HUMAN. The `system`/`compact_boundary` record on
  line 3 is not message-bearing, so the backwards walk steps over it rather
  than treating compaction as a turn or as a reason to stop reading. The
  signal window starts at line 4, the last human-channel `user` record, so
  everything before the compaction is outside this turn's evidence by
  construction.

### `ended-without-stop-hook.jsonl`

- **case:** NO_STOP_HOOK
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** DONE
- **boundary basis:** assistant_final
- **last message-bearing record:** line 2, an `assistant` record
- **unresolved tool_use:** no — this session makes no tool calls
- **latest tool outcome:** none in the window
- **channel:** line 1 is human-channel
- **corroboration:** UNKNOWN — no stop-hook record exists, and mtime is withheld
- **derivation:** The last message-bearing record is line 2, an `assistant`
  record with one `text` block and no `tool_use` block, so `ended = TRUE` and
  rule 5 returns AWAITING_HUMAN. No `system` record with subtype
  `stop_hook_summary` or `turn_duration` appears anywhere in the file, which
  is the ordinary case rather than a degraded one: stop-hook records exist
  only when the observed user has a Stop hook configured, so a boundary that
  needed them would be undefined for most sessions. The status here is
  identical to `ended-with-stop-hook.jsonl` because corroboration is reported
  and never applied.

### `ended-with-stop-hook.jsonl`

- **case:** STOP_HOOK_CONTROL
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** DONE
- **boundary basis:** assistant_final
- **last message-bearing record:** line 2, an `assistant` record
- **unresolved tool_use:** no — this session makes no tool calls
- **latest tool outcome:** none in the window
- **channel:** line 1 is human-channel; line 3 is a `system` record, not a turn
- **corroboration:** TRUE — the stop-hook record sits after the boundary record
- **derivation:** The last message-bearing record is line 2, an `assistant`
  record with one `text` block and no `tool_use` block, so `ended = TRUE` and
  rule 5 returns AWAITING_HUMAN — the same derivation as
  `ended-without-stop-hook.jsonl`, from the same structure. The
  `system`/`stop_hook_summary` record on line 3 sits after the boundary record
  at index 1, so it makes a claim about *this* turn and agrees with it,
  yielding `corroboration = TRUE` with no mtime involved. This fixture is the
  positive control for its sibling: same status, same basis, different
  corroboration, which is what shows the sibling's UNKNOWN comes from the
  missing hook rather than from the corroboration path being dead.

### `slash-command-after-reply.jsonl`

- **case:** SLASH_COMMAND
- **ground truth:** AWAITING_HUMAN
- **derived today:** AWAITING_HUMAN
- **phase 3 target:** DONE
- **boundary basis:** assistant_final
- **last message-bearing record:** line 2, an `assistant` record
- **unresolved tool_use:** no — this session makes no tool calls
- **latest tool outcome:** none in the window
- **channel:** line 3 is injected-channel by prefix, not by `isMeta`
- **derivation:** The file's last record, line 3, has `type: "user"` and
  `isMeta: false`, so the structural flag does not classify it; its text
  begins with `<command-name>`, which is in `INJECTED_TEXT_PREFIXES`, so
  `classify_channel` returns the injected channel and the walk looks through
  it. `isMeta` is false deliberately: with it true, `classify_channel` would
  short-circuit on the flag and the fixture would prove nothing about the
  prefix table. The last message-bearing record the walk stops on is therefore
  line 2, an `assistant` record with one `text` block and no `tool_use` block:
  `ended = TRUE`, and rule 5 returns AWAITING_HUMAN.
- **note (INV-8 tiers):** Line 3 is a harness record *of a human action*. The
  human typed `/status`; Claude Code expanded it into a `<command-name>`
  record. That makes it tier-3 observed evidence — something a person did, on
  the record — and not injected content in the sense `<system-reminder>` is,
  where no human acted at all and the text is the harness talking to the
  model. The distinction matters downstream: tier-3 evidence is quotable as
  *evidence*, never as an instruction, whereas a `<system-reminder>` is not
  even evidence of a human.
- **note (what the code does not yet do):** Phase 1 does not draw that
  distinction. `classify_channel` returns `CHANNEL_INJECTED` for both this
  record and a `<system-reminder>`, and nothing downstream separates them.
  That is the right answer for *status* — neither one is a human turn, so
  neither may hold the boundary, and the status here is correct — but the tier
  distinction itself is unmade today. It is recorded here so that the memory
  write path's provenance tiers have a labelled case to be checked against
  when they land.

### `bookkeeping-only.jsonl`

- **case:** UNKNOWN
- **ground truth:** UNKNOWN
- **derived today:** UNKNOWN
- **phase 3 target:** UNKNOWN
- **boundary basis:** no_conversational_record
- **last message-bearing record:** none
- **unresolved tool_use:** none present
- **latest tool outcome:** none in the window
- **channel:** no `user` or `assistant` record exists, so no channel applies
- **derivation:** There is no message-bearing record in this file: line 1 is
  `mode` and line 2 is `ai-title`, and neither type is in
  `MESSAGE_RECORD_TYPES`. The backwards walk runs off the start of the file
  without settling on anything, so `ended` stays `UNKNOWN` with basis
  `no_conversational_record`, and rule 6 returns UNKNOWN. That is a terminal
  answer, not a fallthrough default: nothing here supports a status claim, and
  the rule list deliberately has no branch that guesses a plausible one.
