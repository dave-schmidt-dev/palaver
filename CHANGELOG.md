# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The Claude Code and Codex summary reducers now share one bounded task-list builder, question-claim builder, and pending-question bootstrap in `palaver/summary/model.py`; snapshots and unknown-reason text are unchanged, and the item cap and reasons are pinned for both sources by `tests/test_summary_collections.py`.
- Moved the twelve identical per-module `_stderr_status` progress helpers into one `palaver.progress.stderr_status`; progress still goes to stderr and stdout stays the result channel.
- `palaver ui --pin` now writes the pane pin through `encode_pin`, the encoder paired with the reader's `parse_pin`, instead of a second hand-built JSON string; output bytes and error text are unchanged.

### Fixed
- Both launchd services now capture stdout/stderr through the shared rotating-log policy, including application startup failures and newline-free floods; ordinary CLI output and MCP HTTP behavior are unchanged. Existing agents require an owner-approved reinstall/reload to adopt the templates.
- The daemon now rolls back its shared SQLite connection after a refused write request and after a failed extraction, so a refused correction can no longer leave evidence rows pending that attach to the next memory or wedge every later commit.
- The write socket now caps a request at 1 MiB, bounds total receive time per request, and answers non-object, invalid-UTF-8 and deeply nested bodies with an error reply instead of crashing the observer.
- Schema migrations now run each migration and its `user_version` bump in one transaction, never overwrite an existing `.bak-N`, and write backups under a temporary name before renaming; a crash mid-migration can no longer destroy the only good backup.
- README and `ledger.yaml` now cite `tests/test_invariants_charter.py`, the file the pre-split `tests/test_invariants.py` became.
- Fixed companion session joining for alternate `CODEX_HOME` directories by discovering roots from live agent open file descriptor paths (`tmp/arg0/codex-arg0*/.lock` and canonical rollout stores) with process identity revalidation and refusal on conflicting homes or stale PIDs.
