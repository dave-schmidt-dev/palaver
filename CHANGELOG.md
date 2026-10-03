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
- README and `ledger.yaml` now cite `tests/test_invariants_charter.py`, the file the pre-split `tests/test_invariants.py` became.
- Fixed companion session joining for alternate `CODEX_HOME` directories by discovering roots from live agent open file descriptor paths (`tmp/arg0/codex-arg0*/.lock` and canonical rollout stores) with process identity revalidation and refusal on conflicting homes or stale PIDs.
