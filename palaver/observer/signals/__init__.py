"""Deterministic status signals and the ordered rule list that consumes them.

This module is the whole of INV-7: *status is computed from deterministic
signals; the model never sets it*. `derive_status()` owns the rule list in
Python, takes no model output of any kind, and is a pure function of a
`Signals` value.

Why the rule list lives here rather than in a prompt. Spike run 1 gave a 4B
model an explicit ordered rule list and it still returned `IDLE` for a
plainly-paused session. Spike run 2 isolated the reason: the model honours
rules whose predicates are *computed signals* and ignores rules whose
predicates are *its own generated fields*. A rule list is only as strong as
the thing evaluating it, so this one is evaluated by the interpreter.

**Without extraction the range is exactly `PHASE1_STATUS_RANGE`** —
`WORKING`, `AWAITING_HUMAN`, `ERROR`, `UNKNOWN`. That is not history: it is
the live contract for every caller that passes no `extraction`, which is
every caller in the tree today, and it is what a model outage degrades to.
With an extraction the range is `REFINED_STATUS_RANGE`, which adds `DONE`,
`WAITING_FOR_USER`, `QUESTION`, and `BLOCKED` (task 3.6). `IDLE` is
unreachable from `derive_status()` under either range and always will be:
it requires process liveness, which is not a signal and never becomes one.
That is a live contract about this function's inputs, not a staging note —
a status that is nameable, storable, and still never returned is a claim the
range test can actually falsify, whereas a status the enum cannot spell is
trivially unreachable.

**Task 5.2: liveness, and why it is a separate layer rather than a signal.**
`Liveness` and `apply_liveness()` add the two things that need to know
whether a *process* is there: `IDLE`, and the demotion of a stale `WORKING`
on a dead process. Liveness deliberately does not join `Signals`. Every
member of `Signals` is read out of the session store, so all four share one
provenance and one coverage measurement (`SIGNAL_NAMES`, task 7.3); process
liveness comes from the pane join (`palaver.ui.pane_join`), is available
only for sessions that have a live pane, and is missing for every session
observed headlessly. Adding it as a fifth field would change `SIGNAL_NAMES`,
and with it the meaning of every coverage percentage already measured, in
order to record a signal whose absence is normal rather than a defect.

So `apply_liveness()` runs *after* the rule list, composes with the coverage
gate rather than bypassing it, and touches exactly two statuses. Unobservable
liveness changes nothing: "turn ended, liveness unobservable" is
`AWAITING_HUMAN`, which is what that status has always meant, and is not a
fifth reason to return `UNKNOWN`.

The single named prohibition, from the brief: **do not equate lack of
terminal output with `DONE`.** An ended turn with no extraction is
`AWAITING_HUMAN`, never `DONE`. Structure can prove that control returned to
the human; it cannot prove the work is finished. `AWAITING_HUMAN` is the
union of `DONE`, `WAITING_FOR_USER`, and `QUESTION`, and it is exactly as
much as the structural signals support on their own. A confident wrong
`DONE` tells the human a session needs nothing when it may be waiting on
them — the most costly error this system can make.

**Task 3.6: refinement, and why it does not weaken any of the above.**
`derive_status()` takes an optional `Extraction` (task 3.4's dataclass, read
here and never modified) and splits the ended-turn branch into the brief's
three values, plus `BLOCKED`. Three properties keep INV-7 intact:

* The model supplies *content* — what work remains, what is blocking, what
  is unanswered — and never a status. `derive_status()` accepts no dict, no
  `status=` argument, and no `**kwargs`; the only refinement input is a
  typed `Extraction`, whose six fields do not include a status and cannot be
  made to. `extraction_from_model_payload()` is the boundary a raw model
  response crosses, and it refuses any status-like key outright, so a prompt
  regression that starts asking for a status is loud rather than silent.
* Refinement lives **inside** the ended-turn branch only. It never overrides
  an unreadable source, an unparsed record, an unresolved tool error, or an
  agent that still holds the turn. Model content refining a coarse structural
  answer is the design; model content overturning a deterministic one is the
  defect INV-7 names, and rule order is what forbids it.
* Absence degrades toward the coarse answer, never toward completion.
  `extraction=None` — the model was unavailable, timed out, or raised —
  returns `AWAITING_HUMAN`, i.e. exactly Phase 1 behaviour. So does an
  extraction that returned nothing about the fields that matter. The caller
  owns that degradation: catch whatever the extractor raises and pass `None`.
  This module opens no socket and calls no model, which is why it can make
  the guarantee at all.

`DONE` is the one status that requires positive evidence rather than the
absence of contrary evidence. The spike's rule was
`WAITING_FOR_USER if remaining_work else DONE`, which reads a *missing*
field as completion — every failed extraction becomes a finished session.
Here, `remaining_work=None` means "this pass had no opinion" (task 3.4's own
reading of the field) and yields `AWAITING_HUMAN`, while `remaining_work=""`
is an affirmative "nothing remains" and is the only thing that yields
`DONE`.

Field text is normalized by stripping whitespace and nothing else. No
`"none"`/`"n/a"`/`"TBD"` special-casing: inferring semantics from model prose
is unbounded, and it is the model deciding status by another route. Strip-only
is safe rather than lazy because every prose form it fails to recognize is
non-empty, and non-empty falls toward `WAITING_FOR_USER` or `BLOCKED` — never
toward `DONE`.

`UNKNOWN` is a first-class value, not an error case. When no signal supports
any status, `derive_status()` returns `UNKNOWN` rather than guessing. There
is no fallthrough branch that picks a plausible default.

Three-valued signals. Every signal here is a `Tri` — true, false, or
*unknown* — because for every one of them absence is a real, observable
condition, and "could not determine" is not the same claim as "determined it
to be false". Collapsing unknown into false is how `UNKNOWN` stops being
reachable and how a guess gets reported as fact, so `Tri` refuses to be used
in a boolean context at all: `if signals.agent_turn_ended:` raises
`TypeError` instead of silently reading `UNKNOWN` as truthy.

Note what the ternary domain does and does not promise. It is the range of
what a *producer* may honestly assert about a signal; it is not a
requirement that each of a signal's three values map to a distinct status.
Under the rule list below, only `agent_turn_ended` discriminates all three.
`source_readable` and `signal_records_parsed` treat `FALSE` and `UNKNOWN`
alike, because "we could not confirm we read the session" is as weak a
footing for a status claim as "we failed to read it". `unresolved_tool_error`
treats `UNKNOWN` and `FALSE` alike in the other direction, because `ERROR`
is a positive claim that requires positive evidence. Both collapses are at
the rule level, deliberate, and named in the rules' own docstrings — the
signal values themselves stay distinct so a caller (and
`palaver diagnose --coverage`, task 1.6) can tell the cases apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence  # noqa: F401
from dataclasses import dataclass, fields  # noqa: F401
from enum import Enum  # noqa: F401

from palaver.extract.persist import Extraction  # noqa: F401

from .liveness import Liveness as Liveness
from .liveness import apply_liveness as apply_liveness
from .liveness import derive_status_with_liveness as derive_status_with_liveness
from .payload import FORBIDDEN_PAYLOAD_KEYS as FORBIDDEN_PAYLOAD_KEYS
from .payload import REFINEMENT_PAYLOAD_KEYS as REFINEMENT_PAYLOAD_KEYS
from .payload import ExtractionPayloadError as ExtractionPayloadError
from .payload import ModelSuppliedStatusError as ModelSuppliedStatusError
from .payload import _normalized_key as _normalized_key
from .payload import _payload_text as _payload_text
from .payload import extraction_from_model_payload as extraction_from_model_payload
from .ranges import LIVE_STATUS_RANGE as LIVE_STATUS_RANGE
from .ranges import PHASE1_STATUS_RANGE as PHASE1_STATUS_RANGE
from .ranges import REFINED_STATUS_RANGE as REFINED_STATUS_RANGE
from .ranges import Status as Status
from .ranges import Tri as Tri
from .rules import StatusDerivation as StatusDerivation
from .rules import _has_content as _has_content
from .rules import _is_affirmatively_empty as _is_affirmatively_empty
from .rules import _refine_ended_turn as _refine_ended_turn
from .rules import derive_status as derive_status
from .rules import derive_status_for_source as derive_status_for_source
from .rules import derive_status_with_provenance as derive_status_with_provenance
from .rules import under_covered as under_covered
from .signal_set import DEFAULT_COVERAGE_THRESHOLD as DEFAULT_COVERAGE_THRESHOLD
from .signal_set import SIGNAL_NAMES as SIGNAL_NAMES
from .signal_set import Signals as Signals
from .signal_set import SourceCoverage as SourceCoverage

__all__ = [
    "DEFAULT_COVERAGE_THRESHOLD",
    "FORBIDDEN_PAYLOAD_KEYS",
    "LIVE_STATUS_RANGE",
    "PHASE1_STATUS_RANGE",
    "REFINED_STATUS_RANGE",
    "REFINEMENT_PAYLOAD_KEYS",
    "SIGNAL_NAMES",
    "ExtractionPayloadError",
    "Liveness",
    "ModelSuppliedStatusError",
    "Signals",
    "SourceCoverage",
    "Status",
    "StatusDerivation",
    "Tri",
    "_has_content",
    "_is_affirmatively_empty",
    "_normalized_key",
    "_payload_text",
    "_refine_ended_turn",
    "apply_liveness",
    "derive_status",
    "derive_status_for_source",
    "derive_status_with_liveness",
    "derive_status_with_provenance",
    "extraction_from_model_payload",
    "under_covered",
]
