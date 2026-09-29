"""The deterministic Signals record, its coverage type, and the coverage threshold."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from .ranges import Tri

#: The share of a source's sampled sessions a signal must be determinable
#: for before a status derived from that signal may be asserted for that
#: source (task 7.3). Below it, `derive_status()` returns `UNKNOWN`.
#:
#: 50% is a policy constant, not a measurement, and it is deliberately
#: coarse. The claim it encodes is structural rather than statistical: a
#: derivation that cannot read a signal for most of a source's sessions has
#: not been shown to fit that source's format at all, and the minority it
#: does answer for are as likely to be shape coincidences — a record that
#: happens to look Claude-Code-like — as observations. Coverage is not
#: accuracy (see `palaver.cli.diagnose`), so this threshold cannot say a
#: source's answers are *right*; it can only refuse to pass on answers from
#: a derivation that demonstrably does not generalize. Every caller may pass
#: its own value.
DEFAULT_COVERAGE_THRESHOLD = 50.0

#: Per-signal coverage for one source, as percentages keyed by the names in
#: `SIGNAL_NAMES`. `palaver.cli.diagnose.CoverageReport` produces these.
SourceCoverage = Mapping[str, float]


@dataclass(frozen=True)
class Signals:
    """The deterministic signal set `derive_status()` reads.

    Every field is required. There are no defaults, because a default would
    let a caller that forgot a signal silently assert something about it —
    and the only safe default (`UNKNOWN`) would then hide the omission
    behind a status that looks deliberate. A producer that cannot determine
    a signal must say so by passing `Tri.UNKNOWN` explicitly.

    Nothing in this set comes from a model, and task 3.6 did not change
    that. Refinement content arrives at `derive_status()` as a separate
    keyword-only `Extraction`, never as a signal: a signal is something
    Palaver observed for itself, and mixing a model's claim in among them
    would put model output behind the same `Tri` a rule trusts absolutely.
    The model supplies the content, Python still owns the rule list (INV-7).

    Attributes:
        source_readable: `TRUE` when the session store was opened and read
            without error on this observation. `FALSE` when the adapter
            raised, the path was missing, or permission was denied.
        signal_records_parsed: `TRUE` when every record the other signals
            were actually derived from decoded successfully. Scoped
            deliberately to *those* records — the message-bearing tail the
            turn boundary and tool-outcome signals read — and not to every
            record in the file. Claude Code transcripts are read from offset
            zero by `last_message_bearing_record`, so a file-wide definition
            would let one corrupt line anywhere in a session's history pin
            that session to `UNKNOWN` for the rest of its life. A trailing
            partial line from an in-flight write is *not* a corruption
            condition here: `read_complete_records` already withholds it and
            re-reads it whole, so it never reaches a parser.
        unresolved_tool_error: `TRUE` when the most recent tool outcome in
            the observed window is an error. A later successful tool outcome
            clears it back to `FALSE` — this is a claim about the session's
            latest outcome, not about whether an error ever occurred.
        agent_turn_ended: `TRUE` when the agent handed control back to the
            human, `FALSE` when it still holds the turn (including
            mid-`tool_use`). Derived structurally by
            `palaver.observer.turn_boundary` (task 1.6) from the last
            message-bearing record's role read *through* INV-8's channel
            classification, plus `tool_use`/`tool_result` pairing. Consumed
            here, never derived here.
    """

    source_readable: Tri
    signal_records_parsed: Tri
    unresolved_tool_error: Tri
    agent_turn_ended: Tri

    def __post_init__(self) -> None:
        """Reject any field that is not a `Tri`.

        A raw `bool` passed where a `Tri` belongs is the collapse this
        module exists to prevent, and it would otherwise fail silently:
        `True is Tri.TRUE` is `False`, so every rule would skip and the
        caller would get a plausible-looking `UNKNOWN`.

        Raises:
            TypeError: If any field's value is not a `Tri` member.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, Tri):
                raise TypeError(
                    f"Signals.{field.name} must be a Tri member, got "
                    f"{type(value).__name__}: {value!r}"
                )


#: Every signal name in `Signals`, in rule-evaluation order. Task 1.6's
#: `palaver diagnose --coverage` reports one coverage percentage per entry.
SIGNAL_NAMES: tuple[str, ...] = tuple(field.name for field in fields(Signals))
