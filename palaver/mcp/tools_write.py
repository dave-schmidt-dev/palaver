"""`palaver_correct`: the only write an MCP client can cause, and its gate.

Three properties, each guarding a different way this could go wrong.

**It never writes.** This module holds a `mode=ro` connection like every
other tool here, and posts the correction to the `palaver observe` daemon's
socket instead. If the daemon is down the tool fails and says so; it does
not open a second writer to get the job done. That refusal is the feature —
two processes writing one SQLite file is a correctness failure that nothing
downstream can detect afterwards.

**It supersedes, it does not edit.** The daemon writes a *new* memory at
tier 1 carrying `supersedes=<old id>`. The corrected row is not touched, so
"what did Palaver believe before I corrected it, and when did that change"
stays answerable. INV-4 is enforced by the schema's own triggers as well,
but the protocol never expresses an edit in the first place.

**Sign-off is elicitation, not annotation.** A destructive-hint annotation
is a label on a tool; the spec is explicit that clients must consider tool
annotations untrusted, so a safety property resting on one rests on
nothing. `ctx.elicit` is a real round trip to the client, and the write
happens only after an accept comes back.

Measured on 2026-08-15 against a live `streamable_http` server and a real
`streamable_http_client`, because whether a *server-initiated* request can
reach a client at all is a property of the transport, not of a signature:

* With a client that declares the elicitation capability, `ctx.elicit`
  round-trips from inside a tool handler. The client's standing GET stream
  is the back-channel.
* Without it, the SDK raises `MCPError: "Elicitation not supported"` —
  named, immediate, and not a hang. So a client that cannot ask its user
  gets a refusal naming the reason, and no correction is written. Failing
  closed here is the only safe direction: a memory rewritten at tier 1
  because nobody could be asked is exactly the confidently-wrong state
  INV-7 exists to prevent.

This repository is public. Nothing in this module is derived from a real
observed session (INV-9).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field

from palaver.memory.tiers import tier_name
from palaver.observer.socket import request


#: What the tool asks a human to confirm. Only primitive fields: the spec's
#: `PrimitiveSchemaDefinition` is all an elicitation schema may contain, and
#: the SDK rejects anything richer at render time.
class CorrectionSignOff(BaseModel):
    """The human's answer to "should this memory be corrected?"."""

    approved: bool = Field(
        description="Write the correction. Leave false to abandon it and change nothing."
    )
    note: str = Field(
        default="",
        description="Optional: why this correction is right, kept with the reply.",
    )


class WriteRefused(RuntimeError):
    """The correction was not written, and the caller is told why."""


def _predecessor(conn: sqlite3.Connection, memory_id: int) -> dict:
    """Read the memory about to be corrected, so the human sees what changes.

    Raises:
        LookupError: No such memory. Checked here, before the human is asked
            anything: a sign-off prompt quoting a memory that does not exist
            invites approval of a change that cannot happen.
    """
    row = conn.execute(
        "SELECT id, statement, origin, tier, created_at FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if row is None:
        raise LookupError(
            f"no memory with id {memory_id}. Ids come from palaver_recall; a memory "
            "cannot be corrected by its statement text."
        )
    successor = conn.execute(
        "SELECT id FROM memories WHERE supersedes = ?", (memory_id,)
    ).fetchone()
    if successor is not None:
        raise LookupError(
            f"memory {memory_id} was already superseded by memory {successor[0]}. "
            "Correct that one instead — a memory has at most one successor (INV-4), "
            "so correcting a superseded row would be refused by the store anyway."
        )
    return {
        "id": row[0],
        "statement": row[1],
        "origin": row[2],
        "tier": row[3],
        "tier_name": tier_name(row[3]),
        "created_at": row[4],
    }


async def correct(
    conn: sqlite3.Connection,
    db_path: Path,
    ctx,
    memory_id: int,
    statement: str,
) -> dict:
    """Supersede one memory with a corrected statement, after a human agrees.

    Args:
        conn: The tool's read-only connection, used to show the human what
            they are about to change.
        db_path: The store, which is also how the daemon's socket is found.
        ctx: The MCP request context, for the elicitation round trip.
        memory_id: `memories.id` of the row to correct, as returned by
            `palaver_recall`.
        statement: The corrected statement.

    Returns:
        `{"ok": True, "memory_id": <new id>, "supersedes": <old id>, ...}`.

    Raises:
        LookupError: No such memory, or it was already superseded.
        ValueError: `statement` is empty.
        WriteRefused: The human declined or cancelled, the client cannot
            elicit at all, or the daemon refused the request. Nothing was
            written in any of those cases.
        DaemonUnavailableError: No daemon is listening. Nothing was written.
    """
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("statement must be a non-empty string")

    before = _predecessor(conn, memory_id)

    # Quote both, in full. A prompt that says "correct memory 41?" asks the
    # human to approve something they cannot see, and an approval given
    # without the text is not sign-off, it is a keystroke.
    try:
        result = await ctx.elicit(
            f"Correct memory {memory_id}?\n\n"
            f"Recorded {before['created_at']} at tier {before['tier']} "
            f"({before['tier_name']}), origin {before['origin']}:\n"
            f"  {before['statement']}\n\n"
            f"Replace with, at tier 1 (user instruction):\n"
            f"  {statement.strip()}\n\n"
            "The original is kept and marked superseded; nothing is deleted.",
            CorrectionSignOff,
        )
    except MCPError as exc:
        # The SDK's own message is "Elicitation not supported" — accurate,
        # and it names neither what was refused nor what to do. A caller
        # reading it alongside a failed correction has to guess whether the
        # write happened. Say both.
        raise WriteRefused(
            f"memory {memory_id} was not corrected: this client cannot present a "
            f"sign-off prompt ({exc}). Palaver will not rewrite a memory at tier 1 "
            "without a human agreeing to it, so the correction was dropped and "
            "nothing was written. Use a client that supports MCP elicitation, or "
            "correct the memory with the palaver CLI."
        ) from exc

    if result.action != "accept":
        raise WriteRefused(
            f"the correction of memory {memory_id} was {result.action}ed, so nothing "
            "was written. The memory is unchanged."
        )
    if not result.data.approved:
        raise WriteRefused(
            f"the sign-off for memory {memory_id} came back with approved=false, so "
            "nothing was written. The memory is unchanged."
        )

    reply = request(
        db_path,
        {"op": "correct", "memory_id": memory_id, "statement": statement.strip()},
    )
    if not reply.get("ok"):
        raise WriteRefused(
            f"the daemon refused the correction: {reply.get('error')}: {reply.get('detail')}"
        )
    return {**reply, "note": result.data.note, "superseded_statement": before["statement"]}


#: Every write tool, by the name it is exposed under. Separate from
#: `READ_TOOLS` because the two register differently: a write tool takes the
#: request context (for the elicitation round trip) and the store path (for
#: the daemon socket), where a read tool needs neither.
WRITE_TOOLS = {"palaver_correct": correct}
