"""Bound a read response by the bytes it becomes, not by the rows it holds.

Task 6.2's premise, as planned, was that `mcp` 2.0.0's 4 MiB body limit
truncates a long recall. Measured on 2026-08-15 against a bare `MCPServer`
and a real `streamable_http_client`, that is not what happens, and the real
behaviour is worse:

* `DEFAULT_MAX_REQUEST_BODY_SIZE` (4 MiB) belongs to
  `RequestBodyLimitMiddleware`, which inspects **incoming POST bodies only**.
  A 5 MiB tool *argument* gets a clean `413 Content Too Large`. A 5 MiB tool
  *result* never reaches that middleware at all.
* What actually bites is `httpx2._config.DEFAULT_MAX_EVENT_SIZE_BYTES`
  — **1 MiB, client-side, per SSE event** — which `mcp.client.streamable_http`
  takes as its default when it builds an `EventSource`. Bisected: a result of
  1 048 000 bytes arrives, 1 048 576 does not.
* Over the limit, the client raises `SSEError` inside its parse loop and the
  caller sees `MCPError: SSE stream ended without a response`. Nothing in
  that message mentions size. It reads like a network fault or a crashed
  server, so a person debugs the wrong thing — the INV-7 shape exactly, which
  is why the bound is asserted here, before the response leaves the tool,
  where the cause can be named and the remedy (page again) can be handed out.

The 1 MiB figure is the most restrictive client limit **we have measured**,
using the Python SDK. Claude Code and Codex CLI are the real consumers and
may use a different SDK with a different cap or none. So `RESPONSE_BUDGET`
is Palaver's own budget, chosen under the one limit we can point at, not a
claim about every client.

**Cursors are keyset, never offset.** `palaver observe` writes to the same
database an agent is paging through. Under `LIMIT/OFFSET`, one insert
between page one and page two shifts every later offset and a row is
silently skipped — the reader gets a complete-looking result that is missing
something, with no way to tell. A keyset cursor carries the last `id` it
saw, so a concurrent insert lands after the reader rather than under it.
`read_memories` orders by `memories.id`, which is unique, so the ordering is
total and no tiebreaker is needed; a ranked (bm25) query would need one,
since score ties are common.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

#: The measured per-event ceiling: `httpx2._config.DEFAULT_MAX_EVENT_SIZE_BYTES`,
#: applied client-side by `EventSource` and taken as the default by
#: `mcp.client.streamable_http`. Recorded as the thing being stayed under, so
#: a future reader can re-check the constant rather than trust this comment.
MAX_SSE_EVENT_BYTES = 1024 * 1024

#: What a single response is allowed to serialize to. Deliberately well under
#: `MAX_SSE_EVENT_BYTES`: the gap absorbs an SDK that frames slightly
#: differently than `wire_size` models, a client with a lower cap than the
#: one measured, and the JSON-RPC envelope growing a field. Paging costs a
#: round trip; guessing the ceiling exactly costs a failure that names
#: nothing.
RESPONSE_BUDGET = 768 * 1024

#: Bumped if the cursor payload's shape ever changes, so an old cursor is
#: refused by name instead of being misread as a new one.
_CURSOR_VERSION = 1

#: A stand-in for the JSON-RPC request id in `wire_size`'s envelope, wide
#: enough that no real session's id is wider. The actual id is not knowable
#: from inside a tool — it belongs to the transport — so it is modelled at
#: its maximum. Erring long costs 19 bytes; erring short means the estimate
#: says a page fits and the wire disagrees.
_WIDEST_REQUEST_ID = 9_999_999_999_999_999_999


class CursorError(ValueError):
    """A cursor that is malformed, stale, or from a different question."""


class RowTooLargeError(ValueError):
    """One row alone exceeds the budget, so paging cannot rescue it."""


def _canonical(value: Any) -> str:
    """Serialize deterministically, so equal scopes fingerprint equally."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scope_fingerprint(scope: Mapping[str, Any]) -> str:
    """Identify the question a cursor belongs to.

    Truncated to 16 hex characters. This is a mistake detector, not an
    authentication token: a caller who forges one reaches only rows they
    could have asked for directly with that scope. What it does prevent is
    the quiet error — a cursor from a project-wide recall replayed against a
    session scope, which under a bare offset would return a page of the
    wrong session's memories and look entirely normal.
    """
    return hashlib.sha256(_canonical(scope).encode()).hexdigest()[:16]


def encode_cursor(scope: Mapping[str, Any], after_id: int) -> str:
    """Build the opaque token a caller passes back to get the next page.

    Args:
        scope: The resolved scope this page answered, echoed in the response.
        after_id: The `id` of the last row on the page just returned.

    Returns:
        A URL-safe base64 token. Opaque by construction: callers that parse
        it will break, which is the point — the encoding is ours to change.
    """
    payload = _canonical(
        {"v": _CURSOR_VERSION, "s": _scope_fingerprint(scope), "a": after_id}
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(token: str, scope: Mapping[str, Any]) -> int:
    """Read a cursor back, refusing one that belongs to a different question.

    Args:
        token: The `next_cursor` from a previous response.
        scope: The resolved scope of the *current* call.

    Returns:
        The `after_id` to resume from.

    Raises:
        CursorError: The token is not decodable, carries a version this build
            does not understand, or was issued for a different scope. Every
            one of these is refused rather than repaired: a cursor that is
            silently ignored restarts the caller at page one without saying
            so, and a cursor honoured across scopes answers a question nobody
            asked.
    """
    if not isinstance(token, str) or not token.strip():
        raise CursorError("cursor must be a non-empty string")

    padded = token + "=" * (-len(token) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CursorError(
            f"cursor is not a cursor this server issued ({exc}). "
            "Pass back the `next_cursor` from the previous response verbatim, "
            "or omit it to start from the beginning."
        ) from exc

    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise CursorError(
            f"cursor version {payload.get('v') if isinstance(payload, dict) else None!r} "
            f"is not {_CURSOR_VERSION}; start the query again without a cursor"
        )

    if payload.get("s") != _scope_fingerprint(scope):
        raise CursorError(
            "cursor was issued for a different scope. A cursor is bound to the "
            "question it answered; reusing one across scopes would return rows "
            "from the wrong scope in a response that looks entirely normal. "
            "Start the new scope without a cursor."
        )

    after_id = payload.get("a")
    if not isinstance(after_id, int) or isinstance(after_id, bool):
        raise CursorError(f"cursor carries a non-integer position {after_id!r}")
    return after_id


def wire_size(payload: Mapping[str, Any]) -> int:
    """The bytes of the SSE event this tool result becomes.

    Models what the client's 1 MiB budget is actually measured against: the
    tool's dict serialized to JSON, that JSON embedded as a *string* in
    `content[0].text` (so every quote, newline and backslash is escaped a
    second time), the JSON-RPC envelope around it, and the `event:`/`data:`
    SSE framing.

    The double encoding is the part an item count cannot stand in for.
    Measured over Palaver's own source prose, the event is 1.077x the tool's
    JSON; over a pathological all-quotes payload it is 1.851x. A row budget
    calibrated on the first would be 70% over on the second.

    The request `id` is modelled at its widest rather than at `0`. A live
    session's ids increment, so a one-digit placeholder makes this function
    *under*-report by a few bytes late in a long session — the one direction
    an estimator of a ceiling must never err in. `_WIDEST_REQUEST_ID` costs
    19 bytes against a 256 KiB gap to the real limit, which is the right
    trade for never being optimistic.

    Args:
        payload: The dict the tool is about to return.

    Returns:
        Bytes on the wire, including framing.
    """
    inner = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    envelope = {
        "jsonrpc": "2.0",
        "id": _WIDEST_REQUEST_ID,
        "result": {"content": [{"type": "text", "text": inner}], "isError": False},
    }
    framed = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return len(f"event: message\r\ndata: {framed}\r\n\r\n".encode())


def paginate(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    scope: Mapping[str, Any],
    items_key: str,
    budget: int = RESPONSE_BUDGET,
) -> dict:
    """Cut `rows` at the last one that fits, and hand back a resume token.

    Args:
        rows: `(cursor_id, item)` pairs, ordered by `cursor_id` ascending and
            already filtered to those after any incoming cursor. The id is
            passed alongside the item rather than read out of it because
            `palaver_sessions` must **not** emit its rowid: this module's
            sibling `tools_read` refuses a rowid as a session identifier
            precisely so a caller never holds one, and handing one back in
            every response would undo that.
        scope: The resolved scope, echoed into the response and bound into
            the cursor.
        items_key: The response key the items go under (`memories`,
            `sessions`).
        budget: Maximum serialized event bytes; `RESPONSE_BUDGET` by default.

    Returns:
        `{"scope": ..., <items_key>: [...], "next_cursor": str | None}`.
        `next_cursor` is `None` exactly when the last row is included, so a
        caller loops until it is `None` rather than until a page comes back
        short — a full page can still be the last one.

    Raises:
        RowTooLargeError: The first row does not fit on its own. Paging
            cannot help, so this is raised rather than returning an empty
            page with a cursor that would loop forever.
        AssertionError: The assembled page exceeds `budget`. Unreachable
            unless the incremental sizing below has drifted from `wire_size`;
            it is checked on the real serialized payload because that is the
            only check that cannot be fooled by an estimator bug.

            **No test kills a mutant that deletes this assertion, and none
            can.** The incremental sizing is exact today, so the branch never
            fires and its removal is unobservable. It is kept as a guard
            against a future change to either `wire_size` or the per-row
            accounting drifting apart silently — the case where the estimate
            says a page fits and the wire says otherwise. Recorded here
            rather than reported as a killed mutant it is not.
    """
    # Reserve for a cursor rather than for `null`: a page that fills to the
    # budget and *then* discovers it needs a ~60-byte token would go over.
    # A large id gives the longest token this encoding produces.
    empty: dict[str, Any] = {
        "scope": dict(scope),
        items_key: [],
        "next_cursor": encode_cursor(scope, 2**62),
    }
    overhead = wire_size(empty)

    # JSON string escaping is per-character, so the escaped length of a
    # concatenation is the sum of its parts. That makes an exact running
    # total possible in one pass, instead of re-serializing the whole page
    # for every candidate row.
    used = overhead
    page: list[Mapping[str, Any]] = []
    last_id = 0
    for cursor_id, item in rows:
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        # +1 for the comma between array elements; the escaped cost of the
        # item is what it contributes once embedded and re-escaped.
        cost = _escaped_size(encoded) + (1 if page else 0)
        if used + cost > budget:
            if not page:
                raise RowTooLargeError(
                    f"row id={cursor_id} serializes to {cost} bytes, over the "
                    f"{budget}-byte response budget on its own. Paging cannot split a "
                    "single row; this row has to be shortened at the source."
                )
            break
        used += cost
        page.append(item)
        last_id = cursor_id

    complete = len(page) == len(rows)
    response: dict[str, Any] = {
        "scope": dict(scope),
        items_key: list(page),
        "next_cursor": None if complete else encode_cursor(scope, last_id),
    }

    actual = wire_size(response)
    assert actual <= budget, (  # noqa: S101 - a wrong answer must not reach the wire
        f"assembled page is {actual} bytes, over the {budget}-byte budget; "
        f"incremental sizing estimated {used}"
    )
    return response


def _escaped_size(encoded: str) -> int:
    """Bytes this JSON fragment costs once re-escaped inside a JSON string.

    `wire_size` embeds the tool's whole JSON as a string, so each fragment is
    escaped a second time. `json.dumps` of the fragment yields exactly that
    escaping; the `- 2` drops the surrounding quotes it adds, which belong to
    the containing string rather than to the fragment.
    """
    return len(json.dumps(encoded, ensure_ascii=False).encode()) - 2
