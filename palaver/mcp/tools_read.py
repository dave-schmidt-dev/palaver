"""Scope-explicit read tools, and the identifier a caller actually holds.

Everything in this module exists to stop one failure: a tool answering a
different question from the one asked, in a response that reads exactly as
authoritative as the right answer would. There are two ways that happens,
and they need separate defences.

**Scope is never defaulted.** `palaver.memory.scope.read_memories` already
refuses to be called with neither `project` nor `session`, or with both.
`parse_scope` applies the identical rule one layer out, at the tool
boundary, because the tool boundary is where the mistake is actually made: a
caller who omits `scope` is not asking for "everything", they have simply
forgotten which question they were asking, and project memory returned where
session memory was meant is wrong in a way nothing downstream can detect.

**The session identifier is the one a human is shown, not a rowid.**
`read_memories(session=...)` takes `sessions.id`, an integer primary key —
correct for a storage helper, useless to an MCP caller. An agent or a person
knows a session by the `session_key` that `palaver status` and `palaver
inspect` print: `<project>/<session-id>`, or the bare `<session-id>`.
`resolve_session_id` converts one to the other.

It **never** falls back to treating the value as a rowid. That fallback is
the trap: session ids are opaque strings today, but nothing stops one from
looking like an integer, and an integer that is a valid rowid would then
resolve to a completely unrelated session and answer confidently. So a value
that does not match an `external_id` raises, even when it would have been a
perfectly good rowid.

Ambiguity is refused rather than guessed, for the same reason `sessions` is
keyed `UNIQUE (source, external_id)` rather than by `external_id` alone: two
sources can legitimately carry the same session id, and picking one would be
a coin flip presented as a lookup. The caller is told which candidates
matched and how to qualify.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from palaver.memory.scope import read_memories
from palaver.memory.tiers import tier_name

#: The scope keys a tool accepts. Exactly one, always.
SCOPE_KEYS = ("project", "session")


class ScopeError(ValueError):
    """A scope argument that names zero questions, or more than one."""


class SessionLookupError(LookupError):
    """A session identifier that resolves to no session, or to several."""


@dataclass(frozen=True)
class Scope:
    """Exactly one of a project name or a session key.

    Attributes:
        project: A `projects.name`, or `None`.
        session: A `session_key` (`<project>/<session-id>`) or bare session
            id, or `None`. Exactly one of the two is set; the constructor
            path (`parse_scope`) is what guarantees it.
    """

    project: str | None = None
    session: str | None = None


def parse_scope(scope: Any) -> Scope:
    """Validate a caller-supplied scope argument into a `Scope`.

    Args:
        scope: The tool's `scope` argument, expected to be a mapping with
            exactly one of the keys in `SCOPE_KEYS` and a non-empty string
            value.

    Returns:
        The parsed `Scope`.

    Raises:
        ScopeError: `scope` is not a mapping, carries no recognised key,
            carries both, names an unknown key, or carries a value that is
            not a non-empty string. Every one of these is a caller error
            that must surface as an error rather than as a default — a
            silently-defaulted scope answers the wrong question.
    """
    if not isinstance(scope, Mapping):
        raise ScopeError(
            f"scope must be a mapping with exactly one of {SCOPE_KEYS}, got {type(scope).__name__}"
        )

    unknown = sorted(set(scope) - set(SCOPE_KEYS))
    if unknown:
        raise ScopeError(f"unknown scope key(s) {unknown}; scope takes exactly one of {SCOPE_KEYS}")

    present = [key for key in SCOPE_KEYS if key in scope]
    if len(present) != 1:
        raise ScopeError(
            f"scope requires exactly one of {SCOPE_KEYS}, never both and never neither "
            f"(got {sorted(scope)}). Scope is never defaulted: a project-wide answer returned "
            "where a session was meant is indistinguishable from the right one."
        )

    key = present[0]
    value = scope[key]
    if not isinstance(value, str) or not value.strip():
        raise ScopeError(f"scope[{key!r}] must be a non-empty string, got {value!r}")

    return Scope(**{key: value.strip()})


def resolve_session_id(conn: sqlite3.Connection, session: str) -> int:
    """Resolve a caller-held session key to the `sessions.id` storage uses.

    Accepts what `palaver status` and `palaver inspect` print: a full
    `session_key` (`<project>/<session-id>`) or a bare `<session-id>`. The
    project half, when given, constrains the search; the session half is
    always matched against `sessions.external_id`.

    Args:
        conn: Open connection to a migrated database.
        session: The caller's session identifier.

    Returns:
        The matching `sessions.id`.

    Raises:
        SessionLookupError: Nothing matched, or several did. A rowid is
            never accepted as a fallback — see the module docstring for why
            that fallback is the specific hazard this function exists to
            avoid.
    """
    project_name, _, external_id = session.rpartition("/")
    if not external_id:
        raise SessionLookupError(f"{session!r} names no session id")

    query = (
        "SELECT s.id, p.name, s.source FROM sessions s "
        "JOIN projects p ON p.id = s.project_id WHERE s.external_id = ?"
    )
    params: tuple[Any, ...] = (external_id,)
    if project_name:
        query += " AND p.name = ?"
        params += (project_name,)

    matches = conn.execute(query + " ORDER BY s.id", params).fetchall()

    if not matches:
        raise SessionLookupError(
            f"no session matches {session!r}. Pass the session_key printed by "
            "`palaver status` (<project>/<session-id>) or its bare session id; "
            "an internal rowid is deliberately not accepted."
        )
    if len(matches) > 1:
        candidates = ", ".join(
            f"{name}/{external_id} (source={source})" for _, name, source in matches
        )
        raise SessionLookupError(
            f"{session!r} matches more than one session ({candidates}); "
            "pass the full session_key (<project>/<session-id>) to disambiguate"
        )
    return int(matches[0][0])


def recall(conn: sqlite3.Connection, scope: Any) -> dict:
    """Read the memories in one scope, each carrying its provenance tier.

    Args:
        conn: Open connection to a migrated database.
        scope: The caller's scope argument; see `parse_scope`.

    Returns:
        A dict with the resolved `scope` echoed back and a `memories` list.
        Every record carries both the numeric `tier` and its `tier_name`,
        so a caller weighing two statements against each other can see which
        one outranks the other without knowing Palaver's tier table.

    Raises:
        ScopeError: The scope was absent, doubled, or malformed.
        SessionLookupError: A session scope resolved to no session, or to
            several.
        LookupError: A project scope named no project.
    """
    parsed = parse_scope(scope)

    if parsed.project is not None:
        rows = read_memories(conn, project=parsed.project)
        echo = {"project": parsed.project}
    else:
        session_id = resolve_session_id(conn, parsed.session)
        rows = read_memories(conn, session=session_id)
        echo = {"session": parsed.session}

    return {
        "scope": echo,
        "memories": [{**row, "tier_name": tier_name(row["tier"])} for row in rows],
    }


def sessions(conn: sqlite3.Connection, scope: Any) -> dict:
    """List the sessions in one scope, so a caller can obtain a session key.

    This is the companion to `recall`'s session scope: a caller that holds
    only a project name has no way to learn the `session_key` a session
    scope requires, and inventing one is how a wrong id gets passed.

    Args:
        conn: Open connection to a migrated database.
        scope: The caller's scope argument; see `parse_scope`. A project
            scope lists every session of that project. A session scope
            returns the single session it resolves to, which is how a caller
            confirms an identifier means what they think before using it.

    Returns:
        A dict with the resolved `scope` echoed back and a `sessions` list,
        each carrying `session_key`, `source`, `started_at`, and `ended_at`.

    Raises:
        ScopeError: The scope was absent, doubled, or malformed.
        SessionLookupError: A session scope resolved to no session, or to
            several.
        LookupError: A project scope named no project.
    """
    parsed = parse_scope(scope)

    columns = "p.name, s.external_id, s.source, s.started_at, s.ended_at"
    if parsed.project is not None:
        project_row = conn.execute(
            "SELECT id FROM projects WHERE name = ?", (parsed.project,)
        ).fetchone()
        if project_row is None:
            raise LookupError(f"no project named {parsed.project!r}")
        rows = conn.execute(
            f"SELECT {columns} FROM sessions s JOIN projects p ON p.id = s.project_id "
            "WHERE s.project_id = ? ORDER BY s.id",
            (project_row[0],),
        ).fetchall()
        echo = {"project": parsed.project}
    else:
        session_id = resolve_session_id(conn, parsed.session)
        rows = conn.execute(
            f"SELECT {columns} FROM sessions s JOIN projects p ON p.id = s.project_id "
            "WHERE s.id = ?",
            (session_id,),
        ).fetchall()
        echo = {"session": parsed.session}

    return {
        "scope": echo,
        "sessions": [
            {
                "session_key": f"{name}/{external_id}",
                "source": source,
                "started_at": started_at,
                "ended_at": ended_at,
            }
            for name, external_id, source, started_at, ended_at in rows
        ],
    }


#: Every read tool, by the name it is exposed under. `server.build_server`
#: registers exactly these; a tool absent from here is not reachable, and one
#: present here without a scope argument would fail this module's own tests.
READ_TOOLS = {"palaver_recall": recall, "palaver_sessions": sessions}
