"""Project and session scoping for memory reads (task 2.3).

Project scope and session scope are different questions, and a caller who
asks one but is silently answered the other cannot tell the difference from
the response alone — a project-scoped answer returned where a session-scoped
one was meant reads exactly as authoritative as the real thing. So
`read_memories` takes exactly one of `project` (a `projects.name`) or
`session` (a `sessions.id`) — never both, never neither, and never a
default. Phase 6's MCP read tools (task 6.1) apply the identical rule at
their own boundary — `scope: {project: <name>} OR {session: <id>}` — for the
same reason: a tool caller has the same opportunity to conflate the two that
any caller of this helper does. This module is what that tool layer calls
into.

**Project scope reads `memories.project_id` directly, never through a join
on `sessions`.** `memories.session_id` is nullable — `write_memory` accepts
`session_id=None` for a project-level observation with no single owning
session (see `palaver/memory/write.py`,
`test_write_memory_defaults_session_id_and_supersedes_to_null`). A `JOIN
sessions ON memories.session_id = sessions.id` would silently drop every
such row from a project-scoped read — exactly the quietly-wrong answer this
module exists to prevent, not merely an edge case of it.

**An unknown scope target raises, rather than returning an empty list.** A
project name or session id that names no row at all is a different failure
from "that project/session exists and currently has zero memories";
collapsing the two into the same empty response hides a likely caller typo
behind a result that looks like a legitimate, if boring, answer. Both
branches resolve the scope target against `projects`/`sessions` first and
raise `LookupError` if it does not exist, before ever querying `memories`.

Supersession is not filtered here — a superseded memory's row is returned
alongside its successor, same as any other row. Deriving current status
from the `supersedes` link is task 2.4's `superseded_memories` view, layered
on top of this module's read, not a filter this module applies itself.
"""

from __future__ import annotations

import sqlite3

_MEMORY_COLUMNS = (
    "id",
    "project_id",
    "session_id",
    "statement",
    "origin",
    "tier",
    "supersedes",
    "created_at",
)


def read_memories(
    conn: sqlite3.Connection,
    *,
    project: str | None = None,
    session: int | None = None,
) -> list[dict]:
    """Read memories in exactly one scope: an entire project, or one session.

    Args:
        conn: Open connection to a database migrated to at least schema
            version 1.
        project: A `projects.name` to scope to. Returns every memory whose
            `project_id` matches that project's id, across every session of
            that project — including project-level memories that carry no
            `session_id` at all.
        session: A `sessions.id` to scope to. Returns only memories whose
            `session_id` matches exactly this session — never a sibling
            session of the same project, and never a project-level memory
            with no session_id.

    Returns:
        Matching `memories` rows, oldest first (`ORDER BY id`), each as a
        dict with keys `id`, `project_id`, `session_id`, `statement`,
        `origin`, `tier`, `supersedes`, `created_at`. Superseded rows are
        included — see the module docstring.

    Raises:
        ValueError: Called with neither `project` nor `session`, or with
            both. Scope is never defaulted and never ambiguous.
        LookupError: `project` names no row in `projects`, or `session`
            names no row in `sessions`.
    """
    if (project is None) == (session is None):
        raise ValueError(
            "read_memories requires exactly one of `project` or `session`, never both and "
            f"never neither (got project={project!r}, session={session!r})"
        )

    columns = ", ".join(_MEMORY_COLUMNS)

    if project is not None:
        project_row = conn.execute("SELECT id FROM projects WHERE name = ?", (project,)).fetchone()
        if project_row is None:
            raise LookupError(f"no project named {project!r}")
        cursor = conn.execute(
            f"SELECT {columns} FROM memories WHERE project_id = ? ORDER BY id",
            (project_row[0],),
        )
    else:
        session_row = conn.execute("SELECT id FROM sessions WHERE id = ?", (session,)).fetchone()
        if session_row is None:
            raise LookupError(f"no session with id {session!r}")
        cursor = conn.execute(
            f"SELECT {columns} FROM memories WHERE session_id = ? ORDER BY id",
            (session,),
        )

    return [dict(zip(_MEMORY_COLUMNS, row, strict=True)) for row in cursor.fetchall()]
