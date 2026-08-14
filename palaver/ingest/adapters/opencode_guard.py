"""Read-only, table-allowlisted access to OpenCode's SQLite session store (INV-3).

`~/.local/share/opencode/opencode.db` holds `account` and `credential` tables
with plaintext `access_token`/`refresh_token` values (verified against the
real store, `docs/research.md` section 3; never queried beyond the schema
names themselves). Palaver's OpenCode adapter (Task 7.2) has no use for
either table, so this module gives it two independent defenses rather than
relying on either one alone:

1. **Read-only at the OS/SQLite level.** `open_guarded_readonly` connects
   with SQLite's `mode=ro` URI parameter, which blocks writes to the
   database file.
2. **A table allowlist enforced through SQLite's own query authorizer.**
   `install_table_allowlist` registers an authorizer callback
   (`sqlite3.Connection.set_authorizer`) that runs while SQLite *compiles*
   a statement, before any row is read — it denies any `SQLITE_READ` of a
   table outside `ALLOWED_TABLES`, structurally, however the query
   references that table (direct, `JOIN`, subquery, any letter case). A
   read-only connection still permits `SELECT`, so mode=ro alone does not
   stop a query from reading `credential.access_token` into memory, a log,
   or (eventually) a model prompt — the one place a leaked token could
   leave the process. The allowlist is what stops that; `tests/test_invariants.py`
   proves the two layers are independent by stripping the allowlist from an
   otherwise-identical still-read-only connection and showing the same query
   then succeeds.

`ALLOWED_TABLES` covers exactly what `docs/research.md` identified as the
adapter's real per-turn data: `session` and `project` for identity, `message`
and `part` for turn content (`session_message` is dead — 0 rows in the real
store — and is deliberately not allowlisted). `account` and `credential` are
never added, here or by any future adapter over a store this project does
not own.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Tables the OpenCode adapter is permitted to read. `account` and
#: `credential` are deliberately absent — see the module docstring and INV-3.
ALLOWED_TABLES = frozenset({"session", "project", "message", "part"})


def readonly_uri(path: str | Path) -> str:
    """Build the SQLite read-only URI for `path`.

    Args:
        path: Filesystem path to the SQLite database.

    Returns:
        A `file:...?mode=ro` URI, resolved to an absolute path so the
        resulting URI is unambiguous regardless of the caller's cwd.
    """
    return f"file:{Path(path).resolve()}?mode=ro"


def _authorize_read_within_allowlist(action, arg1, arg2, dbname, source):
    """SQLite authorizer callback: deny reads of any table outside `ALLOWED_TABLES`.

    Runs during SQLite's statement compilation, before the statement is
    executed, so a denied query never runs — not even far enough to touch
    the row it was asking for. Every other action (including reads of
    allowlisted tables) is left to SQLite's normal permission handling by
    returning `SQLITE_OK`.

    Args:
        action: The `SQLITE_*` action code SQLite is authorizing.
        arg1: For `SQLITE_READ`, the table name being read. Meaning varies
            by action for every other code; not inspected here.
        arg2: Action-specific second argument (e.g. column name for
            `SQLITE_READ`). Not inspected here.
        dbname: The database name being accessed. Not inspected here.
        source: The name of the trigger/view responsible for this access, if
            any. Not inspected here.

    Returns:
        `sqlite3.SQLITE_DENY` for a disallowed table read, `sqlite3.SQLITE_OK`
        otherwise.
    """
    if action == sqlite3.SQLITE_READ and arg1 not in ALLOWED_TABLES:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def install_table_allowlist(conn: sqlite3.Connection) -> None:
    """Attach the `ALLOWED_TABLES` authorizer to an existing connection.

    Args:
        conn: The connection to guard. Mutated in place.
    """
    conn.set_authorizer(_authorize_read_within_allowlist)


def open_guarded_readonly(path: str | Path) -> sqlite3.Connection:
    """Open an OpenCode-shaped SQLite store with both INV-3 defenses installed.

    Args:
        path: Filesystem path to the SQLite database.

    Returns:
        A connection opened via `readonly_uri` (SQLite's own `mode=ro`) with
        `install_table_allowlist` already attached. Every connection this
        function returns carries both defenses; there is no path through
        this module that yields one with only one of them.
    """
    conn = sqlite3.connect(readonly_uri(path), uri=True)
    install_table_allowlist(conn)
    return conn
