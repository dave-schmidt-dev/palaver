"""Migration runner for Palaver's SQLite store.

Every migration takes its rollback point with `VACUUM INTO`, never a file
copy. Under WAL journaling, a committed-but-uncheckpointed transaction tail
lives in the `-wal` sidecar file; copying only the `.db` file would silently
drop it. `VACUUM INTO` folds the live database — including that tail — into
one self-contained snapshot file, and this runner opens and reads that
snapshot before issuing any DDL for the migration it backs, so no assumption
about snapshot isolation is left implicit or untested.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from palaver.store.schema import SCHEMA_MIGRATIONS, Migration


class MigrationError(RuntimeError):
    """Raised when a migration's rollback point can't be trusted, or it failed."""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with WAL journaling and foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def current_version(conn: sqlite3.Connection) -> int:
    """Return the schema version recorded in `PRAGMA user_version`."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _take_backup(db_path: Path, backup_path: Path) -> None:
    """Snapshot db_path into backup_path via `VACUUM INTO`, run to completion."""
    if backup_path.exists():
        backup_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("VACUUM INTO ?", (str(backup_path),))
    finally:
        conn.close()


def _verify_backup(backup_path: Path) -> None:
    """Open and read backup_path, raising if it is absent or unreadable.

    Runs before any DDL statement of the migration it backs, so a missing or
    empty backup is caught while the original database is still untouched,
    rather than discovered only when a restore is later attempted.
    """
    if not backup_path.exists():
        raise MigrationError(f"backup file missing before migration: {backup_path}")
    try:
        ro_conn = sqlite3.connect(f"{backup_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            ro_conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        finally:
            ro_conn.close()
    except sqlite3.Error as exc:
        raise MigrationError(f"backup file unreadable before migration: {backup_path}") from exc


def _restore_backup(db_path: Path, backup_path: Path) -> None:
    """Replace db_path with backup_path, discarding any WAL/SHM sidecars.

    The sidecars are discarded rather than restored: the backup is a
    self-contained VACUUM INTO snapshot with no WAL tail of its own, so a
    stale sidecar left over from the failed migration must not be replayed
    against it.
    """
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{suffix}")
        if sidecar.exists():
            sidecar.unlink()
    shutil.copyfile(backup_path, db_path)


def migrate(
    db_path: str | Path,
    target_version: int | None = None,
    migrations: tuple[Migration, ...] = SCHEMA_MIGRATIONS,
) -> int:
    """Apply pending migrations up to target_version (default: the latest).

    Each migration's statements run one at a time under autocommit, so a
    failure partway through can leave a subset already committed to disk.
    Before any DDL for a migration runs, this function takes a VACUUM INTO
    backup at the migration's starting version and verifies the backup opens
    and reads successfully. If the migration then raises, the backup is
    restored over db_path and a MigrationError is raised from the original
    exception.

    Args:
        db_path: Path to the SQLite database file. Created if absent.
        target_version: Schema version to migrate to. Defaults to the
            highest version among `migrations`.
        migrations: Ordered migrations to consider. Defaults to the real
            schema history; tests may substitute a different sequence to
            exercise the rollback path without touching the real schema.

    Returns:
        The schema version the database ends up at.

    Raises:
        MigrationError: the rollback backup could not be created or
            verified, or a migration failed and was rolled back.
    """
    db_path = Path(db_path)
    ordered = sorted(migrations, key=lambda m: m.version)
    if target_version is None:
        target_version = ordered[-1].version if ordered else 0

    setup_conn = connect(db_path)
    try:
        version = current_version(setup_conn)
    finally:
        setup_conn.close()

    for migration in ordered:
        if migration.version <= version or migration.version > target_version:
            continue

        from_version = version
        backup_path = Path(f"{db_path}.bak-{from_version}")
        _take_backup(db_path, backup_path)
        _verify_backup(backup_path)

        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version={migration.version}")
        except sqlite3.Error as exc:
            conn.close()
            _restore_backup(db_path, backup_path)
            raise MigrationError(
                f"migration to version {migration.version} failed and was rolled back"
            ) from exc
        else:
            conn.close()

        version = migration.version

    return version
