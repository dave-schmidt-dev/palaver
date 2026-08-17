"""Stable project identities shared by observed sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectIdentity:
    """The database identity and canonical path for one project."""

    name: str
    path: Path


def canonical_project_path(path: str | Path) -> Path:
    """Return an absolute, normalized project path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


def encoded_project_path(path: str | Path) -> str:
    """Encode a path using Claude Code's established project-directory spelling."""
    return str(canonical_project_path(path)).replace("/", "-").replace(".", "-")


def project_identity_for_cwd(cwd: str | Path) -> ProjectIdentity:
    """Build a collision-resistant identity from a session working directory.

    The encoded canonical path keeps the identity recognizable, while the
    digest distinguishes paths that collapse under Claude's punctuation
    encoding. The real canonical path remains separately available for the
    database's path uniqueness constraint and exact scope lookup.
    """
    path = canonical_project_path(cwd)
    encoded = encoded_project_path(path)
    suffix = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
    basename = path.name or "root"
    return ProjectIdentity(name=f"{encoded}-{basename}-{suffix}", path=path)


__all__ = [
    "ProjectIdentity",
    "canonical_project_path",
    "encoded_project_path",
    "project_identity_for_cwd",
]
