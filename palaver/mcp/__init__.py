"""Palaver's MCP server: the read surface other agents query (Phase 6).

Two decisions are made here rather than per-tool, because getting either
wrong is invisible from the caller's side:

**Streamable HTTP, never stdio.** stdio is subprocess-per-client — every
Claude Code or Codex session that connects launches its own server process,
and six of those are six independent processes writing one SQLite file. The
whole memory layer rests on a single writer. Streamable HTTP runs the server
once at a fixed localhost endpoint and lets every client connect to it, so
the process count stays at one no matter how many agents attach.

**Scope is required, never defaulted.** See `tools_read`.
"""

from __future__ import annotations
