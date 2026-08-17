"""iTerm2 connection, pane discovery, and companion-pane foundations.

Everything under here is the only part of Palaver that needs iTerm2, and the
only part that cannot run entirely headless. `pyproject.toml` therefore keeps
`iterm2` in an optional `ui` extra rather than in the core requirements, so a
machine running only `palaver observe` never installs it.

The rejected status-bar implementation is intentionally absent. This package
keeps only the connection, lifecycle, and pane-to-session join machinery that
the per-agent companion-pane surface will reuse.
"""
