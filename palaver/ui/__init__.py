"""The pane-local surface: what Palaver shows inside iTerm2.

Everything under here is the only part of Palaver that needs iTerm2, and the
only part that cannot run entirely headless. `pyproject.toml` therefore keeps
`iterm2` in an optional `ui` extra rather than in the core requirements, so a
machine running only `palaver observe` never installs it.

The split inside this package is deliberate and the phase gate depends on it:
connection setup, session bookkeeping, and rendering are separate modules, so
the parts that *can* be proven headlessly are not entangled with the parts
that need a live terminal to prove anything at all.
"""
