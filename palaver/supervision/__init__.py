"""launchd templates for Palaver's supervised long-running processes.

This package holds no code — only the `.plist.tmpl` files rendered by
`palaver.cli.install_agent` (task 5.0) and, later, by the MCP server's own
agent (task 6.5). It exists as a package so the templates ship with an
installed distribution rather than only existing in a source checkout;
`pyproject.toml` names `*.plist.tmpl` as package data for the same reason.
"""
