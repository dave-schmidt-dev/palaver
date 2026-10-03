# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed companion session joining for alternate `CODEX_HOME` directories by discovering roots from live agent open file descriptor paths (`tmp/arg0/codex-arg0*/.lock` and canonical rollout stores) with process identity revalidation and refusal on conflicting homes or stale PIDs.
