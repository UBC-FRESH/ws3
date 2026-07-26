# Child issue: add typing infrastructure and package conventions

## Roadmap task: P2.1

## Status: planned

## Summary

Introduce the project-level typing and developer conventions needed to make `ws3` more interpretable and easier for coding agents to reason about.

## Scope

- Add typed configuration and packaging support for development tooling.
- Introduce a consistent typing strategy for public functions and classes.
- Add a lightweight typing validation workflow to the repository.
- Document the intended conventions in the contributor and developer docs.

## Acceptance criteria

- `pyproject.toml` includes typing-related development tooling and configuration.
- `AGENTS.md` and `CONTRIBUTING.md` make the typing and refactor expectations explicit.
- The repository has a reproducible local check for typing-oriented validation.
- The initial typing infrastructure can be exercised without breaking the existing package import path.
