# Contributing Guidelines

Thank you for contributing to `ws3`. This repository follows the same UBC-FRESH development workflow that is being used in `agent-workbench` and `freshforge`: work is planned and tracked through roadmap phases, tasks, and subtasks, with GitHub issues used as the durable coordination surface.

## Project Workflow

All non-trivial work should be organized as follows:

- One roadmap phase maps to one GitHub parent issue and one feature branch.
- One roadmap task maps to one child GitHub issue linked from the parent issue body.
- Subtasks should be tracked as checklist items in the child issue body unless they are large enough to warrant separate implementation issues.
- `ROADMAP.md`, `CHANGELOG.md`, planning notes, issue bodies, and pull request descriptions should stay synchronized.

Contributors should begin by reviewing the current roadmap and the active issue context before making substantive changes.

## Branching and Issue Hygiene

- Create or use a branch that corresponds to the active roadmap phase.
- Prefer branch names such as `feature/ws3-<phase-id>-<short-name>`.
- Before starting a new feature or substantial task, ensure the relevant GitHub issue exists and is linked to the roadmap.
- Keep work scoped to the current phase and child task. Avoid broad side work unless the maintainer explicitly approves it.
- When a roadmap task is completed, update the corresponding issue body and roadmap status so the repository reflects the current state.

## Development Expectations

- Keep changes small, reviewable, and clearly scoped.
- Prefer evidence-based validation over assumptions.
- Add or update tests when changing behaviour.
- Update documentation when the change materially affects usage, contributor workflow, or package behaviour.
- Keep public repository content free of private data, credentials, machine-specific paths, and unpublished notes.

## Verification

Before concluding work, verify the change with the relevant checks. At minimum, the following are expected for substantive changes:

```bash
python -m pytest
python -m ruff check .
python -m build
sphinx-build -b html docs _build/html -W
```

For larger changes, use the same verification style expected in the UBC-FRESH ecosystem: provide evidence from the repo, tests, or runtime output rather than relying on informal statements.

## Reporting Bugs and Proposing Features

- Report bugs by opening or updating a GitHub issue with a clear description, steps to reproduce, expected behaviour, actual behaviour, and any relevant logs.
- Propose new features through a GitHub issue that includes the motivation, scope, and any implementation sketch.
- Link the relevant issue in pull requests and describe how the change was validated.

## Code of Conduct

All contributors are expected to follow the community standards in [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

