# Contributing

Thank you for contributing to `ws3`. This repository follows the same UBC-FRESH development workflow that is being used in `agent-workbench` and `freshforge`: work is planned and tracked through roadmap phases, tasks, and subtasks, with GitHub issues used as the durable coordination surface.

## Workflow

- Check `ROADMAP.md` before starting non-trivial work.
- Use the active phase branch and linked GitHub issues.
- Keep `CHANGELOG.md`, roadmap checklists, issue comments, and PR descriptions synchronized with completed work.
- Use one parent GitHub issue per roadmap phase.
- Use one child GitHub issue per roadmap task.
- Track roadmap subtasks as checklist items in the child issue body.
- Check off child issue checklist items as subtasks complete.
- Close child issues before closing the parent issue.
- Close the parent issue only after the phase PR merges to `main`.

## Public-Safety Rules

- Do not commit raw agent transcripts, private project notes, credentials, generated local outputs, or machine-specific paths.
- Keep `tmp/`, `runtime/`, `local/`, and `outputs/` as ignored local working areas.
- Promote only sanitized and generally useful findings into tracked `planning/` notes.
- Keep examples generic unless a roadmap phase explicitly introduces a public-safe case study.

## Local Checks

For governance-only changes, run:

```bash
git status --short --branch
git diff --check
```

Also inspect changed Markdown files and search for accidental private paths, credentials, raw transcripts, and unrelated project-specific assumptions.

## Development Workflow Constraints

- Do not commit, push, create PRs, or mutate GitHub without approval.
- Use a dedicated feature branch for each roadmap phase.
- Prefer branch names such as `feature/ws3-<phase-id>-<short-name>`.
- Keep the implementation scoped to the current phase and child task.
- Update the roadmap and changelog as progress is made.
- Make verification part of the work rather than an afterthought.

## Verification

Before concluding work, verify the change with the relevant checks. At minimum, the following are expected for substantive changes:

```bash
python -m pytest
python -m ruff check ws3/ tests/
python -m build
sphinx-build -b html docs _build/html -W
```

For larger changes, use the same verification style expected in the UBC-FRESH ecosystem: provide evidence from the repo, tests, or runtime output rather than relying on informal statements.

## GitHub Issue And Comment Formatting

Formatting matters. GitHub issue bodies and comments must be readable as rendered Markdown, not flattened prose.

Rules:

- Use short section labels on their own lines, such as `Roadmap task: P3.1`, `Parent phase issue: #18`, `Status: active`, and `Checklist:`.
- Use real GitHub task-list syntax, with one checklist item per line.
- Never write inline pseudo-checklists such as `Checklist: [ ] first. [ ] second.`
- Wrap branch names, file paths, commands, and commit hashes in backticks.
- For parent phase issues, list child issues as task-list bullets with issue numbers and task IDs.
- Before creating or editing several issues, prepare bodies as multi-line Markdown strings or temporary body files.

## GitHub Issue Body Quality Standard

Issue bodies are part of the project specification and onboarding material. Write them so a new lab student, external collaborator, or coding agent can understand the task, implement it, verify it, and close it without reading the original chat transcript.

Parent phase issues must include phase identifier, status, branch name, roadmap links, goal, scope, out-of-scope boundaries, architecture notes, child task checklist, acceptance criteria, verification, and closeout requirements.

Child task issues must include task identifier, parent phase issue, status, related planning links, goal, scope, out-of-scope boundaries, subtasks, acceptance criteria, verification commands, artifacts, risks, and completion metadata once closed.

Do not create placeholder issue bodies with only a title and a short checklist unless the maintainer explicitly asks for a placeholder.

## Reporting Bugs and Proposing Features

- Report bugs by opening or updating a GitHub issue with a clear description, steps to reproduce, expected behaviour, actual behaviour, and any relevant logs.
- Propose new features through a GitHub issue that includes the motivation, scope, and any implementation sketch.
- Link the relevant issue in pull requests and describe how the change was validated.

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/UBC-FRESH/ws3/discussions) for usage questions and general discussion.
- **Bug Reports**: Use the [bug report template](https://github.com/UBC-FRESH/ws3/issues/new?template=bug_report.md) for bugs.
- **Feature Requests**: Use the [feature request template](https://github.com/UBC-FRESH/ws3/issues/new?template=feature_request.md) for new features.
- **Documentation**: See the [full documentation](https://ubc-fresh.github.io/ws3/) for tutorials and API reference.

## Code of Conduct

All contributors are expected to follow the community standards in [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

