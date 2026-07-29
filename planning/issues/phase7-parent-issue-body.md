# Phase 7 — Release and Community Building

**Roadmap task**: P7
**Parent phase issue**: this issue (#74)
**Status**: active
**Branch**: `feature/ws3-phase7-release`
**Start date**: 2026-07-27
**Planning doc**: [phase7_release_and_community.md](planning/phase7_release_and_community.md)

---

## Goal

Prepare ws3 for public release as v1.1.0a2 and establish community infrastructure for user feedback and contribution.

## Scope

- Release verification (DONE: version 1.1.0a2, CHANGELOG updated)
- Community infrastructure (GitHub Discussions, issue templates, CONTRIBUTING.md)
- User testing readiness (notebook execution, test files for new modules)
- PyPI publication (requires human credentials)

## Out of Scope

- Textbook chapters 19-20 (future phase)
- New feature development (release phase only)
- CI/CD pipeline configuration (separate infrastructure)
- Major documentation restructuring (Phase 6 complete)

## Architecture Notes

Phase 7 is a release phase, not a feature phase. All code changes are bounded to:
- Version bump (DONE)
- Community files (CONTRIBUTING.md, issue templates, CODE_OF_CONDUCT.md)
- Test files for existing modules (advanced_modeling, perf, integration)
- README.md support channels

No new ws3 package modules are added in this phase.

## Child Task Checklist

- [ ] **#75** — Task 7.1: Release Verification (DONE)
- [ ] **#76** — Task 7.2: Community Infrastructure
- [ ] **#77** — Task 7.3: User Testing Readiness
- [ ] **#78** — Task 7.4: PyPI Release

## Acceptance Criteria

- [ ] Version is `1.1.0a2` in `ws3/__init__.py`
- [ ] CHANGELOG.md has v1.1.0a2 entry
- [ ] All verification checks pass (pytest, ruff, sphinx-build, build)
- [ ] GitHub Discussions enabled
- [ ] Issue templates exist (.github/ISSUE_TEMPLATE/)
- [ ] CONTRIBUTING.md updated with community guidelines
- [ ] Support channels in README.md
- [ ] All example notebooks execute successfully
- [ ] Test files exist for advanced_modeling, perf, integration
- [ ] PyPI release published (requires human)

## Verification

```bash
python -c "import ws3; assert ws3.__version__ == '1.1.0a2'"
python -m pytest
python -m ruff check .
sphinx-build -b html docs _build/html -W
python -m build
```

## Closeout Requirements

- All child issues closed or explicitly deferred
- PR from `feature/ws3-phase7-release` merged to `main`
- PyPI release published
- CHANGELOG.md updated with release date
- Parent issue closed after PR merge

---

**Do not close this issue until all child issues are resolved and the PR is merged.**