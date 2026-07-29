# Phase 7 — Release and Community Building

**Purpose**: Prepare ws3 for public release as v1.1.0a2 and establish community infrastructure for user feedback and contribution.

**Date**: 2026-07-27 (updated 2026-07-29)

---

## Context

Phase 6 (Documentation Cleanup) is complete. All fabricated APIs purged, legacy stubs removed, troubleshooting guide fixed. The documentation now uses real ws3 APIs throughout and builds cleanly.

Phase 7 focuses on:
1. **Release preparation** — version bump (done: 1.1.0a2), CHANGELOG, verification
2. **Community infrastructure** — GitHub Discussions, issue templates, CONTRIBUTING.md
3. **User testing readiness** — ensure notebooks execute, add missing tests

This is a release-focused phase, not a feature-development phase. Scope is bounded.

---

## Task Breakdown

### Task 7.1 — Release Verification (DONE)

**Status**: complete
**Scope**: Version bump to 1.1.0a2, CHANGELOG update, verification suite

- [x] Update `ws3/__init__.py` version from `1.1.0a1` to `1.1.0a2`
- [x] Update `CHANGELOG.md` with v1.1.0a2 entry (Phase 6 summary)
- [x] Verify `sphinx-build -b html docs/source _build/html -W` passes
- [x] Verify `python -c "import ws3; print(ws3.__version__)"` returns `1.1.0a2`

### Task 7.2 — Community Infrastructure

**Status**: not_started
**Scope**: GitHub Discussions, issue templates, CONTRIBUTING.md, support channels

- [ ] Set up GitHub Discussions (Q&A, announcements, showcases)
- [ ] Create issue templates (bug report, feature request, question)
- [ ] Update `CONTRIBUTING.md` with community guidelines
- [ ] Add support channels to `README.md`
- [ ] Create `CODE_OF_CONDUCT.md` if not exists

### Task 7.3 — User Testing Readiness

**Status**: not_started
**Scope**: Ensure notebooks execute, add missing tests, fix remaining issues

- [ ] Verify all example notebooks execute successfully
- [ ] Add dedicated test files for `ws3.advanced_modeling`
- [ ] Add dedicated test files for `ws3.perf`
- [ ] Add dedicated test files for `ws3.integration`
- [ ] Fix any remaining notebook execution failures

### Task 7.4 — PyPI Release (Human Dependencies)

**Status**: not_started
**Scope**: Publish v1.1.0a2 to PyPI

**Human dependencies**:
- [ ] Configure GitHub environments and PyPI trusted publisher (requires human)
- [ ] Create release tag `v1.1.0a2` and push
- [ ] Publish to PyPI via trusted publisher (requires human)
- [ ] Create GitHub release with release notes

**Note**: The agent can prepare the tag and release notes, but the actual publish requires human credentials.

---

## Out of Scope

The following are explicitly NOT part of Phase 7:

- **Textbook chapters 19-20** — Future phase work (case studies, future directions)
- **New feature development** — This is a release phase, not a feature phase
- **CI/CD pipeline configuration** — Separate infrastructure work
- **Major documentation restructuring** — Phase 6 completed this

---

## Workflow Compliance

| Requirement | Implementation |
|-------------|----------------|
| Parent issue | `#74` — "Phase 7 — Release and Community Building" |
| Feature branch | `feature/ws3-phase7-release` |
| Child issues | `#75` (7.1), `#76` (7.2), `#77` (7.3), `#78` (7.4) |
| ROADMAP.md | Updated with Phase 7 and issue numbers |
| CHANGELOG.md | Updated with v1.1.0a2 entry |

---

## Success Criteria

- [ ] Version bumped to `1.1.0a2` (done)
- [ ] CHANGELOG updated with release notes (done)
- [ ] All verification checks pass (pytest, ruff, sphinx-build, build)
- [ ] GitHub Discussions enabled
- [ ] Issue templates created
- [ ] CONTRIBUTING.md updated with community guidelines
- [ ] Support channels documented in README.md
- [ ] All example notebooks execute successfully
- [ ] Dedicated test files for new modules (advanced_modeling, perf, integration)
- [ ] PyPI release published (requires human)

---

## Files to Edit

1. `ws3/__init__.py` — version number (DONE: `1.1.0a2`)
2. `CHANGELOG.md` — release notes (DONE: v1.1.0a2 entry)
3. `ROADMAP.md` — Phase 7 section (needs update)
4. `CONTRIBUTING.md` — community guidelines
5. `README.md` — support channels
6. `.github/ISSUE_TEMPLATE/` — bug report, feature request, question templates
7. `tests/test_advanced_modeling.py` — new test file
8. `tests/test_perf.py` — new test file
9. `tests/test_integration.py` — new test file

---

## References

- Phase 6 parent issue: [#69](https://github.com/UBC-FRESH/ws3/issues/69) (to be closed)
- Phase 6 completion: `planning/phase6_independent_audit.md`
- Current alpha release: [v1.1.0a1](https://pypi.org/project/ws3/1.1.0a1/)
- AGENTS.md workflow policy: `AGENTS.md`

---

**This document is the single source of truth for Phase 7. Do not trust any other status document.**