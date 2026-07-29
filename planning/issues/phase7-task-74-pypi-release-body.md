# Task 7.4 — PyPI Release

**Roadmap task**: P7.4
**Parent phase issue**: #74
**Status**: not_started
**Planning doc**: [phase7_release_and_community.md](planning/phase7_release_and_community.md)#task-74--pypi-release-human-dependencies

---

## Goal

Publish ws3 v1.1.0a2 to PyPI.

## Scope

- Create release tag `v1.1.0a2`
- Publish to PyPI via trusted publisher
- Create GitHub release with release notes

## Subtasks

- [ ] Configure GitHub environments and PyPI trusted publisher (requires human)
- [ ] Create release tag `v1.1.0a2` and push to GitHub
- [ ] Publish to PyPI via trusted publisher (requires human)
- [ ] Create GitHub release with release notes

## Acceptance Criteria

- [ ] Tag `v1.1.0a2` exists in GitHub
- [ ] ws3==1.1.0a2 is publishable on PyPI
- [ ] GitHub release exists with release notes

## Verification

```bash
git tag -l "v1.1.0a2"
pip install ws3==1.1.0a2
```

## Artifacts

- Git tag `v1.1.0a2`
- PyPI package `ws3==1.1.0a2`
- GitHub release with notes

## Human Dependencies

- GitHub environment configuration for PyPI trusted publisher
- PyPI credentials (handled by trusted publisher once configured)
- Release tag creation and push
- GitHub release creation

## Risks

- PyPI trusted publisher requires organization-level GitHub settings
- Release tag must match version in `ws3/__init__.py`

---

**Do not close until PyPI publish is confirmed and GitHub release exists.**