# Task 7.1 — Release Verification

**Roadmap task**: P7.1
**Parent phase issue**: #74
**Status**: complete
**Planning doc**: [phase7_release_and_community.md](planning/phase7_release_and_community.md)#task-71--release-verification-done

---

## Goal

Verify release readiness: version bump, CHANGELOG, and verification suite.

## Scope

- Version bump from `1.1.0a1` to `1.1.0a2`
- CHANGELOG.md update with v1.1.0a2 entry
- Run verification suite

## Subtasks

- [x] Update `ws3/__init__.py` version to `1.1.0a2`
- [x] Update `CHANGELOG.md` with v1.1.0a2 entry
- [x] Verify `sphinx-build -b html docs/source _build/html -W` passes
- [x] Verify `python -c "import ws3; print(ws3.__version__)"` returns `1.1.0a2`

## Acceptance Criteria

- [x] `ws3.__version__` is `'1.1.0a2'`
- [x] CHANGELOG.md has v1.1.0a2 entry with Phase 6 summary
- [x] All verification checks pass

## Verification

```bash
python -c "import ws3; assert ws3.__version__ == '1.1.0a2'"
sphinx-build -b html docs/source _build/html -W
```

## Artifacts

- `ws3/__init__.py` — version `1.1.0a2`
- `CHANGELOG.md` — v1.1.0a2 entry

## Risks

None. This task is straightforward version management.

---

**Closed**: All subtasks complete.