# Task 7.3 — User Testing Readiness

**Roadmap task**: P7.3
**Parent phase issue**: #74
**Status**: not_started
**Planning doc**: [phase7_release_and_community.md](planning/phase7_release_and_community.md)#task-73--user-testing-readiness

---

## Goal

Ensure ws3 is ready for user testing: all notebooks execute, test coverage for new modules.

## Scope

- Verify all example notebooks execute successfully
- Add test files for `ws3.advanced_modeling`
- Add test files for `ws3.perf`
- Add test files for `ws3.integration`
- Fix any remaining notebook execution failures

## Subtasks

- [ ] Run all example notebooks and verify execution
- [ ] Create `tests/test_advanced_modeling.py` with import checks and basic functionality tests
- [ ] Create `tests/test_perf.py` with import checks and basic functionality tests
- [ ] Create `tests/test_integration.py` with import checks and basic functionality tests
- [ ] Fix any notebook execution failures

## Acceptance Criteria

- [ ] All 21 example notebooks execute without errors
- [ ] `tests/test_advanced_modeling.py` exists and passes
- [ ] `tests/test_perf.py` exists and passes
- [ ] `tests/test_integration.py` exists and passes
- [ ] `python -m pytest` passes with no failures

## Verification

```bash
python -m pytest
python -m pytest tests/test_advanced_modeling.py
python -m pytest tests/test_perf.py
python -m pytest tests/test_integration.py
```

## Artifacts

- `tests/test_advanced_modeling.py`
- `tests/test_perf.py`
- `tests/test_integration.py`

## Risks

- Notebook execution may fail due to missing data or environment issues
- Test coverage for advanced modules may be limited without deep domain knowledge

---

**Do not close until all subtasks are complete and verified.**