# Task 7.2 — Community Infrastructure

**Roadmap task**: P7.2
**Parent phase issue**: #74
**Status**: not_started
**Planning doc**: [phase7_release_and_community.md](planning/phase7_release_and_community.md)#task-72--community-infrastructure

---

## Goal

Establish community infrastructure for user feedback and contribution.

## Scope

- GitHub Discussions setup
- Issue templates
- CONTRIBUTING.md update
- Support channels in README.md
- CODE_OF_CONDUCT.md

## Subtasks

- [ ] Set up GitHub Discussions (Q&A, announcements, showcases categories)
- [ ] Create `.github/ISSUE_TEMPLATE/bug_report.md`
- [ ] Create `.github/ISSUE_TEMPLATE/feature_request.md`
- [ ] Create `.github/ISSUE_TEMPLATE/question.md`
- [ ] Update `CONTRIBUTING.md` with community guidelines
- [ ] Add support channels to `README.md`
- [ ] Create `CODE_OF_CONDUCT.md` if not exists

## Acceptance Criteria

- [ ] GitHub Discussions enabled with at least Q&A and announcements categories
- [ ] Three issue templates exist: bug report, feature request, question
- [ ] CONTRIBUTING.md has community guidelines section
- [ ] README.md has support channels section
- [ ] CODE_OF_CONDUCT.md exists (or confirmed not needed)

## Verification

```bash
ls .github/ISSUE_TEMPLATE/
grep -q "Discussion" README.md
grep -q "Contributing" CONTRIBUTING.md
```

## Artifacts

- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/ISSUE_TEMPLATE/question.md`
- `CONTRIBUTING.md` (updated)
- `README.md` (updated)
- `CODE_OF_CONDUCT.md` (new or confirmed)

## Risks

- GitHub Discussions may require organization-level settings (human action)
- Issue templates are markdown files (agent can create)

---

**Do not close until all subtasks are complete and verified.**