# Phase 6: GitHub MCP Server Integration

**Status**: Planning  
**Branch**: `feature/ws3-phase6-github-mcp`  
**Parent Issue**: TBD (create after Phase 5 closeout)  

---

## Overview

Phase 6 focuses on wiring the GitHub MCP server into the ws3 development workflow so that AI coding agents can interact with GitHub issues, PRs, and repository state without manual setup. This eliminates the friction of explaining GitHub integration to agents.

---

## Goals

1. **Automatic GitHub Integration**: Agents can read/write issues, PRs, and repository state without manual configuration
2. **Consistent Workflow**: Enforce the UBC-FRESH phase/task/subtask workflow through GitHub issue structure
3. **Reduced Friction**: New contributors and agents can start working immediately without GitHub setup tutorials

---

## Proposed Tasks

### Task 6.1 — Install and Configure GitHub MCP Server

**Scope**: 
- Add `github-mcp-server` as an optional dependency in `pyproject.toml`
- Create installation guide for developers
- Document configuration requirements (GitHub token, repository URL)

**Deliverables**:
- `pyproject.toml` updated with `github-mcp` optional dependency
- `docs/guides/github_mcp_setup.md` — installation and configuration guide
- Example `.env` or configuration file for local development

**Acceptance Criteria**:
- `pip install ws3[github-mcp]` installs the MCP server
- Configuration guide covers all supported environments (VS Code, CLI, CI/CD)
- Token-based authentication documented

---

### Task 6.2 — Create GitHub Issue Templates

**Scope**:
- Create issue templates for phases, tasks, and bugs
- Document the expected issue body structure
- Add template files to `.github/ISSUE_TEMPLATE/`

**Deliverables**:
- `.github/ISSUE_TEMPLATE/phase.md` — template for new phases
- `.github/ISSUE_TEMPLATE/task.md` — template for tasks
- `.github/ISSUE_TEMPLATE/bug.md` — template for bug reports
- Updated `CONTRIBUTING.md` with template usage instructions

**Acceptance Criteria**:
- Templates enforce consistent issue structure
- Templates include all required sections (goal, scope, subtasks, acceptance criteria, verification)
- Templates are easy to use via GitHub web UI

---

### Task 6.3 — Create GitHub Actions for Issue Automation

**Scope**:
- Automate issue labeling based on content
- Auto-assign issues to appropriate reviewers
- Link issues to branches automatically
- Notify maintainers of stale issues

**Deliverables**:
- `.github/workflows/issue-labeler.yml` — auto-label issues
- `.github/workflows/issue-assign.yml` — auto-assign issues
- `.github/workflows/issue-stale.yml` — stale issue detection
- Updated `CONTRIBUTING.md` with automation behavior documentation

**Acceptance Criteria**:
- Issues are automatically labeled based on content
- Issues are auto-assigned to appropriate reviewers
- Stale issues are detected and notified
- Automation doesn't interfere with normal workflow

---

### Task 6.4 — Create Agent Workflow Documentation

**Scope**:
- Document how agents should interact with GitHub issues
- Create examples of issue creation, updating, and closing
- Document the expected issue body format
- Add examples to `AGENTS.md`

**Deliverables**:
- Updated `AGENTS.md` with GitHub interaction guidelines
- `docs/guides/agent_github_workflow.md` — agent-specific GitHub guide
- Examples of proper issue creation and updates
- Verification commands for issue state

**Acceptance Criteria**:
- Agents can create, update, and close issues without manual guidance
- Issue bodies follow the expected format
- Agents verify their work through issue state changes
- Documentation is clear and actionable

---

### Task 6.5 — Create Verification Scripts

**Scope**:
- Scripts to verify issue state matches roadmap
- Scripts to check issue body completeness
- Scripts to validate issue linking (parent/child relationships)

**Deliverables**:
- `scripts/verify_issues.py` — verify issue state
- `scripts/check_issue_bodies.py` — validate issue body format
- `scripts/link_issues.py` — validate parent/child linking
- Updated `CONTRIBUTING.md` with verification commands

**Acceptance Criteria**:
- Scripts can verify all open issues are properly structured
- Scripts detect missing or incomplete issue bodies
- Scripts validate parent/child issue relationships
- Scripts integrate with CI/CD pipeline

---

## Dependencies

- GitHub repository with admin access
- GitHub token with appropriate permissions
- GitHub MCP server installed and configured
- CI/CD pipeline for automation workflows

---

## Success Metrics

- **Adoption**: 100% of new issues created using templates
- **Consistency**: 95% of issues have complete bodies
- **Efficiency**: Agents can create/update/close issues without manual guidance
- **Automation**: 80% of routine issue tasks automated

---

## Timeline

**Month 1**: Task 6.1 (Install and Configure) + Task 6.2 (Issue Templates)  
**Month 2**: Task 6.3 (GitHub Actions) + Task 6.4 (Agent Workflow Docs)  
**Month 3**: Task 6.5 (Verification Scripts) + Phase closeout

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GitHub MCP server configuration complexity | High | Medium | Provide clear setup guide with examples |
| Issue template rigidity | Medium | Medium | Allow flexibility for edge cases |
| Automation interference | Medium | Low | Test thoroughly before enabling |
| Agent adoption slow | Medium | Medium | Provide clear documentation and examples |

---

## Next Steps

1. **Close Phase 5** — Complete smoke testing and promote v1.1.0a1 to stable
2. **Create Phase 6 parent issue** — Link all child tasks
3. **Start Task 6.1** — Install and configure GitHub MCP server
4. **Begin Task 6.2** — Create issue templates
5. **Iterate** — Adjust based on agent feedback and workflow needs