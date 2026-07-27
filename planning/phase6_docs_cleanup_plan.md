# Phase 6 — Documentation Cleanup and Integration

**Status:** Not Started  
**Branch:** `feature/ws3-phase6-docs-cleanup`  
**Start Date:** 2026-07-26  
**Target Completion:** TBD  

---

## Problem Statement

The current documentation has two issues:

1. **Fragmented structure:** Legacy flat-chapter docs (`Chapt1.rst`, `Chapt2.rst`, `intro.rst`, etc.) sit alongside the new structured sections (`getting_started/`, `textbook/`, `howto/`, `guides/`, `reference/`). This split is confusing and unnecessary.

2. **AI-generated slop:** Some sections contain verbose, filler content that doesn't match the UBC-FRESH brand of tight, precise technical writing. This needs to be purged.

---

## Goals

1. **Integrate legacy content** into the structured sections where it belongs
2. **Remove the "Old Documentation" section** from `index.rst`
3. **Audit and purge AI slop** from all documentation files
4. **Ensure consistent tone** across all docs — tight, precise, no filler
5. **Verify docs build** with zero errors and zero warnings

---

## Tasks

### Task 6.1 — Audit documentation for AI slop

**Scope:** Read every `.rst` and `.md` file in `docs/source/` and identify:
- Verbose filler sentences
- Redundant explanations
- Overly casual language
- Content that adds no technical value

**Deliverable:** `planning/phase6_docs_audit.md` listing all slop instances by file and line

---

### Task 6.2 — Integrate legacy chapters

**Scope:** The legacy flat chapters (`Chapt1.rst`, `Chapt2.rst`, `intro.rst`, `aboutws3.rst`, `appendices.rst`, `common.rst`, `core.rst`, `examples.rst`, `financial.rst`, `forest.rst`, `forest_helper.rst`, `libCBM.rst`, `modules.rst`, `opt.rst`, `spatial.rst`, `SpaDes.rst`) need to be:
- Reviewed for relevance
- Merged into appropriate structured sections OR removed if obsolete
- The "Old Documentation" section removed from `index.rst`

**Deliverable:** Updated `index.rst` without legacy section, legacy files integrated or removed

---

### Task 6.3 — Purge AI slop

**Scope:** Go through every documentation file and:
- Remove filler sentences
- Tighten verbose explanations
- Ensure consistent technical tone
- Match UBC-FRESH brand: precise, no-nonsense, no hand-holding

**Deliverable:** Cleaned documentation files

---

### Task 6.4 — Verify docs build

**Scope:** Ensure `sphinx-build -b html docs/source _build/html` succeeds with zero errors and zero warnings.

**Deliverable:** Clean build output

---

## Success Criteria

- [ ] No "Old Documentation" section in `index.rst`
- [ ] All legacy content either integrated or removed
- [ ] No AI slop remaining (verbose filler, redundant explanations)
- [ ] Consistent tight technical tone throughout
- [ ] Docs build with zero errors and zero warnings
- [ ] GitHub Pages updated
