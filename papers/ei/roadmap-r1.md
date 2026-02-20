# WS3 Ecological Informatics — R1 Resubmission Roadmap

Decision letter dated Nov 27, 2025. Revision due date listed as Jan 11, 2026 (now past as of Feb 19, 2026) — extension granted to February 23, 2026.

- [~] 1. Phase 1 — Intake, scope, and constraints
  - [~] 1.1 Task — Confirm editorial requirements and missing materials
    - [x] 1.1.1 Subtask — Verify current deadline/extension in Editorial Manager
    - [x] 1.1.2 Subtask — Locate/collect the “Review final.docx” attachment (if distinct from PDF)
    - [x] 1.1.3 Subtask — Identify required resubmission artifacts (response letter, highlighted manuscript, etc.)
  - [x] 1.2 Task — Summarize decision letter and reviewer themes
    - [x] 1.2.1 Subtask — Distill high-level concerns (novelty, scope fit, methods clarity, tone)
    - [x] 1.2.2 Subtask — Map reviewer comments to manuscript sections
    - [x] 1.2.3 Subtask — Decide response strategy (revise vs rebut)

- [~] 2. Phase 2 — Repository reorganization (r0/r1 parallel tracks)
  - [~] 2.1 Task — Freeze and archive r0 baseline
    - [x] 2.1.1 Subtask — Create `manuscript-r0/` from current manuscript sources
    - [ ] 2.1.2 Subtask — Record r0 manifest (hashes + file list) for diff baseline
    - [~] 2.1.3 Subtask — Ensure r0 builds cleanly and matches tagged PDF
  - [x] 2.2 Task — Create r1 working tree
    - [x] 2.2.1 Subtask — Create `manuscript-r1/` as copy of r0 baseline
    - [x] 2.2.2 Subtask — Add r1 placeholders (response letter, roadmap, diff output)
    - [x] 2.2.3 Subtask — Update Makefile/scripts to support r0/r1 builds
  - [~] 2.3 Task — Latexdiff workflow
    - [x] 2.3.1 Subtask — Define r0/r1 “flattened” EM submission sources
    - [x] 2.3.2 Subtask — Generate latexdiff source + PDF in `manuscript-r1/latexdiff-em/`
    - [~] 2.3.3 Subtask — Sanity check diff output (readable, anonymized, no broken refs)

- [~] 3. Phase 3 — Manuscript revisions (content and framing)
  - [x] 3.0 Task — Author voice consistency (HIGH PRIORITY)
    - [x] 3.0.1 Subtask — Review prior publications/tech docs to establish voice baseline (internal)
    - [x] 3.0.2 Subtask — Maintain concise voice notes (tone, cadence, vocabulary, structure)
    - [x] 3.0.3 Subtask — Edit manuscript for consistent author voice and clarity
  - [~] 3.1 Task — Tone, claims, and scope alignment
    - [x] 3.1.1 Subtask — Remove “sales-pitch” language and value adjectives (R2/R5)
    - [x] 3.1.2 Subtask — Temper abstract/summary claims (toy dataset, limitations)
    - [x] 3.1.3 Subtask — Strengthen ecological/informatics framing for EI scope (R1/R4)
    - [x] 3.1.4 Subtask — Reduce jargon and generic phrasing; define acronyms and key terms (R1/R5)
    - [ ] 3.1.5 Subtask — Align tone/structure with EI open-software exemplar papers
  - [x] 3.2 Task — Methods clarity and rigor
    - [x] 3.2.1 Subtask — Clarify LP model as strata-based Model I and implications (R2)
    - [x] 3.2.2 Subtask — Explain heuristic allocation process and scheduling logic (R2)
    - [x] 3.2.3 Subtask — Define decision variables (Xi,j) and spatial units explicitly (R3)
    - [x] 3.2.4 Subtask — Add missing Discussion section and integrate methodological context (R1)
  - [x] 3.3 Task — Novelty and comparison positioning
    - [x] 3.3.1 Subtask — Add literature/context on carbon in optimization-based planning (R5)
    - [x] 3.3.2 Subtask — Explicitly compare WS3 vs existing DSS (Woodstock, Patchworks, etc.) (R3)
    - [x] 3.3.3 Subtask — Clarify what is and is not supported (adjacency/opening size, carbon in objective) (R3/R5)
  - [x] 3.4 Task — Results, discussion, and conclusion
    - [x] 3.4.1 Subtask — Expand case-study description to justify benchmarks (R2)
    - [x] 3.4.2 Subtask — Reframe results as software/architecture contribution (R3/R5)
    - [x] 3.4.3 Subtask — Rewrite conclusion to synthesize findings and implications (R3)
    - [x] 3.4.4 Subtask — Remove programming-level detail (Listing 1) and replace with narrative

- [x] 4. Phase 4 — Reviewer-specific responses
  - [x] 4.1 Task — Reviewer #1 response (Review final.docx)
    - [x] 4.1.1 Subtask — Articulate novelty vs existing DSS and narrow “ecological informatics” claim
    - [x] 4.1.2 Subtask — Rebalance narrative away from user-manual tone; add scientific rationale
    - [x] 4.1.3 Subtask — Define Model I, stratification, zones, prescriptions; expand acronyms
    - [x] 4.1.4 Subtask — Remove/repair placeholder doc quotes and incomplete documentation references
    - [x] 4.1.5 Subtask — Add missing Discussion section; fix typos and citation formatting issues
  - [x] 4.2 Task — Reviewer #2 response
    - [x] 4.2.1 Subtask — Address tone/wording issues and remove subjective terms
    - [x] 4.2.2 Subtask — Expand LP/heuristic method details and scheduling logic
    - [x] 4.2.3 Subtask — Fix minor typos and citation details
  - [x] 4.3 Task — Reviewer #3 response
    - [x] 4.3.1 Subtask — Explain methodological framework and innovation claims
    - [x] 4.3.2 Subtask — Demonstrate advantages vs other DSS platforms
    - [x] 4.3.3 Subtask — Improve conclusion and reduce programming-level detail
  - [x] 4.4 Task — Reviewer #4 response
    - [x] 4.4.1 Subtask — Strengthen ecological informatics motivation and scope fit
    - [x] 4.4.2 Subtask — Clarify contribution beyond OR formulation
  - [x] 4.5 Task — Reviewer #5 response
    - [x] 4.5.1 Subtask — Clarify carbon accounting integration (post-hoc vs in-model)
    - [x] 4.5.2 Subtask — Add literature review on carbon in optimization-based planning
    - [x] 4.5.3 Subtask — Reduce jargon and generic phrasing; tighten narrative

- [~] 5. Phase 5 — Figures, tables, references, and artifacts
  - [x] 5.1 Task — Figures/tables updates
    - [x] 5.1.1 Subtask — Regenerate figures/tables if text changes require updates
    - [x] 5.1.2 Subtask — Validate captions and cross-references
  - [~] 5.2 Task — References and citations
    - [x] 5.2.1 Subtask — Add missing citations (Stanley model; carbon planning literature)
    - [x] 5.2.2 Subtask — Re-run reference validation and fix DOI/title mismatches
  - [~] 5.3 Task — Front/back matter
    - [x] 5.3.1 Subtask — Update highlights, keywords, and cover letter for r1
    - [x] 5.3.2 Subtask — Update declarations and AI-use statements if needed

- [x] 6. Phase 6 — Response package and submission prep
  - [x] 6.1 Task — Response to reviewers package
    - [x] 6.1.1 Subtask — Draft point-by-point response letter
    - [x] 6.1.2 Subtask — Build response matrix mapping comments → edits
    - [x] 6.1.3 Subtask — Ensure each response cites manuscript section/line refs
  - [x] 6.2 Task — Build outputs
    - [x] 6.2.1 Subtask — Compile r1 PDF and check warnings
    - [x] 6.2.2 Subtask — Produce latexdiff PDF for reviewers
    - [x] 6.2.3 Subtask — Generate EM-ready flattened submission package

- [ ] 7. Phase 7 — QA and resubmission
  - [ ] 7.1 Task — Final QA
    - [ ] 7.1.1 Subtask — Verify claims against data and benchmarks
    - [ ] 7.1.2 Subtask — Check formatting, line numbers, and anonymity
    - [ ] 7.1.3 Subtask — Confirm reproducibility links and artifacts
  - [ ] 7.2 Task — Resubmission
    - [ ] 7.2.1 Subtask — Upload manuscript, diff, and response letter
    - [ ] 7.2.2 Subtask — Verify metadata, declarations, and keywords in EM
    - [ ] 7.2.3 Subtask — Archive final r1 package and tag commit

## Submission Readiness Checklist (R1)

- [x] Deadline/extension confirmed in Editorial Manager
- [x] r0 baseline archived and reproducible from tag
- [x] r1 manuscript PDF compiles cleanly (no critical LaTeX warnings)
- [x] Latexdiff PDF generated and readable
- [x] Response-to-reviewers letter complete and cross-referenced
- [ ] Highlights/cover letter/metadata updated for r1
- [ ] References validated (DOIs, titles, year/volume/pages)
- [x] Figures/tables consistent with text and captions
- [ ] Data/software availability statements accurate
- [x] Final EM submission package validated (no subdirs, all assets present)

## Current focus

Immediate submission tasks
- Upload final `em-submission.zip` in Editorial Manager.
- Complete EM metadata fields (keywords, funding, COI, AI-use, data statement).
Recently addressed
- Front/back matter updated (highlights, keywords, cover letter); EM package rebuilt.
- Reference validation pass complete (filled missing years; no missing citations).
- Manuscript proof pass complete; response letter and matrix line refs updated.
