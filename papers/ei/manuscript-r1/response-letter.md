# Response to Editor and Reviewers

Manuscript: **WS3: An open-source Python framework for integrated simulation and optimization of forest landscape and wood supply systems**  
Manuscript number: **ECOINF-D-25-03829**  
Revision: **R1** (latexdiff provided)

Dear Editor,

Thank you for the constructive reviews and the opportunity to revise our manuscript. We have carefully addressed each comment and revised the manuscript to clarify scope, strengthen the ecological informatics framing, expand methodological detail, and tighten tone. A point‑by‑point response follows, with line references to the revised manuscript (line numbers from the marked‑line PDF). A latexdiff file and updated submission package are included.

## Summary of major changes
- **Scope and framing:** Repositioned WS3 as open, reproducible informatics infrastructure rather than a new optimization formulation; clarified what the framework does not attempt. (Lines 100–110, 450–456)
- **Methods clarity:** Expanded the Model~I formulation, data model, and workflow narrative; clarified heuristic scheduling and the Woodstock→Stanley spatialization linkage. (Lines 122–126, 174, 193–217, 290–294)
- **Results interpretation:** Framed the case study as a reproducible pipeline demonstration; parity check presented as an illustrative validation rather than a proof. (Lines 346–347, 414–416)
- **Comparative context:** Strengthened comparison narrative and carbon‑planning literature context. (Lines 443–448)
- **Quality assurance and reproducibility:** Added testing/validation context and clarified input stewardship boundaries. (Lines 231–234)

---

# Editor

**ED‑1. Provide point‑by‑point response and outline changes (highlighted manuscript).**  
**Response:** Completed. This response letter provides point‑by‑point replies with line references. A latexdiff PDF is included with the resubmission package.

**ED‑2. Resubmission date and extension.**  
**Response:** Extension granted through **February 23, 2026**; this revision is submitted within that window.

---

# Reviewer 1

**R1‑1. Novelty unclear; “ecological informatics” claim too broad.**  
**Response:** We reframed the contribution as open, reproducible informatics infrastructure rather than a novel formulation, and clarified scope in the Abstract, Introduction, and Discussion. (Lines 62–69, 100–110, 450–456)

**R1‑2. Reads like a technical report/manual; insufficient scientific rationale.**  
**Response:** We reduced feature‑listing tone, tightened framing, and added a Discussion section focused on informatics contribution and limitations. (Lines 100–110, 450–456)

**R1‑3. Model I, stratification, zones, prescriptions unclear; acronyms not defined.**  
**Response:** We clarified the data model and the Model~I formulation and explicitly defined decision variables and strata. (Lines 122–126, 193–217)

**R1‑4. Excessive jargon/vague terms.**  
**Response:** We edited the Introduction/Methods to remove vague phrasing and replaced jargon with concrete descriptions. (Lines 100–110, 122–217)

**R1‑5. Cited docs contain placeholders; manuscript quotes incomplete documentation.**  
**Response:** We removed placeholder quotes and clarified documentation scope/expectations in the “Input stewardship and validity” paragraph. (Lines 233–234)

**R1‑6. Missing Discussion; typos/formatting; questionable test comparisons.**  
**Response:** A Discussion section has been added; parity is framed as an illustrative, reproducible check rather than proof. (Lines 450–460, 414–416)

---

# Reviewer 2

**R2‑1. Tone reads like sales pitch; subjective adjectives.**  
**Response:** Tone has been softened throughout; claims are phrased in method‑centric terms. (Lines 62–69, 100–110)

**R2‑2. Abstract claims too strong; validation on toy dataset; case studies under‑described.**  
**Response:** Case‑study design is now explicitly described and results are framed as an illustrative pipeline demonstration rather than validation. (Lines 290–294, 346–347)

**R2‑3. Modeling/heuristic details unclear; Model I, allocation heuristic, scheduling logic; mention Stanley if relevant.**  
**Response:** We expanded the data model, Model~I formulation, and workflow narrative, and explicitly described the hierarchical strategic→tactical allocation lineage and the Woodstock–Stanley linkage. (Lines 122–126, 174, 193–217, 290–294)

**R2‑4. Minor typos and citation fixes.**  
**Response:** Citation and typographic cleanup performed throughout; the Johnson and Scheurman (1977) entry is formatted as *Forest Science* 23(1), Monograph 18. DOI formatting has been standardized where present. (Line‑level fixes distributed; see updated references.)

---

# Reviewer 3

**R3‑1. Methodological depth/innovation unclear (risk/uncertainty/AI, etc.).**  
**Response:** We now state explicitly what WS3 does not attempt (risk/uncertainty/ML/fully spatial optimization) and focus the contribution on reproducible workflows. (Lines 100–110, 450–456)

**R3‑2. Intro too much jargon; needs clearer rationale.**  
**Response:** Introduction tightened and simplified; rationale now emphasizes the gap between open simulation scaffolds and end‑to‑end planning workflows. (Lines 100–110)

**R3‑3. Advantages vs existing DSS not demonstrated; lacks spatial constraints.**  
**Response:** Comparison narrative strengthened; scope limits (no adjacency/road planning) stated explicitly; parity test framed as illustrative. (Lines 414–416, 443–448, 456–457)

**R3‑4. Conclusion lacks synthesis.**  
**Response:** Conclusion rewritten to synthesize contributions and limitations. (Lines 462–467)

**R3‑5. Programming‑level detail unnecessary; Xi,j definition unclear.**  
**Response:** Listing removed; mathematical definitions clarified in Model~I section. (Lines 174, 203–207)

---

# Reviewer 4

**R4‑1. Objective function not aligned with ecological scope; contribution appears OR‑centric.**  
**Response:** We emphasize the informatics contribution and reproducible workflow integration rather than novelty of formulation. (Lines 62–69, 100–110, 450–456)

---

# Reviewer 5

**R5‑1. Contribution mainly software; carbon accounting is post‑hoc.**  
**Response:** We explicitly state carbon accounting is applied post hoc in this manuscript, and we focus the contribution on transparent, reproducible integration. (Lines 67, 458)

**R5‑2. Should allow carbon objectives/constraints.**  
**Response:** We note that carbon objectives/constraints are possible when users provide coefficients, but they are outside the scope of the present demonstration. (Line 458)

**R5‑3. LLM‑like style/jargon; missing carbon‑optimization literature.**  
**Response:** We tightened cadence and reduced jargon, and added carbon‑planning context in the comparison narrative. (Lines 361–448)

---

We appreciate the reviewers’ feedback and believe the revision substantially improves clarity, scope alignment, and scientific framing. We are happy to address any additional concerns.

Sincerely,  
Gregory Paradis
