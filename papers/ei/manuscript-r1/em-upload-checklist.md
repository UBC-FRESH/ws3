# EI R1 Submission Checklist (EM Upload)

Date: 2026-02-20
Scope: Ecological Informatics (single-anonymized)

## 1) Manuscript compliance (from Guide for Authors)

- [x] Single-anonymized review: author names/affiliation/contact present in manuscript.
- [x] Title page includes author names, affiliation, corresponding author contact.
- [x] Abstract present and self-contained.
- [x] Abstract length <= 250 words.
  - Current estimate: ~192 words.
- [x] Keywords present (1-7).
  - Current count: 7.
  - Note: guide suggests avoiding multi-word keywords if possible.
- [x] Highlights file provided (3-5 bullets, <= 85 chars).
  - Current: 5 bullets, 60-70 chars each.
- [x] Graphical abstract provided (encouraged).
  - Current size: 2550 x 1260 px (w x h). Guide recommends 531 x 1328 (h x w) or proportional.
  - Action: confirm aspect ratio acceptable in EM.
- [x] Data statement present (Data availability section with Zenodo DOI).
- [x] Competing interests statement present.
- [x] Generative AI use statement present (required since AI tools used).
- [x] Funding statement present.
- [x] Tables are editable LaTeX (not images).
- [x] References cited in text and included in reference list.
- [x] Word count within journal guidance (approx 5600; < 7000 recommended, < 10,000 max).

## 2) EM package contents (validated)

All files are flat (no subdirectories) and present in `em-submission.zip`:
- Manuscript source: `paper.tex`, `references.bib`, `paper.bbl`
- Figures: `f1_architecture.png`, `f2_workflow.png`, `f3_spatial_allocation.png`, `f4a_harvest_and_stock.png`, `f4b_carbon_stocks.png`, `f5_neilsonhack_compare.png`, `scaling_*`, `sup_parity_periods.png`, `graphical_abstract.png`, `ws3-manuscript-graphical-abstract.pdf`
- Tables/CSV: `scenario_flows.csv`, `annual_carbon_stocks.csv`, `perf_scaling.csv`, `woodstock_parity*.csv`, `fair_checklist.csv`
- Highlights: `highlights.txt`
- Declarations: `declarationStatement.docx`
- Cover letter: `cover-letter.pdf`
- Style: `prisma-flow-diagram.sty`

## 3) EM form fields to verify (manual)

- [ ] Corresponding author full contact details (email + full postal address + phone) entered in EM.
- [ ] Funding details in EM match manuscript funding section.
- [ ] Competing interests declaration uploaded (declarationStatement.docx) and EM field set to "nothing to declare" or filled appropriately.
- [ ] Generative AI use declaration in EM matches manuscript statement.
- [ ] Data statement in EM matches manuscript (Zenodo DOI).
- [ ] Keywords entered in EM (match manuscript).
- [ ] Highlights file uploaded as separate item.
- [ ] Graphical abstract uploaded as separate item (optional but provided).
- [ ] Permissions confirmed for any third-party material (if any).

## 4) Action list

- Complete EM form fields and upload package.
