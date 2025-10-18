## WS3 Readiness Assessment for EMS Submission

| **Category**                | **Status**   | **Details & Notes**                                                                 |
|----------------------------|--------------|-------------------------------------------------------------------------------------|
| Open Repository            | ✅ Ready     | GitHub repo is public, active, and well-structured                                 |
| Open Source License        | ✅ Ready     | Clearly licensed under MIT (compatible with EMS requirements)                      |
| Versioning                 | ✅ Ready     | Semantic versioning applied (v1.0.0 on PyPI), changelog well maintained            |
| Documentation (Sphinx)     | ✅ Ready     | Excellent: full Sphinx site with install guide, architecture, use cases            |
| ReadTheDocs Hosting        | ✅ Ready     | Published, stable, auto-updating documentation site                                |
| Installation Instructions  | ✅ Ready     | Easy install via PyPI and Conda; documented in README.md and Sphinx docs           |
| Unit Tests / CI            | ✅ Ready     | Full test suite with CI on GitHub Actions                                          |
| User Support Features      | ✅ Ready     | CONTRIBUTING.md, CODE_OF_CONDUCT.md, ISSUE_TEMPLATE, clean project structure       |
| Examples & Tutorials       | ✅ Ready     | Multiple Jupyter Notebooks in `examples/`, with narrative and executable code       |
| Reproducible Case Study    | ✅ Ready     | Manuscript documents hardware, runtime/scale metrics, and libCBM install notes; optional scaling script and perf_scaling.csv ship with repro package |
| Software Archive with DOI  | ✅ Ready     | Release archived on Zenodo (10.5281/zenodo.17219651) and cited in manuscript       |
| Graphical Abstract         | ✅ Ready     | Figure available as `papers/ems/figs/graphical_abstract.png` and referenced        |
| FAIR Checklist Compliance  | ✅ Ready     | Manuscript FAIR checklist table (label `tab:fair-checklist`) and supplementary CSV (`papers/ems/tables/fair_checklist.csv`) document the requirements; paragraph cross-references the artefact |
| Input validation scope     | ⏭️ Out of scope | Exhaustive auto-validation is infeasible for Woodstock’s full domain; targeted checks documented; responsibility remains with qualified analysts |
| Verification vs Woodstock  | ✅ Ready     | Parity tables/figure generated in `papers/ems/repro/generate_case_study.py`; manuscript table (label `tab:woodstock_parity`) and supplementary CSVs provide prescriptive comparison |
| Scalability benchmarks     | ✅ Ready     | DataLad-enabled TSA suite wired; `perf_scaling.csv` covers sorted mash-ups, 1-vs-16 worker runs, spatial timing, and memory profiles |

### FAIR Compliance Summary
- **Findable**: GitHub repo with tagged releases; Zenodo DOI 10.5281/zenodo.17219651 referenced in paper and README.
- **Accessible**: MIT license, public repository, PyPI distribution; data assets derived from open/example datasets bundled with repo.
- **Interoperable**: Uses standard tabular formats (CSV), Woodstock text sections, and GeoTIFF rasters; API integrates with libCBM and SpaDES connectors.
- **Reusable**: Comprehensive docs on ReadTheDocs, deterministic reproduction scripts in `papers/ems/repro`, CI-validated tests/examples, contribution guidelines.

### Baseline from recent EMS software articles (FSLAM 2022, CIMCA 2023, pyMANGA 2024)
- **Structure & narrative**: Each article devotes substantial space to a detailed architecture section, workflow diagrams, and full case-study walkthroughs. pyMANGA, in particular, spends multiple sections on modular design, contribution workflow, and automated benchmarking. We should confirm our Architecture + Implementation sections match that depth (module diagrams, QA automation specifics, contribution pathway).
- **Quantitative validation**: FSLAM reports accuracy (89% with buffered inventory) and runtime (5 min over 30 M cells); CIMCA contrasts outputs across four MCA variants with district-level exposure summaries; pyMANGA benchmarks every module with automated tests. WS3 now documents deterministic scheduling/spatial timings, multi-core speed-ups, and peak memory usage across five real TSAs; retain these tables and callouts.
- **Reproducibility packaging**: All three papers cite GitHub releases plus clear data/benchmark bundles. FSLAM and pyMANGA ship ready-to-run plugins/examples; CIMCA publishes code with lightweight inputs. Our reproduction pipeline requires libCBM and the Woodstock example data—acceptable, but we should document setup friction (libCBM install quirks, data footprint) and provide smoke-test outputs or checksum references.
- **Software availability blocks**: Each paper uses a consistent table/bullet list including hardware requirements, supported OS, license, version, program size. Our availability list should mirror that completeness (e.g., add hardware requirements, approximate package size) to meet reviewer expectations.
- **Community/process transparency**: pyMANGA highlights GitHub workflows (PR review, actions, issue templates) and contributions. WS3 already has CONTRIBUTING.md/CI, but we should ensure the manuscript explicitly describes contribution pathways and governance to match that level of transparency.
