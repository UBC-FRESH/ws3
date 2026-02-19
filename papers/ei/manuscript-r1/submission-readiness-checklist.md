## WS3 Readiness Assessment for *Ecological Informatics*

| **Category** | **Status** | **Details & Notes** |
|--------------|------------|---------------------|
| Manuscript template & formatting | ✅ Ready | EarthArXiv cover removed; standard Elsevier `elsarticle` layout with line numbers, single column, and updated `\journal{Ecological Informatics}` metadata. |
| Front matter | ✅ Ready | Highlights trimmed to ≤85 characters; abstract stresses informatics contributions; keyword list updated (computational ecology, decision support, carbon accounting, geospatial workflows). |
| Method framing | ✅ Ready | Introduction now enumerates the three methodological contributions; Section 4 includes a limitations paragraph covering data assumptions and computational envelopes. |
| Validation & benchmarking | ✅ Ready | Woodstock parity tables/figure and scaling benchmarks retained; narrative references parity CSVs and runtime/memory envelopes. |
| Software availability & reproducibility | ✅ Ready | Software availability box maintained; reproducibility paragraph points to Zenodo DOI 10.5281/zenodo.17331213 and deterministic scripts. |
| Mandatory declarations | ✅ Ready | Funding, data availability, competing interests, and AI-use statements added per EI GfA; acknowledgements now reference AI assistance and collaborators. |
| Figures, tables, and supplementary assets | ✅ Ready | Captions reviewed; artwork already stored as separate PNG/PDF files in `papers/ei/figs`; tables generated from CSVs in `papers/ei/tables`. |
| References & DOIs | ✅ Ready | Harvard author–year style via `elsarticle-harv`; references in `.bib` include DOIs/URLs where available; no unpublished items in the list. |
| Submission package | ✅ Ready | `latexmk` produces `papers/ei/paper.pdf`; cover letter template and highlights ready for Editorial Manager export; zipped EMS submission replaced with EI assets as needed. |

### Final checks before upload
- Export PDF, source `.tex`, figures, tables, highlights, and cover letter from `papers/ei`.
- Ensure the declarations tool reflects funding/competing interests at submission.
- Confirm Zenodo DOI resolves and README enumerates dependencies (libCBM optional install instructions).
- Decide on open access route (UBC has Elsevier OA agreement covering *Ecological Informatics*). |

