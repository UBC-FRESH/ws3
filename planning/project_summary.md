# Postdoctoral Research Project Summary

## WS3: Open-Source Forest Estate Modeling and Wood Supply Simulation

**Researcher:** Gregory Paradis  
**Institution:** University of British Columbia, Faculty of Forestry  
**Supervisor (U. Laval):** Luc LeBel  
**Funding:** FORAC Research Consortium; NSERC; ECCC; NRCan; Government of BC; Mitacs; CFI JELF; BCKDF  
**Timeline:** 2015–present (postdoc work 2017–2018, continued through 2025+)

---

## Chronological Order of the Four Technical Reports

| # | Report | Date | CIRRELT Ref | Title |
|---|--------|------|-------------|-------|
| 1 | **CIRRELT-2017-34** | June 2017 | CIRRELT-2017-34 | *Diameter Distribution Models for Quebec, Canada* |
| 2 | **CIRRELT-2017-43** | July 2017 | CIRRELT-2017-43 | *Compiling Disaggregation Coefficients to Link Long- and Short-Term Planning Models* |
| 3 | **CIRRELT-2018-23** | May 2018 | CIRRELT-2018-23 | *Retro-Fitting Value-Creation Potential Indicators to Long-Term Supply Models* |
| 4 | **CIRRELT-2018-24** | May 2018 | CIRRELT-2018-24 | *Estimating the Value-Creation Potential of Wood Supply Using a Hybrid Simulation-Optimization Approach* |

---

## Detailed Project Summary

### Overview

This postdoctoral project, conducted at the University of British Columbia (Faculty of Forestry) in collaboration with Université Laval (Département des sciences du bois et de la forêt) and CIRRELT, developed open-source computational tools and methodologies for **strategic forest management planning in Quebec, Canada**. The work addresses a fundamental gap: Quebec's wood supply optimization models estimate maximum sustainable timber harvest levels (Annual Allowable Cut, or AAC) but include **no financial performance indicators**. The project built a complete pipeline—from statistical diameter distribution modeling through hierarchical planning linkages to value-creation potential estimation—implemented as the open-source **ws3** (Wood Supply Simulation System) framework.

---

### Report 1: Diameter Distribution Models for Quebec, Canada
**CIRRELT-2017-34 | June 2017**  
*Authors:* Gregory Paradis, Luc LeBel  
*Affiliation at time:* Université Laval

**Problem:** Stem diameter distributions (stand tables) are essential for forestry applications but were not readily available for Quebec's forest inventory data.

**Methodology:**
- Tested 25 truncated distributions from the generalized beta family against a large dataset of stems from Quebec's government-run permanent sample plot inventory
- Developed a two-stage parameter-fitting methodology using non-linear least-squares, producing improved estimates of parameter estimation error and parameter correlation for bounded-domain data
- Generated best-fit distributions, parameter estimates (with standard errors), and AICc for 30 subdatasets covering all combinations of 10 species groups × 3 cover types across the province

**Key Finding:** Best-fit results were clearly dominated by the four distributions in the **generalized gamma family**, providing a robust statistical foundation for downstream volume disaggregation work.

**Significance:** This report established the statistical groundwork for converting aggregate volume estimates into log-assortment-level disaggregations—a prerequisite for linking long-term wood supply models to short-term industrial fiber consumption models.

---

### Report 2: Compiling Disaggregation Coefficients
**CIRRELT-2017-43 | July 2017**  
*Authors:* Gregory Paradis, Luc LeBel  
*Affiliation at time:* Université Laval

**Problem:** Linking long-term wood supply optimization models (upper level) to short-term network flow models (lower level) requires a matrix of **disaggregation coefficients** that convert species-wise timber volume output into log assortments (by size, species, and quality). These coefficients did not exist for Quebec's wood supply models.

**Methodology:**
- Built on the diameter distribution models from Report 1 (CIRRELT-2017-34)
- Described a reproducible methodology for compiling disaggregation coefficient matrices using readily-available data
- The matrix converts aggregate volume outputs from the upper-level model into log-class assortments consumable by the lower-level network flow model

**Key Contribution:** Provided the missing linkage component that enables a **bilevel modeling framework**:
- **Upper level:** Standard long-term wood supply optimization (maximizes sustainable harvest volume)
- **Lower level:** Short-term network flow optimization (maximizes profit from primary forest product sales)

**Significance:** This report bridged the gap between bio-physical fiber estimation and economic value assessment, enabling the hierarchical planning approach where industrial fiber consumption behavior is anticipated within long-term supply models.

---

### Report 3: Retro-Fitting Value-Creation Potential Indicators
**CIRRELT-2018-23 | May 2018**  
*Authors:* Gregory Paradis, Luc LeBel  
*Affiliations:* UBC (Forest Resources Management) + Université Laval + CIRRELT

**Problem:** The bilevel formulation described in Paradis et al. (2018) requires both disaggregation coefficients (from Report 2) and **value-creation potential (VCP) coefficients** to estimate unit profit for all possible fiber flow paths through the lower-level network. VCP coefficients did not exist for Quebec's models.

**Methodology:**
- Extended the disaggregation methodology from Report 2 (CIRRELT-2017-43)
- Linked disaggregated volumes to value-creation data from an existing database
- Re-aggregated these into VCP coefficients usable in long-term wood supply models
- Developed purpose-built software implementation (the ws3 framework) for reproducible compilation

**Key Contribution:** Greatly simplified the otherwise onerous task of compiling VCP indicators from available data. The methodology:
- Uses readily-available wood supply model data and existing databases
- Is specifically compatible with Quebec's data and model structure
- Can be adapted to other jurisdictions with relative ease
- Represents a first step toward **value-driven forest planning** (as opposed to volume-driven planning)

**Significance:** Completed the methodological chain from diameter distributions → disaggregation → VCP estimation, enabling wood supply models to incorporate financial performance indicators for the first time in Quebec.

---

### Report 4: Hybrid Simulation-Optimization Approach for VCP Estimation
**CIRRELT-2018-24 | May 2018**  
*Authors:* Gregory Paradis, Luc LeBel  
*Affiliations:* UBC (Forest Resources Management) + Université Laval + CIRRELT

**Problem:** Quebec's wood supply models produce AAC estimates but lack any financial performance indicators. There was no framework to estimate the **value-creation potential** of standing timber—the marginal profit a mill owner might make from sale of primary forest products, accounting for all costs and revenues from standing tree to delivered product.

**Methodology:**
- Compiled a hybrid simulation-optimization model that retrofits financial performance indicators to the optimal solution of long-term wood supply models
- Linked to a network flow optimization model simulating profit-maximizing fiber consumption of a network of primary processing facilities
- The network flow model simulates the subset of available fiber supply that a given profit-maximizing industrial network configuration will consume
- Applied to all 71 management units of Quebec's managed public forest (76 million hectares)
- Demonstrated on management unit UA 064-51 with multiple scenarios

**Key Contributions:**
1. **VCP as a function of AAC consumption:** Reported VCP across different proportions of AAC consumed, revealing techo-economic factors limiting fiber consumption
2. **Scenario framework:** The model can be solved repeatedly with different network configurations (opening/closing facilities, changing capacity, testing market prices, evaluating stumpage rate models) to generate comparable scenarios
3. **Provincial-scale insight:** Produced state-of-the-art VCP estimates yielding new understanding of relationships between wood supply modeling, timber licensing policy, stumpage policy, industrial network configurations, and market prices

**Significance:** This report delivered the complete end-to-end methodology—transforming Quebec's volume-only wood supply models into value-aware decision-support tools. It demonstrated that the framework could be applied province-wide and could generate actionable insights for policy analysis.

---

## The ws3 Framework: Implementation and Legacy

The methodologies described across these four reports were implemented as **ws3** (Wood Supply Simulation System), an open-source Python framework released under the MIT license. Key milestones:

| Year | Milestone |
|------|-----------|
| 2015 | First public release (v0.1) |
| 2017–2018 | Core methodology development (the four CIRRELT reports) |
| 2018 | Value-creation potential methodology published |
| 2021 | Initial release on PyPI (v0.0.1) |
| 2024 | Major refactor to typed Python (Phase 2), 62 tests passing |
| 2025 | v1.0.5 release; submitted to *Ecological Informatics* (ECOINF-D-25-03829); under revision (R1) |
| 2026 | v1.1.0a1 alpha; documentation expansion (Phases 4–6); submitted to *Environmental Modelling & Software* (October 2025) |

### Core Capabilities of ws3

- **Forest modeling:** `ForestModel` class with development types, actions, transitions, yields, and scenario compilation
- **Optimization:** Model-I linear programming via PuLP/HiGHS (default) or Gurobi (optional)
- **Spatial allocation:** Hybrid aspatial-to-raster workflows via `ForestRaster`
- **Carbon accounting:** Native libCBM linkage for carbon stock and flux estimation
- **Interoperability:** Woodstock-format text file import; GeoTIFF/shapefile I/O
- **Reproducibility:** Deterministic reproduction package, Zenodo-archived releases, FAIR-compliant

### Research Applications

The ws3 framework has been used in:
- Strategic wood supply and bioenergy planning in mixed-wood forests (Cantegril et al., 2019)
- Value-creation potential and supply modeling (Paradis & LeBel, 2018)
- Climate mitigation and carbon-aware planning in BC (Ke, 2024; Yan, 2025)
- Decision-support prototypes for nature-based solutions (avoided fire, avoided harvest)
- Integration with SpaDES for spatial simulation (`spades_ws3` module)
- Backend for web-based decision-support applications (`ecotrust-dss`)

---

## Publications and Submissions

### Peer-Reviewed / Submitted

| Venue | Status | Date | Manuscript ID |
|-------|--------|------|---------------|
| *Ecological Informatics* (Elsevier) | Under revision (R1) | Submitted ~Nov 2025 | ECOINF-D-25-03829 |
| *Environmental Modelling & Software* (Elsevier) | Submitted | October 17, 2025 | — |
| EarthArXiv preprint | Posted | 2025 | — |
| CIRRELT-2017-34 | Technical report | June 2017 | — |
| CIRRELT-2017-43 | Technical report | July 2017 | — |
| CIRRELT-2018-23 | Technical report | May 2018 | — |
| CIRRELT-2018-24 | Technical report | May 2018 | — |

### JOSS Paper

A Journal of Open Source Software (JOSS) paper was prepared for the ws3 package, documenting it as a community-facing software publication.

---

## Funding Sources

- **FORAC Research Consortium** (primary postdoc funding, 2017–2018)
- **NSERC** (grant RGPIN-2023-04197)
- **Environment and Climate Change Canada** (grant 3000770190)
- **Natural Resources Canada** (grant 3000771054)
- **Government of British Columbia** (grant TP24FCCS004)
- **Mitacs** / Newmont Goldcorp Inc. (grant IT32088)
- **Canada Foundation for Innovation** (CFI JELF)
- **British Columbia Knowledge Development Fund** (BCKDF)

---

## Collaborators and Contributors

- **Luc LeBel** (Université Laval, CIRRELT) — Co-author on all four CIRRELT reports
- **Elaheh Ghasemi** — Documentation, testing, code quality
- **Kathleen Coupland** — Documentation, testing, code quality
- **Salar Ghotb** — Documentation, testing, code quality
- **Jinming Ke** — Climate impact assessment research
- **Yancun Yan** — MSc thesis on carbon-aware optimization (Yan, 2025)
- **Boisvenue et al.** — Comparative carbon capacity research

---

## Key Concepts

| Term | Definition |
|------|------------|
| **AAC** | Annual Allowable Cut — maximum sustainable species-wise harvest level |
| **VCP** | Value-Creation Potential — net financial value of standing timber from mill perspective |
| **Disaggregation coefficients** | Matrix converting aggregate volume to log-assortment-level flows |
| **Bilevel model** | Upper-level wood supply optimization linked to lower-level network flow model |
| **Model-I** | Classic strata-based linear programming formulation for harvest scheduling |
| **libCBM** | Canadian Forest Service Carbon Budget Model — carbon stock/flux estimation |
| **ForestModel** | Core ws3 class representing a forest estate with dtypes, actions, transitions, yields |
| **Development type (dtype)** | Aggregate stratum keyed by theme tuples (analysis unit, species, site class, etc.) |
| **Heuristic scheduler** | Area-control priority-queue scheduler (e.g., oldest-first) for aspatial planning |
| **Spatial allocation** | Post-hoc mapping of aspatial schedules to raster cells |

---

## Project Arc (Visual Summary)

```
2017                          2018                          2025+
  │                             │                             │
  ├─ CIRRELT-2017-34            ├─ CIRRELT-2018-23            ├─ ws3 v1.0.5
  │  Diameter distribution      │  VCP retro-fitting          │  EI submission
  │  models for Quebec          │  + software impl            │  (under revision)
  │                             │                             ├─ EMS submission
  ├─ CIRRELT-2017-43            ├─ CIRRELT-2018-24            │  (Oct 2025)
  │  Disaggregation             │  Hybrid sim-opt VCP         ├─ JOSS paper prepared
  │  coefficients               │  estimation (UA 064-51)     ├─ EarthArXiv preprint
  │  (linkage for bilevel)      │                             ├─ Phase 1-5 development
  │                             │                             └─ Phase 6 docs audit
  ▼                             ▼                             ▼
Statistical                    Methodological                  Software
Foundation                     Framework                       Ecosystem
```

---

## References

- Paradis, G., & LeBel, L. (2017). Diameter Distribution Models for Quebec, Canada. *CIRRELT-2017-34*.
- Paradis, G., & LeBel, L. (2017). Compiling Disaggregation Coefficients to Link Long- and Short-Term Planning Models. *CIRRELT-2017-43*.
- Paradis, G., & LeBel, L. (2018). Retro-Fitting Value-Creation Potential Indicators to Long-Term Supply Models. *CIRRELT-2018-23*.
- Paradis, G., & LeBel, L. (2018). Estimating the Value-Creation Potential of Wood Supply Using a Hybrid Simulation-Optimization Approach. *CIRRELT-2018-24*.
- Paradis, G. (2025). WS3: An open-source Python framework for integrated simulation and optimization of forest landscape and wood supply systems. *Submitted to Ecological Informatics* (ECOINF-D-25-03829).
- Paradis, G. (2025). WS3: An open-source Python framework for integrated simulation and optimization of forest landscape and wood supply systems. *Submitted to Environmental Modelling & Software* (October 2025).