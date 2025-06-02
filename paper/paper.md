---
title: 'ws3: A Python Framework for Forest Estate Modeling and Wood Supply Simulation'
tags:
  - forest modeling
  - wood supply
  - sustainable forestry
  - simulation
  - optimization
  - Python
authors:
  - name: Gregory Paradis
    orcid: 0000-0001-9618-8797
    affiliation: 1
affiliations:
  - name: Department of Forest Resources Management, Faculty of Forestry, University of British Columbia
    index: 1
date: 2025-05-10
bibliography: paper.bib
---

# Summary
ws3 is an open-source Python package that provides tools for forest estate modeling and wood supply simulation. The package allows researchers and practitioners to analyze and predict behavior of forest ecosystems under different management scenarios. 

Through its modular architecture and extensive documentation, ws3 facilitates collaboration among stakeholders and encourages the sharing of best practices in forest estate modeling and wood supply simulation. As a community-driven project, ws3 aims to foster innovation and advance our understanding of complex forestry systems.

# Statement of Need
ws3 addresses a current need in forest management by providing an open source framework for forest estate modeling and wood supply simulation.  

To meet this need, ws3 provides a suite of open tools for reproductible forest estate modelling. The software includes features such as:

*   A flexible and extensible framework for representing forest estate modelling problems, including functions for importing from and to industry-standard data formats.
*   An optimization module that enables the identification of optimal activity schedules under different combinations of management objectives and constraints.
*   A spatial module module that allows for interoperability with external tools that export or consume spatial data in raster pixel format.  
*   Linkages to `libcbm_py`, facilitating integration of forest carbon flow analysis.

The development of ws3 has been guided by a collaborative effort among researchers, policymakers, and practitioners from the forestry sector. 

While many forest estate modeling tools exist, they are often proprietary, inflexible, or specialized to narrow jurisdictions---these tools are less well suited to research applications that aspire to implement the PERFICT concept for ecological modelling [@mcintire2022perfict].  ws3 addresses this gap by providing a transparent, open-source alternative that emphasizes reproducibility, extensibility, and accessibility. Its design supports both experimental model development and production-scale scenario analysis. 

# Implementation and Architecture

ws3 is structured as a set of modular Python packages, each corresponding to a distinct modeling role:

- The `forest` module defines classes that represent forest stands and their management history. This is the main module in the ws3 package.
- The `opt` module defines classes that provide a generic representation of linear programming (LP) optimization problem components (e.g., variables, objective function, constraints, etc.), and wrapper functions for interfacing with various LP solvers (uses open source PuLP solver by default, but also includes a Gurobi solver interface).
- The `spatial` module provides tools for applying aspatial schedules to rasterized landscapes, facilitating hybrid spatial/aspatial modeling.
- The `core` and `common` modules define shared utility classes and interpolation functions used across the system.

A ws3 model is constructed by instantiating an object of the `ws3.forest.ForestModel` class and then populating it with relevant input data and parameters, and running one or more scenarios simulating various forest management policy options and their long-term impacts.

The system is designed for clarity and transparency. Each core function is unit tested and documented. ws3 supports both interactive experimentation (e.g., via Jupyter notebooks) and batch processing workflows in various environments (including linux-based servers and workstations). 

# Features

ws3 provides the following capabilities:

- **Flexible model composition**: Users can combine rule-based logic with optimization-based scheduling.
- **Long-term scenario simulation**: Supports multi-period planning over decadal time horizons.
- **Hybrid spatial/aspatial modeling**: Maps aspatial results to spatial rasters, enabling visualization and spatial policy evaluation.
- **Optimization engine integration**: Built-in support for generating and solving mathematical programs for harvest scheduling.
- **Extensible Python codebase**: Modular architecture allows easy customization and extension.
- **Reproducible workflows**: Models are specified through configuration files and scripts designed to support rigorous scenario analysis and transparency.

ws3 is suitable for research into sustainable yield, carbon accounting, biodiversity conservation, and land-use policy. Its design enables researchers and practitioners to build and compare competing strategies using consistent assumptions and datasets.

ws3 has been used in a number of past research projects [@paradis2018estimating; @cantegril2019bioenergy; @smyth2020climate; @blackburn2020applied; @boisvenue2022managing; @ke2024climate; @yan2025mathematical], has been wrapped in R for use in the SpaDES environment ([spades_ws3](https://github.com/UBC-FRESH/spades_ws3/)), is used in a number of new and ongoing projects, and is used in the backend of multiple cloud-based web applications (e.g., [ecotrust_dss](https://github.com/UBC-FRESH/ecotrust-dss)).


# Acknowledgements

Development of ws3 is led by the FRESH Lab at the University of British Columbia (UBC). The project is supported by funding from the Canada Foundation for Innovation (CFI JELF) and the British Columbia Knowledge Development Fund (BCKDF), Mitacs, Environment and Climate Change Canada (ECCC), the National Science and Engineering Research Council (NSERC) of Canada, Natural Resources Canada (NRCan), and the Government of British Columbia. We thank our students, research assistants, and partners who contributed model components, testing use cases, and domain expertise.

# References

<!-- Example BibTeX references should be placed in a separate `paper.bib` file -->

