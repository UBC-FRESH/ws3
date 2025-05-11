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
    orcid: 0000-0002-XXXX-XXXX
    affiliation: 1
affiliations:
  - name: Faculty of Forestry, University of British Columbia
    index: 1
date: 2025-05-10
bibliography: paper.bib
---

# Summary
ws3 is an open-source Python framework that provides a comprehensive set of tools for forest estate modeling and wood supply simulation. This innovative platform enables researchers, policymakers, and practitioners to analyze and predict the behavior of forest ecosystems under different management scenarios, making it an invaluable resource for sustainable forestry practices.

By integrating aspatial forest esatate modeling with optimization techniques, ws3 empowers users to evaluate the impact of various harvesting strategies on forest productivity, biodiversity, and ecosystem services. This cutting-edge framework is particularly useful for addressing pressing issues in sustainable forestry, such as optimizing wood supply chains while minimizing environmental degradation and promoting reforestation efforts.

Through its modular architecture and extensive documentation, ws3 facilitates collaboration among stakeholders and encourages the sharing of best practices in forest estate modeling and wood supply simulation. As a community-driven project, ws3 aims to foster innovation and advance our understanding of complex forestry systems.

# Statement of Need
ws3 addresses a pressing need in forest management by providing a comprehensive framework for forest estate modeling and wood supply simulation. Current modeling approaches often rely on simplified assumptions or neglect important interactions, leading to inadequate predictions and suboptimal decision-making.
To meet this need, ws3 provides a suite of tools for modeling, optimization, and scenario evaluation. The software includes features such as:

*   A flexible and extensible framework for incorporating various forest growth models and ecosystem services
*   An optimization module that enables the identification of optimal harvesting strategies under different management objectives
*   A set of tools for evaluating the impact of various management scenarios on forest productivity, biodiversity, and ecosystem services

The development of ws3 has been guided by a collaborative effort among researchers, policymakers, and practitioners from the forestry sector. As such, the software is designed to be user-friendly, flexible, and extensible, allowing users to adapt it to their specific needs and contexts.

By providing a robust and reliable framework for forest estate modeling and wood supply simulation, ws3 aims to support the development of more effective forest management strategies that balance economic, social, and environmental objectives. We believe that this contribution will have significant implications for the forestry sector and contribute to the sustainable management of forest resources.

Forest estate modeling is a core method used by forest planners, researchers, and policy-makers to assess the implications of different forest management strategies. These models help answer questions such as: How much timber can be harvested sustainably? What are the trade-offs between economic yield and conservation objectives?

While many forest estate modeling tools exist, they are often proprietary, inflexible, or specialized to narrow jurisdictions. ws3 addresses this gap by providing a transparent, open-source alternative that emphasizes reproducibility, extensibility, and accessibility. Its design supports both experimental model development and production-scale scenario analysis. By using Python, ws3 leverages a widely used ecosystem of scientific computing tools, enabling integration with data processing, spatial analysis, and optimization workflows.

# Implementation and Architecture

ws3 is structured as a set of modular Python packages, each corresponding to a distinct modeling role:

- The `forest` module defines classes that represent forest stands and their management history. This is the main module in the ws3 package.
- The `opt` module defines classes that provide a generic representation of linear programming (LP) optimization problem components (e.g., variables, objective function, constraints, etc.), and wrapper functions for interfacing with various LP solvers (uses open source PuLP solver by default, but also includes a Gurobi solver interface).
- The `spatial` module provides tools for applying aspatial schedules to rasterized landscapes, facilitating hybrid spatial/aspatial modeling.
- The `core` and `common` modules define shared utility classes and interpolation functions used across the system.

A ws3 model is constructed by defining a simulation schedule (e.g., harvest rules, regeneration logic), input datasets (e.g., forest inventories, yield tables), and simulation parameters (e.g., planning horizon, time steps). Simulations proceed by advancing time in user-defined steps, applying scheduled management and natural processes, and updating stand states. Output data—such as harvested volumes, area changes, or stand-level attributes—can be written to structured CSVs or other formats for analysis.

The system is designed for clarity and transparency. Each core function is unit tested, and model configurations are explicitly defined and version-controllable. ws3 supports both interactive experimentation (e.g., via Jupyter notebooks) and batch processing workflows. 

# Features

ws3 provides the following capabilities:

- **Flexible model composition**: Users can combine rule-based logic with optimization-based scheduling.
- **Long-term scenario simulation**: Supports multi-period planning over decadal time horizons.
- **Hybrid spatial/aspatial modeling**: Maps aspatial results to spatial rasters, enabling visualization and spatial policy evaluation.
- **Optimization engine integration**: Built-in support for generating and solving mathematical programs for harvest scheduling.
- **Extensible Python codebase**: Modular architecture allows easy customization and extension.
- **Reproducible workflows**: Models are specified through configuration files and scripts designed to support rigorous scenario analysis and transparency.

ws3 is suitable for research into sustainable yield, carbon accounting, biodiversity conservation, and land-use policy. Its design enables researchers and practitioners to build and compare competing strategies using consistent assumptions and datasets.



ws3 has been used in a number of past and ongoing research projects. ws3 was first used to model the value-creation potential of the wood supply in multiple management units in Quebec, Canada 
@smyth2020climate. 
<!--
This early application of ws3 was later extended to model potential benefits of adding a bioenergy facility to a regional supply chain in Quebec, Canada [@cantegril2019potentiel; @cantegril2019bioenergy]. The ws3 package was later extended to include LP optimization and spatial disturbance allocation functions and used to model several optimal management scenarios covering the entire province of British Columbia, Canada in @smythe2020climate. ws3 has also been used to simulate forest harvesting disturbances in an open software framework for simulating forest invasive alien species establishment and spread [@blackburn2020applied]. @boisvenue2022managing used ws3 to help model the carbon carrying capacity of over 50 million hectares of boreal forest. @ke2024climate used ws3 to estimate climate impact of various forest fertilization and harvesting scenarios. @yan2025mathematical linked ws3 to libcbm_py (an open source Python forest carbon budget modelling software package), making it possible use ws3 to optimize the climate change mitigation potential of forest management activities.
-->

# Acknowledgements

Development of ws3 is led by the FRESH Lab at the University of British Columbia (UBC). The project is supported by funding from the Canada Foundation for Innovation (CFI JELF) and the British Columbia Knowledge Development Fund (BCKDF), Mitacs, Environment and Climate Change Canada (ECCC), the National Science and Engineering Research Council (NSERC) of Canada, Natural Resources Canada (NRCan), and the Government of British Columbia. We thank our students, research assistants, and partners who contributed model components, testing use cases, and domain expertise.

# References

<!-- Example BibTeX references should be placed in a separate `paper.bib` file -->

