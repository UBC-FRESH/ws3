.. _contract-module-boundaries:

=========================
Module Boundaries
=========================

This page defines the responsibilities and boundaries of each ws3 module.

Forest Module (:py:mod:`ws3.forest`)
-------------------------------------

**Responsibility**: Model state, development types, actions, and schedule management.

**Owns**:
- :py:class:`ws3.forest.ForestModel` — Central coordinator for all model data
- :py:class:`ws3.forest.DevelopmentType` — Forest stratum (age, area, yield curves, transitions)
- :py:class:`ws3.forest.Action` — Management intervention definition
- :py:class:`ws3.forest.GreedyAreaSelector` — Default area selection strategy

**Key methods on ForestModel**:
- :py:meth:`ws3.forest.ForestModel.__init__` — Constructor
- :py:meth:`ws3.forest.ForestModel.import_areas_section` — Load area data
- :py:meth:`ws3.forest.ForestModel.import_yields_section` — Load yield curves
- :py:meth:`ws3.forest.ForestModel.import_actions_section` — Load action definitions
- :py:meth:`ws3.forest.ForestModel.import_transitions_section` — Load transition rules
- :py:meth:`ws3.forest.ForestModel.register_curve` — Register a Curve with the model
- :py:meth:`ws3.forest.ForestModel.add_problem` — Create and compile an optimization problem
- :py:meth:`ws3.forest.ForestModel.compile_schedule` — Compile schedule from problem solution
- :py:meth:`ws3.forest.ForestModel.apply_schedule` — Apply schedule to model state
- :py:meth:`ws3.forest.ForestModel.compile_product` — Evaluate yield products
- :py:meth:`ws3.forest.ForestModel.inventory` — Query inventory at a period

**Does not own**:
- Curve construction (delegated to :py:mod:`ws3.core`)
- Optimization solving (delegated to :py:mod:`ws3.opt`)
- Spatial allocation (delegated to :py:mod:`ws3.spatial`)

**Dependencies**:
- :py:mod:`ws3.core` for Curve, Node, Tree classes
- :py:mod:`ws3.opt` for Problem, Variable, Constraint classes
- :py:mod:`ws3.common` for constants and utilities

Optimization Module (:py:mod:`ws3.opt`)
----------------------------------------

**Responsibility**: Optimization problem formulation and solving.

**Owns**:
- :py:class:`ws3.opt.Problem` — Optimization problem (variables, constraints, objective)
- :py:class:`ws3.opt.Variable` — Decision variable definition
- :py:class:`ws3.opt.Constraint` — Constraint definition
- Solver dispatch (Gurobi, PuLP, HiGHS)

**Key methods on Problem**:
- :py:meth:`ws3.opt.Problem.__init__` — Constructor (name, sense, solver)
- :py:meth:`ws3.opt.Problem.add_var` — Add a variable
- :py:meth:`ws3.opt.Problem.add_constr` — Add a constraint
- :py:meth:`ws3.opt.Problem.z` — Set/get objective function coefficients
- :py:meth:`ws3.opt.Problem.solve` — Solve the problem
- :py:meth:`ws3.opt.Problem.status` — Get solution status
- :py:meth:`ws3.opt.Problem.merge` — Merge another problem into this one

**Does not own**:
- Model state (reads from :py:mod:`ws3.forest`)
- Schedule compilation (reads from :py:mod:`ws3.forest`)

**Dependencies**:
- :py:mod:`ws3.forest` for model data (via ForestModel.add_problem)

Core Module (:py:mod:`ws3.core`)
---------------------------------

**Responsibility**: Core data structures — curves and dynamic programming state trees.

**Owns**:
- :py:class:`ws3.core.Curve` — Growth/yield curve with interpolation
- :py:class:`ws3.core.Interpolator` — Linear interpolation between curve points
- :py:class:`ws3.core.Node` — State tree node for dynamic programming
- :py:class:`ws3.core.Tree` — Dynamic programming state tree

**Key methods on Curve**:
- :py:meth:`ws3.core.Curve.__init__` — Constructor (label, points, is_volume, period_length, ...)
- :py:meth:`ws3.core.Curve.add_points` — Add data points
- :py:meth:`ws3.core.Curve.simplify` — Simplify curve by removing redundant points
- :py:meth:`ws3.core.Curve.points` — Get current (x,y) point list
- :py:meth:`ws3.core.Curve.__call__` — Interpolate y at given x

**Key methods on Node**:
- :py:meth:`ws3.core.Node.data` — Get/set node data
- :py:meth:`ws3.core.Node.parent` — Get parent node
- :py:meth:`ws3.core.Node.children` — Get child nodes
- :py:meth:`ws3.core.Node.is_root`, :py:meth:`ws3.core.Node.is_leaf` — Tree position queries

**Does not own**:
- Model state (no ForestModel reference)
- Optimization (no Problem reference)

**Dependencies**:
- :py:mod:`ws3.common` for default constants

Spatial Module (:py:mod:`ws3.spatial`)
---------------------------------------

**Responsibility**: Spatial allocation of harvest schedules to rasterized forest inventory.

**Owns**:
- :py:class:`ws3.spatial.ForestRaster` — Raster-based spatial allocation

**Key methods on ForestRaster**:
- :py:meth:`ws3.spatial.ForestRaster.__init__` — Constructor (hdt_map, hdt_func, src_path, snk_path, ...)
- :py:meth:`ws3.spatial.ForestRaster.allocate_schedule` — Allocate schedule to raster
- :py:meth:`ws3.spatial.ForestRaster.commit` — Close output file handles
- :py:meth:`ws3.spatial.ForestRaster.cleanup` — Commit and close input file handle

**Does not own**:
- Optimization (reads schedule from :py:mod:`ws3.forest`)
- Model state (reads development type mapping from :py:mod:`ws3.forest`)

**Dependencies**:
- :py:mod:`ws3.forest` for ForestModel instance and development type mapping
- :py:mod:`rasterio` for GeoTIFF I/O

Common Module (:py:mod:`ws3.common`)
-------------------------------------

**Responsibility**: Shared constants, utilities, and geospatial helpers.

**Owns**:
- Global constants (PERIOD_LENGTH_DEFAULT, HORIZON_DEFAULT, MIN_AGE_DEFAULT, MAX_AGE_DEFAULT, etc.)
- Utility functions (hex_id, is_num, reproject, clean_vector_data)

**Does not own**:
- Curve objects (defined in :py:mod:`ws3.core`)
- Model-specific logic

**Dependencies**:
- None (leaf module)

Integration Module (:py:mod:`ws3.integration`)
-----------------------------------------------

**Responsibility**: Integration with external tools (fhops, FEMIC, FreshForge).

**Owns**:
- :py:class:`ws3.integration.FHOPSIntegrator` — Harvest cost curve generation
- :py:class:`ws3.integration.FEMICIntegrator` — Carbon pool accounting
- :py:class:`ws3.integration.FreshForgeIntegrator` — Workflow automation

**Key methods**:
- :py:meth:`ws3.integration.FEMICIntegrator.get_carbon_pools` — List carbon pools

**Does not own**:
- Core model logic
- Optimization

**Dependencies**:
- :py:mod:`ws3.forest` (optional, for ForestModel integration)

Module Interaction Pattern
--------------------------

.. mermaid::

   graph LR
     FM["ForestModel"] --> OPT["Problem (opt)"]
     FM --> SPA["ForestRaster (spatial)"]
     FM --> CORE["Curve/Node/Tree (core)"]
     FM --> COM["Constants (common)"]
     OPT --> CORE
     OPT --> COM
     SPA --> FM
     INT["Integration"] --> FM