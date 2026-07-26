.. _contract-module-boundaries:

=========================
Module Boundaries
=========================

This page defines the responsibilities and boundaries of each ws3 module.

Core Module (:py:mod:`ws3.forest`)
-----------------------------------

**Responsibility**: Model state and development types

**Owns**:
- DevelopmentType objects
- Action definitions
- GrowthCurve objects
- Model configuration

**Does not own**:
- Optimization logic
- Simulation execution
- Spatial allocation

**Key classes**:
- :py:class:`ws3.forest.ForestModel` — Central coordinator
- :py:class:`ws3.forest.DevelopmentType` — Forest stratum
- :py:class:`ws3.forest.Action` — Management intervention

**Dependencies**:
- :py:mod:`ws3.common` for Curve objects

Optimization Module (:py:mod:`ws3.opt`)
----------------------------------------

**Responsibility**: Solve harvest scheduling problems

**Owns**:
- Optimization problem formulation
- Solver integration
- Solution objects

**Does not own**:
- Model state (reads from :py:mod:`ws3.forest`)
- Result visualization

**Key classes**:
- :py:func:`ws3.opt.solve_optimization` — Main solver entry point
- :py:class:`ws3.opt.AreaSelector` — Base class for area selection

**Dependencies**:
- :py:mod:`ws3.forest` for model data

Simulation Module (:py:mod:`ws3.core`)
---------------------------------------

**Responsibility**: Execute simulations

**Owns**:
- Period-by-period simulation loop
- Callback execution
- Result collection

**Does not own**:
- Model state (reads from :py:mod:`ws3.forest`)
- Optimization (reads from :py:mod:`ws3.opt`)

**Key functions**:
- :py:func:`ws3.core.simulate` — Run simulation

**Dependencies**:
- :py:mod:`ws3.forest` for model data
- :py:mod:`ws3.opt` for schedule input

Spatial Module (:py:mod:`ws3.spatial`)
---------------------------------------

**Responsibility**: Spatial allocation of harvest

**Owns**:
- ForestRaster objects
- Spatial allocation algorithms
- Map generation

**Does not own**:
- Optimization (reads from :py:mod:`ws3.opt`)
- Model state (reads from :py:mod:`ws3.forest`)

**Key classes**:
- :py:class:`ws3.spatial.ForestRaster` — Spatial representation

**Dependencies**:
- :py:mod:`ws3.forest` for development type mapping
- :py:mod:`ws3.opt` for schedule input

Common Module (:py:mod:`ws3.common`)
-------------------------------------

**Responsibility**: Shared utilities and data structures

**Owns**:
- Curve objects
- Utility functions
- Base classes

**Does not own**:
- Model-specific logic
- Solver-specific logic

**Key classes**:
- :py:class:`ws3.common.Curve` — Growth curve data structure

**Dependencies**:
- None (leaf module)

Callback System
---------------

Callbacks are registered with the model and called during simulation:

.. code-block:: python

   model.register_callback('carbon', libcbm_callback)

Callback signature:

.. code-block:: python

   def callback(period, development_type, action, area_ha):
       """Callback function signature."""
       pass

Callbacks should:
- Be side-effect free where possible
- Not modify model state directly
- Return results through the callback system

Module Interaction Pattern
--------------------------

.. mermaid::

   graph LR
     FM["ForestModel"] --> OPT["Optimization"]
     FM --> SIM["Simulation"]
     FM --> SPA["Spatial"]
     OPT --> SIM
     OPT --> SPA
     SIM --> SPA
     COM["Common"] --> FM
     COM --> OPT
     COM --> SIM
     COM --> SPA