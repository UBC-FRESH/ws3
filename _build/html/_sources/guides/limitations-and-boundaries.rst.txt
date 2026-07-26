.. _guide-limitations:

=================
Limitations and Boundaries
=================

This guide honestly documents what ws3 can and cannot do, helping you set
realistic expectations and avoid common pitfalls.

Core Limitations
----------------

**1. Aspatial Core**

ws3's core optimization is aspatial. It optimizes harvest by development type
but does not inherently consider spatial arrangement.

*Implication*: You cannot optimize for contiguous harvest blocks or minimize
edge effects without external spatial allocation.

*Workaround*: Use :py:mod:`ws3.spatial` for post-optimization spatial allocation,
or integrate with SpaDES for spatially-explicit simulation.

**2. Deterministic Growth**

Standard ws3 uses deterministic growth curves. Stochastic growth is not
natively supported in the core optimization.

*Implication*: You cannot directly optimize for risk-adjusted outcomes without
external scenario analysis.

*Workaround*: Run multiple scenarios with different growth parameters and
compare outcomes.

**3. Single-Objective Optimization**

The core solver optimizes a single objective function (volume, NPV, or area
control). Multi-objective optimization requires external frameworks.

*Implication*: You cannot directly trade off volume vs. carbon vs. revenue
in a single optimization.

*Workaround*: Use weighted sums or run separate optimizations for each
objective and compare.

**4. No Built-in Carbon Accounting**

Carbon accounting requires libCBM integration. Without it, ws3 has no
understanding of carbon stocks or fluxes.

*Implication*: You cannot optimize for carbon sequestration without libCBM.

*Workaround*: Install libCBM and use callback system to track carbon.

External Dependencies
---------------------

**libCBM**

*Requirement*: Carbon accounting
*Limitation*: Requires separate installation and configuration
*Alternative*: Use simplified carbon models if libCBM is unavailable

**Gurobi**

*Requirement*: Commercial license for optimization
*Limitation*: Not free for commercial use
*Alternative*: Use open-source solvers (GLPK, CBC) with reduced performance

**SpaDES**

*Requirement*: R installation and spades_ws3 package
*Limitation*: Adds complexity to workflow
*Alternative*: Use aspatial analysis for initial planning

Platform Limitations
--------------------

**Linux/macOS**

Fully supported and tested.

**Windows**

Partially supported. Some users report issues with:
- Path handling in certain configurations
- Reticulate (Python-R bridge) stability
- Parallel processing with multiprocessing

*Recommendation*: Use WSL2 or Docker for Windows users.

Data Limitations
----------------

**Minimum Data Requirements**

ws3 requires:
- Development types with area, age, species, site index
- Growth curves for each species/site index combination
- Actions with transitions

*Insufficient data leads to*: Infeasible problems or unrealistic results.

**Data Quality**

ws3 does not validate data quality beyond basic checks. Garbage in, garbage
out applies.

*Recommendation*: Use :doc:`../howto/model-validation` to check data before
running optimization.

Performance Boundaries
----------------------

**Model Size**

ws3 can handle models with:
- Up to ~1000 development types (tested)
- Planning horizons up to 100 periods (tested)
- Multiple objectives and constraints

*Beyond these limits*: Performance degrades or solver may fail.

*Recommendation*: For very large models, consider:
- Aggregating similar development types
- Reducing planning horizon
- Using decomposition strategies

**Solver Performance**

Typical solve times:
- Small models (<100 DTs): <1 second
- Medium models (100-500 DTs): 1-10 seconds
- Large models (>500 DTs): 10-60 seconds
- Very large models (>1000 DTs): 1-10 minutes

*These are estimates*. Actual times depend on constraint complexity and
solver efficiency.

Known Issues
------------

**Issue 1: Floating-point precision**

*Description*: Area conservation may not be exact due to floating-point
arithmetic.

*Impact*: Small discrepancies (e.g., 0.001 ha) in area balances.

*Workaround*: Round results to reasonable precision (0.01 ha).

**Issue 2: Solver warm-start**

*Description*: Re-solving with small changes may not reuse previous solution
efficiently.

*Impact*: Repeated solves may be slower than expected.

*Workaround*: Use area control mode for rapid scenario testing.

**Issue 3: Callback state persistence**

*Description*: Callback state (e.g., libCBM) may not persist across
simulation restarts.

*Impact*: Need to re-initialize callbacks after restart.

*Workaround*: Save and restore callback state manually.

Unsupported Features
--------------------

The following features are explicitly not supported:

1. **Dynamic land use change** — ws3 assumes fixed forest area
2. **Market price fluctuations** — Prices are static
3. **Climate change adaptation** — No dynamic parameter adjustment
4. **Multi-agent optimization** — Single decision-maker only
5. **Real-time optimization** — Batch processing only

When to Use ws3
---------------

ws3 is a good fit when you need:
- Strategic harvest scheduling over long horizons
- Optimization of harvest volume, revenue, or area control
- Integration with carbon accounting (via libCBM)
- Scenario analysis across multiple objectives

ws3 may not be the right tool when you need:
- Spatially-explicit optimization (use SpaDES or custom spatial models)
- Real-time decision support (ws3 is batch-oriented)
- Short-term operational planning (consider tactical models)
- Multi-scale integration (consider fhops or custom frameworks)

Reporting Issues
----------------

If you encounter issues not covered here:

1. Check the :doc:`troubleshooting` guide for common problems
2. Search GitHub issues for similar reports
3. Create a new issue with:
   - Minimal reproducible example
   - Expected vs. actual behavior
   - System information (Python version, OS, ws3 version)
   - Full error traceback

Further Reading
---------------

- :doc:`troubleshooting` — Common issues and solutions
- :doc:`../howto/model-validation` — Validation procedures
- :doc:`architecture_overview` — Understanding ws3 architecture