.. _contract-output-format:

=========================
Output Format Spec
=========================

This page documents the output formats produced by ws3.

Schedule Output
---------------

The primary output is a harvest schedule compiled via
:py:meth:`ws3.forest.ForestModel.compile_schedule`. The schedule is a list of
tuples, each with the format ``(dtype_key, age, area, acode, period, etype)``.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Element
     - Type
     - Description
   * - dtype_key
     - tuple[str, ...]
     - Development type key (tuple of theme values)
   * - age
     - int
     - Age at which action was applied
   * - area
     - float
     - Area harvested (hectares)
   * - acode
     - str
     - Action code
   * - period
     - int
     - Planning period (1-indexed)
   * - etype
     - str
     - ``'_existing'`` (area existed before action) or ``'_future'`` (area created by action)

Example:

.. code-block:: python

   schedule = model.compile_schedule(problem)
   # schedule is a list of tuples:
   # [('SP', 50, 'T1'), 30, 5.0, 'harvest', 1, '_existing']

Scenario DataFrame
------------------

Scenarios are compiled into DataFrames via the user-defined
:py:func:`docs.source.examples.util.compile_scenario` helper function (not a
built-in ws3 API). The resulting DataFrame has columns:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - period
     - int
     - Planning period
   * - oha
     - float
     - Harvested area (ha)
   * - ohv
     - float
     - Harvested volume (m³)
   * - ogs
     - float
     - Growing stock (m³)

Export Formats
--------------

The schedule list can be converted to a DataFrame and exported:

.. code-block:: python

   import pandas as pd
   df = pd.DataFrame(schedule, columns=['dtype_key', 'age', 'area', 'acode', 'period', 'etype'])
   df.to_csv('output.csv', index=False)
   df.to_excel('output.xlsx', index=False)

Problem Solution
----------------

The :py:class:`ws3.opt.Problem` instance stores the optimal solution after
calling :py:meth:`ws3.opt.Problem.solve`. Access solution values via:

.. code-block:: python

   problem.solve()
   if problem.solved():
       # Variable values:
       for var_name in problem.var_names():
           var = problem.var(var_name)
           print(var_name, var.val)
       # Constraint LHS values:
       lhs = problem.get_all_constraints_lhs_values()

Spatial Output
--------------

When spatial allocation is performed via :py:class:`ws3.spatial.ForestRaster`,
output is written as GeoTIFF files (one per action code per period). The raster
instance manages file handles internally and writes to the directory specified
by ``snk_path`` in the constructor.

.. code-block:: python

   with ForestRaster(
       hdt_map=hdt_map,
       hdt_func=hdt_func,
       src_path='inventory.tif',
       snk_path='output_dir',
       acode_map={'harvest': 'harv'},
       forestmodel=model,
       base_year=2020,
   ) as raster:
       raster.allocate_schedule()
   # GeoTIFF files are written to output_dir/

Error Handling
--------------

Solver status is accessible via :py:meth:`ws3.opt.Problem.status`:

.. code-block:: python

   problem.solve()
   status = problem.status()
   # Returns: 'optimal', 'infeasible', 'unbounded', or None

Common statuses:

- ``'optimal'`` — Optimal solution found
- ``'infeasible'`` — No solution satisfies all constraints
- ``'unbounded'`` — Objective can be improved indefinitely
- ``None`` — Problem not solved or solver unavailable

Carbon Accounting
-----------------

Carbon pool information is available via
:py:meth:`ws3.integration.FEMICIntegrator.get_carbon_pools`:

.. code-block:: python

   from ws3.integration import FEMICIntegrator
   femic = FEMICIntegrator()
   pools = femic.get_carbon_pools()
   # Returns: ['above_ground_biomass', 'below_ground_biomass', 'deadwood',
   #           'litter', 'soil_organic_matter', 'harvested_product']

Validation
----------

Use :doc:`/howto/running-optimization` to validate output against expectations.