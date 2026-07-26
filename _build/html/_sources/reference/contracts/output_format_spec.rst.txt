.. _contract-output-format:

=========================
Output Format Spec
=========================

This page documents the output formats produced by ws3.

Harvest Schedule
----------------

The primary output is a harvest schedule in DataFrame format.

Columns:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Column
     - Type
     - Description
   * - period
     - int
     - Planning period (0-indexed)
   * - development_type
     - str
     - Source development type code
   * - action
     - str
     - Action code
   * - area_ha
     - float
     - Area harvested (hectares)
   * - volume_m3
     - float
     - Volume harvested (cubic meters)
   * - npv
     - float (optional)
     - Net present value contribution

Example output:

.. code-block:: text

   period,development_type,action,area_ha,volume_m3,npv
   0,DT001,CLEARCUT,50.0,1250.0,45000.0
   0,DT002,CLEARCUT,30.0,900.0,32000.0
   1,DT001,CLEARCUT,45.0,1100.0,38000.0

Export Formats
--------------

The schedule can be exported to:

- **CSV**: :code:`schedule.to_csv('output.csv')`
- **Excel**: :code:`schedule.to_excel('output.xlsx')`
- **JSON**: :code:`schedule.to_json('output.json')`

Summary Statistics
------------------

The solution object provides summary statistics:

.. code-block:: python

   summary = solution.get_summary()

   # Total harvest
   total_area = summary['total_area_ha']
   total_volume = summary['total_volume_m3']

   # Per-period statistics
   period_stats = summary['period_stats']

   # Financial metrics (if applicable)
   npv = summary['npv']

Callback Results
----------------

Results from callbacks (e.g., carbon tracking):

.. code-block:: python

   carbon_results = solution.get_callback_results('carbon')

   # Carbon time series
   carbon_series = carbon_results['carbon_stock']

   # Carbon flux from harvest
   carbon_flux = carbon_results['carbon_flux']

Spatial Output
--------------

When spatial allocation is performed:

.. code-block:: python

   harvest_map = solution.get_spatial_output()

   # Export to GeoJSON
   harvest_map.to_file('harvest_map.geojson', driver='GeoJSON')

   # Export to shapefile
   harvest_map.to_file('harvest_map.shp', driver='ESRI Shapefile')

Error Handling
--------------

If optimization fails, the solution object will contain error information:

.. code-block:: python

   if not solution.is_feasible():
       print(f"Solver status: {solution.solver_status}")
       print(f"Message: {solution.solver_message}")

Common error statuses:

- "infeasible" — No solution satisfies all constraints
- "unbounded" — Objective can be improved indefinitely
- "error" — Solver encountered an error
- "optimal" — Optimal solution found

Validation
----------

Use :doc:`../howto/model-validation` to validate output against expectations.