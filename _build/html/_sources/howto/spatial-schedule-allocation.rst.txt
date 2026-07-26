.. _howto-spatial-schedule-allocation:

=================
Spatial Schedule Allocation
=================

Goal
----

Allocate harvest schedule to specific landscape areas:

* Map development types to spatial units
* Assign harvest actions to specific locations
* Generate spatially-explicit harvest maps

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with spatial concepts from :doc:`../textbook/ch06_spatial_allocation`
* A working ws3 installation with spatial data

Step-by-Step Instructions
-------------------------

**Step 1: Prepare Spatial Data**

.. code-block:: python

   import geopandas as gpd

   # Load spatial inventory
   spatial_df = gpd.read_file('spatial_inventory.geojson')

   # Ensure development type codes match model
   spatial_df['dt_code'] = spatial_df['stratum_code'] + '_' + spatial_df['age_class'].astype(str)

**Step 2: Load Optimization Results**

.. code-block:: python

   from ws3.opt import solve_optimization

   # Run optimization (see previous how-to guide)
   solution = solve_optimization(model=model, horizon=5, objective='maximize_volume')

   # Get schedule
   schedule = solution.get_schedule()

**Step 3: Match Schedule to Spatial Units**

.. code-block:: python

   # For each period in schedule, find matching spatial units
   for period in schedule['period'].unique():
       period_schedule = schedule[schedule['period'] == period]

       for _, row in period_schedule.iterrows():
           dt_code = row['development_type']
           action_code = row['action']
           area_ha = row['area_ha']

           # Find spatial units matching this development type
           matching_units = spatial_df[spatial_df['dt_code'] == dt_code]

           if len(matching_units) > 0:
               # Allocate area to matching units
               allocated = min(area_ha, matching_units['area_ha'].sum())
               # (Implementation depends on allocation strategy)

**Step 4: Generate Harvest Map**

.. code-block:: python

   # Create harvest map GeoDataFrame
   harvest_map = gpd.GeoDataFrame()

   for period in schedule['period'].unique():
       period_schedule = schedule[schedule['period'] == period]

       for _, row in period_schedule.iterrows():
           dt_code = row['development_type']
           action_code = row['action']

           # Find spatial units
           matching_units = spatial_df[spatial_df['dt_code'] == dt_code]

           # Add to harvest map
           for _, unit in matching_units.iterrows():
               harvest_entry = {
                   'period': period,
                   'development_type': dt_code,
                   'action': action_code,
                   'geometry': unit.geometry,
                   'area_ha': unit['area_ha']
               }
               harvest_map = pd.concat([harvest_map, pd.DataFrame([harvest_entry])], ignore_index=True)

**Step 5: Export Spatial Results**

.. code-block:: python

   # Export to GeoJSON
   harvest_map.to_file('harvest_map.geojson', driver='GeoJSON')

   # Export to shapefile
   harvest_map.to_file('harvest_map.shp', driver='ESRI Shapefile')

Expected Output
---------------

* Spatially-explicit harvest map
* GeoJSON or shapefile with harvest allocations
* Period-by-period harvest locations

Troubleshooting
---------------

**Issue: Spatial units don't match development types**

* Check that dt_code format matches between spatial data and model
* Verify age class codes align
* Ensure species and site index codes match

**Issue: Allocation exceeds available area**

* Check that total schedule area doesn't exceed spatial inventory
* Verify area calculations are correct
* Check for double-counting

**Issue: Performance issues**

* Process periods in batches
* Use spatial indexing for faster lookups
* Consider simplifying geometry

Next Steps
----------

* :doc:`running-optimization` — Run optimization
* :doc:`libcbm-callbacks` — Integrate with libCBM for carbon
* :doc:`financial-scenarios` — Add financial analysis