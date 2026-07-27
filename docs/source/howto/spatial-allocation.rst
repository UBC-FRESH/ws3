.. _howto-spatial-allocation:

=============================
Spatial Schedule Allocation
=============================

Goal
----

Allocate an aspatial harvest schedule to a rasterized forest inventory.

Prerequisites
-------------

* Completed :doc:`running-optimization`
* A rasterized forest inventory (GeoTIFF with theme, age, and block ID layers)

Step-by-Step Instructions
-------------------------

**Step 1: Prepare Spatial Data**

.. code-block:: python

   from ws3.spatial import ForestRaster

   # Define development type mapping
   hdt_map = {
       1: ('TSA24', 'CWHvm1', 1, 'DWG', 'curve_001'),
       2: ('TSA24', 'CWHvm1', 1, 'SP', 'curve_002'),
   }

   # Define hash function
   def hdt_func(key):
       return hash(key) % 1000000

**Step 2: Create ForestRaster Instance**

.. code-block:: python

   raster = ForestRaster(
       hdt_map=hdt_map,
       hdt_func=hdt_func,
       src_path="path/to/landscape.tif",
       snk_path="path/to/output",
       acode_map={"harvest": "harvest"},
       forestmodel=fm,
       base_year=2020,
       horizon=10,
       period_length=10
   )

**Step 3: Allocate Schedule**

.. code-block:: python

   # Allocate harvest schedule to raster
   raster.allocate_schedule(problem.solution())

**Step 4: Export Results**

.. code-block:: python

   # Export allocated schedule to GeoTIFF
   raster.export_schedule()

Expected Output
---------------

* Spatially allocated harvest schedule
* GeoTIFF output with harvest prescriptions

Troubleshooting
---------------

**Issue: Allocation fails**

* Check that raster dimensions match model extent
* Verify development type mapping is correct
* Ensure schedule is feasible

**Issue: Output files missing**

* Check that output directory exists and is writable
* Verify rasterio is installed

Next Steps
----------

* :doc:`running-optimization` — Run optimization scenarios
* :doc:`multi-objective-optimization` — Run multi-objective scenarios