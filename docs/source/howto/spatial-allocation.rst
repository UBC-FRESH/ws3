.. _howto-spatial-allocation:

=============================
Spatial Schedule Allocation
=============================

This guide shows how to allocate an aspatial harvest schedule to a
rasterized forest inventory.

Prerequisites
-------------

* A loaded :ref:`ForestModel <howto-loading-model>`
* A solved :ref:`Problem <howto-running-optimization>`
* A rasterized forest inventory (GeoTIFF with theme, age, and block ID layers)

Procedure
---------

**1. Prepare spatial data**

.. code-block:: python

   from ws3.spatial import ForestRaster

   hdt_map = {
       1: ('TSA24', 'CWHvm1', 1, 'DWG', 'curve_001'),
       2: ('TSA24', 'CWHvm1', 1, 'SP', 'curve_002'),
   }

   def hdt_func(key):
       return hash(key) % 1000000

**2. Create a ForestRaster instance**

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

**3. Allocate the schedule**

.. code-block:: python

   raster.allocate_schedule(problem.solution())

**4. The allocated schedule is written to the output directory**

The raster writes output files (typically GeoTIFF or schedule files) to
the ``snk_path`` directory specified during construction.

Troubleshooting
---------------

* **Allocation fails** — verify raster dimensions match model extent and
  that the development type mapping is correct.
* **Output files missing** — confirm the output directory exists and is
  writable, and that ``rasterio`` is installed.