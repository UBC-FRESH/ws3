Chapter 6: Spatial Allocation
=============================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain the difference between aspatial and spatial wood supply models
- Use the :py:class:`ws3.spatial.ForestRaster` class for spatial allocation
- Understand how raster data represents forest inventory
- Allocate harvest targets to specific pixels in a landscape

Aspatial vs. Spatial Models
---------------------------

Wood supply models can be either **aspatial** or **spatial**:

**Aspatial models** (default in ws3):
- Track area by development type, not by location
- Assume all stands of the same type are identical
- Faster to solve, simpler to understand
- Good for regional-scale planning

**Spatial models**:
- Track area by location (pixel-by-pixel)
- Account for spatial constraints (contiguity, adjacency)
- Slower to solve, more complex
- Good for landscape-scale planning

The :py:class:`ws3.spatial.ForestRaster` class adds spatial capability
to ws3.

Forest Raster Data
------------------

A **raster** is a grid of pixels (cells), where each pixel has a value.
In forest inventory, each pixel might represent:

- The dominant species at that location
- The age class of the stand
- The volume per hectare
- The development type code

.. mermaid::

   graph TD
     GIS["GIS Data<br/>(shapefiles, LiDAR)"] --> RASTERIZE["Rasterize<br/>to grid"]
     RASTERIZE --> RASTER["ForestRaster<br/>pixel grid"]
     RASTER --> ALLOC["Spatial allocation<br/>assign harvest"]
     ALLOC --> OUTPUT["Harvest map<br/>spatial output"]

Creating a Forest Raster
------------------------

.. code-block:: python

   from ws3.spatial import ForestRaster

   # ForestRaster requires a pre-rasterized inventory GeoTIFF and a
   # ForestModel instance. The constructor signature is:
   #
   #   ForestRaster(hdt_map, hdt_func, src_path, snk_path,
   #                acode_map, forestmodel, base_year, ...)
   #
   # Where:
   #   hdt_map    - dict mapping hash values to development type tuples
   #   hdt_func   - function to hash development type tuples
   #   src_path   - path to input GeoTIFF (rasterized inventory)
   #   snk_path   - directory for output GeoTIFF files
   #   acode_map  - dict mapping disturbance codes to output prefixes
   #   forestmodel - a ForestModel instance
   #   base_year  - base year for output file naming
   #
   # Example:
   #
   #   raster = ForestRaster(
   #       hdt_map=hdt_map,
   #       hdt_func=hdt_func,
   #       src_path="data/inventory.tif",
   #       snk_path="output/spatial",
   #       acode_map={"HARV": "harv", "THIN": "thin"},
   #       forestmodel=model,
   #       base_year=2024
   #   )

Spatial Allocation via allocate_schedule
----------------------------------------

Once you have an aspatial harvest target (e.g., from optimization),
you can allocate it to specific pixels using ``allocate_schedule``:

.. code-block:: python

   # The allocate_schedule method takes a schedule (list of
   # (dtype_key, age, area, acode, period, etype) tuples) and
   # allocates it to the raster.
   #
   # Example schedule from optimization:
   #   schedule = model.compile_schedule(problem)
   #
   # With context manager:
   #   with ForestRaster(...) as raster:
   #       raster.allocate_schedule(schedule)
   #       raster.commit()
   #
   # Output GeoTIFF files are created automatically in snk_path,
   # one per combination of disturbance type and time step.

Spatial Constraints
-------------------

Common spatial constraints include:

- **Contiguity**: Harvested pixels must form a single block
- **Adjacency**: Harvested pixels must be adjacent to existing roads
- **Setback**: Harvested pixels must be a minimum distance from water bodies
- **Block size**: Harvest blocks must be between minimum and maximum sizes

These constraints can be enforced by modifying the allocation algorithm
or by post-processing the harvest map.

Performance Considerations
--------------------------

Spatial allocation is computationally intensive. Tips for improving
performance:

1. **Reduce resolution**: Use coarser pixel sizes for large landscapes
2. **Subset the area**: Only allocate to relevant development types
3. **Use parallel processing**: ws3's :py:class:`ws3.forest_helper.PersistentWorkerPool`
   can parallelize allocation across multiple processors
4. **Cache intermediate results**: Avoid recalculating the same values

Exercises
---------

**Exercise 1 (Easy)**: Load a forest inventory raster and print the
area by development type.

**Exercise 2 (Medium)**: Allocate a harvest target of 50 hectares to
a raster using the greedy method. Save the harvest map to a GeoTIFF.

**Exercise 3 (Hard)**: Modify the allocation algorithm to enforce a
minimum block size of 10 hectares (no harvest blocks smaller than 10 ha).

Further Reading
---------------

- :doc:`ch05_optimization` — Optimization fundamentals
- :doc:`/howto/spatial-allocation` — Spatial schedule allocation guide
- :doc:`/reference/contracts/index` — Data contracts and module boundaries