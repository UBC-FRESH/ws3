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
   import rasterio

   # Load forest inventory as a raster
   raster = ForestRaster(
       input_raster="inventory.tif",
       attribute_column="species"
   )

   print(f"Raster size: {raster.width} x {raster.height} pixels")
   print(f"Pixel size: {raster.res}")
   print(f"Total area: {raster.total_area()} hectares")

Reading Raster Data
-------------------

.. code-block:: python

   # Read the raster data as a numpy array
   data = raster.read()

   # Get unique development types
   unique_dts = raster.get_unique_development_types()
   print(f"Development types: {unique_dts}")

   # Get area by development type
   area_by_dt = raster.area_by_development_type()
   print(area_by_dt)

Spatial Allocation
------------------

Once you have an aspatial harvest target, you can allocate it to specific
pixels:

.. code-block:: python

   # Get aspatial harvest target (e.g., from optimization)
   target_area = 100  # hectares

   # Allocate to pixels
   harvest_map = raster.allocate_harvest(
       target_area=target_area,
       method="greedy",  # Always harvest oldest first
       development_type="DF-SI50"
   )

   # Save harvest map to GeoTIFF
   rasterio.open(
       "harvest_map.tif",
       "w",
       driver="GTiff",
       height=raster.height,
       width=raster.width,
       count=1,
       dtype="int16",
       crs=raster.crs,
       transform=raster.transform
   ).write(harvest_map)

Allocation Methods
------------------

ws3 supports several allocation methods:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Description
   * - ``greedy``
     - Always harvest oldest stands first
   * - ``random``
     - Randomly select pixels
   * - ``closest``
     - Harvest from closest to existing harvest blocks
   * - ``custom``
     - User-defined selection function

Contiguity Constraints
----------------------

For realistic harvest planning, you may want to ensure that harvested
pixels are contiguous (form a single block):

.. code-block:: python

   # Allocate with contiguity constraint
   harvest_map = raster.allocate_harvest(
       target_area=target_area,
       method="contiguous",
       development_type="DF-SI50"
   )

   # Check if harvest is contiguous
   is_contiguous = raster.check_contiguity(harvest_map)
   print(f"Harvest is contiguous: {is_contiguous}")

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
- :doc:`/howto/spatial-schedule-allocation` — Detailed spatial allocation guide
- :doc:`/reference/modules/spatial` — ForestRaster API reference