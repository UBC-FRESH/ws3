Chapter 2: Forest Inventory and Data Preparation
================================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Describe the structure of forest inventory data and how it maps to ws3
  development types
- Prepare raw inventory data for use with ws3
- Aggregate inventory data into development types
- Validate inventory data for consistency and completeness

What Is Forest Inventory Data?
------------------------------

Forest inventory data is the foundation of any wood supply model. It
describes the current state of the forest: what grows where, how much
of it there is, and what condition it's in.

Inventory data typically comes from:

- **Field surveys**: Sample plots measured by foresters
- **Remote sensing**: LiDAR, aerial photography, satellite imagery
- **Management records**: Past harvest plans, silviculture treatments
- **Provincial databases**: Government forest inventory databases

In British Columbia, the primary source is the **Forest Inventory
Database (FIB)** maintained by the Ministry of Forests. This database
contains hundreds of thousands of inventory records covering the entire
province.

Inventory Data Structure
------------------------

A typical inventory dataset has the following columns:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Column
     - Type
     - Description
   * - ``plot_id``
     - string
     - Unique identifier for the sample plot
   * - ``species``
     - string
     - Dominant tree species (e.g., "Douglas-fir", "Western red cedar")
   * - ``site_index``
     - float
     - Site productivity index (height at reference age)
   * - ``age``
     - int
     - Stand age in years
   * - ``height``
     - float
     - Dominant height in meters
   * - ``basal_area``
     - float
     - Basal area in m²/ha
   * - ``volume``
     - float
     - Merchantable volume in m³/ha
   * - ``stocking``
     - float
     - Stems per hectare
   * - ``canopy_cover``
     - float
     - Canopy closure (0-100%)
   * - ``geometry``
     - geometry
     - Spatial location (point, polygon)

Preparing Data for ws3
----------------------

ws3 expects inventory data in a specific format. The :py:class:`ws3.forest.ForestModel`
class can accept pandas DataFrames directly.

Step 1: Load Your Data

.. code-block:: python

   import pandas as pd

   # Load inventory data from CSV
   inventory = pd.read_csv("forest_inventory.csv")

   # Or load from a GeoJSON file
   inventory = pd.read_json("forest_inventory.geojson")

Step 2: Clean the Data

.. code-block:: python

   # Remove rows with missing critical fields
   inventory = inventory.dropna(subset=["species", "age", "volume"])

   # Standardize species names
   species_map = {
       "Douglas-fir": "Pseudotsuga menziesii",
       "DF": "Pseudotsuga menziesii",
       "Western red cedar": "Thuja plicata",
       "Cedar": "Thuja plicata",
       "Sitka spruce": "Picea sitchensis",
       "Spruce": "Picea sitchensis"
   }
   inventory["species"] = inventory["species"].map(species_map)

Step 3: Define Development Types

.. code-block:: python

   # Create development type codes
   inventory["dt_code"] = (
       inventory["species"] + "-" +
       inventory["site_index"].astype(int).astype(str)
   )

   # Aggregate by development type
   dt_summary = inventory.groupby("dt_code").agg(
       area=("plot_id", "count"),
       mean_age=("age", "mean"),
       mean_volume=("volume", "mean"),
       mean_site_index=("site_index", "mean")
   ).reset_index()

   print(dt_summary)

Step 4: Create the Forest Model

.. code-block:: python

   from ws3.forest import ForestModel

   # Create the model with required parameters
   fm = ForestModel(
       model_name="inventory_model",
       model_path=".",
       base_year=2024,
       horizon=20,
       period_length=10,
       max_age=200
   )

   # Development types are loaded from Woodstock-format data files,
   # not constructed individually. The typical workflow is:
   #
   #   fm.import_areas_section("areas.txt")
   #   fm.import_yields_section("yields.txt")
   #   fm.import_actions_section("actions.txt")
   #   fm.import_transitions_section("transitions.txt")
   #
   # For programmatic construction from a pandas DataFrame, you would
   # iterate over your aggregated development types and populate the
   # model's internal data structures accordingly. See the ws3 source
   # for the import methods above.

   # Access development types via the dtypes dict
   print(f"Development types: {len(fm.dtypes)}")
   # Calculate total area across all periods and development types
   total_area = 0.0
   for period in fm.periods:
       for dtype in fm.dtypes.values():
           total_area += dtype.area(period)
   print(f"Total area: {total_area:.1f} hectares")

Data Validation
---------------

Before running simulations, validate your inventory data:

.. code-block:: python

   # Check for negative areas
   assert (inventory["area"] >= 0).all(), "Negative areas found"

   # Check for reasonable ages
   assert (inventory["age"] >= 0).all(), "Negative ages found"
   assert (inventory["age"] <= 500).all(), "Unreasonably high ages found"

   # Check for consistent species names
   assert inventory["species"].notna().all(), "Missing species names"

   # Check that total area is reasonable
   total_area = inventory["area"].sum()
   print(f"Total inventory area: {total_area:.1f} hectares")

Common Data Issues
------------------

1. **Missing species names**: Replace with "Unknown" or exclude from model
2. **Negative values**: Check for data entry errors
3. **Inconsistent units**: Ensure all volumes are in m³/ha, all ages in years
4. **Duplicate records**: Check for duplicate plot IDs
5. **Outlier values**: Check for unreasonable site indices or volumes

Exercises
---------

**Exercise 1 (Easy)**: Load a sample inventory dataset and create a
ws3 model. Print the development type summary.

**Exercise 2 (Medium)**: Write a function that takes a pandas DataFrame
and returns a list of development types suitable for adding to a ForestModel.

**Exercise 3 (Hard)**: Extend the data cleaning function to handle
missing site indices by estimating them from dominant height and age
using a site index curve.

Further Reading
---------------

- :doc:`ch01_forest_estate_models` — Forest estate model fundamentals
- :doc:`ch03_growth_and_yield` — Growth curve fitting and interpolation
- :doc:`/howto/data-preparation` — Detailed data preparation guide
- :doc:`reference/contracts/index` — Data contracts and module boundaries