.. _howto-data-validation:

=====================
Data Validation and Quality Assurance
=====================

Goal
----

Validate input data quality and consistency before running optimizations:

* Check for missing or invalid data
* Validate development type codes and yield curves
* Ensure spatial data consistency
* Verify optimization parameters

Prerequisites
-------------

* Completed :doc:`data-preparation`
* Familiarity with data validation concepts
* Understanding of forest inventory data structures

Common Data Issues
------------------

Forest inventory data often has:

* **Missing values**: Null or zero values in critical fields
* **Inconsistent codes**: Development type codes that don't match yield curves
* **Spatial errors**: Overlapping polygons, gaps in coverage
* **Temporal inconsistencies**: Ages outside reasonable ranges
* **Yield curve issues**: Missing ages, negative volumes, unrealistic growth

Step-by-Step Instructions
-------------------------

**Step 1: Load and Inspect Data**

.. code-block:: python

   import pandas as pd
   import ws3.forest
   
   # Load model
   fm = ws3.forest.ForestModel(
       model_name="validation_test",
       model_path="data/woodstock_model_files"
   )
   
   fm.import_landscape_section()
   fm.import_yields_section()
   fm.import_actions_section()
   
   # Inspect development types
   print("Development Types Shape:", fm.development_types.shape)
   print("Development Types Columns:", fm.development_types.columns.tolist())
   print("\nFirst 5 rows:")
   print(fm.development_types.head())
   
   # Check for missing values
   print("\nMissing values in development types:")
   print(fm.development_types.isnull().sum())

**Step 2: Validate Development Type Codes**

.. code-block:: python

   # Check for duplicate development type codes
   duplicates = fm.development_types[fm.development_types.duplicated(
       subset=['code'], keep=False
   )]
   
   if not duplicates.empty:
       print(f"WARNING: Found {len(duplicates)} duplicate development type codes")
       print(duplicates[['code', 'area', 'age']])
   else:
       print("✓ No duplicate development type codes")
   
   # Check for invalid area values
   invalid_area = fm.development_types[
       (fm.development_types['area'] < 0) | 
       (fm.development_types['area'].isnull())
   ]
   
   if not invalid_area.empty:
       print(f"WARNING: Found {len(invalid_area)} development types with invalid area")
   else:
       print("✓ All area values are valid")
   
   # Check for unrealistic ages
   unrealistic_ages = fm.development_types[
       (fm.development_types['age'] < 0) | 
       (fm.development_types['age'] > 500)
   ]
   
   if not unrealistic_ages.empty:
       print(f"WARNING: Found {len(unrealistic_ages)} development types with unrealistic ages")
   else:
       print("✓ All ages are within reasonable range")

**Step 3: Validate Yield Curves**

.. code-block:: python

   # Check yield curve coverage
   print(f"Number of yield curves: {len(fm.yields)}")
   
   # Check for development types without yield curves
   dt_codes = set(fm.development_types['code'].tolist())
   yield_keys = set(fm.yields.keys())
   
   missing_yields = dt_codes - yield_keys
   if missing_yields:
       print(f"WARNING: {len(missing_yields)} development types missing yield curves")
       print("Missing codes:", list(missing_yields)[:10])
   else:
       print("✓ All development types have yield curves")
   
   # Check yield curve data quality
   for key, curve in list(fm.yields.items())[:5]:  # Check first 5
       if len(curve) == 0:
           print(f"WARNING: Empty yield curve for {key}")
       elif curve['volume'].min() < 0:
           print(f"WARNING: Negative volumes in yield curve {key}")
       elif curve['age'].min() < 0:
           print(f"WARNING: Negative ages in yield curve {key}")

**Step 4: Validate Spatial Data**

.. code-block:: python

   import geopandas as gpd
   
   # Load spatial data if available
   try:
       spatial_df = gpd.read_file("data/spatial_inventory.geojson")
       
       # Check for overlapping polygons
       overlaps = spatial_df.geometry.overlaps(spatial_df.geometry)
       if overlaps.any():
           print(f"WARNING: Found {overlaps.sum()} overlapping polygons")
       else:
           print("✓ No overlapping polygons")
       
       # Check for gaps (uncovered areas)
       total_area = spatial_df.geometry.area.sum()
       management_area = fm.areas['area_ha'].sum() if 'area_ha' in fm.areas.columns else 0
       
       if abs(total_area - management_area) / management_area > 0.01:
           print(f"WARNING: Spatial area ({total_area:.2f} ha) differs from management area ({management_area:.2f} ha)")
       else:
           print("✓ Spatial and management areas match")
           
   except Exception as e:
       print(f"Could not validate spatial data: {e}")

**Step 5: Validate Optimization Parameters**

.. code-block:: python

   # Check planning horizon
   if fm.horizon < 1:
       print("WARNING: Planning horizon must be at least 1 period")
   else:
       print(f"✓ Planning horizon: {fm.horizon} periods")
   
   # Check period length
   if fm.period_length <= 0:
       print("WARNING: Period length must be positive")
   else:
       print(f"✓ Period length: {fm.period_length} years")
   
   # Check max age
   if fm.max_age < fm.horizon * fm.period_length:
       print(f"WARNING: Max age ({fm.max_age}) should be at least horizon * period_length ({fm.horizon * fm.period_length})")
   else:
       print(f"✓ Max age: {fm.max_age} years")

**Step 6: Run Comprehensive Validation**

.. code-block:: python

   def validate_model(fm):
       """Run comprehensive model validation."""
       issues = []
       
       # Check development types
       if fm.development_types.empty:
           issues.append("No development types loaded")
       elif fm.development_types['area'].sum() == 0:
           issues.append("Total area is zero")
       
       # Check yield curves
       if len(fm.yields) == 0:
           issues.append("No yield curves loaded")
       
       # Check actions
       if 'harvest' not in fm.actions:
           issues.append("No harvest action defined")
       
       # Check spatial data
       if not hasattr(fm, 'spatial_df') or fm.spatial_df.empty:
           issues.append("No spatial data loaded")
       
       return issues
   
   issues = validate_model(fm)
   
   if issues:
       print("VALIDATION ISSUES FOUND:")
       for issue in issues:
           print(f"  • {issue}")
   else:
       print("✓ All validation checks passed!")

Expected Output
---------------

* Comprehensive validation report
* List of issues found (if any)
* Recommendations for fixing issues
* Confidence in data quality

Troubleshooting
---------------

**Issue: Missing yield curves for some development types**

* Solution: Check yield curve file format and paths
* Solution: Verify development type codes match between inventory and yields
* Solution: Use wildcard matching in yield curve keys

**Issue: Spatial data doesn't match management areas**

* Solution: Check for clipping errors in spatial data
* Solution: Verify coordinate reference systems match
* Solution: Recalculate areas from spatial data

**Issue: Unrealistic ages or volumes**

* Solution: Check inventory data source and processing
* Solution: Verify growth curve parameters
* Solution: Add data filtering before model creation

Best Practices
--------------

1. **Validate Early**: Run validation before optimization to catch issues early
2. **Automate**: Include validation in your workflow scripts
3. **Document**: Keep track of data quality issues and resolutions
4. **Test**: Validate on sample data before running full models
5. **Monitor**: Track validation results over time to detect data degradation

Related Resources
-----------------

* :doc:`data-preparation`
* :doc:`model-validation`
* :doc:`../textbook/ch02_forest_inventory`
* GeoPandas documentation: https://geopandas.org/