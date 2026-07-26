.. _howto-data-preparation:

=================
Data Preparation
=================

Goal
----

Prepare input data for a ws3 model run, including:

* Inventory data in the correct format
* Development types and their attributes
* Growth curves and yield tables
* Action definitions and transition rules

Prerequisites
-------------

* Completed the :doc:`../getting_started/index` section
* Read chapters 1-3 of the :doc:`../textbook/index`
* A working ws3 installation
* Sample inventory data (CSV or database)

Step-by-Step Instructions
-------------------------

**Step 1: Prepare Inventory Data**

Inventory data should be in a tabular format with columns for:

* Stratum identifiers (e.g., species, site index, age class)
* Area (hectares)
* Volume (cubic meters)
* Basal area (square meters per hectare)

Example CSV structure:

.. code-block:: csv

   stratum_code,species,site_index,age_class,area_ha,vol_m3,ba_m2_ha
   DT001,SP,SI50,10,100.5,250.3,12.5
   DT001,SP,SI50,20,120.2,380.7,18.3
   DT002,HW,SI40,15,85.0,180.2,9.8

**Step 2: Load Data into ws3**

Use the ws3 API to load your inventory:

.. code-block:: python

   from ws3.forest import ForestModel
   import pandas as pd

   # Load inventory data
   df = pd.read_csv('inventory.csv')

   # Create model instance
   model = ForestModel()

   # Add development types from inventory
   for _, row in df.iterrows():
       model.add_development_type(
           code=row['stratum_code'],
           species=row['species'],
           site_index=row['site_index'],
           age=row['age_class'],
           area=row['area_ha']
       )

**Step 3: Define Growth Curves**

Add growth curves for each species/site class combination:

.. code-block:: python

   from ws3.common import GrowthCurve

   # Define a simple volume curve
   curve = GrowthCurve(
       species='SP',
       site_index=50,
       ages=[10, 20, 30, 40, 50, 60, 70, 80],
       volumes=[250, 380, 520, 680, 850, 1020, 1180, 1320]
   )

   model.add_growth_curve(curve)

**Step 4: Define Actions**

Define management actions (harvest, thinning, etc.):

.. code-block:: python

   model.add_action(
       code='CLEARCUT',
       descr='Clearcut harvest',
       components=['volume'],
       transitions={
           'DT001': 'DT001_REGEN'
       }
   )

**Step 5: Define Transitions**

Define post-harvest transitions:

.. code-block:: python

   model.add_development_type(
       code='DT001_REGEN',
       species='SP',
       site_index=50,
       age=0,
       area=0.0
   )

Expected Output
---------------

After completing these steps, you should have:

* A ws3 ForestModel instance with development types
* Growth curves defined for each species/site class
* Actions and transitions configured

Troubleshooting
---------------

**Issue: Missing development types**

* Check that all strata in your inventory are added to the model
* Verify species and site index codes match growth curve definitions

**Issue: Growth curve errors**

* Ensure ages are in ascending order
* Check that volumes are non-negative
* Verify species and site index match development types

**Issue: Transition errors**

* Confirm that target development types exist
* Check that transition codes are unique

Next Steps
----------

* :doc:`curve-definition` — Learn to define custom growth curves
* :doc:`action-definition` — Define complex management actions
* :doc:`running-optimization` — Run your first optimization scenario