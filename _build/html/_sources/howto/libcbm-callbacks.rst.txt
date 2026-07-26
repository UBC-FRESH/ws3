.. _howto-libcbm-callbacks:

=================
libCBM Callbacks Integration
=================

Goal
----

Integrate ws3 with libCBM for carbon accounting:

* Set up libCBM callbacks in ws3
* Track carbon stocks and fluxes
* Include carbon in optimization objectives

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with carbon concepts from :doc:`../textbook/ch10_carbon_modelling`
* A working ws3 installation with libCBM

Step-by-Step Instructions
-------------------------

**Step 1: Initialize libCBM**

.. code-block:: python

   from libcbm import State, Simulator

   # Create libCBM state
   state = State()

   # Add pools (optional)
   # state.add_pool('aboveground', 'vegetation')

**Step 2: Define Callback Function**

.. code-block:: python

   def libcbm_callback(period, development_type, action, area_ha):
       """Callback function for libCBM integration."""

       # Get current carbon stock
       carbon_stock = state.get_carbon(development_type)

       # Apply harvest action
       if action == 'CLEARCUT':
           # Remove all carbon
           state.remove_carbon(development_type, area_ha)

           # Add regeneration carbon
           state.add_carbon(development_type, 0, area_ha)

       elif action == 'COMM_THIN':
           # Remove partial carbon
           carbon_removed = carbon_stock * 0.3  # 30% removal
           state.remove_carbon(development_type, area_ha, carbon_removed)

       # Advance state
       state.advance(period)

       return carbon_stock

**Step 3: Register Callback with ws3**

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   # Add development types and actions
   # (see previous how-to guides)

   # Register libCBM callback
   model.register_callback('carbon', libcbm_callback)

**Step 4: Run Simulation with Carbon Tracking**

.. code-block:: python

   from ws3.core import simulate

   # Run simulation
   results = simulate(
       model=model,
       horizon=5,
       schedule=schedule,
       callbacks=['carbon']
   )

**Step 5: Extract Carbon Results**

.. code-block:: python

   # Get carbon time series
   carbon_series = results.get_callback_results('carbon')

   # Plot carbon stocks
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(carbon_series['period'], carbon_series['carbon_stock'])
   ax.set_xlabel('Period')
   ax.set_ylabel('Carbon Stock (tC/ha)')
   ax.set_title('Carbon Dynamics')
   plt.tight_layout()
   plt.show()

Expected Output
---------------

* Carbon stock time series
* Carbon flux from harvest actions
* Integration between ws3 and libCBM

Troubleshooting
---------------

**Issue: Callback not called**

* Check that callback is registered with correct name
* Verify callback signature matches expected format
* Ensure callbacks list includes 'carbon'

**Issue: libCBM errors**

* Check that libCBM is properly installed
* Verify pool definitions match your model
* Check state initialization

**Issue: Carbon values unrealistic**

* Check pool definitions and initial stocks
* Verify carbon removal fractions
* Ensure state advancement is correct

Next Steps
----------

* :doc:`running-optimization` — Run optimization
* :doc:`financial-scenarios` — Add financial analysis
* :doc:`custom-area-selector` — Custom area selection