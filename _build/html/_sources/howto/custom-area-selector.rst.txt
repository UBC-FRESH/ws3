.. _howto-custom-area-selector:

=================
Custom Area Selector
=================

Goal
----

Implement custom area selection logic for harvest scheduling:

* Priority-based selection
* Spatial constraints
* Custom eligibility rules
* Multi-criteria selection

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with area selection concepts
* A working ws3 installation

Step-by-Step Instructions
-------------------------

**Step 1: Define Selection Criteria**

.. code-block:: python

   def eligibility_criteria(development_type, period):
       """Define eligibility for harvest."""

       # Minimum age
       if development_type.age < 40:
           return False

       # Maximum area per period
       if development_type.area > 500:
           return False

       # Exclude protected areas
       if development_type.protected:
           return False

       return True

**Step 2: Define Priority Function**

.. code-block:: python

   def priority_score(development_type, period):
       """Calculate priority score for selection."""

       # Higher priority for older stands
       age_score = development_type.age / 100.0

       # Higher priority for higher volume
       volume_score = development_type.volume / 1000.0

       # Combine scores
       score = 0.6 * age_score + 0.4 * volume_score

       return score

**Step 3: Implement Custom Selector**

.. code-block:: python

   from ws3.opt import AreaSelector

   class CustomSelector(AreaSelector):
       def select(self, development_types, period, target_area):
           """Select development types for harvest."""

           # Filter eligible types
           eligible = [
               dt for dt in development_types
               if self.eligibility_criteria(dt, period)
           ]

           # Sort by priority
           eligible.sort(
               key=lambda dt: self.priority_score(dt, period),
               reverse=True
           )

           # Select areas
           selected = []
           remaining = target_area

           for dt in eligible:
               if remaining <= 0:
                   break

               alloc = min(dt.available_area, remaining)
               selected.append({
                   'development_type': dt,
                   'area': alloc
               })
               remaining -= alloc

           return selected

**Step 4: Register Custom Selector**

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   # Add development types and actions
   # (see previous how-to guides)

   # Register custom selector
   selector = CustomSelector()
   model.set_area_selector(selector)

**Step 5: Run Optimization with Custom Selector**

.. code-block:: python

   from ws3.opt import solve_optimization

   solution = solve_optimization(
       model=model,
       horizon=5,
       objective='maximize_volume',
       area_selector=selector
   )

Expected Output
---------------

* Custom area selection logic
* Priority-based harvest allocation
* Eligibility filtering

Troubleshooting
---------------

**Issue: No areas selected**

* Check eligibility criteria are not too restrictive
* Verify priority function returns valid scores
* Ensure target area is feasible

**Issue: Selection doesn't match expectations**

* Debug priority scores
* Check eligibility filtering
* Verify area allocation logic

**Issue: Performance issues**

* Optimize eligibility checks
* Reduce number of development types
* Use caching for expensive calculations

Next Steps
----------

* :doc:`running-optimization` — Run optimization
* :doc:`financial-scenarios` — Add financial analysis
* :doc:`libcbm-callbacks` — Integrate with libCBM for carbon