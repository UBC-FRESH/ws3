.. _howto-action-definition:

=================
Action Definition
=================

Goal
----

Define management actions (harvest, thinning, etc.) and their transitions:

* Clearcut harvest actions
* Partial cut actions (selection, commercial thinning)
* Regeneration transitions
* Custom management prescriptions

Prerequisites
-------------

* Completed :doc:`data-preparation` and :doc:`curve-definition`
* Familiarity with actions and transitions from :doc:`../textbook/ch04_actions_and_transitions`
* A working ws3 installation

Step-by-Step Instructions
-------------------------

**Step 1: Define Clearcut Action**

.. code-block:: python

   from ws3.forest import ForestModel

   model = ForestModel()

   model.add_action(
       code='CLEARCUT',
       descr='Clearcut harvest',
       components=['volume'],
       transitions={
           'DT001': 'DT001_REGEN',
           'DT002': 'DT002_REGEN'
       }
   )

**Step 2: Define Partial Cut Action**

.. code-block:: python

   model.add_action(
       code='COMM_THIN',
       descr='Commercial thinning',
       components=['volume'],
       transitions={
           'DT001': 'DT001_THINNED'
       }
   )

**Step 3: Define Regeneration Development Types**

.. code-block:: python

   model.add_development_type(
       code='DT001_REGEN',
       species='SP',
       site_index=50,
       age=0,
       area=0.0
   )

   model.add_development_type(
       code='DT001_THINNED',
       species='SP',
       site_index=50,
       age=25,
       area=0.0
   )

**Step 4: Define Yield Components for Actions**

Some actions may have different yield components:

.. code-block:: python

   model.add_action(
       code='SELECT_CUT',
       descr='Selection cutting',
       components=['volume', 'basal_area'],
       transitions={
           'DT001': 'DT001_SELECT'
       }
   )

**Step 5: Verify Action Configuration**

.. code-block:: python

   # List all actions
   for action in model.actions.values():
       print(f"{action.code}: {action.descr}")
       print(f"  Components: {action.components}")
       print(f"  Transitions: {action.transitions}")

Expected Output
---------------

* Action objects created and registered
* Development types for post-action states defined
* Ability to apply actions in simulation

Troubleshooting
---------------

**Issue: Action not found**

* Check that action code is spelled correctly
* Verify action was added before simulation

**Issue: Transition target doesn't exist**

* Ensure all target development types are defined
* Check that target types have valid attributes

**Issue: Wrong yield components**

* Match components to what the action actually removes
* Volume for harvest, BA for thinning, etc.

Next Steps
----------

* :doc:`data-preparation` — Prepare inventory data
* :doc:`curve-definition` — Define growth curves
* :doc:`running-optimization` — Run optimization scenarios