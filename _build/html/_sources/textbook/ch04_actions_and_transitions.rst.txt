Chapter 4: Actions and Transitions
==================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Define management actions in ws3 using the :py:class:`ws3.forest.Action`
  class
- Specify transitions that define what happens after an action
- Understand the relationship between actions, transitions, and development
  types
- Build a complete set of actions for a realistic forest management scenario

What Are Actions?
-----------------

An **action** is a management intervention applied to a development type
in a specific age class. Actions represent the decisions a forest manager
makes: when to harvest, when to thin, when to plant.

In ws3, actions are defined by:

- A **code** (short identifier, e.g., "HARV", "THIN")
- A **description** (human-readable explanation)
- A set of **components** (what attributes change)
- A set of **transitions** (what development type results)

Types of Actions
----------------

Common forest management actions include:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Action Type
     - Code
     - Description
   * - Clearcut harvest
     - HARV
     - Remove all trees, leave bare site
   * - Selection harvest
     - SEL
     - Remove selected trees, reduce density
   * - Commercial thin
     - CT
     - Remove merchantable trees, reduce density
   * - Pre-commercial thin
     - PCT
     - Remove suppressed trees, improve spacing
   * - Planting
     - PLNT
     - Plant seedlings on bare site
   * - Prescribe
     - PRES
     - Do nothing, let nature take its course

Defining Actions in ws3
-----------------------

.. code-block:: python

   from ws3.forest import ForestModel

   # Actions and transitions are defined in Woodstock-format section files
   # and imported into the ForestModel. The typical workflow is:
   #
   #   model = ForestModel("my_model", "/path/to/data", 2024,
   #                       horizon=20, period_length=10)
   #   model.import_areas_section()       # loads development types and areas
   #   model.import_yields_section()      # loads growth curves
   #   model.import_actions_section()     # loads action definitions
   #   model.import_transitions_section() # loads transition rules
   #
   # Actions are defined in the ACTIONS section file (e.g., model.act):
   #   *action HARV Clearcut harvest - remove all trees
   #   *operable df si50 volume basal_area
   #   *operable sp si40 volume basal_area
   #
   # Transitions are defined in the TRANSITIONS section file (e.g., model.trn):
   #   *case HARV
   #   *source df si50
   #   *target bare
   #   *source sp si40
   #   *target bare

Understanding Transitions
-------------------------

A **transition** defines what happens to a development type after an
action is applied. It maps the "before" state to the "after" state.

For example, a clearcut harvest action on a Douglas-fir stand:

.. mermaid::

   graph LR
     BEFORE["DF-SI50<br/>500 ha, age 40"] --> ACTION["HARV<br/>Clearcut"]
     ACTION --> AFTER["Bare<br/>0 ha, age 0"]

The transition says: "After applying HARV to DF-SI50, the stand becomes
Bare."

Complex Transitions
-------------------

Some actions have more complex transitions. For example, a thinning action
might reduce the age class (because time passes during the treatment):

.. code-block:: python

   # Thinning actions are defined in the ACTIONS/TRANSITIONS section files
   # and imported via model.import_actions_section() and
   # model.import_transitions_section(). The transition file maps source
   # development types to target development types for each action code.

Actions and the Simulation Loop
-------------------------------

During simulation, the model processes actions period by period:

.. mermaid::

   graph TD
     START["Start of period"] --> CHECK["Check which actions<br/>to apply"]
     CHECK --> APPLY["Apply actions<br/>(change development types)"]
     APPLY --> GROW["Advance age<br/>(grow curves)"]
     GROW --> NEXT["Next period"]
     NEXT --> CHECK

The simulation loop:

1. **Check**: Which development types have actions scheduled?
2. **Apply**: Execute actions (change development types)
3. **Grow**: Advance all development types by one age class
4. **Repeat**: Go to step 1 for the next period

Action Components
-----------------

The ``components`` parameter specifies which attributes are affected by
the action:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Effect
   * - ``volume``
     - Volume is removed (for harvest) or reduced (for thinning)
   * - ``basal_area``
     - Basal area is removed or reduced
   * - ``height``
     - Dominant height may change (for planting)
   * - ``stocking``
     - Stems per hectare change

For harvest actions, the volume component is typically set to remove
all volume. For thinning actions, it removes a fraction of volume.

Defining a Complete Action Set
------------------------------

For a realistic forest management scenario, you need a complete set of
actions:

.. code-block:: python

   # Define all actions for a managed forest

   actions = [
       {
           "code": "HARV",
           "descr": "Clearcut harvest",
           "components": ["volume", "basal_area"],
           "transitions": {
               "DF-SI50": "Bare",
               "SP-SI40": "Bare",
               "CE-SI45": "Bare"
           }
       },
       {
           "code": "PLNT",
           "descr": "Plant after harvest",
           "components": ["volume"],
           "transitions": {
               "Bare": "DF-SI50",
               "Bare-SP": "SP-SI40",
               "Bare-CE": "CE-SI45"
           }
       },
       {
           "code": "PRES",
           "descr": "Prescribe (do nothing)",
           "components": [],
           "transitions": {}
       }
   ]

   # In practice, actions are defined in Woodstock-format section files
   # and imported via model.import_actions_section() and
   # model.import_transitions_section(). The dict-based approach above
   # is illustrative only; the real workflow uses file-based import.

Common Mistakes
---------------

1. **Missing transitions**: Every action must define transitions for all
   affected development types. Missing transitions cause errors.

2. **Inconsistent codes**: Development type codes in transitions must
   match exactly (case-sensitive).

3. **Forgetting the bare site**: After harvest, you need a "Bare"
   development type to receive the harvested area.

4. **Not defining planting**: If you harvest but don't plant, the bare
   site stays bare forever.

Exercises
---------

**Exercise 1 (Easy)**: Define a thinning action that reduces volume by
50% and transitions from "DF-SI50-A40" to "DF-SI50-A45".

**Exercise 2 (Medium)**: Create a complete action set for a forest with
Douglas-fir and Spruce, including harvest, planting, and prescribe actions.

**Exercise 3 (Hard)**: Modify the simulation loop to track which actions
were applied each period and output a log of all management decisions.

Further Reading
---------------

- :doc:`ch01_forest_estate_models` — Forest estate model fundamentals
- :doc:`ch05_optimization` — Using actions in optimization problems
- :doc:`/howto/action-definition` — Detailed action definition guide