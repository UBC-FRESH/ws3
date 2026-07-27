.. _howto-faq:

=============================
Frequently Asked Questions
=============================

Q: How do I load a Woodstock model?
------------------------------------

.. code-block:: python

   from ws3.forest import ForestModel

   fm = ForestModel(
       model_name="my_model",
       model_path="path/to/model",
       base_year=2020,
       horizon=10,
       period_length=10
   )
   fm.import_areas_section()
   fm.import_yields_section()
   fm.import_actions_section()
   fm.import_transitions_section()
   fm.initialize_areas()
   fm.add_null_action()
   fm.reset_actions()

Q: How do I define a growth curve?
-----------------------------------

.. code-block:: python

   from ws3.core import Curve

   curve = Curve(
       label="SP_SI50_Volume",
       is_volume=True,
       points=[(10, 15.2), (20, 45.8), (30, 95.3)],
       period_length=10
   )
   fm.register_curve(curve)

Q: How do I run optimization?
------------------------------

.. code-block:: python

   from ws3.opt import Problem

   problem = Problem(name="base", sense=1, solver="highs")
   # Add variables, constraints, objective
   problem.solve()
   solution = problem.solution()

Q: How do I allocate harvest spatially?
----------------------------------------

.. code-block:: python

   from ws3.spatial import ForestRaster

   raster = ForestRaster(
       hdt_map=hdt_map,
       hdt_func=hdt_func,
       src_path="landscape.tif",
       snk_path="output",
       acode_map={"harvest": "harvest"},
       forestmodel=fm,
       base_year=2020,
       horizon=10,
       period_length=10
   )
   raster.allocate_schedule(problem.solution())

Q: How do I run multi-objective optimization?
----------------------------------------------

.. code-block:: python

   from ws3.advanced_modeling import MultiObjectiveOptimizer

   optimizer = MultiObjectiveOptimizer(fm)
   objectives = [
       {"name": "npv", "weight": 0.5},
       {"name": "carbon", "weight": 0.5}
   ]
   pareto_front = optimizer.optimize(objectives)

Q: How do I run parallel optimization?
---------------------------------------

.. code-block:: python

   from ws3.forest_helper import PersistentWorkerPool

   pool = PersistentWorkerPool(n_workers=4)
   results = pool.map(lambda scenario: run_scenario(fm, scenario), scenarios)

Q: What solvers are supported?
-------------------------------

ws3 supports:
* HiGHS (via PuLP)
* Gurobi
* CBC (via PuLP)

Set solver in Problem constructor:

.. code-block:: python

   problem = Problem(name="base", solver="highs")

Q: How do I check if a method exists?
--------------------------------------

Use Python's built-in `dir()` function:

.. code-block:: python

   print(dir(fm))  # List ForestModel methods
   print(dir(problem))  # List Problem methods