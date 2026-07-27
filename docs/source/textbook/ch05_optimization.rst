Chapter 5: Optimization
=======================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Formulate a wood supply optimization problem using the :py:class:`ws3.opt.Problem`
  class
- Define decision variables, objective functions, and constraints
- Solve optimization problems with different solvers (HiGHS, Gurobi)
- Interpret optimization results and extract harvest schedules

What Is Optimization in Forest Planning?
----------------------------------------

Optimization helps forest managers find the **best** management plan
given a set of constraints. Instead of guessing which harvest schedule
is best, optimization systematically searches for the schedule that
maximizes (or minimizes) your objective.

Common objectives:

- **Maximize net present value (NPV)**: Maximize economic returns
- **Maximize sustained yield**: Maximize average annual harvest
- **Minimize cost**: Minimize harvesting and silviculture costs
- **Maximize carbon sequestration**: Maximize standing biomass

Common constraints:

- **Area constraints**: Maximum harvest area per period
- **Volume constraints**: Minimum or maximum volume harvested
- **Inventory constraints**: Minimum ending inventory
- **Spatial constraints**: Contiguity, adjacency requirements

The Optimization Problem
------------------------

A wood supply optimization problem has three components:

1. **Decision variables**: What we can control (e.g., harvest area)
2. **Objective function**: What we want to optimize (e.g., NPV)
3. **Constraints**: What limits our decisions (e.g., max harvest area)

.. mermaid::

   graph TD
     VAR["Decision Variables<br/>What we control"] --> OBJ["Objective Function<br/>What we optimize"]
     VAR --> CON["Constraints<br/>What limits us"]
     OBJ --> SOL["Solution<br/>Optimal values"]
     CON --> SOL

Setting Up an Optimization Problem
----------------------------------

.. code-block:: python

   from ws3.opt import Problem

   # Create an optimization problem
   prob = Problem("example_problem")

   # Add decision variables
   # x1 = harvest area for development type 1
   # x2 = harvest area for development type 2
   prob.add_var("harvest_DT1", vtype="continuous", lb=0, ub=500)
   prob.add_var("harvest_DT2", vtype="continuous", lb=0, ub=300)

   # Add objective: maximize NPV
   # The z() method sets objective coefficients as a dict keyed on variable names
   # NPV = 50 * x1 + 40 * x2 (price per m³ * volume per ha * area)
   prob.z(coeffs={"harvest_DT1": 50.0, "harvest_DT2": 40.0})

   # Add constraints
   # Constraint 1: Total harvest cannot exceed 200 ha per period
   # add_constraint(name, coeffs_dict, sense, rhs) where sense is 'leq', 'geq', or 'eq'
   prob.add_constraint("max_harvest", coeffs={"harvest_DT1": 1.0, "harvest_DT2": 1.0}, sense="leq", rhs=200)

   # Constraint 2: At least 100 ha of DT1 must remain
   prob.add_constraint("min_inventory", coeffs={"harvest_DT1": 1.0}, sense="leq", rhs=400)

Solving the Problem
-------------------

ws3 supports multiple solvers:

.. code-block:: python

   # Set solver and solve
   # The solver is set via prob.solver("highs") before calling solve()
   prob.solver("highs")  # or "gurobi" or "pulp"
   prob.solve()

Extracting Results
------------------

.. code-block:: python

   # Get the optimal solution
   solution = prob.solution()

   # Print decision variable values
   print(f"Optimal harvest for DT1: {solution['harvest_DT1']:.1f} ha")
   print(f"Optimal harvest for DT2: {solution['harvest_DT2']:.1f} ha")

   # Print the objective value
   print(f"Maximum NPV: ${prob.z():,.0f}")

Multi-Period Optimization
-------------------------

For realistic forest planning, you need to optimize over multiple periods:

.. code-block:: python

   # Create variables for each development type and period
   # Track variable names for building objective and constraints
   harvest_var_names = {}  # {(dt_code, period): var_name}
   for dt_code in ["DF-SI50", "SP-SI40"]:
       for period in range(20):
           var_name = f"harv_{dt_code}_p{period}"
           prob.add_var(var_name, vtype="continuous", lb=0, ub=100)  # Max 100 ha per period
           harvest_var_names[(dt_code, period)] = var_name

   # Objective: maximize NPV over all periods
   # z() takes a dict keyed on variable names with coefficient values
   npv_coeffs = {}
   discount_rate = 0.05
   for (dt_code, period), var_name in harvest_var_names.items():
       volume_per_ha = 200  # m³/ha (from growth curve)
       price = 50  # $/m³
       coeff = volume_per_ha * price * (1 + discount_rate) ** (-period * 5)
       npv_coeffs[var_name] = coeff
   prob.z(coeffs=npv_coeffs)

   # Constraint: Maximum harvest area per period
   for period in range(20):
       period_var_names = [
           harvest_var_names[(dt_code, period)]
           for dt_code in ["DF-SI50", "SP-SI40"]
       ]
       period_coeffs = {vn: 1.0 for vn in period_var_names}
       prob.add_constraint(
           f"max_harvest_p{period}",
           coeffs=period_coeffs,
           sense="leq",
           rhs=200
       )

   # Solve
   prob.solver("highs")
   prob.solve()

   # Extract solution
   solution = prob.solution()
   for (dt_code, period), var_name in harvest_var_names.items():
       area = solution[var_name]
       if area > 0:
           print(f"Period {period}: Harvest {area:.1f} ha of {dt_code}")

Solver Comparison
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Solver
     - Type
     - License
     - Best For
   * - HiGHS
     - Open-source
     - Free
     - Small to medium problems
   * - Gurobi
     - Commercial
     - Paid license
     - Large problems, MIP
   * - PuLP
     - Open-source
     - Free
     - Linear problems

HiGHS is the default and works well for most wood supply problems.
Use Gurobi if you need:

- Mixed-integer programming (binary decisions)
- Quadratic objectives
- Very large problems (>10,000 variables)

Common Optimization Patterns
----------------------------

1. **Sustained yield**: Maximize average harvest over all periods
2. **Even flow**: Minimize variance in harvest across periods
3. **Rotation optimization**: Find the optimal rotation age
4. **Multi-objective**: Balance economic and ecological objectives

Exercises
---------

**Exercise 1 (Easy)**: Set up and solve a simple optimization problem
to maximize NPV with two decision variables and two constraints.

**Exercise 2 (Medium)**: Extend the multi-period optimization to include
a constraint that total harvest over all periods does not exceed 2,000 ha.

**Exercise 3 (Hard)**: Formulate a rotation optimization problem to find
the optimal rotation age for Douglas-fir that maximizes NPV.

Further Reading
---------------

- :doc:`ch04_actions_and_transitions` — Defining actions
- :doc:`/howto/running-optimization` — Detailed optimization guide
- :doc:`/howto/parallel-optimization` — Parallel optimization for large models