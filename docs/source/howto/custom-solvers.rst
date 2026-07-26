.. _howto-custom-solvers:

=====================
Custom Solver Integration
=====================

Goal
----

Integrate custom solvers or modify existing solver behavior for ws3:

* Add support for new optimization solvers
* Customize solver parameters and behavior
* Implement custom branching strategies
* Integrate heuristic solvers

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Programming experience in Python
* Understanding of optimization solver interfaces
* Familiarity with PuLP, Gurobi, or similar solver APIs

Available Solvers
-----------------

ws3 currently supports:

* **Gurobi**: Commercial, high-performance MIP solver
* **PuLP/CBC**: Open-source, good for smaller problems
* **HiGHS**: Open-source, fast for linear programs
* **GLPK**: Open-source, included with some distributions

Custom Solver Requirements
--------------------------

To integrate a custom solver, you need:

1. **Solver Interface**: Class that wraps the solver API
2. **Problem Translation**: Convert ws3 Problem to solver format
3. **Solution Extraction**: Convert solver solution back to ws3 format
4. **Parameter Mapping**: Map ws3 parameters to solver parameters

Step-by-Step Instructions
-------------------------

**Step 1: Create Solver Interface Class**

.. code-block:: python

   from ws3.opt import Problem, SOLVER_DEFAULT
   
   class CustomSolver:
       """Interface for a custom optimization solver."""
       
       def __init__(self, problem: Problem):
           self.problem = problem
           self.solver_instance = None
           
       def build_problem(self):
           """Translate ws3 problem to solver format."""
           # Create solver-specific problem structure
           self.solver_instance = self.create_solver_problem()
           
       def create_solver_problem(self):
           """Create solver-specific problem (override in subclass)."""
           raise NotImplementedError
           
       def solve(self, **kwargs):
           """Solve the problem using custom solver."""
           self.build_problem()
           solution = self.solver_instance.solve(**kwargs)
           self.extract_solution(solution)
           
       def extract_solution(self, solver_solution):
           """Extract ws3-compatible solution from solver result."""
           for var_name, var_obj in self.problem._vars.items():
               if var_name in solver_solution:
                   var_obj.val = solver_solution[var_name]
               else:
                   var_obj.val = 0.0
                   
       def get_status(self):
           """Get solution status."""
           return self.problem.status()

**Step 2: Implement Specific Solver**

.. code-block:: python

   class MyCustomSolver(CustomSolver):
       """Example custom solver implementation."""
       
       def create_solver_problem(self):
           """Create problem for custom solver."""
           # Initialize your solver
           solver = YourSolverLibrary()
           
           # Add variables
           for var_name, var_obj in self.problem._vars.items():
               solver.add_variable(
                   name=var_name,
                   lb=var_obj.lb,
                   ub=var_obj.ub,
                   vtype=var_obj.vtype
               )
           
           # Add constraints
           for con_name, con_obj in self.problem._constraints.items():
               solver.add_constraint(
                   name=con_name,
                   coefficients=con_obj.coeffs,
                   sense=con_obj.sense,
                   rhs=con_obj.rhs
               )
           
           # Add objective
           solver.set_objective(self.problem._objective)
           
           return solver
           
       def solve(self, **kwargs):
           """Solve with custom solver."""
           self.build_problem()
           
           # Apply custom parameters
           if 'time_limit' in kwargs:
               self.solver_instance.set_param('TimeLimit', kwargs['time_limit'])
           
           # Solve
           self.solver_instance.solve()
           
           # Extract solution
           solution = {}
           for var_name in self.problem._vars:
               solution[var_name] = self.solver_instance.get_var_value(var_name)
           
           self.extract_solution(solution)

**Step 3: Register Custom Solver**

.. code-block:: python

   from ws3.opt import register_solver
   
   # Register your custom solver
   register_solver('custom', MyCustomSolver)
   
   # Now you can use it
   problem = compile_scenario(fm, scenario_name="test")
   problem.solve(solver="custom", time_limit=3600)

**Step 4: Use Custom Solver Parameters**

.. code-block:: python

   # Pass solver-specific parameters
   problem.solve(
       solver="custom",
       time_limit=1800,           # 30 minutes
       mip_gap=0.01,             # 1% optimality gap
       threads=4,                # Use 4 cores
       custom_param1="value1",   # Solver-specific parameter
       custom_param2=42,         # Another solver-specific parameter
   )

Expected Output
---------------

* Custom solver integrates seamlessly with ws3
* Can be used like any other solver: `problem.solve(solver="custom")`
* Supports custom parameters specific to your solver
* Extracts solutions in ws3 format

Troubleshooting
---------------

**Issue: Solver crashes during problem construction**

* Solution: Check variable and constraint names match between ws3 and solver
* Solution: Verify variable types are compatible
* Solution: Check constraint senses and RHS values

**Issue: Solution extraction fails**

* Solution: Ensure all variables have values in solver solution
* Solution: Check variable naming conventions
* Solution: Verify solution format matches expectations

**Issue: Custom solver is slower than built-in solvers**

* Solution: Profile solver performance
* Solution: Compare solver parameters
* Solution: Consider hybrid approach (custom solver for warm start, then Gurobi)

Best Practices
--------------

1. **Test Thoroughly**: Compare custom solver results with Gurobi on test problems
2. **Document Parameters**: Document all solver-specific parameters
3. **Error Handling**: Handle solver errors gracefully
4. **Performance Monitoring**: Track solve times and solution quality
5. **Fallback Strategy**: Have fallback to built-in solvers if custom fails
6. **Version Control**: Track solver library versions

Related Resources
-----------------

* :doc:`parallel-optimization`
* :doc:`../textbook/ch05_optimization`
* PuLP documentation: https://python-pulp.readthedocs.io/
* Gurobi documentation: https://www.gurobi.com/documentation/