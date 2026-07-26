.. _howto-advanced-optimization:

=========================
Advanced Optimization Techniques
=========================

Goal
----

Implement advanced optimization techniques for complex forest management problems:

* Multi-objective optimization with trade-off analysis
* Stochastic optimization for uncertainty
* Dynamic planning with re-optimization
* Constraint programming for complex rules

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with optimization concepts from :doc:`../textbook/ch05_optimization`
* Understanding of multi-objective optimization principles

Common Optimization Challenges
------------------------------

Forest management optimization often involves:

* **Conflicting objectives**: Maximize revenue vs. conserve carbon
* **Uncertainty**: Future prices, growth rates, disturbances
* **Complex constraints**: Spatial adjacency, contiguous areas, habitat requirements
* **Dynamic decisions**: Re-optimization as conditions change

Step-by-Step Instructions
-------------------------

**Step 1: Define Multiple Objectives**

.. code-block:: python

   from ws3.core import compile_scenario

   # Define multiple objectives with weights
   objectives = {
       'npv': 0.5,           # Net present value
       'even_flow': 0.3,     # Even flow constraint
       'carbon': 0.2,        # Carbon sequestration
   }

   # Compile with multiple objectives
   problem = compile_scenario(
       fm,
       scenario_name="multi_obj",
       objectives=objectives
   )

**Step 2: Solve with Different Weight Combinations**

.. code-block:: python

   import pandas as pd

   # Try different weight combinations
   weight_sets = [
       {'npv': 0.8, 'even_flow': 0.1, 'carbon': 0.1},
       {'npv': 0.5, 'even_flow': 0.3, 'carbon': 0.2},
       {'npv': 0.2, 'even_flow': 0.3, 'carbon': 0.5},
   ]

   results = []
   for weights in weight_sets:
       problem = compile_scenario(fm, scenario_name="test", objectives=weights)
       solution = problem.solve(solver="gurobi")
       
       # Extract objective values
       results.append({
           'weights': weights,
           'npv_value': solution.get_objective_value('npv'),
           'even_flow_dev': solution.get_objective_value('even_flow'),
           'carbon_value': solution.get_objective_value('carbon'),
       })

   results_df = pd.DataFrame(results)
   print(results_df)

**Step 3: Analyze Trade-offs**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot trade-off curves
   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   axes[0].plot(results_df['npv_value'], results_df['carbon_value'], 'o-')
   axes[0].set_xlabel('NPV ($)')
   axes[0].set_ylabel('Carbon (tC)')
   axes[0].set_title('NPV vs Carbon Trade-off')

   axes[1].plot(results_df['npv_value'], results_df['even_flow_dev'], 's-')
   axes[1].set_xlabel('NPV ($)')
   axes[1].set_ylabel('Even Flow Deviation')
   axes[1].set_title('NPV vs Even Flow Trade-off')

   axes[2].plot(results_df['carbon_value'], results_df['even_flow_dev'], '^-')
   axes[2].set_xlabel('Carbon (tC)')
   axes[2].set_ylabel('Even Flow Deviation')
   axes[2].set_title('Carbon vs Even Flow Trade-off')

   plt.tight_layout()
   plt.show()

**Step 4: Identify Pareto-Optimal Solutions**

.. code-block:: python

   from ws3.opt import find_pareto_frontier

   # Find Pareto-optimal solutions
   pareto_solutions = find_pareto_frontier(results_df, 
                                          objectives=['npv_value', 'carbon_value'])
   
   print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
   
   # Plot Pareto frontier
   plt.scatter(results_df['npv_value'], results_df['carbon_value'], 
               alpha=0.5, label='All solutions')
   plt.scatter(pareto_solutions['npv_value'], pareto_solutions['carbon_value'],
               c='red', s=100, label='Pareto-optimal', zorder=5)
   plt.xlabel('NPV ($)')
   plt.ylabel('Carbon (tC)')
   plt.title('Pareto Frontier')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Expected Output
---------------

* Multiple objective values for each weight combination
* Trade-off curves showing relationships between objectives
* Identification of Pareto-optimal solutions
* Visual representation of the efficient frontier

Troubleshooting
---------------

**Issue: Solver takes too long with multiple objectives**

* Solution: Use smaller weight ranges or fewer weight combinations
* Solution: Increase MIP gap tolerance (e.g., 0.05 instead of 0.01)
* Solution: Use time limits to prevent excessive solving

**Issue: No Pareto-optimal solutions found**

* Solution: Check if objectives are truly conflicting
* Solution: Increase number of weight combinations tested
* Solution: Verify objective function formulations

**Issue: Solutions are infeasible**

* Solution: Relax constraints (reduce even_flow tolerance)
* Solution: Check data integrity and consistency
* Solution: Try different solver or solver parameters

Best Practices
--------------

1. **Start Simple**: Begin with single-objective optimization, then add complexity
2. **Weight Selection**: Use domain knowledge to select meaningful weight combinations
3. **Sensitivity Analysis**: Test how sensitive solutions are to weight changes
4. **Visualization**: Always visualize trade-offs to understand relationships
5. **Stakeholder Input**: Involve decision-makers in weight selection
6. **Documentation**: Document all objective functions and constraints clearly

Related Resources
-----------------

* :doc:`multi-objective-optimization` (notebook 074)
* :doc:`scenario-analysis` (notebook 071)
* :doc:`../textbook/ch08_multi_objective_optimization`
* :doc:`../textbook/ch09_stochastic_optimization`