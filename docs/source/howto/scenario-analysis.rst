.. _howto-scenario-analysis:

=========================
Scenario Analysis and Reporting
=========================

Goal
----

Perform comprehensive scenario analysis and generate professional reports:

* Compare multiple optimization scenarios
* Analyze sensitivity to parameters
* Generate publication-quality reports
* Create executive summaries for decision-makers

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with scenario analysis concepts
* Understanding of reporting requirements

Scenario Analysis Workflow
--------------------------

A typical scenario analysis involves:

1. **Define Scenarios**: Different objectives, constraints, or parameters
2. **Run Optimization**: Solve each scenario
3. **Compare Results**: Analyze differences and trade-offs
4. **Sensitivity Analysis**: Test parameter sensitivity
5. **Generate Reports**: Create comprehensive documentation

Step-by-Step Instructions
-------------------------

**Step 1: Define Multiple Scenarios**

.. code-block:: python

   from ws3.core import compile_scenario
   import pandas as pd
   
   # Define scenarios
   scenarios = {
       'base_case': {
           'objective': 'maximize_npv',
           'weights': {'npv': 1.0},
           'description': 'Base case: maximize net present value'
       },
       'conservation': {
           'objective': 'maximize_carbon',
           'weights': {'carbon': 1.0},
           'description': 'Conservation focus: maximize carbon sequestration'
       },
       'balanced': {
           'objective': 'maximize_npv',
           'weights': {'npv': 0.5, 'carbon': 0.3, 'even_flow': 0.2},
           'description': 'Balanced: mix of economic and environmental goals'
       },
       'timber_focus': {
           'objective': 'maximize_volume',
           'weights': {'volume': 1.0},
           'description': 'Timber production focus: maximize volume harvested'
       }
   }

**Step 2: Run All Scenarios**

.. code-block:: python

   results = {}
   
   for name, params in scenarios.items():
       print(f"Running scenario: {name}")
       
       # Compile and solve
       problem = compile_scenario(fm, scenario_name=name, **params)
       solution = problem.solve(solver="gurobi")
       
       # Extract results
       results[name] = {
           'objective_value': solution.get_objective_value(),
           'solve_time': solution.solve_time,
           'status': solution.status(),
           'schedule': solution.get_schedule(),
           'params': params,
       }
       
       print(f"  Status: {solution.status()}")
       print(f"  Objective: {solution.get_objective_value():.2f}")
       print(f"  Solve time: {solution.solve_time:.2f}s")
       print()

**Step 3: Compare Scenarios**

.. code-block:: python

   # Create comparison table
   comparison = pd.DataFrame({
       name: {
           'Objective Value': res['objective_value'],
           'Solve Time (s)': res['solve_time'],
           'Status': res['status'],
       }
       for name, res in results.items()
   })
   
   print("Scenario Comparison:")
   print(comparison.T)
   
   # Visualize comparison
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # Objective values
   axes[0].bar(comparison.index, comparison['Objective Value'])
   axes[0].set_ylabel('Objective Value')
   axes[0].set_title('Objective Values by Scenario')
   axes[0].tick_params(axis='x', rotation=45)
   
   # Solve times
   axes[1].bar(comparison.index, comparison['Solve Time (s)'])
   axes[1].set_ylabel('Solve Time (seconds)')
   axes[1].set_title('Solve Times by Scenario')
   axes[1].tick_params(axis='x', rotation=45)
   
   plt.tight_layout()
   plt.show()

**Step 4: Analyze Schedule Differences**

.. code-block:: python

   # Compare harvest schedules
   fig, ax = plt.subplots(1, 1, figsize=(12, 6))
   
   for name, res in results.items():
       schedule = res['schedule']
       if 'period' in schedule.columns and 'area_ha' in schedule.columns:
           ax.plot(schedule['period'], schedule['area_ha'], 
                  label=name, marker='o', linewidth=2)
   
   ax.set_xlabel('Period')
   ax.set_ylabel('Harvest Area (ha)')
   ax.set_title('Harvest Schedules by Scenario')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

**Step 5: Sensitivity Analysis**

.. code-block:: python

   # Test sensitivity to discount rate
   discount_rates = [0.02, 0.04, 0.06, 0.08, 0.10]
   npv_values = []
   
   for rate in discount_rates:
       problem = compile_scenario(
           fm,
           scenario_name=f"discount_{rate}",
           objective='maximize_npv',
           discount_rate=rate
       )
       solution = problem.solve(solver="gurobi")
       npv_values.append(solution.get_objective_value())
   
   # Plot sensitivity
   plt.figure(figsize=(10, 6))
   plt.plot(discount_rates, npv_values, 'o-', linewidth=2, markersize=8)
   plt.xlabel('Discount Rate')
   plt.ylabel('Net Present Value ($)')
   plt.title('Sensitivity to Discount Rate')
   plt.grid(True, alpha=0.3)
   plt.axvline(x=0.05, color='r', linestyle='--', label='Default (5%)')
   plt.legend()
   plt.tight_layout()
   plt.show()

**Step 6: Generate Report**

.. code-block:: python

   def generate_scenario_report(scenarios, results, output_file):
       """Generate comprehensive scenario analysis report."""
       
       report = []
       report.append("# Scenario Analysis Report\n")
       report.append(f"Generated: {pd.Timestamp.now()}\n")
       report.append(f"Number of scenarios: {len(scenarios)}\n")
       
       # Executive summary
       report.append("## Executive Summary\n")
       best_scenario = max(results.items(), key=lambda x: x[1]['objective_value'])
       report.append(f"Best scenario: **{best_scenario[0]}** "
                    f"with objective value {best_scenario[1]['objective_value']:.2f}\n")
       
       # Detailed results
       report.append("## Detailed Results\n")
       for name, res in results.items():
           report.append(f"### {name}\n")
           report.append(f"- Objective: {res['params'].get('objective', 'N/A')}")
           report.append(f"- Value: {res['objective_value']:.2f}")
           report.append(f"- Solve time: {res['solve_time']:.2f}s")
           report.append(f"- Status: {res['status']}\n")
       
       # Save report
       with open(output_file, 'w') as f:
           f.write('\n'.join(report))
       
       print(f"Report saved to {output_file}")
   
   # Generate report
   generate_scenario_report(scenarios, results, 'scenario_report.md')

Expected Output
---------------

* Comparison table of all scenarios
* Visualizations of objective values and solve times
* Harvest schedule comparisons
* Sensitivity analysis plots
* Comprehensive markdown report

Troubleshooting
---------------

**Issue: Some scenarios fail to solve**

* Solution: Check constraint feasibility
* Solution: Relax constraints or adjust parameters
* Solution: Try different solver or solver settings

**Issue: Comparison tables are empty**

* Solution: Verify results dictionary is populated correctly
* Solution: Check that schedules are extracted properly
* Solution: Ensure all scenarios completed successfully

**Issue: Reports are too large**

* Solution: Summarize key findings instead of full details
* Solution: Use executive summary format
* Solution: Create separate detailed appendices

Best Practices
--------------

1. **Consistent Naming**: Use clear, descriptive scenario names
2. **Documentation**: Document all scenario parameters and assumptions
3. **Visualization**: Always include visual comparisons
4. **Sensitivity**: Test key parameters for sensitivity
5. **Reproducibility**: Save all inputs and outputs for reproducibility
6. **Stakeholder Input**: Involve decision-makers in scenario definition

Related Resources
-----------------

* :doc:`multi-objective-optimization` (notebook 074)
* :doc:`scenario-analysis` (notebook 071)
* :doc:`../textbook/ch10_scenario_analysis`
* Markdown documentation: https://daringfireball.net/projects/markdown/