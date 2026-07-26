.. _howto-financial-scenarios:

=================
Financial Scenarios
=================

Goal
----

Add financial analysis to your ws3 optimization:

* Define cost and revenue parameters
* Calculate net present value (NPV)
* Run financial optimization scenarios
* Compare financial outcomes

Prerequisites
-------------

* Completed :doc:`running-optimization`
* Familiarity with financial concepts from :doc:`../textbook/ch07_financial_analysis`
* A working ws3 installation with sample data

Step-by-Step Instructions
-------------------------

**Step 1: Define Financial Parameters**

.. code-block:: python

   # Harvest costs
   harvest_cost = {
       'CLEARCUT': 45.0,  # $/m3
       'COMM_THIN': 35.0,  # $/m3
   }

   # Transportation costs
   transport_cost = 0.15  # $/m3/km

   # Product prices
   product_price = {
       'sawlog': 120.0,  # $/m3
       'pulpwood': 35.0,  # $/m3
   }

   # Discount rate
   discount_rate = 0.05  # 5%

**Step 2: Define Revenue Function**

.. code-block:: python

   def calculate_revenue(volume, product_mix):
       """Calculate revenue from harvested volume."""

       revenue = 0.0
       for product, proportion in product_mix.items():
           revenue += volume * proportion * product_price[product]

       return revenue

**Step 3: Define Cost Function**

.. code-block:: python

   def calculate_cost(volume, action, distance_km=10.0):
       """Calculate harvest and transport costs."""

       # Harvest cost
       cost = volume * harvest_cost[action]

       # Transport cost
       cost += volume * transport_cost * distance_km

       return cost

**Step 4: Calculate NPV**

.. code-block:: python

   def calculate_npv(cash_flows, discount_rate):
       """Calculate net present value."""

       npv = 0.0
       for period, cash_flow in enumerate(cash_flows):
           npv += cash_flow / ((1 + discount_rate) ** period)

       return npv

**Step 5: Run Financial Optimization**

.. code-block:: python

   from ws3.opt import solve_optimization

   # Define financial objective
   def financial_objective(schedule):
       """Calculate NPV of harvest schedule."""

       cash_flows = []
       for period, row in schedule.iterrows():
           revenue = calculate_revenue(row['volume'], row['product_mix'])
           cost = calculate_cost(row['volume'], row['action'])
           cash_flows.append(revenue - cost)

       return calculate_npv(cash_flows, discount_rate)

   # Run optimization
   solution = solve_optimization(
       model=model,
       horizon=5,
       objective='maximize_npv',
       financial_objective=financial_objective
   )

**Step 6: Analyze Financial Results**

.. code-block:: python

   # Get financial summary
   summary = solution.get_financial_summary()

   print(f"Total Revenue: ${summary['total_revenue']:,.2f}")
   print(f"Total Cost: ${summary['total_cost']:,.2f}")
   print(f"Net Present Value: ${summary['npv']:,.2f}")

   # Plot cash flows
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.bar(summary['period'], summary['revenue'], label='Revenue')
   ax.bar(summary['period'], -summary['cost'], bottom=-summary['cost'], label='Cost')
   ax.set_xlabel('Period')
   ax.set_ylabel('Revenue/Cost ($)')
   ax.set_title('Financial Summary')
   ax.legend()
   plt.tight_layout()
   plt.show()

Expected Output
---------------

* Financial summary with revenue, cost, and NPV
* Cash flow time series
* Financial optimization results

Troubleshooting
---------------

**Issue: Negative NPV**

* Check cost and price assumptions
* Verify discount rate
* Ensure revenue calculations are correct

**Issue: Solver doesn't converge**

* Check that financial objective is well-defined
* Verify cash flow calculations
* Try simpler objective first

**Issue: Unrealistic financial values**

* Compare with industry benchmarks
* Check units ($/m3 vs $/ha)
* Verify cost components are complete

Next Steps
----------

* :doc:`running-optimization` — Run optimization
* :doc:`libcbm-callbacks` — Integrate with libCBM for carbon
* :doc:`custom-area-selector` — Custom area selection