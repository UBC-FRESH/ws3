Chapter 7: Financial Analysis
=============================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Calculate net present value (NPV) for forest management scenarios
- Understand the components of forest financial analysis
- Use ws3's financial functions to evaluate management plans
- Compare alternative management strategies using financial metrics

What Is Financial Analysis in Forest Planning?
----------------------------------------------

Financial analysis evaluates the economic viability of forest management
plans. It answers questions like:

- Is this harvest schedule profitable?
- Which management strategy maximizes returns?
- How sensitive is the plan to changes in timber prices?
- What is the optimal rotation age?

Key financial metrics:

- **Net Present Value (NPV)**: Total value of all cash flows, discounted to present
- **Internal Rate of Return (IRR)**: Discount rate that makes NPV = 0
- **Benefit-Cost Ratio (BCR)**: Total benefits divided by total costs
- **Rotation Age**: Age that maximizes NPV per hectare

Net Present Value
-----------------

**Net Present Value (NPV)** is the most important financial metric in
forest planning. It accounts for the time value of money: a dollar today
is worth more than a dollar tomorrow.

.. math::

   NPV = \\sum_{t=0}^{T} \\frac{R_t - C_t}{(1 + r)^t}

Where:
- :math:`R_t` = Revenue in period :math:`t`
- :math:`C_t` = Cost in period :math:`t`
- :math:`r` = Discount rate
- :math:`T` = Planning horizon

.. mermaid::

   graph TD
     REV["Revenue<br/>(harvest sales)"] --> NPV["NPV Calculation"]
     COST["Costs<br/>(harvesting, silviculture)"] --> NPV
     DISC["Discount rate<br/>(time value of money)"] --> NPV
     NPV --> PROF["Profitability<br/>decision"]

Using ws3's Financial Functions
-------------------------------

ws3 provides financial analysis functions in the :py:mod:`ws3.financial`
module.

Calculating NPV
~~~~~~~~~~~~~~~

.. code-block:: python

   # Financial calculations are done in pure Python (no ws3.financial module).
   # Define cash flows for each period
   revenues = [0, 0, 50000, 80000, 100000, 120000, 110000, 90000, 70000, 50000]
   costs = [10000, 5000, 20000, 25000, 30000, 35000, 30000, 25000, 20000, 15000]

   # Calculate NPV at 5% discount rate
   discount_rate = 0.05
   npv = sum(
       (r - c) / (1 + discount_rate) ** t
       for t, (r, c) in enumerate(zip(revenues, costs))
   )
   print(f"NPV: ${npv:,.0f}")

   # Calculate NPV at different discount rates
   for rate in [0.02, 0.05, 0.08, 0.10]:
       npv = sum(
           (r - c) / (1 + rate) ** t
           for t, (r, c) in enumerate(zip(revenues, costs))
       )
       print(f"NPV at {rate*100:.0f}%: ${npv:,.0f}")

Calculating IRR
~~~~~~~~~~~~~~~

The **Internal Rate of Return (IRR)** is the discount rate that makes
NPV equal to zero. It represents the inherent rate of return of the
investment.

.. code-block:: python

   # Calculate IRR using numpy (or scipy.optimize)
   import numpy as np
   from scipy.optimize import brentq

   # Net cash flows
   net_flows = [r - c for r, c in zip(revenues, costs)]

   # IRR is the discount rate that makes NPV = 0
   def npv_at_rate(rate):
       return sum(cf / (1 + rate) ** t for t, cf in enumerate(net_flows))

   irr = brentq(npv_at_rate, -0.99, 0.99)
   print(f"IRR: {irr*100:.1f}%")

   # Compare to discount rate
   if irr > 0.05:
       print("Project is profitable at 5% discount rate")
   else:
       print("Project is not profitable at 5% discount rate")

Rotation Economics
------------------

The **optimal rotation age** is the age that maximizes NPV per hectare.
This is a fundamental concept in forest economics.

.. mermaid::

   graph TD
     AGE["Rotation age"] --> VOL["Volume at harvest"]
     AGE --> COST["Costs over rotation"]
     VOL --> REV["Revenue at harvest"]
     COST --> NPV["NPV"]
     REV --> NPV
     NPV --> OPT["Optimal rotation age<br/>(max NPV)"]

The Faustmann formula calculates the optimal rotation age:

.. math::

   V'(T) / V(T) = r / (1 - e^{-rT})

Where:
- :math:`V(T)` = Volume at age :math:`T`
- :math:`V'(T)` = Marginal growth at age :math:`T`
- :math:`r` = Discount rate

Using ws3 to Find Optimal Rotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ws3.forest import ForestModel
   from ws3.core import Curve

   # Define a volume curve
   volume_curve = Curve(
       label="DF_volume",
       points=[(age, vol) for age, vol in
               zip(range(0, 201, 10),
                   [0, 2, 10, 30, 70, 130, 210, 300, 390, 460, 510,
                    540, 560, 575, 585, 590, 595, 598, 600, 601, 602, 602])]
   )

   # Calculate NPV for each rotation age
   prices = 50  # $/m³
   costs = 10000  # Fixed costs per hectare
   discount_rate = 0.05

   npv_by_age = []
   for age in range(10, 201, 10):
       volume = volume_curve(age)
       revenue = volume * prices
       # Discounted revenue minus costs
       npv = (revenue - costs) / (1 + discount_rate) ** age
       npv_by_age.append((age, npv))

   # Find optimal rotation age
   optimal_age, max_npv = max(npv_by_age, key=lambda x: x[1])
   print(f"Optimal rotation age: {optimal_age} years")
   print(f"Maximum NPV: ${max_npv:,.0f}")

Sensitivity Analysis
--------------------

Financial analysis should include sensitivity analysis to understand
how results change with different assumptions:

.. code-block:: python

   # Sensitivity to timber prices
   print("Sensitivity to timber prices:")
   base_price = 50
   for price in [30, 40, 50, 60, 70]:
       # Adjust revenues for new price
       adjusted_revenues = [r * price / base_price for r in revenues]
       npv_adj = sum(
           (r - c) / (1 + 0.05) ** t
           for t, (r, c) in enumerate(zip(adjusted_revenues, costs))
       )
       print(f"  Price = ${price}/m³: NPV = ${npv_adj:,.0f}")

   # Sensitivity to discount rates
   print("\nSensitivity to discount rates:")
   for rate in [0.02, 0.05, 0.08, 0.10, 0.15]:
       npv = sum(
           (r - c) / (1 + rate) ** t
           for t, (r, c) in enumerate(zip(revenues, costs))
       )
       print(f"  Rate = {rate*100:.0f}%: NPV = ${npv:,.0f}")

Common Financial Mistakes
-------------------------

1. **Ignoring discounting**: Failing to account for the time value of money
2. **Using nominal vs. real values**: Mixing nominal and real prices
3. **Ignoring costs**: Only considering revenue, not harvesting/silviculture costs
4. **Overlooking risk**: Not accounting for uncertainty in prices and volumes
5. **Incorrect rotation age**: Using biological maturity instead of economic optimum

Exercises
---------

**Exercise 1 (Easy)**: Calculate the NPV of a simple harvest scenario
with revenues of $100,000 in year 20 and costs of $10,000 in year 0.

**Exercise 2 (Medium)**: Find the optimal rotation age for a Douglas-fir
stand with the volume curve defined in this chapter.

**Exercise 3 (Hard)**: Perform a sensitivity analysis on the optimal
rotation age with respect to discount rate and timber price.

Further Reading
---------------

- :doc:`ch05_optimization` — Optimization fundamentals
- :doc:`/howto/faq` — Frequently asked questions
- :doc:`/reference/contracts/index` — Data contracts and module boundaries