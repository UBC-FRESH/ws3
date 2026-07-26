.. _contract-runtime-invariants:

=========================
Runtime Invariants
=========================

This page documents invariants that must hold during ws3 execution.

Model State Invariants
----------------------

1. **Development type codes are unique**

   Each development type must have a unique code within the model. Duplicate
   codes will cause undefined behavior.

2. **Growth curves match development types**

   Every development type must have a corresponding growth curve. If a DT
   has species="SP" and site_index=50, there must be a curve for that
   combination.

3. **Action transitions reference valid DTs**

   All target development types in action transitions must exist in the model.

4. **Area is non-negative**

   All area values must be >= 0. Negative areas indicate a bug.

5. **Ages are non-negative integers**

   Stand ages must be non-negative integers.

6. **Volume curves are monotonically non-decreasing**

   Volume curves should not decrease with age (except for thinning effects).

Optimization Invariants
-----------------------

1. **Flow constraints are feasible**

   Flow constraints must allow at least one feasible solution. If min_ratio >
   max_ratio, the problem is infeasible.

2. **Area constraints don't exceed available area**

   Sum of max_area constraints across all periods must not exceed total
   available area.

3. **Harvest doesn't exceed growth**

   Harvest volume in any period should not exceed available volume from
   standing inventory.

Simulation Invariants
---------------------

1. **Total area is conserved**

   Total area across all development types should remain constant (unless
   area is added or removed explicitly).

2. **Transitions are deterministic**

   Given the same input state and actions, simulation should produce the
   same output.

3. **Callback state is consistent**

   If callbacks modify state (e.g., libCBM), the state must be consistent
   with the development type transitions.

Error Conditions
----------------

The following conditions indicate bugs or misconfiguration:

- Development type with zero area in schedule
- Action with no transitions defined
- Growth curve with fewer than 2 data points
- Negative volume or area values in output
- Infeasible optimization (no solution found)

Validation
----------

Use :doc:`../howto/model-validation` to check these invariants after running
your model.