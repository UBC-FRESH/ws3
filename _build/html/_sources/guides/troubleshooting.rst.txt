.. _guide-troubleshooting:

=================
Troubleshooting
=================

This guide covers common issues you may encounter when using ws3, along with
diagnostic steps and solutions.

Installation Issues
-------------------

**Issue: Cannot install ws3 with pip**

*Symptom*: :code:`pip install ws3` fails with dependency errors

*Solution*:
1. Use a clean virtual environment: :code:`python -m venv .venv && source .venv/bin/activate`
2. Install in development mode: :code:`pip install -e .`
3. Check Python version: ws3 requires Python 3.9+

**Issue: Missing optional dependencies**

*Symptom*: Import errors for scipy, pandas, or other optional packages

*Solution*:
.. code-block:: bash

   # Install all optional dependencies
   pip install "ws3[dev,docs]"

   # Or install specific extras
   pip install "ws3[gurobi]"  # For Gurobi solver

Model Configuration Errors
--------------------------

**Issue: Development type codes conflict**

*Symptom*: Errors when importing development types

*Solution*:
- Ensure all development type keys are unique within the model
- Development types are created automatically when importing the AREAS section
- Check for typos in the landscape section file

**Issue: Growth curve not found**

*Symptom*: :code:`KeyError` when querying volume at certain ages

*Solution*:
- Verify growth curve exists for the species/site_index combination
- Check that curve ages cover the planning horizon
- Ensure curve ages are in ascending order

**Issue: Action transition target doesn't exist**

*Symptom*: :code:`KeyError` when applying action

*Solution*:
- Check that all target development types in transitions are defined
- Verify transition codes match existing development type codes
- List all DTs: :code:`print(list(fm.dtypes.keys()))`

Optimization Errors
-------------------

**Issue: Solver fails to converge**

*Symptom*: Optimization returns no solution or "infeasible" status

*Diagnostic steps*:
1. Check constraint feasibility:
   - Flow constraints: ensure min_ratio <= max_ratio
   - Area constraints: ensure sum of max_area doesn't exceed total area
2. Simplify the problem:
   - Reduce planning horizon
   - Remove some constraints
   - Use simpler objective function
3. Check model size:
   - Too many development types can cause solver memory issues
   - Reduce to essential DTs for debugging

*Solution*:
.. code-block:: python

   # Check if problem was solved
   if problem.solved():
       solution = problem.solution()
       print(f"Objective value: {problem.z()}")
   else:
       print("Problem was not solved successfully")

**Issue: No harvest in schedule**

*Symptom*: Schedule is empty or all areas are zero

*Diagnostic steps*:
1. Verify actions are applicable to development types
2. Check area constraints allow harvest
3. Ensure growth curves are defined for all DTs
4. Check that minimum age constraints aren't too restrictive

*Solution*:
.. code-block:: python

   # List all development types and their areas
   for key, dt in fm.dtypes.items():
       area = sum(dt.areas[p] for p in range(fm.horizon+1))
       print(f"{key}: area={area:.1f} ha")

**Issue: Solver takes too long**

*Symptom*: Optimization runs for hours without completing

*Diagnostic steps*:
1. Check model size (number of development types × periods)
2. Verify constraint complexity
3. Consider parallel solving strategies

*Solution*:
- Reduce planning horizon for testing
- Use area control mode instead of full optimization
- Simplify constraints
- Check that you're using an efficient solver

Simulation Errors
-----------------

**Issue: Callback not called**

*Symptom*: Carbon or other callback results are empty

*Diagnostic steps*:
1. Verify callback is registered with correct name
2. Check callback signature matches expected format
3. Ensure callbacks list includes the callback name

*Solution*:
Callbacks are registered via the ForestModel's callback system. Check the
:doc:`textbook/ch10_carbon_modelling` chapter for integration details.

**Issue: State inconsistency after simulation**

*Symptom*: Development type areas don't sum to expected total

*Diagnostic steps*:
1. Check that all transitions are properly defined
2. Verify no area is lost or gained during transitions
3. Check for floating-point rounding errors

*Solution*:
.. code-block:: python

   # Verify area conservation
   total_area = sum(sum(dt.areas[p] for p in range(fm.horizon+1))
                    for dt in fm.dtypes.values())
   print(f"Total area: {total_area:.2f} ha")

**Issue: Negative volumes or areas in output**

*Symptom*: Schedule contains negative values

*Diagnostic steps*:
1. Check growth curve values are non-negative
2. Verify harvest fractions don't exceed 1.0
3. Check for calculation errors in yield components

*Solution*:
- Validate input data before running simulation
- Add assertions in custom callbacks
- Check the :doc:`textbook/index` for modeling best practices

Performance Issues
------------------

**Issue: Slow simulation for large models**

*Symptom*: Simulation takes hours or days

*Diagnostic steps*:
1. Count number of development types
2. Check planning horizon length
3. Verify callback complexity

*Solution*:
- Reduce model size for testing
- Use parallel processing where possible
- Profile callbacks to identify bottlenecks
- Consider simplifying growth curves

**Issue: High memory usage**

*Symptom*: System runs out of memory

*Diagnostic steps*:
1. Check model size
2. Verify no memory leaks in callbacks
3. Check for large intermediate data structures

*Solution*:
- Process in batches
- Clear intermediate results
- Use generators instead of lists where possible

External Dependency Issues
--------------------------

**Issue: libCBM not installed**

*Symptom*: ImportError when using carbon callbacks

*Solution*:
.. code-block:: bash

   # Install libCBM
   pip install libcbm

   # Or install from source (see libCBM documentation)

**Issue: Gurobi license error**

*Symptom*: "License expired" or "Invalid license"

*Solution*:
1. Check Gurobi license is active
2. Verify environment variables are set correctly
3. Contact Gurobi support if license is valid but not recognized

**Issue: SpaDES integration fails**

*Symptom*: Cannot import spades_ws3 or reticulate errors

*Solution*:
1. Ensure R is installed and in PATH
2. Check that spades_ws3 R package is installed
3. Verify Python-R bridge is working: :code:`reticulate::py_config()`

Recovery Procedures
-------------------

**Issue: Model in inconsistent state after crash**

*Recovery*:
1. Save model state before running: :code:`pickle.dump(model, open('model.pkl', 'wb'))`
2. If crash occurs, reload from pickle: :code:`model = pickle.load(open('model.pkl', 'rb'))`
3. Re-run from last checkpoint

**Issue: Need to restart optimization**

*Recovery*:
1. Save intermediate results periodically
2. Use checkpointing in long simulations
3. Keep backup of input data and configuration

Prevention Best Practices
-------------------------

1. **Validate input data** before running models
2. **Test with small models** before scaling up
3. **Save model state** before long simulations
4. **Log all operations** for debugging
5. **Use version control** for all configuration files
6. **Document assumptions** and parameter choices

Further Reading
---------------

- :doc:`../textbook/index` — Comprehensive modeling concepts
- :doc:`limitations-and-boundaries` — Understanding ws3 boundaries
- :doc:`../reference/contracts/index` — Data contracts and runtime invariants