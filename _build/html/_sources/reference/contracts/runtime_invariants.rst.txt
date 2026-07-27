.. _contract-runtime-invariants:

=========================
Runtime Invariants
=========================

This page documents invariants that must hold during ws3 execution.

Model State Invariants
----------------------

1. **Development type keys are unique tuples**

   Each development type is identified by a unique tuple of theme values
   (e.g., ``('SP', 50, 'T1')``). Duplicate keys in ``ForestModel.dtypes``
   will cause undefined behavior.

2. **Yield curves are registered before use**

   Every yield component curve referenced by a development type must have
   been registered with the model via :py:meth:`ws3.forest.ForestModel.register_curve`.
   Curves are keyed by their point list — two curves with identical points
   share the same registered instance.

3. **Action transitions reference valid development types**

   All target development types produced by action transitions must be
   resolvable from the source development type's theme values.

4. **Area is non-negative**

   All area values stored in ``DevelopmentType._areas`` must be >= 0.
   Negative areas indicate a bug.

5. **Ages are non-negative integers**

   Stand ages must be non-negative integers. The ``ages`` list in
   ``ForestModel`` is ``list(range(max_age + 1))``.

6. **Volume curves are monotonically non-decreasing (with exceptions)**

   Volume curves should not decrease with age, except for thinning effects
   or harvest-related yield components.

7. **Curves are immutable after registration**

   Once a curve is registered via :py:meth:`ws3.forest.ForestModel.register_curve`,
   it is locked (``curve.is_locked = True``). Modifying a locked curve's points
   will corrupt the model's curve registry.

Optimization Invariants
-----------------------

1. **Flow constraints are feasible**

   Flow constraints must allow at least one feasible solution. If
   ``min_ratio > max_ratio``, the problem is infeasible.

2. **Area constraints don't exceed available area**

   Sum of max_area constraints across all periods must not exceed total
   available area.

3. **Harvest doesn't exceed growth**

   Harvest volume in any period should not exceed available volume from
   standing inventory.

4. **Problem variables are unique**

   Variable names must be unique within a :py:class:`ws3.opt.Problem` instance.
   Duplicate variable names will overwrite existing variables.

5. **Problem sense is consistent**

   The objective sense (``SENSE_MAXIMIZE`` or ``SENSE_MINIMIZE``) is set at
   construction and should not change after variables and constraints are added.

Schedule Application Invariants
-------------------------------

1. **Total area is conserved**

   Total area across all development types should remain constant when
   actions are applied, unless transitions create or destroy area.

2. **Transitions are deterministic**

   Given the same input state and actions, :py:meth:`ws3.forest.ForestModel.apply_schedule`
   should produce the same output.

3. **Applied actions are tracked per period**

   ``ForestModel.applied_actions[period][acode][dtype_key][age]`` stores
   the area applied. This structure must remain consistent after each
   schedule application.

4. **GreedyAreaSelector patches missing area**

   When :py:meth:`ws3.forest.ForestModel.apply_schedule` is called with
   ``recourse_enabled=True`` (default), the :py:class:`ws3.forest.GreedyAreaSelector`
   automatically fills any missing area from operable age classes.

Spatial Allocation Invariants
-----------------------------

1. **Raster dimensions match model horizon**

   The :py:class:`ws3.spatial.ForestRaster` creates one output GeoTIFF per
   (period, year_within_period) combination. The number of periods is
   determined by ``forestmodel.horizon``.

2. **Pixel area is computed from raster resolution**

   ``ForestRaster._pixel_area = d^2 * 0.0001`` converts square meters to
   hectares based on the raster's pixel dimension ``d``.

3. **ForestRaster is single-use**

   After calling :py:meth:`ws3.spatial.ForestRaster.commit` or
   :py:meth:`ws3.spatial.ForestRaster.cleanup`, the instance is expired.
   Further calls to :py:meth:`ws3.spatial.ForestRaster.allocate_schedule`
   will raise ``RuntimeError``.

Error Conditions
----------------

The following conditions indicate bugs or misconfiguration:

- Development type with zero area in schedule
- Action with no operability expressions defined
- Curve with fewer than 2 data points after simplification
- Negative volume or area values in output
- Infeasible optimization (no solution found)
- Locked curve modified after registration
- Missing yield component curve referenced by development type

Validation
----------

Use :doc:`../howto/model-validation` to check these invariants after running
your model.