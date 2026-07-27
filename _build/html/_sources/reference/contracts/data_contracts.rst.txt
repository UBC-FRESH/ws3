.. _contract-data-contracts:

=================
Data Contracts
=================

This page defines the data formats that ws3 expects and produces.

Development Type Contract
-------------------------

Development types are represented as :py:class:`ws3.forest.DevelopmentType` instances,
keyed by a tuple of theme values (one per theme) stored in ``ForestModel.dtypes``.

A development type is identified by its **key** — a tuple of theme values (e.g.,
``('SP', 50, 'T1')`` for species=SP, site_index=50, theme1=T1). Each development
type encapsulates:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - key
     - tuple[str, ...]
     - Unique identifier: tuple of theme values
   * - parent
     - ForestModel
     - Reference to owning model
   * - _ages_curve
     - core.Curve
     - Age curve for the development type
   * - _ycomps
     - dict[str, Curve]
     - Yield component curves keyed by name
   * - oper_expr
     - defaultdict(list)
     - Operability expressions per action code
   * - transitions
     - dict[(str, int), list]
     - Action/age → target development types
   * - _areas
     - dict[int, defaultdict(float)]
     - Area by period and age

Example:

.. code-block:: python

   # Development types are created automatically when areas are imported
   model.import_areas_section()
   # Access a development type:
   dt = model.dtypes[('SP', 50, 'T1')]

Action Contract
---------------

Actions are represented as :py:class:`ws3.forest.Action` instances stored in
``ForestModel.actions`` (dict keyed by action code string).

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Attribute
     - Type
     - Description
   * - code
     - str
     - Unique action identifier (e.g., "harvest")
   * - targetage
     - int or None
     - Target age for the action (None = any age)
   * - descr
     - str
     - Human-readable description
   * - lockexempt
     - bool
     - Whether action bypasses age locks
   * - components
     - list[str]
     - Yield components affected (for aggregate actions)
   * - partial
     - list[str]
     - Partial yield components
   * - is_harvest
     - int
     - 1 if harvest action, 0 otherwise
   * - is_sticky
     - int
     - 1 if action persists across periods

Example:

.. code-block:: python

   action = model.actions['harvest']
   print(action.code, action.descr)  # 'harvest', 'Clearcut harvest'

Growth Curve Contract
---------------------

Growth curves are represented as :py:class:`ws3.core.Curve` instances.
Curves are registered with the model via :py:meth:`ws3.forest.ForestModel.register_curve`.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - label
     - str or None
     - Label for the curve
   * - id
     - str or None
     - ID for the curve
   * - is_volume
     - bool
     - Whether the curve tracks volume
   * - points
     - list[tuple[int, float]]
     - List of (x, y) coordinate pairs
   * - type
     - str
     - Curve type: 'a' (age-based), 't' (time-based), 'c' (complex)
   * - is_special
     - bool
     - Immune to simplification
   * - period_length
     - float
     - Length of planning period in years
   * - xmin
     - int
     - Minimum x value (default: 0)
   * - xmax
     - int
     - Maximum x value (default: 200)
   * - epsilon
     - float
     - Tolerance for curve simplification
   * - simplify
     - bool
     - Whether to simplify the curve on construction

Example:

.. code-block:: python

   from ws3.core import Curve
   curve = Curve(
       label='vol_SP50',
       points=[(0, 0), (10, 25.0), (20, 55.0), (30, 95.0), (40, 150.0), (50, 220.0)],
       is_volume=True,
       type='a',
       period_length=10
   )
   registered = model.register_curve(curve)

Yields Data Structure
---------------------

Yields are stored as a **list** of tuples in ``ForestModel.yields``. Each entry
is a tuple of ``(mask, ytype, ycomps)`` where:

- ``mask`` — tuple of theme values (e.g., ``('SP', 50)``)
- ``ytype`` — one of ``'a'`` (age-based), ``'t'`` (time-based), ``'c'`` (complex)
- ``ycomps`` — list of ``(yname, Curve)`` tuples

Schedule Output Contract
------------------------

Schedules are compiled as lists of tuples via :py:meth:`ws3.forest.ForestModel.compile_schedule`.
Each tuple has the format ``(dtype_key, age, area, acode, period, etype)``.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Element
     - Type
     - Description
   * - dtype_key
     - tuple[str, ...]
     - Development type key
   * - age
     - int
     - Age at which action was applied
   * - area
     - float
     - Area harvested (hectares)
   * - acode
     - str
     - Action code
   * - period
     - int
     - Planning period (1-indexed)
   * - etype
     - str
     - ``'_existing'`` or ``'_future'``

Example:

.. code-block:: python

   schedule = model.compile_schedule(problem)
   for dtk, age, area, acode, period, etype in schedule:
       print(f"Period {period}: {acode} on {dtk} at age {age}, {area} ha")