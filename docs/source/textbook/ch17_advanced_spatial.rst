.. _textbook_ch17_advanced_spatial:

==================================
Chapter 17: Advanced Spatial Modeling
==================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Understand advanced spatial constraints in forest optimization
- Implement adjacency and contiguous area requirements
- Model spatial connectivity and fragmentation
- Use raster-based spatial analysis with ws3
- Integrate GIS data into optimization models

Introduction
------------

Spatial modeling is crucial for realistic forest management optimization.
While basic ws3 models treat all areas as homogeneous, real forests have
spatial structure that affects:

- **Harvest logistics**: Distance to roads, landings, mills
- **Ecological requirements**: Habitat connectivity, buffer zones
- **Regulatory constraints**: Adjacency rules, contiguous areas
- **Visual impacts**: Screen distances, viewshed analysis

This chapter explores advanced spatial modeling techniques that go beyond
basic spatial allocation.

Spatial Constraints Overview
----------------------------

Common spatial constraints in forest optimization:

1. **Adjacency Constraints**: Prevent harvesting adjacent areas
2. **Contiguous Area Requirements**: Ensure harvested areas meet size minimums
3. **Spatial Connectivity**: Maintain landscape connectivity
4. **Buffer Zones**: Protect sensitive areas from harvesting
5. **Visual Quality**: Manage visual impacts from harvest blocks

**Mathematical Formulation**:

For adjacency constraints, we typically use:

.. math::

   x_{i,t} + x_{j,t} \leq 1 \quad \forall i,j \in \text{adjacent pairs}, \forall t

Where :math:`x_{i,t}` is a binary variable indicating whether area :math:`i`
is harvested in period :math:`t`.

Implementing Adjacency Constraints
-----------------------------------

**Step 1: Define Adjacency Matrix**

.. code-block:: python

   import numpy as np
   import geopandas as gpd
   
   # Load spatial data
   spatial_df = gpd.read_file("data/spatial_inventory.geojson")
   
   # Create adjacency matrix
   def create_adjacency_matrix(geometry_series):
       """Create binary adjacency matrix from geometries."""
       n = len(geometry_series)
       adj_matrix = np.zeros((n, n), dtype=int)
       
       for i in range(n):
           for j in range(i+1, n):
               if geometry_series[i].touches(geometry_series[j]):
                   adj_matrix[i, j] = 1
                   adj_matrix[j, i] = 1
       
       return adj_matrix
   
   adj_matrix = create_adjacency_matrix(spatial_df.geometry)
   print(f"Adjacency matrix shape: {adj_matrix.shape}")
   print(f"Number of adjacent pairs: {adj_matrix.sum() // 2}")

**Step 2: Add Adjacency Constraints to Problem**

.. code-block:: python

   from ws3.opt import Problem
   
   def add_adjacency_constraints(problem, adj_matrix, dt_mapping):
       """Add adjacency constraints to optimization problem.
       
       :param problem: ws3 optimization problem
       :param adj_matrix: adjacency matrix
       :param dt_mapping: mapping from spatial units to DT codes
       """
       n_spatial = adj_matrix.shape[0]
       
       # Find adjacent pairs
       adjacent_pairs = []
       for i in range(n_spatial):
           for j in range(i+1, n_spatial):
               if adj_matrix[i, j] == 1:
                   adjacent_pairs.append((i, j))
       
       print(f"Adding {len(adjacent_pairs)} adjacency constraints")
       
       # Add constraints
       for idx, (i, j) in enumerate(adjacent_pairs):
           # Get DT codes for adjacent areas
           dt_i = dt_mapping[i]
           dt_j = dt_mapping[j]
           
           # Create constraint: x_{i,t} + x_{j,t} <= 1
           constraint_name = f"adj_{i}_{j}"
           coeffs = {
               f"{dt_i}_t": 1.0,
               f"{dt_j}_t": 1.0,
           }
           
           problem.add_constraint(
               name=constraint_name,
               coeffs=coeffs,
               sense='<',
               rhs=1.0
           )

**Step 3: Solve with Adjacency Constraints**

.. code-block:: python

   # Compile scenario with adjacency constraints
   problem = compile_scenario(fm, scenario_name="adjacency_test")
   
   # Add adjacency constraints
   add_adjacency_constraints(problem, adj_matrix, dt_mapping)
   
   # Solve
   solution = problem.solve(solver="gurobi")
   
   print(f"Status: {solution.status()}")
   print(f"Objective value: {solution.get_objective_value():.2f}")

Contiguous Area Requirements
-----------------------------

**Problem**: Harvested areas must meet minimum size requirements for
economic viability and regulatory compliance.

**Mathematical Formulation**:

For contiguous area requirements, we need to ensure that harvested areas
form connected components of sufficient size:

.. math::

   \sum_{i \in C} x_{i,t} \geq A_{min} \cdot y_{C,t} \quad \forall C \in \mathcal{C}, \forall t

Where :math:`\mathcal{C}` is the set of all possible connected components,
:math:`A_{min}` is the minimum contiguous area, and :math:`y_{C,t}` is a
binary variable indicating whether component :math:`C` is harvested in
period :math:`t`.

**Implementation Approach**:

1. **Identify Connected Components**: Use graph theory to find connected
   components in the harvest schedule
2. **Size Constraints**: Add constraints to ensure minimum size
3. **Binary Variables**: Use binary variables to indicate which components
   are harvested

.. code-block:: python

   from scipy.sparse.csgraph import connected_components
   
   def check_contiguous_areas(schedule, spatial_df, min_area):
       """Check if harvested areas meet contiguous area requirements.
       
       :param schedule: harvest schedule with area and period columns
       :param spatial_df: spatial inventory with geometry
       :param min_area: minimum contiguous area requirement
       :return: boolean indicating if requirements met
       """
       # Get harvested areas
       harvested = spatial_df[spatial_df['harvested'] == 1]
       
       if harvested.empty:
           return True
       
       # Find connected components
       n_components, labels = connected_components(
           adj_matrix[harvested.index.values],
           directed=False
       )
       
       # Check size of each component
       for comp_id in range(n_components):
           comp_area = harvested[labels == comp_id]['area_ha'].sum()
           if comp_area < min_area:
               print(f"Component {comp_id} too small: {comp_area:.2f} ha < {min_area} ha")
               return False
       
       return True

Raster-Based Spatial Analysis
------------------------------

**Advantages of Raster Data**:

- **Uniform resolution**: Consistent cell sizes
- **Easy computation**: Fast neighborhood operations
- **Integration**: Compatible with GIS and remote sensing
- **Flexibility**: Easy to add layers (elevation, slope, etc.)

**Converting Vector to Raster**:

.. code-block:: python

   import rasterio
   from rasterio.features import geometry_mask
   
   def vector_to_raster(geometry_df, raster_template, field='dt_code'):
       """Convert vector geometries to raster.
       
       :param geometry_df: geopandas DataFrame with geometries
       :param raster_template: rasterio Dataset for reference
       :param field: attribute field to use as values
       :return: raster array
       """
       # Create output array
       out_image = np.zeros(raster_template.shape, dtype=raster_template.dtypes[0])
       
       # Create mask from geometries
       shapes = [
           (geom, value) 
           for geom, value in zip(geometry_df.geometry, geometry_df[field])
       ]
       
       # Rasterize
       out_image = rasterio.features.rasterize(
           shapes,
           out_shape=raster_template.shape,
           fill=0,
           transform=raster_template.transform,
           dtype=raster_template.dtypes[0]
       )
       
       return out_image
   
   # Convert spatial data to raster
   dt_raster = vector_to_raster(spatial_df, raster_template)
   print(f"Raster shape: {dt_raster.shape}")

**Using Raster in Optimization**:

.. code-block:: python

   def add_raster_constraints(problem, raster, constraint_type='buffer'):
       """Add constraints based on raster data.
       
       :param problem: ws3 optimization problem
       :param raster: raster array
       :param constraint_type: type of constraint ('buffer', 'slope', etc.)
       """
       if constraint_type == 'buffer':
           # Get buffer zones (e.g., near water bodies)
           buffer_mask = raster == BUFFER_VALUE
           
           # Add constraints to prevent harvesting in buffer zones
           for row in range(raster.shape[0]):
               for col in range(raster.shape[1]):
                   if buffer_mask[row, col]:
                       # Add constraint for this cell
                       cell_var = f"cell_{row}_{col}"
                       if cell_var in problem._vars:
                           problem.add_constraint(
                               name=f"buffer_{row}_{col}",
                               coeffs={cell_var: 1.0},
                               sense='=',
                               rhs=0.0
                           )

Case Study: Adjacency Constraints in TSA 24
--------------------------------------------

**Objective**: Implement adjacency constraints for TSA 24 to ensure
sustainable forest management.

**Data Requirements**:

- Spatial inventory with polygon geometries
- Adjacency relationships between polygons
- Minimum harvest block size (e.g., 50 hectares)

**Implementation Steps**:

1. Load spatial data and create adjacency matrix
2. Compile base optimization scenario
3. Add adjacency constraints
4. Solve and compare results
5. Analyze impact on harvest schedule

.. code-block:: python

   # Load TSA 24 spatial data
   tsa24_spatial = gpd.read_file("data/tsa24_spatial.geojson")
   
   # Create adjacency matrix
   adj_matrix = create_adjacency_matrix(tsa24_spatial.geometry)
   
   # Compile scenario
   problem = compile_scenario(fm, scenario_name="tsa24_adjacency")
   
   # Add adjacency constraints
   add_adjacency_constraints(problem, adj_matrix, tsa24_spatial['dt_code'])
   
   # Solve
   solution = problem.solve(solver="gurobi")
   
   # Analyze results
   schedule = solution.get_schedule()
   print(f"Total harvest area: {schedule['area_ha'].sum():.2f} ha")
   print(f"Number of harvest blocks: {schedule['dt_code'].nunique()}")

Summary
-------

This chapter covered advanced spatial modeling techniques for forest
optimization:

- **Adjacency constraints**: Prevent harvesting adjacent areas
- **Contiguous area requirements**: Ensure minimum harvest block sizes
- **Raster-based analysis**: Use raster data for spatial constraints
- **Case study**: Applied techniques to TSA 24

These techniques enable more realistic and practical forest management
optimization that accounts for spatial structure and constraints.

Exercises
---------

1. **Adjacency Matrix**: Create an adjacency matrix for a simple 3x3 grid
   of forest polygons. Verify that each polygon is adjacent to its
   neighbors.

2. **Contiguous Areas**: Modify the adjacency-constrained optimization
   to add contiguous area requirements. Test with different minimum
   area thresholds.

3. **Raster Constraints**: Convert a slope raster to constraints that
   prevent harvesting on steep slopes (>30%).

4. **Performance Impact**: Measure the impact of adjacency constraints
   on solve time and objective value. At what point do constraints make
   the problem infeasible?

5. **Real Data**: Apply these techniques to a real forest inventory
   dataset. Compare results with and without spatial constraints.

Related Resources
-----------------

* :doc:`spatial-schedule-allocation` (how-to guide)
* :doc:`../textbook/ch06_spatial_allocation` (basic spatial modeling)
* GeoPandas documentation: https://geopandas.org/
* Rasterio documentation: https://rasterio.readthedocs.io/