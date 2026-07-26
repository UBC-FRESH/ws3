Chapter 14: Integrating ws3 with SpaDES
========================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain what SpaDES is and why integrate it with ws3
- Use the `spades_ws3` R module to link ws3 with SpaDES simulations
- Understand the event-driven architecture that connects the two systems
- Configure harvest scheduling modes (optimize vs. areacontrol)
- Interpret combined SpaDES-ws3 simulation output

What Is SpaDES?
---------------

**SpaDES** (SPAtial Event-driven Simulation Engine) is an R framework for
building spatially-explicit, event-driven forest landscape simulations.
It was developed at the University of British Columbia's Faculty of
Forestry and is maintained by the Predictive Ecology group.

SpaDES provides:

- **Event-driven simulation**: Time advances through discrete events
  (harvest, grow, disturbance) rather than fixed time steps
- **Spatial explicitness**: Simulations operate on raster landscapes
- **Modular architecture**: Build complex simulations from reusable modules
- **Stochasticity**: Support for random processes (fire, insects, wind)

The `spades_ws3` R module bridges SpaDES and ws3, allowing ws3's
optimization engine to drive harvest decisions in a SpaDES simulation.

.. mermaid::

   graph TD
     SPADES["SpaDES<br/>(R framework)"] --> MODULE["spades_ws3<br/>R module"]
     MODULE --> WS3["ws3<br/>(Python engine)"]
     MODULE --> RETICULATE["reticulate<br/>(R-Python bridge)"]
     WS3 --> SCHEDULE["Harvest schedule"]
     SCHEDULE --> MODULE
     MODULE --> LANDSCAPE["Landscape raster"]

Architecture
------------

The `spades_ws3` module works as follows:

1. **Initialization**: SpaDES loads the module, which uses `reticulate`
   to create a Python environment and import ws3
2. **Event scheduling**: Three events are scheduled each simulation year:
   - `harvest`: Apply the ws3 harvest schedule to the landscape
   - `grow`: Advance forest growth for all stands
   - `plot`/`save`: Output monitoring and data saving
3. **Harvest scheduling**: Two modes are supported:
   - **optimize**: ws3 solves the full optimization problem each period
   - **areacontrol**: ws3 allocates harvest based on target areas

.. code-block:: r

   # spades_ws3 module parameters
   parameters = rbind(
     defineParameter("horizon", "numeric", 1L, NA, NA,
                     "ws3 simulation horizon (periods)"),
     defineParameter("base.year", "numeric", 2015L, NA, NA,
                     "ws3 simulation base year"),
     defineParameter("scheduler.mode", "character", "optimize", NA, NA,
                     "Switch between 'optimize' and 'areacontrol'"),
     defineParameter("target.scalefactors", "numeric", NULL, NA, NA,
                     "Target areas scale factors"),
     defineParameter("workers", "numeric", 1L, NA, NA,
                     "number of worker threads")
   )

Harvest Scheduling Modes
------------------------

**Optimize mode** (`scheduler.mode = "optimize"`):

ws3 solves the full optimization problem at each time step, considering
the current state of the landscape. This produces the truly optimal
schedule but is computationally expensive.

**Area control mode** (`scheduler.mode = "areacontrol"`):

ws3 allocates harvest to meet target areas derived from a pre-computed
optimization. This is faster but less flexible — it follows a fixed
allocation plan.

.. mermaid::

   graph TD
     OPTIMIZE["Optimize mode"] --> SOLVE["Solve optimization<br/>each period"]
     SOLVE --> SCHEDULE["Optimal schedule"]
     AREACONTROL["Area control mode"] --> PLAN["Pre-computed plan"]
     PLAN --> ALLOCATE["Allocate to targets"]

Configuring the Integration
---------------------------

To use `spades_ws3` in a SpaDES simulation:

.. code-block:: r

   # Load required packages
   library(SpaDES.core)
   library(spades_ws3)

   # Define simulation parameters
   simList <- list(
     landscape = landscape_raster,  # RasterStack of stand age
     parameters = list(
       horizon = 20,
       base.year = 2015,
       scheduler.mode = "optimize",
       workers = 4,
       verbose = 1
     ),
     modules = list(
       spades_ws3
     ),
     events = list(
       list(module = "spades_ws3", eventTime = seq(2015, 2035, 1),
            eventType = c("init", "harvest", "grow", "save"))
     )
   )

   # Run the simulation
   simList <- spaDES::spaDES(simList)

Data Flow
---------

The data flow between SpaDES and ws3:

.. mermaid::

   graph LR
     LANDSCAPE["Landscape raster<br/>(SpaDES)"] --> AGGREGATE["Aggregate to DTs"]
     AGGREGATE --> WS3_MODEL["ws3 ForestModel"]
     WS3_MODEL --> OPTIMIZE["Optimization"]
     OPTIMIZE --> SCHEDULE["Harvest schedule"]
     SCHEDULE --> APPLY["Apply to landscape"]
     APPLY --> LANDSCAPE
     LANDSCAPE --> GROW["Apply growth"]
     GROW --> LANDSCAPE

The `spades_ws3` module aggregates the raster landscape into development
types, creates a ws3 ForestModel, runs optimization, and applies the
resulting harvest schedule back to the landscape.

Output and Monitoring
---------------------

SpaDES provides built-in monitoring through plot and save events:

.. code-block:: r

   # Configure output
   simList$parameters$.plotInitialTime = 2015
   simList$parameters$.plotInterval = 5
   simList$parameters$.saveInitialTime = 2015
   simList$parameters$.saveInterval = 5

After simulation, you can analyze the output:

.. code-block:: r

   # Access simulation results
   harvest_history = sim$harvest_history
   landscape_history = sim$landscape_history

   # Plot results
   plot(harvest_history$area_harvested, type = "l",
        xlab = "Year", ylab = "Area harvested (ha)")

Limitations and Considerations
------------------------------

1. **Performance**: Running ws3 optimization at each time step is
   computationally expensive. Use `areacontrol` mode for faster runs.
2. **Python environment**: The `reticulate` bridge requires a compatible
   Python installation with ws3 installed.
3. **Memory**: Large landscapes with many development types can consume
   significant memory.
4. **Parallelization**: Set `workers > 1` to parallelize ws3 computations.

Exercises
---------

**Exercise 1 (Easy)**: Set up a minimal SpaDES simulation with the
`spades_ws3` module in optimize mode.

**Exercise 2 (Medium)**: Compare the output of optimize vs. areacontrol
modes on the same landscape.

**Exercise 3 (Hard)**: Extend the `spades_ws3` module to include
disturbance events (fire, insects) that modify the landscape between
harvest and grow events.

Further Reading
---------------

- :doc:`ch11_femic_models` — Building models with FEMIC
- :doc:`ch12_fhops_integration` — Using fhops for harvest cost curves
- SpaDES documentation: https://spades.predictiveecology.org
- `spades_ws3` package: https://github.com/UBC-FRESH/spades_ws3