****************************
Wood Supply Models
****************************

Area of Interest
================

Landscape Classification
----------------

Initial Forest Inventory
================

Stratifying the Forest 
================

The forest inventory data is typically aggregated into a manageable number of *strata* (i.e., *development types*),  which simplifies the modelling. 

Development Types
----------------

Each development type is linked to *growth and yield* functions describing the change in key attributes attributes (e.g., species-wise standing timber volume, number of merchantable stems per unit area, wildlife habitat suitability index value, etc.) expressed as a function of stratum age.

Each development type may also be associated with one or more *actions*, which can yield *output products* (e.g., species-wise assortments of raw timber products, cost, treated area, etc.).

Growth and Yield
================

Forest Management Activities
================

Transitions
================

Applying an action to a development type induces a *state transition* (i.e., applying an action may modify one or more stratification variables, effectively transitioning the treated area to a different development type). 

Constraints
================

Scenarios
================

There are two basic approached that can be used (independently, or in combination) to generate the dynamic activity  schedules for each scenario.

Heuristics
----------------

The simplest approach, which we call the *heuristic* activity scheduling method, involves defining period-wise targets for a single key output (e.g., total harvest volume) along with a set of rules that determines the order in  which actions are applied to eligible development types. At each time step, the model iteratively applies actions according to the rules until the output target value is met, or it runs out of eligible area. At this point, the model simulates one time-step worth of growth, and the process repeats until the end of the planning horizon.

Optimization
----------------

A slightly more complex approach, which we call the *optimization* activity scheduling method, involves defining an  optimization problem (i.e., an objective function and constraints), and solving this problem to optimality (using one of several available third-party mathematical solver software packages).

Although the optimization approach is more powerful than the heuristic approach for modelling harvesting and other anthopic activities, an optimization approach is not appropriate for modelling strongly-stochastic disturbance processes (e.g., wildfire, insect invasions, blowdown). Thus, a hybrid heuristic-optimization approach may be best when modelling a combination of anthopic and natural disturbance processes.

Types of Models
================

Spatial
----------------

Aspatial
----------------

Linear Programming
================

Model 1
----------------

Model 2
----------------

Model 3
----------------

