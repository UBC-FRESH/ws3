****************************
Wood Supply Models
****************************

Wood supply models (WSM) are a suite of software and packages that simulate forest activities to assist in the development of forest plans, wood supply analysis and sustainable forest management. 
WSM aim to provide foresters, landowners and stakeholders with information about how a forested land base will progress through time. They help by providing a glimpse into the future of how a forested land base could change through time given a particular set of proposed management objectives and activities. 
Although there are many different WSM available for forest planners there are some commonalities between the required input data, the development process and approach to identifying a solution.

Area of Interest
================
All WSP require the identification of an Area of Interest (AOI). On a simple level an AOI is spatial explicit delineation of a forested area that you wish to project forward through time on which activities are applied. AOI's can be defined by legal or administrative boundaries, terrain or landscape features, forest type or any characteristic that separates the area you are interested vs. the area you are not interested in. In some instances defining an AOI is simple process and potentially has been predefined by higher level management, the landowners or other stakeholders. In other instances AOI's are not predefined and require defining at the start of the project. 

Defining AOI's
----------------


Landscape Classification
----------------

Initial Forest Inventory
================

Since WSM aim to project a forested area through time, it is essential that there is an initial inventory. Initial inventories can have many different structures but, generally contain information about stand delineation or polygons, volume, tree species, 

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

