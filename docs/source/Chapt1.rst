******************
Wood Supply Models
******************

Wood supply models (WSM) are software packages that simulate forest activities 
to assist in the development of forest plans, wood supply analysis and sustainable 
forest management. WSM aim to provide foresters, landowners and stakeholders with 
information about how a forested land base will progress through time. They help 
by providing a glimpse into the future of how a forested land base could change 
through time given a particular set of proposed management objectives and activities. 

Although there are many different WSM software implementations available for forest 
planners, there are some commonalities across these implementations with respect to 
the required input data, the development process and approach to identifying a solution.

Area of Interest
================

All WSP require the identification of an Area of Interest (AOI). On a simple level 
an AOI is spatial explicit delineation of a forested area that you wish to project 
forward through time on which activities are applied. AOI's can be defined by legal 
or administrative boundaries, terrain or landscape features, forest type or any 
characteristic that separates the area you are interested vs. the area you are not 
interested in. In some instances defining an AOI is simple process and potentially 
has been predefined by higher level management, the landowners or other stakeholders. 
In other instances AOI's are not predefined and require defining at the start of the 
project. 

Defining AOI
------------

Defining an AOI is typically the starting point for a WSPP. Generally, an AOI is 
defined in a spatially explicit way based on landscape features or landownership. If 
you are starting a WSP project and are tasked with delineating an AOI, the first place 
to start is by having a discussion with the project stakeholders. The second step is 
finding the required corresponding spatial data layers. Depending on the context, 
this step may be more or less difficult based on availability of geospatial databases.   

Generally, an AOI will be defined by a vector polygon dataset (e.g., stored in ESRI 
Shapefile or Geodatabase layer format, or perhaps as a geoJSON file). This polygon data
layer outlines the boundary of the area you want to model. Anything outside the AOI 
boundary will *not* be included in your model. All other spatial data being used as 
part of the project will be clipped to the extent of your AOI. Often multiple data 
layers will be used to define an AOI. For example, one aspect of an AOI might be defined 
by the presence of private landownership and another aspect might be defined by the 
extent of the forest. 

Landscape Classification
------------------------

In addition to defining an AOI, WSP typically require some initial landscape 
classification. This initial landscape classification is an essential part of WSM 
since features not defined are inevitably unmanageable. When starting an initial 
landscape classification, most WSP will start by ensuring that all legally required 
management objectives are included. For example in most jursidictions in Canada, different 
riparian classes have different management requirements, including different *no harvest* 
buffers. Collecting spatial data of different riparian features is required, and creating 
of legal sized buffers so it is possible to change the potential future harvesting 
activities that can take place on the land base. 

Beyond identifying legally mandated management requirements, this initial landscape 
classification will often include delineation of the Timber Harvest Land Base (THLB) and 
the Non-Timber Harvest Land Base (NTHLB). The THLB includes all forested lands that the 
WSP will be able to harvest volume from for the duration of the planning horizon. The 
NTHLB includes all areas classified as not forested, and any forested stands that are 
excluded from harvesting for the duration of the planning horizon for other reasons
(e.g., inaccessible, low productivity, steep slopes, high host, protected status, etc.). 
THLB and NTHLB net-down are often displayed in summary tables in timber supply review
technical documentation (outlining the different features and areas that are in each 
section of the AOI). Completing this step is typically a time consuming process, and 
requires a strong understanding of regionally appropriate forest legislation and input 
from multiple stakeholders. 

The more care that is taken in this step, the easier it will be to ensure that you create 
a WSM that is adaptable enough to capture the diversity of different scenarios you might 
want to build and model into the future. Ideally, a WSP for one area will have one set 
of input data and landscape classification schema. Sometimes this is not possible, and it 
might be required to build two (or more) concurrent models with distinct classification schema 
to see how this impacts future wood supply. An example of this might be delineating the 
THLB and NTHLB differently which would potentially require two distinct landscape 
classifications.    

Ultimately, all of these different features will be combined into one layer or file, that will 
be joined with initial forest inventory. 

Initial Forest Inventory
========================

Given that the primary purpose of a WSM is to project the state of a forested 
area through time under various patterns of management activities (and possibly 
natural disturbances or other external factors), it is essential to start this 
projection from an initial forest inventory. Initial inventories can have many 
different structures, but generally contain information about stand delineation, 
merchantable wood volume, tree species and age, site productivity, biogeoclimactic 
variables, management or disturbance history, administrative or other zoning, 
slope, watershed membership, etc.  

All initial forest inventories need to have some initial grouping into stands or 
blocks. This can be presented spatially or aspatially. Spatial delineations of 
stands are polygons defined simply by being different from their neighbours. 
These spatial block or boundary delineations often use tree species, volume, age, 
and terrain features (such as rivers, or streams) to determine where one areas 
starts and ends. These initial stand delineations often (but not always) are linked 
with stand origin. An area that had the same disturbance (either anthropomorphic 
or natural) and the same regeneration (planted or natural), at the same time, in 
one spatially continuous extent, will often reflect one group or delineation. 
Aspaital delineation is much the same as spatial except different groups do not 
have neighbours or defined polygon boundaries. 

Stratifying the Forest 
======================

The forest inventory data is typically aggregated into a manageable number of 
*development types* (i.e., *strata* or *analysis units*),  which simplifies the 
modelling, by reducing the state space that needs to be modelled. The exact mechanism 
for stratifying the forest will vary from one WSM implementation to another, but 
typically there will be a way to define a number of *stratification variables* 
that will be used to aggregate stands into stand types. These stratification 
variables are often derived from the forest inventory data (e.g., biogeoclimactic 
zone, leading species, watershed, THLB, first or second growth status, etc.). 
A *stratum* will be defined by unique combinations of these stratification variables. 
The *curse of dimensionality* definitely applies here---the more stratification 
variables there are the more possible combinations that exist, resulting in 
geometric growth of the state space (i.e., potential number of development types) 
as a function of the number of stratification variables.

Stand age (or age class) is generally not used as a stratification in WSM software 
implementations, because it is linked to time and is reserved for use as the 
dependent (input) variable in the yield curves that WSM use to predict how 
stands evolve (grow) between stand-replacing disturbances.

Basic Components of WSM State Logic
===================================

The WSM state logic is a set of rules that define how the stand attributes can
change over time. The set of all possible *states* that a given stand can be in
is the *state space*. A WSM simulates trajectories through state space, primarily 
driven by the passage of time and application of disturbances. 

Below, we define some basic concepts that are used in the WSM state logic.

Development Types
-----------------

Each development type is linked to *growth and yield* functions describing the 
change in key stand attributes.

Each development type may also be associated with one or more *actions*, which can yield 
*output products* (e.g., species-wise assortments of raw timber products, cost incurred 
from exection of managment activities, treated area, etc.). Development types can also
be used to define *transitions*, which are transitions between development types as a
result of actions being applied at specific ages.

Yield Curves
------------

Growth and yield functions are one of the major inputs to WSM. These will be used to 
predict and project stand attributes over time, as a function of stand age (e.g., 
species-wise standing timber volume, number of merchantable stems per unit area, 
wildlife habitat suitability index value, etc.). These growth and yield functions could
be the output from a stand growth model, yield model, or other stand-level models that
predict changes in stand attributes as a function of stand age. They could also be 
derived from expert knowledge, or from output from various simulation or statistical 
regression models. 

Actions 
-------

Actions are used to apply management actions at specific ages in WSM. Actions are typically used
simulate various silviculture treatments (e.g., site preparation, planting, pre-commercial thinning,
fertilization, commercial thinning, final felling, etc.). However, actions may also be used
to model transitions in management policy, such as transitions between management regimes (e.g., 
transition to an *intensive* management regime could trigger eligibility for various intensive
management actions).

Action *eligibility* is typically defined as a function of management regime (which is implicitly
encoded into the stratification variables that define development types), whereas action 
*operability* is defined as a function of stand age (which may vary by development type, e.g., 
some stand types grow faster than others so would be eligible for harvesting at a younger age).

Transitions
-----------

Applying an action to a development type induces a *state transition* (i.e., applying an action 
may modify one or more stratification variables, effectively transitioning the treated area to a 
different development type). 

Scenarios
=========

The set of all possible combinations of actions, across development types and time steps, is 
the *solution space* for a given WSM. One of the primary functions of a WSM is the allow the 
analyst to explore this solution space, and to generate *scenarios*. A scenario is a specific 
combination of actions that simulates a specific management regime. A scenario is typically 
generated by emulating existing mangement policies, or by exploring a new management regime.

There are two basic approached that can be used (independently, or in combination) to generate 
the dynamic activity schedules for each scenario.

Heuristics
----------

The simplest approach, which we call the *heuristic* activity scheduling method, involves defining 
period-wise targets for a single key output (e.g., total harvest volume) along with a set of rules 
that determines the order in  which actions are applied to eligible development types. At each time 
step, the model iteratively applies actions according to the rules until the output target value is 
met, or it runs out of eligible area. At this point, the model simulates one time-step worth of 
growth, and the process repeats until the end of the planning horizon.

Optimization
------------

A slightly more complex approach, which we call the *optimization* activity scheduling method, 
involves defining an  optimization problem (i.e., an objective function and constraints), and solving 
this problem to optimality (using one of several available third-party mathematical solver software 
packages).

Although the optimization approach is more powerful than the heuristic approach for modelling 
harvesting and other anthopic activities, an optimization approach is not appropriate for modelling 
strongly-stochastic disturbance processes (e.g., wildfire, insect invasions, blowdown). Thus, a hybrid 
heuristic-optimization approach may be best when modelling a combination of anthopic and natural 
disturbance processes.

.. Types of Models
.. ================

.. Spatial
.. ----------------

.. Aspatial
.. ----------------

.. Linear Programming
.. ================

.. Model 1
.. ----------------

.. Model 2
.. ----------------

.. Model 3
.. ----------------
