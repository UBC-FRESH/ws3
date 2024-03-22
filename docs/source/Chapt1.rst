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
For WSP defining an AOI is almost always the starting point. Generally, AOI's are defined in a spatially explicit way based on landscape features or landownership. If you are starting a WSP project and are tasked with delineating an AOI the first place to start is by having a discussion with the project stakeholders. The second step is finding the required corresponding spatial layers. Depending on where your geographically this step may be more or less difficult based on availability of geospatial databases.   

Generally, an AOI will be a simple shapefile (or layer in a geodatabase), that is an outline where inside the boundaries is the area you want to project (grow) forward though time and outside the boundary is the area you are not projecting through time. All other spatial data being used as part of the project will be clipped to the extent of your AOI. Often multiple data layers will be used to define an AOI. For example, one side of an AOI might be defined by the presence of private landownership and other sides might be defined by the extent of the forest. 

Landscape Classification
----------------
In addition to defining an AOI, WSP typically require some initial landscape classification. This initial landscape classification is an essential part of WSM since features not defined are inevitably unmanageable. When starting an initial landscape classification, most WSP will start by ensuring that all legally required management objectives are included. For example in B.C. different riparian classes have different management requirements, including different 'no harvest' buffers. Collecting spatial data of different riparian features is required, and creating of legal sized buffers so it is possible to change the potential future harvesting activities that can take place on the land base. 

Beyond identifying legally mandated management requirements this initial landscape classification will include delineation of the Timber Harvest Land Base (THLB) and the Non-Timber Harvest Land Base (NTHLB). The THLB includes all forested lands that the WSP will be able to harvest volume from for the duration of the planning horizon. The NTHLB includes all areas classified as not forested, and any forested stands that are excluded from harvesting for the duration of the planning horizon. THLB's and NTHLB's are often displayed in summary tables, outlining the different features and areas that are in each section of the AOI.  This is often a time consuming process and requires a strong understanding of regionally appropriate forest legislation, and input from multiple stakeholders. 

The more care that is taken in this step the easier it will be to ensure that you create a WSP that is adaptable enough to capture the diversity of different scenarios you might want to build and model into the future. Optimally, a WSP for one area will have one set of input data, landscape classification schema, sometimes this is not possible and it might be required to build two concurrent models with two different classifications to see how this impacts future wood supply. An example of this might be delineating the THLB and NTHLB differently which would potentially require two landscape classifications.    

All of these different features will be combined into one layer or file, that will be joined with initial forest inventory. 

Initial Forest Inventory
================

Since WSM aim is to project a forested area through time, it is essential that there is an initial forest inventory. Initial inventories can have many different structures but, generally contain information about stand delineation, volume, and tree species.  

All initial forest inventories need to have some initial grouping into stands or blocks. This can be presented spatially or aspatially. Spatial delineations of stands are polygons defined simply by being different from their neighbours. These spatial block or boundary delineations often use, tree species, volume, age, and terrain features (such as rivers, or streams) to determine where one areas starts and ends. These initial stand delineations often (but not always) are linked with stand origin. An area that had the same disturbance (either anthropomorphic or natural) and, the same regeneration (planted or natural), at the same time, in one spatially continuous extent, will often reflect one group or delineation. 
Aspaital delineation is much the same as spatial except different groups do not have neighbours or defined polygon boundaries, instead 

Stratifying the Forest 
================

The forest inventory data is typically aggregated into a manageable number of *strata* (i.e., *development types*, *analysis units*),  which simplifies the modelling, by reducing the number of individual growth and yield curves required. 



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

