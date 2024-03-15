****************************
Overview
****************************

Wood Supply Simulation System (``ws3``) is an open-open-source Python software package that is designed to model *wood supply planning problems* (WSPP), in the context of  sustainable forest management. 

A WSPP consists of determining the location, nature, and timing of forest management activities (i.e., *actions*) for a given forest, typically over multiple planning periods or rotations. The planning horizon often spans over 100 years or more. WSPP are intently complicated problems. In practice WSPP are supported by complex software models that that simulate different sequences of *actions* and *growth* for each time step, starting from and initial forest inventory. These software models are typically classified as wood supply models.

All wood supply models (WSM) require complex input data sets, which can be divided into *static* and *dynamic* components. Static WSM input data types include initial forest inventory, growth and yield curves, action definitions, transition definitions, and a schedule of prescribed activities. Dynamic WSM input data may include a combination of heuristic and optimization-based processes to automatically derive a dynamic activity schedule. The dynamic inputs get layered on top of the static activity schedule.

Given a set of static inputs, a given WSM can be used to simulate a number of *scenarios*. Generally, scenarios differ only in terms of the dynamic activity schedule that is simulated. Comparing output from several scenarios is the basic mechanism by which forest managers derive insight from wood supply models.

``ws3`` is backed by a well documented application programming interface (API), which can be customized to control the way ``ws3`` behaves. 

About this User Guide
================

This guide is divided into four main sections. The first section describes general concepts about WSP with specific linkages to the ``ws3`` requirements. This section covers ideas and requirements that are consistent across all WSP and is designed to ensure proper language use and highlight commonalities. Users who are familiar with WSM will likely find this section a review but, it will help ensure clarity for successive sections in this guide. 

The second section provides specific and detailed information about ``ws3`` and provides guidance on data preparation and running a wood supply simulation. 

The third section provides information for connecting ``ws3`` to libCMB, allowing for carbon accounting to be included as a dynamic value within the WSM. 

The fourth section provides information on connecting ``ws3`` to SpaDES. SpaDES is a **SPA**\ tially explicit **D**\ iscrete **E**\ vent **S**\ imulation used for disturbance modelling.  

About F.R.E.S.H.
==========================

words.

Contact information
==========================

Add in contact information - should we have a seperate email that bumps to someone?


Move these sections to different part - don't think that they should count as "over view" seem more complicated
=========================
Overview of Main Classes and Functions
=========================

This section describes some of the main classes and functions that make up.

The ``ForestModel`` class is the core class in the package. This class encapsulates all the information used to 
simulate scenarios from a given dataset (i.e., stratified intial inventory, growth and yield functions, action 
eligibility, transition matrix, action schedule, etc.), as well as a large collection of functions to import and 
export data, generate activity schedules, and simulate application of these schedules  (i.e., run scenarios).

At the heart of the ``ForestModel`` class is a list of ``DevelopentType`` instances. Each ``DevelopmentType`` 
instance encapsulates information about one development type (i.e., a forest stratum, which is an aggregate of 
smaller *stands* that make up the raw forest inventory input data). The ``DevelopmentType`` class also stores a 
list of operable *actions*, maps *state variable transitions* to these actions, stores growth and yield functions, 
and knows how to *grow itself* when time is incremented during a simulation.

.. To Do: Finish documenting main stuff here.
 
Common Use Case and Sample Notebooks
===========================

In this section, we assume an interactive Jupyter Notebook environment is used to interface with ``ws3``.

A typical use case starts with creating an instance of the ``ForestModel`` class. Then, we need to load data into 
this instance, define one or more scenarios (using a mix of heuristic and optimization approaches), run the 
scenarios, and export output data to a format suitable for analysis (or link to the next model in a larger 
modelling pipeline).

The first step in typical workflow is to run a mix of standard ``ws3`` and custom data-importing functions.  These 
functions import data from various sources, *on-the-fly* reformat this data to be compatible with ``ws3``, and load
 the reformated data into the ``ForestModel`` instance using standard methods. For example, ``ws3`` includes 
functions to import legacy Woodstock [#]_ model data (including LANDSCAPE, CONSTANTS, AREAS, YIELDS, LIFESPAN, 
ACTIONS, TRANSITIONS, and SCHEDULE section data), as well as functions to import and rasterize vector stand 
inventory data.

For example, one might define the following custom Python function in a Jupyter Notebook, to import data formatted 
for Woodstock.::

    def instantiate_forestmodel(model_name, model_path, horizon,
                                period_length, max_age, add_null_action=True):
        fm = ForestModel(model_name=model_name, 
	 	 	 model_path=model_path, 
 	 		 horizon=horizon,     
			 period_length=period_length,
			 max_age=max_age)
	fm.import_landscape_section()
	fm.import_areas_section()
	fm.import_yields_section()
	fm.import_actions_section()
	fm.add_null_action()
	fm.import_transitions_section()
	fm.reset_actions()
	fm.initialize_areas()
	fm.grow()
	return fm

The next step in a typical workflow is to define one or more scenarios. Assuming that we are using an optimization 
approach to harvest scheduling, we need to define an objective function (e.g., maximize total harvest volume) and 
constraints (e.g., species-wise volume and area even-flow constraints, ending standing inventory constraints, 
periodic minimum late-seral-stage area constraints) [#]_, build the optimization model matrix, solve the model to 
optimality [#]_. 


.. [#] Woodstock software is part of `Remsoft Solution Suite <http://www.remsoft.com/forestry.php>`_. 
.. [#] ``ws3`` currently implements functions to formulate and solve *Model I* wood supply optimization 
problems---however, the package was deliberately designed to make it easy to transparently switch between *Model I*
,  *Model II* and *Model III* formulations without affecting the rest of the modelling workflow. ``ws3`` currently 
has placeholder function stubs for *Model II* and *Model III* formulations, which will be implemented in later 
versions as the need arises. For more information on wood supply model formulations, see Chapter 16 of the 
`Handbook of Operations Research in Natural Resources <http://www.springer.com/gp/book/9780387718149>`_.
.. [#] ``ws3`` currently uses the `Gurobi <http://www.gurobi.com/>`_ solver to solve the linear programming (LP) 
problems to optimality. We chose Gurobi because it is one of the top two solvers currently available (along with 
the `CPLEX <https://www.ibm.com/analytics/data-science/prescriptive-analytics/cplex-optimizer>`_ solver), has a 
simple and flexible policy for requesting unlimited licences for free use in research projects, has elegant Python 
bindings, and we like the technical documentation. However, we deliberately used a modular design, which allows us 
to transparently switch to a different solver in ``ws3`` without affecting the rest of the workflow---this design 
will make it easy to implement an interface to addional solvers in future releases.


 
