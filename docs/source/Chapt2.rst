**************************
About the ``ws3`` Package
**************************

The ``ws3`` package is implemented using the Python programming language. ``ws3`` is an aspatial wood supply model, which applies actions to development types, simulates growth, and tracks inventory area at each time step. Aspatial models output aspatial activity schedules---each line of the output schedule specifies the stratification variable values (which constitute a unique key into the list of development types), the time step, the action code, and the area treated.

Because the model is aspatial, the area treated on a given line of the output schedule may not be spatially contiguous (i.e., the area may be geographically dispersed throughout the landscape). Furthermore, in the common case where only a subset of development type area is treated in a given time step, the aspatial model provides not information regarding which subset of available area is treated (and, conversely, not treated). Some applications (e.g., linking to spatially-explicit or highly--spatially-referenced models) require a spatially-explicit activity schedule. ``ws3`` includes a *spatial disturbance allocator* sub-module, which contains functions that can map aspatial multi-period action schedules onto a rasterized spatial representation of the forest.

``ws3`` uses a scripted Python interface to control the model, which provides maximum flexibility and makes it very  easy to automate modelling workflows. This ensures reproducible methodologies, and makes it relatively easy to link ``ws3`` models to other software packages to form complex modelling pipelines. The scripted interface also makes it relatively easy to implement custom data-importing functions, which makes it easier to import existing data from a variety of ad-hoc sources without the need to recompile the data into a standard ``ws3``-specific format (i.e., importing functions can be implemented such that the conversion process is fully automated and applied to raw input data *on the fly*). Similarly, users can easily implement custom functions to re-format ``ws3``  output data *on the fly* (either for static serialization to disk, or to be piped live into another process). 

Although we recommend using Jupyter Notebooks as an interactive interface to ``ws3`` (the package was specifically designed with an interactive notebook interface in mind), ``ws3`` functions can also be imported and run in fully scripted workflow (e.g., non-interactive batch processes can be run in a massively-paralleled workflow on high-performance--computing resources, if available). The ability to mix interactive and massively-paralleled non-interactive workflows is a unique feature of ``ws3``.

``ws3`` is a complex and flexible collection of functional software units. The following sections describe some of the main classes and functions in the package, and describe some common use cases, and link to sample notebooks that implement these use cases.

Overview of Main Classes and Functions
=========================

This section describes some of the main classes and functions that make up.

The ``ForestModel`` class is the core class in the package. This class encapsulates all the information used to simulate scenarios from a given dataset (i.e., stratified initial inventory, growth and yield functions, action eligibility, transition matrix, action schedule, etc.), as well as a large collection of functions to import and export data, generate activity schedules, and simulate application of these schedules  (i.e., run scenarios).

At the heart of the ``ForestModel`` class is a list of ``DevelopentType`` instances. Each ``DevelopmentType`` instance encapsulates information about one development type (i.e., a forest stratum, which is an aggregate of smaller *stands* that make up the raw forest inventory input data). The ``DevelopmentType`` class also stores a list of operable *actions*, maps *state variable transitions* to these actions, stores growth and yield functions, and knows how to *grow itself* when time is incremented during a simulation.

.. To Do: Finish documenting main stuff here.
 
Common Use Case and Sample Notebooks
===========================

In this section, we assume an interactive Jupyter Notebook environment is used to interface with ``ws3``.

A typical use case starts with creating an instance of the ``ForestModel`` class. Then, we need to load data into this instance, define one or more scenarios (using a mix of heuristic and optimization approaches), run the scenarios, and export output data to a format suitable for analysis (or link to the next model in a larger modelling pipeline).

The first step in typical workflow is to run a mix of standard ``ws3`` and custom data-importing functions.  These functions import data from various sources, *on-the-fly* reformat this data to be compatible with ``ws3``, and load  the reformated data into the ``ForestModel`` instance using standard methods. For example, ``ws3`` includes functions to import legacy Woodstock [#]_ model data (including LANDSCAPE, CONSTANTS, AREAS, YIELDS, LIFESPAN, ACTIONS, TRANSITIONS, and SCHEDULE section data), as well as functions to import and rasterize vector stand inventory data.

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

The next step in a typical workflow is to define one or more scenarios. Assuming that we are using an optimization approach to harvest scheduling, we need to define an objective function (e.g., maximize total harvest volume) and constraints (e.g., species-wise volume and area even-flow constraints, ending standing inventory constraints, periodic minimum late-seral-stage area constraints) [#]_, build the optimization model matrix, solve the model to optimality [#]_. 

.. [#] Woodstock software is part of `Remsoft Solution Suite <http://www.remsoft.com/forestry.php>`_. 

.. [#] ``ws3`` currently implements functions to formulate and solve *Model I* wood supply optimization problems---however, the package was deliberately designed to make it easy to transparently switch between *Model I* ,  *Model II* and *Model III* formulations without affecting the rest of the modelling workflow. ``ws3`` currently has placeholder function stubs for *Model II* and *Model III* formulations, which will be implemented in later versions as the need arises. For more information on wood supply model formulations, see Chapter 16 of the 
`Handbook of Operations Research in Natural Resources <http://www.springer.com/gp/book/9780387718149>`_.

.. [#] ``ws3`` currently uses the `Gurobi <http://www.gurobi.com/>`_ solver to solve the linear programming (LP) problems to optimality. We chose Gurobi because it is one of the top two solvers currently available (along with the `CPLEX <https://www.ibm.com/analytics/data-science/prescriptive-analytics/cplex-optimizer>`_ solver), has a simple and flexible policy for requesting unlimited licences for free use in research projects, has elegant Python bindings, and we like the technical documentation. However, we deliberately used a modular design, which allows us to transparently switch to a different solver in ``ws3`` without affecting the rest of the workflow---this design will make it easy to implement an interface to additional solvers in future releases.