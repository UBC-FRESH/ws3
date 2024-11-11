**************************
About the :py:mod:`ws3` Package
**************************

The :py:mod:`ws3` package is implemented using the Python programming language. 
:py:mod:`ws3` is an aspatial wood supply model, which applies actions (and matching
transitions) to development types, simulates growth, and tracks inventory 
area (by development type and by age class) at each time step. Aspatial 
models output aspatial activity schedules---each line of the output schedule 
specifies the stratification variable values (which constitute a unique key 
into the list of development types), the age class, the time step, the action 
code, and the area treated.

Because the model is aspatial, the area treated on a given line of the output 
schedule may not be spatially contiguous (i.e., the area may be geographically 
dispersed throughout the landscape). Furthermore, in the common case where only 
a subset of available area in a given development type and age class combination
is treated in a given time step, an aspatial model provides no information 
regarding which spatial subset of available area is treated (and, conversely, 
not treated). 

Some applications (e.g., linking to spatially-explicit 
or highly--spatially-referenced models) require a spatially-explicit activity 
schedule. :py:mod:`ws3` includes a *spatial disturbance allocator* sub-module, which 
contains functions that can map aspatial multi-period action schedules onto a 
rasterized spatial representation of the forest. The spatial disturbance allocator 
is implemented using the :py:mod:`rasterio` package and is available as part of the 
:py:mod:`ws3.spatial` module. The area-based forest inventory input data for a :py:mod:`ws3`
model is typically compiled by *aggregating* stands in a spatially explicity 
forest inventory data layer (i.e., by combining spatially discontiguous stands
into area bins, grouping by stratum and age class). The spatial disturbance 
allocation functions are effectively *disaggregating* the aspatial disturbance
schedule, restoring the spatial dimensionality of the forest inventory data layer.
These same functions are also capable of *disaggregating* :py:mod:`ws3` output schedules 
from multi-year simulation periods into annual time steps (which is useful if you 
plan to pipe :py:mod:`ws3` output to a downstream spatially-explicit model that also
operates on a smaller simulation time step than your source :py:mod:`ws3` model). 
Note that this disaggregation operation is always going to be somewhat 
problematic, because we are essentially fabricating details that are not 
represented in the source data.

:py:mod:`ws3` uses a scripted Python interface to control the model, which provides maximum 
flexibility and makes it easy to automate modelling workflows. This ensures 
reproducible methodologies, and enables linking :py:mod:`ws3` input or outout to other 
software packages to form complex modelling pipelines. The scripted interface also 
makes it relatively easy to implement custom data-importing functions, which makes 
it easier to import existing data from a variety of ad-hoc sources without the 
need to recompile the data into a standard :py:mod:`ws3`-specific format (i.e., importing 
functions can be implemented such that the conversion process is fully automated 
and applied to raw input data *on the fly*). Similarly, users can easily implement 
custom functions to re-format :py:mod:`ws3` output data *on the fly* (either for static 
serialization to disk, or to be piped live into another process). 

Although we recommend using Jupyter Notebooks as an interactive interface to :py:mod:`ws3` 
(the package was specifically designed with an interactive notebook interface in mind), 
:py:mod:`ws3` functions can also be imported and run in fully scripted workflow 
(e.g., non-interactive batch processes can be run in a massively-paralleled workflow 
on high-performance--computing resources, if available). The ability to mix interactive 
and massively-paralleled non-interactive workflows is a unique feature of :py:mod:`ws3`.

:py:mod:`ws3` is a complex and flexible collection of functional software units. The following 
sections describe some of the main classes and functions in the package, and describe 
some common use cases, and link to sample notebooks that implement these use cases.

Overview of Main Classes and Functions
=========================

This section describes some of the main classes and functions that make up the ``ws3`` 
pacakge.

The :py:class:`ws3.forest.ForestModel` class is the core class in the package. This class 
encapsulates all the information used to simulate scenarios from a given dataset 
(i.e., stratified initial inventory, growth and yield functions, action eligibility, 
transition matrix, action schedule, etc.), as well as a collection of functions 
to import and export data, generate activity schedules, and simulate application of 
these schedules (i.e., run scenarios).

At the heart of the :py:class:`~ws3.forest.ForestModel` class is a list of 
:py:class:`ws3.forest.DevelopmentType` instances. Each 
:py:class:`~ws3.forest.DevelopmentType` instance encapsulates information about one 
development type (i.e., a forest stratum, which is an aggregate of smaller *stands* 
that make up the raw forest inventory input data). The :py:class:`~ws3.forest.DevelopmentType` 
class also stores a list of operable *actions*, maps *state variable transitions* 
to these actions, stores growth and yield functions, and knows how to *grow itself* 
when time is incremented during a simulation.

.. To Do: Finish documenting main stuff here.
 
Common Use Case and Sample Notebooks
===========================

In this section, we assume an interactive Jupyter Notebook environment is used to 
interface with :py:mod:`ws3`.

A typical use case starts with creating an instance of the :py:class:`~ws3.forest.ForestModel` 
class. Then, we need to load data into this instance, define one or more scenarios 
(using a mix of heuristic and optimization approaches), run the scenarios, and 
export output data to a format suitable for analysis (or link to the next model in 
a larger modelling pipeline).

The first step in a typical workflow is to run a mix of standard :py:mod:`ws3` and custom 
data-importing functions.  These functions import data from various sources, *on-the-fly* 
reformat this data to be compatible with :py:mod:`ws3`, and load  the reformated data into 
the :py:class:`~ws3.forest.ForestModel` instance using standard methods. For example, 
:py:mod:`ws3` includes functions to import legacy Woodstock [#]_ model data (including 
``LANDSCAPE``, ``CONSTANTS``, ``AREAS``, ``YIELDS``, ``LIFESPAN``, ``ACTIONS``, 
``TRANSITIONS``, and ``SCHEDULE`` section data), as well as functions to import and 
rasterize vector stand inventory data.

For example, one might define the following custom Python function in a Jupyter Notebook, 
to import data formatted for Woodstock.::
	
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

The next step in a typical workflow is to define one or more scenarios. Assuming 
that we are using an optimization approach to harvest scheduling, we need to 
define an objective function (e.g., maximize total harvest volume) and constraints 
(e.g., species-wise volume and area even-flow constraints, ending standing 
inventory constraints, periodic minimum late-seral-stage area constraints) 
[#]_, build the optimization model matrix, solve the model to optimality [#]_. 

.. [#] *Woodstock* is part of suite of 
`Forestry Planning and Scheduling <https://remsoft.com/forestry-planning-and-scheduling/>`_
software tools offered by Remsoft. 

.. [#] :py:mod:`ws3` currently implements functions to formulate and solve *Model I* 
wood supply optimization problems---however, the package was deliberately designed 
to make it easy to transparently switch between *Model I* ,  *Model II* and *Model III* 
formulations without affecting the rest of the modelling workflow. ``ws3`` currently 
has placeholder function stubs for *Model II* and *Model III* formulations, which will 
be implemented in later versions as the need arises. For more information on wood 
supply model formulations, see Chapter 16 of the 
`Handbook of Operations Research in Natural Resources <http://www.springer.com/gp/book/9780387718149>`_.

.. [#] :py:mod:`ws3` currently defaults to the open source `PuLP <https://coin-or.github.io/pulp/>`_ solver
to solve linear programs, and also includes `Gurobi <http://www.gurobi.com/>`_ solver bindings. 
:py:mod:`ws3` may eventually provide a pluggable interface to other solvers (e.g., CPLEX, GLPK, etc.).

