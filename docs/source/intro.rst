****************************
Overview
****************************

Introduction
================

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

The Forest Resources Environmental Services Hub (FRESH) is a research group at the University of British Columbia (UBC), Canada, that focuses on sustainable forest management including wood supply analysis, optimization research and supply chain analysis. 

One of the primary focuses of the research group is to develop open sourced information and software packages that increase the accessibility of higher level forest modelling including tools used in WSPP and WSM. The ``ws3`` package, this user guide and the API all aim to help meet this goal. 

Contact information
==========================

If you have any questions about this guide or want to ask questions about ``ws3``, WSM or WSPP please reach out

|		**Address**
|			2430 - 2424 Main Mall
|			Vancouver B.C.
|			V6T 1Z4
|			Canada
	
|		**Email**
|			kathleen.coupland@ubc.ca
|			gregory.paradis@ubc.ca
	
|		**Phone**
|			(604) 827 - 0845
	
 
