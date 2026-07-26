.. _contract-data-contracts:

=================
Data Contracts
=================

This page defines the data formats that ws3 expects and produces.

Development Type Contract
-------------------------

Each development type must have:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - code
     - str
     - Unique identifier (e.g., "DT001")
   * - species
     - str
     - Species code (e.g., "SP", "HW")
   * - site_index
     - int
     - Site index class (e.g., 40, 50)
   * - age
     - int
     - Stand age in years
   * - area
     - float
     - Area in hectares

Example:

.. code-block:: python

   {
       "code": "DT001",
       "species": "SP",
       "site_index": 50,
       "age": 20,
       "area": 100.0
   }

Action Contract
---------------

Each action must have:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - code
     - str
     - Unique action identifier (e.g., "CLEARCUT")
   * - descr
     - str
     - Human-readable description
   * - components
     - list[str]
     - Yield components affected (e.g., ["volume"])
   * - transitions
     - dict[str, str]
     - Mapping from source DT code to target DT code

Example:

.. code-block:: python

   {
       "code": "CLEARCUT",
       "descr": "Clearcut harvest",
       "components": ["volume"],
       "transitions": {
           "DT001": "DT001_REGEN"
       }
   }

Growth Curve Contract
---------------------

Each growth curve must have:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - species
     - str
     - Species code
   * - site_index
     - int
     - Site index class
   * - ages
     - list[int]
     - Age values (ascending order)
   * - volumes
     - list[float]
     - Volume values (m3/ha)
   * - components
     - list[str]
     - Components included (e.g., ["volume", "basal_area"])

Example:

.. code-block:: python

   {
       "species": "SP",
       "site_index": 50,
       "ages": [10, 20, 30, 40, 50],
       "volumes": [25.0, 55.0, 95.0, 150.0, 220.0],
       "components": ["volume"]
   }

Schedule Output Contract
------------------------

Each schedule row must have:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Type
     - Description
   * - period
     - int
     - Planning period (0-indexed)
   * - development_type
     - str
     - Source development type code
   * - action
     - str
     - Action code
   * - area_ha
     - float
     - Area harvested (hectares)
   * - volume_m3
     - float
     - Volume harvested (cubic meters)

Example:

.. code-block:: python

   {
       "period": 0,
       "development_type": "DT001",
       "action": "CLEARCUT",
       "area_ha": 50.0,
       "volume_m3": 1250.0
   }