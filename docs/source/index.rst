=====================================
Wood Supply Simulation System (ws3)
=====================================

Welcome to the **Wood Supply Simulation System** (:py:mod:`ws3`) documentation.
ws3 is an open-source Python package for landscape-level wood supply simulation
and forest estate modelling.

Who Is This Documentation For?
==============================

This documentation serves three audiences:

* **New users** — forest planners, students, and land managers who want to run
  their first wood supply simulation. Start with :doc:`getting_started/index`.
* **Advanced users** — researchers and analysts building custom scenarios,
  linking ws3 to other tools, or extending the API. See :doc:`howto/index` and
  :doc:`guides/index`.
* **LLM coding agents** — AI assistants that read and edit this repository.
  Follow :doc:`guides/coding-agent-onboarding` and the :doc:`reference/contracts/index`
  to understand the data contracts, runtime invariants, and repository
  conventions the agent must respect.

The ws3 documentation also doubles as an **introduction to forest estate
modelling** for students. The :doc:`textbook/index` section teaches the
underlying concepts — from forest inventory through growth-and-yield projection
to multi-period planning — using ws3 as the working example.

Quick Navigation
================

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Section
     - For Whom
     - What You Will Learn
   * - :doc:`getting_started/index`
     - New users
     - Installation, first run, data preparation, and running your first scenario.
   * - :doc:`textbook/index`
     - Students and newcomers
     - Forest estate modelling concepts: inventory, growth, actions, transitions, and planning horizons.
   * - :doc:`howto/index`
     - Advanced users
     - Step-by-step guides for common modelling tasks.
   * - :doc:`guides/index`
     - Power users and developers
     - Deep dives into architecture, integration, and advanced workflows.
   * - :doc:`reference/index`
     - Everyone
     - API reference, data contracts, and technical specifications.

Old Documentation
-----------------

The legacy flat-chapter documentation is still available for reference:

.. toctree::
  :maxdepth: 2
  :caption: Legacy Chapters

  intro
  Chapt1
  Chapt2
  aboutws3
  examples
  appendices
  modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
