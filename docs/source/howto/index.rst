.. _howto:

=================
How-To Guides
=================

Step-by-step guides for common modelling tasks in ws3. These pages assume you
are already familiar with the basics covered in the Getting Started section
and the textbook.

How-To Guides
=============

Operational guides for common ws3 tasks. Each guide is self-contained and
includes runnable code examples.

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   custom_actions
   transition_rules
   linking_libcbm
   linking_spades
   optimization_integration
   batch_scenarios
   output_postprocessing
   custom_growth_curves
   multi_region_models
   reproducibility_workflows

What You Will Find Here
========================

Each how-to guide follows a consistent format:

* **Goal** — what you will accomplish by the end of the guide.
* **Prerequisites** — what you need to know or have set up first.
* **Step-by-Step Instructions** — numbered steps with code examples.
* **Expected Output** — what the results should look like.
* **Troubleshooting** — common pitfalls and how to fix them.

Common Tasks Covered
====================

* Defining custom forest management actions and transition rules.
* Linking ws3 to libCMB for carbon accounting.
* Linking ws3 to SpaDES for spatially-explicit disturbance modelling.
* Integrating ws3 with optimization solvers for automatic schedule generation.
* Running and comparing multiple scenarios in batch.
* Post-processing simulation output for reporting and visualization.
* Adding custom growth-and-yield curves.
* Building multi-region models that share data across landscapes.
* Setting up reproducible workflows with version-controlled inputs and
  automated validation.

Prerequisites
=============

Before tackling these guides, you should have:

* Completed the :doc:`../getting_started/index` section.
* Read at least the first four chapters of the :doc:`../textbook/index`.
* A working ws3 installation with sample data available.