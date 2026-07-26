.. _textbook:

=====================
The ws3 Textbook
=====================

This section teaches the concepts behind forest estate modelling, using ws3
as the working example. It is designed for students, new users, and anyone
who wants a structured introduction to the theory and practice of wood supply
simulation.

The chapters build on each other in a fixed prerequisite chain. Each chapter
ends with a summary and exercises to reinforce the material.

Who Is This For?
================

* **University students** studying forest resource management, sustainable
  forest management, or forest economics.
* **New ws3 users** who want to understand *why* the model works the way it
  does, not just *how* to run it.
* **Researchers** from adjacent fields (ecology, economics, geography) who
  need a grounding in wood supply modelling concepts.

How to Use This Textbook
========================

Read the chapters in order. Each chapter assumes you have completed the
previous one. The prerequisite chain is:

.. mermaid::

   graph LR
     A[ch01: Forest Estate Models] --> B[ch02: Forest Inventory]
     B --> C[ch03: Growth & Yield]
     C --> D[ch04: Actions & Transitions]
     D --> E[ch05: Multi-Period Planning]
     E --> F[ch06: Scenario Analysis]
     F --> G[ch07: Carbon Accounting]
     G --> H[ch08: Spatial Extension]

Chapter Listing
===============

Textbook: Introduction to Forest Estate Modelling
==================================================

This textbook introduces the concepts, methods, and computational tools used
in forest estate modelling — the discipline that :py:mod:`ws3` serves. Each
chapter builds on the previous ones, with learning objectives, worked
:py:mod:`ws3` examples, and exercises at the end.

The chapters are designed for:

- **Students** learning forest resource analysis and wood supply modelling.
- **Practitioners** who want a structured refresher on core concepts.
- **Developers** who need domain context before extending :py:mod:`ws3`.

Prerequisite Chain
------------------

.. mermaid::

   graph LR
     CH01["ch01: Forest Estate Models"] --> CH02["ch02: Forest Inventory"]
     CH01 --> CH03["ch03: Growth & Yield"]
     CH02 --> CH04["ch04: Actions & Transitions"]
     CH03 --> CH04
     CH04 --> CH05["ch05: Optimization"]
     CH04 --> CH06["ch06: Spatial Allocation"]
     CH05 --> CH07["ch07: Financial Analysis"]
     CH06 --> CH07
     CH07 --> CH08["ch08: Uncertainty & Risk"]
     CH08 --> CH09["ch09: Advanced Topics"]
     CH07 --> CH10["ch10: Carbon Modelling"]
     CH09 --> CH10

.. note::

   Chapters 02 and 03 can be studied in parallel after Chapter 01.
   Chapters 05 and 06 can also be studied in parallel after Chapter 04.
   Chapter 10 (Carbon Modelling) can be studied after Chapter 07 (Financial Analysis).

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Chapters

   ch01_forest_estate_models
   ch02_forest_inventory
   ch03_growth_and_yield
   ch04_actions_and_transitions
   ch05_optimization
   ch06_spatial_allocation
   ch07_financial_analysis
   ch08_uncertainty_and_risk
   ch09_advanced_topics
   ch10_carbon_modelling

Chapter Structure
=================

Every chapter follows the same structure:

* **Learning Objectives** — what you will be able to do after reading.
* **Key Concepts** — the core ideas, with definitions and diagrams.
* **How It Works in ws3** — how the concept maps to the software.
* **Worked Example** — a step-by-step walkthrough with ws3 code.
* **Summary** — a concise recap of the main points.
* **Exercises** — problems to test your understanding.

Prerequisites for This Textbook
===============================

You should be comfortable with:

* Basic algebra and statistics (means, standard deviations, simple equations).
* Reading Python code (you do not need to be an expert programmer).
* A general interest in how forests are managed over time.

No prior experience with forest modelling or wood supply simulation is
required. The textbook builds from first principles.