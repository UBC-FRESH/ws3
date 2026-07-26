Technical Contracts
===================

This section is ws3's compact technical contract surface.

Use it when you need a fast answer about data formats, runtime prerequisites,
module boundaries, or agent-facing invariants.

This is intentionally **not** a separate agent-only documentation universe. The
pages here live in the same Sphinx tree as the narrative guides and API
reference, and they link back to those deeper pages instead of duplicating them
wholesale.

How to Use This Section
-----------------------

- Start here when you need the shortest source-of-truth answer for an
  operational seam.
- Follow the linked how-to guides when you need the full walkthrough or
  interpretation detail.
- Follow the linked API pages when the question is really about code ownership
  or callable behavior rather than workflow contract.

Contract Pages
--------------

.. toctree::
   :maxdepth: 1

   data_contracts
   runtime_invariants
   module_boundaries
   output_format_spec