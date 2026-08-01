Woodstock Format: What ws3 Reads
================================

ws3 imports an essential subset of the Woodstock model input data format. The
boundary of that subset used to be invisible: keywords outside it are not
rejected, they are *ignored*. A dataset can declare an ``OPTIMIZE`` section, or
use ``*ACTIONSERIES``, and import without complaint — producing a model that is
quietly not the model that was written. No error, wrong answer.

This page is generated from the contract shipped with the package
(:py:mod:`ws3.woodstock`), so it cannot drift from what the importers actually
do.

.. woodstock-coverage::

The format is open ended by design
----------------------------------

A Woodstock model instance declares its own themes, in its own order, with its
own stratification variable codes within each theme. There is no fixed
allocation of meaning to theme positions, and no theme count that is standard.

The ``LANDSCAPE`` section is the authoritative source for all of it: the
cardinality and order of the themes in the theme vector, the values each theme
is allowed to take, the ``*AGGREGATE`` theme values, and the constants. The
number of themes declared there determines the length of every development-type
key in the model.

Datasets that look structurally alike usually do so because one author reused
their own conventions, not because the format imposes anything. Do not infer
theme semantics from a sample of models.

``*THEME`` declaration lines carry a descriptive name after the keyword, and
that description is the only statement in a dataset of what a theme position
means. :py:meth:`ws3.forest.ForestModel.import_landscape_section` preserves it
as ``__description__`` on each entry of ``ForestModel._themes``; where a dataset
omits it, ws3 has nothing to go on and says so rather than inventing a meaning.
The generated theme names (``theme0``, ``theme1``, ...) are positional labels
and carry no meaning of their own.

Sections
--------

``stub`` is called out separately from ``no importer`` because the failure mode
differs in a way that matters. A stub method exists and returns successfully, so
a caller has every reason to believe the section was read. It was not.

.. woodstock-sections::

Keywords ws3 reads
------------------

Every other catalogued keyword is ignored on import.

.. woodstock-keywords::

Deliberate divergences from Woodstock
-------------------------------------

These are recorded in the contract rather than treated as defects.

.. woodstock-divergences::

Linting a dataset
-----------------

:py:func:`ws3.woodstock.lint_dataset` reports what ws3 will not read from a
dataset. It reads the section files directly: nothing is imported, no model is
built, and nothing is modified. It is advisory, and it is not required in order
to import a model — but running it before you trust an import is cheap.

.. code-block:: python

   from ws3.woodstock import lint_dataset, format_findings

   findings = lint_dataset('examples/data/woodstock_model_files_tsa24_clipped',
                           'tsa24_clipped')
   print(format_findings(findings))

Run against the ``tsa24_clipped`` dataset shipped with the examples, that
reports:

.. code-block:: text

   error: .../tsa24_clipped.lif: the LIFESPAN section is present but not imported:
     import_lifespan_section is a stub and imports nothing. Everything in this
     file is ignored.
   error: .../tsa24_clipped.opt: the OPTIMIZE section is present but not imported:
     import_optimize_section is a stub and imports nothing. Everything in this
     file is ignored.
   error: .../tsa24_clipped.que: the QUEUE section is present but not imported:
     ws3 has no importer for it. Everything in this file is ignored.
   error: .../tsa24_clipped.rep: the REPORTS section is present but not imported:
     ws3 has no importer for it. Everything in this file is ignored.
   error: .../tsa24_clipped.run: the CONTROL section is present but not imported:
     import_control_section is a stub and imports nothing. Everything in this
     file is ignored.

   5 section(s) not imported, 0 keyword(s) ignored.

Five files in that dataset are read by nobody. That is not a defect report — the
model built from it has carried real work — but it is the difference between
knowing that and assuming otherwise.

Findings are ordered by severity, then file, then line. Each
:py:class:`ws3.woodstock.Finding` carries the severity, section, path, line and
keyword separately, so the report can be rendered any way you like:

.. code-block:: python

   errors = [f for f in findings if f.severity == 'error']
   for f in errors:
       print(f.section, f.path)

To check only part of a dataset, pass the section identifiers:

.. code-block:: python

   findings = lint_dataset(path, name, sections_to_check=['Actions', 'Yields'])

An empty result means ws3 imports everything present in that dataset.

API
---

.. automodule:: ws3.woodstock
   :members:
   :undoc-members:
   :show-inheritance:
