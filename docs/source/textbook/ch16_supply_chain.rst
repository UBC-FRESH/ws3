Chapter 16: Value-Creation and Forest Supply Chain Modelling
============================================================

Learning Objectives
-------------------

After reading this chapter, you should be able to:

- Explain why long-term AAC models and short-term industrial behavior can diverge
- Describe the two-phase workflow used to link ws3-style harvest schedules to supply-chain optimization
- Define value-creation potential (VCP) and its six major components
- Interpret a multi-layer network flow model for forest fibre allocation
- Design policy and capacity scenarios to test how much AAC is likely to be consumed

Why This Chapter Exists
-----------------------

Classical wood supply planning focuses on sustained biophysical harvest levels. In many
jurisdictions, this process estimates annual allowable cut (AAC) by species group while
omitting financial performance at the stand-to-mill system level.

The supply-chain question is different:

- Which part of the available volume is profitable to consume?
- Which mills can process that fibre?
- How do transport, stumpage, product markets, and policy incentives alter behavior?

The four CIRRELT studies in ``tmp/`` form a coherent arc:

1. Build statistically robust diameter distribution models.
2. Compile disaggregation coefficients that link long-term and short-term planning units.
3. Retrofit VCP indicators into long-term model outputs.
4. Run a hybrid simulation-optimization network to estimate industrial consumption behavior.

This chapter synthesizes that arc into a practical ws3-oriented workflow.

Conceptual Gap: AAC vs. Consumed Fibre
--------------------------------------

AAC is an upper bound from a strategic planning model. It is not a guarantee that industry
will consume all offered fibre.

If marginal value is negative, a profit-maximizing network avoids that fibre unless forced by
contracts, regulation, or subsidies.

.. mermaid::

   graph LR
     WS["Long-term supply model<br/>AAC by species group"] --> DG["Disaggregation + valuation"]
     DG --> NF["Network flow model<br/>mill behavior"]
     NF --> OUT["Consumed volume + total VCP"]

The key result is a behavioral filter:

.. math::

   	ext{Consumed AAC} \subseteq \text{Available AAC}

Phase 1: Build Compatible Inputs
--------------------------------

The first phase transforms aggregated long-term outputs into data usable by a supply-chain model.

Step 1: Compile Volume Disaggregation Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2017 method links each harvested unit from the upper model to diameter-class fibre baskets.

For each species group :math:`s`, cover type :math:`c`, and treatment type :math:`t`, harvested
volume is distributed over stem size class :math:`x` using:

.. math::

   v_{cst}(x) = u_{cst} \cdot p_{cst}(x)

where:

- :math:`u_{cst}` is aggregate harvested volume from the long-term model
- :math:`p_{cst}(x)` is a probability vector over DBH size classes

In the Quebec workflow summarized by the papers, :math:`p_{cst}(x)` is assembled from:

- standing inventory diameter distribution
- harvest-selection probability by size class
- stem form factors

This bridges coarse model output and fine-grained valuation tables.

Step 2: Retrofit Value-Creation Indicators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The retrofit step maps disaggregated fibre to value data and then aggregates back to planning units.

The six VCP components used in the CIRRELT workflow are:

.. list-table:: VCP Components
   :header-rows: 1
   :widths: 25 25 50

   * - Component
     - Typical Basis
     - Interpretation
   * - Fixed procurement cost
     - $/ha
     - Administrative and access-related fixed overheads
   * - Harvest cost
     - $/m3
     - Extraction, handling, and roadside operations
   * - Silviculture credit
     - $/ha
     - Policy credit offsetting silviculture obligations
   * - Stumpage cost
     - $/m3
     - Public timber fee by zone and commodity
   * - Transport cost
     - $/m3
     - Fibre movement from forest to processors
   * - Product value net of processing and delivery
     - $/m3 equivalent
     - Revenue-side contribution of resulting products

Net value-creation potential at decision level is then:

.. math::

   	ext{VCP}_{\text{net}} = \text{Product Value} - (\text{Fixed} + \text{Harvest} + \text{Silv} + \text{Stumpage} + \text{Transport})

This gives each harvest decision an economic signal that can drive downstream optimization.

Phase 2: Network Flow Optimization
----------------------------------

After retrofit, we solve a network flow model that emulates profit-maximizing industrial consumption.

Network Topology
~~~~~~~~~~~~~~~~

The hybrid model in the 2018 study uses five node layers:

- Source nodes: harvest decisions from the schedule
- Dispatch nodes: area-to-volume conversion and commodity split
- Commodity nodes: species/product aggregates
- Processor nodes: sawmills, pulpmills, and capacities
- Sink nodes: final accounting of processed flow

.. mermaid::

   graph LR
     SRC["Source\n(harvest areas)"] --> DSP["Dispatch\n(area -> commodities)"]
     DSP --> CDT["Commodity"]
     CDT --> PRC["Processors"]
     PRC --> SNK["Sink"]
     PRC --> PRC2["Secondary chip flows\n(optional)"]

Decision Variables and Objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core decision is flow on arcs linking these node sets. The main objective in the
published formulation is to maximize total network value (often represented through source
flow coefficients already carrying net VCP).

High-level form:

.. math::

   \max \sum_{a \in A} c_a f_a

subject to:

- lower and upper flow bounds on arcs
- flow conservation at each internal node
- optional harvested-area bounds for scenario sweeps

In practice, repeated solves with varying harvested-area constraints produce a curve of total
VCP versus proportion of available AAC consumed.

Interpreting the Scenario Curves
--------------------------------

The 2018 case study (UA 064-51) demonstrates a recurring pattern:

- Base case can be negative over most or all harvested proportions.
- Commodity-targeted subsidies can move some value curves above zero.
- Combined incentives can increase the harvested share a profit-maximizing network will accept.

Conceptually:

- If marginal VCP declines as harvest proportion increases, the model harvests "best" stands first.
- Flat peaks indicate sensitivity: small policy changes can shift consumed volume materially.

This does not imply that subsidies are always desirable; it shows the model can evaluate
system response to policy and market assumptions.

How This Maps to ws3 Workflows
------------------------------

Within ws3-oriented pipelines, the chapter's main implementation message is:

1. Keep long-term planning and industrial simulation as separate modules.
2. Use explicit translation layers for data aggregation mismatches.
3. Attach economics to planning decisions before solving network allocation.
4. Run scenario ensembles, not just a single "best" solve.

Minimal pseudo-workflow:

.. code-block:: python

   # 1) Load solved long-term model schedule
   schedule = load_harvest_schedule()

   # 2) Disaggregate aggregate volumes to size/species baskets
   disagg = apply_disaggregation_coefficients(schedule)

   # 3) Attach value components and aggregate to decision-level net VCP
   vcp = compile_value_components(disagg, costs_db, product_db)

   # 4) Build and solve network model for a scenario
   model = build_network_flow_model(vcp, capacities, licences, transport)
   result = model.solve(harvest_area_ratio=0.60)

   # 5) Repeat across policy/capacity/price scenarios
   frontier = run_scenario_sweep(model, ratios=[i/20 for i in range(21)])

Limitations and Modelling Risks
-------------------------------

Important caveats from the paper sequence still apply:

- Input uncertainty: disaggregation and cost functions introduce estimation error.
- Aggregation lock-in: topology chosen for data convenience may reduce traceability.
- Policy dependence: results can shift substantially with stumpage, subsidy, and price assumptions.
- Validation burden: VCP components should be calibrated with independent evidence where possible.

A practical rule is to treat point estimates as scenario outcomes, not immutable forecasts.

Worked Interpretation Exercise
------------------------------

Suppose a scenario sweep shows:

- total VCP peaks at 55% harvested area,
- then declines as additional area is forced into the system.

Interpretation:

1. The first 55% of available fibre contains most of the positive or least-negative margin.
2. Remaining fibre likely has weaker product mix, higher cost, or weaker market match.
3. Policy options to test next include capacity changes, transport improvements, and commodity-targeted incentives.

Exercises
---------

1. Easy: Define a table schema that maps each harvest decision to cover type, treatment class,
   species group volumes, and net VCP.
2. Medium: Write a short note explaining how you would validate disaggregation vectors before
   using them in policy analysis.
3. Hard: Formulate an alternative objective that maximizes consumed volume subject to a
   minimum acceptable total VCP, and discuss when this objective is preferable.

Further Reading
---------------

- :doc:`ch05_optimization` for optimization fundamentals used by both planning and network models
- :doc:`ch07_financial_analysis` for economic concepts used in value components
- :doc:`ch11_femic_models` and :doc:`ch12_fhops_integration` for ecosystem-level integration context
- CIRRELT technical reports in ``tmp/`` used for this synthesis:

  - CIRRELT-2017-34 (diameter distribution modelling)
  - CIRRELT-2017-43 (disaggregation coefficients)
  - CIRRELT-2018-23 (retrofit VCP indicators)
  - CIRRELT-2018-24 (hybrid simulation-optimization)
- Integrated forest planning research