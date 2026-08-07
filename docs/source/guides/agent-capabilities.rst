.. _agent-capabilities:

===================
Agent Capabilities
===================

``ws3`` ships a small set of operations designed to be driven by an AI coding
agent, where **the output is validated against real model state before it is
returned**.

This guide covers what they are, how to configure them, what is recorded, and how
to add one.

.. note::

   Entirely optional. Core ``ws3`` modelling never needs this, and ``import ws3``
   never loads it.

The problem this solves
=======================

The usual way an agent operates a library is: read the documentation, write
Python, hope it composed the API correctly. The agent is *outside* the package,
guessing at it. Every call is an unbounded generation problem whose only validator
is "did it crash."

Agent capabilities invert that. ``ws3`` owns the prompt, the endpoint, and --
crucially -- validation of the model's output against its own state.

The reliability does not come from embedding an LLM. It comes from a component
*inside* the package being able to cheaply check the answer before returning it.

.. code-block:: text

   build prompt → call model → parse → validate against real ws3 state
                      ↑                            │
                      └──── feed failure back ─────┘   (bounded retries)

Output that fails validation **never reaches the caller**. On exhaustion the
capability returns ``ok=False`` with the accumulated reasons -- never a best guess.

Available capabilities
======================

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Capability
     - What it does
     - What it validates
   * - ``build_mask``
     - Builds a development-type mask from a natural-language description
     - The mask resolves against the :py:class:`~ws3.forest.ForestModel` to at
       least one development type
   * - ``explain_exception``
     - Explains an error in plain language with suggested next actions
     - Every ``ws3`` symbol the explanation cites actually exists in the
       installed package
   * - ``diagnose_import``
     - Diagnoses a failing Woodstock import and proposes a corrected line
     - The fix is applied to a scratch copy and the section genuinely re-imports
   * - ``rtfm``
     - Routes a user goal or error message to the correct capability
     - The returned capability name is real; cited doc URLs return HTTP 200
   * - ``ws3_hint``
     - Gives general modelling guidance with verifiable symbol and URL references
     - Every cited ``ws3`` symbol exists in the installed package; every cited
       doc URL returns HTTP 200
   * - ``inspect_model``
     - Shows a read-only metadata snapshot of the live ForestModel
     - Every reported field is read directly from the live model; numeric
       values are computed by the deterministic executor, never by the provider
   * - ``report_scenario_inventory_products``
     - Replays a selected model schedule and reports inventory and products by period
     - The model and sibling schedule paths are validated; values come directly
       from live ``inventory`` and ``compile_product`` calls

Each rejection is specific. ``build_mask`` names the theme codes that were not
found; ``explain_exception`` names the symbols that do not exist;
``diagnose_import`` reports that the section still fails.

Installation
============

.. code-block:: bash

   pip install ws3[agent]        # capabilities
   pip install ws3[agent-mcp]    # capabilities plus the MCP server

Configuration
=============

Resolution order, first hit wins:

1. an explicit ``AgentConfig`` passed by the caller
2. environment variables
3. ``~/.config/fresh-agent/config.toml``
4. otherwise **unavailable** -- ``available()`` returns ``False``

.. code-block:: bash

   export FRESH_AGENT_ENDPOINT="https://your-host/v1"
   export FRESH_AGENT_MODEL="your-model-id"
   export FRESH_AGENT_API_KEY="..."                    # optional
   export FRESH_AGENT_HEADERS='{"X-Trace": "abc"}'     # optional, JSON

Or:

.. code-block:: toml

   # ~/.config/fresh-agent/config.toml
   [agent]
   endpoint = "https://your-host/v1"
   model = "your-model-id"
   timeout = 60.0

Nothing about any particular endpoint is hardcoded, and credentials are read from
the environment or user config only -- never from a repository.

Usage
=====

.. code-block:: python

   import ws3.agent
   from ws3.forest import ForestModel

   fm = ForestModel(model_name='my_model', model_path='path/to/model',
                    base_year=2020)
   fm.import_landscape_section()
   fm.import_areas_section()

   if not ws3.agent.available():
       raise SystemExit('no agent endpoint configured')

   result = ws3.agent.run('build_mask', 'mature spruce stands', context=fm)

   if result.ok:
       print('validated mask:', result.value)
   else:
       print('no valid mask after', result.attempts, 'attempts')
       for reason in result.errors:
           print('  -', reason)

``available()`` never raises and never touches the network. It answers *"is this
configured"*, not *"is the endpoint reachable"* -- reachability is only knowable by
making a call, and the probe has to be cheap enough to sit inside an ``if``.

Use ``core_installed()`` to distinguish *not installed* from *installed but
unconfigured*; they have different fixes.

IPython and Jupyter line magics
===============================

Load the extension after creating a :py:class:`~ws3.forest.ForestModel` named
``fm`` in the notebook namespace:

.. code-block:: ipython

  %load_ext ws3.agent.ipython_magics
  %ws3_hint How do I add a fire disturbance?

Questions may be written naturally. They do not need quotes, and a trailing
question mark is supported.

Capability results are displayed as rendered Markdown with separate answer,
steps, validated values, fixes, and references sections. The magic displays the
result and returns ``None`` so Jupyter does not wrap the response in a quoted
``Out[...]`` string with escaped newlines. Scripts that need structured data
should use the Python capability API instead of parsing notebook display output.

IPython normally treats a terminal ``?`` as object-help syntax. Without the
``ws3`` extension's input cleanup transform, the example above is rewritten
*before the magic runs* as a request for help on the final word, equivalent to
``%pinfo disturbance``. The resulting ``Object `disturbance` not found.``
message is therefore an IPython parsing failure, not an agent or provider
response.

If that message appears after upgrading or editing ``ws3``, reload the extension
in the active kernel and rerun the original unquoted command:

.. code-block:: ipython

  %reload_ext ws3.agent.ipython_magics
  %ws3_hint How do I add a fire disturbance?

Restarting the kernel is an alternative when module state may be stale.

``%ws3_inspect_model`` — read-only metadata snapshot
---------------------------------------------------

.. code-block:: ipython

  %ws3_inspect_model
  %ws3_inspect_model full snapshot

Displays a Markdown table of the live :py:class:`~ws3.forest.ForestModel`'s
metadata without modifying the model in any way. Supported fields are:

- **identity** — ``model_name`` and ``name``
- **periods** — ``base_year``, ``horizon`` (count of periods), ``period_length``,
  and the list of period integers
- **counts** — number of themes, actions, and development types
- **total area** — unambiguous sum of ``dt.area(1)`` across all development types
  at period 1 (the base period only); ``None`` is reported rather than fabricated
  when the sum cannot be computed

Arbitrary operable-area filters, time-series plots beyond the base period, or any
mutation are **not** executed. The capability returns an explicit unsupported
result instead of producing plausible-looking numbers.

Deterministic selection rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The magic enumerates every ``ForestModel`` in the IPython namespace and applies
these rules:

1. **Exactly one model** — inspect it directly; no user disambiguation needed.
2. **Multiple models, no identifier in the query** — list every candidate with
   its variable name and metadata, refuse to pick, and suggest specifying the
   full variable name, ``model_name``, or public ``name``.
3. **Multiple models, explicit identifier** — the query token is matched as a
   complete identifier (not a substring) against each candidate's variable name,
   ``model_name``, and ``name``. If more than one candidate matches the query is
   flagged as ambiguous; if none match the candidates are listed and the user is
   prompted to be more specific.
4. **No model** — display a short actionable message pointing at how to create
   one.

The capability never fabricates numeric values and never invokes model-generated
Python. It displays Markdown and returns ``None`` so Jupyter does not wrap the
output in a quoted ``Out[...]`` string.

Scenario inventory/products report
----------------------------------

``ws3.agent.report_scenario_inventory_products`` is a deterministic, offline
field-test entry point for a bundled WS3 model and its sibling ``.seq`` schedule:

.. code-block:: python

   from pathlib import Path
   import ws3.agent

   result = ws3.agent.report_scenario_inventory_products(
     Path('examples/data/woodstock_model_files_tsa24_clipped'),
     'tsa24_clipped',
   )
   assert result.ok
   print(result.initial_area)
   for row in result.rows:
     print(row.period, row.harvested_area, row.harvested_volume,
       row.standing_volume)

The workflow imports the model sections into a newly constructed in-memory
``ForestModel``, reads the initial area with ``ForestModel.inventory(0)``,
applies only the selected model's sibling schedule, and reports each period with
the exact calls ``compile_product(period, '1.', acode='harvest')``,
``compile_product(period, 'totvol', acode='harvest')``, and
``inventory(period, 'totvol')``. It does not accept a mask or provider-generated
actions. Schedule application can change the fresh in-memory model, but source
model files are hashed before and after the run and the result states explicitly
whether they remained unchanged.

The direct Python entry point makes no provider call and needs no credentials.
The same operation is advertised as the
``report_scenario_inventory_products`` MCP tool. The current shared MCP server
still has its existing provider configuration requirement at server startup, but
this tool ignores the provider and performs all computation on the host. A small
runnable example is available at ``examples/agent_scenario_report.py``.

What the guarantee is, and what it is not
=========================================

**It is**: a capability returns validated output or it returns nothing.

**It is not**: a promise the answer is the one you wanted. ``build_mask`` returns a
mask that matches at least one development type -- not necessarily the stands you
had in mind. The oracle rules out the *demonstrably wrong*, which is a smaller
claim than being right, and a much larger one than being plausible.

``result.ok is False`` means every attempt was rejected. That is information, not
an error to route around: ``result.errors`` says what the model kept getting wrong.

Provider-backed capabilities are **advisory**. They return proposals; applying
them is your decision. The deterministic scenario report is the bounded
exception: it applies a source schedule only to a newly loaded in-memory model,
never to a caller's model object or source model files.

MCP server
==========

.. code-block:: bash

   ws3-agent-mcp --model-path path/to/model --model-name my_model

The model path matters. ``build_mask`` validates by resolving proposals against a
real model, so without one it will correctly -- but uselessly -- reject everything.
Some context cannot travel over the wire, which is why the server is constructed
with access to it rather than receiving it per call.

Inspect the tool descriptors without starting a server:

.. code-block:: bash

   ws3-agent-mcp --list-tools

Register with an MCP client:

.. code-block:: json

   {
     "mcpServers": {
       "ws3": {
         "command": "ws3-agent-mcp",
         "args": [
           "--model-path", "/path/to/model",
           "--model-name", "my_model"
         ]
       }
     }
   }

Exposing capabilities as tools rather than documenting them as conventions is
deliberate. Instructions get ignored; tools in the tool list get called.

Worked example
==============

A complete, runnable example is at ``examples/agent_capability_example.py``.
It loads a real ForestModel, calls three capabilities with FakeProvider (so it
runs offline with no credentials), and shows both validated and rejected outputs:

.. code-block:: bash

   python examples/agent_capability_example.py

The same pattern works with a live endpoint by omitting the ``provider=``
argument so ``ws3.agent`` resolves configuration from the environment or
``~/.config/fresh-agent/config.toml``.

Provenance
==========

Every attempt is recorded, including the ones that failed validation. A
nondeterministic component in a scientific pipeline without an audit trail is not
defensible, so the log *is* the evidence.

Records are JSON Lines, at ``$FRESH_AGENT_LOG`` or
``./.fresh-agent/provenance.jsonl``:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Field
     - Notes
   * - ``capability``
     - Which capability ran
   * - ``model``
     - Model identifier from configuration
   * - ``endpoint_host``
     - Host only -- never the full URL, which can carry credentials
   * - ``prompt_sha256``
     - Digest, not the prompt body, which can embed user data
   * - ``raw_output``
     - The model's completion
   * - ``ok`` / ``verdict``
     - Whether it passed, and why not if it did not
   * - ``attempt``
     - 1-based attempt number
   * - ``duration_ms``
     - Wall clock

There is no field capable of holding a credential. Header values and API keys are
redacted before write, matched by substring so unfamiliar vendor headers redact by
default rather than leaking by omission.

Adding a capability
===================

.. important::

   **No oracle, no capability. Write the validator first.**

A capability is a prompt plus a validator plus a retry budget. The validator must
check the proposal against **real state** -- resolve the mask, re-parse the file,
confirm the symbol exists.

Validating model output against another model, against a regex over its own text,
or against a mock proves nothing. If you cannot write a validator that can
genuinely fail, what you are building is not a capability, and adding it would
quietly convert a trustworthy surface into a plausible-sounding one.

This is not a stylistic preference. Fabricated APIs reached this repository's
documentation, its test suite, and its shipped module code before being caught.
``explain_exception`` exists because that failure mode is well attested here.

Start from the oracle:

.. code-block:: python

   def validate(self, candidate, context):
       """Can this be checked against something real?"""
       if context is None:
           return Verdict.invalid('no model supplied; cannot validate')
       matches = context.unmask(candidate)          # real ws3 call
       if not matches:
           return Verdict.invalid(
               f'{candidate} matches zero development types'
           )
       return Verdict.valid()

Then the rest:

.. code-block:: python

   from fresh_agent_core.capability import Capability, ParseError, Verdict

   class MyCapability(Capability[MyResult]):
       name = 'my_capability'
       description = (
           'What it does. State what it validates, so a caller knows what '
           'guarantee it is getting.'
       )
       max_attempts = 3

       def build_messages(self, inputs, failures):
           content = f'Do the thing for: {inputs}'
           if failures:
               content += '\n\nPrevious attempts were rejected:\n'
               content += '\n'.join(f'  - {f}' for f in failures)
           return [{'role': 'user', 'content': content}]

       def parse(self, raw):
           ...   # raise ParseError with a specific message if unusable

       def validate(self, candidate, context):
           ...   # the oracle, written first

Three things that are easy to get wrong:

**Incorporate the failures.** A retry that re-sends the identical prompt is a
re-roll, not a repair, and it burns the budget for nothing.

**Give reasons the model can act on.** ``Verdict.invalid()`` requires at least one
reason precisely because a reasonless rejection degrades the loop into repeated
identical sampling -- which looks like a retry budget but is not one.

**Test with bad output, not just good.** The assertion that matters is *invalid
output never escapes the loop*, and you cannot make it without scripting invalid
output:

.. code-block:: python

   from fresh_agent_core import FakeProvider

   provider = FakeProvider([
       'not json at all',                  # malformed
       '{"mask": "? ? nonexistent"}',      # well-formed, fails validation
       '{"mask": "? ? ?"}',                # finally valid
   ])

A capability that only survives well-formed input has not been tested.

See also
========

- :doc:`coding-agent-onboarding` -- architecture and conventions for agents
  working on ``ws3`` itself
- `fresh-agent-core <https://github.com/UBC-FRESH/fresh-agent-core>`_ -- the
  shared runtime
