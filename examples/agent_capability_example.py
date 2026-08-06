"""
Example: calling ws3 agent capabilities from Python.

This demonstrates the Python API. For a live endpoint with a real model,
see ws3.agent.mcp_server and ``ws3-agent-mcp --list-tools``.

FakeProvider is used here so this runs fully offline with no credentials.
"""

from pathlib import Path

from fresh_agent_core import AgentConfig, FakeProvider

from ws3.agent.capabilities import build_registry
from ws3.agent.capabilities.build_mask import BuildMask, MaskRequest
from ws3.agent.capabilities.explain_exception import ExplainException, ExceptionReport
from ws3.agent.capabilities.rtfm_capability import RTFMCapability, RTFMInputs
from ws3.forest import ForestModel

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'

fm = ForestModel(
    model_name=MODEL_NAME,
    model_path=str(MODEL_DIR),
    base_year=2020,
    horizon=10,
    period_length=10,
    max_age=1000,
)
fm.import_landscape_section()
fm.import_areas_section(convert_periods_to_years=10)

CONFIG = AgentConfig(endpoint='offline://example', model='fake')

# ---------------------------------------------------------------------------
# 1. Registry introspection
# ---------------------------------------------------------------------------

print('Registered capabilities:')
for cap in build_registry(fm):
    print(f'  {cap.name}: {cap.description[:70]}...')

# ---------------------------------------------------------------------------
# 2. build_mask -- oracle: mask resolves against real model themes
# ---------------------------------------------------------------------------

# A wildcard mask resolves to at least one development type, so the oracle
# accepts it.  A real provider would omit ``provider=`` so ws3.agent uses
# the configured OpenAI-compatible endpoint.
valid_provider = FakeProvider([
    '{"mask": "? ? ? ? ?", "reasoning": "wildcard every theme"}',
], repeat_last=True)

capability = BuildMask(fm)
result = capability.run(
    MaskRequest(description='all stands'),
    provider=valid_provider,
    config=CONFIG,
    context=fm,
)

if result.ok:
    print('\nbuild_mask: VALID')
    print('  mask tuple:', result.value.mask)
else:
    print('\nbuild_mask: REJECTED -- every attempt failed:')
    for err in result.errors:
        print(' ', err)

# ---------------------------------------------------------------------------
# 3. explain_exception -- oracle: every cited symbol exists in ws3
# ---------------------------------------------------------------------------

symbol_ok_provider = FakeProvider([
    '{"cause": "theme not found", '
    '"next_actions": ["Call fm.themes() to see available themes."], '
    '"symbols_referenced": ["ForestModel.themes", "fm.themes"]}',
], repeat_last=True)

capability = ExplainException()
result = capability.run(
    ExceptionReport(
        exc_type='KeyError',
        message='theme not found: VRI',
        traceback_text='KeyError: "theme not found: VRI"\n  File "...forest.py", line 42',
    ),
    provider=symbol_ok_provider,
    config=CONFIG,
)

if result.ok:
    print('\nexplain_exception: VALID')
    print('  cause:', result.value.cause)
else:
    print('\nexplain_exception: REJECTED:')
    for err in result.errors:
        print(' ', err)

# ---------------------------------------------------------------------------
# 4. rtfm -- oracle: capability name is real, doc URLs are live
# ---------------------------------------------------------------------------

rtfm_provider = FakeProvider([
    '{"capability": "build_mask", '
    '"parameters": {"description": "mature spruce stands"}, '
    '"rationale": "User wants to select mature spruce stands."}',
], repeat_last=True)

capability = RTFMCapability()
result = capability.run(
    RTFMInputs(query='I want to select mature spruce stands'),
    provider=rtfm_provider,
    config=CONFIG,
)

if result.ok:
    print('\nrtfm: VALID')
    print('  recommended:', result.value.capability)
    print('  params:', result.value.parameters)
else:
    print('\nrtfm: REJECTED:')
    for err in result.errors:
        print(' ', err)
