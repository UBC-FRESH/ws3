"""
RTFM — "Which capability do I use?"

This is a specialised index over the ws3 capability registry. Its job is:
**given a user goal or error message, return which capability to call and
what parameters to pass**.

This is NOT a general RAG system. It does NOT search the modelling corpus.
It does NOT give forestry modelling advice. It routes to known capabilities.

Spec: planning/phase8_embedded_agents.md — Task 8.7a RTFMCapability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fresh_agent_core.capability import Capability, ParseError, Verdict

from ws3.agent.capabilities.rtfm import (
    RTFM_FOOTER_INSTRUCTION,
    extract_json,
    validate_rtfm_footer,
)


@dataclass(frozen=True)
class RTFMInputs:
    """What the user is trying to do or what went wrong."""

    query: str
    error_text: str = ''


@dataclass(frozen=True)
class RTFMResult:
    """
    A recommended capability and how to call it.

    :param capability: Name of the recommended capability, e.g. ``"explain_exception"``.
    :param parameters: Dict of parameter names to values to pass.
    :param rationale: One-sentence explanation of why this capability fits.
    :param alternative: Name of a second capability to try if the first fails,
        or the empty string if no good alternative exists.
    """

    capability: str
    parameters: dict
    rationale: str
    alternative: str = ''
    rtfm_footer: str = ''
    raw: str = ''


#: Static capability index. Each entry maps a capability name to the scenarios
#: it handles. This is the only "knowledge base" RTFMCapability consults.
_CAPABILITY_INDEX = {
    'inspect_model': {
        'keywords': [
            'inspect', 'model info', 'metadata', 'snapshot', 'overview',
            'what is the model', 'base_year', 'horizon', 'period_length',
            'periods', 'nthemes', 'nactions', 'ndtypes', 'total_area',
        ],
        'triggers': [
            'show me the model', 'model info', 'model metadata',
            'what is in the model', 'model snapshot', 'inspect the model',
        ],
        'context_params': ['query'],
        'description': (
            'Read-only metadata snapshot of a ForestModel. Returns model_name, '
            'base_year, horizon, period_length, periods, theme/action/dtype '
            'counts, and total area. Does not execute model-generated Python. '
            'Use when the user wants to see basic information about the loaded '
            'ForestModel.'
        ),
    },
    'explain_exception': {
        'keywords': [
            'exception', 'error', 'traceback', 'crash', 'fail', 'raised',
            'AttributeError', 'TypeError', 'ValueError', 'KeyError',
            'NameError', 'ImportError', 'RuntimeError', ' ws3 ',
        ],
        'triggers': [
            'why did this fail', 'what does this error mean', 'explain error',
            'error message', 'exception occurred', ' crashed ', ' crashed\n',
        ],
        'context_params': ['exc_type', 'message', 'traceback_text', 'context'],
        'description': (
            'Explains a ws3 Python exception in plain language to a forest '
            'modeller who is not a Python specialist. Use when the user sees '
            'a Python traceback or error message they do not understand.'
        ),
    },
    'diagnose_import': {
        'keywords': [
            'import', 'load', 'read', 'parse', 'section', 'woodstock',
            'failed to import', 'could not import', 'import error',
            '.lan', '.are', '.yld', '.act', '.trn', '.con', '.out',
        ],
        'triggers': [
            'failed to import', 'could not load', 'section failed',
            'model import error', 'cannot import section', 'invalid syntax',
            'section file', 'woodstock file',
        ],
        'context_params': ['model_path', 'model_name', 'section', 'error', 'excerpt'],
        'description': (
            'Diagnoses why a Woodstock section file failed to import and proposes '
            'a corrected line. The fix is validated by re-importing. Use when the '
            'user reports that ws3 cannot read their Woodstock model files.'
        ),
    },
    'build_mask': {
        'keywords': [
            'mask', 'development type', 'select', 'filter', 'stand',
            'stratum', 'cover type', 'species', 'age', 'site index',
        ],
        'triggers': [
            'build a mask', 'select stands', 'mask for', 'which development',
            'development type mask', 'filter by', 'age class',
        ],
        'context_params': ['description'],
        'description': (
            'Builds a ws3 development-type mask from a natural-language description. '
            'The mask is validated against the actual ForestModel to guarantee it '
            'matches at least one real development type. Use when the user wants '
            'to select stands by type, species, age, or any combination.'
        ),
    },
    'report_scenario_inventory_products': {
        'keywords': [
            'scenario report', 'inventory report', 'inventory products',
            'harvested volume', 'standing volume', 'schedule report',
        ],
        'triggers': [
            'report scenario inventory', 'report inventory products',
            'show scenario products', 'scenario harvest report',
        ],
        'context_params': ['model_path', 'model_name', 'schedule_path'],
        'description': (
            'Produces a deterministic inventory and products report for a '
            'Woodstock scenario and optional schedule. Use when the user wants '
            'period-by-period harvested area, harvested volume, or standing '
            'volume from a model scenario.'
        ),
    },
    'ws3_hint': {
        'keywords': [
            'how do i', 'how to', 'tutorial', 'explain how',
            'what is the best way', 'what are the steps', 'can you show me',
            'example', 'guide', 'manual', 'best practice', 'recommend',
            'suggest', 'set up', 'add a ', 'create a ', 'build a ',
        ],
        'triggers': [
            'how do i add', 'how do i create', 'how do i set up',
            'what is a development type', 'explain the difference between',
            'how does ws3 work', 'getting started', 'beginner',
        ],
        'context_params': ['goal', 'context'],
        'description': (
            'Gives general modelling guidance for ws3: suggested steps, '
            'cited ws3 symbols (validated against the installed package), '
            'and cited doc URLs (validated with HTTP 200). '
            'Use when the user wants modelling guidance or "how do I..." '
            'help rather than a specific validated output.'
        ),
    },
}

#: Keywords that mean the user is asking for general modelling help, not a
#: specific capability — RTFMCapability routes these to ``ws3_hint``.
_GENERAL_MODELLING_KEYWORDS = {
    'how do i', 'how to', 'tutorial', 'explain how', 'what is a ',
    'what are the steps', 'can you show me', 'example', 'documentation',
    'guide', 'manual', 'best practice', 'recommend', 'suggest',
}


class RTFMCapability(Capability[RTFMResult]):
    """
    Route a user goal or error to the correct ws3 agent capability.

    This is a specialised index over the capability registry. It does not
    search the modelling corpus or give forestry advice — it only answers:
    "which capability should I call and with what parameters?".
    """

    name = 'rtfm'
    description = (
        'Route a user goal or error message to the correct ws3 agent capability. '
        'Returns which capability to call and what parameters to pass. '
        'Validated by checking the capability name is real. '
        'Does not give modelling advice or search documentation.'
    )
    max_attempts = 1  # No oracle; single shot

    input_schema = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': (
                    'What the user is trying to do, e.g. "I want to select '
                    'mature spruce stands" or "ws3 crashed with a KeyError".'
                ),
            },
            'error_text': {
                'type': 'string',
                'description': 'Full error text or traceback, if applicable.',
            },
        },
        'required': ['query'],
    }

    def from_payload(self, payload: dict) -> RTFMInputs:
        """Build an :py:class:`RTFMInputs` from MCP tool arguments."""
        return RTFMInputs(
            query=str(payload.get('query', '')),
            error_text=str(payload.get('error_text', '')),
        )

    def render(self, value: RTFMResult) -> str:
        """Render as a human-readable capability recommendation."""
        if not value.capability or value.capability == 'none':
            return (
                'No specific validated capability matches your request. '
                'Try the ws3_hint capability for general modelling guidance: '
                '{"capability": "ws3_hint", "parameters": {"goal": "<what you want to do>"}}'
            )
        alt = (
            f'\nIf that does not help, try: {value.alternative}'
            if value.alternative
            else ''
        )
        param_lines = '\n  '.join(
            f'{k}={v!r}' for k, v in value.parameters.items()
        )
        return (
            f'Use the `{value.capability}` capability with these parameters:\n'
            f'  {param_lines}\n'
            f'\nRationale: {value.rationale}{alt}\n'
            '\nRTFM links: none'
        )

    def build_messages(
        self,
        inputs: RTFMInputs,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        combined = inputs.query
        if inputs.error_text:
            combined += f'\n\nError text:\n{inputs.error_text}'

        content = (
            'You are a specialised router for ws3 agent capabilities.\n'
            '\n'
            'Your ONLY job: given a user goal or error message, return which '
            'ws3 capability to call and what parameters to pass.\n'
            '\n'
            'Do NOT give modelling advice. Do NOT search documentation. '
            'Do NOT explain how the capability works. '
            'Only return the routing decision.\n'
            '\n'
            'Available capabilities and when to use them:\n'
        )
        for name, info in _CAPABILITY_INDEX.items():
            content += (
                f'\n  {name}:\n'
                f'    {info["description"]}\n'
                f'    Parameters: {", ".join(info["context_params"])}\n'
            )

        content += (
            f'\nUser request:\n{combined}\n'
            '\n'
            'Respond with a JSON object and nothing else:\n'
            '  {"capability": "<name or "none">", '
            '"parameters": {{"<param>": "<value>", ...}}, '
            '"rationale": "<one sentence>", '
            '"alternative": "<name or "">"}\n'
        )
        content += RTFM_FOOTER_INSTRUCTION
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> RTFMResult:
        json_text, footer = extract_json(raw)
        payload = json.loads(json_text)

        if not isinstance(payload, dict):
            raise ParseError(f'expected a JSON object, got {type(payload).__name__}')
        for key in ('capability', 'parameters', 'rationale'):
            if key not in payload:
                raise ParseError(f'missing required key {key!r}')

        capability = str(payload.get('capability', '')).strip()
        parameters = dict(payload.get('parameters', {}))
        rationale = str(payload.get('rationale', ''))
        alternative = str(payload.get('alternative', ''))

        # Validate capability name
        if capability != 'none' and capability not in _CAPABILITY_INDEX:
            raise ParseError(
                f'unknown capability {capability!r}; must be one of '
                f'{", ".join(sorted(_CAPABILITY_INDEX))} or "none"'
            )

        return RTFMResult(
            capability=capability,
            parameters=parameters,
            rationale=rationale,
            alternative=alternative,
            rtfm_footer=footer,
            raw=raw,
        )

    def validate(self, candidate: RTFMResult, context: Any) -> Verdict:
        """
        Validate that the routing decision is sensible.

        :param candidate: The parsed routing result.
        :param context: Optional dict with ``include_rtfm`` (bool).
        """
        # Check RTFM footer
        include_rtfm = True
        if isinstance(context, dict):
            include_rtfm = context.get('include_rtfm', True)

        rtfm_verdict = validate_rtfm_footer(
            candidate.raw,
            footer_text=candidate.rtfm_footer,
            include_rtfm=include_rtfm,
        )
        if not rtfm_verdict.ok:
            return rtfm_verdict

        if candidate.capability == 'none':
            # General modelling query — routing is correct by definition
            return Verdict.valid()

        # Capability name should be in the index
        if candidate.capability not in _CAPABILITY_INDEX:
            return Verdict.invalid(
                f'unrecognised capability {candidate.capability!r}'
            )

        # All recommended parameters should be in the capability's context_params
        info = _CAPABILITY_INDEX[candidate.capability]
        valid_params = set(info['context_params'])
        for param in candidate.parameters:
            if param not in valid_params:
                return Verdict.invalid(
                    f'parameter {param!r} is not valid for capability '
                    f'{candidate.capability!r}; valid parameters are: '
                    f'{", ".join(sorted(valid_params))}'
                )

        return Verdict.valid()
