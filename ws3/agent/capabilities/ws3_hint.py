"""
``ws3_hint`` — general modelling guidance backed by a partial oracle.

Unlike the three narrow capabilities (``build_mask``, ``explain_exception``,
``diagnose_import``), this one has no hard domain oracle. A model cannot
prove modelling advice is correct. What it *can* prove is that its references
are real:

- Every cited ws3 symbol actually exists in the installed package
- Every cited doc URL actually returns HTTP 200

That partial oracle is still valuable: fabricated APIs are the primary failure
mode for coding agents operating on unfamiliar packages, which is precisely what
Phase 6 documented across this codebase. Catching hallucinated symbols and
broken links is a meaningful constraint even without a full modelling oracle.

The capability therefore works as a "guided walk" through the docs — it gives
direction, names the right functions, and points to the right URLs, with the
validator guaranteeing those references are not fabrications.
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
class HintResult:
    """
    A modelling hint with verifiable references.

    :param hint: What to do and why.
    :param suggested_steps: Ordered list of steps to take.
    :param suggested_symbols: ws3 symbols the hint references (verifiable).
    :param suggested_urls: Doc URLs the hint references (verifiable).
    :param rtfm_footer: Extracted RTFM footer from the response.
    :param raw: Raw model output.
    """

    hint: str
    suggested_steps: list[str]
    suggested_symbols: list[str]
    suggested_urls: list[str]
    rtfm_footer: str = ''
    raw: str = ''


@dataclass(frozen=True)
class HintInputs:
    """
    What the user wants help with.

    :param goal: Natural-language description of what they are trying to do.
        e.g. "add a new species to my model" or "set up spatial allocation".
    :param context: Optional context about the current state, e.g. what they
        have already tried, relevant error messages, or the model being used.
    """

    goal: str
    context: str = ''


class Ws3Hint(Capability[HintResult]):  # type: ignore[misc]
    """
    Give general modelling guidance for ws3, with verifiable symbol and URL references.

    Validated by checking every cited ws3 symbol actually exists in the installed
    package, and every cited doc URL returns HTTP 200. The modelling advice itself
    cannot be verified — but the references can, which catches hallucinated APIs.

    Use when ``rtfm`` routes to this capability, or when the user explicitly asks
    "how do I..." / "what is the best way to..." without a specific error.
    """

    name = 'ws3_hint'
    description = (
        'Get general guidance on how to use ws3 for a modelling task. '
        'Returns suggested steps, cited ws3 symbols (validated against the '
        'installed package), and cited doc URLs (validated with HTTP 200). '
        'Use when the user wants modelling guidance rather than a specific '
        'capability result.'
    )
    max_attempts = 2  # Fallback on rejection lets model try again with corrections

    input_schema = {
        'type': 'object',
        'properties': {
            'goal': {
                'type': 'string',
                'description': (
                    'What the user is trying to do, e.g. '
                    '"add a species to my yield table" or '
                    '"set up a spatial allocation problem".'
                ),
            },
            'context': {
                'type': 'string',
                'description': (
                    'Optional context: what they have already tried, '
                    'relevant error messages, or the model being used.'
                ),
            },
        },
        'required': ['goal'],
    }

    def from_payload(self, payload: dict[str, Any]) -> HintInputs:
        """Build :py:class:`HintInputs` from MCP tool arguments."""
        return HintInputs(
            goal=str(payload.get('goal', '')),
            context=str(payload.get('context', '')),
        )

    def render(self, value: HintResult) -> str:
        """Render as readable text."""
        lines = [value.hint, '']
        if value.suggested_steps:
            lines.append('Suggested steps:')
            for i, step in enumerate(value.suggested_steps, 1):
                lines.append(f'  {i}. {step}')
            lines.append('')
        if value.suggested_symbols:
            lines.append('ws3 symbols referenced:')
            for sym in value.suggested_symbols:
                lines.append(f'  - {sym}')
            lines.append('')
        return '\n'.join(lines)

    def build_messages(
        self,
        inputs: HintInputs,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        failure_context = ''
        if failures:
            failure_context = (
                '\n\nPrevious attempts failed with:\n'
                + '\n'.join(f'  - {f}' for f in failures)
                + '\nDo not repeat approaches that failed.'
            )

        context_block = (
            f'\n\nContext provided by caller:\n{inputs.context}'
            if inputs.context
            else ''
        )

        content = (
            'You are a ws3 modelling guide. The user wants help with:\n'
            f'{inputs.goal}\n'
            f'{context_block}\n'
            f'{failure_context}\n'
            '\n'
            'Domain rules you must follow:\n'
            '- ws3 has no ws3.fire module and no ws3.fire.add_fire function.\n'
            '- Disturbances are represented by action definitions and transition '
            'rules in the ACTIONS and TRANSITIONS sections.\n'
            '- Harvests and partial treatments use the action/transition model; '
            'there is no generic partial-cut helper. A partial treatment must be '
            'defined by the model data before it can be applied.\n'
            '- If the live model context does not list the requested action, say '
            'that it is unavailable and explain which section files must define it.\n'
            '- Never invent a module, function, action code, or transition.\n'
            '\n'
            'Allowed ws3 references for this answer are:\n'
            '  ws3.forest.ForestModel\n'
            '  ws3.forest.Action\n'
            '  ws3.forest.ForestModel.import_actions_section\n'
            '  ws3.forest.ForestModel.import_transitions_section\n'
            '  ws3.forest.ForestModel.apply_action\n'
            '  ws3.forest.ForestModel.apply_schedule\n'
            '  ws3.forest.ForestModel.is_harvest\n'
            'Use only references from this list in suggested_symbols.\n'
            'Use only these documentation URLs in suggested_urls or the RTFM '
            'footer:\n'
            '  https://ubc-fresh.github.io/ws3/forest.html\n'
            '  https://ubc-fresh.github.io/ws3/reference/contracts/module_boundaries.html\n'
            '  https://ubc-fresh.github.io/ws3/textbook/ch04_actions_and_transitions.html\n'
            '  https://ubc-fresh.github.io/ws3/textbook/ch15_disturbance_modelling.html\n'
            '\n'
            'Respond with a JSON object and nothing else. The response must '
            'include:\n'
            '  {\n'
            '    "hint": "<one-paragraph summary of the best approach>",\n'
            '    "suggested_steps": ["<step 1>", "<step 2>", ...],\n'
            '    "suggested_symbols": ["ws3.forest.ForestModel.themes", ...],\n'
            '    "suggested_urls": ["https://ubc-fresh.github.io/ws3/core.html#...", ...]\n'
            '  }\n'
            '\n'
            'Rules:\n'
            '- Only cite ws3 symbols that actually exist in the installed package\n'
            '- Only cite doc URLs that return HTTP 200\n'
            '- suggested_symbols must name the *specific* ws3 function or class, '
            'not a general concept\n'
            '- suggested_urls must point to the exact section that covers the topic\n'
            '- Do not give forestry or silvicultural advice beyond what is needed '
            'to use the API correctly\n'
            '\n'
            'RTFM links:\n'
            'List every ws3 symbol and doc URL referenced in your response '
            'in the "suggested_symbols" and "suggested_urls" fields.\n'
        )
        content += RTFM_FOOTER_INSTRUCTION
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> HintResult:
        """Parse the JSON response and extract the RTFM footer."""
        json_text, footer = extract_json(raw)
        payload = json.loads(json_text)

        if not isinstance(payload, dict):
            raise ParseError(f'expected JSON object, got {type(payload).__name__}')

        hint = str(payload.get('hint', ''))
        suggested_steps = [
            str(s) for s in payload.get('suggested_steps', [])
            if isinstance(s, str)
        ]
        suggested_symbols = [
            str(s) for s in payload.get('suggested_symbols', [])
            if isinstance(s, str)
        ]
        suggested_urls = [
            str(u) for u in payload.get('suggested_urls', [])
            if isinstance(u, str) and u.startswith('http')
        ]

        return HintResult(
            hint=hint,
            suggested_steps=suggested_steps,
            suggested_symbols=suggested_symbols,
            suggested_urls=suggested_urls,
            rtfm_footer=footer,
            raw=raw,
        )

    def validate(self, candidate: HintResult, context: Any) -> Verdict:
        """
        Check that all cited symbols and URLs are real.

        The modelling advice itself is not verified — only the references are.
        """
        # Check RTFM footer (presence, symbol existence, URL validity)
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

        # Validate suggested_symbols against ws3 symbol table
        from ws3.agent.capabilities.rtfm import _extract_symbols, _known_symbols

        all_cited = _extract_symbols(
            ' '.join(candidate.suggested_symbols + candidate.suggested_urls)
        )
        known = _known_symbols()
        fabricated = [
            s for s in all_cited
            if s.split('.')[-1] not in known
            and any(part in known for part in s.split('.')[:-1])
        ]
        if fabricated:
            return Verdict.invalid(
                'suggested_symbols cites non-existent ws3 symbols: '
                + ', '.join(sorted(set(fabricated)))
            )

        # Validate suggested_urls return HTTP 200
        from ws3.agent.capabilities.rtfm import _doc_url_valid

        bad_urls = [
            url for url in candidate.suggested_urls
            if not _doc_url_valid(url)
        ]
        if bad_urls:
            return Verdict.invalid(
                'suggested_urls return errors: ' + ', '.join(bad_urls)
            )

        return Verdict.valid()
