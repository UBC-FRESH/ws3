"""
Build a development-type mask from a natural-language description.

Oracle: :py:meth:`ws3.forest.ForestModel.unmask` resolves the proposed mask
against the actual model. A mask matching zero development types is rejected --
it is syntactically fine and operationally useless, which is exactly the failure
mode a human hits and cannot easily diagnose.

Mask arity is *not* the model's problem. The number of themes is authoritative
model state, read from the LANDSCAPE section of the input dataset (or from
whatever built the model), so the language model is asked only which themes to
constrain and to what code. The full-arity mask is assembled here, in code, with
every unmentioned position filled in as a wildcard.

This is deliberate. An earlier version asked for the whole space-separated mask
and let the validator police the length; a live 9B model duly returned ten
entries for a five-theme model, and the retry cost about 28 seconds to recover a
fact the code already knew. Do not ask a model to reproduce state you can read.

Know what the oracle does *not* prove. Theme positions carry no fixed meaning:
the Woodstock format lets each model define its own themes and its own codes
within them, so nothing about position 2 is knowable from the format. Similar
layouts across a family of models are a convention of whoever built them, not a
property of ws3. ``unmask`` therefore establishes that a mask *resolves to real
development types*, and nothing whatsoever about whether it means what the user
asked for. A mask that quietly selects the wrong stratum passes this oracle.

The prompt is written accordingly: it presents only what the model instance
actually declares, forbids assuming a position holds any particular kind of
information, and offers an explicit way to report that a request cannot be
grounded -- because guessing produces output the validator will happily accept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from fresh_agent_core.capability import Capability, ParseError, Verdict

from ws3.agent.themes import ThemeError, ThemeSchema, schema_for


@dataclass(frozen=True)
class MaskRequest:
    """
    What to build a mask for.

    :param description: Natural-language description of the stands to select,
        e.g. "mature spruce on good sites".
    """

    description: str


class BuildMask(Capability[tuple]):
    """Propose a development-type mask, validated by resolving it against the model."""

    name = 'build_mask'
    description = (
        'Build a ws3 development-type mask from a natural-language description. '
        'Validated by resolving the proposed mask against the supplied ForestModel: '
        'a mask matching zero development types is rejected, so a returned mask is '
        'guaranteed to select at least one real development type.'
    )
    max_attempts = 3

    input_schema = {
        'type': 'object',
        'properties': {
            'description': {
                'type': 'string',
                'description': 'Natural-language description of the stands to '
                               'select, e.g. "mature spruce on good sites".',
            },
        },
        'required': ['description'],
    }

    def from_payload(self, payload: dict) -> MaskRequest:
        """Build a :py:class:`MaskRequest` from MCP tool arguments."""
        return MaskRequest(description=str(payload.get('description', '')))

    def coerce_input(self, inputs: Any) -> MaskRequest:
        """
        Accept a bare description string as well as a :py:class:`MaskRequest`.

        The single-string form is the documented convenience call
        (``ws3.agent.run('build_mask', 'mature spruce stands', context=fm)``), so
        it has to actually work.
        """
        if isinstance(inputs, MaskRequest):
            return inputs
        if isinstance(inputs, str):
            return MaskRequest(description=inputs)
        if isinstance(inputs, dict):
            return self.from_payload(inputs)
        raise TypeError(
            f'build_mask takes a description string, a dict, or a MaskRequest; '
            f'got {type(inputs).__name__}'
        )

    def render(self, value: tuple) -> str:
        """Render as a Woodstock-style space-separated mask, ready to paste."""
        return ' '.join(value)

    def build_messages(self, inputs: MaskRequest, failures: tuple[str, ...]) -> list[dict[str, str]]:
        """
        Build the prompt, folding in why previous attempts were rejected.

        Asks for theme constraints rather than a finished mask, so the model never
        has to count theme positions and cannot get the arity wrong.
        """
        count = 'an unknown number of' if self._schema is None else str(self._schema.nthemes)
        listing = '  (not available)' if self._schema is None else self._schema.describe()
        content = (
            'You are selecting development types in a ws3 forest model.\n'
            '\n'
            'Theme positions have no fixed meaning. This is a Woodstock-format '
            'model: how many themes there are, what each one represents, and which '
            'codes it admits are all chosen by whoever built this particular model. '
            'Do not assume a position holds species, age, site quality, ownership, '
            'or any other attribute. Other models you have seen tell you nothing '
            'about this one.\n'
            '\n'
            f'This model has {count} themes. Available codes:\n'
            f'{listing}\n'
            '\n'
            f'Requested selection: {inputs.description}\n'
            '\n'
            'List only the theme positions you want to constrain, and the code to '
            'constrain each one to. Every position you do not mention is left '
            'unconstrained, matching anything. Do not assemble the mask yourself and '
            'do not pad it with wildcards -- that is done for you.\n'
            '\n'
            'Use only codes listed above. Where an aggregate expresses the request, '
            'prefer it: aggregates are groupings the modeller defined, so they carry '
            'the intended meaning.\n'
            '\n'
            'Respond with a JSON object and nothing else:\n'
            '  {"constraints": {"<theme position>": "<code>"}, '
            '"reasoning": "<one sentence>"}\n'
            '\n'
            'To select everything, constrain nothing: '
            '{"constraints": {}, "reasoning": "..."}\n'
            '\n'
            'If the request cannot be grounded in the themes and codes listed above, '
            'say so instead of guessing -- a wrong mask resolves perfectly well and '
            'will not be caught:\n'
            '  {"insufficient_information": "<what is missing>"}\n'
        )
        if failures:
            content += (
                '\nPrevious attempts were rejected:\n'
                + '\n'.join(f'  - {f}' for f in failures)
                + '\nPropose different constraints that avoid these problems.\n'
            )
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> tuple:
        """
        Build the mask from the model's theme constraints.

        Accepts a finished ``mask`` too, since a model that volunteers one is not
        worth a retry -- but that path is checked for arity by
        :py:meth:`validate`, whereas the ``constraints`` path cannot get arity
        wrong by construction.
        """
        text = raw.strip()
        # Models frequently wrap JSON in a fenced code block despite instructions.
        # Tolerating that is cheaper than burning a retry on formatting.
        if text.startswith('```'):
            text = text.strip('`')
            if text.lstrip().lower().startswith('json'):
                text = text.lstrip()[4:]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f'expected a JSON object with a "constraints" key, got: '
                f'{raw[:200]!r} ({exc})'
            ) from exc

        if not isinstance(payload, dict):
            raise ParseError('expected a JSON object')

        # An explicit "I cannot ground this" is a better answer than a mask that
        # resolves and means the wrong thing, so it is surfaced rather than
        # retried into something more confident.
        if 'insufficient_information' in payload:
            raise ParseError(
                f'the request could not be grounded in this model: '
                f'{payload["insufficient_information"]}'
            )

        if 'constraints' in payload:
            if self._schema is None:
                raise ParseError(
                    'no ForestModel was supplied, so the number of themes is '
                    'unknown and the mask cannot be assembled'
                )
            try:
                return self._schema.assemble(payload['constraints'])
            except ThemeError as exc:
                raise ParseError(str(exc)) from None

        if 'mask' in payload:
            mask = payload['mask']
            if isinstance(mask, str):
                return tuple(mask.lower().split())
            if isinstance(mask, list):
                return tuple(str(m).lower() for m in mask)
            raise ParseError(
                f'"mask" must be a string or list, got {type(mask).__name__}'
            )

        raise ParseError('expected a JSON object containing a "constraints" key')

    def validate(self, candidate: tuple, context: Any) -> Verdict:
        """
        Resolve the mask against the real model.

        :param candidate: Proposed mask.
        :param context: The :py:class:`~ws3.forest.ForestModel` to resolve against.
        """
        if context is None:
            return Verdict.invalid(
                'No ForestModel was supplied as context, so the mask cannot be '
                'validated. This is a caller error, not a model error.'
            )

        expected = context.nthemes()
        if len(candidate) != expected:
            return Verdict.invalid(
                f'mask has {len(candidate)} entries but the model has {expected} '
                f'themes; supply exactly one code or ? per theme'
            )

        try:
            matches = context.unmask(candidate)
        except Exception as exc:
            # unmask asserts on malformed input rather than raising something
            # specific, so anything escaping it is reported as a rejection with the
            # underlying reason attached.
            return Verdict.invalid(f'mask could not be resolved: {type(exc).__name__}: {exc}')

        if not matches:
            unknown = ThemeSchema.from_model(context).unknown_codes(candidate)
            detail = ('; ' + '; '.join(unknown)) if unknown else ''
            return Verdict.invalid(
                f'mask matches zero development types{detail}'
            )

        return Verdict.valid()

    def __init__(self, fm: Optional[Any] = None) -> None:
        """
        :param fm: Model supplying the theme schema. The theme count is what makes
            mask assembly deterministic, so without a model the capability can
            describe nothing and assemble nothing.
        """
        super().__init__()
        self._schema: Optional[ThemeSchema] = schema_for(fm)
