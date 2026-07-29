"""
Build a development-type mask from a natural-language description.

Oracle: :py:meth:`ws3.forest.ForestModel.unmask` resolves the proposed mask
against the actual model. A mask matching zero development types is rejected --
it is syntactically fine and operationally useless, which is exactly the failure
mode a human hits and cannot easily diagnose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from fresh_agent_core.capability import Capability, ParseError, Verdict

#: Cap on theme codes listed per theme in the prompt. Real models can carry
#: hundreds; listing all of them buries the task and wastes context.
MAX_CODES_PER_THEME = 40


@dataclass(frozen=True)
class MaskRequest:
    """
    What to build a mask for.

    :param description: Natural-language description of the stands to select,
        e.g. "mature spruce on good sites".
    """

    description: str


def _theme_summary(fm: Any) -> str:
    """
    Describe the model's themes so the model can only propose real codes.

    Without this the model invents plausible-looking theme codes, which the
    validator then rejects -- correct but wasteful. Showing the actual codes turns
    most of the task into selection rather than generation.
    """
    lines = []
    for index, theme in enumerate(fm._themes):
        name = theme.get('__name__', f'theme{index}')
        codes = [k for k in theme if not k.startswith('__')]
        shown = sorted(codes)[:MAX_CODES_PER_THEME]
        suffix = '' if len(codes) <= MAX_CODES_PER_THEME else f' ... ({len(codes)} total)'
        lines.append(f'  position {index} ({name}): {", ".join(shown)}{suffix}')
    return '\n'.join(lines)


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

    def build_messages(self, inputs: MaskRequest, failures: tuple[str, ...]) -> list[dict[str, str]]:
        """Build the prompt, folding in why previous attempts were rejected."""
        content = (
            'You are constructing a development-type mask for a ws3 forest model.\n'
            '\n'
            'A mask is a space-separated list of theme codes, one per theme position, '
            'in order. Use ? as a wildcard for any position that should not be '
            'constrained.\n'
            '\n'
            f'This model has {self._nthemes} themes. Available codes:\n'
            f'{self._themes}\n'
            '\n'
            f'Requested selection: {inputs.description}\n'
            '\n'
            'Respond with a JSON object and nothing else:\n'
            '  {"mask": "<code-or-? for each theme, space separated>", '
            '"reasoning": "<one sentence>"}\n'
        )
        if failures:
            content += (
                '\nPrevious attempts were rejected:\n'
                + '\n'.join(f'  - {f}' for f in failures)
                + '\nPropose a different mask that avoids these problems.\n'
            )
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> tuple:
        """Extract the mask as a tuple of theme codes."""
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
                f'expected a JSON object with a "mask" key, got: {raw[:200]!r} ({exc})'
            ) from exc

        if not isinstance(payload, dict) or 'mask' not in payload:
            raise ParseError('expected a JSON object containing a "mask" key')

        mask = payload['mask']
        if isinstance(mask, str):
            return tuple(mask.lower().split())
        if isinstance(mask, list):
            return tuple(str(m).lower() for m in mask)
        raise ParseError(f'"mask" must be a string or list, got {type(mask).__name__}')

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
            unknown = self._unknown_codes(candidate, context)
            detail = (
                f'; these codes are not defined for their theme: {", ".join(unknown)}'
                if unknown else ''
            )
            return Verdict.invalid(
                f'mask matches zero development types{detail}'
            )

        return Verdict.valid()

    @staticmethod
    def _unknown_codes(mask: tuple, fm: Any) -> list[str]:
        """
        Identify codes absent from their theme.

        Turns "matched nothing" into an actionable reason, which is what the retry
        needs in order to do better than re-rolling.
        """
        unknown = []
        for index, code in enumerate(mask):
            if code == '?':
                continue
            if index < len(fm._themes) and code not in fm._themes[index]:
                unknown.append(f'position {index}: {code!r}')
        return unknown

    def __init__(self, fm: Optional[Any] = None) -> None:
        """
        :param fm: Optional model used to describe themes in the prompt. When
            omitted the prompt omits the code listing and the model must guess,
            which the validator will usually reject.
        """
        super().__init__()
        self._nthemes = fm.nthemes() if fm is not None else 'an unknown number of'
        self._themes = _theme_summary(fm) if fm is not None else '  (not available)'
