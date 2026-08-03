"""
Explain a ws3 exception in plain language.

Oracle: every ws3 symbol the explanation mentions must actually exist.

This capability is deliberate. Phase 6 removed fabricated APIs from the
documentation, Phase 7.5 found the same defect class in the test suite
(``ws3.core.interpolate_curves``), and Phase 7.6 found it in shipped module code
(``ws3.core.compile_scenario``, ``Problem.get_objective_value``). In every case a
plausible-sounding name that was never written passed unnoticed.

An LLM asked to explain an error is under exactly the same pressure to invent a
helpful-sounding method. Here the same check runs automatically, on every
attempt, before the caller sees anything.
"""

from __future__ import annotations

import importlib
import json
import re
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from fresh_agent_core.capability import Capability, ParseError, Verdict

#: Modules whose public surface counts as "real ws3".
WS3_MODULES = (
    'ws3.common',
    'ws3.core',
    'ws3.forest',
    'ws3.opt',
    'ws3.spatial',
    'ws3.financial',
)

#: Matches dotted references like ``ws3.opt.Problem.solve`` or ``Problem.solve``.
_SYMBOL_PATTERN = re.compile(r'\b((?:ws3\.)?[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b')

#: Roots that look like dotted references but are not ws3 API. Matched on the
#: first component, so 'e.g' (from prose) and 'np.array' are both skipped.
_IGNORE_ROOTS = frozenset({
    'self', 'cls', 'os', 'sys', 'np', 'numpy', 'pd', 'pandas', 'json', 're',
    'math', 'pathlib', 'typing', 'dict', 'list', 'str', 'int', 'float', 'bool',
    'set', 'tuple', 'e', 'i', 'g', 'etc', 'vs',
})


@dataclass(frozen=True)
class ExceptionReport:
    """
    The failure to explain.

    :param exc_type: Exception class name, e.g. ``ValueError``.
    :param message: The exception message.
    :param traceback_text: Formatted traceback, optional but strongly preferred.
    :param context: Anything else worth knowing, e.g. what the caller was doing.
    """

    exc_type: str
    message: str
    traceback_text: str = ''
    context: str = ''


@dataclass(frozen=True)
class Explanation:
    """A validated explanation."""

    cause: str
    next_actions: tuple[str, ...]
    symbols_referenced: tuple[str, ...]


def _public_names(module_name: str) -> set[str]:
    """Public attribute names of a module, plus its classes' public methods."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return set()

    names: set[str] = set()
    for attr in dir(module):
        if attr.startswith('_'):
            continue
        names.add(attr)
        value = getattr(module, attr, None)
        if isinstance(value, type):
            names.update(m for m in dir(value) if not m.startswith('_'))
    return names


def known_symbols(modules: Iterable[str] = WS3_MODULES) -> set[str]:
    """
    Every public name across the given ws3 modules.

    Deliberately a flat set rather than a dotted index: an explanation may
    reasonably write ``Problem.solve`` or ``ws3.opt.Problem.solve``, and rejecting
    one form on a technicality would generate noise rather than catch defects.
    """
    names: set[str] = set()
    for module_name in modules:
        names.update(_public_names(module_name))
        names.add(module_name.split('.')[-1])
    return names


def extract_symbols(text: str) -> list[str]:
    """Pull dotted references that look like ws3 API mentions out of *text*."""
    found = []
    for match in _SYMBOL_PATTERN.finditer(text):
        symbol = match.group(1)
        if symbol.split('.')[0].lower() in _IGNORE_ROOTS:
            continue
        found.append(symbol)
    return found


class ExplainException(Capability[Explanation]):  # type: ignore[misc]
    """Explain a failure, validated so it cannot cite APIs that do not exist."""

    name = 'explain_exception'
    description = (
        'Explain a ws3 exception in plain language and suggest next actions. '
        'Validated by checking that every ws3 symbol the explanation references '
        'actually exists in the installed package, so the explanation cannot cite '
        'methods or functions that were never written.'
    )
    max_attempts = 3

    input_schema = {
        'type': 'object',
        'properties': {
            'exc_type': {'type': 'string', 'description': 'Exception class name.'},
            'message': {'type': 'string', 'description': 'Exception message.'},
            'traceback_text': {
                'type': 'string',
                'description': 'Formatted traceback. Optional but strongly preferred.',
            },
            'context': {
                'type': 'string',
                'description': 'What the caller was doing when it failed.',
            },
        },
        'required': ['exc_type', 'message'],
    }

    def from_payload(self, payload: dict[str, Any]) -> ExceptionReport:
        """Build an :py:class:`ExceptionReport` from MCP tool arguments."""
        return ExceptionReport(
            exc_type=str(payload.get('exc_type', '')),
            message=str(payload.get('message', '')),
            traceback_text=str(payload.get('traceback_text', '')),
            context=str(payload.get('context', '')),
        )

    def coerce_input(self, inputs: Any) -> ExceptionReport:
        """
        Accept a live exception as well as an :py:class:`ExceptionReport`.

        Passing the caught exception straight through is the obvious call at a
        ``except`` site, and it also captures the traceback automatically -- which
        materially improves the explanation, and which a hand-built report usually
        omits.
        """
        if isinstance(inputs, ExceptionReport):
            return inputs
        if isinstance(inputs, BaseException):
            return ExceptionReport(
                exc_type=type(inputs).__name__,
                message=str(inputs),
                traceback_text=''.join(
                    traceback.format_exception(
                        type(inputs), inputs, inputs.__traceback__
                    )
                ),
            )
        if isinstance(inputs, dict):
            return self.from_payload(inputs)
        raise TypeError(
            f'explain_exception takes an Exception, a dict, or an ExceptionReport; '
            f'got {type(inputs).__name__}'
        )

    def render(self, value: Explanation) -> str:
        """Render cause and next actions as readable text."""
        actions = '\n'.join(f'  - {a}' for a in value.next_actions)
        return f'{value.cause}\n\nSuggested next steps:\n{actions}'

    def build_messages(
        self,
        inputs: ExceptionReport,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        content = (
            'Explain the following ws3 error to a forest modeller who is not a '
            'Python specialist.\n'
            '\n'
            f'Exception: {inputs.exc_type}\n'
            f'Message: {inputs.message}\n'
        )
        if inputs.traceback_text:
            content += f'\nTraceback:\n{inputs.traceback_text}\n'
        if inputs.context:
            content += f'\nContext: {inputs.context}\n'
        content += (
            '\nOnly reference ws3 functions, classes and methods that genuinely '
            'exist. Do not invent plausible-sounding API names -- an explanation '
            'citing a method that does not exist is worse than no explanation.\n'
            '\n'
            'Respond with a JSON object and nothing else:\n'
            '  {"cause": "<one or two sentences>", '
            '"next_actions": ["<action>", ...]}\n'
        )
        if failures:
            content += (
                '\nPrevious attempts were rejected:\n'
                + '\n'.join(f'  - {f}' for f in failures)
                + '\nRewrite the explanation without the offending references.\n'
            )
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> Explanation:
        text = raw.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.lstrip().lower().startswith('json'):
                text = text.lstrip()[4:]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f'expected a JSON object with "cause" and "next_actions", got '
                f'{raw[:200]!r} ({exc})'
            ) from exc

        if not isinstance(payload, dict):
            raise ParseError(f'expected a JSON object, got {type(payload).__name__}')
        for key in ('cause', 'next_actions'):
            if key not in payload:
                raise ParseError(f'missing required key {key!r}')

        actions = payload['next_actions']
        if not isinstance(actions, list):
            raise ParseError('"next_actions" must be a list of strings')

        cause = str(payload['cause'])
        action_texts = tuple(str(a) for a in actions)
        referenced = tuple(extract_symbols(' '.join((cause,) + action_texts)))
        return Explanation(
            cause=cause,
            next_actions=action_texts,
            symbols_referenced=referenced,
        )

    def validate(self, candidate: Explanation, context: Any) -> Verdict:
        """
        Reject explanations citing ws3 symbols that do not exist.

        :param candidate: The parsed explanation.
        :param context: Optional iterable of module names to check against.
            Defaults to the standard ws3 modules.
        """
        modules = context if context else WS3_MODULES
        known = known_symbols(modules)

        fabricated = []
        for symbol in candidate.symbols_referenced:
            parts = symbol.split('.')
            leaf = parts[-1]
            # A reference counts as a ws3 reference if any component names
            # something real in ws3. Once it does, the *leaf* must also be real.
            #
            # The looser "root or leaf is known" rule was wrong and this is the
            # case that proves it: `Problem.get_objective_value` has a genuine
            # root and an invented method, which is precisely the defect class
            # this capability exists to catch. A reference with no ws3 component
            # at all (`pandas.DataFrame`) is not our business and is ignored.
            touches_ws3 = any(part in known for part in parts[:-1])
            if touches_ws3 and leaf not in known:
                fabricated.append(symbol)

        if fabricated:
            return Verdict.invalid(
                'the explanation references ws3 names that do not exist: '
                + ', '.join(sorted(set(fabricated)))
            )

        if not candidate.cause.strip():
            return Verdict.invalid('the explanation has an empty "cause"')

        if not candidate.next_actions:
            return Verdict.invalid('the explanation suggests no next actions')

        return Verdict.valid()
