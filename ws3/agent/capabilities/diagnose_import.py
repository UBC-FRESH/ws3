"""
Diagnose a failing Woodstock model import.

Oracle: the suggested fix is applied to a scratch copy and the failing section is
re-parsed. If it still fails, the suggestion is rejected.

This is the strongest oracle of the three: it does not check that the answer looks
plausible, it checks that the answer *works*. The cost is that each validation
attempt actually re-parses a file, which is why the retry budget here is smaller.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fresh_agent_core.capability import Capability, ParseError, Verdict

#: Woodstock section suffixes and the ForestModel method that imports each.
SECTION_IMPORTERS = {
    'lan': 'import_landscape_section',
    'are': 'import_areas_section',
    'yld': 'import_yields_section',
    'act': 'import_actions_section',
    'trn': 'import_transitions_section',
    'con': 'import_constants_section',
    'out': 'import_outputs_section',
}


@dataclass(frozen=True)
class ImportFailure:
    """
    A failing model import.

    :param model_path: Directory holding the Woodstock files.
    :param model_name: Base name shared by the section files.
    :param section: Section suffix that failed, e.g. ``'lan'``.
    :param error: The error text produced by the failed import.
    :param excerpt: The relevant portion of the offending file, if known.
    """

    model_path: str
    model_name: str
    section: str
    error: str
    excerpt: str = ''


@dataclass(frozen=True)
class Diagnosis:
    """A proposed repair."""

    cause: str
    original_line: str
    corrected_line: str


class DiagnoseImport(Capability[Diagnosis]):
    """Diagnose an import failure, validated by re-parsing with the fix applied."""

    name = 'diagnose_import'
    description = (
        'Diagnose why a Woodstock section failed to import and propose a corrected '
        'line. Validated by applying the correction to a scratch copy of the model '
        'and re-running the import: a suggestion that does not actually make the '
        'section parse is rejected.'
    )
    max_attempts = 2

    input_schema = {
        'type': 'object',
        'properties': {
            'model_path': {'type': 'string', 'description': 'Directory holding the model files.'},
            'model_name': {'type': 'string', 'description': 'Base name shared by section files.'},
            'section': {
                'type': 'string',
                'description': 'Failing section suffix.',
                'enum': sorted(SECTION_IMPORTERS),
            },
            'error': {'type': 'string', 'description': 'Error text from the failed import.'},
            'excerpt': {
                'type': 'string',
                'description': 'Relevant portion of the offending file, if known.',
            },
        },
        'required': ['model_path', 'model_name', 'section', 'error'],
    }

    def from_payload(self, payload: dict) -> ImportFailure:
        """Build an :py:class:`ImportFailure` from MCP tool arguments."""
        return ImportFailure(
            model_path=str(payload.get('model_path', '')),
            model_name=str(payload.get('model_name', '')),
            section=str(payload.get('section', '')),
            error=str(payload.get('error', '')),
            excerpt=str(payload.get('excerpt', '')),
        )

    def coerce_input(self, inputs: Any) -> ImportFailure:
        """
        Accept a dict as well as an :py:class:`ImportFailure`.

        No string shorthand here: this capability needs the model path and section
        to run its oracle at all, so there is no single field that could stand in
        for the rest.
        """
        if isinstance(inputs, ImportFailure):
            return inputs
        if isinstance(inputs, dict):
            return self.from_payload(inputs)
        raise TypeError(
            f'diagnose_import takes a dict or an ImportFailure; '
            f'got {type(inputs).__name__}'
        )

    def render(self, value: Diagnosis) -> str:
        """Render the diagnosis and the verified replacement line."""
        return (
            f'{value.cause}\n'
            f'\nReplace:\n  {value.original_line}\n'
            f'\nWith:\n  {value.corrected_line}\n'
            f'\n(verified: the section re-imports successfully with this change)'
        )

    def build_messages(
        self,
        inputs: ImportFailure,
        failures: tuple[str, ...],
    ) -> list[dict[str, str]]:
        content = (
            'A Woodstock model section failed to import into ws3. Diagnose it and '
            'propose a single corrected line.\n'
            '\n'
            f'Section: {inputs.section}\n'
            f'Error: {inputs.error}\n'
        )
        if inputs.excerpt:
            content += f'\nRelevant file content:\n{inputs.excerpt}\n'
        content += (
            '\nPropose a change to exactly one line. The corrected line must be a '
            'drop-in replacement for the original.\n'
            '\n'
            'Respond with a JSON object and nothing else:\n'
            '  {"cause": "<why it failed>", "original_line": "<exact text to '
            'replace>", "corrected_line": "<replacement text>"}\n'
        )
        if failures:
            content += (
                '\nPrevious attempts were rejected:\n'
                + '\n'.join(f'  - {f}' for f in failures)
                + '\nPropose a different correction.\n'
            )
        return [{'role': 'user', 'content': content}]

    def parse(self, raw: str) -> Diagnosis:
        text = raw.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.lstrip().lower().startswith('json'):
                text = text.lstrip()[4:]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f'expected a JSON object with "cause", "original_line" and '
                f'"corrected_line", got {raw[:200]!r} ({exc})'
            ) from exc

        if not isinstance(payload, dict):
            raise ParseError(f'expected a JSON object, got {type(payload).__name__}')
        for key in ('cause', 'original_line', 'corrected_line'):
            if key not in payload:
                raise ParseError(f'missing required key {key!r}')

        return Diagnosis(
            cause=str(payload['cause']),
            original_line=str(payload['original_line']),
            corrected_line=str(payload['corrected_line']),
        )

    def validate(self, candidate: Diagnosis, context: Any) -> Verdict:
        """
        Apply the fix to a scratch copy and re-import.

        :param candidate: The proposed repair.
        :param context: The :py:class:`ImportFailure` being diagnosed.
        """
        if not isinstance(context, ImportFailure):
            return Verdict.invalid(
                'No ImportFailure was supplied as context, so the fix cannot be '
                'tested. This is a caller error, not a model error.'
            )

        if candidate.original_line == candidate.corrected_line:
            return Verdict.invalid(
                'the corrected line is identical to the original, so it changes nothing'
            )

        importer = SECTION_IMPORTERS.get(context.section)
        if importer is None:
            return Verdict.invalid(
                f'unknown section {context.section!r}; expected one of '
                f'{", ".join(sorted(SECTION_IMPORTERS))}'
            )

        source = Path(context.model_path)
        if not source.is_dir():
            return Verdict.invalid(
                f'model path {context.model_path!r} does not exist, so the fix '
                f'cannot be tested'
            )

        with tempfile.TemporaryDirectory(prefix='ws3-diagnose-') as tmp:
            scratch = Path(tmp) / source.name
            shutil.copytree(source, scratch)

            target = scratch / f'{context.model_name}.{context.section}'
            if not target.is_file():
                return Verdict.invalid(f'section file {target.name!r} not found')

            text = target.read_text()
            if candidate.original_line not in text:
                return Verdict.invalid(
                    f'the line to replace was not found in {target.name}: '
                    f'{candidate.original_line!r}'
                )

            target.write_text(text.replace(candidate.original_line, candidate.corrected_line))

            outcome = _try_import(scratch, context.model_name, importer)

        if outcome is None:
            return Verdict.valid()
        return Verdict.invalid(f'the section still fails after the fix: {outcome}')


def _try_import(model_path: Path, model_name: str, importer: str) -> str | None:
    """
    Re-run the failing import against a patched copy.

    :return: ``None`` if the import succeeded, otherwise the error text.
    """
    from ws3.forest import ForestModel

    try:
        fm = ForestModel(
            model_name=model_name,
            model_path=str(model_path),
            base_year=2020,
        )
        # Landscape defines the themes every other section depends on, so it is
        # imported first unless it is itself the section under test.
        if importer != 'import_landscape_section':
            fm.import_landscape_section()
        getattr(fm, importer)()
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    return None
