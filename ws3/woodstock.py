"""
The Woodstock input data format contract, and what ws3 does with it.

The Woodstock format is deliberately open ended. A model instance declares its
own set of themes, in its own order, with its own stratification variable codes
within each theme, and the LANDSCAPE section is the authoritative source for all
of it. Much of ws3's internal complexity follows from that flexibility.

ws3 implements an essential subset of the format -- enough to have carried real
projects for years -- but the boundary of that subset was previously undocumented
and unenforced. Keywords outside it are not rejected; they are *ignored*. A
dataset can declare an OPTIMIZE section, or use ``*ACTIONSERIES``, and import
without complaint, producing a model that is quietly not the model that was
written. No error, wrong answer.

This module exists to make that boundary visible:

- :py:func:`contract` loads the machine-readable keyword contract shipped as
  package data.
- :py:func:`lint_dataset` reports which parts of a dataset ws3 will not read.

Linting is advisory and reads nothing but the files. It never mutates a model,
and it is not required in order to import one.

Two ws3 divergences from Woodstock are recorded in the contract rather than
treated as defects:

**Time unit.** Woodstock measures stand age and action timing in periods; ws3
measures them in years. Any keyword documented as taking a number of periods --
``_LOCK`` most visibly -- is therefore not directly comparable between the two.

**Theme indexing.** Woodstock counts themes from one and writes ``_THn``
accordingly; ws3 stores themes zero-indexed, so ``_THn`` is ws3 theme ``n-1``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Location of the contract, shipped as package data.
CONTRACT_PATH = Path(__file__).parent / 'data' / 'woodstock_format.yaml'

#: What ws3 does with each section: the importer, and whether it does anything.
#:
#: ``stub`` is called out separately from ``none`` because the failure mode
#: differs in a way that matters. A stub method exists and can be called
#: successfully, so a caller has every reason to believe the section was read.
#: It was not.
SECTION_SUPPORT: dict[str, tuple[str | None, str]] = {
    'Landscape': ('import_landscape_section', 'implemented'),
    'Areas': ('import_areas_section', 'implemented'),
    'Yields': ('import_yields_section', 'implemented'),
    'Transitions': ('import_transitions_section', 'implemented'),
    'Outputs': ('import_outputs_section', 'implemented'),
    'Constants': ('import_constants_section', 'implemented'),
    'Schedule': ('import_schedule_section', 'implemented'),
    'Actions': ('import_actions_section', 'partial'),
    'Control': ('import_control_section', 'stub'),
    'Optimize': ('import_optimize_section', 'stub'),
    'Graphics': ('import_graphics_section', 'stub'),
    'Lifespan': ('import_lifespan_section', 'stub'),
    'Regimes': (None, 'none'),
    'Reports': (None, 'none'),
    'Queue': (None, 'none'),
    'Allocation': (None, 'none'),
    'LpSchedule': (None, 'none'),
}

#: Keywords ws3 reads.
#:
#: Maintained here, by hand, rather than derived by searching the source for
#: literal tokens. That approach does not work: ``import_outputs_section``
#: recognises ``_AREA`` and ``_INVENT`` through a generic ``startswith('_')``
#: check, so there is no literal to find, and a token search reported two
#: working keywords as unsupported. Anything added to an importer must be added
#: here too; :py:mod:`tests.test_woodstock` guards the parts that can be checked.
SUPPORTED_KEYWORDS: frozenset[str] = frozenset({
    # import_landscape_section
    '*THEME', '*AGGREGATE',
    # import_areas_section, standard positional form
    '*A',
    # import_yields_section
    '*Y', '*YC', '*YT', '_AGE',
    # complex yield functions resolved by the yields importer
    '_SUM',
    # import_actions_section
    '*ACTION', '*OPERABLE', '*PARTIAL', '_TH', '@AGE', '@YLD',
    # import_transitions_section
    '*CASE', '*SOURCE', '*TARGET', '_LOCK', '_REPLACE', '_APPEND',
    # outputs buffer resolver, including the built-in inventory keywords
    '*OUTPUT', '*LEVEL', '*GROUP', '_AREA', '_INVENT',
})

#: Deliberate departures from Woodstock semantics, recorded so that they are not
#: mistaken for defects.
DIVERGENCES: dict[str, str] = {
    'time_unit': (
        'Woodstock measures stand age and action timing in periods. ws3 measures '
        'them in years. ForestModel.import_areas_section takes '
        'convert_periods_to_years and multiplies imported ages by the period '
        'length. Any keyword documented as taking a number of periods, _LOCK '
        'most visibly, is therefore not directly comparable between the two.'
    ),
    'theme_indexing': (
        'Woodstock counts themes from one and writes _THn accordingly. ws3 '
        'stores themes zero-indexed, so _THn refers to ws3 theme n-1.'
    ),
}

#: Tokens that look like keywords but are format punctuation rather than
#: keywords, and would otherwise be reported on every line of a mask.
_NOT_KEYWORDS = {'*', '?'}

#: A Woodstock keyword: a sigil followed by a name. Function-style keywords such
#: as ``_SUM(...)`` are matched without their parentheses.
_KEYWORD = re.compile(r'(?<![\w])([*_@][A-Za-z][A-Za-z0-9]*)')


@dataclass(frozen=True)
class Finding:
    """
    One thing ws3 will not read from a dataset.

    :param severity: ``error`` when data is silently dropped, ``warning`` when a
        keyword is unrecognised, ``info`` for advisory notes.
    :param section: Section identifier, e.g. ``Yields``.
    :param path: File the finding refers to, if any.
    :param line: 1-based line number, if the finding is line-specific.
    :param keyword: Keyword involved, if the finding is keyword-specific.
    :param message: What ws3 will do, stated plainly.
    """

    severity: str
    section: str
    message: str
    path: str | None = None
    line: int | None = None
    keyword: str | None = None

    def __str__(self) -> str:
        where = self.path or self.section
        if self.line is not None:
            where = f'{where}:{self.line}'
        return f'{self.severity}: {where}: {self.message}'


@lru_cache(maxsize=1)
def contract() -> Any:
    """
    Load the Woodstock format contract.

    Cached, because it is read-only reference data and parsing it repeatedly
    inside a lint loop would be wasteful.

    :raises FileNotFoundError: If the package data is missing, which means an
        incomplete installation rather than a user error.
    """
    import yaml

    if not CONTRACT_PATH.is_file():
        raise FileNotFoundError(
            f'The Woodstock format contract is missing from the installed '
            f'package (expected at {CONTRACT_PATH}). Reinstall ws3.'
        )
    with open(CONTRACT_PATH) as f:
        return yaml.safe_load(f)


def sections() -> Any:
    """Section identifiers mapped to their contract entry, e.g. file extension."""
    return contract()['meta']['sections']


def keywords() -> Any:
    """Every catalogued keyword, mapped to its contract entry."""
    return contract()['keywords']


def supported_keywords() -> frozenset[str]:
    """Keywords ws3 reads. See :py:data:`SUPPORTED_KEYWORDS`."""
    return SUPPORTED_KEYWORDS


def section_support(name: str) -> tuple[str | None, str]:
    """
    The importer for a section and whether it does anything.

    :return: ``(importer_name, status)`` where status is one of ``implemented``,
        ``partial``, ``stub`` or ``none``.
    """
    return SECTION_SUPPORT.get(name, (None, 'none'))


def canonical(token: str) -> str:
    """
    Reduce a keyword token to its contract spelling.

    Indexed keywords appear in real files with a number (``_TH1``) but are
    catalogued under a placeholder (``_THn``), and the contract stores the
    placeholder stripped. Normalising both sides means a keyword has one
    spelling everywhere, which is what makes the support set checkable.
    """
    return re.sub(r'^([*_@]TH)(?:n|\d+)$', r'\1', token)


def _scan_keywords(text: str) -> list[tuple[int, str]]:
    """
    Find keyword tokens in a section file, with line numbers.

    Comments are stripped first: Woodstock comments start with ``;`` and run to
    end of line, and prose in a comment should not be reported as a keyword.
    """
    found = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.partition(';')[0]
        if not line.strip():
            continue
        for match in _KEYWORD.finditer(line):
            token = match.group(1)
            if token in _NOT_KEYWORDS:
                continue
            found.append((number, canonical(token)))
    return found


def lint_dataset(
    model_path: str,
    model_name: str,
    sections_to_check: Iterable[str] | None = None,
) -> list[Finding]:
    """
    Report what ws3 will not read from a Woodstock dataset.

    Reads the section files directly. Nothing is imported, no model is built,
    and nothing is modified -- so this is safe to run before deciding whether to
    trust an import.

    Findings are ordered by severity, then by file, then by line.

    :param model_path: Directory holding the section files.
    :param model_name: Base file name shared by the section files.
    :param sections_to_check: Restrict to these section identifiers. Defaults to
        every section in the contract.
    :return: Findings, empty when ws3 reads everything present.
    """
    base = Path(model_path)
    all_sections = sections()
    wanted = set(sections_to_check) if sections_to_check else set(all_sections)
    known = keywords()
    supported = supported_keywords()

    findings: list[Finding] = []

    for name in sorted(wanted):
        spec = all_sections.get(name)
        if spec is None:
            continue
        extension = spec.get('file_extension')
        if not extension:
            continue

        path = base / f'{model_name}.{extension}'
        if not path.is_file():
            continue

        try:
            text = path.read_text(errors='replace')
        except OSError as exc:
            findings.append(Finding(
                severity='warning', section=name, path=str(path),
                message=f'could not be read: {exc}',
            ))
            continue

        support = section_support(name)[1]
        if support in ('stub', 'none'):
            importer = section_support(name)[0]
            detail = (
                'ws3 has no importer for it'
                if not importer else
                f'{importer} is a stub and imports nothing'
            )
            findings.append(Finding(
                severity='error', section=name, path=str(path),
                message=(
                    f'the {name.upper()} section is present but not imported: '
                    f'{detail}. Everything in this file is ignored.'
                ),
            ))
            # No point listing individual keywords in a file that is wholly
            # ignored; the section-level finding already says so.
            continue

        seen: set[str] = set()
        for line, token in _scan_keywords(text):
            if token in supported or token in seen:
                continue
            entry = known.get(token)
            if entry is None:
                # Unknown to the contract entirely. Most likely a theme code or
                # yield name that happens to start with a sigil, so this is
                # advisory rather than an error.
                continue
            seen.add(token)
            findings.append(Finding(
                severity='warning', section=name, path=str(path), line=line,
                keyword=token,
                message=(
                    f'{token} is a documented Woodstock keyword that ws3 does '
                    f'not import; it is ignored'
                ),
            ))

    order = {'error': 0, 'warning': 1, 'info': 2}
    findings.sort(key=lambda f: (order.get(f.severity, 3), f.path or '', f.line or 0))
    return findings


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as a readable report."""
    findings = list(findings)
    if not findings:
        return 'No findings: ws3 imports everything present in this dataset.'
    lines = [str(f) for f in findings]
    errors = sum(1 for f in findings if f.severity == 'error')
    warnings = sum(1 for f in findings if f.severity == 'warning')
    lines.append('')
    lines.append(f'{errors} section(s) not imported, {warnings} keyword(s) ignored.')
    return '\n'.join(lines)
