"""
Sphinx directives that generate the Woodstock support tables from the contract.

The point is that the documented subset cannot drift from the implemented one.
Everything here reads :py:mod:`ws3.woodstock` at build time: the section support
map, the supported-keyword set, and the shipped keyword contract. Nothing is
transcribed by hand.
"""

from __future__ import annotations

from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from ws3 import woodstock

#: Rendered in the status column, in the order sections are grouped.
STATUS_ORDER = ['implemented', 'partial', 'stub', 'none']

STATUS_TEXT = {
    'implemented': 'implemented',
    'partial': 'partial',
    'stub': 'stub — imports nothing',
    'none': 'no importer',
}


class _Generated(Directive):
    """A directive whose body is reStructuredText generated at build time."""

    has_content = False

    def lines(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self):
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1)
        node = self.state.parent
        self.state.nested_parse(
            StringList(self.lines(), source=source), self.content_offset, node)
        return []


class WoodstockSections(_Generated):
    """Table of every contract section and what ws3 does with it."""

    def lines(self) -> list[str]:
        sections = woodstock.sections()
        rows = []
        for name in sorted(
                sections,
                key=lambda n: (STATUS_ORDER.index(woodstock.section_support(n)[1]), n)):
            importer, status = woodstock.section_support(name)
            extension = sections[name].get('file_extension', '')
            rows.append((
                name,
                f'``.{extension}``' if extension else '',
                f'``{importer}``' if importer else '—',
                STATUS_TEXT[status],
            ))

        out = [
            '.. list-table::',
            '   :header-rows: 1',
            '   :widths: 20 12 38 30',
            '',
            '   * - Section',
            '     - File',
            '     - Importer',
            '     - Status',
        ]
        for row in rows:
            out.append(f'   * - {row[0]}')
            out.extend(f'     - {cell}' for cell in row[1:])
        return out


class WoodstockKeywords(_Generated):
    """Table of the keywords ws3 reads, with the section each belongs to."""

    def lines(self) -> list[str]:
        known = woodstock.keywords()
        rows = []
        for keyword in sorted(woodstock.supported_keywords()):
            entry = known.get(keyword, {})
            used_in = ', '.join(entry.get('used_in', [])) or '—'
            rows.append((f'``{keyword}``', used_in))

        out = [
            '.. list-table::',
            '   :header-rows: 1',
            '   :widths: 25 75',
            '',
            '   * - Keyword',
            '     - Sections it appears in',
        ]
        for keyword, used_in in rows:
            out.append(f'   * - {keyword}')
            out.append(f'     - {used_in}')
        return out


class WoodstockCoverage(_Generated):
    """One sentence of measured coverage, counted from the contract."""

    def lines(self) -> list[str]:
        catalogued = len(woodstock.keywords())
        supported = len(woodstock.supported_keywords())
        counts = dict.fromkeys(STATUS_ORDER, 0)
        for name in woodstock.sections():
            counts[woodstock.section_support(name)[1]] += 1
        return [
            f'The contract catalogues **{catalogued} keywords** across '
            f'**{len(woodstock.sections())} sections**. ws3 reads '
            f'**{supported}** of those keywords. Of the sections, '
            f'{counts["implemented"]} are implemented, {counts["partial"]} is '
            f'partial, {counts["stub"]} are stubs that import nothing, and '
            f'{counts["none"]} have no importer at all.',
        ]


class WoodstockDivergences(_Generated):
    """Definition list of the recorded, deliberate departures from Woodstock."""

    TITLES = {
        'time_unit': 'Time unit: periods versus years',
        'theme_indexing': 'Theme indexing: one-based versus zero-based',
    }

    def lines(self) -> list[str]:
        out: list[str] = []
        for key, text in woodstock.DIVERGENCES.items():
            out.append(self.TITLES.get(key, key))
            out.append(f'   {text}')
            out.append('')
        return out


def setup(app):
    app.add_directive('woodstock-sections', WoodstockSections)
    app.add_directive('woodstock-keywords', WoodstockKeywords)
    app.add_directive('woodstock-coverage', WoodstockCoverage)
    app.add_directive('woodstock-divergences', WoodstockDivergences)
    return {'version': '1', 'parallel_read_safe': True}
