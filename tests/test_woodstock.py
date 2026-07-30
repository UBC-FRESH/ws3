"""
Tests for the Woodstock format contract and the dataset linter.

The point of the linter is that ws3 reads a subset of the Woodstock format and
ignores the rest *silently*. A dataset can declare an OPTIMIZE section, import
without error, and produce a model that is not the model that was written. These
tests check that the linter says so, and -- just as importantly -- that it does
not cry wolf about keywords ws3 handles perfectly well.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ws3 import woodstock

MODEL_DIR = (Path(__file__).parent.parent / 'examples' / 'data'
             / 'woodstock_model_files_tsa24_clipped')
MODEL_NAME = 'tsa24_clipped'


class TestContract:
    def test_contract_loads(self):
        assert woodstock.contract()['keywords']

    def test_contract_ships_with_the_package(self):
        assert woodstock.CONTRACT_PATH.is_file()

    def test_contract_carries_no_vendor_prose(self):
        """
        The contract is interface facts only.

        The source documentation is vendor copyright; reproducing its prose in
        this repository would be a licensing problem, so the generator emits
        syntax skeletons and argument names and nothing else.
        """
        text = woodstock.CONTRACT_PATH.read_text()
        assert 'REMSOFT' not in text.upper()
        assert 'COPYRIGHT' not in text.upper()

    def test_every_section_declares_a_file_extension(self):
        for name, spec in woodstock.sections().items():
            assert spec.get('file_extension'), f'{name} has no file extension'

    def test_keywords_are_scoped_to_sections(self):
        """Most keywords say which section they belong to; some are global."""
        scoped = [k for k, v in woodstock.keywords().items() if v.get('used_in')]
        assert len(scoped) > 150


class TestSupportDeclarations:
    """
    ws3's support declarations must stay in step with ws3.

    They are hand-maintained because token searching does not work -- the outputs
    importer recognises _AREA and _INVENT through a generic prefix check. What
    *can* be checked mechanically is checked here.
    """

    def test_every_supported_section_names_a_real_importer(self):
        from ws3.forest import ForestModel
        for name, (importer, status) in woodstock.SECTION_SUPPORT.items():
            if importer is None:
                assert status == 'none', f'{name} has no importer but status {status}'
                continue
            assert hasattr(ForestModel, importer), f'{name}: no {importer}'

    def test_sections_with_an_importer_are_not_marked_absent(self):
        for name, (importer, status) in woodstock.SECTION_SUPPORT.items():
            if importer is not None:
                assert status != 'none', f'{name} has {importer} but is marked none'

    def test_every_section_in_the_contract_has_a_support_declaration(self):
        missing = set(woodstock.sections()) - set(woodstock.SECTION_SUPPORT)
        assert not missing, f'no ws3 support declared for: {sorted(missing)}'

    def test_supported_keywords_are_real_woodstock_keywords(self):
        """A typo here would silently suppress a finding forever."""
        known = set(woodstock.keywords())
        unknown = {k for k in woodstock.SUPPORTED_KEYWORDS if k not in known}
        assert not unknown, f'not in the contract: {sorted(unknown)}'

    def test_divergences_are_recorded(self):
        assert 'time_unit' in woodstock.DIVERGENCES
        assert 'periods' in woodstock.DIVERGENCES['time_unit']
        assert 'years' in woodstock.DIVERGENCES['time_unit']


@pytest.mark.skipif(not MODEL_DIR.is_dir(), reason='example model not available')
class TestLintRealDataset:
    @pytest.fixture(scope='class')
    def findings(self):
        return woodstock.lint_dataset(str(MODEL_DIR), MODEL_NAME)

    def test_reports_the_unread_optimize_section(self, findings):
        """
        The case that motivates the linter.

        import_optimize_section is a stub: calling it succeeds and imports
        nothing, so nothing else in ws3 will ever tell the user.
        """
        opt = [f for f in findings if f.section == 'Optimize']
        assert opt, 'OPTIMIZE section present but not reported'
        assert opt[0].severity == 'error'
        assert 'stub' in opt[0].message

    def test_reports_sections_with_no_importer_at_all(self, findings):
        reported = {f.section for f in findings if f.severity == 'error'}
        assert {'Queue', 'Reports'} <= reported

    def test_does_not_report_sections_ws3_reads(self, findings):
        """Landscape, Areas, Yields and Transitions all import correctly."""
        reported = {f.section for f in findings}
        assert not reported & {'Landscape', 'Areas', 'Yields', 'Transitions'}

    def test_no_false_positives_on_supported_keywords(self, findings):
        """
        Regression test.

        A hand-written support list initially omitted _AREA and _INVENT, which
        the outputs importer handles through a generic prefix check, so the
        linter reported two working keywords as ignored. A linter that cries
        wolf gets switched off.
        """
        flagged = {f.keyword for f in findings if f.keyword}
        assert not flagged & woodstock.SUPPORTED_KEYWORDS

    def test_findings_are_ordered_errors_first(self, findings):
        severities = [f.severity for f in findings]
        assert severities == sorted(severities, key=lambda s: {'error': 0, 'warning': 1}.get(s, 2))

    def test_findings_render_with_location(self, findings):
        assert all(MODEL_NAME in str(f) for f in findings)


class TestLintEdgeCases:
    def test_missing_dataset_yields_no_findings(self, tmp_path):
        """Absent files are not findings; there is nothing to be wrong about."""
        assert woodstock.lint_dataset(str(tmp_path), 'nothing_here') == []

    def test_unread_section_is_reported_even_when_trivial(self, tmp_path):
        (tmp_path / 'm.opt').write_text('; just a comment\n')
        findings = woodstock.lint_dataset(str(tmp_path), 'm')
        assert [f.section for f in findings] == ['Optimize']

    def test_comments_are_not_scanned_for_keywords(self, tmp_path):
        """A keyword named in a comment is prose, not a declaration."""
        (tmp_path / 'm.yld').write_text('; this model would use *YT if ws3 read it\n')
        assert woodstock.lint_dataset(str(tmp_path), 'm') == []

    def test_unsupported_keyword_in_a_read_section_is_reported(self, tmp_path):
        (tmp_path / 'm.are').write_text('*AA ? ? 0 100 aacode\n')
        findings = woodstock.lint_dataset(str(tmp_path), 'm')
        assert any(f.keyword == '*AA' for f in findings)
        assert all(f.line == 1 for f in findings)

    def test_each_unsupported_keyword_is_reported_once(self, tmp_path):
        (tmp_path / 'm.are').write_text('*AA a\n*AA b\n*AA c\n')
        findings = [f for f in woodstock.lint_dataset(str(tmp_path), 'm')
                    if f.keyword == '*AA']
        assert len(findings) == 1

    def test_restricting_sections(self, tmp_path):
        (tmp_path / 'm.opt').write_text('_MAXIMIZE x\n')
        (tmp_path / 'm.que').write_text('*SELECT harvest\n')
        only = woodstock.lint_dataset(str(tmp_path), 'm', sections_to_check=['Queue'])
        assert {f.section for f in only} == {'Queue'}

    def test_format_findings_is_explicit_when_clean(self):
        assert 'No findings' in woodstock.format_findings([])
