"""
Tests for the ws3 model contract: extraction and structural verification.

The contract is the typed, JSON-serialisable specification surface attached to
existing ``ForestModel`` / ``ThemeSchema`` APIs. It captures what a downstream
consumer (an optimiser, a report generator, an agent capability) needs to know
about a model *without* holding a reference to the model itself.

Verification returns findings, never raises, so ordinary model invalidity is
observable rather than fatal.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ws3.agent.themes import (
    SEVERITY_ERROR,
    ModelContract,
    VerificationFinding,
    VerificationResult,
    contract_for,
)
from ws3.forest import ForestModel


@pytest.fixture
def tsa24_model(tmp_path: Path) -> ForestModel:
    """
    Build a minimal but realistic ForestModel from a Woodstock-style landscape
    file. The model has two themes (TSA and species) and a handful of development
    types with area inventory.
    """
    (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
        *THEME Timber Supply Area (TSA)
        tsa24
        *THEME Leading tree species
        sw
        pl
        *AGGREGATE conifer
        sw pl
        *THEME Site quality
        1
        2
        """))
    (tmp_path / 'm.are').write_text(textwrap.dedent("""\
        *A tsa24 sw 1 0 100.0
        *A tsa24 sw 2 0 200.0
        *A tsa24 pl 1 0 150.0
        *A tsa24 pl 2 0 50.0
        """))
    fm = ForestModel(
        model_name='m',
        model_path=str(tmp_path),
        base_year=2020,
        horizon=10,
        period_length=10,
        max_age=100,
    )
    fm.import_landscape_section()
    fm.import_areas_section(convert_periods_to_years=10)
    return fm


@pytest.fixture
def tsa24_contract(tsa24_model: ForestModel) -> ModelContract:
    return ModelContract.from_model(tsa24_model)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestModelContractExtraction:
    """The contract must faithfully reproduce the model's structural surface."""

    def test_metadata_captures_scalar_properties(self, tsa24_contract: ModelContract):
        md = tsa24_contract.metadata
        assert md['model_name'] == 'm'
        assert md['base_year'] == 2020
        assert md['horizon'] == 10
        assert md['period_length'] == 10
        assert md['max_age'] == 100
        assert md['n_development_types'] == 4
        assert md['n_actions'] == 0

    def test_schema_carries_all_themes(self, tsa24_contract: ModelContract):
        themes = tsa24_contract.schema.themes
        assert len(themes) == 3
        # import_landscape_section names themes theme0, theme1, ...
        assert themes[0].name == 'theme0'
        assert themes[1].name == 'theme1'
        assert themes[2].name == 'theme2'

    def test_schema_captures_aggregates(self, tsa24_contract: ModelContract):
        species = tsa24_contract.schema.themes[1]
        assert 'conifer' in species.aggregates
        assert set(species.aggregates['conifer']) == {'sw', 'pl'}

    def test_development_types_count_and_keys(self, tsa24_contract: ModelContract):
        keys = {k for k, _ in tsa24_contract.development_types}
        assert keys == {
            ('tsa24', 'sw', '1'),
            ('tsa24', 'sw', '2'),
            ('tsa24', 'pl', '1'),
            ('tsa24', 'pl', '2'),
        }

    def test_development_types_include_age_class_count(self, tsa24_contract: ModelContract):
        for _key, n_ages in tsa24_contract.development_types:
            assert isinstance(n_ages, int)
            assert n_ages >= 1

    def test_to_dict_is_json_roundtrip(self, tsa24_contract: ModelContract):
        """The dict form must survive a JSON round-trip -- it is the portable surface."""
        payload = tsa24_contract.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip['metadata']['model_name'] == 'm'
        assert len(roundtrip['development_types']) == 4


# ---------------------------------------------------------------------------
# Verification -- happy path
# ---------------------------------------------------------------------------

class TestVerificationHappyPath:
    """A well-formed model must verify clean."""

    def test_valid_model_has_no_findings(self, tsa24_contract: ModelContract):
        result = tsa24_contract.verify()
        assert result.is_valid
        assert result.errors == []

    def test_verify_returns_verification_result(self, tsa24_contract: ModelContract):
        result = tsa24_contract.verify()
        assert isinstance(result, VerificationResult)

    def test_to_dict_summary_counts(self, tsa24_contract: ModelContract):
        result = tsa24_contract.verify()
        summary = result.to_dict()['summary']
        assert summary['total'] == 0
        assert summary['errors'] == 0
        assert summary['warnings'] == 0


# ---------------------------------------------------------------------------
# Verification -- L0 errors
# ---------------------------------------------------------------------------

class TestVerificationL0Errors:
    """L0 checks must catch structural defects and report them as errors."""

    def test_dtype_key_wrong_length_is_error(self):
        """
        A development-type key whose length does not match nthemes is structurally
        incoherent: no mask can address it.
        """
        from ws3.agent.themes import Theme, ThemeSchema

        schema = ThemeSchema(
            themes=(
                Theme(
                    index=0, name='tsa', description='',
                    basecodes=('tsa24',), aggregates={},
                ),
                Theme(
                    index=1, name='species', description='',
                    basecodes=('sw', 'pl'), aggregates={'conifer': ('sw', 'pl')},
                ),
            )
        )
        contract = ModelContract(
            metadata={'model_name': 'broken', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[(('tsa24', 'sw', 'extra'), 1)],
        )
        result = contract.verify()
        assert not result.is_valid
        categories = {f.category for f in result.errors}
        assert 'dtype_key_length' in categories

    def test_dtype_code_unknown_to_theme_is_error(self):
        """A code that does not belong to its theme position must be flagged."""
        from ws3.agent.themes import Theme, ThemeSchema

        schema = ThemeSchema(
            themes=(
                Theme(
                    index=0, name='tsa', description='',
                    basecodes=('tsa24',), aggregates={},
                ),
                Theme(
                    index=1, name='species', description='',
                    basecodes=('sw', 'pl'), aggregates={'conifer': ('sw', 'pl')},
                ),
            )
        )
        contract = ModelContract(
            metadata={'model_name': 'broken', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[(('tsa24', 'norway_spruce'), 1)],
        )
        result = contract.verify()
        assert not result.is_valid
        code_errors = [f for f in result.errors if f.category == 'dtype_code_known']
        assert len(code_errors) == 1
        assert 'norway_spruce' in code_errors[0].message
        assert 'species' in code_errors[0].message

    def test_theme_with_no_basecodes_is_error(self):
        """A theme without basecodes cannot select any development type."""
        from ws3.agent.themes import Theme, ThemeSchema

        schema = ThemeSchema(
            themes=(
                Theme(
                    index=0, name='tsa', description='',
                    basecodes=('tsa24',), aggregates={},
                ),
                Theme(
                    index=1, name='empty_theme', description='',
                    basecodes=(), aggregates={},
                ),
            )
        )
        contract = ModelContract(
            metadata={'model_name': 'broken', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 0, 'n_actions': 0},
            schema=schema,
            development_types=[],
        )
        result = contract.verify()
        assert not result.is_valid
        empty_errors = [f for f in result.errors if f.category == 'theme_has_basecodes']
        assert len(empty_errors) == 1
        assert 'empty_theme' in empty_errors[0].message

    def test_non_contiguous_theme_indices_are_error(self):
        """Theme indices must form a contiguous range starting at 0."""
        from ws3.agent.themes import Theme, ThemeSchema

        schema = ThemeSchema(
            themes=(
                Theme(
                    index=0, name='tsa', description='',
                    basecodes=('tsa24',), aggregates={},
                ),
                Theme(
                    index=2, name='skipped', description='',
                    basecodes=('sw',), aggregates={},
                ),
            )
        )
        contract = ModelContract(
            metadata={'model_name': 'broken', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 0, 'n_actions': 0},
            schema=schema,
            development_types=[],
        )
        result = contract.verify()
        assert not result.is_valid
        arity_errors = [f for f in result.errors if f.category == 'theme_arity']
        assert len(arity_errors) >= 1
        # Index 1 is missing from the contiguous range 0..1
        assert '1' in arity_errors[0].message


# ---------------------------------------------------------------------------
# Verification -- L1 warnings
# ---------------------------------------------------------------------------

class TestVerificationL1Warnings:
    """L1 checks surface as warnings, not errors."""

    def test_duplicate_dtype_key_is_warning_not_error(self):
        """Duplicates violate an invariant but do not block use of the contract."""
        from ws3.agent.themes import Theme, ThemeSchema

        schema = ThemeSchema(
            themes=(
                Theme(
                    index=0, name='tsa', description='',
                    basecodes=('tsa24',), aggregates={},
                ),
            )
        )
        contract = ModelContract(
            metadata={'model_name': 'dup', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[(('tsa24',), 1), (('tsa24',), 2)],
        )
        result = contract.verify()
        assert result.is_valid  # warnings do not invalidate
        dup_warnings = [f for f in result.warnings if f.category == 'dtype_duplicate_key']
        assert len(dup_warnings) == 1


# ---------------------------------------------------------------------------
# Integration: extraction + verification round-trip
# ---------------------------------------------------------------------------

class TestContractEndToEnd:
    """The full pipeline: extract, verify, serialise."""

    def test_valid_model_roundtrips_through_json(self, tsa24_model: ForestModel):
        contract = ModelContract.from_model(tsa24_model)
        result = contract.verify()
        assert result.is_valid
        payload = contract.to_dict()
        serialised = json.dumps(payload)
        deserialised = json.loads(serialised)
        assert deserialised['metadata']['n_development_types'] == 4
        assert deserialised['schema']['nthemes'] == 3
        assert len(deserialised['development_types']) == 4

    def test_contract_for_convenience_function(self, tsa24_model: ForestModel):
        """The convenience accessor must return the same contract as from_model."""
        a = contract_for(tsa24_model)
        b = ModelContract.from_model(tsa24_model)
        assert a.to_dict() == b.to_dict()

    def test_finding_to_dict_preserves_all_fields(self):
        f = VerificationFinding(
            level='L0',
            category='dtype_code_known',
            message='test message',
            severity=SEVERITY_ERROR,
        )
        d = f.to_dict()
        assert d['level'] == 'L0'
        assert d['category'] == 'dtype_code_known'
        assert d['message'] == 'test message'
        assert d['severity'] == SEVERITY_ERROR
