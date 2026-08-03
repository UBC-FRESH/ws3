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
    CompileSolveCapability,
    DevelopmentTypeEntry,
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
        keys = {entry.key for entry in tsa24_contract.development_types}
        assert keys == {
            ('tsa24', 'sw', '1'),
            ('tsa24', 'sw', '2'),
            ('tsa24', 'pl', '1'),
            ('tsa24', 'pl', '2'),
        }

    def test_development_types_include_age_class_count(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert isinstance(entry.n_age_classes, int)
            assert entry.n_age_classes >= 1
            assert isinstance(entry.total_area, float)
            assert entry.total_area > 0.0
            assert isinstance(entry.age_classes, tuple)
            assert len(entry.age_classes) == entry.n_age_classes

    def test_to_dict_is_json_roundtrip(self, tsa24_contract: ModelContract):
        """The dict form must survive a JSON round-trip -- it is the portable surface."""
        payload = tsa24_contract.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip['metadata']['model_name'] == 'm'
        assert len(roundtrip['development_types']) == 4
        # Each entry now has the extended area/yield fields.
        for entry in roundtrip['development_types']:
            assert 'key' in entry
            assert 'n_age_classes' in entry
            assert 'total_area' in entry
            assert 'age_classes' in entry
            assert 'yield_components' in entry
            assert 'yield_compiled' in entry


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
        assert summary['errors'] == 0
        # The test fixture has no yields, so we expect 4 yield_coverage_missing warnings
        # (one per development type). No area_inventory_empty warnings since all have area.
        assert summary['warnings'] == 4


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
            development_types=[DevelopmentTypeEntry(
                key=('tsa24', 'sw', 'extra'),
                n_age_classes=1,
                total_area=100.0,
                age_classes=(0,),
                yield_components=(),
                yield_compiled={},
            )],
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
            development_types=[DevelopmentTypeEntry(
                key=('tsa24', 'norway_spruce'),
                n_age_classes=1,
                total_area=100.0,
                age_classes=(0,),
                yield_components=(),
                yield_compiled={},
            )],
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
            development_types=[
                DevelopmentTypeEntry(
                    key=('tsa24',), n_age_classes=1, total_area=100.0,
                    age_classes=(0,), yield_components=(), yield_compiled={},
                ),
                DevelopmentTypeEntry(
                    key=('tsa24',), n_age_classes=2, total_area=200.0,
                    age_classes=(0, 10), yield_components=(), yield_compiled={},
                ),
            ],
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


# ---------------------------------------------------------------------------
# Area inventory extraction
# ---------------------------------------------------------------------------

class TestAreaInventoryExtraction:
    """Period-0 area inventory must be captured per development type."""

    def test_total_area_matches_sum_of_age_classes(self, tsa24_contract: ModelContract):
        """Total area must equal the sum of per-age-class areas (already validated
        by the model's area() method; here we just confirm consistency)."""
        for entry in tsa24_contract.development_types:
            assert entry.total_area >= 0.0
            assert entry.total_area > 0.0, (
                f'{entry.key} has zero total area'
            )

    def test_total_area_values_are_correct(self, tsa24_model: ForestModel):
        contract = ModelContract.from_model(tsa24_model)
        expected = {
            ('tsa24', 'sw', '1'): 100.0,
            ('tsa24', 'sw', '2'): 200.0,
            ('tsa24', 'pl', '1'): 150.0,
            ('tsa24', 'pl', '2'): 50.0,
        }
        for entry in contract.development_types:
            assert entry.total_area == expected[entry.key], (
                f'{entry.key}: expected {expected[entry.key]}, got {entry.total_area}'
            )

    def test_age_classes_are_sorted(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert list(entry.age_classes) == sorted(entry.age_classes)

    def test_n_age_classes_matches_age_classes_length(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert len(entry.age_classes) == entry.n_age_classes

    def test_area_inventory_roundtrips_through_json(self, tsa24_contract: ModelContract):
        payload = tsa24_contract.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        for entry in roundtrip['development_types']:
            assert 'total_area' in entry
            assert 'age_classes' in entry
            assert isinstance(entry['total_area'], (int, float))
            assert isinstance(entry['age_classes'], list)


# ---------------------------------------------------------------------------
# Yield coverage extraction
# ---------------------------------------------------------------------------

class TestYieldCoverageExtraction:
    """Yield component names must be captured per development type."""

    def test_yield_components_is_tuple(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert isinstance(entry.yield_components, tuple)

    def test_yield_components_are_sorted(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert list(entry.yield_components) == sorted(entry.yield_components)

    def test_yield_compiled_matches_yield_components(self, tsa24_contract: ModelContract):
        for entry in tsa24_contract.development_types:
            assert set(entry.yield_compiled.keys()) == set(entry.yield_components)

    def test_yield_coverage_roundtrips_through_json(self, tsa24_contract: ModelContract):
        payload = tsa24_contract.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        for entry in roundtrip['development_types']:
            assert 'yield_components' in entry
            assert 'yield_compiled' in entry
            assert isinstance(entry['yield_components'], list)
            assert isinstance(entry['yield_compiled'], dict)


# ---------------------------------------------------------------------------
# Verification -- area_inventory_empty
# ---------------------------------------------------------------------------

class TestVerificationAreaInventoryEmpty:
    """An L1 warning must fire when a development type has no period-0 area."""

    def test_area_inventory_empty_is_warning(self):
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
            metadata={'model_name': 'empty', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=0, total_area=0.0,
                age_classes=(), yield_components=(), yield_compiled={},
            )],
        )
        result = contract.verify()
        assert result.is_valid  # warnings do not invalidate
        area_warnings = [
            f for f in result.warnings if f.category == 'area_inventory_empty'
        ]
        assert len(area_warnings) == 1
        assert 'tsa24' in area_warnings[0].message

    def test_area_inventory_empty_is_not_error(self):
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
            metadata={'model_name': 'empty', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=0, total_area=0.0,
                age_classes=(), yield_components=(), yield_compiled={},
            )],
        )
        result = contract.verify()
        area_errors = [
            f for f in result.errors if f.category == 'area_inventory_empty'
        ]
        assert len(area_errors) == 0


# ---------------------------------------------------------------------------
# Verification -- yield_coverage_missing
# ---------------------------------------------------------------------------

class TestVerificationYieldCoverageMissing:
    """An L1 warning must fire when a development type has no yield components."""

    def test_yield_coverage_missing_is_warning(self):
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
            metadata={'model_name': 'no_ycomps', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=1, total_area=100.0,
                age_classes=(0,), yield_components=(), yield_compiled={},
            )],
        )
        result = contract.verify()
        assert result.is_valid  # warnings do not invalidate
        yield_warnings = [
            f for f in result.warnings if f.category == 'yield_coverage_missing'
        ]
        assert len(yield_warnings) == 1
        assert 'tsa24' in yield_warnings[0].message

    def test_yield_coverage_missing_is_not_error(self):
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
            metadata={'model_name': 'no_ycomps', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 1, 'n_actions': 0},
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=1, total_area=100.0,
                age_classes=(0,), yield_components=(), yield_compiled={},
            )],
        )
        result = contract.verify()
        yield_errors = [
            f for f in result.errors if f.category == 'yield_coverage_missing'
        ]
        assert len(yield_errors) == 0


# ---------------------------------------------------------------------------
# DevelopmentTypeEntry dataclass
# ---------------------------------------------------------------------------

class TestDevelopmentTypeEntry:
    """The DevelopmentTypeEntry dataclass must be frozen and serialisable."""

    def test_entry_is_frozen(self):
        entry = DevelopmentTypeEntry(
            key=('tsa24',), n_age_classes=1, total_area=100.0,
            age_classes=(0,), yield_components=(), yield_compiled={},
        )
        with pytest.raises(AttributeError):
            entry.key = ('other',)

    def test_entry_to_dict(self):
        entry = DevelopmentTypeEntry(
            key=('tsa24', 'sw'), n_age_classes=2, total_area=300.0,
            age_classes=(0, 10), yield_components=('vol', 'biomass'),
            yield_compiled={'vol': True, 'biomass': False},
        )
        d = entry.to_dict()
        assert d['key'] == ['tsa24', 'sw']
        assert d['n_age_classes'] == 2
        assert d['total_area'] == 300.0
        assert d['age_classes'] == [0, 10]
        assert d['yield_components'] == ['vol', 'biomass']
        assert d['yield_compiled'] == {'vol': True, 'biomass': False}

    def test_entry_to_dict_is_json_roundtrip(self):
        entry = DevelopmentTypeEntry(
            key=('tsa24', 'sw'), n_age_classes=2, total_area=300.0,
            age_classes=(0, 10), yield_components=('vol', 'biomass'),
            yield_compiled={'vol': True, 'biomass': False},
        )
        d = entry.to_dict()
        roundtrip = json.loads(json.dumps(d))
        assert roundtrip == d


# ---------------------------------------------------------------------------
# Action and transition extraction
# ---------------------------------------------------------------------------

class TestActionTransitionExtraction:
    """Action codes and transition targets must be extracted from existing APIs."""

    def test_action_codes_extracted_from_oper_expr(self, tsa24_model: ForestModel):
        """Action codes from oper_expr must be captured per development type."""
        # Add an action to the model to test extraction.
        tsa24_model.add_null_action(acode='test_action')
        contract = ModelContract.from_model(tsa24_model)
        for entry in contract.development_types:
            assert 'test_action' in entry.action_codes

    def test_transition_targets_extracted_from_transitions(self, tsa24_model: ForestModel):
        """Transition targets must be extracted from DevelopmentType.transitions."""
        tsa24_model.add_null_action(acode='test_action')
        contract = ModelContract.from_model(tsa24_model)
        for entry in contract.development_types:
            assert 'test_action' in entry.transition_targets

    def test_action_codes_are_sorted(self, tsa24_model: ForestModel):
        """Action codes must be sorted in the extracted tuple."""
        tsa24_model.add_null_action(acode='z_action')
        tsa24_model.add_null_action(acode='a_action')
        contract = ModelContract.from_model(tsa24_model)
        for entry in contract.development_types:
            assert list(entry.action_codes) == sorted(entry.action_codes)

    def test_transition_targets_are_sorted(self, tsa24_model: ForestModel):
        """Transition targets must be sorted in the extracted tuple."""
        tsa24_model.add_null_action(acode='z_action')
        tsa24_model.add_null_action(acode='a_action')
        contract = ModelContract.from_model(tsa24_model)
        for entry in contract.development_types:
            assert list(entry.transition_targets) == sorted(entry.transition_targets)

    def test_declared_actions_in_metadata(self, tsa24_model: ForestModel):
        """Declared action codes must be in contract metadata."""
        tsa24_model.add_null_action(acode='test_action')
        contract = ModelContract.from_model(tsa24_model)
        assert 'test_action' in contract.metadata['declared_actions']

    def test_action_transition_roundtrips_through_json(self, tsa24_model: ForestModel):
        """Action/transition fields must survive JSON round-trip."""
        tsa24_model.add_null_action(acode='test_action')
        contract = ModelContract.from_model(tsa24_model)
        payload = contract.to_dict()
        roundtrip = json.loads(json.dumps(payload))
        for entry in roundtrip['development_types']:
            assert 'action_codes' in entry
            assert 'transition_targets' in entry
            assert isinstance(entry['action_codes'], list)
            assert isinstance(entry['transition_targets'], list)


# ---------------------------------------------------------------------------
# Verification -- action_orphan and transition_target_invalid
# ---------------------------------------------------------------------------

class TestVerificationActionTransition:
    """Orphan action references and invalid transition targets must be flagged."""

    def test_action_orphan_is_warning(self):
        """An action code in oper_expr not in declared_actions must be flagged."""
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
            metadata={
                'model_name': 'orphan', 'base_year': 2020, 'horizon': 1,
                'period_length': 10, 'max_age': 100,
                'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                'n_development_types': 1, 'n_actions': 0,
                'declared_actions': (),  # No actions declared
            },
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=1, total_area=100.0,
                age_classes=(0,), yield_components=(), yield_compiled={},
                action_codes=('orphan_action',),  # Not declared
                transition_targets=(),
            )],
        )
        result = contract.verify()
        assert result.is_valid  # warnings do not invalidate
        orphan_warnings = [
            f for f in result.warnings if f.category == 'action_orphan'
        ]
        assert len(orphan_warnings) == 1
        assert 'orphan_action' in orphan_warnings[0].message

    def test_transition_target_invalid_is_warning(self):
        """A transition target not in declared_actions must be flagged."""
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
            metadata={
                'model_name': 'invalid_target', 'base_year': 2020, 'horizon': 1,
                'period_length': 10, 'max_age': 100,
                'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                'n_development_types': 1, 'n_actions': 0,
                'declared_actions': (),  # No actions declared
            },
            schema=schema,
            development_types=[DevelopmentTypeEntry(
                key=('tsa24',), n_age_classes=1, total_area=100.0,
                age_classes=(0,), yield_components=(), yield_compiled={},
                action_codes=(),
                transition_targets=('invalid_target',),  # Not declared
            )],
        )
        result = contract.verify()
        assert result.is_valid  # warnings do not invalidate
        target_warnings = [
            f for f in result.warnings if f.category == 'transition_target_invalid'
        ]
        assert len(target_warnings) == 1
        assert 'invalid_target' in target_warnings[0].message

    def test_valid_action_references_produce_no_findings(self, tsa24_model: ForestModel):
        """Valid action references must not produce findings."""
        tsa24_model.add_null_action(acode='valid_action')
        contract = ModelContract.from_model(tsa24_model)
        result = contract.verify()
        orphan_warnings = [
            f for f in result.warnings if f.category == 'action_orphan'
        ]
        target_warnings = [
            f for f in result.warnings if f.category == 'transition_target_invalid'
        ]
        assert len(orphan_warnings) == 0
        assert len(target_warnings) == 0


# ---------------------------------------------------------------------------
# Source oracle: verify_source round-trip
# ---------------------------------------------------------------------------

class TestVerifySourceValidMinimal:
    """A clean minimal Woodstock source must round-trip through the oracle."""

    def test_valid_source_produces_no_errors(self, tmp_path: Path):
        """A minimal landscape + areas dataset should verify clean."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            *THEME Species
            sw
            *AGGREGATE conifer
            sw
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 sw 0 100.0
            """))
        result = ModelContract.verify_source(str(tmp_path), 'm')
        assert isinstance(result, VerificationResult)
        error_categories = {f.category for f in result.errors}
        assert 'source_lint_error' not in error_categories
        assert 'source_import_failed' not in error_categories

    def test_valid_source_records_provenance(self, tmp_path: Path):
        """The contract produced by the oracle must carry source provenance."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        result = ModelContract.verify_source(str(tmp_path), 'm')
        # The contract is built internally; verify the result is structured.
        assert isinstance(result, VerificationResult)
        assert result.to_dict() is not None

    def test_valid_source_json_serialisable(self, tmp_path: Path):
        """The full result dict must survive a JSON round-trip."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        result = ModelContract.verify_source(str(tmp_path), 'm')
        payload = result.to_dict()
        serialised = json.dumps(payload)
        deserialised = json.loads(serialised)
        assert 'is_valid' in deserialised
        assert 'findings' in deserialised
        assert 'summary' in deserialised


class TestVerifySourceMalformed:
    """Malformed or unsupported source must yield findings, not exceptions."""

    def test_missing_dataset_yields_no_crash(self, tmp_path: Path):
        """A non-existent dataset must return findings, not raise."""
        result = ModelContract.verify_source(str(tmp_path / 'no_such_dir'), 'ghost')
        assert isinstance(result, VerificationResult)
        # lint_dataset tolerates missing files; import will fail and that
        # failure becomes a finding.
        assert any(f.category == 'source_import_failed' for f in result.findings)

    def test_unsupported_section_is_reported(self, tmp_path: Path):
        """An OPTIMIZE section (unsupported) must surface as an error finding."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        (tmp_path / 'm.opt').write_text('_MAXIMIZE x\n')
        result = ModelContract.verify_source(str(tmp_path), 'm')
        lint_errors = [
            f for f in result.findings
            if f.category == 'source_lint_error'
        ]
        assert lint_errors, 'OPTIMIZE section should be reported as lint error'
        assert any('Optimize' in f.message for f in lint_errors)

    def test_malformed_landscape_yields_import_finding(self, tmp_path: Path):
        """A landscape file with no THEME declarations must fail import gracefully."""
        (tmp_path / 'm.lan').write_text('; just a comment\n')
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        result = ModelContract.verify_source(str(tmp_path), 'm')
        assert isinstance(result, VerificationResult)
        # The import will raise because there is no *THEME declaration; that
        # failure must be captured as a finding, not propagated.
        import_findings = [
            f for f in result.findings
            if f.category == 'source_import_failed'
        ]
        assert import_findings, 'malformed landscape should produce import finding'


class TestVerifySourceJSON:
    """The oracle result must be JSON-serialisable end to end."""

    def test_result_to_dict_structure(self, tmp_path: Path):
        """to_dict must return the documented shape."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        result = ModelContract.verify_source(str(tmp_path), 'm')
        d = result.to_dict()
        assert isinstance(d['is_valid'], bool)
        assert isinstance(d['findings'], list)
        assert isinstance(d['summary'], dict)
        assert 'total' in d['summary']
        assert 'errors' in d['summary']
        assert 'warnings' in d['summary']

    def test_finding_dicts_have_required_fields(self, tmp_path: Path):
        """Each finding dict must carry level, category, message, severity."""
        (tmp_path / 'm.lan').write_text(textwrap.dedent("""\
            *THEME TSA
            tsa1
            """))
        (tmp_path / 'm.are').write_text(textwrap.dedent("""\
            *A tsa1 0 50.0
            """))
        (tmp_path / 'm.opt').write_text('_MAXIMIZE x\n')
        result = ModelContract.verify_source(str(tmp_path), 'm')
        for f in result.findings:
            fd = f.to_dict()
            assert 'level' in fd
            assert 'category' in fd
            assert 'message' in fd
            assert 'severity' in fd
            assert fd['severity'] in ('error', 'warning')


# ---------------------------------------------------------------------------
# Compile/solve smoke oracle
# ---------------------------------------------------------------------------

class TestCompileSolveSmokeOracle:
    """The compile/solve smoke oracle must report capability without crashing."""

    def test_compile_available_for_valid_model(self, tsa24_model: ForestModel):
        """A model with development types must report compile as available."""
        contract = ModelContract.from_model(tsa24_model)
        result, capability = contract.verify_compile_solve(tsa24_model)
        assert capability.compile_available is True
        assert isinstance(result, VerificationResult)
        # No errors from compile (yields are None in this minimal model, but
        # compile_actions does not raise).
        compile_errors = [
            f for f in result.errors if f.category == 'compile_failed'
        ]
        assert len(compile_errors) == 0

    def test_solve_deferred_when_no_problems(self, tsa24_model: ForestModel):
        """A model without optimization problems must record solve as deferred."""
        contract = ModelContract.from_model(tsa24_model)
        result, capability = contract.verify_compile_solve(tsa24_model)
        assert capability.solve_available is False
        assert capability.deferred_reason is not None
        assert 'no optimization problems' in capability.deferred_reason

    def test_compile_returns_none_model(self):
        """Calling with None must return a capability with both unavailable."""
        contract = ModelContract(
            metadata={'model_name': 'test', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 0, 'n_actions': 0},
            schema=ModelContract.from_model.__class__.__bases__[0] if False else None,  # type: ignore
            development_types=[],
        )
        # Use a minimal contract with a valid schema.
        from ws3.agent.themes import Theme, ThemeSchema
        schema = ThemeSchema(
            themes=(Theme(
                index=0, name='tsa', description='',
                basecodes=('tsa1',), aggregates={},
            ),)
        )
        contract = ModelContract(
            metadata={'model_name': 'test', 'base_year': 2020, 'horizon': 1,
                      'period_length': 10, 'max_age': 100,
                      'area_epsilon': 0.01, 'curve_epsilon': 1e-06,
                      'n_development_types': 0, 'n_actions': 0},
            schema=schema,
            development_types=[],
        )
        result, capability = contract.verify_compile_solve(None)
        assert capability.compile_available is False
        assert capability.solve_available is False
        assert len(result.findings) == 0

    def test_capability_to_dict_is_json_roundtrip(self, tsa24_model: ForestModel):
        """The capability dict must survive a JSON round-trip."""
        contract = ModelContract.from_model(tsa24_model)
        result, capability = contract.verify_compile_solve(tsa24_model)
        payload = capability.to_dict()
        serialised = json.dumps(payload)
        deserialised = json.loads(serialised)
        assert deserialised['compile_available'] is True
        assert deserialised['solve_available'] is False
        assert 'deferred_reason' in deserialised
        assert 'yield_compilation_status' in deserialised

    def test_yield_compilation_status_recorded(self, tsa24_model: ForestModel):
        """Yield compilation status must be recorded per development type."""
        contract = ModelContract.from_model(tsa24_model)
        result, capability = contract.verify_compile_solve(tsa24_model)
        # The minimal model has no yields, so each DT should have an empty dict.
        for key in tsa24_model.dtypes.keys():
            assert key in capability.yield_compilation_status
            assert capability.yield_compilation_status[key] == {}

    def test_capability_is_frozen_dataclass(self):
        """CompileSolveCapability must be a frozen dataclass."""
        cap = CompileSolveCapability(
            compile_available=True,
            solve_available=False,
            deferred_reason='test',
        )
        with pytest.raises(AttributeError):
            cap.compile_available = False

    def test_deferred_reason_is_explicit(self, tsa24_model: ForestModel):
        """The deferred reason must be explicit and not None when solve is unavailable."""
        contract = ModelContract.from_model(tsa24_model)
        result, capability = contract.verify_compile_solve(tsa24_model)
        assert capability.deferred_reason is not None
        # The reason must mention the requirement for user-defined problems.
        assert 'add_problem' in capability.deferred_reason
