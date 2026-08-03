"""
Focused tests for ws3.agent.spec and ws3.agent.emitter.

Covers:
- JSON round-trip fidelity
- Deterministic serialized bytes
- Real ModelBuilder fresh import
- Existing-model isolation (spec does not mutate imported models)
- Explicit unsupported transition behavior
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ws3.agent.builder import BuildResult, ModelBuilder
from ws3.agent.emitter import emit_all, emit_outputs, emit_yields
from ws3.agent.spec import (
    ActionSpec,
    ModelSpec,
    ModelSpecError,
    OperableMask,
    ThemeSpec,
    TransitionSpec,
    UnsupportedTransitionError,
    YieldSpec,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_theme_spec():
    """Minimal single-theme spec with areas and yields (no actions/transitions)."""
    return ModelSpec(
        model_name='single',
        base_year=2000,
        horizon=10,
        period_length=10,
        max_age=100,
        themes=(ThemeSpec(name='theme0', description='main', basecodes=('A',)),),
        areas={('A',): {1: 100.0, 2: 200.0, 3: 50.0}},
        yields=(
            YieldSpec(
                mask=('A',),
                ytype='a',
                ynames=('BA', 'HWP'),
                points={'BA': [0.0, 10.0, 20.0, 30.0], 'HWP': [0.0, 1.0, 2.0, 3.0]},
            ),
        ),
    )


@pytest.fixture
def two_theme_spec():
    """Two-theme spec with multi-element tuple keys."""
    return ModelSpec(
        model_name='dual',
        base_year=2000,
        horizon=5,
        period_length=10,
        max_age=80,
        themes=(
            ThemeSpec(name='theme0', basecodes=('P',)),
            ThemeSpec(name='theme1', basecodes=('S',)),
        ),
        areas={
            ('P', 'S'): {1: 50.0, 2: 75.0},
            ('P', 'P'): {1: 25.0},
        },
        yields=(
            YieldSpec(
                mask=('P', 'S'),
                ytype='t',
                ynames=('V',),
                points={'V': [100.0, 200.0, 300.0]},
            ),
        ),
        transitions={
            'trans1': TransitionSpec(
                case='trans1',
                source=('P', 'S'),
                target=('P', 'P'),
                action='harvest',
                theme_replace='_TH1',
            ),
        },
    )


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_single_theme_round_trip(self, single_theme_spec):
        """Spec -> dict -> JSON string -> dict -> Spec preserves all fields."""
        d = single_theme_spec.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        spec2 = ModelSpec.from_dict(d2)

        assert spec2.model_name == single_theme_spec.model_name
        assert spec2.base_year == single_theme_spec.base_year
        assert spec2.horizon == single_theme_spec.horizon
        assert spec2.period_length == single_theme_spec.period_length
        assert spec2.max_age == single_theme_spec.max_age
        assert spec2.themes == single_theme_spec.themes
        assert spec2.areas == single_theme_spec.areas
        assert spec2.yields == single_theme_spec.yields
        assert spec2.actions == single_theme_spec.actions
        assert spec2.metadata == single_theme_spec.metadata

    def test_two_theme_tuple_keys_round_trip(self, two_theme_spec):
        """Multi-element tuple keys survive JSON serialization."""
        d = two_theme_spec.to_dict()
        spec2 = ModelSpec.from_dict(d)

        # Area keys should be tuples of strings
        assert ('P', 'S') in spec2.areas
        assert ('P', 'P') in spec2.areas
        assert spec2.areas[('P', 'S')] == {1: 50.0, 2: 75.0}

        # Yield mask should be a tuple
        assert spec2.yields[0].mask == ('P', 'S')

        # Action operable_masks keys should be tuples
        # (no actions in two_theme_spec, but verify structure)

        # Transition source/target should be tuples
        assert spec2.transitions['trans1'].source == ('P', 'S')
        assert spec2.transitions['trans1'].target == ('P', 'P')

    def test_json_keys_are_strings(self, single_theme_spec):
        """Top-level and structural dict keys are strings (JSON requirement).

        Note: area value dicts (age->area) have int keys, which is expected
        and correct - we only verify the spec structure keys.
        """
        d = single_theme_spec.to_dict()

        # Top-level keys must be strings
        for k in d.keys():
            assert isinstance(k, str)

        # Theme keys must be strings
        for theme in d['themes']:
            for k in theme.keys():
                assert isinstance(k, str)

        # Yield keys must be strings
        for yield_spec in d['yields']:
            for k in yield_spec.keys():
                assert isinstance(k, str)

        # Action keys must be strings
        for acode, action in d['actions'].items():
            assert isinstance(acode, str)
            for k in action.keys():
                assert isinstance(k, str)
            # operable_masks keys must be strings
            for mk in action['operable_masks'].keys():
                assert isinstance(mk, str)

        # Transition keys must be strings
        for tcode, trans in d['transitions'].items():
            assert isinstance(tcode, str)
            for k in trans.keys():
                assert isinstance(k, str)

    def test_nested_operable_masks_round_trip(self):
        """Action operable_masks with OperableMask entries round-trip correctly."""
        spec = ModelSpec(
            model_name='test',
            base_year=2000,
            horizon=10,
            period_length=10,
            max_age=100,
            themes=(ThemeSpec(name='t0', basecodes=('A', 'B')),),
            actions={
                'harvest': ActionSpec(
                    acode='harvest',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=0, max_age=50),
                        OperableMask(mask=('B',), min_age=30, max_age=100),
                    ),
                ),
            },
        )
        d = spec.to_dict()
        spec2 = ModelSpec.from_dict(d)
        assert spec2.actions['harvest'].operable_masks == (
            OperableMask(mask=('A',), min_age=0, max_age=50),
            OperableMask(mask=('B',), min_age=30, max_age=100),
        )


# ---------------------------------------------------------------------------
# Deterministic bytes
# ---------------------------------------------------------------------------

class TestDeterministicBytes:
    def test_emit_yields_deterministic(self, single_theme_spec):
        """Same spec produces identical bytes on every emit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = emit_yields(single_theme_spec, Path(tmpdir))
            content1 = path1.read_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            path2 = emit_yields(single_theme_spec, Path(tmpdir))
            content2 = path2.read_bytes()

        assert content1 == content2

    def test_emit_all_deterministic(self, single_theme_spec):
        """emit_all produces identical bytes across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = emit_all(single_theme_spec, Path(tmpdir))
            bytes1 = {k: v.read_bytes() for k, v in result1.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            result2 = emit_all(single_theme_spec, Path(tmpdir))
            bytes2 = {k: v.read_bytes() for k, v in result2.items()}

        assert bytes1 == bytes2

    def test_emit_yields_has_age_token(self, single_theme_spec):
        """Emitting age-based yields includes the age column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(single_theme_spec, Path(tmpdir))
            content = path.read_text()

        lines = [line for line in content.strip().split('\n') if line and not line.startswith('*') and not line.startswith('_')]
        assert len(lines) >= 1
        # Each data line should start with an integer age
        for line in lines:
            parts = line.split()
            assert parts[0].isdigit(), f'First token must be age, got: {parts[0]!r}'


# ---------------------------------------------------------------------------
# Real ModelBuilder fresh import
# ---------------------------------------------------------------------------

class TestBuilderFreshImport:
    def test_builder_creates_forest_model(self, single_theme_spec):
        """ModelBuilder can construct a ForestModel from a fresh spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ModelBuilder(single_theme_spec)
            result = builder.build(Path(tmpdir))

        assert isinstance(result, BuildResult)
        assert result.model is not None
        # Verify the model has themes (nthemes is a method)
        assert result.model.nthemes() >= 1

    def test_builder_emits_sections(self, single_theme_spec):
        """ModelBuilder emits section files to the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ModelBuilder(single_theme_spec)
            builder.build(Path(tmpdir))
            # Check that at least the yields file was emitted
            yld_files = list(Path(tmpdir).glob('*.yld'))
            assert len(yld_files) >= 1


# ---------------------------------------------------------------------------
# Builder rejects non-empty actions/transitions (regression)
# ---------------------------------------------------------------------------

class TestBuilderActionsSupported:
    """Actions are supported in the five-section implementation."""

    def test_action_spec_builds_forest_model(self, tmp_path):
        """A spec with supported actions builds a ForestModel with actions."""
        spec = ModelSpec(
            model_name='action_test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            actions={
                'harvest': ActionSpec(
                    acode='harvest',
                    description='cut trees',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=0, max_age=100),
                    ),
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert result.model is not None
        assert len(result.model.actions) >= 1
        # Verify the .act file was emitted
        act_files = list(tmp_path.glob('*.act'))
        assert len(act_files) == 1

    def test_action_import_populates_model(self, tmp_path):
        """Real .act import populates ForestModel.actions."""
        spec = ModelSpec(
            model_name='action_import',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            actions={
                'harvest': ActionSpec(
                    acode='harvest',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=0, max_age=100),
                    ),
                ),
                'thinning': ActionSpec(
                    acode='thinning',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=20, max_age=60),
                    ),
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert len(result.model.actions) == 2
        action_codes = set(result.model.actions.keys())
        assert 'harvest' in action_codes
        assert 'thinning' in action_codes


class TestBuilderTransitionsSupported:
    """Transitions with _TH prefix are supported in the five-section implementation."""

    def test_supported_transition_builds_forest_model(self, tmp_path):
        """A spec with supported transitions builds a ForestModel with transitions."""
        spec = ModelSpec(
            model_name='trans_test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            transitions={
                'case1': TransitionSpec(
                    case='case1',
                    source=('A',),
                    target=('A',),
                    action='harvest',
                    theme_replace='_TH1',
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert result.model is not None
        # Verify the .trn file was emitted
        trn_files = list(tmp_path.glob('*.trn'))
        assert len(trn_files) == 1

    def test_transition_import_populates_model(self, tmp_path):
        """Real .trn import populates ForestModel.transitions."""
        spec = ModelSpec(
            model_name='trans_import',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            transitions={
                'case1': TransitionSpec(
                    case='case1',
                    source=('A',),
                    target=('A',),
                    action='harvest',
                    theme_replace='_TH1',
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert len(result.model.transitions) >= 1
        trans_cases = set(result.model.transitions.keys())
        assert 'case1' in trans_cases


class TestBuilderRejectsUnsupportedTransitions:
    """_APPEND transitions are still rejected."""

    def test_theme_append_rejected(self, tmp_path):
        """theme_append transitions are explicitly rejected."""
        with pytest.raises(UnsupportedTransitionError):
            ModelSpec(
                model_name='reject_append',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                areas={('A',): {1: 100.0}},
                yields=(
                    YieldSpec(
                        mask=('A',),
                        ytype='a',
                        ynames=('BA',),
                        points={'BA': [0.0, 10.0]},
                    ),
                ),
                transitions={
                    'case1': TransitionSpec(
                        case='case1',
                        source=('A',),
                        target=('A',),
                        action='harvest',
                        theme_append='_TH1',
                    ),
                },
            )

    def test_rejection_names_case(self):
        """Rejection message names the offending transition case."""
        with pytest.raises(UnsupportedTransitionError, match='bad_case'):
            ModelSpec(
                model_name='reject_append_named',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                transitions={
                    'bad_case': TransitionSpec(
                        case='bad_case',
                        source=('A',),
                        target=('A',),
                        action='harvest',
                        theme_append='_TH1',
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Existing-model isolation
# ---------------------------------------------------------------------------

class TestExistingModelIsolation:
    def test_spec_does_not_mutate_existing_model(self):
        """Building from a spec does not affect pre-existing models."""
        # This test verifies that ModelSpec is construction-oriented and
        # does not have side effects on any existing ForestModel state.
        spec = ModelSpec(
            model_name='iso_test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
        )
        # to_dict and from_dict should be pure
        d = spec.to_dict()
        spec2 = ModelSpec.from_dict(d)
        assert spec == spec2
        # No mutation of original
        assert spec.areas == {('A',): {1: 100.0}}


# ---------------------------------------------------------------------------
# Unsupported transition behavior
# ---------------------------------------------------------------------------

class TestUnsupportedTransitions:
    def test_theme_append_rejected(self):
        """theme_append transitions are explicitly rejected."""
        with pytest.raises(UnsupportedTransitionError):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=10,
                period_length=10,
                max_age=100,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                transitions={
                    'bad': TransitionSpec(
                        case='bad',
                        source=('A',),
                        target=('A',),
                        action='harvest',
                        theme_append='_TH1',
                    ),
                },
            )

    def test_invalid_theme_replace_rejected(self):
        """theme_replace not starting with '_TH' is rejected."""
        with pytest.raises(UnsupportedTransitionError):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=10,
                period_length=10,
                max_age=100,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                transitions={
                    'bad': TransitionSpec(
                        case='bad',
                        source=('A',),
                        target=('A',),
                        action='harvest',
                        theme_replace='invalid',
                    ),
                },
            )

    def test_valid_theme_replace_accepted(self):
        """theme_replace starting with '_TH' is accepted."""
        spec = ModelSpec(
            model_name='test',
            base_year=2000,
            horizon=10,
            period_length=10,
            max_age=100,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            transitions={
                'good': TransitionSpec(
                    case='good',
                    source=('A',),
                    target=('A',),
                    action='harvest',
                    theme_replace='_TH1',
                ),
            },
        )
        assert 'good' in spec.transitions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_duplicate_theme_names_rejected(self):
        """Duplicate theme names raise ModelSpecError."""
        with pytest.raises(ModelSpecError):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=10,
                period_length=10,
                max_age=100,
                themes=(
                    ThemeSpec(name='t0', basecodes=('A',)),
                    ThemeSpec(name='t0', basecodes=('B',)),
                ),
            )

    def test_area_key_theme_count_mismatch(self):
        """Area keys with wrong theme count raise ModelSpecError."""
        with pytest.raises(ModelSpecError):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=10,
                period_length=10,
                max_age=100,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                areas={('A', 'B'): {1: 100.0}},  # 2 elements, expected 1
            )


# ---------------------------------------------------------------------------
# Phase 10 — Output-boundary validation, overwrite, and loss reporting
# ---------------------------------------------------------------------------

class TestUnsafeModelName:
    """ModelBuilder rejects model_name values that could escape output_dir."""

    def _base_spec(self, model_name: str):
        return ModelSpec(
            model_name=model_name,
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
        )

    def test_slash_rejected(self):
        """model_name with forward slash raises ModelSpecError."""
        spec = self._base_spec('foo/bar')
        with pytest.raises(ModelSpecError, match='path separators'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_backslash_rejected(self):
        """model_name with backslash raises ModelSpecError."""
        spec = self._base_spec('foo\\bar')
        with pytest.raises(ModelSpecError, match='path separators'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_dotdot_rejected(self):
        """model_name with '..' traversal raises ModelSpecError."""
        # Forward-slash variant is caught by the path-separator check first.
        spec = self._base_spec('foo/../bar')
        with pytest.raises(ModelSpecError):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_bare_dotdot_rejected(self):
        """A model_name that is literally '..' is rejected as traversal."""
        spec = self._base_spec('..')
        with pytest.raises(ModelSpecError, match='path traversal'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_empty_name_rejected(self):
        """Empty model_name raises ModelSpecError."""
        spec = self._base_spec('')
        with pytest.raises(ModelSpecError, match='empty'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_whitespace_only_rejected(self):
        """Whitespace-only model_name raises ModelSpecError."""
        spec = self._base_spec('   ')
        with pytest.raises(ModelSpecError, match='empty|blank'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_leading_trailing_whitespace_rejected(self):
        """model_name with leading/trailing whitespace raises ModelSpecError."""
        spec = self._base_spec('  safe  ')
        with pytest.raises(ModelSpecError, match='whitespace'):
            ModelBuilder(spec).build(Path(tempfile.gettempdir()))

    def test_safe_name_accepted(self):
        """A normal alphanumeric model_name is accepted."""
        spec = self._base_spec('my_safe_model_v2')
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ModelBuilder(spec).build(Path(tmpdir))
        assert result.model is not None


class TestOutputDirNotEmpty:
    """ModelBuilder rejects non-empty output_dir by default."""

    def _base_spec(self):
        return ModelSpec(
            model_name='test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
        )

    def test_nonempty_dir_rejected_by_default(self, tmp_path):
        """Building into a non-empty dir without overwrite raises."""
        tmp_path.joinpath('unrelated.txt').write_text('do not touch')
        spec = self._base_spec()
        with pytest.raises(ModelSpecError, match='non-empty'):
            ModelBuilder(spec).build(tmp_path)
        # Unrelated file must still exist.
        assert tmp_path.joinpath('unrelated.txt').exists()

    def test_nonempty_dir_allowed_with_overwrite(self, tmp_path):
        """overwrite=True permits writing into a non-empty directory."""
        tmp_path.joinpath('unrelated.txt').write_text('old content')
        spec = self._base_spec()
        result = ModelBuilder(spec).build(tmp_path, overwrite=True)
        assert result.model is not None
        # The unrelated file is NOT deleted — overwrite only allows emission,
        # it does not silently wipe the directory.
        assert tmp_path.joinpath('unrelated.txt').read_text() == 'old content'

    def test_empty_dir_accepted(self, tmp_path):
        """An empty directory is accepted without overwrite."""
        spec = self._base_spec()
        result = ModelBuilder(spec).build(tmp_path)
        assert result.model is not None

    def test_nonexistent_dir_accepted(self):
        """A non-existent directory is created and accepted."""
        spec = self._base_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            new_subdir = Path(tmpdir) / 'new' / 'subdir'
            result = ModelBuilder(spec).build(new_subdir)
            assert result.model is not None
            assert new_subdir.is_dir()


class TestLossReporting:
    """BuildResult carries explicit loss/unsupported record."""

    def _base_spec(self):
        return ModelSpec(
            model_name='loss_test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
        )

    def test_loss_empty_when_no_unsupported_features(self, tmp_path):
        """Loss dict is empty when no unsupported features are present."""
        spec = self._base_spec()
        result = ModelBuilder(spec).build(tmp_path)
        assert isinstance(result.loss, dict)
        assert result.loss == {}

    def test_loss_reports_unsupported_action_features(self, tmp_path):
        """Loss reports unsupported action features (target_age, lock_exempt, description)."""
        spec = ModelSpec(
            model_name='loss_unsupported',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            actions={
                'harvest': ActionSpec(
                    acode='harvest',
                    target_age=50,
                    lock_exempt=True,
                    description='cut trees',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=0, max_age=100),
                    ),
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert 'actions' in result.loss
        assert any('target_age' in msg for msg in result.loss['actions'])
        assert any('lock_exempt' in msg for msg in result.loss['actions'])
        assert any('description' in msg for msg in result.loss['actions'])

    def test_loss_reports_unsupported_transition_features(self, tmp_path):
        """Loss reports unsupported transition features (theme_mask)."""
        spec = ModelSpec(
            model_name='loss_trans_unsupported',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            transitions={
                'case1': TransitionSpec(
                    case='case1',
                    source=('A',),
                    target=('A',),
                    action='harvest',
                    theme_replace='_TH1',
                    theme_mask=('A',),
                ),
            },
        )
        result = ModelBuilder(spec).build(tmp_path)
        assert 'transitions' in result.loss
        assert any('theme_mask' in msg for msg in result.loss['transitions'])

    def test_loss_record_json_serializable(self, tmp_path):
        """BuildResult.loss round-trips through JSON."""
        spec = self._base_spec()
        result = ModelBuilder(spec).build(tmp_path)
        # Should not raise.
        json_str = json.dumps(result.loss)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_loss_in_to_dict(self, tmp_path):
        """BuildResult.to_dict includes the loss record."""
        spec = self._base_spec()
        result = ModelBuilder(spec).build(tmp_path)
        d = result.to_dict()
        assert 'loss' in d
        assert d['loss'] == {}

    def test_pre_emission_rejection_preserved(self):
        """Unsupported transitions are rejected before any files are emitted."""
        with pytest.raises(UnsupportedTransitionError):
            ModelSpec(
                model_name='reject_still',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                transitions={
                    'bad': TransitionSpec(
                        case='bad',
                        source=('A',),
                        target=('A',),
                        action='harvest',
                        theme_append='_TH1',
                    ),
                },
            )


# ---------------------------------------------------------------------------
# FEMIC-shaped pair-point regression (issue: TypeError in emit_yields)
# ---------------------------------------------------------------------------

class TestFemicPairPoints:
    """Regression tests for explicit (age, value) pair yield points.

    These tests verify that YieldSpec points can be supplied as either
    bare numeric value sequences (existing behavior) or explicit
    ``(age, value)`` pair sequences, with proper validation.
    """

    def _femic_five_theme_spec(self, points_format: str = 'pairs'):
        """Build a FEMIC-shaped five-theme spec with pair points."""
        themes = (
            ThemeSpec(name='tsa', description='Timber Supply Area', basecodes=('tsa1',)),
            ThemeSpec(name='ifm', description='Managed state', basecodes=('managed',)),
            ThemeSpec(name='au', description='Analysis Unit', basecodes=('1',)),
            ThemeSpec(name='stratum', description='Stratum code', basecodes=('s1',)),
            ThemeSpec(name='curve', description='Yield curve ID', basecodes=('7',)),
        )
        if points_format == 'pairs':
            points = {'totvol': [(0, 0.0), (2, 40.0), (4, 80.0), (6, 120.0)]}
        else:
            points = {'totvol': [0.0, 40.0, 80.0, 120.0]}
        return ModelSpec(
            model_name='femic_parity',
            base_year=2020,
            horizon=5,
            period_length=10,
            max_age=100,
            themes=themes,
            areas={('tsa1', 'managed', '1', 's1', '7'): {0: 12.5, 2: 8.0}},
            yields=(YieldSpec(
                mask=('tsa1', 'managed', '1', 's1', '7'),
                ytype='a',
                ynames=('totvol',),
                points=points,
            ),),
        )

    def test_pair_points_emit_success(self):
        """FEMIC-shaped spec with pair points does not crash emit_yields."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(spec, Path(tmpdir))
            content = path.read_text()
        assert content  # Non-empty output
        assert '*Y tsa1 managed 1 s1 7' in content
        assert '_AGE totvol' in content

    def test_pair_points_deterministic_bytes(self):
        """Same pair-point spec produces identical bytes across runs."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = emit_yields(spec, Path(tmpdir))
            bytes1 = path1.read_bytes()
        with tempfile.TemporaryDirectory() as tmpdir:
            path2 = emit_yields(spec, Path(tmpdir))
            bytes2 = path2.read_bytes()
        assert bytes1 == bytes2

    def test_pair_points_emit_correct_ages(self):
        """Pair points emit with the supplied ages, not implicit 1..N."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(spec, Path(tmpdir))
            lines = [ln for ln in path.read_text().strip().split('\n')
                     if ln and not ln.startswith('*') and not ln.startswith('_')]
        ages = [int(ln.split()[0]) for ln in lines]
        assert ages == [0, 2, 4, 6], f'Expected explicit ages [0,2,4,6], got {ages}'

    def test_all_femic_sections_emit_and_import(self, tmp_path):
        """The FEMIC five-section shape survives deterministic fresh import."""
        spec = ModelSpec(
            model_name='femic_full',
            base_year=2020,
            horizon=5,
            period_length=10,
            max_age=100,
            themes=(
                ThemeSpec(name='tsa', description='TSA', basecodes=('tsa1',)),
                ThemeSpec(name='ifm', description='Managed state', basecodes=('managed',)),
                ThemeSpec(name='au', description='AU', basecodes=('1',)),
                ThemeSpec(name='stratum', description='Stratum', basecodes=('s1',)),
                ThemeSpec(name='curve', description='Curve', basecodes=('7',)),
            ),
            areas={('tsa1', 'managed', '1', 's1', '7'): {0: 12.5}},
            yields=(YieldSpec(
                mask=('tsa1', 'managed', '1', 's1', '7'),
                ytype='a',
                ynames=('totvol',),
                points={'totvol': [(0, 0.0), (2, 40.0), (4, 80.0)]},
            ),),
            actions={
                'cc': ActionSpec(
                    acode='cc',
                    operable_masks=(OperableMask(
                        mask=('tsa1', 'managed', '1', '?', '?'),
                        min_age=1,
                        max_age=25,
                    ),),
                ),
            },
            transitions={
                'cc': TransitionSpec(
                    case='cc',
                    source=('tsa1', 'managed', '1', '?', '?'),
                    target=('tsa1', 'managed', '1', 's1', '7'),
                    action='cc',
                    proportion=0.75,
                ),
            },
        )

        first_dir = tmp_path / 'first'
        second_dir = tmp_path / 'second'
        first_dir.mkdir()
        second_dir.mkdir()
        first = emit_all(spec, first_dir)
        second = emit_all(spec, second_dir)
        assert set(first) == {'landscape', 'areas', 'yields', 'actions', 'transitions'}
        assert {name: path.read_bytes() for name, path in first.items()} == {
            name: path.read_bytes() for name, path in second.items()
        }
        assert '*ACTION cc Y' in first['actions'].read_text()
        assert '*TARGET tsa1 managed 1 s1 7 75' in first['transitions'].read_text()

        result = ModelBuilder(spec).build(tmp_path / 'built')
        model = result.model
        assert model.oper_expr['cc'][('tsa1', 'managed', '1', '?', '?')] == (
            '_age >= 10 and _age <= 250'
        )
        transition = model.transitions['cc'][
            ('tsa1', 'managed', '1', '?', '?')
        ][''][0]
        assert transition[0] == ('tsa1', 'managed', '1', 's1', '7')
        assert transition[1] == 0.75


# ---------------------------------------------------------------------------
# Phase 10 — Unsupported action type handling (regression)
# ---------------------------------------------------------------------------

class TestUnsupportedActionType:
    """Unsupported action types are silently accepted, not rejected."""

    def test_unsupported_action_type_not_rejected(self, tmp_path):
        """An action with an acode not in SUPPORTED_ACTION_TYPES is silently accepted."""
        spec = ModelSpec(
            model_name='unsupported_action',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            actions={
                'weird_action': ActionSpec(
                    acode='weird_action',
                    operable_masks=(
                        OperableMask(mask=('A',), min_age=0, max_age=100),
                    ),
                ),
            },
        )
        # Should NOT raise — unsupported action types are silently accepted
        result = ModelBuilder(spec).build(tmp_path)
        assert result.model is not None
        # The action is still emitted (silently)
        assert 'weird_action' in result.model.actions


# ---------------------------------------------------------------------------
# Regression — FEMIC pair-point tests (preserved from prior state)
# ---------------------------------------------------------------------------

class TestFemicPairPointsRegression:
    """Additional regression tests for pair-point yield handling."""

    def _femic_five_theme_spec(self, points_format: str = 'pairs'):
        """Build a FEMIC-shaped five-theme spec with pair points."""
        themes = (
            ThemeSpec(name='tsa', description='Timber Supply Area', basecodes=('tsa1',)),
            ThemeSpec(name='ifm', description='Managed state', basecodes=('managed',)),
            ThemeSpec(name='au', description='Analysis Unit', basecodes=('1',)),
            ThemeSpec(name='stratum', description='Stratum code', basecodes=('s1',)),
            ThemeSpec(name='curve', description='Yield curve ID', basecodes=('7',)),
        )
        if points_format == 'pairs':
            points = {'totvol': [(0, 0.0), (2, 40.0), (4, 80.0), (6, 120.0)]}
        else:
            points = {'totvol': [0.0, 40.0, 80.0, 120.0]}
        return ModelSpec(
            model_name='femic_parity',
            base_year=2020,
            horizon=5,
            period_length=10,
            max_age=100,
            themes=themes,
            areas={('tsa1', 'managed', '1', 's1', '7'): {0: 12.5, 2: 8.0}},
            yields=(YieldSpec(
                mask=('tsa1', 'managed', '1', 's1', '7'),
                ytype='a',
                ynames=('totvol',),
                points=points,
            ),),
        )

    def test_bare_points_still_work(self):
        """Bare numeric value sequences still produce implicit sequential ages."""
        spec = self._femic_five_theme_spec('bare')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(spec, Path(tmpdir))
            lines = [ln for ln in path.read_text().strip().split('\n')
                     if ln and not ln.startswith('*') and not ln.startswith('_')]
        ages = [int(ln.split()[0]) for ln in lines]
        assert ages == [1, 2, 3, 4], f'Expected implicit ages [1,2,3,4], got {ages}'

    def test_pair_points_curves_match_values(self):
        """Pair point values match the supplied values exactly."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(spec, Path(tmpdir))
            lines = [ln for ln in path.read_text().strip().split('\n')
                     if ln and not ln.startswith('*') and not ln.startswith('_')]
        values = [float(ln.split()[1]) for ln in lines]
        assert values == [0.0, 40.0, 80.0, 120.0]

    def test_pair_points_real_builder_import(self):
        """FEMIC-shaped pair-point spec builds a ForestModel via ModelBuilder."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ModelBuilder(spec)
            result = builder.build(Path(tmpdir))
        assert result.model is not None
        assert result.model.nthemes() == 5

    def test_mixed_formats_rejected(self):
        """Mixed bare/pair formats within a yield raise ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='mixed|format'):
            ModelSpec(
                model_name='mixed_fmt',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA', 'HWP'),
                    points={
                        'BA': [0.0, 10.0, 20.0],           # bare
                        'HWP': [(0, 0.0), (1, 1.0), (2, 2.0)],  # pairs
                    },
                ),),
            )

    def test_unaligned_ages_rejected(self):
        """Components with different age sequences raise ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='ages'):
            ModelSpec(
                model_name='unaligned',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA', 'HWP'),
                    points={
                        'BA': [(0, 0.0), (2, 10.0), (4, 20.0)],
                        'HWP': [(0, 0.0), (3, 1.0), (6, 2.0)],
                    },
                ),),
            )

    def test_non_increasing_ages_rejected(self):
        """Non-monotonic ages in a component raise ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='strictly increasing'):
            ModelSpec(
                model_name='nonmono',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [(5, 5.0), (3, 3.0), (1, 1.0)]},
                ),),
            )

    def test_duplicate_ages_rejected(self):
        """Duplicate ages in a component raise ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='strictly increasing'):
            ModelSpec(
                model_name='dup_ages',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [(1, 1.0), (1, 2.0), (3, 3.0)]},
                ),),
            )

    def test_malformed_pair_rejected(self):
        """A non-tuple item in a pair sequence raises ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='pair'):
            ModelSpec(
                model_name='malformed',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [(1, 1.0), 'not_a_pair', (3, 3.0)]},
                ),),
            )

    def test_non_numeric_age_rejected(self):
        """Non-numeric age in a pair raises ModelSpecError."""
        themes = (ThemeSpec(name='t0', basecodes=('A',)),)
        with pytest.raises(ModelSpecError, match='not numeric|non-numeric'):
            ModelSpec(
                model_name='bad_age',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=themes,
                yields=(YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [('x', 1.0)]},
                ),),
            )

    def test_emit_all_deterministic_pairs(self):
        """emit_all with pair points produces deterministic bytes."""
        spec = self._femic_five_theme_spec('pairs')
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = emit_all(spec, Path(tmpdir))
            b1 = {k: v.read_bytes() for k, v in r1.items()}
        with tempfile.TemporaryDirectory() as tmpdir:
            r2 = emit_all(spec, Path(tmpdir))
            b2 = {k: v.read_bytes() for k, v in r2.items()}
        assert b1 == b2

    def test_time_based_pair_points(self):
        """Time-based (ytype='t') yields also accept pair points."""
        spec = ModelSpec(
            model_name='time_pairs',
            base_year=2020,
            horizon=5,
            period_length=10,
            max_age=100,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            yields=(YieldSpec(
                mask=('A',),
                ytype='t',
                ynames=('V',),
                points={'V': [(0, 0.0), (2, 50.0), (4, 100.0)]},
            ),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_yields(spec, Path(tmpdir))
            lines = [ln for ln in path.read_text().strip().split('\n')
                     if ln and not ln.startswith('*') and not ln.startswith('_')]
        ages = [int(ln.split()[0]) for ln in lines]
        assert ages == [0, 2, 4]


# ---------------------------------------------------------------------------
# Phase 10 — Output section support
# ---------------------------------------------------------------------------
class TestOutputSpec:
    """OutputSpec and OutputGroupSpec dataclasses."""

    def test_output_spec_creation(self):
        """OutputSpec can be created with all fields."""
        from ws3.agent.spec import OutputSpec

        output = OutputSpec(
            code='totvol',
            theme_index='1',
            description='Total volume',
            expression='_AREA * BA',
            is_level=False,
        )
        assert output.code == 'totvol'
        assert output.theme_index == '1'
        assert output.description == 'Total volume'
        assert output.expression == '_AREA * BA'
        assert output.is_level is False

    def test_output_group_spec_creation(self):
        """OutputGroupSpec can be created with output codes."""
        from ws3.agent.spec import OutputGroupSpec

        group = OutputGroupSpec(
            name='inventory',
            output_codes=('totvol', 'totvolb'),
        )
        assert group.name == 'inventory'
        assert group.output_codes == ('totvol', 'totvolb')

    def test_output_spec_to_dict(self):
        """OutputSpec.to_dict produces JSON-safe dict."""
        from ws3.agent.spec import OutputSpec

        output = OutputSpec(
            code='totvol',
            theme_index='1',
            description='Total volume',
            expression='_AREA * BA',
            is_level=False,
        )
        d = output.to_dict()
        assert d == {
            'code': 'totvol',
            'theme_index': '1',
            'description': 'Total volume',
            'expression': '_AREA * BA',
            'is_level': False,
        }

    def test_output_group_spec_to_dict(self):
        """OutputGroupSpec.to_dict produces JSON-safe dict."""
        from ws3.agent.spec import OutputGroupSpec

        group = OutputGroupSpec(
            name='inventory',
            output_codes=('totvol', 'totvolb'),
        )
        d = group.to_dict()
        assert d == {
            'name': 'inventory',
            'output_codes': ['totvol', 'totvolb'],
        }


class TestOutputModelSpec:
    """ModelSpec with outputs field."""

    def test_model_spec_with_outputs(self):
        """ModelSpec accepts outputs and output_groups."""
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        spec = ModelSpec(
            model_name='output_test',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
                OutputSpec(
                    code='totvolb',
                    theme_index='1',
                    description='Total volume boards',
                    expression='_AREA * BA * 10',
                ),
            ),
            output_groups=(
                OutputGroupSpec(
                    name='inventory',
                    output_codes=('totvol', 'totvolb'),
                ),
            ),
        )
        assert len(spec.outputs) == 2
        assert len(spec.output_groups) == 1
        assert spec.output_groups[0].name == 'inventory'

    def test_duplicate_output_codes_rejected(self):
        """Duplicate output codes raise ModelSpecError."""
        from ws3.agent.spec import OutputSpec

        with pytest.raises(ModelSpecError, match='Duplicate output codes'):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                outputs=(
                    OutputSpec(code='totvol', description='First'),
                    OutputSpec(code='totvol', description='Second'),
                ),
            )

    def test_output_group_unknown_reference_rejected(self):
        """Output group referencing unknown output code raises ModelSpecError."""
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        with pytest.raises(ModelSpecError, match='unknown output code'):
            ModelSpec(
                model_name='test',
                base_year=2000,
                horizon=5,
                period_length=10,
                max_age=80,
                themes=(ThemeSpec(name='t0', basecodes=('A',)),),
                outputs=(
                    OutputSpec(code='totvol', description='Total volume'),
                ),
                output_groups=(
                    OutputGroupSpec(
                        name='inventory',
                        output_codes=('totvol', 'unknown'),
                    ),
                ),
            )


class TestOutputJsonRoundTrip:
    """Output section round-trips through JSON."""

    def test_outputs_round_trip(self):
        """Outputs survive JSON serialization."""
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        spec = ModelSpec(
            model_name='output_rt',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
                OutputSpec(
                    code='level1',
                    theme_index='1',
                    description='A level',
                    expression='',
                    is_level=True,
                ),
            ),
            output_groups=(
                OutputGroupSpec(
                    name='inventory',
                    output_codes=('totvol', 'level1'),
                ),
            ),
        )
        d = spec.to_dict()
        spec2 = ModelSpec.from_dict(d)

        assert len(spec2.outputs) == 2
        assert spec2.outputs[0].code == 'totvol'
        assert spec2.outputs[0].theme_index == '1'
        assert spec2.outputs[0].expression == '_AREA * BA'
        assert spec2.outputs[1].is_level is True
        assert len(spec2.output_groups) == 1
        assert spec2.output_groups[0].output_codes == ('totvol', 'level1')


class TestOutputEmission:
    """Output section emission produces correct Woodstock format."""

    def test_emit_outputs_basic(self):
        """Basic output emission produces *OUTPUT lines."""
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name='emit_out',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_outputs(spec, Path(tmpdir))
            content = path.read_text()

        assert '*OUTPUT totvol(1) Total volume' in content
        assert '*SOURCE _AREA * BA' in content

    def test_emit_outputs_level(self):
        """Level outputs use *LEVEL keyword."""
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name='emit_level',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            outputs=(
                OutputSpec(
                    code='level1',
                    description='A level',
                    is_level=True,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_outputs(spec, Path(tmpdir))
            content = path.read_text()

        assert '*LEVEL level1' in content

    def test_emit_outputs_with_group(self):
        """Output groups are emitted as *GROUP lines."""
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        spec = ModelSpec(
            model_name='emit_group',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            outputs=(
                OutputSpec(code='totvol', description='Total volume'),
            ),
            output_groups=(
                OutputGroupSpec(
                    name='inventory',
                    output_codes=('totvol',),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = emit_outputs(spec, Path(tmpdir))
            content = path.read_text()

        assert '*GROUP inventory totvol' in content

    def test_emit_outputs_deterministic(self):
        """Same spec produces identical bytes on every emit."""
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name='emit_det',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = emit_outputs(spec, Path(tmpdir))
            content1 = path1.read_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            path2 = emit_outputs(spec, Path(tmpdir))
            content2 = path2.read_bytes()

        assert content1 == content2


class TestOutputBuilderImport:
    """ModelBuilder imports outputs section."""

    def test_builder_emits_outputs(self):
        """ModelBuilder emits .out file when outputs are present."""
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name='builder_out',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ModelBuilder(spec).build(Path(tmpdir))
            out_files = list(Path(tmpdir).glob('*.out'))
            assert len(out_files) == 1
            content = out_files[0].read_text()
            assert '*OUTPUT totvol' in content

    def test_builder_imports_outputs(self):
        """ModelBuilder emits and imports outputs with theme_index."""
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name="builder_import_out",
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name="t0", basecodes=("A",)),),
            areas={("A",): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=("A",),
                    ytype="a",
                    ynames=("BA",),
                    points={"BA": [0.0, 10.0]},
                ),
            ),
            outputs=(
                OutputSpec(
                    code="totvol",
                    theme_index="1",
                    description="Total volume",
                    expression="_AREA * BA",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ModelBuilder(spec).build(Path(tmpdir))
            # Outputs are now imported successfully after parser fix
            assert "totvol" in result.model.outputs
            # The .out file was emitted
            out_files = list(Path(tmpdir).glob("*.out"))
            assert len(out_files) == 1
            # No loss for outputs (import succeeded)
            assert "outputs" not in result.loss

    def test_builder_no_outputs(self):
        """ModelBuilder does not emit .out file when no outputs."""
        spec = ModelSpec(
            model_name='no_outputs',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ModelBuilder(spec).build(Path(tmpdir))
            out_files = list(Path(tmpdir).glob('*.out'))
            assert len(out_files) == 0


class TestOutputFiveSectionSmoke:
    """Five-section smoke test with outputs."""

    def test_five_section_with_outputs(self):
        """Full five-section model with outputs builds successfully."""
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        spec = ModelSpec(
            model_name='five_section_smoke_out',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            actions={
                'harvest': ActionSpec(
                    acode='harvest',
                    operable_masks=(OperableMask(('A',), 2, 6),),
                ),
            },
            transitions={
                'harvest': TransitionSpec(
                    case='harvest',
                    source=('A',),
                    target=('A',),
                    action='harvest',
                    proportion=0.75,
                ),
            },
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
            output_groups=(
                OutputGroupSpec(
                    name='inventory',
                    output_codes=('totvol',),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ModelBuilder(spec).build(Path(tmpdir))
            assert set(result.emitted_paths) == {
                'landscape', 'areas', 'yields', 'actions', 'transitions', 'outputs'
            }
            # Outputs are now imported successfully after parser fix
            assert 'totvol' in result.model.outputs
            assert 'outputs' not in result.loss


# ---------------------------------------------------------------------------
# Regression test for parser fix (theme_index extraction)
# ---------------------------------------------------------------------------

class TestOutputParserFix:
    """Regression test for _resolve_outputs_buffer theme_index extraction."""

    def test_parser_extracts_theme_index(self):
        """Parser correctly extracts theme_index from code(N) format."""
        from ws3.forest import ForestModel

        # Create a minimal ForestModel with required parameters.
        fm = ForestModel(
            model_name='test_parser',
            model_path='/tmp',
            base_year=2000,
        )

        # Simulate the parsed output string with theme_index.
        output_str = '*OUTPUT totvol(1) Total volume\n*SOURCE _AREA * BA'

        # Parse it.
        fm._resolve_outputs_buffer(output_str)

        # Verify theme_index was extracted correctly.
        # Convention: Woodstock uses 1-based theme indices; Output normalizes
        # to 0-based (consistent with _REPLACE/_APPEND keyword handling).
        assert 'totvol' in fm.outputs
        assert fm.outputs['totvol'].theme_index == 0
        assert fm.outputs['totvol'].description == 'total volume'
        # Expression preserves original case (parser lowercases only for tokenization).
        assert fm.outputs['totvol'].expression == '_AREA * BA'

    def test_parser_no_theme_index(self):
        """Parser handles outputs without theme_index."""
        from ws3.forest import ForestModel

        fm = ForestModel(
            model_name='test_parser_no_theme',
            model_path='/tmp',
            base_year=2000,
        )

        # Output without theme_index.
        output_str = '*OUTPUT totvol Total volume\n*SOURCE _AREA * BA'

        fm._resolve_outputs_buffer(output_str)

        assert 'totvol' in fm.outputs
        assert fm.outputs['totvol'].theme_index is None

    def test_full_roundtrip_with_theme_index(self):
        """Full roundtrip: spec -> emit -> import -> model has outputs with theme_index."""
        from ws3.agent.builder import ModelBuilder
        from ws3.agent.spec import OutputSpec

        spec = ModelSpec(
            model_name='roundtrip_theme',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ModelBuilder(spec).build(Path(tmpdir))
            # Verify output was imported with correct (normalized) theme_index.
            assert 'totvol' in result.model.outputs
            assert result.model.outputs['totvol'].theme_index == 0
            # Verify no import errors.
            assert 'outputs' not in result.loss


class TestOutputGroupImport:
    """Regression test for output group importer semantic defect."""

    def test_output_group_parser_imports_codes(self):
        """Parser parses output codes from *GROUP lines."""
        from ws3.forest import ForestModel

        fm = ForestModel(
            model_name='test_group_parser',
            model_path='/tmp',
            base_year=2000,
        )

        # Simulate a *GROUP line with output codes (as emitted by the emitter).
        output_str = (
            '*OUTPUT totvol Total volume\n'
            '*SOURCE _AREA * BA\n'
            '*GROUP summary totvol\n'
        )

        fm._resolve_outputs_buffer(output_str)

        # The 'summary' group should contain 'totvol'.
        assert 'summary' in fm.output_groups
        assert fm.output_groups['summary'] == {'totvol'}
        # no_group should also be populated (existing contract).
        assert 'no_group' in fm.output_groups
        assert fm.output_groups['no_group'] == {'totvol'}

    def test_output_group_roundtrip(self):
        """Full roundtrip: spec with output group -> emit -> import -> model.output_groups."""
        from ws3.agent.builder import ModelBuilder
        from ws3.agent.spec import OutputGroupSpec, OutputSpec

        spec = ModelSpec(
            model_name='group_roundtrip',
            base_year=2000,
            horizon=5,
            period_length=10,
            max_age=80,
            themes=(ThemeSpec(name='t0', basecodes=('A',)),),
            areas={('A',): {1: 100.0}},
            yields=(
                YieldSpec(
                    mask=('A',),
                    ytype='a',
                    ynames=('BA',),
                    points={'BA': [0.0, 10.0]},
                ),
            ),
            outputs=(
                OutputSpec(
                    code='totvol',
                    theme_index='1',
                    description='Total volume',
                    expression='_AREA * BA',
                ),
            ),
            output_groups=(
                OutputGroupSpec(
                    name='summary',
                    output_codes=('totvol',),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ModelBuilder(spec).build(Path(tmpdir))
            # Verify output group was imported correctly.
            assert 'summary' in result.model.output_groups
            assert result.model.output_groups['summary'] == {'totvol'}
            # Verify no import errors.
            assert 'outputs' not in result.loss


# ---------------------------------------------------------------------------
# Phase 10 — Output section support
# ---------------------------------------------------------------------------
