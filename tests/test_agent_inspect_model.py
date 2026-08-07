"""
Tests for the ws3 ``inspect_model`` read-only metadata capability.

Everything runs offline against ``FakeProvider``. No endpoint, no credentials.

Covers:
- Deterministic snapshot against a real ForestModel
- Validator validates model identity and computed claims
- Malformed/unsafe provider output is rejected
- Retry recovers when the provider self-corrects
- No mutation of the live model
- Unsupported complex queries return explicit unsupported result
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip('fresh_agent_core', reason='ws3[agent] not installed')

from fresh_agent_core import AgentConfig, FakeProvider  # noqa: E402

from ws3.agent.capabilities.inspect_model import (  # noqa: E402
    InspectInputs,
    InspectModel,
    InspectResult,
    _snapshot,
)

CONFIG = AgentConfig(endpoint='offline://test', model='test-model')

MODEL_DIR = Path(__file__).parent.parent / 'examples' / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'


@pytest.fixture(scope='module')
def fm():
    """A real ForestModel. The snapshot is only meaningful against real state."""
    if not MODEL_DIR.is_dir():
        pytest.skip(f'test model not available at {MODEL_DIR}')
    from ws3.forest import ForestModel  # noqa: E402

    model = ForestModel(
        model_name=MODEL_NAME,
        model_path=str(MODEL_DIR),
        base_year=2020,
        horizon=10,
        period_length=10,
        max_age=1000,
    )
    model.import_landscape_section()
    model.import_areas_section(convert_periods_to_years=10)
    return model


def _snap_response() -> str:
    """JSON snapshot that looks plausible; values are ignored by the executor."""
    return json.dumps({
        'operation': 'full_snapshot',
        'model_name': 'should-be-ignored',
        'base_year': 9999,
    })


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class TestSnapshotExecutor:
    def test_snapshot_returns_all_metadata_keys(self, fm):
        snap = _snapshot(fm)
        assert set(snap.keys()) >= {
            'model_name', 'name', 'base_year', 'horizon', 'period_length',
            'periods', 'nthemes', 'nactions', 'ndtypes', 'total_area',
        }

    def test_snapshot_fills_scalar_fields(self, fm):
        snap = _snapshot(fm)
        assert snap['model_name'] == MODEL_NAME
        assert snap['base_year'] == 2020
        assert snap['period_length'] == 10
        assert snap['horizon'] == 10
        assert snap['periods'] == list(range(1, 11))

    def test_snapshot_counts_are_integers(self, fm):
        snap = _snapshot(fm)
        assert isinstance(snap['nthemes'], int)
        assert snap['nthemes'] > 0
        assert isinstance(snap['nactions'], int)
        assert snap['nactions'] >= 0  # actions not imported in fixture
        assert isinstance(snap['ndtypes'], int)
        assert snap['ndtypes'] > 0

    def test_total_area_is_safe_sum(self, fm):
        snap = _snapshot(fm)
        # Total area should be computed from dt.area(1) sum, must be a float
        assert isinstance(snap['total_area'], float)
        # Area may be 0 if no period-1 data; must be non-negative
        assert snap['total_area'] >= 0

    def test_snapshot_does_not_mutate_model(self, fm):
        before = _snapshot(fm)
        _snapshot(fm)
        after = _snapshot(fm)
        assert before == after

    def test_snapshot_handles_empty_model(self):
        """An unloaded model still produces a safe (mostly-None) snapshot."""
        from ws3.forest import ForestModel

        bare = ForestModel(
            model_name='bare',
            model_path='/dev/null',
            base_year=2000,
            horizon=5,
            period_length=10,
        )
        snap = _snapshot(bare)
        assert snap['model_name'] == 'bare'
        assert snap['base_year'] == 2000
        assert snap['horizon'] == 5
        assert snap['nthemes'] == 0
        assert snap['nactions'] == 0
        assert snap['ndtypes'] == 0
        # No areas imported — total_area should be None or 0
        assert snap['total_area'] in (None, 0.0)


# ---------------------------------------------------------------------------
# Capability — inputs and parsing
# ---------------------------------------------------------------------------


class TestInspectInputs:
    def test_defaults(self):
        inp = InspectInputs(query='show me the model')
        assert inp.model_name == ''

    def test_from_payload(self):
        inp = InspectModel().from_payload({
            'query': 'counts',
            'model_name': 'tsa24',
        })
        assert inp.query == 'counts'
        assert inp.model_name == 'tsa24'

    def test_from_payload_missing_query(self):
        inp = InspectModel().from_payload({})
        assert inp.query == ''


class TestInspectParsing:
    def test_parses_valid_operation(self):
        cap = InspectModel()
        result = cap.parse('{"operation": "full_snapshot"}')
        assert isinstance(result, InspectResult)
        assert result.unsupported == ''

    def test_parses_unsupported(self):
        cap = InspectModel()
        result = cap.parse('{"operation": "unsupported"}')
        assert result.unsupported == 'query outside bounded operations'

    @pytest.mark.parametrize('operation', [
        'model_identity', 'temporal_summary', 'counts', 'area',
    ])
    def test_parses_each_bounded_operation(self, operation):
        cap = InspectModel()
        result = cap.parse(f'{{"operation": "{operation}"}}')
        assert result.unsupported == ''

    def test_rejects_unknown_operation(self):
        cap = InspectModel()
        with pytest.raises(Exception, match='unrecognized operation'):
            cap.parse('{"operation": "plot"}')

    def test_rejects_non_json(self):
        cap = InspectModel()
        with pytest.raises(Exception, match='JSON'):
            cap.parse('nope')

    def test_rejects_json_without_operation_key(self):
        cap = InspectModel()
        with pytest.raises(Exception, match='JSON'):
            cap.parse('{"foo": "bar"}')

    def test_tolerates_fenced_json(self):
        cap = InspectModel()
        raw = '```json\n{"operation": "full_snapshot"}\n```'
        result = cap.parse(raw)
        assert result.unsupported == ''


# ---------------------------------------------------------------------------
# Capability — validate
# ---------------------------------------------------------------------------


class TestInspectValidation:
    def test_validate_requires_forest_model(self):
        cap = InspectModel()
        candidate = InspectResult(
            model_name=None, name=None, base_year=None, horizon=None,
            period_length=None, periods=None, nthemes=None, nactions=None,
            ndtypes=None, total_area=None, unsupported='',
        )
        verdict = cap.validate(candidate, 'not a forest model')
        assert verdict.ok is False

    def test_validate_accepts_real_forest_model(self, fm):
        cap = InspectModel()
        candidate = InspectResult(
            model_name=None, name=None, base_year=None, horizon=None,
            period_length=None, periods=None, nthemes=None, nactions=None,
            ndtypes=None, total_area=None, unsupported='',
        )
        verdict = cap.validate(candidate, fm)
        assert verdict.ok is True

    def test_validate_accepts_unsupported_query(self, fm):
        """Unsupported queries are a valid provider selection; run() handles them."""
        cap = InspectModel()
        candidate = InspectResult(
            model_name=None, name=None, base_year=None, horizon=None,
            period_length=None, periods=None, nthemes=None, nactions=None,
            ndtypes=None, total_area=None,
            unsupported='query outside bounded operations',
            operation='unsupported',
        )
        verdict = cap.validate(candidate, fm)
        # validate() treats unsupported as a valid selection; run() returns the
        # explicit unsupported result without fabricating values.
        assert verdict.ok is True


# ---------------------------------------------------------------------------
# End-to-end: FakeProvider rejection then retry success
# ---------------------------------------------------------------------------


class TestInspectEndToEnd:
    def test_provider_rejection_then_retry(self, fm):
        """
        Provider outputs malformed response first, then a valid one.

        The capability must retry and succeed — never fabricate values.
        """
        cap = InspectModel()
        provider = FakeProvider([
            'not json at all',
            '{"operation": "full_snapshot"}',
        ])
        result = cap.run(
            InspectInputs(query='show me the model'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        assert result.value is not None
        assert result.value.model_name == MODEL_NAME
        assert result.value.base_year == 2020

    def test_full_snapshot_populates_all_fields(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "full_snapshot"}'])
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.model_name == MODEL_NAME
        assert v.base_year == 2020
        assert v.horizon == 10
        assert v.period_length == 10
        assert v.periods == list(range(1, 11))
        assert v.nthemes > 0
        assert v.nactions >= 0  # actions not imported in fixture
        assert v.ndtypes > 0
        assert v.total_area is not None and v.total_area >= 0
        assert v.unsupported == ''

    def test_unsupported_query_returns_explicit_result(self, fm):
        """
        Unsupported queries must return an explicit unsupported result,
        never fabricated values.
        """
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "unsupported"}'])
        result = cap.run(
            InspectInputs(query='plot the age distribution'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        assert result.value is not None
        assert result.value.unsupported != ''
        assert result.value.operation == 'unsupported'
        # All numeric fields must be None — no fabrication
        assert result.value.model_name is None
        assert result.value.base_year is None
        assert result.value.total_area is None
        assert result.value.horizon is None
        assert result.value.periods is None

    def test_no_mutation(self, fm):
        """
        The capability is read-only. Running it must not change the model.
        """
        snap_before = _snapshot(fm)
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "full_snapshot"}'])
        cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        snap_after = _snapshot(fm)
        assert snap_before == snap_after

    def test_model_identity_rejected_against_wrong_model(self):
        """
        The validator checks model identity. A wrong model must fail.
        No file imports — just compare model_name against the filter.
        """
        from ws3.forest import ForestModel

        wrong = ForestModel(
            model_name='wrong',
            model_path=str(MODEL_DIR),
            base_year=2020,
            horizon=10,
            period_length=10,
            max_age=1000,
        )

        cap = InspectModel()
        provider = FakeProvider(
            ['{"operation": "full_snapshot"}'],
            repeat_last=True,
        )
        result = cap.run(
            InspectInputs(query='show me', model_name='tsa24_clipped'),
            provider=provider,
            config=CONFIG,
            context=wrong,
        )
        # Should fail because the model_name doesn't match
        assert result.ok is False

    def test_model_identity_filter_uses_live_snapshot_before_projection(self, fm):
        cap = InspectModel()
        result = cap.run(
            InspectInputs(query='identify this model', model_name=MODEL_NAME),
            provider=FakeProvider(['{"operation": "model_identity"}']),
            config=CONFIG,
            context=fm,
        )

        assert result.ok is True
        assert result.value.model_name == MODEL_NAME

    def test_values_coming_from_executor_not_provider(self, fm):
        """
        Provider output must not supply trusted numeric facts.

        We script a provider response with wrong values and verify the
        executor overrides them.
        """
        cap = InspectModel()
        provider = FakeProvider([
            '{"operation": "full_snapshot", "base_year": 1899}',
        ])
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        assert result.value.base_year == 2020  # Not 1899
        assert result.value.model_name == MODEL_NAME  # Not "should-be-ignored"

    def test_render_produces_markdown(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "full_snapshot"}'])
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        rendered = cap.render(result.value)
        assert 'WS3 Inspect Model' in rendered
        assert MODEL_NAME in rendered
        assert 'base_year' in rendered

    def test_unsupported_render(self):
        cap = InspectModel()
        result = InspectResult(
            model_name=None, name=None, base_year=None, horizon=None,
            period_length=None, periods=None, nthemes=None, nactions=None,
            ndtypes=None, total_area=None,
            unsupported='plotting is not a bounded operation',
        )
        rendered = cap.render(result)
        assert 'Unsupported' in rendered


# ---------------------------------------------------------------------------
# Operation field selection — only intended fields returned
# ---------------------------------------------------------------------------


class TestOperationFieldSelection:
    """Each bounded operation must return only its intended fields."""

    def test_model_identity_returns_only_identity_fields(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "model_identity"}'])
        result = cap.run(
            InspectInputs(query='model identity'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.operation == 'model_identity'
        assert v.model_name == MODEL_NAME
        assert v.name is not None
        # All other fields must be None
        assert v.base_year is None
        assert v.horizon is None
        assert v.period_length is None
        assert v.periods is None
        assert v.nthemes is None
        assert v.nactions is None
        assert v.ndtypes is None
        assert v.total_area is None

    def test_temporal_summary_returns_only_temporal_fields(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "temporal_summary"}'])
        result = cap.run(
            InspectInputs(query='temporal summary'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.operation == 'temporal_summary'
        assert v.base_year == 2020
        assert v.horizon == 10
        assert v.period_length == 10
        assert v.periods == list(range(1, 11))
        # All other fields must be None
        assert v.model_name is None
        assert v.name is None
        assert v.nthemes is None
        assert v.nactions is None
        assert v.ndtypes is None
        assert v.total_area is None

    def test_counts_returns_only_count_fields(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "counts"}'])
        result = cap.run(
            InspectInputs(query='counts'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.operation == 'counts'
        assert v.nthemes is not None and v.nthemes > 0
        assert v.ndtypes > 0
        # Identity and temporal fields must be None
        assert v.model_name is None
        assert v.base_year is None
        assert v.horizon is None
        assert v.total_area is None

    def test_area_returns_only_area_field(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "area"}'])
        result = cap.run(
            InspectInputs(query='total area'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.operation == 'area'
        assert v.total_area is not None and v.total_area >= 0
        # All other fields must be None
        assert v.model_name is None
        assert v.base_year is None
        assert v.horizon is None
        assert v.nthemes is None
        assert v.ndtypes is None

    def test_full_snapshot_returns_all_fields(self, fm):
        cap = InspectModel()
        provider = FakeProvider(['{"operation": "full_snapshot"}'])
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
        )
        assert result.ok is True
        v = result.value
        assert v.operation == 'full_snapshot'
        # All fields must be populated
        assert v.model_name is not None
        assert v.base_year is not None
        assert v.horizon is not None
        assert v.total_area is not None


# ---------------------------------------------------------------------------
# Provenance — MemorySink records attempts
# ---------------------------------------------------------------------------


class TestProvenance:
    """MemorySink must record every attempt, successful or not."""

    def test_memory_sink_records_successful_attempt(self, fm):
        from fresh_agent_core.provenance import MemorySink

        cap = InspectModel()
        provider = FakeProvider(['{"operation": "full_snapshot"}'])
        sink = MemorySink()
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
            sink=sink,
        )
        assert result.ok is True
        assert len(sink.records) == 1
        record = sink.records[0]
        assert record.capability == 'inspect_model'
        assert record.ok is True
        assert record.attempt == 1

    def test_memory_sink_records_parse_failure_then_success(self, fm):
        from fresh_agent_core.provenance import MemorySink

        cap = InspectModel()
        provider = FakeProvider([
            'not json',
            '{"operation": "full_snapshot"}',
        ])
        sink = MemorySink()
        result = cap.run(
            InspectInputs(query='full snapshot'),
            provider=provider,
            config=CONFIG,
            context=fm,
            sink=sink,
        )
        assert result.ok is True
        # Two attempts: first failed parse, second succeeded
        assert len(sink.records) == 2
        assert sink.records[0].ok is False
        assert sink.records[1].ok is True

    def test_memory_sink_records_unsupported_query(self, fm):
        from fresh_agent_core.provenance import MemorySink

        cap = InspectModel()
        provider = FakeProvider(['{"operation": "unsupported"}'])
        sink = MemorySink()
        result = cap.run(
            InspectInputs(query='plot the age distribution'),
            provider=provider,
            config=CONFIG,
            context=fm,
            sink=sink,
        )
        assert result.ok is True
        assert len(sink.records) == 1
        # Unsupported is a valid provider selection, so ok=True
        assert sink.records[0].ok is True
