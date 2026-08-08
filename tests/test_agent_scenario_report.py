"""Focused tests for the deterministic scenario inventory/products report."""

from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path

import pytest

pytest.importorskip('fresh_agent_core', reason='ws3[agent] not installed')

from fresh_agent_core import AgentConfig, FakeProvider  # noqa: E402

import ws3.agent  # noqa: E402
from ws3.agent.capabilities.scenario_report import (  # noqa: E402
    ScenarioReport,
    ScenarioReportInputs,
    report_scenario_inventory_products,
)
from ws3.forest import ForestModel  # noqa: E402

CONFIG = AgentConfig(endpoint='offline://test', model='test-model')
MODEL_DIR = Path(__file__).parent.parent / 'examples' / 'data' / 'woodstock_model_files_tsa24_clipped'
MODEL_NAME = 'tsa24_clipped'


def _load_model(model_dir: Path = MODEL_DIR) -> ForestModel:
    model = ForestModel(
        model_name=MODEL_NAME,
        model_path=str(model_dir),
        base_year=2020,
        horizon=10,
        period_length=10,
        max_age=1000,
    )
    model.import_landscape_section()
    model.import_areas_section(convert_periods_to_years=10)
    model.import_yields_section(convert_periods_to_years=10)
    model.import_actions_section(convert_periods_to_years=10)
    model.import_transitions_section(convert_periods_to_years=10)
    model.reset_actions()
    return model


def _hashes(model_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(model_dir.iterdir())
        if path.is_file()
    }


class TestScenarioReport:
    def test_fixture_report_has_live_inventory_and_harvest_rows(self):
        before = _hashes(MODEL_DIR)

        result = ws3.agent.report_scenario_inventory_products(MODEL_DIR, MODEL_NAME)

        assert result.ok is True
        assert result.model_identity.model_name == MODEL_NAME
        assert result.model_identity.nthemes == 5
        assert result.initial_area is not None and result.initial_area > 0
        assert result.initial_volume is not None and result.initial_volume > 0
        assert result.schedule_provenance.entries == 24
        assert result.schedule_provenance.periods == tuple(range(1, 11))
        assert result.schedule_provenance.action_codes == ('harvest',)
        assert result.schedule_provenance.applied_in_fresh_model is True
        assert len(result.rows) == 10
        assert [row.period for row in result.rows] == list(range(1, 11))
        assert all(row.standing_volume > 0 for row in result.rows)
        assert any(
            row.harvested_area > 0 and row.harvested_volume > 0
            for row in result.rows
        )
        assert result.warnings == ()
        assert result.errors == ()
        assert result.source_model_files_unchanged is True
        assert _hashes(MODEL_DIR) == before
        assert 'No source model file was mutated' in result.source_model_mutation_statement

    def test_request_schema_has_no_selection_or_mask_inputs(self):
        properties = set(ScenarioReport.input_schema['properties'])

        assert properties == {'model_path', 'model_name', 'schedule_path'}
        assert 'age' not in properties
        assert 'mask' not in properties

    def test_report_calls_exact_inventory_and_product_apis(self, monkeypatch):
        inventory_calls = []
        product_calls = []
        original_inventory = ForestModel.inventory
        original_compile_product = ForestModel.compile_product

        def tracked_inventory(model, period, yname=None, *args, **kwargs):
            inventory_calls.append((period, yname))
            return original_inventory(model, period, yname, *args, **kwargs)

        def tracked_compile_product(model, period, expr, acode=None, *args, **kwargs):
            product_calls.append((period, expr, acode))
            return original_compile_product(model, period, expr, acode, *args, **kwargs)

        monkeypatch.setattr(ForestModel, 'inventory', tracked_inventory)
        monkeypatch.setattr(ForestModel, 'compile_product', tracked_compile_product)

        result = report_scenario_inventory_products(MODEL_DIR, MODEL_NAME)

        assert result.ok is True
        assert (0, None) in inventory_calls
        assert (0, 'totvol') in inventory_calls
        for period in range(1, 11):
            assert (period, '1.', 'harvest') in product_calls
            assert (period, 'totvol', 'harvest') in product_calls
            assert (period, 'totvol') in inventory_calls

    def test_compiled_schedule_without_application_has_no_products(self):
        model = _load_model()
        schedule = model.import_schedule_section(convert_periods_to_years=10)

        assert len(schedule) == 24
        assert model.compile_product(1, '1.', acode='harvest') == 0.0
        assert model.compile_product(1, 'totvol', acode='harvest') == 0.0

    def test_empty_schedule_is_a_structured_no_harvest_report(self, tmp_path):
        copied_model_dir = tmp_path / MODEL_NAME
        shutil.copytree(MODEL_DIR, copied_model_dir)
        (copied_model_dir / f'{MODEL_NAME}.seq').write_text('', encoding='utf-8')

        result = report_scenario_inventory_products(copied_model_dir, MODEL_NAME)

        assert result.ok is True
        assert result.initial_area is not None and result.initial_area > 0
        assert result.schedule_provenance.entries == 0
        assert result.schedule_provenance.applied_in_fresh_model is True
        assert all(row.harvested_area == 0 for row in result.rows)
        assert all(row.harvested_volume == 0 for row in result.rows)
        assert result.warnings == (
            'The selected schedule contained no entries; harvested products are expected to be zero.',
        )
        assert result.source_model_files_unchanged is True

    def test_fresh_model_boundary_does_not_change_callers_model(self):
        caller_model = _load_model()
        before_actions = copy.deepcopy(caller_model.applied_actions)
        before_dtypes = tuple(sorted(caller_model.dtypes))
        before_inventory = (
            caller_model.inventory(0),
            caller_model.inventory(0, 'totvol'),
        )

        result = report_scenario_inventory_products(MODEL_DIR, MODEL_NAME)

        assert result.ok is True
        assert caller_model.applied_actions == before_actions
        assert tuple(sorted(caller_model.dtypes)) == before_dtypes
        assert caller_model.inventory(0) == before_inventory[0]
        assert caller_model.inventory(0, 'totvol') == before_inventory[1]

    def test_host_side_adapter_does_not_call_provider(self):
        provider = FakeProvider(['this must not be requested'])
        capability = ScenarioReport()

        result = capability.run(
            ScenarioReportInputs(str(MODEL_DIR), MODEL_NAME),
            provider=provider,
            config=CONFIG,
        )

        assert result.ok is True
        assert provider.calls == []
