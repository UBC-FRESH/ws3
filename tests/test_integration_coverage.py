"""
Targeted coverage tests for ws3.integration.

Covers uncovered branches:
- FHOPSIntegrator when fhops not available
- FEMICIntegrator when femic not available
- FreshForgeIntegrator when freshforge not available
- SpaDESIntegrator when reticulate not available
- RESTAPIServer without fastapi
- Convenience factory functions
- FHOPSIntegrator.generate_cost_curves / export_curves
- FEMICIntegrator.calculate_carbon_budget / export_carbon_report
- FreshForgeIntegrator.create_pipeline / export_pipeline / run_optimization_pipeline
- SpaDESIntegrator.create_spades_config / export_config
- FHOPSIntegrator.inject_into_model → NotImplementedError
- RESTAPIServer.run_server without uvicorn
"""

import json
import sys

sys.path.append('../ws3/')

import pandas as pd
import pytest

from ws3.integration import (
    FEMICIntegrator,
    FHOPSIntegrationConfig,
    FHOPSIntegrator,
    FreshForgeIntegrator,
    RESTAPIServer,
    SpaDESIntegrator,
    create_femic_integrator,
    create_fhops_integrator,
    create_freshforge_integrator,
    create_rest_api,
    create_spades_integrator,
)

# ---------------------------------------------------------------------------
# FHOPSIntegrator
# ---------------------------------------------------------------------------

class TestFHOPSIntegrator:
    def test_check_fhops_unavailable(self):
        """fhops is not installed → _fhops_available is False."""
        integ = FHOPSIntegrator()
        assert integ._fhops_available is False

    def test_generate_cost_curves_raises_when_unavailable(self):
        integ = FHOPSIntegrator()
        df = pd.DataFrame({'age': [10], 'species': ['sw']})
        with pytest.raises(ImportError, match="fhops is not installed"):
            integ.generate_cost_curves(df, "sw", 50.0)

    def test_generate_cost_curves_when_available(self, monkeypatch):
        """Mock fhops as available and test cost curve generation."""
        monkeypatch.setattr(
            "ws3.integration.FHOPSIntegrator._check_fhops_available",
            lambda self: True,
        )
        integ = FHOPSIntegrator()
        df = pd.DataFrame({'age': [10, 20, 30], 'species': ['sw', 'sw', 'sw']})
        result = integ.generate_cost_curves(df, "sw", 50.0, distance_to_landing=200.0, slope=15.0)
        assert isinstance(result, pd.DataFrame)
        assert 'age' in result.columns
        assert 'volume_m3_ha' in result.columns
        assert 'cost_per_m3' in result.columns
        assert len(result) > 0

    def test_export_curves(self, tmp_path):
        integ = FHOPSIntegrator()
        df = pd.DataFrame({'age': [10, 20], 'cost_per_m3': [50.0, 45.0]})
        path = str(tmp_path / "costs.csv")
        integ.export_curves(df, path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_inject_into_model_raises(self):
        integ = FHOPSIntegrator()
        df = pd.DataFrame({'age': [10], 'cost_per_m3': [50.0]})
        with pytest.raises(NotImplementedError, match="inject_into_model"):
            integ.inject_into_model(None, df, "sw", 50.0)

    def test_config_to_dict(self):
        cfg = FHOPSIntegrationConfig(
            inventory_file="inv.csv",
            terrain_file="terrain.tif",
            roads_file="roads.shp",
            output_dir="/tmp/out",
        )
        d = cfg.to_dict()
        assert d['inventory_file'] == "inv.csv"
        assert d['terrain_file'] == "terrain.tif"
        assert d['roads_file'] == "roads.shp"
        assert d['output_dir'] == "/tmp/out"


# ---------------------------------------------------------------------------
# FEMICIntegrator
# ---------------------------------------------------------------------------

class TestFEMICIntegrator:
    def test_check_femic_unavailable(self):
        integ = FEMICIntegrator()
        assert integ._femic_available is False

    def test_calculate_carbon_budget_raises_when_unavailable(self):
        integ = FEMICIntegrator()
        sched = pd.DataFrame({'area_ha': [100], 'period': [1]})
        land = pd.DataFrame({'carbon': [150]})
        with pytest.raises(ImportError, match="femic is not installed"):
            integ.calculate_carbon_budget(sched, land)

    def test_calculate_carbon_budget_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "ws3.integration.FEMICIntegrator._check_femic_available",
            lambda self: True,
        )
        integ = FEMICIntegrator()
        sched = pd.DataFrame({'area_ha': [100.0, 200.0], 'period': [1, 1]})
        land = pd.DataFrame({'carbon': [150, 160]})
        result = integ.calculate_carbon_budget(sched, land)
        assert isinstance(result, dict)
        assert result['total_area_ha'] == 300.0
        assert result['carbon_flux_tC'] > 0

    def test_get_carbon_pools(self):
        integ = FEMICIntegrator()
        pools = integ.get_carbon_pools()
        assert isinstance(pools, list)
        assert len(pools) > 0
        assert 'above_ground_biomass' in pools

    def test_export_carbon_report(self, tmp_path):
        integ = FEMICIntegrator()
        budget = {'total_area_ha': 100.0, 'carbon_flux_tC': 5000.0}
        path = str(tmp_path / "carbon.json")
        integ.export_carbon_report(budget, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['total_area_ha'] == 100.0


# ---------------------------------------------------------------------------
# FreshForgeIntegrator
# ---------------------------------------------------------------------------

class TestFreshForgeIntegrator:
    def test_check_freshforge_unavailable(self):
        integ = FreshForgeIntegrator()
        assert integ._freshforge_available is False

    def test_create_pipeline_raises_when_unavailable(self):
        integ = FreshForgeIntegrator()
        with pytest.raises(ImportError, match="freshforge is not installed"):
            integ.create_pipeline("test", [])

    def test_create_pipeline_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "ws3.integration.FreshForgeIntegrator._check_freshforge_available",
            lambda self: True,
        )
        integ = FreshForgeIntegrator()
        steps = [{'id': 'step1', 'type': 'ws3.load'}]
        result = integ.create_pipeline("test_pipeline", steps)
        assert result['name'] == "test_pipeline"
        assert result['version'] == '1.0'
        assert 'created' in result
        assert result['steps'] == steps

    def test_run_optimization_pipeline(self, monkeypatch):
        monkeypatch.setattr(
            "ws3.integration.FreshForgeIntegrator._check_freshforge_available",
            lambda self: True,
        )
        integ = FreshForgeIntegrator()
        result = integ.run_optimization_pipeline(
            model_path="/data",
            scenario_name="test",
            objective="max_npv",
            output_dir="/out",
        )
        assert 'steps' in result
        step_ids = [s['id'] for s in result['steps']]
        assert 'load_model' in step_ids
        assert 'solve' in step_ids

    def test_export_pipeline(self, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "ws3.integration.FreshForgeIntegrator._check_freshforge_available",
            lambda self: True,
        )
        integ = FreshForgeIntegrator()
        pipeline = {'name': 'test', 'steps': []}
        path = str(tmp_path / "pipeline.json")
        integ.export_pipeline(pipeline, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['name'] == 'test'


# ---------------------------------------------------------------------------
# SpaDESIntegrator
# ---------------------------------------------------------------------------

class TestSpaDESIntegrator:
    def test_check_spades_unavailable(self):
        integ = SpaDESIntegrator()
        assert integ._spades_available is False

    def test_create_spades_config(self):
        integ = SpaDESIntegrator()
        config = integ.create_spades_config(
            model_path="/data",
            landscape_raster="/raster.tif",
            scheduling_mode="optimize",
        )
        assert 'ws3' in config
        assert 'spades' in config
        assert 'integration' in config
        assert config['ws3']['model_path'] == "/data"
        assert config['spades']['scheduling_mode'] == "optimize"

    def test_export_config(self, tmp_path):
        integ = SpaDESIntegrator()
        config = integ.create_spades_config("/data", "/raster.tif")
        path = str(tmp_path / "spades.json")
        integ.export_config(config, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded['ws3']['model_path'] == "/data"


# ---------------------------------------------------------------------------
# RESTAPIServer
# ---------------------------------------------------------------------------

class TestRESTAPIServer:
    def test_init_defaults(self):
        srv = RESTAPIServer()
        assert srv.host == '0.0.0.0'
        assert srv.port == 8000
        assert srv._app is None

    def test_create_app_without_fastapi_raises(self):
        srv = RESTAPIServer()
        with pytest.raises(ImportError, match="FastAPI is required"):
            srv.create_app()

    def test_run_server_without_app_creates_one(self, monkeypatch):
        """run_server with no app tries to create one; without fastapi it raises."""
        srv = RESTAPIServer()
        with pytest.raises(ImportError):
            srv.run_server()

    def test_run_server_without_uvicorn(self, monkeypatch):
        """If FastAPI somehow succeeds but uvicorn is missing, prints message."""
        srv = RESTAPIServer()
        fake_app = object()
        # Mock uvicorn import to fail
        imported = {}
        def fake_import(name, *args, **kwargs):
            if name == 'uvicorn':
                raise ImportError("no uvicorn")
            return imported.get(name)
        monkeypatch.setattr('builtins.__import__', fake_import)
        # Should not crash — prints message
        srv.run_server(fake_app)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_create_fhops_integrator(self):
        integ = create_fhops_integrator()
        assert isinstance(integ, FHOPSIntegrator)

    def test_create_femic_integrator(self):
        integ = create_femic_integrator()
        assert isinstance(integ, FEMICIntegrator)

    def test_create_freshforge_integrator(self):
        integ = create_freshforge_integrator()
        assert isinstance(integ, FreshForgeIntegrator)

    def test_create_spades_integrator(self):
        integ = create_spades_integrator()
        assert isinstance(integ, SpaDESIntegrator)

    def test_create_rest_api(self):
        srv = create_rest_api(host='127.0.0.1', port=9000)
        assert isinstance(srv, RESTAPIServer)
        assert srv.host == '127.0.0.1'
        assert srv.port == 9000
