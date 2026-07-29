"""
Unit tests for ws3.integration module.

Tests FHOPSIntegrator, FEMICIntegrator, FreshForgeIntegrator,
SpaDESIntegrator, and RESTAPIServer classes.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from ws3.integration import (
    FHOPSIntegrator,
    FHOPSIntegrationConfig,
    FEMICIntegrator,
    FreshForgeIntegrator,
    SpaDESIntegrator,
    RESTAPIServer,
)


class TestFHOPSIntegrationConfig:
    """Tests for FHOPSIntegrationConfig dataclass."""

    def test_creation(self):
        """Test creating config with required fields."""
        config = FHOPSIntegrationConfig(inventory_file="inventory.csv")
        assert config.inventory_file == "inventory.csv"
        assert config.terrain_file is None
        assert config.output_dir == "."

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = FHOPSIntegrationConfig(
            inventory_file="inv.csv",
            terrain_file="terrain.tif",
            output_dir="/tmp/output"
        )
        d = config.to_dict()
        assert d["inventory_file"] == "inv.csv"
        assert d["terrain_file"] == "terrain.tif"
        assert d["output_dir"] == "/tmp/output"


class TestFHOPSIntegrator:
    """Tests for FHOPSIntegrator class."""

    def test_initialization_no_config(self):
        """Test creating integrator without config."""
        integrator = FHOPSIntegrator()
        assert integrator.config is None

    def test_initialization_with_config(self):
        """Test creating integrator with config."""
        config = FHOPSIntegrationConfig(inventory_file="inv.csv")
        integrator = FHOPSIntegrator(config)
        assert integrator.config == config

    @patch('ws3.integration.FHOPSIntegrator._check_fhops_available', return_value=False)
    def test_generate_cost_curves_not_available(self, mock_check):
        """Test cost curve generation when fhops not available."""
        integrator = FHOPSIntegrator()
        inventory = pd.DataFrame({'age': [20, 30, 40], 'volume': [100, 200, 300]})

        with pytest.raises(ImportError):
            integrator.generate_cost_curves(
                inventory=inventory,
                species="DF",
                site_index=50
            )

    @patch('ws3.integration.FHOPSIntegrator._check_fhops_available', return_value=True)
    def test_generate_cost_curves(self, mock_check):
        """Test cost curve generation (mocked)."""
        integrator = FHOPSIntegrator()
        inventory = pd.DataFrame({'age': [20, 30, 40], 'volume': [100, 200, 300]})

        result = integrator.generate_cost_curves(
            inventory=inventory,
            species="DF",
            site_index=50,
            distance_to_landing=200.0,
            slope=15.0
        )
        assert isinstance(result, pd.DataFrame)
        assert 'age' in result.columns
        assert 'cost_per_m3' in result.columns


class TestFEMICIntegrator:
    """Tests for FEMICIntegrator class."""

    def test_initialization(self):
        """Test creating integrator."""
        integrator = FEMICIntegrator()
        assert integrator._femic_available is False

    def test_calculate_carbon_budget(self):
        """Test carbon budget calculation (requires femic)."""
        try:
            import femic
        except ImportError:
            pytest.skip("femic not installed")
        
        integrator = FEMICIntegrator()

        schedule = pd.DataFrame({
            'area_ha': [100.0, 200.0],
            'period': [1, 1]
        })
        landscape = pd.DataFrame({'carbon': [150.0, 160.0]})

        result = integrator.calculate_carbon_budget(schedule, landscape)
        assert 'total_area_ha' in result
        assert 'carbon_stock_pre_harvest_tC' in result
        assert 'carbon_flux_tC' in result
        assert result['total_area_ha'] == 300.0

    def test_get_carbon_pools(self):
        """Test getting list of carbon pools."""
        integrator = FEMICIntegrator()
        pools = integrator.get_carbon_pools()
        assert isinstance(pools, list)
        assert 'above_ground_biomass' in pools


class TestFreshForgeIntegrator:
    """Tests for FreshForgeIntegrator class."""

    def test_initialization(self):
        """Test creating integrator."""
        integrator = FreshForgeIntegrator()
        assert integrator._freshforge_available is False

    def test_create_pipeline(self):
        """Test creating a pipeline (mocked - freshforge not available raises)."""
        integrator = FreshForgeIntegrator()
        # Since freshforge is not installed, this should raise ImportError
        with pytest.raises(ImportError):
            integrator.create_pipeline(
                name="test_pipeline",
                steps=[{"id": "step1", "type": "ws3.load"}]
            )

    def test_run_optimization_pipeline(self):
        """Test running an optimization pipeline (mocked)."""
        integrator = FreshForgeIntegrator()
        # Mock the _check_freshforge_available to return True
        integrator._freshforge_available = True

        result = integrator.run_optimization_pipeline(
            model_path="/path/to/model",
            scenario_name="test_scenario",
            objective="maximize_npv",
            output_dir="/tmp/output"
        )
        assert 'name' in result
        assert 'steps' in result
        assert len(result['steps']) == 4


class TestSpaDESIntegrator:
    """Tests for SpaDESIntegrator class."""

    def test_initialization(self):
        """Test creating integrator."""
        integrator = SpaDESIntegrator()
        assert integrator._spades_available is False

    def test_create_spades_config(self):
        """Test creating SpaDES configuration."""
        integrator = SpaDESIntegrator()
        config = integrator.create_spades_config(
            model_path="/path/to/model",
            landscape_raster="/path/to/raster.tif",
            scheduling_mode="optimize"
        )
        assert 'ws3' in config
        assert 'spades' in config
        assert 'integration' in config
        assert config['ws3']['model_path'] == "/path/to/model"


class TestRESTAPIServer:
    """Tests for RESTAPIServer class."""

    def test_initialization(self):
        """Test creating server."""
        server = RESTAPIServer(port=8080)
        assert server.port == 8080
        assert server.host == '0.0.0.0'
        assert server._app is None

    def test_create_app(self):
        """Test creating FastAPI app (mocked imports)."""
        server = RESTAPIServer()

        # Mock fastapi and pydantic imports
        mock_fastapi = MagicMock()
        mock_pydantic = MagicMock()
        mock_app_instance = MagicMock()
        mock_fastapi.FastAPI.return_value = mock_app_instance

        with patch.dict('sys.modules', {
            'fastapi': mock_fastapi,
            'fastapi.responses': MagicMock(),
            'pydantic': mock_pydantic,
        }):
            app = server.create_app()
            assert app is not None
            assert server._app is not None

    def test_run_server_not_installed(self):
        """Test run_server when uvicorn is not installed."""
        server = RESTAPIServer()
        server._app = MagicMock()

        with patch.dict('sys.modules', {'uvicorn': None}):
            # Should print a message but not crash
            server.run_server()
