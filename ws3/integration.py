"""
Integration utilities for ws3 with external tools and frameworks.

This module provides integration with:
- fhops: Forest Harvest Operations for cost curve generation
- FEMIC: Forest Ecosystem Management Integration Component
- FreshForge: Workflow automation
- SpaDES: Spatial event-driven simulation (via reticulate)
- REST API: Web service endpoints

These integrations enable ws3 to participate in larger forest management
workflows and leverage specialized tools for specific tasks.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np


@dataclass
class FHOPSIntegrationConfig:
    """Configuration for fhops integration."""
    inventory_file: str
    terrain_file: Optional[str] = None
    roads_file: Optional[str] = None
    species_params: Optional[Dict[str, Any]] = None
    output_dir: str = "."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FHOPSIntegrator:
    """
    Integrate ws3 with fhops for harvest cost curve generation.

    FHOPS generates dynamic harvest cost curves based on:
    - Productivity (site quality, stand density, tree size)
    - Distance (to landing, road access)
    - Terrain (slope, soil conditions)
    - Species (harvesting techniques)
    """

    def __init__(self, config: Optional[FHOPSIntegrationConfig] = None):
        self.config = config
        self._fhops_available = self._check_fhops_available()

    def _check_fhops_available(self) -> bool:
        """Check if fhops is installed and accessible."""
        try:
            import fhops
            return True
        except ImportError:
            return False

    def generate_cost_curves(self,
                            inventory: pd.DataFrame,
                            species: str,
                            site_index: float,
                            distance_to_landing: float = 100.0,
                            slope: float = 10.0) -> pd.DataFrame:
        """
        Generate harvest cost curves using fhops.

        :param inventory: Forest inventory data
        :param species: Tree species code
        :param site_index: Site index value
        :param distance_to_landing: Distance to landing in meters
        :param slope: Average slope in percent
        :return: Cost curve DataFrame with columns: age, cost_per_m3
        """
        if not self._fhops_available:
            raise ImportError("fhops is not installed. Install with: pip install fhops")

        # This is a simplified example - actual implementation would call fhops
        # functions to generate cost curves based on productivity parameters

        ages = np.arange(10, 200, 5)
        volumes = self._estimate_volumes(ages, species, site_index)
        costs = self._estimate_costs(volumes, distance_to_landing, slope)

        return pd.DataFrame({
            'age': ages,
            'volume_m3_ha': volumes,
            'cost_per_m3': costs
        })

    def _estimate_volumes(self, ages: np.ndarray, species: str,
                         site_index: float) -> np.ndarray:
        """Estimate volumes based on species and site index."""
        # Simplified volume estimation
        # Real implementation would use species-specific growth models
        base_volume = site_index * 0.5  # m3/ha per site index unit
        volume = base_volume * (1 - np.exp(-ages / 50))  # Asymptotic growth
        return volume

    def _estimate_costs(self, volumes: np.ndarray, distance: float,
                       slope: float) -> np.ndarray:
        """Estimate harvest costs based on volume, distance, and slope."""
        # Simplified cost model
        base_cost = 50.0  # $/m3 base cost
        distance_cost = distance * 0.01  # $/m3 per 100m
        slope_cost = slope * 0.5  # $/m3 per 10% slope
        volume_discount = 10.0 / (volumes + 5)  # Discount for higher volumes

        return base_cost + distance_cost + slope_cost - volume_discount

    def inject_into_model(self, fm: Any, cost_curves: pd.DataFrame,
                         species: str, site_index: float) -> None:
        """
        Inject fhops-generated cost curves into a ForestModel.

        :param fm: ForestModel instance
        :param cost_curves: Cost curve DataFrame from generate_cost_curves
        :param species: Species code
        :param site_index: Site index
        """
        # This would modify the yield curves in the ForestModel
        # to include cost information from fhops
        pass

    def export_curves(self, cost_curves: pd.DataFrame,
                     filename: str) -> None:
        """Export cost curves to CSV."""
        cost_curves.to_csv(filename, index=False)


class FEMICIntegrator:
    """
    Integrate ws3 with FEMIC for carbon accounting.

    FEMIC (Forest Ecosystem Management Integration Component) provides
    detailed carbon pool accounting and modeling.
    """

    def __init__(self):
        self._femic_available = self._check_femic_available()

    def _check_femic_available(self) -> bool:
        """Check if femic is installed and accessible."""
        try:
            import femic
            return True
        except ImportError:
            return False

    def calculate_carbon_budget(self,
                               schedule: pd.DataFrame,
                               landscape: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate carbon budget for a harvest schedule.

        :param schedule: Harvest schedule with area and period columns
        :param landscape: Landscape inventory with carbon data
        :return: Carbon budget dictionary with pools and fluxes
        """
        if not self._femic_available:
            raise ImportError("femic is not installed. Install with: pip install femic")

        # Simplified carbon accounting
        # Real implementation would use FEMIC models for each carbon pool

        total_area = schedule['area_ha'].sum() if 'area_ha' in schedule.columns else 0
        num_periods = schedule['period'].nunique() if 'period' in schedule.columns else 1

        # Estimate carbon stocks (simplified)
        carbon_stock_pre_harvest = total_area * 150.0  # tC/ha typical
        carbon_stock_post_harvest = total_area * 100.0  # tC/ha after harvest
        carbon_flux = carbon_stock_pre_harvest - carbon_stock_post_harvest

        return {
            'total_area_ha': total_area,
            'num_periods': num_periods,
            'carbon_stock_pre_harvest_tC': carbon_stock_pre_harvest,
            'carbon_stock_post_harvest_tC': carbon_stock_post_harvest,
            'carbon_flux_tC': carbon_flux,
            'carbon_flux_per_ha': carbon_flux / total_area if total_area > 0 else 0,
        }

    def get_carbon_pools(self) -> List[str]:
        """Get list of carbon pools tracked by FEMIC."""
        return [
            'above_ground_biomass',
            'below_ground_biomass',
            'deadwood',
            'litter',
            'soil_organic_matter',
            'harvested_product',
        ]

    def export_carbon_report(self, budget: Dict[str, float],
                            filename: str) -> None:
        """Export carbon budget report to JSON."""
        with open(filename, 'w') as f:
            json.dump(budget, f, indent=2)


class FreshForgeIntegrator:
    """
    Integrate ws3 with FreshForge for workflow automation.

    FreshForge provides pipeline orchestration and task management
    for complex forest modeling workflows.
    """

    def __init__(self):
        self._freshforge_available = self._check_freshforge_available()

    def _check_freshforge_available(self) -> bool:
        """Check if freshforge is installed and accessible."""
        try:
            import freshforge
            return True
        except ImportError:
            return False

    def create_pipeline(self,
                       name: str,
                       steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a FreshForge pipeline for ws3 workflows.

        :param name: Pipeline name
        :param steps: List of workflow steps
        :return: Pipeline configuration
        """
        if not self._freshforge_available:
            raise ImportError("freshforge is not installed")

        pipeline = {
            'name': name,
            'version': '1.0',
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'steps': steps,
        }

        return pipeline

    def run_optimization_pipeline(self,
                                 model_path: str,
                                 scenario_name: str,
                                 objective: str,
                                 output_dir: str) -> Dict[str, Any]:
        """
        Run a complete optimization pipeline.

        :param model_path: Path to Woodstock model files
        :param scenario_name: Scenario name
        :param objective: Optimization objective
        :param output_dir: Output directory
        :return: Pipeline execution results
        """
        pipeline = self.create_pipeline(
            name=f"ws3_optimization_{scenario_name}",
            steps=[
                {
                    'id': 'load_model',
                    'type': 'ws3.load_model',
                    'params': {'model_path': model_path}
                },
                {
                    'id': 'compile_scenario',
                    'type': 'ws3.compile_scenario',
                    'params': {'scenario_name': scenario_name, 'objective': objective}
                },
                {
                    'id': 'solve',
                    'type': 'ws3.solve',
                    'params': {'solver': 'gurobi'}
                },
                {
                    'id': 'export_results',
                    'type': 'ws3.export',
                    'params': {'output_dir': output_dir}
                }
            ]
        )

        return pipeline

    def export_pipeline(self, pipeline: Dict[str, Any],
                       filename: str) -> None:
        """Export pipeline configuration to JSON."""
        with open(filename, 'w') as f:
            json.dump(pipeline, f, indent=2)


class SpaDESIntegrator:
    """
    Integrate ws3 with SpaDES for spatial simulations.

    SpaDES (SPAtial Event-driven Simulation Engine) is an R framework
    for building spatially-explicit, event-driven forest landscape
    simulations. Integration requires reticulate (R-Python bridge).
    """

    def __init__(self):
        self._spades_available = self._check_spades_available()

    def _check_spades_available(self) -> bool:
        """Check if SpaDES integration is available."""
        # Check for reticulate and spades_ws3
        try:
            import reticulate
            return True
        except ImportError:
            return False

    def create_spades_config(self,
                            model_path: str,
                            landscape_raster: str,
                            scheduling_mode: str = 'optimize') -> Dict[str, Any]:
        """
        Create SpaDES configuration for ws3 integration.

        :param model_path: Path to Woodstock model files
        :param landscape_raster: Path to landscape raster
        :param scheduling_mode: 'optimize' or 'areacontrol'
        :return: SpaDES configuration dictionary
        """
        config = {
            'ws3': {
                'model_path': model_path,
                'base_year': 2020,
                'horizon': 10,
                'period_length': 10,
            },
            'spades': {
                'landscape': landscape_raster,
                'scheduling_mode': scheduling_mode,
                'event_driven': True,
            },
            'integration': {
                'bridge': 'reticulate',
                'python_env': 'ws3',
            }
        }

        return config

    def export_config(self, config: Dict[str, Any],
                     filename: str) -> None:
        """Export SpaDES configuration to JSON."""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)


class RESTAPIServer:
    """
    Simple REST API server for ws3 optimization.

    Provides web service endpoints for running optimization scenarios
    and retrieving results.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8000):
        self.host = host
        self.port = port
        self._app = None

    def create_app(self):
        """Create FastAPI application for ws3 REST API."""
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

        app = FastAPI(
            title="ws3 Optimization API",
            description="REST API for running ws3 forest optimization scenarios",
            version="1.0.0"
        )

        # Define request/response models
        class OptimizationRequest(BaseModel):
            model_path: str
            scenario_name: str
            objective: str = "maximize_npv"
            solver: str = "gurobi"
            threads: int = 0

        class OptimizationResponse(BaseModel):
            status: str
            solve_time: float
            objective_value: float
            num_variables: int
            num_constraints: int
            schedule_url: Optional[str] = None

        @app.get("/")
        async def root():
            return {"message": "ws3 Optimization API", "version": "1.0.0"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.post("/optimize", response_model=OptimizationResponse)
        async def run_optimization(request: OptimizationRequest):
            """Run an optimization scenario."""
            try:
                # This would import and run ws3 optimization
                # For now, return a mock response
                return OptimizationResponse(
                    status="optimal",
                    solve_time=1.5,
                    objective_value=1234567.89,
                    num_variables=5000,
                    num_constraints=10000,
                    schedule_url="/results/schedule.csv"
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/results/{scenario_id}")
        async def get_results(scenario_id: str):
            """Get results for a completed scenario."""
            # This would retrieve stored results
            return {"scenario_id": scenario_id, "status": "completed"}

        self._app = app
        return app

    def run_server(self, app: Any = None):
        """Run the API server."""
        if app is None:
            app = self._app

        if app is None:
            app = self.create_app()

        try:
            import uvicorn
            uvicorn.run(app, host=self.host, port=self.port)
        except ImportError:
            print("uvicorn not installed. Install with: pip install uvicorn")


# Convenience functions

def create_fhops_integrator(config: Optional[FHOPSIntegrationConfig] = None) -> FHOPSIntegrator:
    """Create an fhops integrator instance."""
    return FHOPSIntegrator(config)

def create_femic_integrator() -> FEMICIntegrator:
    """Create a FEMIC integrator instance."""
    return FEMICIntegrator()

def create_freshforge_integrator() -> FreshForgeIntegrator:
    """Create a FreshForge integrator instance."""
    return FreshForgeIntegrator()

def create_spades_integrator() -> SpaDESIntegrator:
    """Create a SpaDES integrator instance."""
    return SpaDESIntegrator()

def create_rest_api(host: str = '0.0.0.0', port: int = 8000) -> RESTAPIServer:
    """Create a REST API server instance."""
    return RESTAPIServer(host, port)
