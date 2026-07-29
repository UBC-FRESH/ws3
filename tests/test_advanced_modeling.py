"""
Unit tests for ws3.advanced_modeling module.

Tests StochasticOptimizer, MultiObjectiveOptimizer, DynamicPlanner,
and ClimateScenarioManager classes.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from ws3.advanced_modeling import (
    StochasticOptimizer,
    MultiObjectiveOptimizer,
    DynamicPlanner,
    ClimateScenarioManager,
    StochasticScenario,
    UncertaintyType,
)


class TestStochasticScenario:
    """Tests for StochasticScenario dataclass."""

    def test_creation(self):
        """Test creating a scenario with required fields."""
        scenario = StochasticScenario(name="test", probability=0.5)
        assert scenario.name == "test"
        assert scenario.probability == 0.5
        assert scenario.parameters == {}

    def test_to_dict(self):
        """Test converting scenario to dictionary."""
        scenario = StochasticScenario(
            name="test",
            probability=0.5,
            parameters={"growth_factor": 1.2}
        )
        d = scenario.to_dict()
        assert d["name"] == "test"
        assert d["probability"] == 0.5
        assert d["parameters"]["growth_factor"] == 1.2


class TestStochasticOptimizer:
    """Tests for StochasticOptimizer class."""

    def test_initialization(self):
        """Test creating optimizer with a problem."""
        mock_problem = MagicMock()
        optimizer = StochasticOptimizer(mock_problem)
        assert optimizer.problem == mock_problem
        assert optimizer.scenarios == []

    def test_add_scenario(self):
        """Test adding a scenario."""
        mock_problem = MagicMock()
        optimizer = StochasticOptimizer(mock_problem)
        scenario = StochasticScenario(name="s1", probability=0.5)
        optimizer.add_scenario(scenario)
        assert len(optimizer.scenarios) == 1
        assert optimizer.scenarios[0].name == "s1"

    def test_generate_scenarios_growth(self):
        """Test generating growth uncertainty scenarios."""
        mock_problem = MagicMock()
        optimizer = StochasticOptimizer(mock_problem)
        scenarios = optimizer.generate_scenarios(
            UncertaintyType.GROWTH,
            n_scenarios=10,
            mean=1.0,
            std=0.1
        )
        assert len(scenarios) == 10
        for s in scenarios:
            assert "growth_factor" in s.parameters
            assert 0.5 <= s.parameters["growth_factor"] <= 1.5  # Reasonable range

    def test_generate_scenarios_prices(self):
        """Test generating price uncertainty scenarios."""
        mock_problem = MagicMock()
        optimizer = StochasticOptimizer(mock_problem)
        scenarios = optimizer.generate_scenarios(
            UncertaintyType.PRICES,
            n_scenarios=5
        )
        assert len(scenarios) == 5
        for s in scenarios:
            assert "price_factor" in s.parameters

    def test_generate_scenarios_climate(self):
        """Test generating climate scenarios."""
        mock_problem = MagicMock()
        optimizer = StochasticOptimizer(mock_problem)
        scenarios = optimizer.generate_scenarios(
            UncertaintyType.CLIMATE,
            n_scenarios=5
        )
        assert len(scenarios) == 5
        for s in scenarios:
            assert "temperature_anomaly" in s.parameters
            assert "precipitation_anomaly" in s.parameters


class TestMultiObjectiveOptimizer:
    """Tests for MultiObjectiveOptimizer class."""

    def test_initialization(self):
        """Test creating optimizer with a problem."""
        mock_problem = MagicMock()
        optimizer = MultiObjectiveOptimizer(mock_problem)
        assert optimizer.problem == mock_problem
        assert optimizer.objectives == []

    def test_add_objective(self):
        """Test adding an objective."""
        mock_problem = MagicMock()
        optimizer = MultiObjectiveOptimizer(mock_problem)
        optimizer.add_objective("npv", weight=0.5, direction="maximize")
        assert len(optimizer.objectives) == 1
        assert optimizer.objectives[0]["name"] == "npv"

    def test_solve_weighted_sum(self):
        """Test weighted sum optimization (mocked)."""
        mock_problem = MagicMock()
        mock_problem.get_objective_value.return_value = 100.0
        mock_problem.get_solution.return_value = {"x": [1, 2, 3]}
        optimizer = MultiObjectiveOptimizer(mock_problem)
        optimizer.add_objective("npv", weight=1.0, direction="maximize")
        result = optimizer.solve_weighted_sum()
        assert result["method"] == "weighted_sum"
        assert result["objective_values"] == {}


class TestDynamicPlanner:
    """Tests for DynamicPlanner class."""

    def test_initialization(self):
        """Test creating planner with a problem."""
        mock_problem = MagicMock()
        planner = DynamicPlanner(mock_problem, n_periods=10)
        assert planner.problem == mock_problem
        assert planner.n_periods == 10
        assert planner.plans == []

    def test_plan_static(self):
        """Test static planning (mocked)."""
        mock_problem = MagicMock()
        mock_problem.get_objective_value.return_value = 100.0
        mock_problem.get_solution.return_value = {"x": [1, 2, 3]}
        planner = DynamicPlanner(mock_problem, n_periods=5)
        plan = planner.plan_static()
        assert plan["type"] == "static"
        assert plan["n_periods"] == 5
        assert len(planner.plans) == 1

    def test_plan_dynamic(self):
        """Test dynamic planning with re-optimization (mocked)."""
        mock_problem = MagicMock()
        mock_problem.get_objective_value.return_value = 100.0
        mock_problem.get_solution.return_value = {"x": [1, 2, 3]}
        planner = DynamicPlanner(mock_problem, n_periods=10)
        plan = planner.plan_dynamic(reoptimize_every=5)
        assert plan["type"] == "dynamic"
        assert plan["reoptimize_every"] == 5
        assert len(plan["plans"]) == 2  # 0 and 5


class TestClimateScenarioManager:
    """Tests for ClimateScenarioManager class."""

    def test_initialization(self):
        """Test creating manager with no arguments."""
        manager = ClimateScenarioManager()
        assert manager.scenarios == []

    def test_add_scenario(self):
        """Test adding a climate scenario."""
        manager = ClimateScenarioManager()
        manager.add_scenario(
            name="RCP85",
            temperature_change=4.5,
            precipitation_change=0.2,
            co2_change=936
        )
        assert len(manager.scenarios) == 1
        assert manager.scenarios[0]["name"] == "RCP85"
        assert manager.scenarios[0]["temperature_change"] == 4.5

    def test_get_rcp_scenarios(self):
        """Test getting standard RCP scenarios."""
        manager = ClimateScenarioManager()
        scenarios = manager.get_rcp_scenarios()
        assert len(scenarios) == 4
        names = [s["name"] for s in scenarios]
        assert "RCP2.6" in names
        assert "RCP4.5" in names
        assert "RCP6.0" in names
        assert "RCP8.5" in names


class TestUncertaintyType:
    """Tests for UncertaintyType enum."""

    def test_values(self):
        """Test enum values."""
        assert UncertaintyType.GROWTH.value == "growth"
        assert UncertaintyType.PRICES.value == "prices"
        assert UncertaintyType.DEMAND.value == "demand"
        assert UncertaintyType.DISTURBANCE.value == "disturbance"
        assert UncertaintyType.CLIMATE.value == "climate"
