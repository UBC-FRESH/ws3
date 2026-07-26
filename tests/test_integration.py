"""
Integration tests for ws3 modules.

Tests interaction between different ws3 components.
"""

import sys
sys.path.append('../ws3/')

import pytest
import pandas as pd
import numpy as np
from ws3.forest import ForestModel
from ws3.opt import Problem, Variable, Constraint
from ws3.core import interpolate_curves


class TestForestOptIntegration:
    """Test integration between ForestModel and optimization."""
    
    def test_forest_to_opt_conversion(self):
        """Test converting ForestModel to optimization problem."""
        try:
            fm = ForestModel(
                model_name="test",
                model_path="data/woodstock_model_files_tsa24"
            )
            fm.import_landscape_section()
            fm.import_areas_section(convert_periods_to_years=10)
            fm.import_yields_section(convert_periods_to_years=10)
            fm.import_actions_section(convert_periods_to_years=10)
            fm.import_transitions_section(convert_periods_to_years=10)
            fm.initialize_areas()
            fm.add_null_action()
            fm.reset_actions()
            
            # Check that model has required components
            assert len(fm.development_types) > 0, "No development types"
            assert len(fm.yields) > 0, "No yield curves"
            assert len(fm.actions) > 0, "No actions"
            
            # Compile scenario
            from ws3.core import compile_scenario
            problem = compile_scenario(fm, scenario_name="integration_test")
            
            # Check problem structure
            assert len(problem._vars) > 0, "No variables in problem"
            assert len(problem._constraints) > 0, "No constraints in problem"
            
        except FileNotFoundError:
            pytest.skip("Test data not available")
    
    def test_yield_curve_interpolation(self):
        """Test yield curve interpolation in optimization context."""
        try:
            fm = ForestModel(
                model_name="test",
                model_path="data/woodstock_model_files_tsa24"
            )
            fm.import_yields_section(convert_periods_to_years=10)
            
            # Get a yield curve
            if len(fm.yields) > 0:
                first_key = list(fm.yields.keys())[0]
                curve = fm.yields[first_key]
                
                # Check that curve has data
                assert len(curve) > 0, "Empty yield curve"
                assert 'age' in curve.columns, "Missing age column"
                assert 'volume' in curve.columns, "Missing volume column"
                
                # Test interpolation
                test_ages = np.array([10, 20, 30, 40, 50])
                volumes = interpolate_curves(curve, test_ages)
                
                assert len(volumes) == len(test_ages), "Wrong number of volumes"
                assert all(v >= 0 for v in volumes), "Negative volumes"
                
        except FileNotFoundError:
            pytest.skip("Test data not available")
    
    def test_action_definitions(self):
        """Test action definitions in optimization context."""
        try:
            fm = ForestModel(
                model_name="test",
                model_path="data/woodstock_model_files_tsa24"
            )
            fm.import_actions_section(convert_periods_to_years=10)
            
            # Check that actions are defined
            assert len(fm.actions) > 0, "No actions defined"
            
            # Check that harvest action exists
            assert "harvest" in fm.actions, "No harvest action"
            assert fm.actions["harvest"].is_harvest, "Harvest action not marked as harvest"
            
        except FileNotFoundError:
            pytest.skip("Test data not available")


class TestCoreOptIntegration:
    """Test integration between core and opt modules."""
    
    def test_variable_constraint_interaction(self):
        """Test that variables and constraints interact correctly."""
        problem = Problem()
        
        # Add variables
        x = Variable("x", "continuous", 0, 100)
        y = Variable("y", "continuous", 0, 100)
        
        problem.add_variable("x", "continuous", 0, 100)
        problem.add_variable("y", "continuous", 0, 100)
        
        # Add constraint
        problem.add_constraint("con1", {"x": 1.0, "y": 1.0}, "<", 150)
        
        # Set objective
        problem.set_objective({"x": 1.0, "y": 2.0})
        
        # Solve
        problem.solve(solver="highs")
        
        # Check solution
        solution = problem.get_solution()
        assert "x" in solution, "Missing x in solution"
        assert "y" in solution, "Missing y in solution"
        
        # Check constraint is satisfied
        lhs = solution["x"] + solution["y"]
        assert lhs <= 150.0 + 1e-6, f"Constraint violated: {lhs} > 150"
    
    def test_multiple_objectives(self):
        """Test optimization with multiple objectives."""
        problem = Problem()
        
        for i in range(10):
            problem.add_variable(f"x{i}", "continuous", 0, 100)
        
        problem.add_constraint("con1", {f"x{i}": 1.0 for i in range(10)}, "<", 200)
        
        # Set multi-objective (weighted sum)
        objective = {f"x{i}": float(i+1) for i in range(10)}
        problem.set_objective(objective)
        
        problem.solve(solver="highs")
        
        # Check that problem solved
        assert problem.status() == "optimal"
        
        # Check that solution is reasonable
        solution = problem.get_solution()
        total = sum(solution.values())
        assert total > 0, "Total objective value should be positive"


class TestSpatialOptIntegration:
    """Test integration between spatial and optimization modules."""
    
    def test_spatial_constraints(self):
        """Test adding spatial constraints to optimization."""
        try:
            import geopandas as gpd
            
            # Create simple spatial data
            from shapely.geometry import Polygon
            
            polygons = [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
            ]
            
            spatial_df = gpd.GeoDataFrame({
                'dt_code': ['DT1', 'DT2', 'DT3'],
                'area_ha': [100.0, 100.0, 100.0],
                'geometry': polygons
            })
            
            # Create adjacency matrix
            adj_matrix = np.zeros((3, 3), dtype=int)
            adj_matrix[0, 1] = 1
            adj_matrix[1, 0] = 1
            adj_matrix[0, 2] = 1
            adj_matrix[2, 0] = 1
            
            # Create optimization problem
            problem = Problem()
            
            for i in range(3):
                problem.add_variable(f"x{i}", "binary", 0, 1)
            
            # Add adjacency constraints
            for i in range(3):
                for j in range(i+1, 3):
                    if adj_matrix[i, j] == 1:
                        problem.add_constraint(
                            f"adj_{i}_{j}",
                            {f"x{i}": 1.0, f"x{j}": 1.0},
                            "<", 1
                        )
            
            # Set objective
            problem.set_objective({f"x{i}": float(i+1) for i in range(3)})
            
            # Solve
            problem.solve(solver="highs")
            
            # Check solution
            solution = problem.get_solution()
            assert problem.status() == "optimal"
            
            # Verify adjacency constraints
            if solution["x0"] > 0.5 and solution["x1"] > 0.5:
                pytest.fail("Adjacent areas both harvested")
            
        except ImportError:
            pytest.skip("geopandas or shapely not available")


class TestFinancialOptIntegration:
    """Test integration between financial and optimization modules."""
    
    def test_financial_objective(self):
        """Test financial objective in optimization."""
        problem = Problem()
        
        # Add variables for harvest in each period
        for t in range(5):
            problem.add_variable(f"harvest_t{t}", "continuous", 0, 1000)
        
        # Add even-flow constraint
        problem.add_constraint(
            "even_flow",
            {f"harvest_t{t}": 1.0 for t in range(5)},
            "=",
            500  # Total harvest
        )
        
        # Set financial objective (simplified NPV)
        discount_rate = 0.05
        objective = {}
        for t in range(5):
            npv_factor = 1.0 / (1 + discount_rate) ** t
            objective[f"harvest_t{t}"] = npv_factor * 100  # $100 per unit
        
        problem.set_objective(objective)
        
        # Solve
        problem.solve(solver="highs")
        
        # Check solution
        assert problem.status() == "optimal"
        
        solution = problem.get_solution()
        
        # Check even-flow constraint
        total_harvest = sum(solution[f"harvest_t{t}"] for t in range(5))
        assert abs(total_harvest - 500) < 1e-6, f"Even-flow constraint violated: {total_harvest}"
        
        # Check that earlier periods have higher harvest (due to discounting)
        assert solution["harvest_t0"] >= solution["harvest_t4"] - 1e-6, \
            "Earlier periods should have higher harvest due to discounting"


class TestReproducibility:
    """Test reproducibility of optimization results."""
    
    def test_deterministic_solving(self):
        """Test that solving same problem gives same results."""
        problem1 = Problem()
        problem2 = Problem()
        
        # Add same variables
        for i in range(20):
            problem1.add_variable(f"x{i}", "continuous", 0, 100)
            problem2.add_variable(f"x{i}", "continuous", 0, 100)
        
        # Add same constraints
        for i in range(5):
            coeffs = {f"x{j}": 1.0 for j in range(i, min(i+3, 20))}
            problem1.add_constraint(f"con{i}", coeffs, "<", 50)
            problem2.add_constraint(f"con{i}", coeffs, "<", 50)
        
        # Set same objective
        objective = {f"x{i}": float(i+1) for i in range(20)}
        problem1.set_objective(objective)
        problem2.set_objective(objective)
        
        # Solve both
        problem1.solve(solver="highs")
        problem2.solve(solver="highs")
        
        # Get solutions
        sol1 = problem1.get_solution()
        sol2 = problem2.get_solution()
        
        # Check that solutions are identical
        for var in sol1:
            assert abs(sol1[var] - sol2[var]) < 1e-6, \
                f"Solution for {var} differs: {sol1[var]} vs {sol2[var]}"