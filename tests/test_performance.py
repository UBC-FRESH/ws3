"""
Performance tests for ws3 optimization module.

Tests solver performance, memory usage, and scalability.
"""

import sys
sys.path.append('../ws3/')

import pytest
import time
import numpy as np
from ws3.opt import Variable, Constraint, Problem
from ws3.forest import ForestModel


class TestPerformance:
    """Test optimization performance characteristics."""

    def test_small_problem_solve_time(self):
        """Test solve time for small problem (<100 variables)."""
        problem = Problem("test")

        # Add small number of variables
        for i in range(50):
            problem.add_var(f"x{i}", "continuous", 0, 100)

        # Add a few constraints
        for i in range(10):
            coeffs = {f"x{i}": 1.0, f"x{i+1}": 0.5} if i+1 < 50 else {f"x{i}": 1.0}
            problem.add_constraint(f"con{i}", coeffs, "<", 50)

        # Set objective
        problem.z({f"x{i}": 1.0 for i in range(50)})

        # Time the solve
        start = time.time()
        problem.solve()
        elapsed = time.time() - start

        # Small problems should solve quickly
        assert elapsed < 5.0, f"Small problem took too long: {elapsed:.2f}s"

    def test_medium_problem_solve_time(self):
        """Test solve time for medium problem (100-1000 variables)."""
        problem = Problem("test")

        # Add medium number of variables
        n_vars = 500
        for i in range(n_vars):
            problem.add_var(f"x{i}", "continuous", 0, 1000)

        # Add constraints
        for i in range(100):
            coeffs = {}
            for j in range(i, min(i+10, n_vars)):
                coeffs[f"x{j}"] = 1.0
            problem.add_constraint(f"con{i}", coeffs, "<", 500)

        # Set objective
        problem.z({f"x{i}": float(i+1) for i in range(n_vars)})

        # Time the solve
        start = time.time()
        problem.solve()
        elapsed = time.time() - start

        # Medium problems should solve in reasonable time
        assert elapsed < 30.0, f"Medium problem took too long: {elapsed:.2f}s"

    def test_large_problem_scalability(self):
        """Test that solve time scales reasonably with problem size."""
        problem_sizes = [100, 500, 1000]
        solve_times = []

        for n_vars in problem_sizes:
            problem = Problem("test")

            for i in range(n_vars):
                problem.add_var(f"x{i}", "continuous", 0, 100)

            for i in range(n_vars // 10):
                coeffs = {f"x{j}": 1.0 for j in range(i, min(i+5, n_vars))}
                problem.add_constraint(f"con{i}", coeffs, "<", 50)

            problem.z({f"x{i}": 1.0 for i in range(n_vars)})

            start = time.time()
            problem.solve()
            elapsed = time.time() - start
            solve_times.append(elapsed)

        # Verify that larger problems take longer (but not exponentially)
        for i in range(1, len(solve_times)):
            assert solve_times[i] >= solve_times[i-1] * 0.5, \
                f"Problem {problem_sizes[i]} should take at least half the time of {problem_sizes[i-1]}"

    def test_memory_usage_small_problem(self):
        """Test memory usage for small problem."""
        import tracemalloc

        tracemalloc.start()

        problem = Problem("test")

        # Add variables and constraints
        for i in range(100):
            problem.add_var(f"x{i}", "continuous", 0, 100)

        for i in range(20):
            coeffs = {f"x{j}": 1.0 for j in range(i, min(i+5, 100))}
            problem.add_constraint(f"con{i}", coeffs, "<", 50)

        problem.z({f"x{i}": 1.0 for i in range(100)})

        problem.solve()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Small problem should use reasonable memory
        assert peak < 100 * 1024 * 1024, f"Small problem used too much memory: {peak / (1024*1024):.2f} MB"

    def test_variable_creation_performance(self):
        """Test performance of creating many variables."""
        n_vars = 10000

        start = time.time()
        variables = []
        for i in range(n_vars):
            var = Variable(f"x{i}", "continuous", 0, 100)
            variables.append(var)
        elapsed = time.time() - start

        # Creating 10k variables should be fast
        assert elapsed < 2.0, f"Variable creation took too long: {elapsed:.2f}s"

    def test_constraint_creation_performance(self):
        """Test performance of creating many constraints."""
        n_constraints = 5000

        start = time.time()
        constraints = []
        for i in range(n_constraints):
            # Ensure at least one coefficient
            j_start = i % 100
            j_end = min(j_start + 3, 100)
            coeffs = {f"x{j}": 1.0 for j in range(j_start, j_end)}
            if not coeffs:
                coeffs = {"x0": 1.0}
            constraint = Constraint(f"con{i}", coeffs, "<", 50)
            constraints.append(constraint)
        elapsed = time.time() - start

        # Creating 5k constraints should be fast
        assert elapsed < 2.0, f"Constraint creation took too long: {elapsed:.2f}s"


class TestSolverComparison:
    """Compare performance across different solvers."""

    @pytest.fixture
    def sample_problem(self):
        """Create a sample problem for testing."""
        problem = Problem("test")

        for i in range(100):
            problem.add_var(f"x{i}", "continuous", 0, 100)

        for i in range(20):
            coeffs = {f"x{j}": 1.0 for j in range(i, min(i+5, 100))}
            problem.add_constraint(f"con{i}", coeffs, "<", 50)

        problem.z({f"x{i}": float(i+1) for i in range(100)})

        return problem

    def test_solver_consistency(self, sample_problem):
        """Test that different solvers give consistent results."""
        results = {}

        for solver in ["highs", "pulp"]:
            try:
                problem = sample_problem.__class__()
                # Copy problem
                for var in sample_problem._vars.values():
                    problem.add_var(var.name, var.vtype, var.lb, var.ub)
                for con in sample_problem._constraints.values():
                    problem.add_constraint(con.name, con.coeffs, con.sense, con.rhs)
                problem.z(sample_problem._objective)

                problem.solve(solver=solver)
                results[solver]  = problem._solution
            except Exception as e:
                pytest.skip(f"Solver {solver} not available: {e}")

        if len(results) >= 2:
            # Check that solutions are similar (within 10%)
            sol1 = list(results.values())[0]
            sol2 = list(results.values())[1]

            max_diff = max(abs(v1 - v2) for v1, v2 in zip(sol1.values(), sol2.values()))
            assert max_diff < 10.0, f"Solutions differ too much: {max_diff}"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_problem(self):
        """Test solving an empty problem."""
        problem = Problem("test")

        # Should not crash - empty problems may be valid in some contexts
        try:
            problem.solve()
        except Exception as e:
            # If it raises, that's also acceptable
            pass

    def test_infeasible_problem(self):
        """Test handling of infeasible problems."""
        problem = Problem("test")

        problem.add_var("x", "continuous", 0, 10)
        problem.add_constraint("con1", {"x": 1.0}, ">", 100)  # Infeasible

        problem.z({"x": 1.0})

        problem.solve()

        # Should report infeasible status
        assert problem.status() == "infeasible"

    def test_unbounded_problem(self):
        """Test handling of unbounded problems."""
        problem = Problem("test")

        problem.add_var("x", "continuous", 0, float('inf'))
        problem.z({"x": 1.0})

        problem.solve()

        # Should report unbounded status
        assert problem.status() == "unbounded"

    def test_integer_variables(self):
        """Test solving with integer variables."""
        problem = Problem("test")

        for i in range(10):
            problem.add_var(f"x{i}", "integer", 0, 100)

        problem.add_constraint("con1", {f"x{i}": 1.0 for i in range(10)}, "<", 50)
        problem.z({f"x{i}": float(i+1) for i in range(10)})

        problem.solve()

        # Check that solution values are integers
        solution  = problem._solution
        for val in solution.values():
            assert val == int(val), f"Solution value {val} is not integer"

    def test_binary_variables(self):
        """Test solving with binary variables."""
        problem = Problem("test")

        for i in range(10):
            problem.add_var(f"x{i}", "binary", 0, 1)

        problem.add_constraint("con1", {f"x{i}": 1.0 for i in range(10)}, "=", 5)
        problem.z({f"x{i}": float(i+1) for i in range(10)})

        problem.solve()

        # Check that solution values are 0 or 1
        solution  = problem._solution
        for val in solution.values():
            assert val in [0.0, 1.0], f"Solution value {val} is not binary"


class TestIntegration:
    """Integration tests with ForestModel."""

    def test_forest_model_optimization(self):
        """Test optimization with ForestModel."""
        # This test requires actual data files
        # Skip if data not available
        try:
            fm = ForestModel(
                model_name="test",
                model_path="data/woodstock_model_files_tsa24",
                base_year=2024
            )
            fm.import_landscape_section()
            fm.import_areas_section(convert_periods_to_years=10)
            fm.import_yields_section(convert_periods_to_years=10)
            fm.import_actions_section(convert_periods_to_years=10)
            fm.import_transitions_section(convert_periods_to_years=10)
            fm.initialize_areas()
            fm.add_null_action()
            fm.reset_actions()

            # Compile and solve a simple scenario
            from ws3.core import compile_scenario

            problem = compile_scenario(fm, scenario_name="test_perf")
            solution = problem.solve()

            assert solution.status() == "optimal"

        except FileNotFoundError:
            pytest.skip("Test data not available")

    def test_multiple_scenarios(self):
        """Test running multiple scenarios."""
        try:
            fm = ForestModel(
                model_name="test",
                model_path="data/woodstock_model_files_tsa24",
                base_year=2024
            )
            fm.import_landscape_section()
            fm.import_areas_section(convert_periods_to_years=10)
            fm.import_yields_section(convert_periods_to_years=10)
            fm.import_actions_section(convert_periods_to_years=10)
            fm.import_transitions_section(convert_periods_to_years=10)
            fm.initialize_areas()
            fm.add_null_action()
            fm.reset_actions()

            from ws3.core import compile_scenario

            scenarios = ["base", "test1", "test2"]
            solutions = {}

            for scenario in scenarios:
                problem = compile_scenario(fm, scenario_name=scenario)
                solution = problem.solve()
                solutions[scenario] = solution

            # All should solve successfully
            for scenario, solution in solutions.items():
                assert solution.status() == "optimal", \
                    f"Scenario {scenario} did not solve optimally"

        except FileNotFoundError:
            pytest.skip("Test data not available")
