"""
Performance optimization utilities for ws3.

This module provides tools for:
- Solver parameter tuning and optimization
- Memory profiling and leak detection
- Performance benchmarking
- Incremental solving (warm-starting)
- Result caching for repeated scenarios
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SolverTuner:
    """
    Optimizes solver parameters for forest optimization problems.

    Provides systematic parameter tuning for Gurobi, CBC, GLPK, and HiGHS solvers.
    """

    def __init__(self, problem: Any, solver: str = 'gurobi'):
        """
        Initialize solver tuner.

        :param problem: Optimization problem to tune
        :param solver: Solver name (gurobi, pulp, highs)
        """
        self.problem = problem
        self.solver = solver
        self.baseline_params = self._get_default_params()

    def _get_default_params(self) -> Any:
        """Get default solver parameters."""
        defaults: dict[str, Any] = {
            'gurobi': {
                'Threads': 0,  # Auto
                'TimeLimit': 3600,  # 1 hour
                'MIPGap': 0.01,  # 1% optimality gap
                'NodeLimit': 0,  # No limit
                'Heuristics': 0.05,
                'Cuts': -1,  # Auto
                'Presolve': -1,  # Auto
            },
            'pulp': {
                'seconds': 3600,
                'msg': 0,
            },
            'highs': {
                'time_limit': 3600.0,
                'mip_rel_gap': 0.01,
            }
        }
        return defaults.get(self.solver, {})

    def tune_parameters(self, param_grid: dict[str, list[Any]],
                       n_iterations: int = 5) -> Any:
        """
        Tune solver parameters using grid search.

        :param param_grid: Dictionary of parameter names to lists of values
        :param n_iterations: Number of iterations per parameter combination
        :return: Best parameter set
        """
        from itertools import product

        best_params = self.baseline_params.copy()
        best_time = float('inf')

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        print(f"Tuning {self.solver} with {len(combinations)} parameter combinations...")

        for _i, combo in enumerate(combinations):
            params = self.baseline_params.copy()
            for j, name in enumerate(param_names):
                params[name] = combo[j]

            # Test this parameter set
            avg_time = self._test_parameters(params, n_iterations)

            if avg_time < best_time:
                best_time = avg_time
                best_params = params.copy()
                best_params['_best_time'] = avg_time

        print(f"Best parameters: {best_params}")
        print(f"Best time: {best_time:.2f}s")

        return best_params

    def _test_parameters(self, params: dict[str, Any], n_iterations: int) -> float:
        """Test a parameter set and return average solve time."""
        times = []

        for _ in range(n_iterations):
            # Apply parameters
            self._apply_parameters(params)

            # Solve
            start = time.time()
            self.problem.solve(threads=params.get('Threads', 0))
            elapsed = time.time() - start

            times.append(elapsed)

        return float(np.mean(times))

    def _apply_parameters(self, params: dict[str, Any]) -> None:
        """Apply parameters to the solver."""
        if self.solver == 'gurobi' and hasattr(self.problem, '_model'):
            try:
                import gurobipy as gp  # noqa: F401  (availability probe)
                for key, value in params.items():
                    if key != '_best_time':
                        self.problem._model.setParam(key, value)
                self.problem._model.optimize()
            except Exception as e:
                print(f"Error applying Gurobi parameters: {e}")

        elif self.solver == 'pulp' and hasattr(self.problem, '_model'):
            try:
                import pulp
                solver = pulp.PULP_CBC_CMD(**params)
                self.problem._model.solve(solver)
            except Exception as e:
                print(f"Error applying PuLP parameters: {e}")

    def get_recommendations(self) -> Any:
        """Get solver parameter recommendations based on problem size."""
        n_vars = len(self.problem._vars) if hasattr(self.problem, '_vars') else 0
        n_constraints = len(self.problem._constraints) if hasattr(self.problem, '_constraints') else 0

        recommendations = self.baseline_params.copy()

        # Adjust based on problem size
        if n_vars > 10000:
            recommendations['MIPGap'] = 0.05  # Looser gap for large problems
            recommendations['TimeLimit'] = 7200  # 2 hours
        elif n_vars < 1000:
            recommendations['MIPGap'] = 0.001  # Tighter gap for small problems

        if n_constraints > 50000:
            recommendations['Presolve'] = 2  # Aggressive presolve

        return recommendations


class MemoryProfiler:
    """
    Profile memory usage of optimization problems.

    Identifies memory leaks and optimization opportunities.
    """

    def __init__(self):
        self.snapshots = []

    def take_snapshot(self, label: str = '') -> dict[str, Any]:
        """
        Take a memory usage snapshot.

        :param label: Label for this snapshot
        :return: Memory statistics
        """
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        current, peak = tracemalloc.get_traced_memory()

        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'current_mb': current / (1024 * 1024),
            'peak_mb': peak / (1024 * 1024),
            'process_memory_mb': self._get_process_memory(),
        }

        self.snapshots.append(snapshot)
        return snapshot

    def _get_process_memory(self) -> float:
        """Get current process memory in MB."""
        try:
            import resource
            # Unix only
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert KB to MB
        except ImportError:
            return 0.0

    def profile_solve(self, solve_func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Profile memory usage during a solve operation.

        :param solve_func: Function to profile
        :return: Memory profiling results
        """
        import tracemalloc

        tracemalloc.start()

        # Take before snapshot
        before = self.take_snapshot('before_solve')

        # Run solve
        start_time = time.time()
        solve_func(*args, **kwargs)
        solve_time = time.time() - start_time

        # Take after snapshot
        after = self.take_snapshot('after_solve')

        # Stop tracing
        tracemalloc.stop()

        return {
            'solve_time': solve_time,
            'memory_before': before,
            'memory_after': after,
            'memory_delta': after['current_mb'] - before['current_mb'],
        }

    def get_report(self) -> pd.DataFrame:
        """Generate memory profiling report."""
        if not self.snapshots:
            return pd.DataFrame()

        return pd.DataFrame(self.snapshots)

    def reset(self):
        """Reset profiler state."""
        self.snapshots = []


class PerformanceBenchmark:
    """
    Benchmark optimization performance.

    Provides standardized benchmarks for comparing solver configurations.
    """

    def __init__(self, problem: Any):
        self.problem = problem
        self.results: list[Any] = []

    def benchmark_solve(self, n_runs: int = 5, **solve_kwargs: Any) -> dict[str, Any]:
        """
        Benchmark solve performance.

        :param n_runs: Number of runs to average
        :param solve_kwargs: Additional arguments to solve()
        :return: Benchmark results
        """
        times = []
        solutions = []

        for _i in range(n_runs):
            start = time.time()
            self.problem.solve(**solve_kwargs)
            elapsed = time.time() - start

            times.append(elapsed)
            solutions.append(self.problem.status())

        results = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'solutions': solutions,
            'status': solutions[-1] if solutions else None,
        }

        self.results.append(results)
        return results

    def benchmark_parallel(self, threads_list: list[int]) -> pd.DataFrame:
        """
        Benchmark parallel speedup.

        :param threads_list: List of thread counts to test
        :return: Performance comparison DataFrame
        """
        results = []

        for n_threads in threads_list:
            result = self.benchmark_solve(threads=n_threads, n_runs=3)
            result['threads'] = n_threads
            results.append(result)

        return pd.DataFrame(results)

    def get_speedup(self, baseline_threads: int = 1) -> dict[int, float]:
        """Calculate speedup relative to baseline."""
        if not self.results:
            return {}

        baseline_time = None
        for result in self.results:
            if result.get('threads') == baseline_threads:
                baseline_time = result['mean_time']
                break

        if baseline_time is None:
            return {}

        speedups = {}
        for result in self.results:
            threads = result['threads']
            if threads > 0:
                speedups[threads] = baseline_time / result['mean_time']

        return speedups

    def plot_speedup(self, ax=None):
        """Plot speedup curve."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        speedups = self.get_speedup()
        threads = list(speedups.keys())
        speeds = list(speedups.values())

        ax.plot(threads, speeds, 'o-', label='Measured')
        ax.plot(threads, threads, '--', label='Ideal linear speedup')

        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Speedup')
        ax.set_title('Parallel Speedup Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class ResultCache:
    """
    Cache optimization results for repeated scenarios.

    Enables fast re-solving of similar problems by caching intermediate results.
    """

    def __init__(self, cache_dir: str = '.ws3_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache: dict[str, Any] = {}

    def _compute_cache_key(self, problem: Any, **kwargs: Any) -> str:
        """Compute a unique cache key for a problem configuration."""
        # Use problem attributes and kwargs to generate key
        key_data = {
            'n_vars': len(problem._vars) if hasattr(problem, '_vars') else 0,
            'n_constraints': len(problem._constraints) if hasattr(problem, '_constraints') else 0,
            'kwargs': sorted(kwargs.items()),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, problem: Any, **kwargs: Any) -> Any:
        """
        Get cached result if available.

        :param problem: Optimization problem
        :param kwargs: Problem configuration
        :return: Cached result or None
        """
        key = self._compute_cache_key(problem, **kwargs)

        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        return None

    def put(self, problem: Any, result: Any, **kwargs: Any) -> None:
        """
        Store result in cache.

        :param problem: Optimization problem
        :param result: Solution result
        :param kwargs: Problem configuration
        """
        key = self._compute_cache_key(problem, **kwargs)

        cache_file = self.cache_dir / f"{key}.json"

        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        self.cache[key] = result

    def clear(self):
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir()
        self.cache = {}

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob('*.json'))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'n_cached': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
        }


class IncrementalSolver:
    """
    Support incremental solving (warm-starting) from previous solutions.

    Enables efficient re-optimization when problem parameters change slightly.
    """

    def __init__(self, problem: Any):
        self.problem = problem
        self.previous_solution: dict[str, float] | None = None

    def warm_start(self, solution: dict[str, float]) -> bool:
        """
        Set warm start solution.

        :param solution: Dictionary of variable names to values
        :return: True if warm start accepted
        """
        self.previous_solution = solution

        # Apply warm start to problem
        if hasattr(self.problem, '_warm_start'):
            self.problem._warm_start = list(solution.values())
            return True

        return False

    def solve_with_warmstart(self, **kwargs: Any) -> bool:
        """
        Solve with warm start from previous solution.

        :param kwargs: Additional solve arguments
        :return: True if solution improved
        """
        raise NotImplementedError(
            "IncrementalSolver.solve_with_warmstart() is an experimental stub and is "
            "not production-ready.\n"
            "\n"
            "Missing: it calls Problem.get_solution(), which does not exist. The real "
            "API is Problem.solution().\n"
            "\n"
            "The rest of this module -- MemoryProfiler, ResultCache, "
            "PerformanceBenchmark and SolverTuner -- is functional and unaffected. "
            "Tracked in #103."
        )
        if self.previous_solution is None:
            print("No warm start solution available")
            return False

        # Solve with warm start
        self.problem.solve(warm_start=list(self.previous_solution.values()), **kwargs)

        # Check if solution improved
        new_solution = self.problem.get_solution()

        if new_solution and self.previous_solution:
            # Compare objective values
            old_obj = self._compute_objective(self.previous_solution)
            new_obj = self._compute_objective(new_solution)

            return new_obj < old_obj  # Assuming minimization

        return True

    def _compute_objective(self, solution: dict[str, float]) -> float:
        """Compute objective value for a solution."""
        # This is solver-specific and would need implementation
        return 0.0

    def get_solution(self) -> Any:
        """Get current solution."""
        if hasattr(self.problem, '_solution'):
            return self.problem._solution
        return None


# Convenience functions

def tune_solver(problem: Any, solver: str = 'gurobi', **kwargs: Any) -> SolverTuner:
    """Create and return a SolverTuner instance."""
    return SolverTuner(problem, solver)

def profile_memory() -> MemoryProfiler:
    """Create and return a MemoryProfiler instance."""
    return MemoryProfiler()  # type: ignore[no-untyped-call]

def benchmark(problem: Any) -> PerformanceBenchmark:
    """Create and return a PerformanceBenchmark instance."""
    return PerformanceBenchmark(problem)

def cache_results(cache_dir: str = '.ws3_cache') -> ResultCache:
    """Create and return a ResultCache instance."""
    return ResultCache(cache_dir)

def incremental_solve(problem: Any) -> IncrementalSolver:
    """Create and return an IncrementalSolver instance."""
    return IncrementalSolver(problem)
