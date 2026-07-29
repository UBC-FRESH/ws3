"""
Unit tests for ws3.perf module.

Tests SolverTuner, MemoryProfiler, PerformanceBenchmark, ResultCache,
and IncrementalSolver classes.
"""

from unittest.mock import MagicMock, patch
from ws3.perf import (
    SolverTuner,
    MemoryProfiler,
    PerformanceBenchmark,
    ResultCache,
    IncrementalSolver,
)


class TestSolverTuner:
    """Tests for SolverTuner class."""

    def test_initialization_gurobi(self):
        """Test creating tuner with gurobi solver."""
        mock_problem = MagicMock()
        tuner = SolverTuner(mock_problem, solver='gurobi')
        assert tuner.problem == mock_problem
        assert tuner.solver == 'gurobi'
        assert 'Threads' in tuner.baseline_params

    def test_initialization_highs(self):
        """Test creating tuner with highs solver."""
        mock_problem = MagicMock()
        tuner = SolverTuner(mock_problem, solver='highs')
        assert tuner.solver == 'highs'
        assert 'time_limit' in tuner.baseline_params

    def test_tune_parameters(self):
        """Test parameter tuning (mocked)."""
        mock_problem = MagicMock()
        mock_problem.solve.return_value = None
        mock_problem.status = 'Optimal'
        tuner = SolverTuner(mock_problem, solver='highs')

        param_grid = {
            'time_limit': [10.0, 100.0],
            'mip_rel_gap': [0.01, 0.05]
        }

        with patch.object(tuner, '_test_parameters') as mock_test:
            mock_test.return_value = 0.5
            result = tuner.tune_parameters(param_grid, n_iterations=1)
            assert '_best_time' in result
            assert result['_best_time'] == 0.5


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    def test_initialization(self):
        """Test creating profiler."""
        profiler = MemoryProfiler()
        assert profiler.snapshots == []

    def test_take_snapshot(self):
        """Test taking a memory snapshot."""
        profiler = MemoryProfiler()

        snapshot = profiler.take_snapshot(label='test')
        assert snapshot['label'] == 'test'
        assert 'current_mb' in snapshot
        assert 'peak_mb' in snapshot
        assert 'timestamp' in snapshot

    def test_profile_solve(self):
        """Test profiling a solve function."""
        profiler = MemoryProfiler()

        def dummy_solve():
            return 42

        result = profiler.profile_solve(dummy_solve)
        assert result['solve_time'] >= 0
        assert 'memory_before' in result
        assert 'memory_after' in result
        assert 'memory_delta' in result

    def test_get_report(self):
        """Test generating a profiling report."""
        profiler = MemoryProfiler()
        profiler.take_snapshot('s1')
        profiler.take_snapshot('s2')
        report = profiler.get_report()
        assert len(report) == 2

    def test_reset(self):
        """Test resetting profiler state."""
        profiler = MemoryProfiler()
        profiler.take_snapshot('s1')
        profiler.reset()
        assert profiler.snapshots == []


class TestPerformanceBenchmark:
    """Tests for PerformanceBenchmark class."""

    def test_initialization(self):
        """Test creating benchmark with a problem."""
        mock_problem = MagicMock()
        benchmark = PerformanceBenchmark(mock_problem)
        assert benchmark.problem == mock_problem
        assert benchmark.results == []

    def test_benchmark_solve(self):
        """Test benchmarking solve performance (mocked)."""
        mock_problem = MagicMock()
        mock_problem.solve.return_value = None
        mock_problem.status.return_value = 'Optimal'
        benchmark = PerformanceBenchmark(mock_problem)

        result = benchmark.benchmark_solve(n_runs=3)
        assert result['mean_time'] >= 0
        assert result['std_time'] >= 0
        assert 'min_time' in result
        assert 'max_time' in result
        assert result['status'] == 'Optimal'

    def test_benchmark_parallel(self):
        """Test benchmarking parallel performance (mocked)."""
        mock_problem = MagicMock()
        mock_problem.solve.return_value = None
        mock_problem.status.return_value = 'Optimal'
        benchmark = PerformanceBenchmark(mock_problem)

        df = benchmark.benchmark_parallel(threads_list=[1, 2, 4])
        assert len(df) == 3
        assert 'threads' in df.columns


class TestResultCache:
    """Tests for ResultCache class."""

    def test_initialization(self):
        """Test creating cache with default dir."""
        cache = ResultCache(cache_dir='/tmp/test_ws3_cache')
        assert cache.cache_dir is not None
        assert cache.cache == {}

    def test_put_and_get(self):
        """Test caching and retrieving a result."""
        cache = ResultCache(cache_dir='/tmp/test_ws3_cache')
        mock_problem = MagicMock()
        mock_problem._vars = [1, 2, 3]
        mock_problem._constraints = [1, 2]

        value = {"result": 42}
        cache.put(mock_problem, value)
        cached = cache.get(mock_problem)
        assert cached == value

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResultCache(cache_dir='/tmp/test_ws3_cache')
        mock_problem = MagicMock()
        mock_problem._vars = [999]
        mock_problem._constraints = []
        result = cache.get(mock_problem)
        assert result is None

    def test_clear_cache(self):
        """Test clearing cache."""
        cache = ResultCache(cache_dir='/tmp/test_ws3_cache_clear')
        mock_problem = MagicMock()
        mock_problem._vars = [1]
        mock_problem._constraints = []
        cache.put(mock_problem, {"a": 1})
        cache.clear()
        assert cache.cache == {}

    def test_stats(self):
        """Test cache statistics."""
        cache = ResultCache(cache_dir='/tmp/test_ws3_cache_stats')
        mock_problem = MagicMock()
        mock_problem._vars = [1]
        mock_problem._constraints = []
        cache.put(mock_problem, {"a": 1})
        stats = cache.stats()
        assert 'n_cached' in stats
        assert 'total_size_mb' in stats


class TestIncrementalSolver:
    """Tests for IncrementalSolver class."""

    def test_initialization(self):
        """Test creating incremental solver."""
        mock_problem = MagicMock()
        solver = IncrementalSolver(mock_problem)
        assert solver.problem == mock_problem
        assert solver.previous_solution is None

    def test_warm_start(self):
        """Test warm starting from previous solution."""
        mock_problem = MagicMock()
        mock_problem._warm_start = None
        solver = IncrementalSolver(mock_problem)

        warm_start = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0]}
        result = solver.warm_start(warm_start)
        assert result is True
        assert solver.previous_solution == warm_start

    def test_solve_with_warm_start(self):
        """Test solving with warm start (mocked)."""
        mock_problem = MagicMock()
        mock_problem.solve.return_value = None
        mock_problem.get_solution.return_value = {"x": [1.0, 2.0]}
        solver = IncrementalSolver(mock_problem)
        solver.warm_start({"x": [1.0, 2.0]})

        result = solver.solve_with_warmstart()
        # Returns False when objective values are equal (0.0 < 0.0 is False)
        assert result is False

    def test_solve_without_warm_start(self):
        """Test solving without warm start returns False."""
        mock_problem = MagicMock()
        solver = IncrementalSolver(mock_problem)

        result = solver.solve_with_warmstart()
        assert result is False

    def test_get_solution(self):
        """Test getting current solution."""
        mock_problem = MagicMock()
        mock_problem._solution = {"x": [1.0]}
        solver = IncrementalSolver(mock_problem)
        sol = solver.get_solution()
        assert sol == {"x": [1.0]}

    def test_get_solution_none(self):
        """Test getting solution when none available."""
        mock_problem = MagicMock()
        del mock_problem._solution
        solver = IncrementalSolver(mock_problem)
        sol = solver.get_solution()
        assert sol is None
