"""
Tests that non-functional Phase 5 code paths refuse to run.

The modules added in Phase 5 (`ws3.advanced_modeling`, `ws3.perf`,
`ws3.integration`) contain working data structures wrapped around unimplemented
solve/apply methods. Before gating, several returned confident-looking results
that could not mean anything -- most notably
``StochasticOptimizer.solve_stochastic``, which generated random scenarios,
applied none of them, solved the identical problem N times, and reported a
variance across N identical values that was always exactly 0.

These tests pin two things:

1. every hollow entry point raises ``NotImplementedError``, so it cannot silently
   return fabricated output;
2. the parts that genuinely work are untouched and still work.

See #103.
"""

import pytest

import ws3.advanced_modeling as am
import ws3.integration as ig
import ws3.perf as perf


GATED_CALLS = [
    ('StochasticOptimizer.solve_stochastic',
     lambda: am.StochasticOptimizer(None).solve_stochastic()),
    ('StochasticOptimizer._apply_scenario',
     lambda: am.StochasticOptimizer(None)._apply_scenario(None)),
    ('MultiObjectiveOptimizer.solve_weighted_sum',
     lambda: am.MultiObjectiveOptimizer(None).solve_weighted_sum()),
    ('MultiObjectiveOptimizer.solve_epsilon_constraint',
     lambda: am.MultiObjectiveOptimizer(None).solve_epsilon_constraint('a', {})),
    ('MultiObjectiveOptimizer.find_pareto_frontier',
     lambda: am.MultiObjectiveOptimizer(None).find_pareto_frontier()),
    ('MultiObjectiveOptimizer._extract_objective_values',
     lambda: am.MultiObjectiveOptimizer(None)._extract_objective_values()),
    ('DynamicPlanner.plan_static',
     lambda: am.DynamicPlanner(None).plan_static()),
    ('DynamicPlanner.plan_dynamic',
     lambda: am.DynamicPlanner(None).plan_dynamic()),
    ('ClimateScenarioManager.apply_climate_effects',
     lambda: am.ClimateScenarioManager().apply_climate_effects(None, {})),
    ('ClimateScenarioManager.run_climate_analysis',
     lambda: am.ClimateScenarioManager().run_climate_analysis(None)),
    ('IncrementalSolver.solve_with_warmstart',
     lambda: perf.IncrementalSolver(None).solve_with_warmstart()),
    ('FHOPSIntegrator.inject_into_model',
     lambda: ig.FHOPSIntegrator().inject_into_model(None, None, 'x', 1.0)),
]


@pytest.mark.parametrize('name, call', GATED_CALLS, ids=[n for n, _ in GATED_CALLS])
def test_non_functional_paths_raise(name, call):
    """Each hollow entry point must raise rather than return fabricated results."""
    with pytest.raises(NotImplementedError):
        call()


@pytest.mark.parametrize('name, call', GATED_CALLS, ids=[n for n, _ in GATED_CALLS])
def test_gate_messages_are_actionable(name, call):
    """The message must say it is a stub and name what is missing."""
    with pytest.raises(NotImplementedError) as exc:
        call()
    msg = str(exc.value)
    assert 'stub' in msg.lower()
    assert 'missing' in msg.lower()


def test_scenario_generation_still_works():
    """Scenario generation is real and must remain usable."""
    opt = am.StochasticOptimizer(None)
    scenarios = opt.generate_scenarios(
        n_scenarios=5,
        uncertainty_type=am.UncertaintyType.GROWTH,
        mean=1.0,
        std=0.1,
    )
    assert len(scenarios) == 5
    assert all(isinstance(s, am.StochasticScenario) for s in scenarios)

    summary = opt.get_scenario_summary()
    assert len(summary) == 5


def test_generated_scenarios_actually_vary():
    """
    Guards the premise behind the gate.

    The scenario draws were never the problem -- they are genuine random draws.
    The defect was that they were never applied. If these ever stop varying, the
    generation side has broken too.
    """
    opt = am.StochasticOptimizer(None)
    scenarios = opt.generate_scenarios(
        n_scenarios=20,
        uncertainty_type=am.UncertaintyType.GROWTH,
        mean=1.0,
        std=0.2,
    )
    factors = {s.parameters['growth_factor'] for s in scenarios}
    assert len(factors) > 1, 'generated scenarios should not be identical'


def test_rcp_scenarios_still_work():
    """Static RCP scenario data is real and must remain usable."""
    mgr = am.ClimateScenarioManager()
    scenarios = mgr.get_rcp_scenarios()
    assert len(scenarios) == 4
    assert {s['name'] for s in scenarios} == {'RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5'}


def test_add_objective_still_works():
    """Objective registration is real and must remain usable."""
    opt = am.MultiObjectiveOptimizer(None)
    opt.add_objective('volume', weight=0.7)
    opt.add_objective('carbon', weight=0.3)
    assert len(opt.objectives) == 2


def test_memory_profiler_still_works():
    """MemoryProfiler uses tracemalloc and genuinely works."""
    profiler = perf.MemoryProfiler()
    snapshot = profiler.take_snapshot('test')
    assert 'current_mb' in snapshot
    assert isinstance(snapshot['current_mb'], float)
