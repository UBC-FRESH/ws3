"""
Targeted coverage tests for ws3.forest — DevelopmentType, ForestModel helpers.

Covers uncovered branches:
- DevelopmentType: grow, reset_areas, ycomps, ycomp (silent_fail), add_ycomp
- DevelopmentType: operable_ages, is_operable, operable_area, area
- DevelopmentType: resolve_condition, compile_actions, compile_action
- DevelopmentType: _compile_oper_expr edge cases
- ForestModel: nthemes, set_horizon, _resolve_period_multiplier
- ForestModel: compile_actions
- ForestModel: register_curve (via common_curves)
"""

import sys

sys.path.append('../ws3/')


import pytest

from ws3.core import Curve
from ws3.forest import (
    Action,
    DevelopmentType,
    ForestModel,
    GreedyAreaSelector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubParent:
    """Minimal stand-in for ForestModel exposing only what DevelopmentType reads."""
    horizon = 10
    max_age = 200
    period_length = 10
    area_epsilon = 0.01

    def __init__(self):
        self.common_curves = {
            'zero': Curve('zero', is_special=True, type=''),
            'unit': Curve('unit', points=[(0, 1)], is_special=True, type=''),
            'ages': Curve('ages', points=[(0, 0), (200, 200)], is_special=True, type=''),
        }
        self.constants = {}
        self.horizon = 10
        self.periods = list(range(1, 11))
        self.ages = list(range(201))
        self.actions = {}
        self.dtypes = {}
        self._themes = []
        self.nthemes = lambda: 0
        self.theme_basecodes = lambda i: []
        self.register_curve = lambda c: c
        self.unmask = lambda m: list(self.dtypes.keys())

    def set_horizon(self, h):
        self.horizon = h


def _make_dtype(parent=None):
    """Build a DevelopmentType with a stub parent."""
    if parent is None:
        parent = _StubParent()
    dt = DevelopmentType(('test',), parent)
    return dt


def _make_fm():
    """Build a ForestModel with one dtype and area for testing ForestModel methods."""
    fm = ForestModel.__new__(ForestModel)
    fm.horizon = 10
    fm.periods = list(range(1, 11))
    fm.ages = list(range(201))
    fm.max_age = 200
    fm.period_length = 10
    fm._themes = [{'code': 't1'}, {'code': 't2'}]
    fm.common_curves = {
        'zero': Curve('zero', is_special=True, type=''),
        'unit': Curve('unit', points=[(0, 1)], is_special=True, type=''),
        'ages': Curve('ages', points=[(0, 0), (200, 200)], is_special=True, type=''),
    }
    fm.register_curve = lambda c: c
    fm.dtypes = {}
    fm.actions = {'harv': Action('harv', is_sticky=0),
                  'harv_sticky': Action('harv_sticky', is_sticky=1)}
    fm.applied_actions = {p: {} for p in range(1, 11)}
    fm.ynames = set()
    dt = DevelopmentType(('a', '1'), fm)
    dt._areas = {1: {10: 5.0, 20: 3.0}, 2: {10: 4.0}}
    dt.oper_expr = {}
    dt.operability = {}
    fm.dtypes[('a', '1')] = dt
    return fm


# ---------------------------------------------------------------------------
# DevelopmentType — area
# ---------------------------------------------------------------------------

class TestDevelopmentTypeArea:
    def test_area_return_total(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0, 20: 3.0}
        assert dt.area(1) == 8.0

    def test_area_return_by_age(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0, 20: 3.0}
        assert dt.area(1, age=10) == 5.0

    def test_area_return_missing_age(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        # age=20 not in inventory
        assert dt.area(1, age=20) == 0.0

    def test_area_set_delta(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt.area(1, age=10, area=2.0, delta=True)
        assert dt.area(1, age=10) == 7.0

    def test_area_set_absolute(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt.area(1, age=10, area=2.0, delta=False)
        assert dt.area(1, age=10) == 2.0

    def test_area_set_returns_none(self):
        dt = _make_dtype()
        result = dt.area(1, age=10, area=5.0)
        assert result is None


# ---------------------------------------------------------------------------
# DevelopmentType — reset_areas, grow
# ---------------------------------------------------------------------------

class TestDevelopmentTypeResetGrow:
    def test_reset_areas_all(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt._areas[2] = {20: 3.0}
        dt.reset_areas()
        assert dt._areas[1] == {}
        assert dt._areas[2] == {}

    def test_reset_areas_specific_period(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt._areas[2] = {20: 3.0}
        dt.reset_areas(period=1)
        assert dt._areas[1] == {}
        assert dt._areas[2] == {20: 3.0}

    def test_grow_cascades(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt.grow(start_period=1, cascade=True)
        # Age 10 + period_length(10) = 20 in period 2
        assert 20 in dt._areas[2]
        assert dt._areas[2][20] == 5.0

    def test_grow_single_period(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        dt.grow(start_period=1, cascade=False)
        assert 20 in dt._areas[2]
        # Period 3 should not be affected
        assert dt._areas[3] == {}


# ---------------------------------------------------------------------------
# DevelopmentType — ycomps, ycomp, add_ycomp
# ---------------------------------------------------------------------------

class TestDevelopmentTypeYcomps:
    def test_ycomps_empty(self):
        dt = _make_dtype()
        assert dt.ycomps() == []

    def test_ycomp_missing_silent(self):
        dt = _make_dtype()
        assert dt.ycomp('nonexistent', silent_fail=True) is None

    def test_ycomp_missing_raises(self):
        dt = _make_dtype()
        with pytest.raises(KeyError):
            dt.ycomp('nonexistent', silent_fail=False)

    def test_add_ycomp_curve(self):
        dt = _make_dtype()
        c = Curve('vol', points=[(0, 0), (100, 100)])
        dt.add_ycomp('a', 'vol', c)
        assert 'vol' in dt.ycomps()
        assert dt.ycomp('vol') is c

    def test_add_ycomp_complex(self):
        dt = _make_dtype()
        dt.add_ycomp('c', 'complex_yield', 'MAI(vol)')
        assert 'complex_yield' in dt._complex_ycomps
        assert dt._ycomps['complex_yield'] is None

    def test_add_ycomp_first_match_rejects_duplicate(self):
        dt = _make_dtype()
        c1 = Curve('vol', points=[(0, 0), (100, 100)])
        c2 = Curve('vol2', points=[(0, 0), (100, 200)])
        dt.add_ycomp('a', 'vol', c1)
        dt.add_ycomp('a', 'vol', c2, first_match=True)
        # Should still be c1
        assert dt.ycomp('vol') is c1


# ---------------------------------------------------------------------------
# DevelopmentType — operable_ages, is_operable, operable_area
# ---------------------------------------------------------------------------

class TestDevelopmentTypeOperable:
    def test_operable_ages_no_action(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        result = dt.operable_ages('nonexistent', 1)
        assert result is None

    def test_is_operable_no_action(self):
        dt = _make_dtype()
        assert dt.is_operable('nonexistent', 1) is False

    def test_operable_area_no_action(self):
        dt = _make_dtype()
        dt._areas[1] = {10: 5.0}
        assert dt.operable_area('nonexistent', 1) == 0.0

    def test_operable_area_no_inventory(self):
        dt = _make_dtype()
        dt.oper_expr['harv'] = ['_age >= 50']
        dt.operability['harv'] = {1: (50, 200)}
        # No inventory at period 1
        assert dt.operable_area('harv', 1, age=100) == 0.0

    def test_operable_area_negligible_cleanup(self):
        dt = _make_dtype()
        dt.oper_expr['harv'] = ['_age >= 50']
        dt.operability['harv'] = {1: (50, 200)}
        dt._areas[1] = {100: 0.001}  # below epsilon
        result = dt.operable_area('harv', 1, age=100, cleanup=True)
        assert result == 0.0
        assert 100 not in dt._areas[1]


# ---------------------------------------------------------------------------
# DevelopmentType — resolve_condition
# ---------------------------------------------------------------------------

class TestDevelopmentTypeResolveCondition:
    def test_resolve_condition(self):
        dt = _make_dtype()
        c = Curve('vol', points=[(0, 0), (10, 10), (20, 20), (30, 30)])
        dt.add_ycomp('a', 'vol', c)
        ages = dt.resolve_condition('vol', 10, 20)
        assert 10 in ages
        assert 20 in ages


# ---------------------------------------------------------------------------
# DevelopmentType — compile_action
# ---------------------------------------------------------------------------

class TestDevelopmentTypeCompileAction:
    def test_compile_action_never_operable(self):
        dt = _make_dtype()
        dt.oper_expr['harv'] = ['_age >= 300']  # beyond max_age
        result = dt.compile_action('harv')
        assert result == -1
        assert 'harv' not in dt.operability

    def test_compile_action_operable(self):
        dt = _make_dtype()
        dt.oper_expr['harv'] = ['_age >= 50']
        result = dt.compile_action('harv')
        assert result == 0
        assert 'harv' in dt.operability


# ---------------------------------------------------------------------------
# DevelopmentType — _compile_oper_expr
# ---------------------------------------------------------------------------

class TestDevelopmentTypeCompileOperExpr:
    def test_period_exact(self):
        dt = _make_dtype()
        dt.operability['a'] = {}
        dt._compile_oper_expr('a', '_cp = 5')
        assert 5 in dt.operability['a']
        assert dt.operability['a'][5] == (0, 200)

    def test_period_gte(self):
        dt = _make_dtype()
        dt.operability['a'] = {}
        dt._compile_oper_expr('a', '_cp >= 5')
        for p in range(5, 11):
            assert dt.operability['a'][p] == (0, 200)

    def test_period_lte(self):
        dt = _make_dtype()
        dt.operability['a'] = {}
        dt._compile_oper_expr('a', '_cp <= 3')
        for p in range(1, 4):
            assert dt.operability['a'][p] == (0, 200)
        for p in range(4, 11):
            assert p not in dt.operability['a']

    def test_age_gte(self):
        dt = _make_dtype()
        dt.operability['a'] = {}
        c = Curve('vol', points=[(0, 0), (50, 50), (100, 100)])
        dt.add_ycomp('a', 'vol', c)
        dt._compile_oper_expr('a', '_age >= 50')
        for p in range(1, 11):
            assert dt.operability['a'][p] == (50, 200)

    def test_bad_relational_operator(self):
        dt = _make_dtype()
        dt.operability['a'] = {}
        with pytest.raises(ValueError, match="Bad relational operator"):
            dt._compile_oper_expr('a', '_cp != 5')

    def test_age_and_conditions(self):
        """_age >= 50 and _age <= 100 narrows the age range (intersection)."""
        dt = _make_dtype()
        dt.operability['a'] = {}
        dt._compile_oper_expr('a', '_age >= 50 and _age <= 100')
        for p in range(1, 11):
            assert dt.operability['a'][p] == (50, 100)

    def test_age_or_conditions(self):
        """_age >= 50 or _age <= 100 widens the age range (union)."""
        dt = _make_dtype()
        dt.operability['a'] = {}
        dt._compile_oper_expr('a', '_age >= 50 or _age <= 100')
        # OR union: lower bound is min, upper bound is max
        for p in range(1, 11):
            assert dt.operability['a'][p] == (50, 100)


# ---------------------------------------------------------------------------
# ForestModel helpers
# ---------------------------------------------------------------------------

class TestForestModelHelpers:
    def test_set_horizon(self):
        fm = ForestModel.__new__(ForestModel)
        fm.set_horizon(5)
        assert fm.horizon == 5
        assert fm.periods == [1, 2, 3, 4, 5]

    def test_resolve_period_multiplier_valid(self):
        fm = ForestModel.__new__(ForestModel)
        fm._period_to_years_factor = None
        result = fm._resolve_period_multiplier(10)
        assert result == 10
        assert fm._period_to_years_factor == 10

    def test_resolve_period_multiplier_conflict(self):
        fm = ForestModel.__new__(ForestModel)
        fm._period_to_years_factor = 10
        with pytest.raises(ValueError, match="conflicts"):
            fm._resolve_period_multiplier(20)

    def test_resolve_period_multiplier_invalid_type(self):
        fm = ForestModel.__new__(ForestModel)
        fm._period_to_years_factor = None
        with pytest.raises(ValueError, match="must be an integer"):
            fm._resolve_period_multiplier("ten")

    def test_resolve_period_multiplier_non_positive(self):
        fm = ForestModel.__new__(ForestModel)
        fm._period_to_years_factor = None
        with pytest.raises(ValueError, match="must be positive"):
            fm._resolve_period_multiplier(0)

    def test_nthemes(self):
        fm = ForestModel.__new__(ForestModel)
        fm._themes = [{'code': 'tsa'}, {'code': 'spec'}]
        assert fm.nthemes() == 2

    def test_common_curves_registered(self):
        """ForestModel.__init__ registers zero, unit, and ages curves."""
        fm = ForestModel(
            model_name='test',
            model_path='/tmp',
            base_year=2020,
            horizon=1,
            period_length=10,
            max_age=100,
        )
        assert 'zero' in fm.common_curves
        assert 'unit' in fm.common_curves
        assert 'ages' in fm.common_curves

    def test_age_class_distribution(self):
        fm = _make_fm()
        acd = fm.age_class_distribution(period=1)
        assert acd[10] == 5.0
        assert acd[20] == 3.0
        assert acd[30] == 0.0

    def test_age_class_distribution_omit_null(self):
        fm = _make_fm()
        acd = fm.age_class_distribution(period=1, omit_null=True)
        assert 10 in acd
        assert 20 in acd
        assert 30 not in acd

    def test_operable_dtypes(self):
        fm = _make_fm()
        dt = list(fm.dtypes.values())[0]
        dt.oper_expr['harv'] = ['_age >= 5']
        dt.operability['harv'] = {1: (5, 200)}
        result = fm.operable_dtypes('harv', 1)
        assert ('a', '1') in result
        assert 10 in result[('a', '1')]

    def test_operable_dtypes_no_match(self):
        fm = _make_fm()
        result = fm.operable_dtypes('nonexistent', 1)
        assert result == {}

    def test_inventory(self):
        fm = _make_fm()
        inv = fm.inventory(period=1)
        # period 1 area: ages 10 (5.0) and 20 (3.0), shifted by period_length=10
        # aged ages: 20 (5.0) and 30 (3.0), total = 8.0
        assert inv == 8.0

    def test_reset_areas_single_period(self):
        fm = _make_fm()
        dt = list(fm.dtypes.values())[0]
        assert 10 in dt._areas[1]
        fm.reset_areas(period=1)
        assert dt._areas[1] == {}  # period 1 cleared
        assert 10 in dt._areas[2]  # period 2 untouched

    def test_reset_areas_all_periods(self):
        fm = _make_fm()
        dt = list(fm.dtypes.values())[0]
        fm.reset_areas()  # all periods
        assert dt._areas[1] == {}
        assert dt._areas[2] == {}

    def test_reset_actions_specific_period_acode(self):
        fm = _make_fm()
        fm.applied_actions[1] = {'harv': {('a', '1'): {10: [2.0], 20: [1.0]}}}
        fm.applied_actions[2] = {'harv': {('a', '1'): {10: [1.5]}}}
        fm.reset_actions(period=1, acode='harv')
        assert fm.applied_actions[1]['harv'] == {}  # period 1 harv cleared
        assert ('a', '1') in fm.applied_actions[2]['harv']  # period 2 untouched

    def test_reset_actions_all_periods(self):
        fm = _make_fm()
        fm.applied_actions[1] = {'harv': {('a', '1'): {10: [2.0]}}}
        fm.applied_actions[2] = {'harv': {('a', '1'): {10: [1.5]}}}
        fm.reset_actions()  # all periods and all acodes
        # reset_actions clears inner dict but keeps period keys
        assert fm.applied_actions[1]['harv'] == {}
        assert fm.applied_actions[2]['harv'] == {}

    def test_reset_actions_sticky_preserved(self):
        fm = _make_fm()
        fm.applied_actions[1] = {'harv_sticky': {('a', '1'): {10: [2.0]}}}
        fm.reset_actions(period=1, acode='harv_sticky')
        # sticky actions should NOT be reset by default
        assert ('a', '1') in fm.applied_actions[1]['harv_sticky']

    def test_reset_actions_sticky_override(self):
        fm = _make_fm()
        fm.applied_actions[1] = {'harv_sticky': {('a', '1'): {10: [2.0]}}}
        fm.reset_actions(period=1, acode='harv_sticky', override_sticky=True)
        # with override, sticky actions ARE reset
        assert fm.applied_actions[1]['harv_sticky'] == {}

    def test_operated_area_basic(self):
        """operated_area sums area from applied_actions for given acode/period."""
        fm = _make_fm()
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}], 20: [3.0, {}]}}
        }
        result = fm.operated_area('harv', 1)
        assert result == 5.0

    def test_operated_area_with_dtype_filter(self):
        fm = _make_fm()
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}]}, ('b', '2'): {15: [1.5, {}]}}
        }
        result = fm.operated_area('harv', 1, dtype_key=('a', '1'))
        assert result == 2.0

    def test_operated_area_with_age_filter(self):
        fm = _make_fm()
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}], 20: [3.0, {}]}}
        }
        result = fm.operated_area('harv', 1, age=20)
        assert result == 3.0

    def test_operated_area_no_actions(self):
        """operated_area returns 0 when no actions applied for period/acode."""
        fm = _make_fm()
        fm.applied_actions[1] = {'harv': {}}  # period exists, harv is empty
        result = fm.operated_area('harv', 1)
        assert result == 0.0

    def test_compile_product_simple_expr(self):
        """compile_product with a numeric expression uses area * expr value."""
        fm = _make_fm()
        # Structure: [area, {yname: value}] - ycomp products dict is empty
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}], 20: [3.0, {}]}}
        }
        result = fm.compile_product(period=1, expr='2.0', acode='harv')
        assert result == 10.0  # (2.0 + 3.0) * 2.0

    def test_compile_product_with_dtype_filter(self):
        fm = _make_fm()
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}]}, ('b', '2'): {15: [4.0, {}]}}
        }
        result = fm.compile_product(period=1, expr='1.0', acode='harv',
                                    dtype_keys=[('a', '1')])
        assert result == 2.0

    def test_compile_product_coeff_mode(self):
        """coeff=True forces area to 1 regardless of actual area."""
        fm = _make_fm()
        fm.applied_actions[1] = {
            'harv': {('a', '1'): {10: [2.0, {}], 20: [3.0, {}]}}
        }
        result = fm.compile_product(period=1, expr='5.0', acode='harv', coeff=True)
        # With coeff=True, each age slot contributes 5.0 * 1.0 = 5.0
        assert result == 10.0  # 2 slots * 5.0

    def test_compile_product_no_applied_actions(self):
        """compile_product returns 0 when no actions applied."""
        fm = _make_fm()
        fm.applied_actions[1] = {'harv': {}}  # period exists, harv is empty
        result = fm.compile_product(period=1, expr='10.0', acode='harv')
        assert result == 0.0


# ---------------------------------------------------------------------------
# GreedyAreaSelector
# ---------------------------------------------------------------------------

class TestGreedyAreaSelector:
    def test_init(self):
        fm = ForestModel.__new__(ForestModel)
        sel = GreedyAreaSelector(fm)
        assert sel.parent is fm
