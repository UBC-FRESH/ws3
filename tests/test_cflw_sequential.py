"""Unit tests for consecutive-period sequential-flow even-flow constraints.

Covers the ``cflw_e`` extension (ws3 issue #152): the legacy symmetric
reference-period band keeps working, and the extended form supports separate
decrease/increase tolerances and a ``"consecutive"`` (previous-period)
reference, enabling the classic FORPLAN policies (NDY, bounded decline,
bounded deviation; cf. Daugherty 1991, eq. 3-4/3-5, Table 5.6).
"""

from __future__ import annotations

import ws3.opt as opt
from ws3.forest import _normalize_cflw_e
from ws3.forest_helper import worker_cmp_cflw_phase3


class TestNormalizeCflwE:
    periods = [1, 2, 3]

    def test_legacy_tuple_is_symmetric_and_anchored(self):
        spec = _normalize_cflw_e(({1: 0.05, 2: 0.05, 3: 0.05}, 1), self.periods)
        # Legacy form: alpha == beta == eps, anchored to period 1 for all periods.
        assert spec == {1: (0.05, 0.05, 1), 2: (0.05, 0.05, 1), 3: (0.05, 0.05, 1)}

    def test_consecutive_ndy_anchors_to_previous_period(self):
        spec = _normalize_cflw_e(
            {"decrease": {1: 0.0, 2: 0.0, 3: 0.0}, "increase": None, "ref": "consecutive"},
            self.periods,
        )
        # Period 1 has no previous period (ref None); periods 2,3 anchor to t-1.
        assert spec[2] == (0.0, None, 1)
        assert spec[3] == (0.0, None, 2)

    def test_bounded_deviation_sets_both_bounds(self):
        spec = _normalize_cflw_e(
            {"decrease": {2: 0.1}, "increase": {2: 0.1}, "ref": "consecutive"}, self.periods
        )
        assert spec[2] == (0.1, 0.1, 1)

    def test_period_with_no_tolerance_is_omitted(self):
        spec = _normalize_cflw_e({"decrease": None, "increase": None, "ref": "consecutive"}, self.periods)
        assert spec == {}


class TestWorkerCmpCflwPhase3:
    # mu maps variable key -> coefficient; xnames maps key -> variable name.
    mu_t = {("i", "j1"): 2.0, ("i", "j2"): 3.0}
    mu_ref = {("i", "j1"): 1.0, ("i", "j2"): 4.0}
    xnames = {("i", "j1"): "x_a", ("i", "j2"): "x_b"}

    def test_ndy_builds_only_lower_bound(self):
        rows = worker_cmp_cflw_phase3((2, "hv", self.mu_t, self.mu_ref, 0.0, None, self.xnames))
        assert len(rows) == 1
        name, coeffs, sense, rhs = rows[0]
        assert sense == opt.SENSE_GEQ and rhs == 0.0
        # H_t - (1 - 0) * H_ref >= 0
        assert coeffs["x_a"] == 2.0 - 1.0
        assert coeffs["x_b"] == 3.0 - 4.0

    def test_bounded_deviation_builds_both_bounds(self):
        rows = worker_cmp_cflw_phase3((2, "hv", self.mu_t, self.mu_ref, 0.1, 0.1, self.xnames))
        assert len(rows) == 2
        lb = next(r for r in rows if r[2] == opt.SENSE_GEQ)
        ub = next(r for r in rows if r[2] == opt.SENSE_LEQ)
        # lb: H_t - 0.9 * H_ref ; ub: H_t - 1.1 * H_ref
        assert lb[1]["x_a"] == 2.0 - 0.9 * 1.0
        assert ub[1]["x_b"] == 3.0 - 1.1 * 4.0

    def test_missing_ref_key_defaults_to_zero(self):
        rows = worker_cmp_cflw_phase3((2, "hv", self.mu_t, {("i", "j1"): 1.0}, 0.0, None, self.xnames))
        coeffs = rows[0][1]
        assert coeffs["x_b"] == 3.0 - 0.0  # ref missing -> 0.0
