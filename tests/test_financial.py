import sys
sys.path.append('../ws3/')
import pytest
#import numpy as np
#import fiona
#import os
import math
import ws3.financial
from ws3.financial import sylv_cred, sylv_cred_formula, piece_size_ratio, harv_cost


def test_math_funcs_returns_exp_and_log_not_exp_twice():
    """
    Regression test for the log/exp mixup.

    Seven of the eight hand-copied binding pairs in ws3/financial.py bound ``log``
    to ``math.exp`` rather than ``math.log``, so the deterministic silvicultural
    credit results were wrong by roughly 40x while raising nothing. The bindings
    are now produced by a single helper; this pins its contract.
    """
    exp, log = ws3.financial._math_funcs(False)
    assert exp is math.exp
    assert log is math.log
    assert exp is not log


def test_sylv_cred_f1_matches_independently_computed_value():
    """
    Pin _sylv_cred_f1 against the formula evaluated independently.

    Written against the published expression rather than by calling the function
    and recording whatever it returned, so that it fails if the exp/log mixup is
    ever reintroduced.
    """
    P, vr, vp = 50.0, 0.5, 0.35
    C1a, C2a = 4.511, -0.628
    C7d, C8d = -0.391, 1.939
    C15h, C16h, C17i, C18j = 3.912, -0.0094, 0.0698, 9.2529

    expected = (C1a * vr ** C2a
                - math.exp(C7d * math.log(vp) + C8d)
                + C15h * math.exp(C16h * P)
                - C17i * P
                + C18j) * P

    assert ws3.financial._sylv_cred_f1(P, vr, vp) == pytest.approx(expected)


def test_sylv_cred_f1_rejects_the_buggy_formulation():
    """The buggy exp-for-log form differs by a wide margin, so the pin is meaningful."""
    P, vr, vp = 50.0, 0.5, 0.35
    C1a, C2a = 4.511, -0.628
    C7d, C8d = -0.391, 1.939
    C15h, C16h, C17i, C18j = 3.912, -0.0094, 0.0698, 9.2529

    buggy = (C1a * vr ** C2a
             - math.exp(C7d * math.exp(vp) + C8d)      # exp where log belongs
             + C15h * math.exp(C16h * P)
             - C17i * P
             + C18j) * P

    assert ws3.financial._sylv_cred_f1(P, vr, vp) != pytest.approx(buggy)


def test_pacal_guard_raises_informative_error_when_unavailable(monkeypatch):
    """
    rv=True must fail with an actionable message, not a bare NameError.

    Previously ``pacal`` was never imported at all (the import sat behind a
    permanently-true PACAL_BROKEN flag), so every probabilistic path raised
    ``NameError: name 'pacal' is not defined``.
    """
    monkeypatch.setattr(ws3.financial, 'pacal', None)
    with pytest.raises(NotImplementedError, match='PaCal'):
        ws3.financial._require_pacal()
    with pytest.raises(NotImplementedError, match='rv=False'):
        ws3.financial._math_funcs(True)


def test_pacal_available_reports_a_bool():
    assert isinstance(ws3.financial.pacal_available(), bool)


@pytest.mark.skipif(not ws3.financial.pacal_available(),
                    reason='PaCal not installed (pip install ws3[rv])')
def test_rv_path_runs_when_pacal_present():
    """The probabilistic path produces a finite result when PaCal is available."""
    result = harv_cost(0.35, True, False, rv=True)
    assert math.isfinite(result)


def test_sylv_cred():
    # Test data
    P = 10.0
    vr = 2.0
    vp = 1.0
    formula = 1

    # Call the function
    result = sylv_cred(P, vr, vp, formula)

    # Corrected 2026-07-29. This previously asserted 126.33, which was the output
    # of the buggy implementation that bound `log` to `math.exp` (see #100). The
    # test had been written by running the code and recording what it returned, so
    # it pinned the defect in place rather than catching it.
    #
    # Evaluating the published formula independently at P=10, vr=2, vp=1:
    #
    #   with math.log (correct):  80.83076   <- current implementation
    #   with math.exp (as coded): 126.33232  <- former expected value
    #
    # The former value matches the buggy form to five decimal places, which is
    # what confirms the origin of the number.
    expected_result = 80.83076

    # Assertion
    assert result == pytest.approx(expected_result, rel=1.3e-04)


def test_sylv_cred_formula_ec_m():
    treatment_type = 'ec'
    cover_type = 'M'

    assert sylv_cred_formula(treatment_type, cover_type) == 1


def test_sylv_cred_formula_ec_other():
    treatment_type = 'ec'
    cover_type = 'S'

    assert sylv_cred_formula(treatment_type, cover_type) == 2


def test_sylv_cred_formula_cj():
    treatment_type = 'cj'
    cover_type = 'S'

    assert sylv_cred_formula(treatment_type, cover_type) == 4


def test_sylv_cred_formula_cprog_r():
    treatment_type = 'cprog'
    cover_type = 'R'

    assert sylv_cred_formula(treatment_type, cover_type) == 7


def test_sylv_cred_formula_cprog_m():
    treatment_type = 'cprog'
    cover_type = 'M'

    assert sylv_cred_formula(treatment_type, cover_type) == 7


def test_piece_size_ratio_valid():
    treatment_type = 2
    cover_type = 'm'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0.8


def test_piece_size_ratio_invalid_treatment_type():
    treatment_type = 4
    cover_type = 'r'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0


def test_piece_size_ratio_invalid_cover_type():
    treatment_type = 2
    cover_type = 's'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0


def test_piece_size_ratio_empty_piece_size_ratios():
    treatment_type = 2
    cover_type = 'r'
    piece_size_ratios = {}

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 1.0


def test_harv_cost():
    piece_size = 10
    is_finalcut = True
    is_toleranthw = False
    partialcut_extracare = False
    A, B, C, D, E, F, G, K = 1.97, 0.405, 0.169, 0.164, 0.202, 13.6, 8.83, 0.0

    expected_result_1 = (
        A - (B * math.log(piece_size)) + (C * float(partialcut_extracare)) +
        (D * float(is_finalcut)) - (E * (1 - float(is_toleranthw)))
    )
    expected_result = math.exp(expected_result_1)+ ((F * float(is_toleranthw)) + (G * (1 - float(is_toleranthw)))) + K

    assert harv_cost(piece_size, is_finalcut, is_toleranthw, partialcut_extracare, A, B, C, D, E, F, G, K) == expected_result
