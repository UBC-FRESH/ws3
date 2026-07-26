"""
This module contains a number of functions used for calculating 
silviculture credits and harvest costs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm

#################################################################################################
# PaCal breaks when trying to import numpy.fft.fftpack (names have changed or some such... yuck).
# Note that this will breaks the folowing functions in this ws3.common
#   _sylv_credit_f1
#   _sylv_credit_f2
#   _sylv_credit_f3
#   _sylv_credit_f4
#   _sylv_credit_f5
#   _sylv_credit_f6
#   _sylv_credit_f7
#   sylv_cred_rv
#   harv_cost_rv
# TO DO:
#   Patch PaCal 1.6, maybe using pypatch (as part of the ws3 build process, in setup.py).
# The fix:
#   Patch line 29 in pacal/utils.py from
#     from numpy.fft.fftpack import fft, ifft
#   to
#     from numpy.fft import fft, ifft 
# 
PACAL_BROKEN = True
if not PACAL_BROKEN:
    import pacal
#################################################################################################
import math

    
def _sylv_cred_f1(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C1a: float = 4.511,
                  C2a: float = -0.628,
                  C7d: float = -0.391,
                  C8d: float = 1.939,
                  C15h: float = 3.912,
                  C16h: float = -0.0094,
                  C17i: float = 0.0698,
                  C18j: float = 9.2529,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (C1a*vr**C2a-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f2(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.237,
                  C4b: float = 2.592,
                  C7d: float = -0.237,
                  C8d: float = 2.247,
                  C11f: float = 4.3546,
                  C12f: float = 0.34,
                  C13g: float = 4.3543,
                  C14g: float = 0.34,
                  C15h: float = 3.912,
                  C16h: float = -0.0094,
                  C17i: float = 0.0698,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp  # type: ignore[assignment]
    log = pacal.log if rv else math.exp  # type: ignore[assignment]
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f3(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.237,
                  C4b: float = 2.247,
                  C7d: float = -0.237,
                  C8d: float = 2.247,
                  C15h: float = 3.912,
                  C16h: float = -0.0094,
                  C17i: float = 0.0698,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp  # type: ignore[assignment]
    log = pacal.log if rv else math.exp  # type: ignore[assignment]
    sc = (exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f4(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.237,
                  C4b: float = 2.592,
                  C7d: float = -0.237,
                  C8d: float = 2.247,
                  C11f: float = 4.3546,
                  C12f: float = 0.34,
                  C13g: float = 4.3546,
                  C14g: float = 0.34,
                  C15h: float = 3.912,
                  C16h: float = -0.0069,
                  C17i: float = 0.0517,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp  # type: ignore[assignment]
    log = pacal.log if rv else math.exp  # type: ignore[assignment]
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f5(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.237,
                  C4b: float = 2.519,
                  C7d: float = -0.237,
                  C8d: float = 2.247,
                  C11f: float = 4.3546,
                  C12f: float = 0.34,
                  C13g: float = 4.3546,
                  C14g: float = 0.34,
                  C15h: float = 3.912,
                  C16h: float = -0.0069,
                  C17i: float = 0.0517,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f6(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.237,
                  C4b: float = 2.519,
                  C5c: float = -0.391,
                  C6c: float = 2.017,
                  C7d: float = -0.237,
                  C8d: float = 2.247,
                  C9e: float = -0.391,
                  C10e: float = 1.939,
                  C11f: float = 4.3546,
                  C12f: float = 0.34,
                  C13g: float = 4.3546,
                  C14g: float = 0.34,
                  C15h: float = 3.912,
                  C16h: float = -0.0069,
                  C17i: float = 0.0517,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (((exp(C3b*log(vr)+C4b)+exp(C5c*log(vr)+C6c)-exp(C7d*log(vp)+C8d)-exp(C9e*log(vp)+C10e))/2
            +C11f/vr**C12f-C13g/vp**C14g+C15h*exp(C16h*P)-C17i*P+C18j*P)*Kmult+Kplus)
    if rv:
        return float(sc.mean())  # type: ignore[union-attr]
    else:
        return float(sc)  # type: ignore[misc]


def _sylv_cred_f7(P: float,
                  vr: float,
                  vp: float,
                  rv: bool = False,
                  C3b: float = -0.391,
                  C4b: float = 2.2,
                  C7d: float = -0.391,
                  C8d: float = 1.939,
                  C15h: float = 3.912,
                  C16h: float = -0.0069,
                  C17i: float = 0.0517,
                  C18j: float = 7.1029,
                  Kmult: float = 1.,
                  Kplus: float = 0.) -> float:
    exp = pacal.exp if rv else math.exp  # type: ignore[assignment]
    log = pacal.log if rv else math.exp  # type: ignore[assignment]
    sc = (exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return float(sc.mean())  # type: ignore[union-attr] # expected value, given random variates
    else:
        return float(sc)  # type: ignore[misc]


def sylv_cred(P: float, vr: float, vp: float, formula: int) -> float:
    """
    This function returns sylviculture credit ($ per hectare).

    :param float P: Volume harvested per hectare.
    :param float vr: Mean piece size of harvested stems.
    :param float vp: mean piece size of stand before harvesting.
    :param formula: formula index (1 to 7).        
    """
    f = {1:_sylv_cred_f1,
         2:_sylv_cred_f2,
         3:_sylv_cred_f3,
         4:_sylv_cred_f4,
         5:_sylv_cred_f5,
         6:_sylv_cred_f6,
         7:_sylv_cred_f7}
    return f[formula](P, vr, vp)  # type: ignore[operator,no-any-return]


def sylv_cred_rv(P_mu: float, P_sigma: float, tv_mu: float, tv_sigma: float, N_mu: float, N_sigma: float, psr: float,
                 treatment_type: Optional[str] = None, cover_type: Optional[str] = None, formula: Optional[int] = None,
                 P_min: float = 20., tv_min: float = 50., N_min: float = 200., ps_min: float = 0.05,
                 E_fromintegral: bool = False, e: float = 0.01, n: int = 1000) -> float:
    
    """
    This function returns sylviculture credit ($ per hectare).

    :param float P: Volume harvested per hectare.
    :param float vr: Mean piece size of harvested stems.
    :param float vp: mean piece size of stand before harvesting.
    :param formula: formula index (1 to 7).

    .. note:: Assumes that variables ``(P, vr, vp)`` are random variates (returns expected value of function, using PaCAL packages to model random variates, assuming normal distribution for all three variables).
        Can use either PaCAL numerical integration (sssslow!), or custom numerical integration using Monte Carlo sampling (default).   
    """
    if treatment_type and cover_type:
        formula = sylv_cred_formula(treatment_type, cover_type)
    assert formula
    # PaCAL overrides the | operator to implement conditional distributions
    P = pacal.NormalDistr(P_mu, P_sigma) | pacal.Gt(P_min)
    tv = pacal.NormalDistr(tv_mu, tv_sigma) | pacal.Gt(tv_min)
    N = pacal.NormalDistr(N_mu, N_sigma) | pacal.Gt(N_min)
    vp = (tv / N) | pacal.Gt(ps_min)
    #vr = vp + (vp.mean() * (1 - psr))
    # truncate again in case psr < 1 (shifts distn to the left)
    vr = (vp + (vp.mean() * (psr - 1.))) | pacal.Gt(ps_min)  
    f = {1:_sylv_cred_f1,
         2:_sylv_cred_f2,
         3:_sylv_cred_f3,
         4:_sylv_cred_f4,
         5:_sylv_cred_f5,
         6:_sylv_cred_f6,
         7:_sylv_cred_f7}
    #print ' formula', formula
    if E_fromintegral:
        # estimate expected value E(f(P, vr, vp)) using PaCAL numerical integration functions (sssssslow!) 
        E = f[formula](P, vr, vp, rv=True)  # type: ignore[operator]
    else:
        # estimate expected value E(f(P, vr, vp)) using Monte Carlo simulation (until convergence to E_tol)
        E = 0.
        dE = np.inf
        i = 1
        while dE > e:
            args = list(zip(P.rand(n), vr.rand(n), vp.rand(n)))
            while len(args) > 0: # process random args in in n-length chunks
                _E = E
                E = ((i - 1) * E + f[formula](*args.pop())) / i  # type: ignore[operator]
                dE = abs((E - _E) / _E) if _E else np.inf
                i += 1
    return E  # type: ignore[no-any-return]


def sylv_cred_formula(treatment_type: str, cover_type: str) -> int:
    """
    Returns sylviculture credit formula index.

    :param str treatment_type: Treatment type.
    :param str cover_type: Cover type.
    """
    if treatment_type == 'ec':
        return 1 if cover_type.lower() in ['r', 'm'] else 2
    if treatment_type == 'cj':
        return 4
    if treatment_type == 'cprog':
        return 7 if cover_type.lower() in ['r', 'm'] else 4        
    return 0


def piece_size_ratio(treatment_type: int, cover_type: str, piece_size_ratios: Optional[Dict[int, Dict[str, float]]]) -> float:
    """
    Returns piece size ratio.
    
    Assume Action.is_harvest in [0, 1, 2, 3]
    
    Assume cover_type in ['r', 'm', 'f']
    
    Return vr/vp ratio, where
      - vr is mean piece size of harvested stems, and
      - vp is mean piece size of stand before harvesting.
    """
    if treatment_type in [1, 2, 3] and cover_type in ['r', 'm', 'f']:
        if piece_size_ratios:
            return piece_size_ratios[treatment_type][cover_type]
        else:
            return 1.
    else:
        return 0.


def harv_cost(piece_size: float,
              is_finalcut: bool,
              is_toleranthw: bool,
              partialcut_extracare: bool = False,              
              A: float = 1.97, B: float = 0.405, C: float = 0.169, D: float = 0.164, E: float = 0.202, F: float = 13.6, G: float = 8.83, K: float = 0.,
              rv: bool = False) -> float:
    """
    Returns harvest cost.

    :param float piece_size: Piece size.
    :param bool is_finalcut: Treatment type (final cut or not).
    :param bool is_toleranthw: Stand type (tolerant hardwood or not).
    :param bool partialcut_extracare: Partialcut "extra care" flag.
    :param float A: Series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults that are extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB).
    :param bool rv: Types of variables (default Variables are deterministic).        
    """

    _ifc = float(is_finalcut)
    _ith = float(is_toleranthw)
    _pce = float(partialcut_extracare)
    log = pacal.log if rv else math.log  # type: ignore[assignment]
    exp = pacal.exp if rv else math.exp  # type: ignore[assignment]
    _exp = A - (B * log(piece_size)) + (C * _pce) + (D * _ifc) - (E * (1 - _ith))
    hc = exp(_exp) + ((F * _ith) + (G * (1 - _ith))) + K
    if rv:
        return float(hc.mean())  # type: ignore[union-attr]
    else:
        return float(hc)  # type: ignore[misc]

    
def harv_cost_rv(tv_mu: float, tv_sigma: float, N_mu: float, N_sigma: float, psr: float,
                 is_finalcut: bool,
                 is_toleranthw: bool,
                 partialcut_extracare: bool = False,
                 tv_min: float = 50., N_min: float = 200., ps_min: float = 0.05,
                 E_fromintegral: bool = False, e: float = 0.01, n: int = 1000) -> float:

    """
    Returns harvest cost.

    
    :param bool is_finalcut: Treatment type (final cut or not).
    :param bool is_toleranthw: Stand type (tolerant hardwood or not).
    :param bool partialcut_extracare: Partialcut "extra care" flag.
    :param float A: Series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults that are extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB).
    :param bool rv: Types of variables (default Variables random variates).
        Can use either PaCAL numerical integration (sssslow!), or custom numerical integration using Monte Carlo sampling (default).       
    """
    

    # PaCAL overrides the | operator to implement conditional distributions
    tv = pacal.NormalDistr(tv_mu, tv_sigma) | pacal.Gt(tv_min)
    N = pacal.NormalDistr(N_mu, N_sigma) | pacal.Gt(N_min)
    vp = (tv / N) | pacal.Gt(ps_min)
    #vr = vp + (vp.mean() * (1 - psr))
    # truncate again in case psr < 1 (shifts distn to the left)
    vr = (vp + (vp.mean() * (psr - 1.))) | pacal.Gt(ps_min)
    if E_fromintegral:
        # estimate expected value E(f(vr)) using PaCAL numerical integration functions (sssssslow!) 
        E = harv_cost(vr, is_finalcut, is_toleranthw, rv=True)  # type: ignore[operator]
    else:
        # estimate expected value E(f(vr)) using Monte Carlo simulation (until convergence to E_tol)
        E = 0.
        dE = np.inf
        i = 1
        while dE > e:
            args = list(vr.rand(n))
            while len(args) > 0: # process random args in in n-length chunks
                _E = E
                E = ((i - 1) * E + harv_cost(args.pop(), is_finalcut, is_toleranthw)) / i  # type: ignore[operator]
                dE = abs((E - _E) / _E) if _E else np.inf
                i += 1
    return E


def harv_cost_wec(piece_size: float,
                  is_finalcut: bool,
                  is_toleranthw: bool,
                  sigma: float,
                  nsigmas: int = 3,
                  **kwargs: Any) -> float:
    """
    Estimate harvest cost with error correction.

    :param float piece_size: Mean piece size.
    :param bool is_finalcut: True if harvest treatment is final cut, False otherwise.
    :param bool is_toleranthw: True if tolerant hardwood cover type, False otherwise.
    :param bool sigma: Standard deviation of piece size estimator.
    :param int nsigmas: Number of standard deviations to model on either side of the mean (default 3).
    :param float binw: Width of bins for weighted numerical integration, in multiples of sigma (default 1.0).       
    """

    # bin centerpoints
    rv = norm(loc=piece_size, scale=sigma)
    X = sorted([(piece_size + (sigma * (i - (1. * 0.5)) * sign)) 
               for i in range(1, nsigmas+1) for sign in [-1, +1]])
    return sum(harv_cost(x, is_finalcut, is_toleranthw, **kwargs) * sigma * rv.pdf(x) for x in X)  # type: ignore[misc,no-any-return]
        
