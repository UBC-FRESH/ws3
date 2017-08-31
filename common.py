# -*- coding: utf-8 -*-
###################################################################################
# MIT License

# Copyright (c) 2015-2017 Gregory Paradis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

"""
This module contains definitions for global attributes, functions, and classes that might be used anywhere in the package.

Attributes:
    HORIZON_DEFAULT (int): Default value for ''.
    PERIOD_LENGTH_DEFAULT (int): Default number of years per period.
    MIN_AGE_DEFAULT (int): Default value for `core.Curve.xmin`.
    MAX_AGE_DEFAULT (int): Default value for `core.Curve.xmax`.
    CURVE_EPSILON_DEFAULT (float): Defalut value for `core.Curve.epsilon`.
AREA_EPSILON_DEFAULT = 0.01
    
"""


import time
import scipy
import numpy as np
import pacal


try:
    import cPickle as pickle
except:
    import pickle
import math
#from math import exp, log


def timed(func):
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print '%s took %.3f seconds.' % (func.func_name, t)
        return result
    return wrapper
from scipy.stats import norm

HORIZON_DEFAULT = 30
PERIOD_LENGTH_DEFAULT = 5
MIN_AGE_DEFAULT = 0
MAX_AGE_DEFAULT = 1000
CURVE_EPSILON_DEFAULT = 0.01
AREA_EPSILON_DEFAULT = 0.01

##################################################
# not used (delete) [commenting out]
SPECIES_GROUPS_QC  = {
    'ERR':'ERR',
    'ERS':'ERS',
    'BOP':'BOP',
    'EPR':'SEP',
    'CHB':'FTO',
    'EPN':'SEP',
    'EPO':'SEP',
    'BOJ':'BOJ',
    'PEH':'PEU',
    'ERA':'ERR',
    'CAC':'FTO',
    'ERN':'ERR',
    'PEG':'PEU',
    'EPB':'SEP',
    'CAF':'FTO',
    'PEB':'PEU',
    'BOG':'BOP',
    'SOA':'NCO',
    'SAL':'NCO',
    'SAB':'SAB',
    'PIB':'PIN',
    'PIG':'SEP',
    'PRU':'AUR',
    'PET':'PEU',
    'CET':'FTO',
    'PRP':'NCO',
    'PIR':'PIN',
    'PIS':'SEP',
    'PED':'PEU',
    'FRA':'FTO',
    'CHE':'FTO',
    'CHG':'FTO',
    'FRN':'FTO',
    'THO':'AUR',
    'CHR':'FTO',
    'FRP':'FTO',
    'TIL':'FTO',
    'MEL':'AUR',
    'ORT':'FTO',
    'ORR':'FTO',
    'MEH':'AUR',
    'NOC':'FTO',
    'HEG':'HEG',
    'OSV':'FTO',
    'ORA':'FTO'
}

##################################################
# not used (delete) [commenting out]
SPECIES_GROUPS_WOODSTOCK_QC  = {
    'ERR':'ERR',
    'ERS':'ERS',
    'BOP':'BOP',
    'EPR':'SEP',
    'CHB':'FTO',
    'EPN':'SEP',
    'EPO':'SEP',
    'BOJ':'BOJ',
    'PEH':'PEU',
    'ERA':'ERR',
    'CAC':'FTO',
    'ERN':'ERR',
    'PEG':'PEU',
    'EPB':'SEP',
    'CAF':'FTO',
    'PEB':'PEU',
    'BOG':'BOP',
    'SOA':'NCO',
    'SAL':'NCO',
    'SAB':'SAB',
    'PIB':'PIN',
    'PIG':'SEP',
    'PRU':'AUR',
    'PET':'PEU',
    'CET':'FTO',
    'PRP':'NCO',
    'PIR':'PIN',
    'PIS':'SEP',
    'PED':'PEU',
    'FRA':'FTO',
    'CHE':'FTO',
    'CHG':'FTO',
    'FRN':'FTO',
    'THO':'AUR',
    'CHR':'FTO',
    'FRP':'FTO',
    'TIL':'FTO',
    'MEL':'AUR',
    'ORT':'FTO',
    'ORR':'FTO',
    'MEH':'AUR',
    'NOC':'FTO',
    'HEG':'HEG',
    'OSV':'FTO',
    'ORA':'FTO'
}

##################################################
# not used (delete) [commenting out]
##########################################
# keys correspond to bin labels
# values correspond to bin upper bounds (inclusive)
# AGE_CLASS_BINS_DEFAULT = {
#     '10':20,
#     '30':40,
#     '50':60,
#     '70':80,
#     '90':100,
#     '120+':MAX_AGE_DEFAULT
# }
##########################################
    

def is_num(s):
    """
    Returns True if s is a number.
    """
    try:
        float(s)
        return True
    except:
        return False

    
def _sylv_cred_f1(P,
                  vr,
                  vp,
                  rv=False,
                  C1a=4.511,
                  C2a=-0.628,
                  C7d=-0.391,
                  C8d=1.939,
                  C15h=3.912,
                  C16h=-0.0094,
                  C17i=0.0698,
                  C18j=9.2529,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (C1a*vr**C2a-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc

    
def _sylv_cred_f2(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.237,
                  C4b=2.592,
                  C7d=-0.237,
                  C8d=2.247,
                  C11f=4.3546,
                  C12f=0.34,
                  C13g=4.3543,
                  C14g=0.34,
                  C15h=3.912,
                  C16h=-0.0094,
                  C17i=0.0698,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def _sylv_cred_f3(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.237,
                  C4b=2.247,
                  C7d=-0.237,
                  C8d=2.247,
                  C15h=3.912,
                  C16h=-0.0094,
                  C17i=0.0698,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def _sylv_cred_f4(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.237,
                  C4b=2.592,
                  C7d=-0.237,
                  C8d=2.247,
                  C11f=4.3546,
                  C12f=0.34,
                  C13g=4.3546,
                  C14g=0.34,
                  C15h=3.912,
                  C16h=-0.0069,
                  C17i=0.0517,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def _sylv_cred_f5(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.237,
                  C4b=2.519,
                  C7d=-0.237,
                  C8d=2.247,
                  C11f=4.3546,
                  C12f=0.34,
                  C13g=4.3546,
                  C14g=0.34,
                  C15h=3.912,
                  C16h=-0.0069,
                  C17i=0.0517,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = ((exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C11f/vr**C12f-C13g/vp**C14g
           +C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus)
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def _sylv_cred_f6(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.237,
                  C4b=2.519,
                  C5c=-0.391,
                  C6c=2.017,
                  C7d=-0.237,
                  C8d=2.247,
                  C9e=-0.391,
                  C10e=1.939,
                  C11f=4.3546,
                  C12f=0.34,
                  C13g=4.3546,
                  C14g=0.34,
                  C15h=3.912,
                  C16h=-0.0069,
                  C17i=0.0517,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (((exp(C3b*log(vr)+C4b)+exp(C5c*log(vr)+C6c)-exp(C7d*log(vp)+C8d)-exp(C9e*log(vp)+C10e))/2
            +C11f/vr**C12f-C13g/vp**C14g+C15h*exp(C16h*P)-C17i*P+C18j*P)*Kmult+Kplus)
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def _sylv_cred_f7(P,
                  vr,
                  vp,
                  rv=False,
                  C3b=-0.391,
                  C4b=2.2,
                  C7d=-0.391,
                  C8d=1.939,
                  C15h=3.912,
                  C16h=-0.0069,
                  C17i=0.0517,
                  C18j=7.1029,
                  Kmult=1.,
                  Kplus=0.):
    exp = pacal.exp if rv else math.exp
    log = pacal.log if rv else math.exp
    sc = (exp(C3b*log(vr)+2.2)-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
    if rv:
        return sc.mean() # expected value, given random variates
    else:
        return sc


def sylv_cred(P, vr, vp, formula):
    """
    Returns sylviculture credit ($ per hectare), given P (volume harvested per hectare), vr (mean piece size of harvested stems), vp (mean piece size of stand before harvesting), and formula index (1 to 7).
    Assumes that variables (P, vr, vp) are deterministic.
    """
    f = {1:_sylv_cred_f1,
         2:_sylv_cred_f2,
         3:_sylv_cred_f3,
         4:_sylv_cred_f4,
         5:_sylv_cred_f5,
         6:_sylv_cred_f6,
         7:_sylv_cred_f7}
    return f[formula](P, vr, vp)


def sylv_cred_rv(P_mu, P_sigma, tv_mu, tv_sigma, N_mu, N_sigma, psr,
                 treatment_type=None, cover_type=None, formula=None,
                 P_min=20., tv_min=50., N_min=200., ps_min=0.05,
                 E_fromintegral=False, e=0.01, n=1000):
    """
    Returns sylviculture credit ($ per hectare), given P (volume harvested per hectare), vr (mean piece size of harvested stems), vp (mean piece size of stand before harvesting), and formula index (1 to 7).
    Assumes that variables (P, vr, vp) are random variates (returns expected value of function, using PaCAL packages to model random variates, assuming normal distribution for all three variables).
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
        E = f[formula](P, vr, vp, rv=True)
    else:
        # estimate expected value E(f(P, vr, vp)) using Monte Carlo simulation (until convergence to E_tol)
        E = 0.
        dE = np.inf
        i = 1
        while dE > e:
            args = zip(P.rand(n), vr.rand(n), vp.rand(n))
            while len(args) > 0: # process random args in in n-length chunks
                _E = E
                E = ((i - 1) * E + f[formula](*args.pop())) / i
                dE = abs((E - _E) / _E) if _E else np.inf
                i += 1
    return E


def sylv_cred_formula(treatment_type, cover_type):
    """
    Returns sylviculture credit formula index, given treatment type and cover type.
    """
    if treatment_type == 'ec':
        return 1 if cover_type.lower() in ['r', 'm'] else 2
    if treatment_type == 'cj':
        return 4
    if treatment_type == 'cprog':
        return 7 if cover_type.lower() in ['r', 'm'] else 4        
    return 0


def piece_size_ratio(treatment_type, cover_type, piece_size_ratios):
    """
    Returns piece size ratio.
    Assume Action.is_harvest in [0, 1, 2, 3]
    Assume cover_type in ['r', 'm', 'f']
    Return vr/vp ratio, where
      vr is mean piece size of harvested stems, and
      vp is mean piece size of stand before harvesting.
    """
    if treatment_type in [1, 2, 3] and cover_type in ['r', 'm', 'f']:
        if piece_size_ratios:
            return piece_size_ratios[treatment_type][cover_type]
        else:
            return 1.
    else:
        return 0.


def harv_cost(piece_size,
              is_finalcut,
              is_toleranthw,
              partialcut_extracare=False,              
              A=1.97, B=0.405, C=0.169, D=0.164, E=0.202, F=13.6, G=8.83, K=0.,
              rv=False):
    """
    Returns harvest cost, given piece size, treatment type (final cut or not), stand type (tolerant hardwood or not), partialcut "extra care" flag, and a series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults [extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB]). 
    Assumes that variables are deterministic.
    """
    _ifc = float(is_finalcut)
    _ith = float(is_toleranthw)
    _pce = float(partialcut_extracare)
    log = pacal.log if rv else math.log
    exp = pacal.exp if rv else math.exp
    _exp = A - (B * log(piece_size)) + (C * _pce) + (D * _ifc) - (E * (1 - _ith))
    hc = exp(_exp) + ((F * _ith) + (G * (1 - _ith))) + K
    if rv:
        return hc.mean()
    else:
        return hc

    
def harv_cost_rv(tv_mu, tv_sigma, N_mu, N_sigma, psr,
                 is_finalcut,
                 is_toleranthw,
                 partialcut_extracare=False,
                 tv_min=50., N_min=200., ps_min=0.05,
                 E_fromintegral=False, e=0.01, n=1000):
    """
    Returns harvest cost, given piece size, treatment type (final cut or not), stand type (tolerant hardwood or not), partialcut "extra care" flag, and a series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults [extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB]). 
    Assumes that variables are random variates (returns expected value of function, using PaCAL packages to model random variates, assuming normal distribution for all three variables).
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
        E = harv_cost(vr, is_finalcut, is_toleranthw, rv=True)
    else:
        # estimate expected value E(f(vr)) using Monte Carlo simulation (until convergence to E_tol)
        E = 0.
        dE = np.inf
        i = 1
        while dE > e:
            args = list(vr.rand(n))
            while len(args) > 0: # process random args in in n-length chunks
                _E = E
                E = ((i - 1) * E + harv_cost(args.pop(), is_finalcut, is_toleranthw)) / i
                dE = abs((E - _E) / _E) if _E else np.inf
                i += 1
    return E


def harv_cost_wec(piece_size,
                  is_finalcut,
                  is_toleranthw,
                  sigma,
                  nsigmas=3,
                  **kwargs):
    """
    Estimate harvest cost with error correction.
    :float piece_size: mean piece size
    :bool is_finalcut: True if harvest treatment is final cut, False otherwise
    :bool is_toleranthw: True if tolerant hardwood cover type, False otherwise
    :float sigma: standard deviation of piece size estimator
    :int nsigmas: number of standard deviations to model on either side of the mean (default 3)
    :float binw: width of bins for weighted numerical integration, in multiples of sigma (default 1.0)
    """
    # bin centerpoints
    rv = norm(loc=piece_size, scale=sigma)
    X = sorted([(piece_size + (sigma * (i - (1. * 0.5)) * sign)) 
               for i in range(1, nsigmas+1) for sign in [-1, +1]])
    return sum(harv_cost(x, is_finalcut, is_toleranthw, **kwargs) * sigma * rv.pdf(x) for x in X)
        

# class rvquot_gen(scipy.stats.rv_continuous):
#     def __init__(self, 
#                  locn, scalen, 
#                  locd, scaled,
#                  a=0.,
#                  b=1.,
#                  name='rvquot'):
#         super(rvquot_gen, self).__init__(a=a,b=b,name=name)
#         self.pacal_dist = pacal.NormalDistr(locn, scalen) / pacal.NormalDistr(locd, scaled)
#         self.integral = scipy.integrate.quad(self.f, self.a, self.b)[0]

#     def f(self, x):
#         return self.pacal_dist.pdf(x)
        
#     def _pdf(self, x):
#         return self.f(x)/self.integral


# class rvquot_gen(scipy.stats.rv_continuous):
#     def __init__(self, 
#                  locn, scalen, 
#                  locd, scaled,
#                  a=0.0,
#                  b=1.0,
#                  name='rvquot',
#                  loc=0.):
#         super(rvquot_gen, self).__init__(a=a,b=b,name=name)
#         self.pacal_dist = pacal.NormalDistr(locn, scalen) / pacal.NormalDistr(locd, scaled)
#         self.loc = loc
#         self.integral = scipy.integrate.quad(self.f, self.a, self.b)[0]
        
#     def f(self, x):
#         return self.pacal_dist.pdf(x-self.loc)
        
#     def _pdf(self, x):
#         return self.f(x)/self.integral

