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

PACAL_BROKEN = True

import time
import scipy
import numpy as np
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
if not PACAL_BROKEN:
    import pacal
#################################################################################################
import rasterio
import hashlib
import re
import binascii

try:
    import pickle as pickle
except:
    import pickle
import math
#from math import exp, log
import fiona
from fiona.transform import transform_geom
from fiona.crs import from_epsg

def is_num(s):
    try:
        float(s)
        return True
    except:
        return False

def reproject(f, srs_crs, dst_crs):
    f['geometry'] = transform_geom(srs_crs, dst_crs, f['geometry'],
                          antimeridian_cutting=False,
                          precision=-1)
    return f

def clean_vector_data(src_path, dst_path, dst_name, prop_names, clean=True, tolerance=0.,
                      preserve_topology=True, logfn='clean_stand_shapefile.log', max_records=None,
                      theme0=None, prop_types=None, driver='ESRI Shapefile', dst_epsg=None,
                      update_area_prop=''):
    import logging
    import sys
    from shapely.geometry import mapping, shape, Polygon, MultiPolygon
    import fiona
    from collections import OrderedDict
    logging.basicConfig(filename=logfn, level=logging.INFO)
    snk1_path = '%s/%s.shp' % (dst_path, dst_name) 
    #snk2_path = dst_path[:-4]+'_error.shp' 
    snk2_path = '%s/%s_error.shp' % (dst_path, dst_name) 
    with fiona.open(src_path, 'r') as src:
        kwds1 = src.meta.copy()
        kwds2 = src.meta.copy()
        kwds1.update(driver=driver)
        kwds2.update(driver=driver)
        if dst_epsg:
            dst_crs = from_epsg(dst_epsg)
            kwds1.update(crs=dst_crs, crs_wkt=None)
        if not prop_types:
            prop_types = [('theme0', 'str:10')] if theme0 else []
            prop_types = prop_types + [(pn.lower(), src.schema['properties'][pn]) for pn in prop_names]
        kwds1['schema']['properties'] = OrderedDict(prop_types)
        kwds2['schema']['properties'] = OrderedDict(prop_types)
        print(kwds1)
        with fiona.open(snk1_path, 'w', **kwds1) as snk1, fiona.open(snk2_path, 'w', **kwds2) as snk2:
            n = len(src) if not max_records else max_records
            i = 0
            for f in src[:n]:
                i += 1
                prop_data = [('theme0', theme0)] if theme0 else []
                if prop_types:
                    prop_data = prop_data + [(prop_types[i+len(prop_data)][0], f['properties'][pn])
                                             for i, pn in enumerate(prop_names)]   
                else:
                    prop_data = prop_data + [(pn.lower(), f['properties'][pn]) for pn in prop_names]
                f.update(properties = OrderedDict(prop_data))
                try:
                    g = shape(f['geometry'])
                    if not g.is_valid:
                        _g = g.buffer(0)
                        ################################
                        # HACK
                        # Something changed (maybe in fiona?) and now all GDB datasets are
                        # loading as MultiPolygon geometry type (instead of Polygon). 
                        # The buffer(0) trick smashes the geometry back to Polygon, 
                        # so this hack upcasts it back to MultiPolygon.
                        # 
                        # Not sure how robust this is going to be (guessing not robust).
                        _g = MultiPolygon([_g])
                        assert _g.is_valid
                        assert _g.geom_type == 'MultiPolygon'
                        g = _g
                        ################################
                    ##################################################################
                    # The idea was to remove redundant vertices from polygons
                    # (to make datasets smaller, but also speed up geometry processing).
                    # This sort of worked, but was unstable so commented out for now.
                    # g = g.simplify(tolerance=tolerance, preserve_topology=True)
                    # if not g.is_valid:
                    #     _g = g.buffer(0)
                    #     assert _g.is_valid
                    #     assert _g.geom_type == 'Polygon'
                    #     g = _g
                    ##################################################################
                    f['geometry'] = mapping(g)
                    #print('geometry type 2', f['geometry']['type'])
                    if dst_epsg: f = reproject(f, src.crs, dst_crs)
                    if update_area_prop:
                        f['properties'][update_area_prop] = shape(f['geometry']).area
                    snk1.write(f)
                except Exception as e: # log exception and write uncleanable feature a separate shapefile
                    logging.exception("Error cleaning feature %s:", f['id'])
                    snk2.write(f)
    return snk1_path, snk2_path


def reproject_vector_data(src_path, snk_path, snk_epsg, driver='ESRI Shapefile'):
    import fiona
    from fiona.crs import from_epsg
    from pyproj import Proj, transform
    with fiona.open(src_path, 'r') as src:
        snk_crs = from_epsg(snk_epsg)
        src_proj, snk_proj = Proj(src.crs), Proj(snk_crs)
        kwds = src.meta.copy()
        kwds.update(crs=snk_crs, crs_wkt=None)
        kwds.update(driver=driver)
        with fiona.open(snk_path, 'w', **kwds) as snk:
            #print snk.meta
            for f in src: snk.write(reproject(f, src.crs, snk_crs))

                          
def rasterize_stands(shp_path, tif_path, theme_cols, age_col, blk_col='', age_divisor=1., d=100.,
                     dtype=rasterio.int32, compress='lzw', round_coords=True,
                     value_func=lambda x: re.sub(r'(-| )+', '_', str(x).lower()), cap_age=None,
                     verbose=False):
    """
    Rasterize vector stand data.
    """
    import fiona
    from rasterio.features import rasterize
    if verbose: print('rasterizing', shp_path)
    if dtype == rasterio.int32: 
        nbytes = 4
    else:
        raise TypeError('Data type not implemented: %s' % dtype)
    hdt = {}
    shapes = [[], [], []]
    crs = None
    with fiona.open(shp_path, 'r') as src:
        crs = src.crs
        b = src.bounds #(x_min, y_min, x_max, y_max)
        w, h = b[2] - b[0], b[3] - b[1]
        m, n = int((h - (h%d) + d) / d), int((w - (w%d) + d) /  d)
        W = b[0] - (b[0]%d) if round_coords else b[0]
        N = b[1] - (b[1]%d) +d*m if round_coords else b[1] + d*m
        transform = rasterio.transform.from_origin(W, N, d, d)
        for i, f in enumerate(src):
            fp = f['properties']
            dt = tuple(value_func(fp[t]) for t in theme_cols)
            h = hash_dt(dt, dtype, nbytes)
            hdt[h] = dt
            try:
                age = np.int32(math.ceil(fp[age_col]/float(age_divisor)))
            except:
                #######################################
                # DEBUG
                # print(i, fp)                
                #######################################
                if fp[age_col] == None: 
                    age = np.int32(1)
                else:
                    raise ValueError('Bad age value in record %i: %s' % (i, str(fp[age_col])))
            if cap_age and age > cap_age: age = cap_age
            try:
                assert age > 0
            except:
                if fp[age_col] == 0:
                    age = np.int32(1)
                else:
                    print('bad age', age, fp[age_col], age_divisor)
                    raise
            blk = i if not blk_col else fp[blk_col]
            shapes[0].append((f['geometry'], h))   # themes
            shapes[1].append((f['geometry'], age)) # age
            shapes[2].append((f['geometry'], blk)) # block identifier
    #rst_path = shp_path[:-4]+'.tif' if not rst_path else rst_path
    nodata_value = -2147483648
    kwargs = {'out_shape':(m, n), 'transform':transform, 'dtype':dtype, 'fill':nodata_value}
    r = np.stack([rasterize(s, **kwargs) for s in shapes])
    kwargs = {'driver':'GTiff', 
              'width':n, 
              'height':m, 
              'count':3, 
              'crs':crs,
              'transform':transform,
              'dtype':dtype,
              'nodata':nodata_value,
              'compress':compress}
    #print(shp_path)
    #print(src.crs)
    #print(kwargs)
    with rasterio.open(tif_path, 'w', **kwargs) as snk:
        snk.write(r[0], indexes=1)
        snk.write(r[1], indexes=2)
        snk.write(r[2], indexes=3)
    return hdt
        

def hash_dt(dt, dtype=rasterio.int32, nbytes=4):
    s = '.'.join(map(str, dt)).encode('utf-8')
    d = hashlib.md5(s).digest() # first n bytes of md5 digest
    return np.dtype(dtype).type(int(binascii.hexlify(d[:4]), 16))


def warp_raster(src, dst_path, dst_crs={'init':'EPSG:4326'}):
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.enums import Resampling
    dst_t, dst_w, dst_h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
    profile = src.profile.copy()
    profile.update({'crs':dst_crs, 'transform':dst_t, 'width':dst_w, 'height':dst_h})
    with rasterio.open(dst_path, 'w', **profile) as dst:
        for i in range(1, src.count+1):
            reproject(source=rasterio.band(src, i),
                      destination=rasterio.band(dst, i),
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=dst_t,
                      dst_crs=dst_crs,
                      resampling=Resampling.nearest)


def timed(func):
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print('%s took %.3f seconds.' % (func.__name__, t))
        return result
    return wrapper
from scipy.stats import norm

HORIZON_DEFAULT = 30
PERIOD_LENGTH_DEFAULT = 10
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
    sc = (exp(C3b*log(vr)+C4b)-exp(C7d*log(vp)+C8d)+C15h*exp(C16h*P)-C17i*P+C18j)*P*Kmult+Kplus
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
            args = list(zip(P.rand(n), vr.rand(n), vp.rand(n)))
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


class Node:
    def __init__(self, nid, data=None, parent=None):
        self.nid = nid
        self._data = data
        self._parent = parent
        self._children = []

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return not self._children

    def add_child(self, child):
        self._children.append(child)

    def parent(self):
        return self._parent

    def children(self):
        return self._children
    
    def data(self, key=None):
        if key:
            return self._data[key]
        else:
            return self._data # if not self.is_root() else None

#from graphviz import Digraph        
class Tree:   
    def __init__(self, period=1):
        self._period = period
        self._nodes = [Node(0)]
        self._path = [self._nodes[0]]

    def children(self, nid):
        return [self._nodes[cid] for cid in self._nodes[nid].children()]
        
    def nodes(self):
        return self._nodes

    def node(self, nid):
        return self._nodes[nid]
    
    def add_node(self, data, parent=None):
        n = Node(len(self._nodes), data, parent)
        self._nodes.append(n)
        return n

    def grow(self, data):
        parent = self._path[-1]
        child = self.add_node(data, parent=parent.nid)
        parent.add_child(child.nid)
        self._path.append(child)
        return child
        
    def ungrow(self):
        self._path.pop()
        
    def leaves(self):
        return [n for n in self._nodes if n.is_leaf()]
    
    def root(self):
        return self._nodes[0]
    
    #def path(self):
    #    return self._path
    
    #def period(self):
    #    return len(self._path) - 1

    def path(self, leaf=None):
        if not leaf: return self._path[1:]
        path = []
        n = leaf
        while not (n.is_root()):
            path.append(n)
            parent = self.node(n.parent())
            n=parent
        path.reverse()
        return tuple(path)
    
    def paths(self):
        return [self.path(leaf) for leaf in self.leaves()]
    
    #def draw(self):
    #    graph = Digraph()
    #    for n in self.nodes():
    #        print n.data()
    #        graph.node(str(n.nid), n.data())
    #    for n in self.nodes():
    #        for c in self.children(n.nid):
    #            graph.edge(str(n.nid), str(c.nid))
    #    graph.graph_attr.update(size='10,10')
    #    return graph
