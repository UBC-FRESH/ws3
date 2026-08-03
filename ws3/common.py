"""
This module contains definitions for global attributes, functions, and classes
that might be used anywhere in the package.

Attributes:
    HORIZON_DEFAULT (int): Default value for ''.
    PERIOD_LENGTH_DEFAULT (int): Default number of years per period.
    MIN_AGE_DEFAULT (int): Default value for `core.Curve.xmin`.
    MAX_AGE_DEFAULT (int): Default value for `core.Curve.xmax`.
    CURVE_EPSILON_DEFAULT (float): Defalut value for `core.Curve.epsilon`.
    AREA_EPSILON_DEFAULT = 0.01

"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rasterio

try:
    import pickle as pickle
except ImportError:
    import pickle

import fiona
from fiona.crs import from_epsg
from fiona.transform import transform_geom

PACAL_BROKEN = True

#################################################################################################
# PaCal breaks when trying to import numpy.fft.fftpack (names have changed or some such... yuck).
# Note that this will breaks the folowing functions in this ws3.common
#   _sylv_credit_f1
#   _sylv_credit_f2
#   _sylv_credit_f3
#   _sylv_credit_f4
#   g
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

# from math import exp, log


def hex_id(obj: Any, digest_size: int = 10) -> str:
    """
    Convert an object to a hexadecimal string via SHA-1 hashing.

    :param obj: The object to hash.
    :param digest_size: Number of digest bytes to truncate to (unused).
    :return: Hexadecimal digest string.
    """
    # return binascii.hexlify(hashlib.sha1(pickle.dumps(obj)).digest(10))
    return hashlib.sha1(pickle.dumps(obj)).hexdigest()


def is_num(s: Any) -> bool:
    """
    Check whether the given input has a numerical value.

    :param s: Input value to test.
    :return: ``True`` if ``float(s)`` succeeds, ``False`` otherwise.
    """
    try:
        float(s)
        return True
    except Exception:
        return False

def reproject(
    f: dict[str, Any],
    srs_crs: dict[str, Any],
    dst_crs: dict[str, Any],
) -> dict[str, Any]:
    """
    Reproject a geometry from a source coordinate reference system (CRS) to a destination CRS.

    :param f: Feature dictionary with ``geometry`` and ``properties`` keys.
    :param srs_crs: Source CRS dictionary.
    :param dst_crs: Destination CRS dictionary.
    :return: The feature dictionary with the reprojected geometry.
    """
    f['geometry'] = transform_geom(
        srs_crs,
        dst_crs,
        f['geometry'],
        antimeridian_cutting=False,
        precision=-1,
    )
    return f

def clean_vector_data(
    src_path: str,
    dst_path: str,
    dst_name: str,
    prop_names: list[str],
    clean: bool = True,
    tolerance: float = 0.0,
    preserve_topology: bool = True,
    logfn: str = 'clean_stand_shapefile.log',
    max_records: int | None = None,
    theme0: str | None = None,
    prop_types: list[tuple[str, str]] | None = None,
    driver: str = 'ESRI Shapefile',
    dst_epsg: int | None = None,
    update_area_prop: str = '',
) -> tuple[str, str]:
    """
    Clean vector data obtained from a shapefile and reproject to a destination shapefile.

    :param src_path: Path to the source shapefile.
    :param dst_path: Path to the destination shapefile.
    :param dst_name: The name for the destination shapefile.
    :param prop_names: List of property names.
    :param clean: If True, performs cleaning; otherwise only reprojects.
    :param tolerance: Adjusts the level of geometry modifications.
    :param preserve_topology: If True, preserves topology.
    :param logfn: Filename for the log file to store cleaned info.
    :param max_records: Maximum number of records to process.
    :param theme0: Theme value for the cleaned shapefile.
    :param prop_types: List of tuples showing property types.
    :param driver: Driver for writing shapefiles.
    :param dst_epsg: EPSG code for the destination CRS.
    :param update_area_prop: Property that includes updated area information.
    :return: Tuple of paths (cleaned shapefile path, error shapefile path).
    """
    import logging
    from collections import OrderedDict

    from shapely.geometry import MultiPolygon, mapping, shape
    logging.basicConfig(filename=logfn, level=logging.INFO)
    snk1_path = f'{dst_path}/{dst_name}.shp'
    #snk2_path = dst_path[:-4]+'_error.shp'
    snk2_path = f'{dst_path}/{dst_name}_error.shp'
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
                    if dst_epsg:
                        f = reproject(f, src.crs, dst_crs)
                    if update_area_prop:
                        f['properties'][update_area_prop] = shape(f['geometry']).area
                    snk1.write(f)
                except Exception: # log exception and write uncleanable feature a separate shapefile
                    logging.exception("Error cleaning feature %s:", f['id'])
                    snk2.write(f)
    return snk1_path, snk2_path


def reproject_vector_data(
    src_path: str,
    snk_path: str,
    snk_epsg: int,
    driver: str = 'ESRI Shapefile',
) -> None:
    """
    Reproject vector data from a source shapefile to a destination shapefile using ESRI Shapefile as the default driver.

    :param src_path: Path to the source shapefile.
    :param snk_path: Path to the destination shapefile.
    :param snk_epsg: EPSG code for the destination CRS.
    :param driver: The driver for writing the shapefiles.
    """
    from fiona.crs import from_epsg
    with fiona.open(src_path, 'r') as src:
        snk_crs = from_epsg(snk_epsg)
        kwds = src.meta.copy()
        kwds.update(crs=snk_crs, crs_wkt=None)
        kwds.update(driver=driver)
        with fiona.open(snk_path, 'w', **kwds) as snk:
            #print snk.meta
            for f in src:
                snk.write(reproject(f, src.crs, snk_crs))


def rasterize_stands(
    shp_path: str,
    tif_path: str,
    theme_cols: list[str],
    age_col: str,
    blk_col: str = '',
    age_divisor: float = 1.0,
    d: float = 100.0,
    dtype: rasterio.dtype = rasterio.int32,
    compress: str = 'lzw',
    round_coords: bool = True,
    value_func: Callable[[Any], str] = lambda x: re.sub(r'(-| )+', '_', str(x).lower()),
    cap_age: int | None = None,
    verbose: bool = False,
) -> dict[int, tuple[str, ...]]:
    """
    Rasterize stand data and store the data as a TIFF file.

    :param shp_path: Path to the source shapefile.
    :param tif_path: Path to the resulting TIFF file.
    :param theme_cols: List of theme columns.
    :param age_col: Age column name.
    :param blk_col: Block identifier column name.
    :param age_divisor: A number to scale stand age values.
    :param d: The pixel size of the raster.
    :param dtype: The type of the output file (default is rasterio.int32).
    :param compress: The compression method (default is lzw).
    :param round_coords: If True, rounds the coordinates of the output file.
    :param value_func: A function applied to theme columns.
    :param cap_age: Maximum stand age (optional).
    :param verbose: Verbosity flag (defaults to False).
    :return: Dictionary mapping hash values to development type tuples.
    """
    from rasterio.features import rasterize
    if verbose:
        print('rasterizing', shp_path)
    if dtype == rasterio.int32:
        nbytes = 4
    else:
        raise TypeError(f'Data type not implemented: {dtype}')
    hdt: dict[int, tuple[str, ...]] = {}
    shapes: list[list[tuple[Any, Any]]] = [[], [], []]
    crs: dict[str, Any] | None = None
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
            except Exception:
                if fp[age_col] is None:
                    age = np.int32(1)
                else:
                    raise ValueError(f'Bad age value in record {i}: {str(fp[age_col])}') from None
            if cap_age and age > cap_age:
                age = np.int32(cap_age)
            try:
                assert age > 0
            except Exception:
                if fp[age_col] == 0:
                    age = np.int32(1)
                else:
                    print('bad age', age, fp[age_col], age_divisor)
                    raise
            blk = i if not blk_col else fp[blk_col]
            shapes[0].append((f['geometry'], h))   # themes
            shapes[1].append((f['geometry'], age)) # age
            shapes[2].append((f['geometry'], blk)) # block identifier
    nodata_value: int = -2147483648  # this really should be a function arg
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
    with rasterio.open(tif_path, 'w', **kwargs) as snk:
        snk.write(r[0], indexes=1)
        snk.write(r[1], indexes=2)
        snk.write(r[2], indexes=3)
    return hdt


def hash_dt(
    dt: tuple[Any, ...],
    dtype: rasterio.dtype = rasterio.int32,
    nbytes: int = 4,
) -> int:
    """
    Hash the development type and return an integer value.

    :param dt: Development type tuple.
    :param dtype: The type of the output file (default is rasterio.int32).
    :param nbytes: The number of bytes to consider from the hash (default is 4).
    :return: Integer hash value.
    """
    import struct
    s = '.'.join(map(str, dt)).encode('utf-8')
    d = hashlib.md5(s).digest()  # first n bytes of md5 digest
    # return np.dtype(dtype).type(int(binascii.hexlify(d[:4]), 16))
    return np.dtype(dtype).type(struct.unpack('<i', d[:4])[0])  # type: ignore[no-any-return]

def warp_raster(
    src: rasterio.DatasetReader,
    dst_path: str,
    dst_crs: dict[str, str] | None = None,
) -> None:
    """
    Warp a raster from its original CRS to a new CRS.

    :param src: The source rasterio dataset to be warped.
    :param dst_path: The path to save the warped raster.
    :param dst_crs: The destination CRS in rasterio format (default is EPSG:4326).
    """
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    if dst_crs is None:
        dst_crs = {'init': 'EPSG:4326'}
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


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Record the execution time of a function.

    :param func: The function to be timed.
    :return: Wrapped function that prints execution time.
    """
    def wrapper(*args: Any) -> Any:
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print(f'{func.__name__} took {t:.3f} seconds.')
        return result
    return wrapper


from scipy.stats import norm  # noqa: E402

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
    return f[formula](P, vr, vp)


def sylv_cred_rv(P_mu, P_sigma, tv_mu, tv_sigma, N_mu, N_sigma, psr,
                 treatment_type=None, cover_type=None, formula=None,
                 P_min=20., tv_min=50., N_min=200., ps_min=0.05,
                 E_fromintegral=False, e=0.01, n=1000):

    """
    This function returns sylviculture credit ($ per hectare).

    :param float P: Volume harvested per hectare.
    :param float vr: Mean piece size of harvested stems.
    :param float vp: mean piece size of stand before harvesting.
    :param formula: formula index (1 to 7).

    .. Note:: Assumes that variables (P, vr, vp) are random variates (returns expected value of function, using PaCAL packages to model random variates, assuming normal distribution for all three variables).
        Can use either PaCAL numerical integration (sssslow!), or custom numerical integration using Monte Carlo sampling (default).
    """
    if treatment_type and cover_type:
        formula = sylv_cred_formula(treatment_type, cover_type)  # type: ignore[no-untyped-call]
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
            args = list(zip(P.rand(n), vr.rand(n), vp.rand(n), strict=False))
            while len(args) > 0:  # process random args in in n-length chunks
                _E = E
                E = ((i - 1) * E + f[formula](*args.pop())) / i
                dE = abs((E - _E) / _E) if _E else np.inf
                i += 1
    return E  # type: ignore[return-value]


def sylv_cred_formula(treatment_type, cover_type):  # type: ignore[no-untyped-call]
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


def piece_size_ratio(treatment_type, cover_type, piece_size_ratios):
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


def harv_cost(piece_size: Any,
              is_finalcut: Any,
              is_toleranthw: Any,
              partialcut_extracare: bool = False,
              A: float = 1.97, B: float = 0.405, C: float = 0.169, D: float = 0.164,
              E: float = 0.202, F: float = 13.6, G: float = 8.83, K: float = 0.,
              rv: bool = False) -> Any:
    """
    Returns harvest cost.

    :param float piece_size: Piece size.
    :param bool is_finalcut: Treatment type (final cut or not).
    :param bool is_toleranthw: Stand type (tolerant hardwood or not).
    :param bool partialcut_extracare: Partialcut "extra care" flag.
    :param float A: Series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults that are extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB).
    :param bool rv: Types of variables (default: Variables are deterministic).
    """

    _ifc = float(is_finalcut)
    _ith = float(is_toleranthw)
    _pce = float(partialcut_extracare)
    log = pacal.log if rv else math.log
    exp = pacal.exp if rv else math.exp
    _exp = A - (B * log(piece_size)) + (C * _pce) + (D * _ifc) - (E * (1 - _ith))
    hc = exp(_exp) + ((F * _ith) + (G * (1 - _ith))) + K
    if rv:
        return hc.mean()  # type: ignore[union-attr]
    else:
        return hc


def harv_cost_rv(tv_mu, tv_sigma, N_mu, N_sigma, psr,
                 is_finalcut,
                 is_toleranthw,
                 partialcut_extracare=False,
                 tv_min=50., N_min=200., ps_min=0.05,
                 E_fromintegral=False, e=0.01, n=1000):

    """
    Returns harvest cost.

    :param bool is_finalcut: Treatment type (final cut or not).
    :param bool is_toleranthw: Stand type (tolerant hardwood or not).
    :param bool partialcut_extracare: Partialcut "extra care" flag.
    :param float A: Series of regression coefficients (A, B, C, D, E, F, G, K, all with defaults that are extracted from MERIS technical documentation; also see Sebastien Lacroix, BMMB).
    :param bool rv: Types of variables (default: Variables random variates).

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
            while len(args) > 0:  # process random args in in n-length chunks
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
    return sum(harv_cost(x, is_finalcut, is_toleranthw, **kwargs) * sigma * rv.pdf(x) for x in X)
