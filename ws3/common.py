"""
This module contains definitions for global attributes, functions, and classes that 
might be used anywhere in the package.

.. py:data:: HORIZON_DEFAULT
   
   Default value for the length of the simulation horizon (number of periods).

.. py:data:: PERIOD_LENGTH_DEFAULT

   Default period length (number of years).

.. py:data:: MIN_AGE_DEFAULT
   
   Default value for :py:attr:`ws3.core.Curve.xmin`.

.. py:data:: MAX_AGE_DEFAULT
   
   Default value for :py:attr:`ws3.core.Curve.xmax`.

.. py:data:: CURVE_EPSILON_DEFAULT
   
   Defalut value for :py:attr:`ws3.core.Curve.epsilon`.

.. py:data:: AREA_EPSILON_DEFAULT = 0.01
    
"""

from scipy.stats import norm
import time
import scipy
import numpy as np
import rasterio
import hashlib
import re
import binascii

try:
    import pickle as pickle
except:
    import pickle

import math
import fiona
from fiona.transform import transform_geom
from fiona.crs import from_epsg

HORIZON_DEFAULT = 30
PERIOD_LENGTH_DEFAULT = 10
MIN_AGE_DEFAULT = 0
MAX_AGE_DEFAULT = 1000
CURVE_EPSILON_DEFAULT = 0.01
AREA_EPSILON_DEFAULT = 0.01


def hex_id(object, digest_size=4):
    """
    Converts an object to a hexadecimal string using a SHAKE-128 algorithm.
    Used in several places in the code base to generate unique identifiers for objects.

    :param object: The object to hash.
    :param digest_size: The size of the resulting hex string (in bytes). 
        The default value is 4, which means that the resulting hex string will be 8 characters long.

    :return: The hexadecimal hash string of the input object.

    """
    return hashlib.shake_128(pickle.dumps(object)).hexdigest(digest_size)
   
   
def is_num(s):
    """
    Checks if a given input is numerical value.

    :param s: The string to check for numericality.

    :return: ``True`` or ``False`` depending on whether the input was numeric or other.
        
    """
    try:
        float(s)
        return True
    except:
        return False


def reproject(f, srs_crs, dst_crs):
    """
    Reproject a geometry feature from a source coordinate reference system (CRS) 
    to a destination CRS.

    :param f: The geometry feature to reproject.
    :param srs_crs: The source CRS of the input geometry feature.
    :param dst_crs: The destination CRS for the output geometry feature.

    :return: A reprojected geometry feature in the destination CRS.

    """
    f['geometry'] = transform_geom(srs_crs, dst_crs, f['geometry'],
                          antimeridian_cutting=False,
                          precision=-1)
    return f


def clean_vector_data(src_path, dst_path, dst_name, prop_names, clean=True, tolerance=0.,
                      preserve_topology=True, logfn='clean_stand_shapefile.log', max_records=None,
                      theme0=None, prop_types=None, driver='ESRI Shapefile', dst_epsg=None,
                      update_area_prop=''):
    """
    The function cleans a vector data obtained form shapefile and reprojects to a destination shapefile.
    The output of the function is the path for cleaned shapefile and uncleaned shapefile.

    :param str src_path: Path to the source dataset.
    :param str dst_path: Path to the destination dataset.
    :param str dst_name: The name for the destination dataset.
    :param list prop_names: List of property names.
    :param bool clean: If the value of clean is True, the function will do cleaning; otherwise, it will do only reprojecting.
    :param float tolerance: This tolerance adjust the level of geometry modifications.
    :param bool preserve_topology: If the value of ``preserve_topology`` is ``True``, it will perserve the topology.
    :param str logfn: The filename for the log file to store the cleaned info.
    :param int max_records: If required, the user can define the maximum number of records for processing the source shapefile.
    :param str theme0: If required, the user can define theme0 for the cleaned shapefile.
    :param list prop_types: List of tuples showing the property types for the cleaned shapefile.
    :param str driver: The driver for writing the shapfiles.
    :param int dst_epsg: If the user specifies dst_epsg, the geometries will be reprojected to the specific CRS.
    :param str update_area_prop: The property that includes updated area information.

    :return: A tuple of two paths to the cleaned and uncleaned shapefiles.

    """
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
    """
    When a specific ESPG is defined, this function reprojects vector data from a source 
    shapefile to a destinaiton shapefile using ESRI shapefile as the default driver.

    :param str src_path: Path to the source shapefile.
    :param str snk_path: Path to the destination shapefile.
    :param int snk_epsg: EPSG code for the destination CRS.
    :param str driver: The driver for writing the shapfiles.  

    :return: None (output written directly to ``snk_path``).

    """
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
    The function rasterizes stands data and writes output to a geoTIFF file.

    :param str shp_path: Path to the source shapefile.
    :param str tif_path: Path to the resulted TIFF file.
    :param list theme_cols: List of theme column names.
    :param int age_col: Age column name.
    :param str blk_col: Block column name.
    :param float age_divisor: A number to scale stand age values.
    :param float d: The pixel size of the raster.
    :param rasterio.dtype dtype: The type of the output file (default type is :py:attr:`rasterio.int32`).
    :param str compress: The compression method (defaults to ``'lzw'``).
    :param bool round_coords: If true, the function rounds the coordinates of the ouput file.
    :param function value_func: A function that is applied to theme columns (in this case, the function replaces hyphens and spaces with underscores and changes all letters to lowercase)
    :param int cap_age: Maximum stand age defined by usder that will be considered as a cap age for stands (optional)
    :param bool verbose: (Optional) Verbosity flag.

    :return: Dictionary mapping hashed :py:attr:`ws3.forest.DevelopmentType.key` development
      type key values to the original development type key tuple value (i.e., the objects used 
      to generate the hash ID values).
      For some workflows (e.g., if calling spatio-temporal disaggregation functions in the 
      :py:mod:`ws3.spatial` module), this dictionary is used to map the hashed values back to the original values 
      (i.e., to "unhash" the hashed values).
    :rtype: dict
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
    nodata_value = -2147483648 # this really should be a function arg
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
        

def hash_dt(dt, dtype=rasterio.int32, nbytes=4):
    """
    The function hashes the development type and returns an integer value.

    :param str dt: Development type.
    :param rasterio.dtype dtype: The type of the output file (default type is rasterio.int32).
    :param int nbytes: The number of bytes to consider from the hash (The default value is 4).

    :return int: The integer value of the hash.
    :rtype: Data type specified in ``dtype`` argument (defaults to :py:class:`rasterio.int32`).

    """
    import struct
    s = '.'.join(map(str, dt)).encode('utf-8')
    d = hashlib.md5(s).digest() # first n bytes of md5 digest
    return np.dtype(dtype).type(struct.unpack('<i', d[:4])[0])


def warp_raster(src, dst_path, dst_crs={'init':'EPSG:4326'}):
    """
    The function warpes a raster from its original CRS to a new CRS.

    :param raserio.DatasetReader src: The source rasterio dataset to be warped.
    :param str dst_path: The path to save the warped raster
    :param dict dst_crs: The destination CRS in rasterio format 
      (default is :py:type:`dict` ``{'init':'EPSG:4326'}``).

    :return: None. The warped raster is saved to the path specified in ``dst_path``.

    """
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
    """
    The function records the execution time of a function.

    :param function func: The function to be timed.
    :return: The wrapped function.
    """
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print('%s took %.3f seconds.' % (func.__name__, t))
        return result
    return wrapper
