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

HORIZON_DEFAULT = 30
PERIOD_LENGTH_DEFAULT = 10
MIN_AGE_DEFAULT = 0
MAX_AGE_DEFAULT = 1000
CURVE_EPSILON_DEFAULT = 0.01
AREA_EPSILON_DEFAULT = 0.01


PACAL_BROKEN = True

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
#from math import exp, log
import fiona
from fiona.transform import transform_geom
from fiona.crs import from_epsg

def hex_id(object, digest_size=10):
    """
    This function converts an object to a hexadecimal string.
    
    """
    #return binascii.hexlify(hashlib.sha1(pickle.dumps(object)).digest(10))
    return hashlib.sha1(pickle.dumps(object)).hexdigest()
   
def is_num(s):
    """
    This function checks if a given input has a numerical value.
        
    """
    try:
        float(s)
        return True
    except:
        return False

def reproject(f, srs_crs, dst_crs):
    """
    Reproject a geometry from a source coordinate reference system (CRS) to a destination CRS.
        
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

    :param str src_path: Path to the source shapefile.
    :param str dst_path: Path to the destination shapefile.
    :param str dst_name: The name for the destination shapefile.
    :param list prop_names: List of property names.
    :param bool clean: If the value of clean is True, the function will do cleaning; otherwise, it will do only reprojecting.
    :param float tolerance: This tolerance adjust the level of geometry modifications.
    :param bool preserve_topology: If the value of preserve_topology is True, it will perserve the topology.
    :param str logfn: The filename for the log file to store the cleaned info.
    :param int max_records: If required, the user can define the maximum number of records for processing the source shapefile.
    :param str theme0: If required, the user can define theme0 for the cleaned shapefile.
    :param list prop_types: List of tuples showing the property types for the cleaned shapefile.
    :param str driver: The driver for writing the shapfiles.
    :param int dst_epsg: If the user specifies dst_epsg, the geometries will be reprojected to the specific CRS.
    :param str update_area_prop: The property that includes updated area information.

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
    When a specific ESPG is defined, this function reprojects vector data from a source shapefile to a destinaiton shapefile using ESRI shapefile as the default driver.

    :param str src_path: Path to the source shapefile.
    :param str snk_path: Path to the destination shapefile.
    :param int snk_epsg: EPSG code for the destination CRS.
    :param str driver: The driver for writing the shapfiles.        
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

  
# def rasterize_stands(shp_path, tif_path, theme_cols, age_col, blk_col='', age_divisor=1., d=100.,
#                      dtype=rasterio.int32, compress='lzw', round_coords=True,
#                      value_func=lambda x: re.sub(r'(-| )+', '_', str(x).lower()), cap_age=None,
#                      verbose=False):
#     """
#     The function rasterizes stands data and stores the data as TIFF file.

#     :param str shp_path: Path to the source shapefile.
#     :param str tif_path: Path to the resulted TIFF file.
#     :param list theme_cols: List of themes.
#     :param int age_col: Age column.
#     :param str blk_col: 
#     :param float age_divisor: A number to scale stand age values.
#     :param float d: The pixel size of the raster.
#     :param rasterio.dtype dtype: The type of the output file (default type is rasterio.int32).
#     :param str compress: The compression method (The default one is lzw)
#     :param bool round_coords: If ture, the function rounds the coordinates of the ouput file.
#     :param function value_func: A function that is applied to theme columns (in this case, the function replaces hyphens and spaces with underscores and changes all letters to lowercase)
#     :param int cap_age: Maximum stand age defined by usder that will be considered as a cap age for stands (optional)
#     :param bool verbose: (Optional) Verbosity flag. Defaults to False
#     """
#     import fiona
#     from rasterio.features import rasterize
#     if verbose: print('rasterizing', shp_path)
#     if dtype == rasterio.int32: 
#         nbytes = 4
#     else:
#         raise TypeError('Data type not implemented: %s' % dtype)
#     hdt = {}
#     shapes = [[], [], []]
#     crs = None
#     with fiona.open(shp_path, 'r') as src:
#         crs = src.crs
#         b = src.bounds #(x_min, y_min, x_max, y_max)
#         w, h = b[2] - b[0], b[3] - b[1]
#         m, n = int((h - (h%d) + d) / d), int((w - (w%d) + d) /  d)
#         W = b[0] - (b[0]%d) if round_coords else b[0]
#         N = b[1] - (b[1]%d) +d*m if round_coords else b[1] + d*m
#         transform = rasterio.transform.from_origin(W, N, d, d)
#         for i, f in enumerate(src):
#             fp = f['properties']
#             dt = tuple(value_func(fp[t]) for t in theme_cols)
#             h = hash_dt(dt, dtype, nbytes)
#             hdt[h] = dt
#             try:
#                 age = np.int32(math.ceil(fp[age_col]/float(age_divisor)))
#             except:
#                 #######################################
#                 # DEBUG
#                 # print(i, fp)                
#                 #######################################
#                 if fp[age_col] == None: 
#                     age = np.int32(1)
#                 else:
#                     raise ValueError('Bad age value in record %i: %s' % (i, str(fp[age_col])))
#             if cap_age and age > cap_age: age = cap_age
#             try:
#                 assert age > 0
#             except:
#                 if fp[age_col] == 0:
#                     age = np.int32(1)
#                 else:
#                     print('bad age', age, fp[age_col], age_divisor)
#                     raise
#             blk = i if not blk_col else fp[blk_col]
#             shapes[0].append((f['geometry'], h))   # themes
#             shapes[1].append((f['geometry'], age)) # age
#             shapes[2].append((f['geometry'], blk)) # block identifier
#     #rst_path = shp_path[:-4]+'.tif' if not rst_path else rst_path
#     nodata_value = -2147483648
#     kwargs = {'out_shape':(m, n), 'transform':transform, 'dtype':dtype, 'fill':nodata_value}
#     r = np.stack([rasterize(s, **kwargs) for s in shapes])
#     kwargs = {'driver':'GTiff', 
#               'width':n, 
#               'height':m, 
#               'count':3, 
#               'crs':crs,
#               'transform':transform,
#               'dtype':dtype,
#               'nodata':nodata_value,
#               'compress':compress}
#     #print(shp_path)
#     #print(src.crs)
#     #print(kwargs)
#     with rasterio.open(tif_path, 'w', **kwargs) as snk:
#         snk.write(r[0], indexes=1)
#         snk.write(r[1], indexes=2)
#         snk.write(r[2], indexes=3)
#     return hdt
        
def rasterize_stands(shp_path, tif_path, theme_cols, age_col, blk_col='', age_divisor=1., d=100.,
                     dtype=rasterio.int32, compress='lzw', round_coords=True,
                     value_func=lambda x: re.sub(r'(-| )+', '_', str(x).lower()), cap_age=None,
                     verbose=False, extra_bands=0):  # Added extra_bands parameter
    """
    The function rasterizes stands data and stores the data as TIFF file.

    :param str shp_path: Path to the source shapefile.
    :param str tif_path: Path to the resulted TIFF file.
    :param list theme_cols: List of themes.
    :param int age_col: Age column.
    :param str blk_col: 
    :param float age_divisor: A number to scale stand age values.
    :param float d: The pixel size of the raster.
    :param rasterio.dtype dtype: The type of the output file (default type is rasterio.int32).
    :param str compress: The compression method (The default one is lzw)
    :param bool round_coords: If ture, the function rounds the coordinates of the ouput file.
    :param function value_func: A function that is applied to theme columns (in this case, the function replaces hyphens and spaces with underscores and changes all letters to lowercase)
    :param int cap_age: Maximum stand age defined by usder that will be considered as a cap age for stands (optional)
    :param bool verbose: (Optional) Verbosity flag. Defaults to False
    :param int extra_bands: Number of extra bands to add to the output file with dtype=numpy.uint32. Defaults to 0.
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
    # Modify kwargs for additional bands
    kwargs = {'driver':'GTiff', 
              'width':n, 
              'height':m, 
              'count':3 + extra_bands, # Adding extra bands
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
        # Write extra bands with zero-initialized values
        for i in range(3, 3 + extra_bands):
            snk.write(np.zeros((m, n), dtype=np.uint32), indexes=i + 1)
    return hdt

def hash_dt(dt, dtype=rasterio.int32, nbytes=4):
    """
    The function hashes the development type and returns an integer value.

    :param str dt: Development type.
    :param rasterio.dtype dtype: The type of the output file (default type is rasterio.int32).
    :param int nbytes: The number of bytes to consider from the hash (The default value is 4).

    """
    import struct
    s = '.'.join(map(str, dt)).encode('utf-8')
    d = hashlib.md5(s).digest() # first n bytes of md5 digest
    #return np.dtype(dtype).type(int(binascii.hexlify(d[:4]), 16))
    return np.dtype(dtype).type(struct.unpack('<i', d[:4])[0])

def warp_raster(src, dst_path, dst_crs={'init':'EPSG:4326'}):
    """
    The function warpes a raster from its original CRS to a new CRS.

    :param raserio.DatasetReader src: The source rasterio dataset to be warped.
    :param str dst_path: The path to save the warped raster
    :param dict dst_crs: The destination CRS in rasterio format (Default is init':'EPSG:4326')
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
    """
    def wrapper(*args):
        t = time.time()
        result = func(*args)
        t = time.time() - t
        print('%s took %.3f seconds.' % (func.__name__, t))
        return result
    return wrapper
from scipy.stats import norm


def is_num(s):
    """
    This function checks if a given input has a numerical value.
        
    """
    try:
        float(s)
        return True
    except:
        return False

    

 