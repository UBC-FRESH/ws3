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

#import logging
import pandas as pd
#import geopandas as gpd
#import pysal as ps
#import random
#import ogr
import fiona
#import shapely
import numpy as np
import rasterio
#from osgeo import gdal


class ForestRaster:
    def __init__(self,
                 gdb_path,
                 gdb_cols,
                 geotiff_path,
                 theme_cols,
                 age_col,
                 area_col,
                 cid_col,
                 x_col,
                 y_col,
                 acodes,
                 horizon,
                 base_year,
                 index_col='index',
                 acode_col='acode',
                 tiff_compress='lzw',
                 area_coeff=1.):
        self._gdb_path = gdb_path
        self._gdb_cols = gdb_cols
        self._age_col = age_col
        self._area_col = area_col
        self._cid_col = cid_col
        self._x_col = x_col
        self._y_col = y_col
        self._acodes = acodes
        self._horizon = horizon
        self._base_year = base_year
        #self._profile = profile
        self._theme_cols = theme_cols
        self._index_col = index_col
        self._acode_col = acode_col
        self._read_gdb(area_coeff)
        if 'index' not in self._c.columns: self._c.reset_index(inplace=True)
        if 'acode' not in self._c.columns: self._c['acode'] = -1
        for tc in self._theme_cols:
            self._c[tc] = self._c[tc].astype(str).str.lower()
        self._c[self._index_col] = self._c[self._index_col].astype(int)
        self._C = self._c.copy().set_index(self._theme_cols + [self._age_col]).sort_index()
        self._theme_data = self._c.as_matrix(self._theme_cols)
        self._age_data = self._c.as_matrix([self._age_col])
        self._acode_data = self._c.as_matrix([self._acode_col])
        self._i2a = {i: a for i, a in enumerate(acodes)}
        self._a2i = {a: i for i, a in enumerate(acodes)}
        self._p = 1
        self._xmin, self._ymin = self._c[x_col].min(), self._c[y_col].min()
        self._pixel_size = pow(self._c[area_col].mean() * 10000., 0.5)
        self._tiffs = {p:{acode:rasterio.open(geotiff_path + '/%s_%i.tiff' % (acode, base_year + p - 1),
                                              'w',
                                              **self._build_profile())
                          for acode in acodes}
                       for p in range(1, horizon + 1)}
        self._tiff_data = {acode: self._read_tiff(acode) for acode in acodes}

    def _build_profile(self, tiff_compress='lzw', dtype=rasterio.uint8):
        xmin, xmax = self._c[self._x_col].min(), self._c[self._x_col].max()
        ymin, ymax = self._c[self._y_col].min(), self._c[self._y_col].max()
        h, w = int(1 + (xmax - xmin) / self._pixel_size), int(1 + (ymax - ymin) / self._pixel_size)
        return {'driver':'GTiff', 
                'height':h, 'width':w, 'count':1, 
                'dtype':dtype,
                'crs':self._gdb_meta['crs'], 'crk_wkt':self._gdb_meta['crs_wkt'],
                'compress':tiff_compress}

    def _read_gdb(self, area_coeff=1., gdb_path=None, gdb_cols=None, compress='lzw', verbose=False):
        gdb_path = gdb_path if gdb_path else self._gdb_path
        gdb_cols = gdb_cols if gdb_cols else self._gdb_cols
        with fiona.open(gdb_path, 'r') as src:
            self._gdb_meta = src.meta.copy()
            _d = {c: [] for c in gdb_cols}
            for f in src:
                try:
                    for c in gdb_cols: _d[c].append(f['properties'][c])
                except Exception, e:
                    print f['properties']
                    #logger.exception('Error processing feature %s:', [f['id']])
                    raise
        self._c = pd.DataFrame(_d).reset_index()
        self._c[self._area_col] = self._c[self._area_col] * area_coeff

        
    def __enter__(self):
        # The value returned by this method is
        # assigned to the variable after ``as``
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # returns either True or False
        # Don't raise any exceptions in this method
        print exc_type, exc_value, exc_traceback
        for p in self._tiffs:
            for acode in self._tiffs[p]:
                self._tiffs[p][acode].close()
        return True

    def _read_tiff(self, acode, verbose=False):
        if verbose: print 'ForestRaster._read_riff()', self._p, acode
        return self._tiffs[self._p][acode].read()

    def _write_tiff(self, acode, data):
        self._tiffs[self._p][acode].write(data)

    def allocate_schedule(self, wm, sch, verbose=False, show_progress=True):
        from ipywidgets import FloatProgress
        #fpmax = len([_ for _ in sch if _[4] <= self._horizon])
        fpmax = len([None for p in wm.applied_actions for a in wm.applied_actions[p]])
        if show_progress:
            fp = FloatProgress(min=0, max=fpmax, description='Allocating schedule')
            display(fp)
            fp.value = 0
        for p in range(1, self._horizon+1):
            if verbose: print 'processing schedule for period %i' % p
            for acode in wm.applied_actions[p]:
                if show_progress: fp.value += 1
                for dtk in wm.applied_actions[p][acode]:
                    for from_age in wm.applied_actions[p][acode][dtk]:
                        area = wm.applied_actions[p][acode][dtk][from_age][0]
                        if not area: continue
                        from_dtk = list(dtk)
                        tmask, tprop, tyield, tage, tlock, treplace, tappend = wm.dtypes[dtk].transitions[acode, from_age][0]
                        to_dtk = [t if tmask[i] == '?' else tmask[i] for i, t in enumerate(from_dtk)] 
                        if treplace: to_dtk[treplace[0]] = wm.resolve_replace(from_dtk, treplace[1])
                        to_dtk = tuple(to_dtk)
                        to_age = wm.resolve_targetage(to_dtk, tyield, from_age, tage, acode, verbose=False)
                        result = self.transition_cells_random(from_dtk, from_age, to_dtk, to_age, area, acode, verbose=False)
                        if result: print 'failed', from_dtk, from_age, to_dtk, to_age, area, acode 
                        #if show_progress: fp.value += 1
            self.commit_buffered_data(verbose=verbose)
            if p < self._horizon: self.grow()

    def transition_cells_random(self,
                                from_dtk,
                                from_age,
                                to_dtk,
                                to_age,
                                tarea,
                                acode,
                                verbose=False):
        key = tuple(from_dtk) + (from_age,)
        cand = self._C.loc[key]  # candidates
        
        # #print 'key', key
        # try:
        #     cand = self._C.loc[key]  # candidates
        # except Exception, e:
        #     #print e
        #     #print self._C.index.levels
        #     return 1 # fail
        #print len(cand)
        cand_area = cand[self._area_col].sum()
        frac = tarea / cand_area
        cand_samp = cand.sample(frac=frac)[self._index_col]
        #print 'tcr', self._p, acode, from_dtk, from_age, to_dtk, to_age, tarea, cand_area, frac
        #print 'tcr', cand_samp
        #print self._acode_data
        for ix in cand_samp:
            self._theme_data[ix] = to_dtk
            self._age_data[ix] = to_age
            self._acode_data[ix] = self._a2i[acode]
        return 0 # all is well
            
    def _resolve_indices(self, x, y):
        i = int((x - self._xmin) / self._pixel_size)
        j = int((y - self._ymin) / self._pixel_size)
        return i, j

    def commit_buffered_data(self, verbose=False):
        self._c[self._theme_cols] = self._theme_data
        self._c[self._age_col] = self._age_data
        self._c['acode'] = self._acode_data
        df = self._c.set_index(self._acode_col)
        #print self._p, self._tiff_data
        for acode in self._tiff_data:
            _df = None
            try:
                _df = df.loc[[self._a2i[acode]]]
            except:
                pass
            #if verbose: print acode, self._tiff_data[acode][0].sum()
            if _df is None or not len(_df):
                #print 'skipping', acode, self._p
                continue #acode not scheduled
            #if not self._tiff_data[acode][0].sum(): continue # acode not scheduled
            for _, x, y in _df[[self._x_col, self._y_col]].itertuples():
                i, j = self._resolve_indices(x, y)
                self._tiff_data[acode][0][i][j] = 1  # assume single-band TIFF
            #print 'foo'
            #if False:#verbose:
            #    print 'Committing buffered data', self._p, acode, len(_df), self._tiff_data[acode][0].sum()
            #    print _df
            #    print self._tiff_data[acode][0]
            self._write_tiff(acode, self._tiff_data[acode])

    def grow(self):
        self._p += 1
        self._tiff_data = {acode: self._read_tiff(acode) for acode in self._acodes}
        self._c[self._age_col] += 1
        self._c[self._acode_col] = -1
        #self._theme_data = self._c.as_matrix(self._theme_cols)
        self._age_data = self._c.as_matrix([self._age_col])
        self._acode_data = self._c.as_matrix([self._acode_col])
        self._C = self._c.copy().set_index(self._theme_cols + [self._age_col]).sort_index()
        
