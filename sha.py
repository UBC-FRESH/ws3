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

import pandas as pd
#import fiona
import numpy as np
import rasterio
import os

class ForestRaster:
    def __init__(self,
                 hdt_map,
                 hdt_func,
                 src_path,
                 acodes,
                 horizon,
                 base_year,
                 tiff_compress='lzw',
                 tiff_dtype=rasterio.uint8):
        self._hdt_map = hdt_map
        self._hdt_func = hdt_func
        self._acodes = acodes
        self._horizon = horizon
        self._base_year = base_year
        self._i2a = {i: a for i, a in enumerate(acodes)}
        self._a2i = {a: i for i, a in enumerate(acodes)}
        self._p = 1 # initialize current period
        self._src = rasterio.open(src_path, 'r')
        self._x = self._src.read()
        self._d = self._src.transform.a # pixel width
        self._pixel_area = pow(self._d, 2) * 0.0001 # m to hectares 
        profile = self._src.profile
        profile.update(dtype=tiff_dtype, compress=tiff_compress, count=1, nodata=0)
        snk_path = os.path.split(src_path)[0]
        self._snk = {p:{acode:rasterio.open(snk_path + '/%s_%i.tiff' % (acode, base_year + p - 1), 'w', **profile)
                        for acode in acodes}
                      for p in range(1, horizon + 1)}
        self._snkd = {acode: self._read_snk(acode) for acode in acodes}
        self._is_valid = True
        
    def commit(self):
        for p in self._snk:
            for acode in self._snk[p]:
                self._snk[p][acode].close()
        self._is_valid = False
        
    def __enter__(self):
        # The value returned by this method is
        # assigned to the variable after ``as``
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # returns either True or False
        # Don't raise any exceptions in this method
        self.commit()
        return True
        
    def _read_snk(self, acode, verbose=False):
        if verbose: print 'ForestRaster._read_snk()', self._p, acode
        return self._snk[self._p][acode].read(1)

    def _write_snk(self):
        for acode in self._acodes:
            self._snk[self._p][acode].write(self._snkd[acode], indexes=1)
 
    def allocate_schedule(self, wm, verbose=False):
        if not self._is_valid: raise RuntimeError('ForestRaster.commit() has already been called (i.e., this instance is toast).')
        for p in range(1, self._horizon+1):
            if verbose: print 'processing schedule for period %i' % p
            for acode in wm.applied_actions[p]:
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
                        tk = tuple(to_dtk)+(to_age,)
                        th = self._hdt_func(tk)
                        result = self.transition_cells_random(from_dtk, from_age, to_dtk, to_age, area, acode, verbose=False)
                        if result: print 'failed', from_dtk, from_age, to_dtk, to_age, area, acode 
            self._write_snk()
            if p < self._horizon: self.grow()

    def transition_cells_random(self, from_dtk, from_age, to_dtk, to_age, tarea, acode, verbose=False):
        fk, tk = tuple(from_dtk), tuple(to_dtk)
        fh, th = self._hdt_func(fk), self._hdt_func(tk)
        x = np.where((self._x[0] == fh) & (self._x[1] == from_age))
        xn = len(x[0])
        xa = float(xn * self._pixel_area)
        c = tarea / xa
        if c > 1. and verbose: print 'missing area', from_dtk, tarea - xa
        c = min(c, 1.)
        n = int(xa * c / self._pixel_area)
        r = np.random.choice(xn, n, replace=False)
        ix = x[0][r],x[1][r]
        self._x[0][ix] = th
        self._x[1][ix] = to_age
        self._snkd[acode][ix] = 1 #self._a2i[acode]
        return 0 # all is well
          
    def grow(self):
        self._p += 1
        self._x[1] += 1 # age
        self._snkd = {acode: self._read_snk(acode) for acode in self._acodes}
