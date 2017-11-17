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
from profilehooks import profile


class ForestRaster:
    def __init__(self,
                 hdt_map,
                 hdt_func,
                 src_path,
                 snk_path,
                 acodes,
                 horizon,
                 base_year,
                 period_length=1,
                 tif_compress='lzw',
                 tif_dtype=rasterio.uint8,
                 piggyback_acodes=[]):
        self._hdt_map = hdt_map
        self._hdt_func = hdt_func
        self._acodes = acodes
        self._horizon = horizon
        self._base_year = base_year
        self._period_length = period_length
        self._i2a = {i: a for i, a in enumerate(acodes)}
        self._a2i = {a: i for i, a in enumerate(acodes)}
        self._p = 1 # initialize current period
        self._src = rasterio.open(src_path, 'r')
        self._x = self._src.read()
        self._d = self._src.transform.a # pixel width
        self._pixel_area = pow(self._d, 2) * 0.0001 # m to hectares
        profile = self._src.profile
        profile.update(dtype=tif_dtype, compress=tif_compress, count=1, nodata=0)
        self._piggyback_acodes = piggyback_acodes
        for acode1 in piggyback_acodes:
            for acode2, _ in piggyback_acodes[acode1]:
                self._acodes.append(acode2)
        self._snk = {(p, dy):{acode:rasterio.open(snk_path+'/%s_%i.tif' % (acode, base_year+(p-1)*period_length + dy), 'w', **profile)
                      for acode in self._acodes}
                      for dy in range(period_length) for p in range(1, (horizon+1))}
        self._snkd = {(acode, dy):self._read_snk(acode, dy) for dy in range(period_length) for acode in self._acodes}
        self._is_valid = True
        
    def commit(self):
        for p in self._snk:
            for acode in self._snk[p]:
                self._snk[p][acode].close()
        self._is_valid = False

    def cleanup(self):
        self.commit()
        self._src.close()
        
    def __enter__(self):
        # The value returned by this method is
        # assigned to the variable after ``as``
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # returns either True or False
        # Don't raise any exceptions in this method
        self.cleanup()
        return True
        
    def _read_snk(self, acode, dy, verbose=False):
        if verbose: print 'ForestRaster._read_snk()', self._p, acode
        return self._snk[(self._p, dy)][acode].read(1)

    def _write_snk(self):
        for dy in range(self._period_length):
            for acode in self._acodes:
                snk = self._snk[(self._p, dy)][acode]
                snk.write(self._snkd[(acode, dy)], indexes=1)
                snk.close()

    #@profile(immediate=True)
    def allocate_schedule(self, forestmodel, da=0, fudge=1., mask=None, verbose=False):
        if not self._is_valid: raise RuntimeError('ForestRaster.commit() has already been called (i.e., this instance is toast).')
        if mask: dtype_keys = forestmodel.unmask(mask)
        for p in range(1, self._horizon+1):
            if verbose: print 'processing schedule for period %i' % p
            for acode in forestmodel.applied_actions[p]:
                for dtk in forestmodel.applied_actions[p][acode]:
                    if mask:
                        if dtk not in dtype_keys: continue
                    for from_age in forestmodel.applied_actions[p][acode][dtk]:
                        area = forestmodel.applied_actions[p][acode][dtk][from_age][0]
                        if not area: continue
                        from_dtk = list(dtk)
                        tmask, tprop, tyield, tage, tlock, treplace, tappend = forestmodel.dtypes[dtk].transitions[acode, from_age][0]
                        to_dtk = [t if tmask[i] == '?' else tmask[i] for i, t in enumerate(from_dtk)] 
                        if treplace: to_dtk[treplace[0]] = forestmodel.resolve_replace(from_dtk, treplace[1])
                        to_dtk = tuple(to_dtk)
                        to_age = forestmodel.resolve_targetage(to_dtk, tyield, from_age, tage, acode, verbose=False)
                        tk = tuple(to_dtk)+(to_age,)
                        th = self._hdt_func(tk)
                        for dy in range(self._period_length):
                            from_ages = [from_age]
                            target_area = area / self._period_length
                            while from_ages and target_area:
                                from_age = from_ages.pop()
                                target_area = self.transition_cells_random(from_dtk, from_age, to_dtk, to_age,
                                                                           target_area, acode, dy,
                                                                           da=da, fudge=fudge, verbose=False)
                            if target_area:
                                print 'failed', (from_dtk, from_age, to_dtk, to_age, acode),
                                print '(missing %4.1f of %4.1f)' % (target_area, area / self._period_length), 'in p%i dy%i' % (p, dy)
                if acode in self._piggyback_acodes:
                    for _acode, _p in self._piggyback_acodes[acode]:
                        for dy in range(self._period_length):
                            x = np.where(self._snkd[(acode, dy)] == 1)
                            xn = len(x[0])
                            r = np.random.choice(xn, int(_p * xn), replace=False)
                            ix = x[0][r], x[1][r]
                            self._snkd[(_acode, dy)][ix] = 1
                            #print acode, _acode, _p, p, dy, len(x), int(_p * len(x))
            self._write_snk()
            if p < self._horizon: self.grow()

    def transition_cells_random(self, from_dtk, from_age, to_dtk, to_age, tarea, acode, dy, da=0, fudge=1., verbose=False):
        fk, tk = tuple(from_dtk), tuple(to_dtk)
        fh, th = self._hdt_func(fk), self._hdt_func(tk)
        x = np.where((self._x[0] == fh) & (self._x[1]+da == from_age))
        xn = len(x[0])
        xa = float(xn * self._pixel_area)
        missing_area = max(0., tarea - xa)
        c = tarea / xa if xa else np.inf
        if c > 1. and verbose: print 'missing area', from_dtk, tarea - xa
        c = min(c, 1.)
        n = int(xa * c / self._pixel_area)
        if not n: return # found nothing to transition
        r = np.random.choice(xn, n, replace=False)
        ix = x[0][r], x[1][r]
        self._x[0][ix] = th
        self._x[1][ix] = to_age
        self._snkd[(acode, dy)][ix] = 1 #self._a2i[acode]
        return missing_area
          
    def grow(self):
        self._p += 1
        self._x[1] += 1 # age
        self._snkd = {(acode, dy):self._read_snk(acode, dy) for dy in range(self._period_length) for acode in self._acodes}
