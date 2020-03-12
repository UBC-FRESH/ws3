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

#from numpy import argmax, array, logical_and
#from scipy.interpolate import interp1d
from bisect import bisect_left
from itertools import repeat
import copy
import math

from ws3 import common

"""
Used by ``Curve`` class to interpolate between real data points.
"""
class Interpolator(object):
    """
    Interpolates x and y values from sparse curve point list.
    """
    def __init__(self, points):
        #if any([_y - _x <= 0 for _x, _y in zip(x, x[1:])]):
        #    raise ValueError("x_list must be in strictly ascending order!")
        x, y = list(zip(*points))
        self.x = list(map(float, x))
        self.y = list(map(float, y))
        self.n = len(x)
        intervals = list(zip(self.x, self.x[1:], self.y, self.y[1:]))
        #print intervals
        try:
            self.m = [(y2 - y1)/(x2 - x1) for x1, x2, y1, y2 in intervals]
        except:
            print(intervals)
            raise
        
    def points(self):
        return list(zip(list(map(int, self.x)), self.y))
        
    def __call__(self, x):
        if x == 0: return self.y[0]
        i = bisect_left(self.x, x) - 1
        #print x, self.m[i]
        return self.y[i] + self.m[i] * (x - self.x[i])          
        # try:
        #     return self.y_list[i] + self.slopes[i] * (x - self.x_list[i])    
        # except IndexError:
        #     print 'i', i
        #     print 'x', x
        #     print 'y_list[i]', self.y_list[i]
        #     print 'slopes[i]', self.slopes[i]
        #     print 'x_list[i]', self.x_list[i]
        #     assert False

    def lookup(self, y, from_right=False):
        ##########################################################################
        # NOTE: This seemed to work fine at first, but breaks badly if y-values
        #       are not monotonically increasing from left to right...
        # if not from_right:
        #     i = bisect_left(self.y, y)
        #     if i == 0: return self.x[0]
        #     i -= 1
        #     if i == self.n - 1: return self.x[-1]
        #     try:
        #         return self.x[i] + (y - self.y[i])/self.m[i] if self.m[i] else self.x[i]
        #     except:
        #         print i, self.n, self.x, self.y
        #         raise
        ##########################################################################
        if not from_right:
            #_x = self.x[0]
            for i, x in enumerate(self.x):
                if self.y[i] > y: break
            i -= 1
            if i == self.n - 1: return self.x[-1]
            try:
                return self.x[i] + (y - self.y[i])/self.m[i] if self.m[i] else self.x[i]
            except:
                print(i, self.n, self.x, self.y)
                raise
            return x
        else:
            raise # not implemented yet...
        
            
class Curve:
    """
    Describes change in state over time (between treatments)
    """
    _type_default = 'a'
    
    def __init__(self,
                 label=None,
                 id=None,
                 is_volume=False,
                 points=None,
                 type=_type_default,
                 is_special=False,
                 period_length=common.PERIOD_LENGTH_DEFAULT,
                 xmin=common.MIN_AGE_DEFAULT,
                 xmax=common.MAX_AGE_DEFAULT,
                 epsilon=common.CURVE_EPSILON_DEFAULT,
                 simplify=True):
        self.label = label
        self.id = id
        self.is_volume = is_volume
        self.type = type
        self.period_length = period_length
        self.xmin = xmin
        self.xmax = xmax
        self.x = range(xmin, xmax+1)
        self.is_special = is_special
        self._y = None
        self.epsilon = epsilon
        self.is_locked = False
        self.add_points(points or [(0, 0)], simplify=simplify) # defaults to zero curve

    def simplify(self, points=None, autotune=True, compile_y=False, verbose=False):
        if self.is_special: return
        assert not self.is_locked
        points = self.points() if points is None else points
        n = len(points)
        ysum = sum(self)
        if n <= 2 or ysum < self.epsilon: return
        estep = self.epsilon
        error = 0.
        e = 0.
        sentinel = 0
        max_iters = 100
        while error < self.epsilon and sentinel < max_iters and len(self.points()) > 2:
            _points = copy.copy(self.points()) # backup
            self._simplify(e)
            if sentinel > 0 and len(self.points()) == len(_points): break
            error = abs(sum(self) - ysum) / ysum            
            if error >= self.epsilon: break
            e += estep
            sentinel += 1
        self.interp = Interpolator(_points) # restore from backup
        self._y = None
        if compile_y: self._compile_y()
        if verbose:
            error = abs(sum(self) - ysum) / ysum
            print('after final simplify', n, len(self.points()), float(n)/float(len(self.points())), error, ysum, sentinel) #, e, abs(sum(self) - ysum) / ysum

        
    def _simplify(self, e, compile_y=False):
        """
        NOTE: Implementation was modified so that point list is stored only once (in interp).
        """
        points = self.points()
        p = copy.copy(points)
        #print self.label, p
        n = 0
        for i in range(1, len(p) - 1):
            s1, s2 = [(p[i+j][1] - p[i+j-1][1]) / (p[i+j][0] - p[i+j-1][0]) for j in [0, 1]]
            if abs(s2 - s1) < e:
                n += 1
                points.remove(p[i]) # remove redundant point
        self.interp = Interpolator(points)
        self._y = None
        if compile_y: self._compile_y()
            
    def add_points(self, points, simplify=True, compile_y=False):
        assert not self.is_locked
        x, y = list(zip(*points)) # assume sorted ascending x
        x = list(x)
        y = [float(_y) for _y in y]
        # seems ok... (never tripped the assertion so far)
        # assert x[0] >= 0 and x[-1] <= self.xmax
        x_min = x[0]
        if x_min > 0:
            if x_min>1:
                x.insert(0, x_min-1)
                y.insert(0, 0.)
            x.insert(0, 0)
            y.insert(0, 0.)
        if x[-1] < self.xmax:
            x.append(self.xmax)
            y.append(y[-1])
        points = list(zip(x, y))
        self.interp = Interpolator(points)
        if simplify:
            self.simplify(points, compile_y)
        else:
            if compile_y: self._compile_y()

    def points(self):
        return self.interp.points()

    ##########################################################################################
    # NOTE: This is confirmed to work!... but has two (minor) problems:
    #       * evaluates more points than necessary
    #       * instantiate a new Curve with shitloads of points
    #         (although can be compressed with Curve.simplify(), it is more work to compress.
    #       * can return a double-range (would rather return a single window, searching L to R)
    #       
    # def range(self, lo=None, hi=None, verbose=False):
    #     y = [0. if ((lo is not None and y < lo) or (hi is not None and y > hi)) else 1. for y in self.y()]
    #     if verbose:
    #         for i, j in enumerate(self.y()):
    #             print i, j, lo, hi, y[i], (hi is not None), (j > hi)
    #     if verbose:
    #         print 'range', lo, hi
    #         print 'range', y
    #         print 'range', self._y
    #     return Curve(points=zip(self.x, y))

    def lookup(self, y, from_right=False, roundx=False):
        x = self.interp.lookup(y, from_right)
        if roundx:
            return int(round(x))
        else:
            return int(x)
    
    def range(self, lo=None, hi=None, as_bounds=False, left_range=True):
        """
        left_range True:  ub lookup from left (default)
        left_range False: ub lookup from right (widest possible range)
        """
        lb = int(round(self.interp.lookup(lo))) if lo is not None else 0
        ub = int(round(self.interp.lookup(hi, from_right=not left_range))) if hi is not None else self.xmax
        #print self.label, 'lo', lo, 'hi', hi, 'lb', lb, 'ub', ub
        #print self.points()
        points = [(lb, 1), (ub, 1)] if ub > lb else [(lb, 1)]
        if lb > 0:
            if lb > 1:
                points.insert(0, (lb-1, 0))
            points.insert(0, (0, 0))
        if ub < self.xmax:
            if ub < self.xmax - 1:
                points.append((ub+1, 0))      
            points.append((self.xmax, 0))
        if as_bounds: 
            return lb, ub
        else:
            #print points
            return Curve(points=points)
        
    def cai(self):
        x = self.x
        y = self.interp(x)
        return Curve(points=list(zip(x, (y[1:]-y[:-1])/self.period_length)))
            
    def mai(self):
        try:
            p = [(0, 0.)] + [(x, self[x]/(float(x)*self.period_length)) for x in range(1, self.xmax+1)]
        except:
            print(self.x) #[1:]
        return Curve(points=p)
            
    def ytp(self):
        y = self.y()
        argmax = y.index(max(y))
        return Curve(points=[(x, argmax-x) for x in self.x])

    def _compile_y(self):
        self._y = [self.interp(x) for x in self.x]
    
    def y(self, compile_y=False):
        if compile_y and not self._y:
            self._compile_y()
            return self._y
        else:
            return [self.interp(x) for x in self.x]
        
    def __iter__(self):
        for y in self.y(): yield y
           
    def __getitem__(self, x):
        return self._y[x] if self._y else self.interp(x)

    def __and__(self, other):
        y = [self[x] and other[x] for x in self.x] 
        return Curve(points=list(zip(self.x, y)))  
    
    def __or__(self, other):
        y = [self[x] or other[x] for x in self.x] 
        return Curve(points=list(zip(self.x, y)))  
    
    def __mul__(self, other):
        y = [_y*other for _y in self.y()] if isinstance(other, float) else [a*b for a,b in zip(self.y(), other.y())]
        return Curve(points=list(zip(self.x, y)))  
    
    def __div__(self, other):
        y = [a/b for a, b in zip(self.y(), [1. if not y else y for y in other.y()])]
        return Curve(points=list(zip(self.x, y)))
        
    def __add__(self, other):
        y = [_y+other for _y in self.y()] if isinstance(other, float) else [a+b for a,b in zip(self.y(), other.y())]
        return Curve(points=list(zip(self.x, y)))  

    def __sub__(self, other):
        y = [_y-other for _y in self.y()] if isinstance(other, float) else [a-b for a,b in zip(self.y(), other.y())]
        return Curve(points=list(zip(self.x, y)))
    
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    
    
if __name__ in '__main__':
    c1 = Curve('foo', points=[(1, 1.), (2, 2.), (3, 3.)])
    c2 = Curve('bar', points=[(1, 11.), (2, 22.), (3, 33.)])
    c3 = Curve('qux', points=[(1, 111.), (2, 222.), (3, 333.)])

    #c1 = Curve('foo', points=[(22, 2.), (33, 3.)])
    #c2 = Curve('bar', points=[(22, 22.), (33, 33.)])
    #c3 = Curve('qux', points=[(22, 222.), (33, 333.)])

    print('c1')
    for x in range(10): print(x, c1[x])
    print('c2')
    for x in range(10): print(x, c2[x])
    print('c3')
    for x in range(10): print(x, c3[x])

        
    print()
    print()
    print('test __mul__')
    print()
    c4 = c1 * c2
    for x in range(10): print(x, c4[x])
    print()
    c4 = c1 * 2.
    for x in range(10): print(x, c4[x])
    print()
    #c4 = 3. * c1
    #for x in range(10): print x, c4.y[x]
    print()
    print()
    print('test __div__')
    print()
    c4 = c1 / c2
    for x in range(10): print(x, c1[x], c2[x], c4[x])
          
    print()
    print()
    print('test __add__')
    c4 = c1 + c2
    for x in range(10): print(x, c4[x])
    print()
    c4 = c1 + 2.
    for x in range(10): print(x, c4[x])
    print()
    c4 = c1 + c2 + c3
    for x in range(10): print(x, c4[x])
        
    print()
    print('test __sub__')
    c4 = c1 - c2
    for x in range(10): print(x, c4[x])
    print()
    c4 = c1 - 2.
    for x in range(10): print(x, c4[x])

    #assert False


    c5 = Curve('baz', points=[(10, 20.), (50, 100.), (60, 101.), (80, 100.), (100, 50.)])

    c5_mai = c5.mai()
    c5_mai_ytp = c5_mai.ytp()
    c5_range1 = c5.range(22., 38.)
    c5_range2 = c5.range(36., 38.)
    #for x in range(10, 50, 1):
    #    #print x, c5_mai[x], c5_mai_ytp[x]
    #    print x,  c5[x], int(c5_range1[x]), int(c5_range2[x]),(c5_range1*c5_range2)[x]

    #for x, y in enumerate(c5): print x, y
        
