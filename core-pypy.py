from numpy import argmax, array, logical_and
from scipy.interpolate import interp1d


try:
    from . import common
    #from . import util
except: # "__main__" case
    import common
    #import util

class Forest:
    """
    Encapsulates the forest system state machine (system state, events, state transitions).
    """
    
    
    
    def __init__(self,
                 startyear,
                 horizon=common.HORIZON_DEFAULT,
                 period=common.PERIOD_DEFAULT,
                 description="",
                 species_groups=common.SPECIES_GROUPS_QC):
        self.startyear = startyear 
        self.horizon = horizon
        self.period = period
        self.description = description
        self._species_groups = species_groups
        self._strata = {}
        self._curves = {}
        self._treatments = {}
        #self._stands = {} # ?
        #self._zones = [] # ?
        #self._stratazones = [] # ?


class Treatment:
    """
    A state-transition-inducing event.
    """

    def __init__(self,
                 label,
                 id):
        self._label = label
        self._id = id


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
                 period=common.PERIOD_DEFAULT,
                 max_age=common.MAX_AGE_DEFAULT):
        self.label = label
        self.id = id
        self.is_volume = is_volume
        self.type = type
        self.period = period
        self.max_age = max_age
        self.x = [x for x in range(0, max_age+1)]
        #self._points = {}
        self.add_points(points)
        self.is_special = is_special
        
    def add_points(self, points):
        #self._points = dict(points)
        #for x, y in points: self._points[x] = y
    #    self._compile()
        
    #def _compile(self):
        #p = self._points
        x, y = zip(*points)
        x = list(x)
        y = list(y)
        #print points
        #print x
        #print y
        assert x[0] >= 0 and x[-1] <= self.max_age
        if x[0] > 0:
            x.insert(0, 0)
            y.insert(0, 0.)
        if x[-1] < self.max_age:
            x.append(self.max_age)
            y.append(y[-1])
        
        #if keys[0] < 0 or keys[-1] > self.max_age:
        #    for k in [k for k in p.keys() if k < 0 or k > self.max_age]: del p[k] # delete OOB data
        #if keys[0]: p[keys[0]-1] = p[0] = 0. # assume 0 before first point
        #p[self.max_age] = p[keys[-1]] # extend flat beyond last point
        #x, y = zip(*p.items())
        #self.interp = interp1d(p.keys(), p.values())
        self.interp = interp1d(x, y)
        
    def range(self, lo, hi):
        y = self.interp(self.x)
        #return Curve(points=zip(self.x, logical_and(lo<=y, y<=hi).astype(float)))
        return Curve(points=zip(self.x, logical_and(lo<=y, y<=hi)))
        
    def cai(self):
        x = self.x
        y = self.interp(x)
        return Curve(points=zip(x, (y[1:]-y[:-1])/self.period))
            
    def mai(self):
        p = [(0, 0.)] + [(x, self[x]/(float(x)*self.period)) for x in self.x[1:]]
        return Curve(points=p)
            
    def ytp(self):
        #argmax = lambda x, y: max(x, key=y)
        argmax = max(self.x, self)
        print argmax
        assert False
        return Curve(points=[(x, argmax(self)-x) for x in self.x])

    def __iter__(self):
        for y in self.interp(self.x): yield y
    
    def __getitem__(self, x):
        return float(self.interp(x))
    
    def __mul__(self, other):
        x = self.x
        y1 = self.interp(x)
        y2 = other if isinstance(other, float) else other.interp(x)
        return Curve(points=zip(x, y1*y2))
    
    def __div__(self, other):
        x = self.x
        y1 = self.interp(x)
        y2 = other if isinstance(other, float) else other.interp(x)
        y2[y2 == 0.] = 1. # numpy array magic (avoid division by 0)
        return Curve(points=zip(x, y1 / y2))

    def __add__(self, other):
        x = self.x
        y1 = self.interp(x)
        y2 = other if isinstance(other, float) else other.interp(x)
        return Curve(points=zip(x, y1 + y2))
    
    def __sub__(self, other):
        x = self.x
        y1 = self.interp(x)
        y2 = other if isinstance(other, float) else other.interp(x)
        return Curve(points=zip(x, y1 - y2))
    
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    
    
if __name__ in '__main__':
    c1 = Curve('foo', points=[(2, 1.), (2, 2.), (3, 3.)])
    c2 = Curve('bar', points=[(2, 11.), (2, 22.), (3, 33.)])
    c3 = Curve('qux', points=[(2, 111.), (2, 222.), (3, 333.)])
    
    print
    print
    print 'test __mul__'
    print
    c4 = c1 * c2
    for x in range(10): print x, c4[x]
    print
    c4 = c1 * 2.
    for x in range(10): print x, c4[x]
    print
    #c4 = 3. * c1
    #for x in range(10): print x, c4.y[x]

    print
    print
    print 'test __div__'
    print
    c4 = c1 / c2
    for x in range(10): print x, c1[x], c2[x], c4[x]

    print
    print
    print 'test __add__'
    c4 = c1 + c2
    for x in range(10): print x, c4[x]
    print
    c4 = c1 + 2.
    for x in range(10): print x, c4[x]
    print
    c4 = c1 + c2 + c3
    for x in range(10): print x, c4[x]
            
    print
    print 'test __sub__'
    c4 = c1 - c2
    for x in range(10): print x, c4[x]
    print
    c4 = c1 - 2.
    for x in range(10): print x, c4[x]

    c5 = Curve('baz', points=[(10, 20.), (50, 100.), (60, 101.), (80, 100.), (100, 50.)])

    c5_mai = c5.mai()
    c5_mai_ytp = c5_mai.ytp()
    c5_range1 = c5.range(22., 38.)
    c5_range2 = c5.range(36., 38.)
    #for x in range(10, 50, 1):
    #    #print x, c5_mai[x], c5_mai_ytp[x]
    #    print x,  c5[x], int(c5_range1[x]), int(c5_range2[x]),(c5_range1*c5_range2)[x]

    #for x, y in enumerate(c5): print x, y
        
