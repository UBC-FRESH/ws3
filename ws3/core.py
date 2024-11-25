"""
This module defines some core classes used elsewhere in the package.
These include classes to represent yield curves and dynamcic programming
state trees.
"""


from bisect import bisect_left
from itertools import repeat
import copy
import math

from ws3 import common

class Interpolator(object):
    """
    Interpolates x and y values from sparse curve point list.

    Used by the :py:class:`ws3.core.Curve` class to interpolate between real data points.

    """
    def __init__(self, points):
        """
        :param list points: A list of (x,y) coordinate pairs.
        """
        x, y = list(zip(*points))
        self.x = list(map(float, x))
        self.y = list(map(float, y))
        self.n = len(x)
        intervals = list(zip(self.x, self.x[1:], self.y, self.y[1:]))
        try:
            self.m = [(y2 - y1)/(x2 - x1) for x1, x2, y1, y2 in intervals]
        except:
            print(intervals)
            raise
        
    def points(self):
        """
        Returns the points as a list of tuples representing the points.

        :return list: A list of (x, y) coordinate pairs.
        """
        return list(zip(list(map(int, self.x)), self.y))
        
    def __call__(self, x):
        """
        Interpolates the value of y at a given x.

        :param x: The x coordinate to interpolate.
        :return float: The y value at the given x.
        """
        if x == 0: return self.y[0]
        i = bisect_left(self.x, x) - 1
        return self.y[i] + self.m[i] * (x - self.x[i])          

    def lookup(self, y, from_right=False):
        """
        Looks up the x-coordinate corresponding to the given y-coordinate.

        :param float y: The y-coordinate to look up.
        :param bool from_right: Flag indicating whether to search from the right. Defaults to `False`.
        :return int: The x-coordinate corresponding to the given y-coordinate.
        """
        if not from_right:
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
    Describes change in state over time (between treatments).
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
        """
        :param str label: A label for the curve.
        :param str id: An ID for the curve.
        :param bool is_volume: Flag indicating whether the curve tracks volume. Defaults to ``False``.
        :param list points: A list of (x,y) pairs defining the curve.
        :param str type: A string indicating the type of curve. Defaults to ``'a'``.
        :param bool is_special: Flag indicating whether the curve is special. Defaults to ``False``.
            Special curves are immune to simplification.
        :param float period_length: The length of the period. Defaults to :py:attr:`ws3.common.PERIOD_LENGTH_DEFAULT`.
        :param float xmin: The minimum age. Defaults to :py:attr:`ws3.common.MIN_AGE_DEFAULT`.
        :param float xmax: The maximum age. Defaults to :py:attr:`ws3.common.MAX_AGE_DEFAULT`.
        :param float epsilon: The tolerance for simplifying the curve. Defaults to :py:attr:`ws3.common.CURVE_EPSILON_DEFAULT`.
        :param bool simplify: Flag indicating whether to simplify the curve. Defaults to ``True``.
        """
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
        """
        Simplifies the curve by removing redundant points.

        :param list of tuples points: The points to simplify. Defaults to None.
        :param bool autotune: Flag indicating whether to automatically tune the simplification process. Defaults to True.
        :param bool compile_y: Flag indicating whether to compile the y-component. Defaults to False.
        :param bool verbose: Flag indicating whether to print verbose output. Defaults to False.
        """
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
        Simplify the curve using a linear interpolation. Internal method, called from ``self.simplify()``.
        .. note:: 
           Implementation was modified so that point list is stored only once (in interp).
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
        """
        Adds points to the curve and optionally simplifies point geometry.
    
        :param list of tuples points: The points to add to the curve.
        :param bool simplify: Flag indicating whether to simplify the curve after adding points. Defaults to ``True``.
        :param bool compile_y: Flag indicating whether to compile the y-component after adding points. Defaults to ``False``.
        """
        assert not self.is_locked
        x, y = list(zip(*points)) # assume sorted ascending x
        x = list(x)
        y = [float(_y) for _y in y]
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
        """
        :return list: list of curve points
        """
        return self.interp.points()

    def lookup(self, y, from_right=False, roundx=False):
        """
        Looks up the x-coordinate corresponding to the given y-coordinate.
    
        :param float y: The y-coordinate to look up.
        :param bool from_right: Flag indicating whether to search from the right. Defaults to ``False``.
        :param bool roundx: Flag indicating whether to round the x-coordinate to the nearest integer. Defaults to ``False``.
        :return float: The x-coordinate corresponding to the given y-coordinate.
        """
        x = self.interp.lookup(y, from_right)
        if roundx:
            return int(round(x))
        else:
            return int(x)
    
    def range(self, lo=None, hi=None, as_bounds=False, left_range=True):
        """
        Returns a Curve representing the range within the specified bounds.

        :param float lo: The lower bound of the range. Defaults to None.
        :param float hi: The upper bound of the range. Defaults to None.
        :param bool as_bounds: Flag indicating whether to return the range as a 
            tuple of bounds. Defaults to ``False``.
        :param bool left_range: Flag indicating whether to look up the upper bound 
            from the left (default) or from the right (widest possible range).
        :return: Returns either curve representing 
          the range within the specified bounds, or a tuple representing lower- and upper-bound 
          values (if ``as_bounds`` set to ``True``).
        :rtype: :py:class:`ws3.core.Curve` or tuple
        """
        lb = int(round(self.interp.lookup(lo))) if lo is not None else 0
        ub = int(round(self.interp.lookup(hi, from_right=not left_range))) if hi is not None else self.xmax
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
            return Curve(points=points)
        
    def cai(self):
        """
        Calculates a current annual increment (CAI) curve.

        :return: A curve representing the current annual increment.
        :rtype: :py:class:`ws3.core.Curve`
        """
        X = list(range(1, self.xmax))
        Y = [self[x] - self[x-1] for x in X]
        points = list(zip(X, Y))
        return Curve(points=points)
            
    def mai(self):
        """
        Calculates a mean annual increment (MAI) curve.

        :return: A curve representing the mean annual increment.
        :rtype: :py:class:`ws3.core.Curve`
        """
        X = range(1, self.xmax)
        Y = [self[x] / x for x in X[1:]]
        points = list(zip(X, Y)) 
        return Curve(points=points)

    def ytp(self):
        """
        Returns a years-to-peak (YTP) curve. This curve is a measure of how many years 
        it takes for the curve to reach its peak (positive values to the left of the peak,
        and negative values to the right of the peak).
        :return: A curve representing the years to peak.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = self.y()
        argmax = y.index(max(y))
        return Curve(points=[(x, argmax-x) for x in self.x])

    def _compile_y(self):
        """
        Compiles the y values from the x values stored in ``self.x``, 
        and stores them in ``self._y``.
        """
        self._y = [self.interp(x) for x in self.x]
    
    def y(self, compile_y=False):
        """
        Returns the y-values of the curve stored in ``self._y`` (will first compile them if ``compile_y`` is set 
        to ``True`` and ``self._y`` is empty), else will interpolate a list of y values 
        for each x value in ``self.x``.

        :param bool compile_y: Flag indicating whether to compile the y-component of the curve. Defaults to ``False``.
        :return list: A list of y values.
        """
        if compile_y and not self._y:
            self._compile_y()
            return self._y
        else:
            return [self.interp(x) for x in self.x]
        
    def __iter__(self):
        """
        Returns an iterator that iterates through the y values of this curve.
        """
        for y in self.y(): yield y
           
    def __getitem__(self, x):
        """
        Returns the y value of this curve at a given x-value ``x``.
        """
        return self._y[x] if self._y else self.interp(x)

    def __and__(self, other):
        """
        Returns a new curve that is the intersection of this curve with another curve ``other``.
        :param :py:class:`ws3.core.Curve` other: The curve to intersect with this curve.
        :return: A new curve that is the intersection of this curve with another curve ``other``.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [self[x] and other[x] for x in self.x]
        points = list(zip(self.x, y)) 
        return Curve(points=points)  
    
    def __or__(self, other):
        """
        Returns a new curve that is the union of this curve with another curve ``other``.
        :param :py:class:`ws3.core.Curve` other: The curve to union with this curve.
        :return: A new curve that is the union of this curve with another curve ``other``.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [self[x] or other[x] for x in self.x]
        point = list(zip(self.x, y)) 
        return Curve(points=points)  
    
    def __mul__(self, other):
        """
        Returns a new curve that is the product of this curve with another curve ``other`` or a constant value.
        :param :py:class:`ws3.core.Curve` other: The curve to multiply with this curve or the constant value ``other``.
        :return: A new curve that is the product of this curve with another curve ``other`` or a constant value.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [_y*other for _y in self.y()] if isinstance(other, float) else [a*b for a,b in zip(self.y(), other.y())]
        points = list(zip(self.x, y))
        return Curve(points=points)  
    
    def __div__(self, other):
        """
        Returns a new curve that is the quotient of this curve with another curve ``other`` or a constant value.
        :param  :py:class:`ws3.core.Curve` other: The curve to divide with this curve or the constant value ``other``.
        :return: A new curve that is the quotient of this curve with another curve ``other`` or a constant value.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [a/b for a, b in zip(self.y(), [1. if not y else y for y in other.y()])]
        points = list(zip(self.x, y))
        return Curve(points=points)
        
    def __add__(self, other):
        """
        Returns a new curve that is the sum of this curve with another curve ``other`` or a constant value.
        :param  :py:class:`ws3.core.Curve` other: The curve to add with this curve or the constant value ``other``
        :return: A new curve that is the sum of this curve with another curve ``other`` or a constant value.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [_y+other for _y in self.y()] if isinstance(other, float) else [a+b for a,b in zip(self.y(), other.y())]
        points = list(zip(self.x, y))
        return Curve(points=points)  

    def __sub__(self, other):
        """
        Returns a new curve that is the difference of this curve with another curve ``other`` or a constant value.
        :param  :py:class:`ws3.core.Curve` other: The curve to subtract with this curve or the constant value ``other``
        :return: A new curve that is the difference of this curve with another curve ``other`` or a constant value.
        :rtype: :py:class:`ws3.core.Curve`
        """
        y = [_y-other for _y in self.y()] if isinstance(other, float) else [a-b for a,b in zip(self.y(), other.y())]
        points = list(zip(self.x, y))
        return Curve(points=points)
    
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__
    

class Node:
    """
    A node class representing a state in a dynamic programming state tree.
    """

    def __init__(self, nid, data=None, parent=None):
        """
        The constructor for a node class.

        :param nid: The unique ID of this node
        :param data: The data stored in this node
        :param parent: The parent of this node
        """
        self.nid = nid
        self._data = data
        self._parent = parent
        self._children = []

    def is_root(self):
        """
        Check if the current object is the root node.
        
        :return: ``True`` if the object is the root node, ``False`` otherwise.
        :rtype: bool
        """
        return self._parent is None

    def is_leaf(self):
        """
        Checks if the current object is a leaf node (i.e., node has no children).
        
        :return: ``True`` if the object is a leaf node, ``False`` otherwise.
        :rtype: bool
        """
        return not self._children

    def add_child(self, child):
        """
        The function adds a child node to the current object.
        
        :param :py:class:`ws3.tree.Node` child: The child node to be added.
        """
        self._children.append(child)

    def parent(self):
        """
        The function gets the parent node of the current object.
       
        :return: The parent node.
        :rtype:  :py:class:`ws3.tree.Node`
        """
        return self._parent

    def children(self):
        """
        The function gets the list of child nodes of the current object.
        
        :return: List of child nodes.
        :rtype: list of :py:class:`ws3.tree.Node` objects.
        """ 
        return self._children
    
    def data(self, key=None):
        """
        The function gets the data associated with the current object.
        If a specific key is provided, return the corresponding value.
        If no key is provided, return the entire data dictionary.
        
        :param key: The key to retrieve a specific value (default is None).   
        :return: The data associated with the ``key`` if a key is specified 
            (or the entire data dictionary if a key is not specified).
        """
        if key:
            return self._data[key]
        else:
            return self._data


class Tree:
    """
    Represents a tree object.
    """
    def __init__(self, period=1):
        self._period = period
        self._nodes = [Node(0)]
        self._path = [self._nodes[0]]

    def children(self, nid):
        """
        The function gets the child nodes of the node with the specified ID.
        
        :param nid: The ID of the node for which to retrieve children.
        :return: List of child nodes.
        :rtype: list of :py:class:`ws3.tree.Node` objects.
        """
        return [self._nodes[cid] for cid in self._nodes[nid].children()]
        
    def nodes(self):
        """
        Returns all nodes in the tree.
        :returns: List of all nodes in the tree.
        :rtype: list of :py:class:`ws3.tree.Node` objects.
        """
        return self._nodes

    def node(self, nid):
        """
        Returns a node with the specified ID.
        
        :param nid: The unique identifier of the node to be retrieved.
        :return: The node object corresponding to the specified ID.
        :rtype: :py:class:`ws3.tree.Node`
        """
        return self._nodes[nid]
    
    def add_node(self, data, parent=None):
        """
        Adds a new node to the tree.
        
        :param data: The data associated with the new node.
        :param parent: The parent node to which the new node will be attached.    
        :return: The newly created node.
        :rtype: :py:class:`ws3.tree.Node`
        """
        n = Node(len(self._nodes), data, parent)
        self._nodes.append(n)
        return n

    def grow(self, data):
        """
        Expands the current path by adding a new child node.
        The new node is added as a child of the last node in the current path.
        The current path used by the optimization problem formulation functions
        to iterate over all possible states (in a depth-first-search pattern).
        
        :param data: The data associated with the new node.
        :return: The newly created node.
        :rtype: :py:class:`ws3.tree.Node`
        """
        parent = self._path[-1]
        child = self.add_node(data, parent=parent.nid)
        parent.add_child(child.nid)
        self._path.append(child)
        return child
        
    def ungrow(self):
        """
        Reduces the current path by removing the last node.
        """
        self._path.pop()
        
    def leaves(self):
        """
        Returns all leaf nodes.

        :return: A list of all leaf nodes.
        :rtype: list of :py:class:`ws3.tree.Node` objects
        """
        return [n for n in self._nodes if n.is_leaf()]
    
    def root(self):
        """
        Returns the root node.

        :return: The root node.
        :rtype: :py:class:`ws3.tree.Node`
        """
        return self._nodes[0]
    
    def path(self, leaf=None):
        """
        Retrieves the path from the root to a specific leaf node or to the current path.
        
        :param leaf: The leaf node for which the path is to be retrieved. 
            Default is ``None`` (which returns the current path).
        :return: a path
        :rtype: tuple of :py:class:`ws3.tree.Node` objects
        """
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
        """
        Retrieves paths from the root to all leaf nodes.

        :return: A list of paths from the root to all leaf nodes.
        :rtype: list of tuples of :py:class:`ws3.tree.Node` objects
        """
        return [self.path(leaf) for leaf in self.leaves()]



