"""
This module defines some core classes used elsewhere in the package.
These include classes to represent yield curves and dynamcic programming
state trees.
"""

from __future__ import annotations

import copy
from bisect import bisect_left

from ws3 import common


class Interpolator:
    """
    Interpolates x and y values from sparse curve point list.

    Used by the :py:class:`ws3.core.Curve` class to interpolate between real data points.

    """
    x: list[float]
    y: list[float]
    n: int
    m: list[float]
    _points: list[tuple[int, float]]

    def __init__(self, points: list[tuple[int, float]]) -> None:
        """
        :param points: A list of (x,y) coordinate pairs.
        """
        x, y = list(zip(*points, strict=False))
        self.x = list(map(float, x))
        self.y = list(map(float, y))
        self.n = len(x)
        intervals = list(zip(self.x, self.x[1:], self.y, self.y[1:], strict=False))
        try:
            self.m = [(y2 - y1)/(x2 - x1) for x1, x2, y1, y2 in intervals]
        except Exception:
            print(intervals)
            raise
        self._points = points

    def points(self) -> list[tuple[int, float]]:
        """
        Returns the points as a list of tuples representing the points.

        :return: A list of (x, y) coordinate pairs.
        """
        return list(zip(list(map(int, self.x)), self.y, strict=False))

    def __call__(self, x: float) -> float:
        """
        Interpolates the value of y at a given x.

        :param x: The x coordinate to interpolate.
        :return: The y value at the given x.
        """
        if x == 0:
            return self.y[0]
        i = bisect_left(self.x, x) - 1
        return self.y[i] + self.m[i] * (x - self.x[i])

    def lookup(self, y: float, from_right: bool = False) -> float:
        """
        Looks up the x-coordinate corresponding to the given y-coordinate.

        :param y: The y-coordinate to look up.
        :param from_right: Flag indicating whether to search from the right. Defaults to ``False``.
        :return: The x-coordinate corresponding to the given y-coordinate.
        """
        if not from_right:
            for i, _x in enumerate(self.x):
                if self.y[i] > y:
                    break
            i -= 1
            if i == self.n - 1:
                return self.x[-1]
            try:
                return self.x[i] + (y - self.y[i])/self.m[i] if self.m[i] else self.x[i]
            except Exception:
                print(i, self.n, self.x, self.y)
                raise
            return _x
        else:
            raise NotImplementedError("lookup from_right not yet implemented")


class Curve:
    """
    Describes change in state over time (between treatments).
    """
    _type_default: str = 'a'

    label: str | None
    id: str | None
    is_volume: bool
    type: str
    period_length: float
    xmin: int
    xmax: int
    x: range | list[int]
    is_special: bool
    _y: list[float] | None
    epsilon: float
    is_locked: bool
    interp: Interpolator

    def __init__(
        self,
        label: str | None = None,
        id: str | None = None,
        is_volume: bool = False,
        points: list[tuple[int, float]] | None = None,
        type: str = _type_default,
        is_special: bool = False,
        period_length: float = common.PERIOD_LENGTH_DEFAULT,
        xmin: int = common.MIN_AGE_DEFAULT,
        xmax: int = common.MAX_AGE_DEFAULT,
        epsilon: float = common.CURVE_EPSILON_DEFAULT,
        simplify: bool = True,
    ) -> None:
        """
        :param label: A label for the curve.
        :param id: An ID for the curve.
        :param is_volume: Flag indicating whether the curve tracks volume. Defaults to ``False``.
        :param points: A list of (x,y) pairs defining the curve.
        :param type: A string indicating the type of curve. Defaults to ``'a'``.
        :param is_special: Flag indicating whether the curve is special. Defaults to ``False``.
            Special curves are immune to simplification.
        :param period_length: The length of the period. Defaults to :py:attr:`ws3.common.PERIOD_LENGTH_DEFAULT`.
        :param xmin: The minimum age. Defaults to :py:attr:`ws3.common.MIN_AGE_DEFAULT`.
        :param xmax: The maximum age. Defaults to :py:attr:`ws3.common.MAX_AGE_DEFAULT`.
        :param epsilon: The tolerance for simplifying the curve. Defaults to :py:attr:`ws3.common.CURVE_EPSILON_DEFAULT`.
        :param simplify: Flag indicating whether to simplify the curve. Defaults to ``True``.
        """
        self.label = label
        self.id = id
        self.is_volume = is_volume
        self.type = type
        self.period_length = period_length
        self.xmin = xmin
        self.xmax = xmax
        self.x = range(xmin, xmax + 1)
        self.is_special = is_special
        self._y = None
        self.epsilon = epsilon
        self.is_locked = False
        self.add_points(points or [(0, 0)], simplify=simplify) # defaults to zero curve

    def simplify(
        self,
        points: list[tuple[int, float]] | None = None,
        autotune: bool = True,
        compile_y: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Simplifies the curve by removing redundant points.

        :param points: The points to simplify. Defaults to None.
        :param autotune: Flag indicating whether to automatically tune the simplification process. Defaults to True.
        :param compile_y: Flag indicating whether to compile the y-component. Defaults to False.
        :param verbose: Flag indicating whether to print verbose output. Defaults to False.
        """
        if self.is_special:
            return
        assert not self.is_locked
        points = self.points() if points is None else points
        n = len(points)
        ysum = sum(self)
        if n <= 2 or ysum < self.epsilon:
            return
        estep = self.epsilon
        error = 0.
        e = 0.
        sentinel = 0
        max_iters = 100
        while error < self.epsilon and sentinel < max_iters and len(self.points()) > 2:
            _points = copy.copy(self.points()) # backup
            self._simplify(e)
            if sentinel > 0 and len(self.points()) == len(_points):
                break
            error = abs(sum(self) - ysum) / ysum
            if error >= self.epsilon:
                break
            e += estep
            sentinel += 1
        self.interp = Interpolator(_points) # restore from backup
        self._y = None
        if compile_y:
            self._compile_y()
        if verbose:
            error = abs(sum(self) - ysum) / ysum
            print('after final simplify', n, len(self.points()), float(n)/float(len(self.points())), error, ysum, sentinel) #, e, abs(sum(self) - ysum) / ysum

    def _simplify(self, e: float, compile_y: bool = False) -> None:
        """
        Simplify the curve using a linear interpolation. Internal method, called from ``self.simplify()``.
        .. note::
           Implementation was modified so that point list is stored only once (in interp).
        """
        points = self.points()
        p = copy.copy(points)
        # print self.label, p
        n = 0
        for i in range(1, len(p) - 1):
            s1, s2 = [(p[i+j][1] - p[i+j-1][1]) / (p[i+j][0] - p[i+j-1][0]) for j in [0, 1]]
            if abs(s2 - s1) < e:
                n += 1
                points.remove(p[i]) # remove redundant point
        self.interp = Interpolator(points)
        self._y = None
        if compile_y:
            self._compile_y()

    def add_points(
        self,
        points: list[tuple[int, float]],
        simplify: bool = True,
        compile_y: bool = False,
    ) -> None:
        """
        Adds points to the curve and optionally simplifies point geometry.

        :param points: The points to add to the curve.
        :param simplify: Flag indicating whether to simplify the curve after adding points. Defaults to ``True``.
        :param compile_y: Flag indicating whether to compile the y-component after adding points. Defaults to ``False``.
        """
        assert not self.is_locked
        _x, _y = list(zip(*points, strict=False))
        x: list[float] = list(map(float, _x))
        y: list[float] = [float(_v) for _v in _y]
        x_min = x[0]
        if x_min > 0:
            if x_min > 1:
                x.insert(0, x_min - 1)
                y.insert(0, 0.)
            x.insert(0, 0)
            y.insert(0, 0.)
        if x[-1] < self.xmax:
            x.append(self.xmax)
            y.append(y[-1])
        points = list(zip(map(int, x), y, strict=False))
        self.interp = Interpolator(points)
        if simplify:
            self.simplify(points, compile_y)
        elif compile_y:
            self._compile_y()

    def points(self) -> list[tuple[int, float]]:
        """
        :return: List of curve points.
        """
        return self.interp.points()

    def lookup(self, y: float, from_right: bool = False, roundx: bool = False) -> int:
        """
        Looks up the x-coordinate corresponding to the given y-coordinate.

        :param y: The y-coordinate to look up.
        :param from_right: Flag indicating whether to search from the right. Defaults to ``False``.
        :param roundx: Flag indicating whether to round the x-coordinate to the nearest integer. Defaults to ``False``.
        :return: The x-coordinate corresponding to the given y-coordinate.
        """
        x = self.interp.lookup(y, from_right)
        if roundx:
            return int(round(x))
        else:
            return int(x)

    def range(
        self,
        lo: float | None = None,
        hi: float | None = None,
        as_bounds: bool = False,
        left_range: bool = True,
    ) -> Curve | tuple[int, int]:
        """
        Returns a Curve representing the range within the specified bounds.

        :param lo: The lower bound of the range. Defaults to None.
        :param hi: The upper bound of the range. Defaults to None.
        :param as_bounds: Flag indicating whether to return the range as a
            tuple of bounds. Defaults to ``False``.
        :param left_range: Flag indicating whether to look up the upper bound
            from the left (default) or from the right (widest possible range).
        :return: Returns either curve representing
          the range within the specified bounds, or a tuple representing lower- and upper-bound
          values (if ``as_bounds`` set to ``True``).
        :rtype: Curve or tuple
        """
        lb = int(round(self.interp.lookup(lo))) if lo is not None else 0
        ub = int(round(self.interp.lookup(hi, from_right=not left_range))) if hi is not None else self.xmax
        points: list[tuple[int, float]] = [(lb, 1.0), (ub, 1.0)] if ub > lb else [(lb, 1.0)]
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

    def cai(self) -> Curve:
        """
        Calculates a current annual increment (CAI) curve.

        :return: A curve representing the current annual increment.
        :rtype: Curve
        """
        X = list(range(1, self.xmax))
        Y = [self[x] - self[x-1] for x in X]
        points = list(zip(X, Y, strict=False))
        return Curve(points=points)

    def mai(self) -> Curve:
        """
        Calculates a mean annual increment (MAI) curve.

        :return: A curve representing the mean annual increment.
        :rtype: Curve
        """
        X = range(1, self.xmax)
        Y = [self[x] / x for x in X[1:]]
        points = list(zip(X, Y, strict=False))
        return Curve(points=points)

    def ytp(self) -> Curve:
        """
        Returns a years-to-peak (YTP) curve. This curve is a measure of how many years
        it takes for the curve to reach its peak (positive values to the left of the peak,
        and negative values to the right of the peak).
        :return: A curve representing the years to peak.
        :rtype: Curve
        """
        y = self.y()
        argmax = y.index(max(y))
        return Curve(points=[(x, argmax-x) for x in self.x])

    def _compile_y(self) -> None:
        """
        Compiles the y values from the x values stored in ``self.x``,
        and stores them in ``self._y``.
        """
        self._y = [self.interp(x) for x in self.x]

    def y(self, compile_y: bool = False) -> list[float]:
        """
        Returns the y-values of the curve stored in ``self._y`` (will first compile them if ``compile_y`` is set
        to ``True`` and ``self._y`` is empty), else will interpolate a list of y values
        for each x value in ``self.x``.

        :param compile_y: Flag indicating whether to compile the y-component of the curve. Defaults to ``False``.
        :return: A list of y values.
        """
        if compile_y and not self._y:
            self._compile_y()
            if self._y is not None:
                return self._y
            return []
        return [self.interp(x) for x in self.x]

    def __iter__(self):
        """
        Returns an iterator that iterates through the y values of this curve.
        """
        yield from self.y()

    def __getitem__(self, x: int) -> float:
        """
        Returns the y value of this curve at a given x-value ``x``.
        """
        return self._y[x] if self._y else self.interp(x)

    def __and__(self, other: Curve) -> Curve:
        """
        Returns a new curve that is the intersection of this curve with another curve ``other``.
        :param other: The curve to intersect with this curve.
        :return: A new curve that is the intersection of this curve with another curve ``other``.
        :rtype: Curve
        """
        y = [self[x] and other[x] for x in self.x]
        points = list(zip(self.x, y, strict=False))
        return Curve(points=points)

    def __or__(self, other: Curve) -> Curve:
        """
        Returns a new curve that is the union of this curve with another curve ``other``.
        :param other: The curve to union with this curve.
        :return: A new curve that is the union of this curve with another curve ``other``.
        :rtype: Curve
        """
        y = [self[x] or other[x] for x in self.x]
        points = list(zip(self.x, y, strict=False))
        return Curve(points=points)

    def __mul__(self, other: Curve | float) -> Curve:
        """
        Returns a new curve that is the product of this curve with another curve ``other`` or a constant value.
        :param other: The curve to multiply with this curve or the constant value ``other``.
        :return: A new curve that is the product of this curve with another curve ``other`` or a constant value.
        :rtype: Curve
        """
        y = [_y*other for _y in self.y()] if isinstance(other, float) else [a*b for a,b in zip(self.y(), other.y(), strict=False)]
        points = list(zip(self.x, y, strict=False))
        return Curve(points=points)

    def __truediv__(self, other: Curve | float) -> Curve:
        """
        Returns a new curve that is the quotient of this curve with another curve ``other`` or a constant value.
        :param other: The curve to divide with this curve or the constant value ``other``.
        :return: A new curve that is the quotient of this curve with another curve ``other`` or a constant value.
        :rtype: Curve
        """
        if isinstance(other, float):
            y = [_y / other for _y in self.y()]
        else:
            y = [a/b for a, b in zip(self.y(), other.y(), strict=False)]
        points = list(zip(self.x, y, strict=False))
        return Curve(points=points)

    def __add__(self, other: Curve | float) -> Curve:
        """
        Returns a new curve that is the sum of this curve with another curve ``other`` or a constant value.
        :param other: The curve to add with this curve or the constant value ``other``
        :return: A new curve that is the sum of this curve with another curve ``other`` or a constant value.
        :rtype: Curve
        """
        y = [_y+other for _y in self.y()] if isinstance(other, float) else [a+b for a,b in zip(self.y(), other.y(), strict=False)]
        points = list(zip(self.x, y, strict=False))
        return Curve(points=points)

    def __sub__(self, other: Curve | float) -> Curve:
        """
        Returns a new curve that is the difference of this curve with another curve ``other`` or a constant value.
        :param other: The curve to subtract with this curve or the constant value ``other``
        :return: A new curve that is the difference of this curve with another curve ``other`` or a constant value.
        :rtype: Curve
        """
        y = [_y-other for _y in self.y()] if isinstance(other, float) else [a-b for a,b in zip(self.y(), other.y(), strict=False)]
        points = list(zip(self.x, y, strict=False))
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

        :param :py:class:`ws3.core.Node` child: The child node to be added.
        """
        self._children.append(child)

    def parent(self):
        """
        The function gets the parent node of the current object.

        :return: The parent node.
        :rtype:  :py:class:`ws3.core.Node`
        """
        return self._parent

    def children(self):
        """
        The function gets the list of child nodes of the current object.

        :return: List of child nodes.
        :rtype: list of :py:class:`ws3.core.Node` objects.
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
        :rtype: list of :py:class:`ws3.core.Node` objects.
        """
        return [self._nodes[cid] for cid in self._nodes[nid].children()]

    def nodes(self):
        """
        Returns all nodes in the tree.
        :returns: List of all nodes in the tree.
        :rtype: list of :py:class:`ws3.core.Node` objects.
        """
        return self._nodes

    def node(self, nid):
        """
        Returns a node with the specified ID.

        :param nid: The unique identifier of the node to be retrieved.
        :return: The node object corresponding to the specified ID.
        :rtype: :py:class:`ws3.core.Node`
        """
        return self._nodes[nid]

    def add_node(self, data, parent=None):
        """
        Adds a new node to the tree.

        :param data: The data associated with the new node.
        :param parent: The parent node to which the new node will be attached.
        :return: The newly created node.
        :rtype: :py:class:`ws3.core.Node`
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
        :rtype: :py:class:`ws3.core.Node`
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
        :rtype: list of :py:class:`ws3.core.Node` objects
        """
        return [n for n in self._nodes if n.is_leaf()]

    def root(self):
        """
        Returns the root node.

        :return: The root node.
        :rtype: :py:class:`ws3.core.Node`
        """
        return self._nodes[0]

    def path(self, leaf=None):
        """
        Retrieves the path from the root to a specific leaf node or to the current path.

        :param leaf: The leaf node for which the path is to be retrieved.
            Default is ``None`` (which returns the current path).
        :return: a path
        :rtype: tuple of :py:class:`ws3.core.Node` objects
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
        :rtype: list of tuples of :py:class:`ws3.core.Node` objects
        """
        return [self.path(leaf) for leaf in self.leaves()]
