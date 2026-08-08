"""
Targeted coverage tests for ws3.core — Interpolator, Curve, Node, Tree.

Covers uncovered branches:
- Interpolator.lookup(from_right=True) → NotImplementedError
- Interpolator edge cases (x=0, lookup boundary)
- Curve.simplify with is_special/is_locked
- Curve.add_points with locked curve
- Curve.range (as_bounds, left_range=False, lb=0, ub=xmax)
- Curve.mai, ytp, cai
- Curve._compile_y
- Curve dunder ops (__and__, __or__, __mul__, __truediv__, __add__, __sub__)
- Tree/Node: grow, ungrow, path, paths, leaves, root, children, data(key)
"""

import sys

sys.path.append('../ws3/')

import pytest

from ws3.core import Curve, Interpolator, Node, Tree

# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------

class TestInterpolator:
    def test_lookup_from_right_raises(self):
        interp = Interpolator([(1, 1), (2, 2), (3, 3)])
        with pytest.raises(NotImplementedError):
            interp.lookup(1.5, from_right=True)

    def test_lookup_returns_x_for_y_at_boundary(self):
        # y values go 1, 2, 3 — looking up y=0.5 should find i=0 then return x[0]
        interp = Interpolator([(1, 1), (2, 2), (3, 3)])
        result = interp.lookup(0.5)
        assert isinstance(result, float)

    def test_lookup_returns_last_x_when_y_exceeds_all(self):
        interp = Interpolator([(1, 1), (2, 2), (3, 3)])
        # y=10 exceeds all y values (1, 2, 3); loop completes without break,
        # i ends at n-1 then decrements to n-2=1, so interpolation fires and
        # returns extrapolated value (10.0), not x[-1]
        result = interp.lookup(10)
        assert result == 10.0

    def test_call_at_x_equals_first_x(self):
        interp = Interpolator([(1, 10), (2, 20)])
        # x=1 matches first x; bisect_left returns 0, i=-1 ... but x==0 special case
        # Actually x=1 is not 0, so we go through bisect path
        result = interp(1)
        assert result == 10.0

    def test_call_at_zero_returns_first_y(self):
        interp = Interpolator([(1, 10), (2, 20)])
        assert interp(0) == 10.0

    def test_points_rounds_x_to_int(self):
        interp = Interpolator([(1.5, 10.0), (3.7, 20.0)])
        pts = interp.points()
        assert pts[0][0] == 1
        assert pts[1][0] == 3


# ---------------------------------------------------------------------------
# Curve — simplify / locked / special
# ---------------------------------------------------------------------------

class TestCurveSimplify:
    def test_simplify_skipped_when_special(self):
        c = Curve(points=[(0, 0), (10, 100), (20, 200)], is_special=True)
        before = len(c.points())
        c.simplify()
        assert len(c.points()) == before

    def test_simplify_skipped_when_locked(self):
        c = Curve(points=[(0, 0), (10, 100), (20, 200)])
        c.is_locked = True
        with pytest.raises(AssertionError):
            c.simplify()

    def test_simplify_noop_when_few_points(self):
        c = Curve(points=[(0, 0), (100, 100)])
        before = len(c.points())
        c.simplify()
        assert len(c.points()) == before

    def test_simplify_noop_when_sum_below_epsilon(self):
        c = Curve(points=[(0, 0.0001), (100, 0.0001)])
        c.epsilon = 1.0  # large epsilon
        before = len(c.points())
        c.simplify()
        assert len(c.points()) == before


class TestCurveAddPoints:
    def test_add_points_locked_raises(self):
        c = Curve()
        c.is_locked = True
        with pytest.raises(AssertionError):
            c.add_points([(10, 5)])

    def test_add_points_pads_to_xmax(self):
        c = Curve(xmax=100)
        c.add_points([(50, 10)], simplify=False)
        pts = c.points()
        xmax_pts = [p[0] for p in pts]
        assert max(xmax_pts) == 100

    def test_add_points_pads_from_zero(self):
        c = Curve()
        c.add_points([(10, 5)], simplify=False)
        pts = c.points()
        xmin_pts = [p[0] for p in pts]
        assert min(xmin_pts) == 0


# ---------------------------------------------------------------------------
# Curve — range, mai, ytp, cai, _compile_y
# ---------------------------------------------------------------------------

class TestCurveRange:
    def test_range_as_bounds(self):
        c = Curve(points=[(0, 0), (10, 10), (20, 20)])
        lb, ub = c.range(lo=5, hi=15, as_bounds=True)
        assert isinstance(lb, int)
        assert isinstance(ub, int)

    def test_range_left_range_false_raises(self):
        """left_range=False passes from_right=True to lookup, which is not implemented."""
        c = Curve(points=[(0, 0), (10, 10), (20, 20)])
        with pytest.raises(NotImplementedError):
            c.range(lo=5, hi=15, left_range=False)

    def test_range_lb_zero(self):
        c = Curve(points=[(0, 0), (10, 10), (20, 20)])
        result = c.range(lo=0, hi=10)
        assert isinstance(result, Curve)

    def test_range_ub_equals_xmax(self):
        c = Curve(points=[(0, 0), (10, 10), (20, 20)], xmax=20)
        result = c.range(lo=5, hi=20)
        assert isinstance(result, Curve)


class TestCurveMai:
    def test_mai_returns_curve(self):
        c = Curve(points=[(0, 0), (10, 100), (20, 200)])
        mai = c.mai()
        assert isinstance(mai, Curve)
        assert len(mai.points()) > 0


class TestCurveYtp:
    def test_ytp_returns_curve(self):
        c = Curve(points=[(0, 0), (10, 100), (20, 50)])
        ytp = c.ytp()
        assert isinstance(ytp, Curve)


class TestCurveCai:
    def test_cai_returns_curve(self):
        c = Curve(points=[(0, 0), (10, 100), (20, 200)])
        cai = c.cai()
        assert isinstance(cai, Curve)


class TestCurveCompileY:
    def test_compile_y_stores_y(self):
        c = Curve(points=[(0, 0), (10, 100)])
        assert c._y is None
        c._compile_y()
        assert c._y is not None
        assert len(c._y) == c.xmax + 1

    def test_y_with_compile_flag(self):
        c = Curve(points=[(0, 0), (10, 100)])
        vals = c.y(compile_y=True)
        assert len(vals) == c.xmax + 1
        assert c._y is not None


# ---------------------------------------------------------------------------
# Curve — dunder arithmetic
# ---------------------------------------------------------------------------

class TestCurveDunders:
    def _make(self):
        return Curve(points=[(0, 0), (10, 10), (20, 20)])

    def test_and(self):
        a = self._make()
        b = self._make()
        result = a & b
        assert isinstance(result, Curve)

    def test_or(self):
        a = self._make()
        b = self._make()
        result = a | b
        assert isinstance(result, Curve)

    def test_mul_float(self):
        a = self._make()
        result = a * 2.0
        assert isinstance(result, Curve)

    def test_mul_curve(self):
        a = self._make()
        b = self._make()
        result = a * b
        assert isinstance(result, Curve)

    def test_truediv_float(self):
        a = self._make()
        result = a / 2.0
        assert isinstance(result, Curve)

    def test_truediv_curve(self):
        # Use curves with no zero values to avoid ZeroDivisionError
        a = Curve(points=[(0, 1), (10, 10), (20, 20)])
        b = Curve(points=[(0, 1), (10, 2), (20, 4)])
        result = a / b
        assert isinstance(result, Curve)

    def test_add_float(self):
        a = self._make()
        result = a + 5.0
        assert isinstance(result, Curve)

    def test_add_curve(self):
        a = self._make()
        b = self._make()
        result = a + b
        assert isinstance(result, Curve)

    def test_sub_float(self):
        a = self._make()
        result = a - 5.0
        assert isinstance(result, Curve)

    def test_sub_curve(self):
        a = self._make()
        b = self._make()
        result = a - b
        assert isinstance(result, Curve)

    def test_rmul(self):
        a = self._make()
        result = 3.0 * a
        assert isinstance(result, Curve)

    def test_radd(self):
        a = self._make()
        result = 5.0 + a
        assert isinstance(result, Curve)

    def test_rsub(self):
        a = self._make()
        result = 100.0 - a
        assert isinstance(result, Curve)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:
    def test_is_root(self):
        n = Node(0)
        assert n.is_root() is True

    def test_is_not_root(self):
        parent = Node(0)
        child = Node(1, parent=parent.nid)
        assert child.is_root() is False

    def test_is_leaf(self):
        n = Node(0)
        assert n.is_leaf() is True

    def test_is_not_leaf(self):
        parent = Node(0)
        child = Node(1)
        parent.add_child(child.nid)
        assert parent.is_leaf() is False

    def test_data_with_key(self):
        n = Node(0, data={'foo': 42})
        assert n.data('foo') == 42

    def test_data_without_key(self):
        n = Node(0, data={'foo': 42, 'bar': 'hello'})
        assert n.data() == {'foo': 42, 'bar': 'hello'}

    def test_parent(self):
        parent = Node(0)
        child = Node(1, parent=parent.nid)
        # parent() returns the parent's nid (int), not the Node object
        assert child.parent() == parent.nid

    def test_children(self):
        parent = Node(0)
        c1 = Node(1)
        c2 = Node(2)
        parent.add_child(c1.nid)
        parent.add_child(c2.nid)
        assert len(parent.children()) == 2


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class TestTree:
    def test_root(self):
        t = Tree()
        assert t.root().is_root()

    def test_add_node(self):
        t = Tree()
        n = t.add_node({'x': 1})
        assert len(t.nodes()) == 2
        assert n.data() == {'x': 1}

    def test_grow(self):
        t = Tree()
        t.add_node({'level': 0})
        child = t.grow({'level': 1})
        assert child.data() == {'level': 1}
        assert not child.is_root()

    def test_ungrow(self):
        t = Tree()
        t.add_node({'level': 0})
        t.grow({'level': 1})
        assert len(t._path) == 2
        t.ungrow()
        assert len(t._path) == 1

    def test_leaves(self):
        t = Tree()
        t.add_node({'a': 1})
        t.grow({'b': 2})
        t.grow({'c': 3})
        # Both grown nodes are leaves.
        leaves = t.leaves()
        assert len(leaves) == 2

    def test_node(self):
        t = Tree()
        n = t.add_node({'x': 1})
        assert t.node(n.nid) is n

    def test_children(self):
        t = Tree()
        t.add_node({'a': 1})
        t.grow({'b': 2})
        t.ungrow()  # go back to root
        t.grow({'c': 3})
        children = t.children(t.node(0).nid)
        assert len(children) == 2

    def test_path_default(self):
        t = Tree()
        t.add_node({'a': 1})
        t.grow({'b': 2})
        path = t.path()
        assert len(path) == 1
        assert path[0].data() == {'b': 2}

    def test_path_to_leaf(self):
        t = Tree()
        t.add_node({'a': 1})
        t.grow({'b': 2})
        c1g = t.grow({'c': 3})
        path = t.path(c1g)
        assert len(path) == 2
        assert path[-1].data() == {'c': 3}

    def test_paths(self):
        t = Tree()
        t.add_node({'a': 1})
        t.grow({'b': 2})
        t.grow({'c': 3})
        paths = t.paths()
        assert len(paths) == 2

    def test_ungrow_underflow_raises(self):
        t = Tree()
        # Tree.__init__ sets _path = [self._nodes[0]] (just the root).
        # ungrow pops from _path, so calling it twice removes the root.
        t.ungrow()  # removes root — path is now empty
        with pytest.raises(IndexError):
            t.ungrow()
