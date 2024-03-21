import sys
sys.path.append('./ws3/')
import core, common
import pytest
# import ws3
# import ws3.core, ws3.common
from core import Interpolator, Curve
from bisect import bisect_left

@pytest.fixture
def sample_points():
    return [(1, 2), (3, 4), (5, 6)]

def test_interpolator_initialization(sample_points):
    interpolator = Interpolator(sample_points)
    assert interpolator.x == [1.0, 3.0, 5.0]
    assert interpolator.y == [2.0, 4.0, 6.0]
    assert interpolator.n == 3

def test_interpolator_points(sample_points):
    interpolator = Interpolator(sample_points)
    assert interpolator.points() == [(1, 2), (3, 4), (5, 6)]

def test_interpolator_call(sample_points):
    interpolator = Interpolator(sample_points)
    assert interpolator(0) == 2.0  # Should return the first y value
    assert interpolator(1) == 2.0  # Should interpolate between (1, 2) and (3, 4)

def test_interpolator_lookup(sample_points):
    interpolator = Interpolator(sample_points)
    assert interpolator.lookup(4) == 3  # Should return the x-coordinate corresponding to y=4

@pytest.fixture
def sample_curve():
    # Create a sample Curve instance for testing
    return Curve(label="Sample Curve", points=[(1, 1), (2, 2), (3, 3)])

def test_curve_initialization():
    # Test Curve initialization with default parameters
    curve = Curve()
    assert curve.label is None
    assert curve.id is None
    assert curve.is_volume == False
    assert curve.type == 'a'
    assert curve.period_length == common.PERIOD_LENGTH_DEFAULT
    assert curve.xmin == common.MIN_AGE_DEFAULT
    assert curve.xmax == common.MAX_AGE_DEFAULT
    assert curve.is_special == False
    assert curve.epsilon == common.CURVE_EPSILON_DEFAULT
    assert curve.is_locked == False
    assert curve.points() == [(common.MIN_AGE_DEFAULT, 0), (common.MAX_AGE_DEFAULT, 0)]

# def test_curve_add_points(sample_curve):
#     # Test adding points to a Curve instance
#     sample_curve.add_points([(4, 0), (5, 0)])
#     assert sample_curve.points() == [(common.MIN_AGE_DEFAULT, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (common.MAX_AGE_DEFAULT, 0)]

# def test_curve_simplify(sample_curve):
#     # Test simplifying a Curve instance
#     sample_curve.simplify()
#     assert sample_curve.points() == [(0, 0), (common.MAX_AGE_DEFAULT, 0)]

# def test_curve_lookup(sample_curve):
#     # Test lookup method of a Curve instance
#     assert sample_curve.lookup(2) == 2

# def test_curve_range(sample_curve):
#     # Test range method of a Curve instance
#     range_curve = sample_curve.range(1, 3)
#     assert range_curve.points() == [(0, 0), (1, 1), (2, 2), (3, 3)]

# def test_curve_mai(sample_curve):
#     # Test calculating mean annual increment (MAI) of a Curve instance
#     mai_curve = sample_curve.mai()
#     assert mai_curve.points() == [(0, 0), (1, 1), (2, 1), (3, 1)]

# def test_curve_ytp(sample_curve):
#     # Test calculating yield-to-point (YTP) of a Curve instance
#     ytp_curve = sample_curve.ytp()
#     assert ytp_curve.points() == [(0, 2), (1, 1), (2, 0), (3, 0)]







