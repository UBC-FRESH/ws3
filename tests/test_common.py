import sys

sys.path.append('../ws3/')
import time

import numpy as np
import pytest

from ws3.common import hash_dt, is_num, reproject, timed


def test_is_num():
    # Test with a valid numerical input
    assert is_num("241")
    assert is_num("0.15")
    assert is_num("-1000")
    assert is_num("4.55")

    # Test with invalid inputs
    assert not is_num("abc")
    assert not is_num("")
    assert not is_num(" ")
    assert not is_num("1.2.3")
    assert not is_num("123a")


def test_hash_dt():
    # Test with a simple development type
    dt_1 = ['tsa24', '0', '439', '500', '439']
    dt_2 = [12466]
    dt_3 = ['test']
    result_1 = hash_dt(dt_1)
    result_2 = hash_dt(dt_2)
    result_3 = hash_dt(dt_3)
    assert isinstance(result_1, np.int32)
    assert isinstance(result_2, np.int32)
    assert isinstance(result_3, np.int32)

    # Determinism: same input must produce same output across calls
    assert hash_dt(dt_1) == result_1
    assert hash_dt(dt_2) == result_2
    assert hash_dt(dt_3) == result_3

    # Different inputs must produce different outputs (MD5 collision-resistant)
    assert hash_dt(dt_1) != hash_dt(dt_2)
    assert hash_dt(dt_1) != hash_dt(dt_3)
    assert hash_dt(dt_2) != hash_dt(dt_3)

    # Test with larger/more complex inputs that stress the int32 conversion
    dt_large = ['tsa24', '0', '999999', '999999', '999999', '100', '200', '300']
    result_large = hash_dt(dt_large)
    assert isinstance(result_large, np.int32)
    # Should be deterministic
    assert hash_dt(dt_large) == result_large

    # Stress the int32 range with inputs that produce very large md5 prefixes.
    # The struct.unpack('<i', ...) path is what replaced the overflow-prone
    # int(binascii.hexlify(...), 16).  The struct approach wraps correctly
    # (two's complement) whereas the old path would overflow on values > 2**31-1.
    dt_overflow = ['z' * 100, str(2**30), 'x' * 50]
    result_overflow = hash_dt(dt_overflow)
    assert isinstance(result_overflow, np.int32)
    # Result must be a valid int32 — np.int32 range is [-2**31, 2**31-1]
    assert np.iinfo(np.int32).min <= int(result_overflow) <= np.iinfo(np.int32).max


def test_reproject():
    # Create a sample feature dictionary with geometry
    feature = {'geometry': {'type': 'Point', 'coordinates': [0, 0]}}

    # Define source and destination coordinate reference systems (CRS)
    srs_crs = 'EPSG:4326'  # WGS 84
    dst_crs = 'EPSG:3857'  # Web Mercator

    # Test reprojecting a point from WGS 84 to Web Mercator
    result = reproject(feature, srs_crs, dst_crs)
    assert 'geometry' in result  # Ensure geometry is still present in the result
    assert result['geometry']['type'] == 'Point'
    # Since the point is near the origin, the coordinates should remain similar in Web Mercator
    assert result['geometry']['coordinates'] == pytest.approx([0, 0], abs=1e-6)


def test_timed(capsys):
    # Define a sample function to be timed
    @timed
    def sample_function():
        time.sleep(1)

    # Call the sample function
    sample_function()

    # Capture the printed output
    captured = capsys.readouterr()

    # Check if the output contains the function name and the elapsed time
    assert 'sample_function took' in captured.out

    # Ensure that the elapsed time is non-zero
    assert float(captured.out.split()[2]) > 0
