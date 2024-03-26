import sys
sys.path.append('./ws3/')
import pytest,time
import numpy as np
from common import is_num, hash_dt, reproject, timed


def test_is_num():
    # Test with a valid numerical input
    assert is_num("241") == True
    assert is_num("0.15") == True
    assert is_num("-1000") == True
    assert is_num("4.55") == True
     
    # Test with invalid inputs
    assert is_num("abc") == False
    assert is_num("") == False
    assert is_num(" ") == False
    assert is_num("1.2.3") == False
    assert is_num("123a") == False

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
