import sys
sys.path.append('../ws3/')
import pytest, time
import numpy as np
import fiona
import os
import math
from ws3.common import is_num, hash_dt, reproject, timed, reproject_vector_data, sylv_cred, sylv_cred_formula, piece_size_ratio, harv_cost


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


def test_sylv_cred():
    # Test data
    P = 10.0
    vr = 2.0
    vp = 1.0
    formula = 1
    
    # Call the function
    result = sylv_cred(P, vr, vp, formula)

    expected_result = 126.33 
    
    # Assertion
    assert result == pytest.approx(expected_result, rel=1.3e-04)


def test_sylv_cred_formula_ec_m():
    treatment_type = 'ec'
    cover_type = 'M'

    assert sylv_cred_formula(treatment_type, cover_type) == 1

def test_sylv_cred_formula_ec_other():
    treatment_type = 'ec'
    cover_type = 'S'

    assert sylv_cred_formula(treatment_type, cover_type) == 2

def test_sylv_cred_formula_cj():
    treatment_type = 'cj'
    cover_type = 'S'

    assert sylv_cred_formula(treatment_type, cover_type) == 4

def test_sylv_cred_formula_cprog_r():
    treatment_type = 'cprog'
    cover_type = 'R'

    assert sylv_cred_formula(treatment_type, cover_type) == 7

def test_sylv_cred_formula_cprog_m():
    treatment_type = 'cprog'
    cover_type = 'M'

    assert sylv_cred_formula(treatment_type, cover_type) == 7



def test_piece_size_ratio_valid():
    treatment_type = 2
    cover_type = 'm'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0.8

def test_piece_size_ratio_invalid_treatment_type():
    treatment_type = 4
    cover_type = 'r'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0

def test_piece_size_ratio_invalid_cover_type():
    treatment_type = 2
    cover_type = 's'
    piece_size_ratios = {
        1: {'r': 0.45, 'm': 0.7, 'f': 0.8},
        2: {'r': 0.6, 'm': 0.8, 'f': 0.9},
        3: {'r': 0.1, 'm': 0.9, 'f': 1.0}
    }

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 0

def test_piece_size_ratio_empty_piece_size_ratios():
    treatment_type = 2
    cover_type = 'r'
    piece_size_ratios = {}

    assert piece_size_ratio(treatment_type, cover_type, piece_size_ratios) == 1.0



def test_harv_cost():
    piece_size = 10
    is_finalcut = True
    is_toleranthw = False
    partialcut_extracare = False
    A, B, C, D, E, F, G, K = 1.97, 0.405, 0.169, 0.164, 0.202, 13.6, 8.83, 0.0

    expected_result_1 = (
        A - (B * math.log(piece_size)) + (C * float(partialcut_extracare)) + 
        (D * float(is_finalcut)) - (E * (1 - float(is_toleranthw)))  
    )
    expected_result = math.exp(expected_result_1)+ ((F * float(is_toleranthw)) + (G * (1 - float(is_toleranthw)))) + K
    assert harv_cost(piece_size, is_finalcut, is_toleranthw, partialcut_extracare, A, B, C, D, E, F, G, K) == expected_result

