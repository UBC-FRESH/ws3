import sys
sys.path.append('./ws3/')
import pytest
from common import is_num


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
