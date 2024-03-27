import sys
sys.path.append('./ws3/')
from opt import Variable, Constraint, Problem
import pytest


VBNDS_INF = float('inf')  # Define VBNDS_INF for testing
SENSE_EQ = '=' # same as GRB.EQUAL
SENSE_GEQ = '>' # same as GRB.GREATER_EQUAL
SENSE_LEQ = '<' # same as GRB.LESS_EQUAL

# test class Variable
def test_variable_initialization():
    name = "x"
    vtype = "continuous"
    lb = 0.0
    ub = VBNDS_INF
    val = None

    var = Variable(name, vtype, lb, ub, val)

    assert var.name == name
    assert var.vtype == vtype
    assert var.lb == lb
    assert var.ub == ub
    assert var.val == val

def test_variable_defaults():
    name = "x"
    vtype = "continuous"

    var = Variable(name, vtype)

    assert var.name == name
    assert var.vtype == vtype
    assert var.lb == 0.0
    assert var.ub == VBNDS_INF
    assert var.val == None

def test_variable_with_value():
    name = "x"
    vtype = "continuous"
    val = 5.0

    var = Variable(name, vtype, val=val)

    assert var.name == name
    assert var.vtype == vtype
    assert var.lb == 0.0
    assert var.ub == VBNDS_INF
    assert var.val == val

def test_variable_with_invalid_bound():
    name = "x"
    vtype = "continuous"
    lb = 10.0
    ub = -5.0

    with pytest.raises(ValueError):
        Variable(name, vtype, lb, ub)

# test class Constraint
def test_constraint_initialization():
    name = "constraint1"
    coeffs = [1, 2, 3]
    sense = SENSE_EQ
    rhs = 10

    constraint = Constraint(name, coeffs, sense, rhs)

    assert constraint.name == name
    assert constraint.coeffs == coeffs
    assert constraint.sense == sense
    assert constraint.rhs == rhs

def test_constraint_invalid_coefficients():
    name = "constraint2"
    coeffs = []  # Invalid coefficients
    sense = SENSE_LEQ
    rhs = 5

    with pytest.raises(ValueError):
        Constraint(name, coeffs, sense, rhs)

def test_constraint_invalid_sense():
    name = "constraint3"
    coeffs = [1, 2, 3]
    sense = "invalid_sense"
    rhs = 8

    with pytest.raises(ValueError):
        Constraint(name, coeffs, sense, rhs)

if __name__ == "__main__":
    pytest.main()