###################################################################################
# MIT License

# Copyright (c) 2015-2017 Gregory Paradis

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################################################

"""
This module implements functions for formulating and solving optimization problems.
"""

import gurobipy as grb

SENSE_MINIMIZE = +1 # same as GRB.MINIMIZE
SENSE_MAXIMIZE = -1 # same as GRB.MAXIMIZE
SENSE_EQ = '=' # same as GRB.EQUAL
SENSE_GEQ = '>' # same as GRB.GREATER_EQUAL
SENSE_LEQ = '<' # same as GRB.LESS_EQUAL
VTYPE_INTEGER = 'I' # same as GRB.INTEGER
VTYPE_BINARY = 'B' # same as GRB.BINARY
VTYPE_CONTINUOUS = 'C' # same as GRB.CONTINUOUS
VBNDS_INF = float('inf')
SOLVR_GUROBI = 'gurobi'

GUROBI_MAP = {
    SENSE_MINIMIZE:grb.GRB.MINIMIZE,
    SENSE_MAXIMIZE:grb.GRB.MAXIMIZE,
    VTYPE_INTEGER:grb.GRB.INTEGER,
    VTYPE_BINARY:grb.GRB.BINARY,
    VTYPE_CONTINUOUS:grb.GRB.CONTINUOUS,
    SENSE_EQ:grb.GRB.EQUAL,
    SENSE_GEQ:grb.GRB.GREATER_EQUAL,
    SENSE_LEQ:grb.GRB.LESS_EQUAL}

class Variable:
    def __init__(self, name, vtype, lb=0., ub=VBNDS_INF, val=None):
        self.name = name
        self.vtype = vtype
        self.lb = lb
        self.ub = ub
        self.val = val

class Constraint:
    def __init__(self, name, coeffs, sense, rhs):
        self.name = name
        self.coeffs = coeffs
        self.sense = sense
        self.rhs = rhs
                
class Problem:
    def __init__(self, name, sense=SENSE_MAXIMIZE, solver=SOLVR_GUROBI):
        self._name = name
        self._vars = {}
        self._z = {}
        self._constraints = {}
        #self._solution = None
        self._sense = sense
        self._solver = solver
        self._dispatch_map = {SOLVR_GUROBI:self._solve_gurobi}

    def add_var(self, name, vtype, lb=0., ub=VBNDS_INF):
        self._vars[name] = Variable(name, vtype, lb, ub)
        self._solution = None # modifying problem kills solution

    def var_names(self):
        return self._vars.keys()

    def constraint_names(self):
        return self._constraints.keys()

    def name(self):
        return self._name
        
    def var(self, name):
        return self._vars[name]

    def sense(self, val=None):
        if val:
            self._sense = val
            self._solution = None # modifying problem kills solution
        else:
            return self._sense

    def solved(self):
        return self._solution is not None
        
    def z(self, coeffs, validate=False):
        if coeffs:
            if validate:
                for v in coeffs:
                    assert v in self._vars
            self._z = coeffs
            self._solution = None # modifying problem kills solution
        else:
            assert self.solved()
            return sum([self._z[v] * self._solution[v] for v in self._vars.keys()])
        
    def add_constraint(self, name, coeffs, sense, rhs, validate=False):
        if validate:
            for v in coeffs:
                assert v in self._vars
        self._constraints[name] = Constraint(name, coeffs, sense, rhs)
        self._solution = None # modifying problem kills solution

    def solver(self, val):
        if val:
            self._solver = val
        else:
            return self._solver

    def solution(self):
        #return self._solution
        return {x:self._vars[x].val for x in self._vars}

    def solve(self, validate=False):
        if validate:
            assert False # not implemented yet, but later check that all systems are GO before launching...
        self._dispatch_map[self._solver].__get__(self, type(self))()

    def _solve_gurobi(self):
        self._m = m = grb.Model(self._name)
        vars = {v.name:m.addVar(name=v.name, vtype=v.vtype) for v in self._vars.values()}
        m.update()
        z = grb.LinExpr()
        for v in vars:
            z += self._z[v] * vars[v]
        m.setObjective(expr=z, sense=GUROBI_MAP[self._sense])
        for name, constraint in self._constraints.items():            
            lhs = grb.LinExpr()
            for x in constraint.coeffs:
                lhs += constraint.coeffs[x] * vars[x]
            m.addConstr(lhs=lhs,
                        sense=GUROBI_MAP[constraint.sense],
                        rhs=constraint.rhs,
                        name=name)
        m.optimize()
        for k, v in self._vars.items():
            _v = m.getVarByName(k)
            v._solver_var = _v # might want to poke around this later...
            v.val = _v.X
