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
The notation is very generic (i.e., refers to variables, constraints, problems, solutions, etc.).
All the wood-supply-problem--specific references are implemented in the ``forest`` module.

The ``Problem`` class is the main functional unit here. It encapsulates optimization problem data (i.e., variables, constraints, objective function, and optimal solution), as well as methods to operate on this data (i.e., methods to build and solve the problem, and report on the optimal solution).

Note that we implemented a modular design that decouples the implementation from the choice of solver. Currently, only bindings to the Gurobi solver are implemented, although bindings to other solvers can easilty be added (we will add more binding in later releases, as the need arises). 
"""


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


class Variable:
    """
    Encapsulates data describing a variable in an optimization problem. This includes a variable name (should be unique within a problem, although the user is responsible for enforcing this condition), a variable type (should be one of ``VTYPE_CONTINUOUS``, ``VTYPE_INTEGER``, or ``VTYPE_BINARY``), variable value bound (lower bound defaults to zero, upper bound defaults to positive infinity), and variable value (defaults to ``None``).
    """
    def __init__(self, name, vtype, lb=0., ub=VBNDS_INF, val=None):
        self.name = name
        self.vtype = vtype
        self.lb = lb
        self.ub = ub
        self.val = val

class Constraint:
    """
    Encapsulates data describing a constraint in an optimization problem. This includes a constraint name (should be unique within a problem, although the user is responsible for enforcing this condition), a vector of coefficient values (length of vector should match the number of variables in the problem, although the user is responsible for enforcing this condition), a sense (should be one of ``SENSE_EQ``, ``SENSE_GEQ``, or ``SENSE_LEQ``), and a right-hand-side value.
    """
    def __init__(self, name, coeffs, sense, rhs):
        self.name = name
        self.coeffs = coeffs
        self.sense = sense
        self.rhs = rhs
                
class Problem:
    """
    This is the main class of the ``opt`` module---it encapsulates optimization problem data (i.e., variables, constraints, objective function, optimal solution, and choice of solver), as well as methods to operate on this data (i.e., methods to build and solve the problem, and report on the optimal solution).
    """
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
        """
        Adds a variable to the problem. The variable name should be unique within the problem (user is responsible for enforcing this condition). Variable type should be one of ``VTYPE_CONTINUOUS``, ``VTYPE_INTEGER``, or ``VTYPE_BINARY``. Variable value bounds default to zero for the lower bound and positive infinity for the upper bound.

        Note that calling this method resets the value of the optimal solution to ``None``. 
        """
        self._vars[name] = Variable(name, vtype, lb, ub)
        self._solution = None # modifying problem kills solution

    def var_names(self):
        """
        Return a list of variable names.
        """
        return list(self._vars.keys())

    def constraint_names(self):
        """
        Returns a list of constraint names.
        """
        return list(self._constraints.keys())

    def name(self):
        """
        Returns problem name.
        """
        return self._name
        
    def var(self, name):
        """
        Returns a ``Variable`` instance, given a variable name.
        """
        return self._vars[name]

    def sense(self, val=None):
        """
        Returns (or sets) objective function sense. Value should be one of ``SENSE_MINIMIZE`` or ``SENSE_MAXIMIZE``.
        """
        if val:
            self._sense = val
            self._solution = None # modifying problem kills solution
        else:
            return self._sense

    def solved(self):
        """
        Returns ``True`` if the problem has been solved, ``False`` otherwise.
        """
        return self._solution is not None
        
    def z(self, coeffs=None, validate=False):
        """
        Returns the objective function value if ``coeffs`` is not provided (triggers an exception if problem has not been solved yet), or updates the objective function coefficient vector (resets the value of the optimal solution to ``None``).
        """
        if coeffs:
            if validate:
                for v in coeffs:
                    assert v in self._vars
            self._z = coeffs
            self._solution = None # modifying problem kills solution
        else:
            assert self.solved()
            return sum([self._z[v] * self._solution[v] for v in list(self._vars.keys())])
        
    def add_constraint(self, name, coeffs, sense, rhs, validate=False):
        """
        Adds a constraint to the problem. The constraint name should be unique within the problem (user is responsible for enforcing this condition). Constraint coeffients should be provided as a ``dict``, keyed on variable names---length of constraint coefficient ``dict`` should match number of variables in the problem (user is responsible for enforcing this condition). Constraint sense should be one of ``SENSE_EQ``, ``SENSE_GEQ``, or ``SENSE_LEQ``. 

        Note that calling this method resets the value of the optimal solution to ``None``. 
        """
        if validate:
            for v in coeffs:
                assert v in self._vars
        self._constraints[name] = Constraint(name, coeffs, sense, rhs)
        self._solution = None # modifying problem kills solution

    def solver(self, val):
        """
        Sets the solver (defaults to ```SOLVER_GUROBI``` in the class constructor). Note that only Gurobi solver bindings are implemented at this time.
        """
        if val:
            self._solver = val
        else:
            return self._solver

    def solution(self):
        """
        Returns a ``dict`` of variable values, keyed on variable names.
        """
        #return self._solution
        return {x:self._vars[x].val for x in self._vars}

    def solve(self, validate=False):
        """
        Solves the optimization problem. Dispatches to a solver-specific method (only Gurobi bindings are implemented at this time).
        """
        if validate:
            assert False # not implemented yet, but later check that all systems are GO before launching...
        return self._dispatch_map[self._solver].__get__(self, type(self))()

    def _solve_gurobi(self, allow_feasrelax=True):
        import gurobipy as grb
        GUROBI_MAP = {
            SENSE_MINIMIZE:grb.GRB.MINIMIZE,
            SENSE_MAXIMIZE:grb.GRB.MAXIMIZE,
            VTYPE_INTEGER:grb.GRB.INTEGER,
            VTYPE_BINARY:grb.GRB.BINARY,
            VTYPE_CONTINUOUS:grb.GRB.CONTINUOUS,
            SENSE_EQ:grb.GRB.EQUAL,
            SENSE_GEQ:grb.GRB.GREATER_EQUAL,
            SENSE_LEQ:grb.GRB.LESS_EQUAL}
        GUROBI_IU = grb.GRB.status.INF_OR_UNBD, grb.GRB.status.INFEASIBLE, grb.GRB.status.UNBOUNDED
        self._m = m = grb.Model(self._name)
        vars = {v.name:m.addVar(name=v.name, vtype=v.vtype) for v in list(self._vars.values())}
        m.update()
        z = grb.LinExpr()
        for v in vars:
            z += self._z[v] * vars[v]
        m.setObjective(expr=z, sense=GUROBI_MAP[self._sense])
        for name, constraint in list(self._constraints.items()):            
            lhs = grb.LinExpr()
            for x in constraint.coeffs:
                lhs += constraint.coeffs[x] * vars[x]
            m.addConstr(lhs=lhs,
                        sense=GUROBI_MAP[constraint.sense],
                        rhs=constraint.rhs,
                        name=name)
        m.optimize()
        print('foo')
        if allow_feasrelax and m.status in GUROBI_IU: # infeasible or unbounded model
            print('ws3.opt._solve_gurobi: Model infeasible, enabling feasRelaxS mode.')
            m.feasRelaxS(1, False, False, True)
            m.optimize()
        if m.status == grb.GRB.OPTIMAL:
            for k, v in list(self._vars.items()):
                _v = m.getVarByName(k)
                v._solver_var = _v # might want to poke around this later...
                v.val = _v.X
        return m
