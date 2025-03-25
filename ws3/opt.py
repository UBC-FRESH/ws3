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
SOLVER_GUROBI = 'gurobi'
SOLVER_PULP = 'pulp'
STATUS_OPTIMAL = 'optimal'
STATUS_INFEASIBLE = 'infeasible'
STATUS_UNBOUNDED = 'unbounded'

class Variable:
    """
    Encapsulates data describing a variable in an optimization problem. This includes a variable name (should be unique within a problem, although the user is responsible for enforcing this condition), a variable type (should be one of ``VTYPE_CONTINUOUS``, ``VTYPE_INTEGER``, or ``VTYPE_BINARY``), variable value bound (lower bound defaults to zero, upper bound defaults to positive infinity), and variable value (defaults to ``None``).
    """
    def __init__(self, name, vtype, lb=0., ub=VBNDS_INF, val=None):
        if lb > ub:
            raise ValueError("Lower bound cannot be greater than upper bound")
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
        if not isinstance(coeffs, dict) or len(coeffs) == 0:
            raise ValueError("Coefficients must be a non-empty list")
        if not all(isinstance(coeff, (int, float)) for coeff in coeffs.values()):
            raise ValueError("Coefficients must be integers or floats")
        if not isinstance(sense, str) or sense not in {'=', '>', '<'}:
            raise ValueError("Sense must be one of '=', '>', or '<'")
        self.name = name
        self.coeffs = coeffs
        self.sense = sense
        self.rhs = rhs
                
class Problem:
    """
    This is the main class of the ``opt`` module---it encapsulates optimization problem data (i.e., variables, constraints, objective function, optimal solution, and choice of solver), as well as methods to operate on this data (i.e., methods to build and solve the problem, and report on the optimal solution).
    """
    def __init__(self, name, sense=SENSE_MAXIMIZE, solver=SOLVER_PULP):
        self._name = name
        self._vars = {}
        self._z = {}
        self._constraints = {}
        #self._solution = None
        self._sense = sense
        self._solver = solver
        self._solver_backend = None
        self._dispatch_map = {SOLVER_PULP:self._solve_pulp, 
                              SOLVER_GUROBI:self._solve_gurobi}

    def add_var(self, name, vtype, lb=0., ub=VBNDS_INF):
        """
        The function adds a variable to the problem.
    
        :param str name: The variable name that needs to be unique within the problem (user is responsible for enforcing this condition) type.
        :param str vtype: The variable type that has to be one of ``VTYPE_CONTINUOUS``, ``VTYPE_INTEGER``, or ``VTYPE_BINARY``.
        :param float lb: The lower bound value for the variable (Default is zero).
        :param float ub: The upper bound value for the variable (Default is positive infinity).
    
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
        Returns (or sets) objective function sense.
        :param str val: Value should be one of ``SENSE_MINIMIZE`` or ``SENSE_MAXIMIZE``.
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
        This function adds a constraint to the problem.
    
        :param str name: The constraint name should be unique within the problem (user is responsible for enforcing this condition).
        :param dict coeffs: Constraint coeffients should be provided as a ``dict``, keyed on variable names---length of constraint coefficient ``dict`` should match number of variables in the problem (user is responsible for enforcing this condition).
        :param float sense: Constraint sense should be one of ``SENSE_EQ``, ``SENSE_GEQ``, or ``SENSE_LEQ``.
        :param float rhs: The right hand side of the constraint.

        Note that calling this method resets the value of the optimal solution to ``None``
    
        """

        if validate:
            for v in coeffs:
                assert v in self._vars
        self._constraints[name] = Constraint(name, coeffs, sense, rhs)
        self._solution = None # modifying problem kills solution

    def solver(self, val):
        """
        Sets the solver backend (defaults to ``SOLVER_PULP`` in the class constructor). 
        
        Use ``SOLVER_GUROBI`` to use Gurobi solver bindings.
        """
        if val:
            self._solver = val
        else:
            return self._solver

    def solution(self):
        """
        Returns a ``dict`` of variable values, keyed on variable names.
        """
        return self._solution
        #return {x:self._vars[x].val for x in self._vars}

    def solve(self, validate=False):
        """
        Solves the optimization problem. Dispatches to a solver-specific method (only Gurobi bindings are implemented at this time).
        """
        if validate:
            assert False # not implemented yet, but later check that all systems are GO before launching...
        self._dispatch_map[self._solver].__get__(self, type(self))()
        if self.status() == STATUS_OPTIMAL:
            self._solution = {x:self._vars[x].val for x in self._vars}

    def status(self):
        """
        Checks whether the current solution is infeasible (i.e., not feasible).
        """
        import ws3.opt
        # import gurobipy
        import pulp
        match self._solver:
            case ws3.opt.SOLVER_PULP:
                match self._model.status:
                    case pulp.constants.LpStatusInfeasible:
                        return STATUS_INFEASIBLE
                    case pulp.constants.LpStatusUnbounded:
                        return STATUS_UNBOUNDED
                    case pulp.constants.LpStatusOptimal:
                        return STATUS_OPTIMAL
            case ws3.opt.SOLVER_GUROBI:
                match self._model.status:
                    case gurobipy.GRB.INFEASIBLE:
                        return STATUS_INFEASIBLE
                    case gurobipy.GRB.UNBOUNDED:
                        return STATUS_UNBOUNDED
                    case gurobipy.GRB.OPTIMAL:
                        return STATUS_OPTIMAL
            # add last case to return None if no match
            case _:
                return None


    def _solve_gurobi(self, allow_feasrelax=True):
        """
        Solve the LP optimization problem using Gurobi.

        Returns
        -------
        None
        """
        import gurobipy as grb
        const_map = {
            SENSE_MINIMIZE:grb.GRB.MINIMIZE,
            SENSE_MAXIMIZE:grb.GRB.MAXIMIZE,
            VTYPE_INTEGER:grb.GRB.INTEGER,
            VTYPE_BINARY:grb.GRB.BINARY,
            VTYPE_CONTINUOUS:grb.GRB.CONTINUOUS,
            SENSE_EQ:grb.GRB.EQUAL,
            SENSE_GEQ:grb.GRB.GREATER_EQUAL,
            SENSE_LEQ:grb.GRB.LESS_EQUAL}
        GUROBI_IU = grb.GRB.status.INF_OR_UNBD, grb.GRB.status.INFEASIBLE, grb.GRB.status.UNBOUNDED
        self._model = grb.Model(self._name)
        vars = {v.name:self._model.addVar(name=v.name, vtype=v.vtype) for v in list(self._vars.values())}
        self._model.update()
        z = grb.LinExpr()
        for v in vars:
            z += self._z[v] * vars[v]
        self._model.setObjective(expr=z, sense=const_map[self._sense])
        for name, constraint in list(self._constraints.items()):
            lhs = grb.LinExpr()
            for x in constraint.coeffs:
                lhs += constraint.coeffs[x] * vars[x]
            self._model.addConstr(lhs=lhs,
                        sense=const_map[constraint.sense],
                        rhs=constraint.rhs,
                        name=name)
        self._model.optimize()
        if allow_feasrelax and self._model.status in GUROBI_IU: # infeasible or unbounded model
            print('ws3.opt._solve_gurobi: Model infeasible, enabling feasRelaxS mode.')
            self._model.feasRelaxS(1, False, False, True)
            self._model.optimize()
        if self._model.status == grb.GRB.OPTIMAL:
            for k, v in list(self._vars.items()):
                _v = self._model.getVarByName(k)
                v._solver_var = _v # might want to poke around this later...
                v.val = _v.X

    def _solve_pulp(self):
        """
        Solve the LP problem using the pulp solver.

        Returns
        -------
        None
        """
        import pulp
        const_map = {
            SENSE_MINIMIZE:pulp.constants.LpMinimize,
            SENSE_MAXIMIZE:pulp.constants.LpMaximize,
            VTYPE_INTEGER:pulp.constants.LpInteger,
            VTYPE_BINARY:pulp.constants.LpBinary,
            VTYPE_CONTINUOUS:pulp.constants.LpContinuous,
            SENSE_EQ:pulp.constants.LpConstraintEQ,
            SENSE_GEQ:pulp.constants.LpConstraintGE,
            SENSE_LEQ:pulp.constants.LpConstraintLE}
        self._model = pulp.LpProblem(name=self._name, sense=const_map[self._sense])
        vars = pulp.LpVariable.dicts(name='',
                                     indices=self._vars.keys(),
                                     lowBound=0.,
                                     upBound=1.,
                                     cat=const_map[VTYPE_CONTINUOUS])
        obj = pulp.lpSum([self._z[v] * vars[v] for v in self._vars])
        self._model += obj, 'objective'
        for name, constraint in list(self._constraints.items()):
            lhs = pulp.lpSum([constraint.coeffs[v] * vars[v] for v in constraint.coeffs])
            if constraint.sense == SENSE_EQ:
                self._model += lhs == constraint.rhs, name
            elif constraint.sense == SENSE_GEQ:
                self._model += lhs >= constraint.rhs, name
            elif constraint.sense == SENSE_LEQ:
                self._model += lhs <= constraint.rhs, name
        # self._model.solve(solver=pulp.LpSolverDefault) # use default LP solver for now, but expland later to allow other backends
        self._model.solve(solver=pulp.PULP_CBC_CMD(msg=False)) # use default LP solver for now, but expland later to allow other backends
        if pulp.LpStatus[self._model.status] in [pulp.constants.LpStatusInfeasible, pulp.constants.LpStatusUnbounded]:
            print(f"ws3.opt._solve_pulp: Model {pulp.LpStatus[self._model.status]}")
        else:
            for k, v in list(self._vars.items()):
                self._vars[k].val = vars[k].varValue


    
    def get_all_constraints_lhs_values(self):
        """
        Returns the left-hand side (LHS) values for all constraints in the problem after solving.
        
        :return: A dictionary where keys are constraint names and values are the LHS values.
        """
        if not self.solved():
            raise ValueError("The problem has not been solved yet.")           
        lhs_values = {}
        if self._solver == SOLVER_PULP:
            import pulp
            for constraint_name, constraint in self._constraints.items():
                lhs_value = sum(constraint.coeffs[v] * self._vars[v].val for v in constraint.coeffs)
                lhs_values[constraint_name] = lhs_value
        elif self._solver == SOLVER_GUROBI:
            for constraint_name, constraint in self._constraints.items():
                lhs_value = sum(constraint.coeffs[v] * self._vars[v].val for v in constraint.coeffs)
                lhs_values[constraint_name] = lhs_value
        else:
            raise ValueError("Unsupported solver backend.")      
        return lhs_values

        

