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
SOLVER_HIGHS = 'highs'
SOLVER_DEFAULT = SOLVER_HIGHS
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
    def __init__(self, name, sense=SENSE_MAXIMIZE, solver=SOLVER_DEFAULT):
        self._name = name
        self._vars = {}
        self._z = {}
        self._constraints = {}
        #self._solution = None
        self._sense = sense
        self._solver = solver
        self._solver_backend = None
        self._dispatch_map = {SOLVER_PULP:self._solve_pulp, 
                              SOLVER_GUROBI:self._solve_gurobi,
                              SOLVER_HIGHS:self._solve_highs}

    def merge(self, problem):
        """
        Merge problem with data from another problem. 
        
        :param Problem: The problem to be merged with this one.
        """
        self._vars.update(problem._vars)
        self._z.update(problem._z)
        self._constraints.update(problem._constraints)


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
        Checks the solution status of the current model for PuLP, Gurobi, or HiGHS (highspy).

        Returns
        -------
        STATUS_INFEASIBLE, STATUS_UNBOUNDED, STATUS_OPTIMAL, or None
        """
        import ws3.opt
        import pulp

        # Optional import: only if Gurobi used
        try:
            import gurobipy
        except ImportError:
            gurobipy = None

        # Optional import: only if HiGHS used
        try:
            import highspy
        except ImportError:
            highspy = None

        match self._solver:
            # --- PuLP Solver ---
            case ws3.opt.SOLVER_PULP:
                match self._model.status:
                    case pulp.constants.LpStatusInfeasible:
                        return STATUS_INFEASIBLE
                    case pulp.constants.LpStatusUnbounded:
                        return STATUS_UNBOUNDED
                    case pulp.constants.LpStatusOptimal:
                        return STATUS_OPTIMAL

            # --- Gurobi Solver ---
            case ws3.opt.SOLVER_GUROBI:
                if gurobipy is None:
                    return None
                match self._model.status:
                    case gurobipy.GRB.INFEASIBLE:
                        return STATUS_INFEASIBLE
                    case gurobipy.GRB.UNBOUNDED:
                        return STATUS_UNBOUNDED
                    case gurobipy.GRB.OPTIMAL:
                        return STATUS_OPTIMAL

            # --- HiGHS Solver (direct highspy) ---
            case ws3.opt.SOLVER_HIGHS:
                if highspy is None:
                    return None
                # HiGHS uses integer codes:
                # 7 = Optimal, 8 = Infeasible, 9 = Unbounded (see HiGHSStatus)
                highs_status = self._model.getModelStatus()  # integer code
                if highs_status == highspy.HighsModelStatus.kOptimal:
                    return STATUS_OPTIMAL
                elif highs_status == highspy.HighsModelStatus.kInfeasible:
                    return STATUS_INFEASIBLE
                elif highs_status == highspy.HighsModelStatus.kUnbounded:
                    return STATUS_UNBOUNDED

            # --- Fallback ---
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
        #self._model.solve(solver=pulp.PULP_CBC_CMD(msg=False, threads=64)) # use default LP solver for now, but expland later to allow other backends
        #self._model.solve(solver=pulp.HiGHS(msg=True, threads=64, solver="pdlp")) # use default LP solver for now, but expland later to allow other backends
        self._model.solve(solver=pulp.HiGHS(msg=True, 
                                            threads=0, 
                                            solver="simplex",
                                            simplex_strategy=2,
                                            simplex_min_concurrency=8,
                                            simplex_max_concurrency=8)) # use default LP solver for now, but expland later to allow other backends
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

    def ___solve_highs(self):
        """
        Solve the current LP using highspy (continuous variables only, no integers).
        Automatically merges duplicate variable entries in constraints to match PuLP behavior.
        """
        import numpy as np
        import highspy
        from collections import defaultdict

        highs = highspy.Highs()

        # ----------------------------
        # 1. Prepare variable arrays
        # ----------------------------
        var_names = list(self._vars.keys())
        var_index = {name: i for i, name in enumerate(var_names)}
        num_vars = len(var_names)

        # Bounds and objective
        lb = np.array([v.lb for v in self._vars.values()], dtype=np.float64)
        ub = np.array([v.ub for v in self._vars.values()], dtype=np.float64)
        obj = np.array([self._z[v.name] for v in self._vars.values()], dtype=np.float64)

        # Add variables and objective
        highs.addVars(num_vars, lb, ub)
        highs.changeColsCost(num_vars, np.arange(num_vars, dtype=np.int32), obj)

        # ----------------------------
        # 2. Prepare constraints (deduplicated)
        # ----------------------------
        num_constraints = len(self._constraints)
        if num_constraints > 0:
            row_lower = np.empty(num_constraints, dtype=np.float64)
            row_upper = np.empty(num_constraints, dtype=np.float64)

            row_idx = []
            col_idx = []
            data = []

            for i, cname in enumerate(self._constraints):
                con = self._constraints[cname]
                sense = con.sense
                rhs = con.rhs

                # Row bounds
                if sense == SENSE_EQ:
                    row_lower[i] = rhs
                    row_upper[i] = rhs
                elif sense == SENSE_LEQ:
                    row_lower[i] = -highspy.kHighsInf
                    row_upper[i] = rhs
                elif sense == SENSE_GEQ:
                    row_lower[i] = rhs
                    row_upper[i] = highspy.kHighsInf
                else:
                    raise ValueError(f"Unknown sense {sense}")

                # Merge duplicate variable entries in this constraint
                coeff_accum = defaultdict(float)
                for vname, coef in con.coeffs.items():
                    j = var_index[vname]
                    coeff_accum[j] += coef

                # Add merged entries to matrix
                for j, coef in coeff_accum.items():
                    row_idx.append(i)
                    col_idx.append(j)
                    data.append(coef)

            # Debug: print matrix dimensions
            print(f"Adding {num_constraints} rows and {len(data)} nonzeros to HiGHS")

            highs.addRows(num_constraints,
                        row_lower, row_upper,
                        len(data),
                        np.array(row_idx, dtype=np.int32),
                        np.array(col_idx, dtype=np.int32),
                        np.array(data, dtype=np.float64))

        # ----------------------------
        # 3. Set objective sense (use correct enum)
        # ----------------------------
        if self._sense == SENSE_MINIMIZE:
            highs.changeObjectiveSense(highspy.ObjSense.kMinimize)
        else:
            highs.changeObjectiveSense(highspy.ObjSense.kMaximize)

        # ----------------------------
        # 4. Solve
        # ----------------------------
        status = highs.run()
        self._model = highs  # for status()

        # ----------------------------
        # 5. Store results
        # ----------------------------
        if status == highspy.HighsStatus.kOk:
            sol = highs.getSolution()
            self._solution = {}
            for i, (vname, var) in enumerate(self._vars.items()):
                var.val = sol.col_value[i]
                self._vars[vname].val = varsol.col_value[i]
        else:
            print(f"ws3.opt._solve_highs: not OK")

        return status

    def __solve_highs(self, debug_limit=10_000_000):
        """
        Solve the current LP using the real Highs Python class.
        Processes constraints incrementally to avoid large upfront allocations.
        Continuous variables only.
        """
        import numpy as np
        import highspy
        from collections import defaultdict

        highs = highspy.Highs()

        # ----------------------------
        # 1. Variables
        # ----------------------------
        var_names = list(self._vars.keys())
        num_vars = len(var_names)
        print(f"Number of variables: {num_vars}")

        col_costs = [self._z[v.name] for v in self._vars.values()]
        col_lower = [v.lb for v in self._vars.values()]
        col_upper = [v.ub for v in self._vars.values()]
        print()
        highs.addVariables(col_lower, col_upper, col_costs, var_names)
        print('vars done')

        # ----------------------------
        # 2. Incremental constraint assembly
        # ----------------------------
        row_lower = []
        row_upper = []
        starts = [0]
        all_vars = []
        all_coeffs = []

        var_index = {name: i for i, name in enumerate(var_names)}

        for idx, (cname, con) in enumerate(self._constraints.items()):
            # Row bounds
            if con.sense == SENSE_EQ:
                row_lower.append(con.rhs)
                row_upper.append(con.rhs)
            elif con.sense == SENSE_LEQ:
                row_lower.append(-highspy.inf())
                row_upper.append(con.rhs)
            elif con.sense == SENSE_GEQ:
                row_lower.append(con.rhs)
                row_upper.append(highspy.inf())
            else:
                raise ValueError(f"Unknown sense {con.sense}")
            print('con rhs defined', cname)

            # Deduplicate coefficients per row
            coeff_accum = defaultdict(float)
            for vname, coef in con.coeffs.items():
                coeff_accum[var_index[vname]] += coef
            print('coeff accum done')

            # Append to CSR arrays
            for j, coef in coeff_accum.items():
                all_vars.append(j)
                all_coeffs.append(coef)

            starts.append(len(all_vars))

            # ---- Incremental debug ----
            if idx % 100 == 0:
                print(f"Constraint {idx}, row_nnz={len(coeff_accum)}, total_nnz={len(all_vars)}")
            if len(all_vars) > debug_limit:
                print(f"ABORT: Nonzeros exceeded {debug_limit}, stopping early.")
                return None

        num_constraints = len(row_lower)
        nnz = len(all_coeffs)
        print(f"Assembly complete: rows={num_constraints}, cols={num_vars}, nnz={nnz}")

        # ----------------------------
        # 3. Convert to NumPy arrays
        # ----------------------------
        starts = np.array(starts, dtype=np.int32)
        all_vars = np.array(all_vars, dtype=np.int32)
        all_coeffs = np.array(all_coeffs, dtype=np.float64)
        row_lower = np.array(row_lower, dtype=np.float64)
        row_upper = np.array(row_upper, dtype=np.float64)

        # CSR integrity check
        if starts[-1] != nnz:
            raise RuntimeError(f"CSR structure invalid: starts[-1]={starts[-1]} != nnz={nnz}")

        # ----------------------------
        # 4. Pass constraints
        # ----------------------------
        highs.addConstrs(row_lower, row_upper, starts, all_vars, all_coeffs)

        # ----------------------------
        # 5. Objective sense
        # ----------------------------
        if self._sense == SENSE_MINIMIZE:
            highs.setMinimize()
        else:
            highs.setMaximize()

        # ----------------------------
        # 6. Solve
        # ----------------------------
        status = highs.solve()
        self._model = highs

        # ----------------------------
        # 7. Store results
        # ----------------------------
        if status == highspy.HighsStatus.kOk:
            sol = highs.getSolution()
            self._solution = {}
            for i, (vname, var) in enumerate(self._vars.items()):
                var.val = sol.col_value[i]
                self._solution[vname] = sol.col_value[i]
        else:
            self._solution = {vname: None for vname in self._vars}

        return status

    def _solve_highs(self):
        """
        Solve the current LP using HiGHS in the same way PuLP does in its buildSolverModel:
        - Uses addCol() and addRow() for each variable and constraint
        - Handles bounds and objective signs like PuLP
        - Deduplicates variable coefficients per constraint
        """
        import highspy
        import numpy as np
        from collections import defaultdict

        highs = highspy.Highs()
        inf = highspy.kHighsInf

        # ----------------------------
        # 1. Variables
        # ----------------------------
        obj_mult = -1 if self._sense == SENSE_MAXIMIZE else 1
        var_names = list(self._vars.keys())
        var_index = {}

        for i, (vname, var) in enumerate(self._vars.items()):
            lb = var.lb if var.lb is not None else -inf
            ub = var.ub if var.ub is not None else inf

            obj_coef = obj_mult * self._z.get(vname, 0.0)

            # No column entries at add time (we'll add rows separately)
            highs.addCol(obj_coef, lb, ub, 0, [], [])
            var_index[vname] = i
            var.index = i  # Optional: for debugging or later reference

        # ----------------------------
        # 2. Constraints
        # ----------------------------
        for i, (cname, con) in enumerate(self._constraints.items()):
            # Compute bounds for the row
            if con.sense == SENSE_EQ:
                lb = con.rhs
                ub = con.rhs
            elif con.sense == SENSE_LEQ:
                lb = -inf
                ub = con.rhs
            elif con.sense == SENSE_GEQ:
                lb = con.rhs
                ub = inf
            else:
                raise ValueError(f"Unknown sense {con.sense}")

            # Deduplicate coefficients per row
            coeff_accum = defaultdict(float)
            for vname, coef in con.coeffs.items():
                coeff_accum[var_index[vname]] += coef

            # Remove explicit zeros
            coeff_accum = {j: c for j, c in coeff_accum.items() if c != 0.0}

            if len(coeff_accum) == 0:
                indices, coefs = [], []
            else:
                indices, coefs = zip(*coeff_accum.items())

            # Add the row to HiGHS
            highs.addRow(lb, ub, len(indices), indices, coefs)

        # ----------------------------
        # 3. Solve
        # ----------------------------
        status = highs.run()
        self._model = highs  # store for status checks

        # ----------------------------
        # 4. Store solution
        # ----------------------------
        if status == highspy.HighsStatus.kOk:
            sol = highs.getSolution()
            col_values = sol.col_value
            for i, var in enumerate(self._vars.values()):
                var.val = col_values[i]
        else:
            for var in self._vars.values():
                var.val = None
                
        return status