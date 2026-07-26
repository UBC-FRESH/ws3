###################################################################################
# MIT License

# Copyright (c) 2015-2025 Gregory Paradis

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

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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
    name: str
    vtype: str
    lb: float
    ub: float
    val: Optional[float]
    _solver_var: Any

    def __init__(self, name: str, vtype: str, lb: float = 0., ub: float = VBNDS_INF, val: Optional[float] = None) -> None:
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
    name: str
    coeffs: Dict[str, float]
    sense: str
    rhs: float

    def __init__(self, name: str, coeffs: Dict[str, float], sense: str, rhs: float) -> None:
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
    _name: str
    _vars: Dict[str, Variable]
    _z: Dict[str, float]
    _constraints: Dict[str, Constraint]
    _sense: int
    _solver: str
    _solver_backend: Optional[str]
    _solution: Optional[Dict[str, float]]
    _warm_start: Optional[List[float]]
    _model: Any
    _dispatch_map: Dict[str, Callable[..., None]]

    def __init__(self, name: str, sense: int = SENSE_MAXIMIZE, solver: str = SOLVER_DEFAULT) -> None:
        self._name = name
        self._vars: Dict[str, Variable] = {}
        self._z: Dict[str, float] = {}
        self._constraints: Dict[str, Constraint] = {}
        self._sense = sense
        self._solver = solver
        self._solver_backend = None
        self._solution = None
        self._warm_start = None
        self._model = None
        self._dispatch_map = {
            SOLVER_PULP: self._solve_pulp,
            SOLVER_GUROBI: self._solve_gurobi,
            SOLVER_HIGHS: self._solve_highs
        }

    def merge(self, problem: Problem) -> None:
        """
        Merge problem with data from another problem. 
        
        :param problem: The problem to be merged with this one.
        """
        self._vars.update(problem._vars)
        self._z.update(problem._z)
        self._constraints.update(problem._constraints)

    def add_var(self, name: str, vtype: str, lb: float = 0., ub: float = VBNDS_INF) -> None:
        """
        The function adds a variable to the problem.
    
        :param name: The variable name that needs to be unique within the problem (user is responsible for enforcing this condition) type.
        :param vtype: The variable type that has to be one of ``VTYPE_CONTINUOUS``, ``VTYPE_INTEGER``, or ``VTYPE_BINARY``.
        :param lb: The lower bound value for the variable (Default is zero).
        :param ub: The upper bound value for the variable (Default is positive infinity).
        """
        self._vars[name] = Variable(name, vtype, lb, ub)
        self._solution = None # modifying problem kills solution

    def var_names(self) -> List[str]:
        """
        Return a list of variable names.
        """
        return list(self._vars.keys())

    def constraint_names(self) -> List[str]:
        """
        Returns a list of constraint names.
        """
        return list(self._constraints.keys())

    def name(self) -> str:
        """
        Returns problem name.
        """
        return self._name

    def var(self, name: str) -> Variable:
        """
        Returns a :py:class:`ws3.opt.Variable` object, given a variable name.

        :param name: Variable name.
        :return: Variable object
        :rtype: :py:class:`ws3.opt.Variable`

        """
        return self._vars[name]

    def sense(self, val: Optional[int] = None) -> Optional[int]:
        """
        Returns (or sets) objective function sense.
        :param val: Value should be one of ``SENSE_MINIMIZE`` or ``SENSE_MAXIMIZE``.
        """
        if val is not None:
            self._sense = val
            self._solution = None # modifying problem kills solution
            return None
        return self._sense

    def solved(self) -> bool:
        """
        Returns ``True`` if the problem has been solved, ``False`` otherwise.
        """
        return self._solution is not None

    def z(self, coeffs: Optional[Dict[str, float]] = None, validate: bool = False) -> Optional[float]:
        """
        Returns the objective function value if ``coeffs`` is not provided (triggers an exception if problem has not been solved yet), or updates the objective function coefficient vector (resets the value of the optimal solution to ``None``).
        """
        if coeffs is not None:
            if validate:
                for v in coeffs:
                    assert v in self._vars
            self._z = coeffs
            self._solution = None # modifying problem kills solution
            return None
        assert self.solved()
        assert self._solution is not None
        return sum([self._z[v] * self._solution[v] for v in list(self._vars.keys())])

    def add_constraint(self, name: str, coeffs: Dict[str, float], sense: str, rhs: float, validate: bool = False) -> None:
        """
        This function adds a constraint to the problem.
    
        :param name: The constraint name should be unique within the problem (user is responsible for enforcing this condition).
        :param coeffs: Constraint coeffients should be provided as a ``dict``, keyed on variable names---length of constraint coefficient ``dict`` should match number of variables in the problem (user is responsible for enforcing this condition).
        :param sense: Constraint sense should be one of ``SENSE_EQ``, ``SENSE_GEQ``, or ``SENSE_LEQ``.
        :param rhs: The right hand side of the constraint.

        Note that calling this method resets the value of the optimal solution to ``None``
    
        """
        if validate:
            for v in coeffs:
                assert v in self._vars
        self._constraints[name] = Constraint(name, coeffs, sense, rhs)
        self._solution = None # modifying problem kills solution

    def solver(self, val: Optional[str] = None) -> Optional[str]:
        """
        Sets the solver backend (defaults to ``SOLVER_PULP`` in the class constructor). 
        
        Use ``SOLVER_GUROBI`` to use Gurobi solver bindings.
        """
        if val is not None:
            self._solver = val
            return None
        return self._solver

    def solution(self) -> Optional[Dict[str, float]]:
        """
        Returns a ``dict`` of variable values, keyed on variable names.
        """
        return self._solution

    def solve(self, validate: bool = False, threads: int = 0, warm_start: Optional[List[float]] = None, verbose: bool = False) -> None:
        """
        Solve the optimization problem.

        :param bool validate: If True, performs pre-solve checks (not implemented).
        :param int threads: Number of solver threads (0 = auto).
        :param list[float] or None warm_start: Optional initial solution vector (in column order) to warm start the solver.

        :return None: Solution is stored in self._solution if optimal.
        """
        if validate:
            assert False, "Validation not implemented yet"

        # Store warm start for use by solver
        self._warm_start = warm_start

        # Dispatch to solver-specific method
        self._dispatch_map[self._solver].__get__(self, type(self))(threads=threads, verbose=verbose)

        # Capture solution if optimal
        if self.status() == STATUS_OPTIMAL:
            assert self._solution is not None
            self._solution = {x: (self._vars[x].val or 0.0) for x in self._vars}
                
    def status(self):
        """
        Checks the solution status of the current model for PuLP, Gurobi, or HiGHS (highspy).

        :returns:  STATUS_INFEASIBLE, STATUS_UNBOUNDED, STATUS_OPTIMAL, or None
        """
        import ws3.opt

        # Optional imports
        try:
            import pulp
        except ImportError:
            pulp = None

        try:
            import gurobipy
        except ImportError:
            gurobipy = None

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
    
    def get_all_constraints_lhs_values(self):
        """
        Returns the left-hand side (LHS) values for all constraints in the problem after solving.
        
        :return: A dictionary where keys are constraint names and values are the LHS values.
        """
        if not self.solved():
            raise ValueError("The problem has not been solved yet.")           
        lhs_values = {}
        if self._solver == SOLVER_PULP:
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

    def _solve_gurobi(self, allow_feasrelax=True, threads=1, verbose=False):
        import gurobipy as grb

        # If you build a custom Env, you must pass it to Model OR set params on the model.
        env = grb.Env(empty=True)
        env.setParam("Threads", threads)
        env.setParam("OutputFlag", int(verbose))
        env.start()

        const_map = {
            SENSE_MINIMIZE: grb.GRB.MINIMIZE,
            SENSE_MAXIMIZE: grb.GRB.MAXIMIZE,
            VTYPE_INTEGER:  grb.GRB.INTEGER,
            VTYPE_BINARY:   grb.GRB.BINARY,
            VTYPE_CONTINUOUS: grb.GRB.CONTINUOUS,
            SENSE_EQ:  grb.GRB.EQUAL,
            SENSE_GEQ: grb.GRB.GREATER_EQUAL,
            SENSE_LEQ: grb.GRB.LESS_EQUAL,
        }

        # attach env so the Threads param actually applies
        self._model = grb.Model(self._name, env=env)

        vars = {v.name: self._model.addVar(name=v.name, vtype=v.vtype)
                for v in self._vars.values()}
        self._model.update()

        z = grb.LinExpr()
        for vname in vars:
            z += self._z[vname] * vars[vname]
        self._model.setObjective(z, sense=const_map[self._sense])

        for name, constraint in self._constraints.items():
            lhs = grb.LinExpr()
            for x in constraint.coeffs:
                lhs += constraint.coeffs[x] * vars[x]

            # NEW: use operator overloads instead of sense/rhs keywords
            if constraint.sense == SENSE_EQ:
                self._model.addConstr(lhs == constraint.rhs, name=name)
            elif constraint.sense == SENSE_LEQ:
                self._model.addConstr(lhs <= constraint.rhs, name=name)
            elif constraint.sense == SENSE_GEQ:
                self._model.addConstr(lhs >= constraint.rhs, name=name)
            else:
                raise ValueError(f"Unknown sense {constraint.sense}")

        self._model.optimize()

        # Use the modern status enum casing
        GUROBI_IU = (grb.GRB.Status.INF_OR_UNBD, grb.GRB.Status.INFEASIBLE, grb.GRB.Status.UNBOUNDED)

        if allow_feasrelax and self._model.Status in GUROBI_IU:
            print('ws3.opt._solve_gurobi: Model infeasible, enabling feasRelaxS mode.')
            # relaxobjtype=1 (squared violations), minrelax=False, vrelax=False, crelax=True
            self._model.feasRelaxS(1, False, False, True)
            self._model.optimize()

        if self._model.Status == grb.GRB.Status.OPTIMAL:
            for k, v in self._vars.items():
                _v = self._model.getVarByName(k)
                v._solver_var = _v
                v.val = _v.X
                
    def _solve_pulp(self, threads=0, verbose=False, solver_backend="CBC"):
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
            SENSE_LEQ:pulp.constants.LpConstraintLE
        }
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
        if solver_backend == "CBC":
            self._model.solve(solver=pulp.PULP_CBC_CMD(msg=verbose, threads=0))
        elif solver_backend == "HiGHS":
            self._model.solve(solver=pulp.HiGHS(msg=verbose, threads=0, solver="pdlp"))
        else:
            print("Solver backend not supported:", solver_backend)
            raise ValueError
        if pulp.LpStatus[self._model.status] in [pulp.constants.LpStatusInfeasible, pulp.constants.LpStatusUnbounded]:
            print(f"ws3.opt._solve_pulp: Model {pulp.LpStatus[self._model.status]}")
        else:
            for k, v in list(self._vars.items()):
                self._vars[k].val = vars[k].varValue

    def _solve_highs(self, threads=0, simplex_strategy=2, verbose=False):
        """
        Solve the current LP using HiGHS in the same way PuLP does in its buildSolverModel:
        - Uses addCol() and addRow() for each variable and constraint
        - Handles bounds and objective signs like PuLP
        - Deduplicates variable coefficients per constraint
        - Optionally applies a warm start solution vector

        Parameters
        ----------
        threads : int
            Number of threads for HiGHS. 0 = auto-detect.
        simplex_strategy : int
            HiGHS simplex strategy. 2 = parallel dual simplex (recommended for multi-core).
        
        Returns
        -------
        status : highspy.HighsStatus
            HiGHS solver status.
        """
        from collections import defaultdict

        import highspy
        import numpy as np

        highs = highspy.Highs()
        highs.resetGlobalScheduler(True) 
        inf = highspy.kHighsInf

        # ----------------------------
        # Solver options
        # ----------------------------
        highs.setOptionValue("threads", threads)
        highs.setOptionValue("solver", "simplex")
        highs.setOptionValue("simplex_strategy", simplex_strategy)
        highs.setOptionValue("simplex_min_concurrency", 2)
        highs.setOptionValue("simplex_max_concurrency", 8)
        highs.setOptionValue("output_flag", verbose)

        # ----------------------------
        # Variables
        # ----------------------------
        obj_mult = -1 if self._sense == SENSE_MAXIMIZE else 1
        var_index = {}

        for i, (vname, var) in enumerate(self._vars.items()):
            lb = var.lb if var.lb is not None else -inf
            ub = var.ub if var.ub is not None else inf
            obj_coef = obj_mult * self._z.get(vname, 0.0)

            highs.addCol(obj_coef, lb, ub, 0, [], [])
            var_index[vname] = i
            var.index = i

        # ----------------------------
        # Constraints
        # ----------------------------
        for cname, con in self._constraints.items():
            # Compute row bounds
            if con.sense == SENSE_EQ:
                lb, ub = con.rhs, con.rhs
            elif con.sense == SENSE_LEQ:
                lb, ub = -inf, con.rhs
            elif con.sense == SENSE_GEQ:
                lb, ub = con.rhs, inf
            else:
                raise ValueError(f"Unknown sense {con.sense}")

            # Deduplicate coefficients
            coeff_accum = defaultdict(float)
            for vname, coef in con.coeffs.items():
                coeff_accum[var_index[vname]] += coef
            coeff_accum = {j: c for j, c in coeff_accum.items() if c != 0.0}

            indices, coefs = zip(*coeff_accum.items()) if coeff_accum else ([], [])
            highs.addRow(lb, ub, len(indices), indices, coefs)

        # ----------------------------
        # Warm start (if provided)
        # ----------------------------
        if getattr(self, "_warm_start", None) is not None:
            print('ws3.opt.Proble._solve_highs: detected _warm_start solution')
            highs.setOptionValue("run_crossover", "choose")  # let HiGHS auto-decide            
            warm_start = self._warm_start
            ncols = len(warm_start)
            idx = np.arange(ncols, dtype=np.int32)
            highs.setSolution(ncols, idx, warm_start)

        # ----------------------------
        # Solve
        # ----------------------------
        status = highs.run()
        self._model = highs

        # ----------------------------
        # Store solution
        # ----------------------------
        if status == highspy.HighsStatus.kOk:
            sol = highs.getSolution()
            col_values = sol.col_value
            for i, var in enumerate(self._vars.values()):
                var.val = col_values[i]
            self._solution = {x: (self._vars[x].val or 0.0) for x in self._vars}
        else:
            for var in self._vars.values():
                var.val = None
            self._solution = None

        return status
