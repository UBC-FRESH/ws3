from ws3 import common
from ws3 import opt
from concurrent.futures import ProcessPoolExecutor #, as_completed
from multiprocessing import get_context


MP_CONTEXT = "fork"

_GLOBAL_MODEL_GEN_VARS = None
_GLOBAL_COEFF_FUNCS_GEN_VARS = None
_GLOBAL_WORKERS_GEN_VARS = 1


def choose_max_batch_factor(workers):
    """
    Adaptive max_batch_factor for auto_batch based on number of workers.

    :param workers: Number of worker processes. Integer value representing the number of worker processes available.
    :type workers: int
    :return: Optimized max_batch_factor value. An integer value that represents the optimized max_batch_factor value based on the number of workers.
    :rtype: int
 
    Usage notes:

    * This function is designed to work with auto_batch, which controls batch sizes
      for parallel processing.
    * The function's output is an integer representing the optimal max_batch_factor
      value for a given number of worker processes.

    Examples:
        >>> choose_max_batch_factor(1)
        2

        >>> choose_max_batch_factor(8)
        4

    Edge case warnings:

    * If workers <= 0, this function will raise a ValueError.
    """
    if workers <= 2:
        return 2
    elif workers <= 8:
        return 4
    elif workers <= 16:
        return 8
    else:
        return 16

def auto_batch(tasks, workers, max_batch_factor = None, size_fn= lambda x: 1.):
    """Split tasks into batches for parallel processing. Optionally sorts tasks by size (descending) and greedily fills batches.

    :param tasks: List of tasks to batch
    :type tasks: list
    :param workers: Number of workers (cores) that will be used later to process batches
    :type workers: int
    :param max_batch_factor: Scaling parameter (larger value yields more smaller batches), defaults to None
    :type max_batch_factor: int, optional
    :param size_fn: Task size estimation function returning float for greedy task sort, defaults to `lambda x: 1.`
    :type size_fn: function, optional
    :return: List of task batches
    :rtype: list[list]
    """    

    if not tasks:
        return []

    if max_batch_factor is None:
        max_batch_factor = choose_max_batch_factor(workers)

    target_batches = max(1, workers * max_batch_factor)
    batch_size = max(1, len(tasks) // target_batches)

    # Default size function if not given
    if size_fn is None:
        size_fn = lambda x: 1

    # Sort tasks by size (descending)
    sized_tasks = sorted(tasks, key=size_fn, reverse=True)

    # Initialize batches and their current total size
    batches = [[] for _ in range(target_batches)]
    batch_loads = [0] * target_batches

    # Greedy fill: always append to the lightest batch
    for task in sized_tasks:
        idx = batch_loads.index(min(batch_loads))
        batches[idx].append(task)
        batch_loads[idx] += size_fn(task)

    # Remove empty batches (if tasks < batches)
    batches = [b for b in batches if b]

    # Optionally further split overly large batches if needed
    final_batches = []
    for batch in batches:
        if len(batch) > batch_size * 2:  # prevent one batch from being huge
            for i in range(0, len(batch), batch_size):
                final_batches.append(batch[i:i + batch_size])
        else:
            final_batches.append(batch)

    return final_batches

def worker_summarize_tree_batch(args):
    """Summarize a batch of trees into coverage constraints and leaf outputs.

    :param args: [batch, z_coeff_key]
    :type args: list
    :return: [(cname, coeffs, z_coeffs), ...]
    :rtype: list[list]
    """    
    batch, z_coeff_key = args
    results = []
    for i, tree in batch:
        cname = f'cov_{common.hex_id(i)}'
        coeffs = {}
        z_coeffs = {}
        for path in tree.paths():
            j = tuple(n.data('acode') for n in path)
            leaf_id = path[-1].data('leaf_id')
            vname = f"x_{leaf_id}"
            coeffs[vname] = 1.0
            z_coeffs[vname] = path[-1].data(z_coeff_key)
        results.append((cname, coeffs, z_coeffs))
    return results

def sanitize_func(f):
    """Make a version of f that is safe to serialize via dill in `spawn` mode
    
    :param f: Function to sanitize
    :type f: function
    :return: Sanitized function
    :rtype: function
    """
    import functools
    import types
    if isinstance(f, functools.partial):
        return functools.partial(sanitize_func(f.func), *f.args, **(f.keywords or {}))
    if isinstance(f, types.FunctionType):
        new_f = types.FunctionType(
            f.__code__,
            {},  # empty globals dict — no module context
            name=f.__name__,
            argdefs=f.__defaults__,
            closure=f.__closure__,
        )
        new_f.__module__ = '__main__'
        return new_f
    raise TypeError(f"Don't know how to sanitize function of type {type(f)}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

def init_worker_gen_vars(blob_bytes_local, serialized_funcs_local, workers=1):
    """Initializer for `_gen_vars_m1` workers: load model and coefficient functions once.
    Also stores desired worker count for `_bld_tree_m1`.

    :param blob_bytes_local: Serialized `ForestModel` object
    :type blob_bytes_local: bytes
    :param serialized_funcs_local: dict of serialized functions keyed on `coeff_funcs` keys 
    :type serialized_funcs_local: dict[str, bytes]
    :param workers: Number of workers, defaults to 1
    :type workers: int, optional
    """    
    global _GLOBAL_MODEL_GEN_VARS, _GLOBAL_COEFF_FUNCS_GEN_VARS, _GLOBAL_WORKERS_GEN_VARS
    import dill
    _GLOBAL_MODEL_GEN_VARS = dill.loads(blob_bytes_local)
    _GLOBAL_COEFF_FUNCS_GEN_VARS = {k: dill.loads(f_bytes) for k, f_bytes in serialized_funcs_local.items()}
    _GLOBAL_WORKERS_GEN_VARS = workers

def worker_gen_vars(tasks, acodes):
    """    Worker for building trees in `_gen_vars_m1`.

    :param tasks: list of (dtk, age) tuples to process
    :type tasks: list[(str, ...), int)]
    :param acodes: list of action codes to use when building trees
    :type acodes: list[str]
    :return: list of (dtk, age, tree) tuples
    :rtype: list[(str, ...), int, Tree]
    """
    model = _GLOBAL_MODEL_GEN_VARS
    coeff_funcs = _GLOBAL_COEFF_FUNCS_GEN_VARS
    workers = _GLOBAL_WORKERS_GEN_VARS
    
    results = []
    for (dtk, age) in tasks: 
        model.reset()
        area = model.dtypes[dtk].area(1, age)
        if not area: continue
        tree = model._bld_tree_m1(
            area, dtk, age, coeff_funcs,
            tree=None, period=1,
            acodes=acodes, compile_c_ycomps=True)
        results.append((dtk, age, tree))
    return results

# ----------------------------
# Globals for _cmp_cflw_m1 parallel execution
# ----------------------------

def worker_cmp_cflw_batch(args):
    """Worker function to process batches of tasks for `_cmp_cflw_m1`

    :param args: (batch, cflw_keys, periods)
    :type args: list[list, dict, list]
    :return: list of (t, o, i, j, value) tuples
    :rtype: list[int, str, tuple, tuple, float]
    """    
    batch, cflw_keys, periods = args
    results = []
    for i, tree in batch:
        for path in tree.paths():
            j = tuple(n.data('acode') for n in path)
            for o in cflw_keys:
                _mu = path[-1].data(o)
                for t in periods:
                    results.append((t, o, i, j, _mu.get(t, 0.0)))
    return results

def worker_cmp_cflw_phase3(args):
    """ Worker function to compute (name, coeffs, sense, rhs) tuples for Phase 3 of `_cmp_cflw_m1`.

    :param args: (t, o, mu_t_o, mu_ref_o, eps, xnames)
    :type args: tuple(int, str, float, float, float, list[str])
    :return: list of (constraint_name, mu_lb, sense, 0.) tuples
    :rtype: list[(str, float, str, float)]
    """    
    t, o, mu_t_o, mu_ref_o, eps, xnames = args
    results = []

    keys = list(mu_t_o.keys())
    x_keys = [xnames[k] for k in keys]
    mu_vals = [mu_t_o[k] for k in keys]
    mu_ref = [mu_ref_o[k] for k in keys]

    # Lower bound row
    mu_lb_vals = [v - (1 - eps) * r for v, r in zip(mu_vals, mu_ref)]
    mu_lb = dict(zip(x_keys, mu_lb_vals))
    results.append((f'flw-lb_{t:03d}_{o}', mu_lb, opt.SENSE_GEQ, 0.0))

    # Upper bound row
    mu_ub_vals = [v - (1 + eps) * r for v, r in zip(mu_vals, mu_ref)]
    mu_ub = dict(zip(x_keys, mu_ub_vals))
    results.append((f'flw-ub_{t:03d}_{o}', mu_ub, opt.SENSE_LEQ, 0.0))

    return results

def worker_cmp_cflw_phase3_batch(batch):
    """Worker function to process batches of phase 3 tasks for `_cmp_cflw_m1`

    :param batch: list of tasks (tuples)
    :type batch: list[tuple]
    :return: list of results
    :rtype: list[tuple]
    """    
    batch_results = []
    for task in batch:
        batch_results.extend(worker_cmp_cflw_phase3(task))
    return batch_results

# ----------------------------
# Globals for _cmp_cgen_m1 parallel execution
# ----------------------------

def worker_cmp_cgen_batch(args):
    """Worker function to process batches of tasks for `_cmp_cgen_m1`

    :param args: (batch, cgen_keys, periods)
    :type args: list[list, dict, list]
    :return: list of (t, o, i, j, value) tuples
    :rtype: list[int, str, tuple, tuple, float]
    """    
    batch, cgen_keys, periods = args
    results = []
    for i, tree in batch:
        for path in tree.paths():
            j = tuple(n.data('acode') for n in path)
            leaf = path[-1]
            for o in cgen_keys:
                _mu = leaf.data(o)  # dict {period: value}
                for t in periods:
                    results.append((t, o, i, j, _mu.get(t, 0.0)))
    return results

def worker_cmp_cgen_phase3(args):
    """
    Args: (t, o, mu_t_o, lb, ub)
    Returns: [(name, coeffs, sense, rhs), ...]
    """
    t, o, mu_t_o, lb, ub = args

    # Build coeffs exactly like the known-good path:
    # NOTE: keys in mu_t_o are (i, j)
    coeffs = {'x_%s' % common.hex_id(k): v for k, v in mu_t_o.items()}

    res = []
    if lb is not None and t in lb:
        res.append((f'gen-lb_{t:03d}_{o}', coeffs, opt.SENSE_GEQ, lb[t]))
    if ub is not None and t in ub:
        res.append((f'gen-ub_{t:03d}_{o}', coeffs, opt.SENSE_LEQ, ub[t]))
    return res

def worker_cmp_cgen_phase3_batch(batch):
    """Process a batch of Phase 3 CGEN tasks."""
    out = []
    for task in batch:
        out.extend(worker_cmp_cgen_phase3(task))
    return out

class PersistentWorkerPool:
    """
    Context manager for a persistent ProcessPoolExecutor that initializes
    workers with ForestModel and coeff_funcs.
    """

    def __init__(self, workers, blob_bytes=None, serialized_funcs=None):
        """Constructor

        :param workers: Number of workers
        :type workers: int
        :param blob_bytes: Serialized `ForestModel` objects, defaults to None
        :type blob_bytes: bytes, optional
        :param serialized_funcs: dict of serialzed `coeff_funcs` functions, defaults to None
        :type serialized_funcs: dict[str, function], optional
        """        
        self.workers = workers
        self.blob_bytes = blob_bytes
        self.serialized_funcs = serialized_funcs
        self.executor = None

    def __enter__(self):
        """Create persistent worker pool executor

        :return: 
        :rtype: ProcessPoolExecutor
        """        
        if self.workers > 1:
            ctx = get_context(MP_CONTEXT)
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=ctx,
                initializer=init_worker_gen_vars,
                initargs=(self.blob_bytes, self.serialized_funcs, self.workers),
            )
        return self.executor

    def __exit__(self, exc_type, exc_value, traceback):
        """Shut down persisten pool executor when with block exits"""        
        if self.executor is not None:
            self.executor.shutdown()