
MP_CONTEXT = "fork"
_global_model_gen_vars = None
_global_coeff_funcs_gen_vars = None
_global_workers_gen_vars = 1


def choose_max_batch_factor(workers: int) -> int:
    """
    Adaptive max_batch_factor for auto_batch based on number of workers.
    Keeps IPC overhead low for small core counts while ensuring enough batches
    for parallel saturation on large core counts.
    """
    if workers <= 2:
        return 2
    elif workers <= 8:
        return 4
    elif workers <= 16:
        return 8
    else:
        return 16

def auto_batch(tasks, workers, max_batch_factor=None, size_fn=None):
    """
    Split tasks into batches for parallel processing.
    - Preserves your adaptive batch size logic.
    - Optionally sorts tasks by size (descending) and greedily fills batches.

    Args:
        tasks: List of tasks (any type)
        workers: Number of process pool workers
        max_batch_factor: Multiplier for target number of batches (~workers*factor)
        size_fn: Optional function returning a numeric cost per task (default=1)

    Returns:
        List of task batches (list of lists)
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
    """
    Summarize a batch of trees into coverage constraints and leaf outputs.
    Returns: [(cname, coeffs, z_coeffs), ...]
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
    """Make a version of f that is safe to serialize via dill in 'spawn' mode"""
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
    """
    Initializer for _gen_vars_m1 workers: load model and coefficient functions once.
    Also stores desired worker count for _bld_tree_m1.
    """
    global _global_model_gen_vars, _global_coeff_funcs_gen_vars, _global_workers_gen_vars
    import dill
    _global_model_gen_vars = dill.loads(blob_bytes_local)
    _global_coeff_funcs_gen_vars = {k: dill.loads(f_bytes) for k, f_bytes in serialized_funcs_local.items()}
    _global_workers_gen_vars = workers

def worker_gen_vars(tasks, acodes_local):
    """
    Worker for building trees in _gen_vars_m1.
    Returns list of (dtk, age, tree).
    """
    model = _global_model_gen_vars
    coeff_funcs = _global_coeff_funcs_gen_vars
    workers = _global_workers_gen_vars
    acodes_eff = list(model.actions.keys()) if not acodes_local else acodes_local
    results = []
    for (dtk, age) in tasks:
        area = model.dtypes[dtk].area(1, age)
        if not area: continue
        tree = model._bld_tree_m1(
            area, dtk, age, coeff_funcs,
            tree=None, period=1,
            acodes=acodes_eff, compile_c_ycomps=True)
        results.append((dtk, age, tree))
    return results

# ----------------------------
# Globals for _cmp_cflw_m1 parallel execution
# ----------------------------

def worker_cmp_cflw_batch(args):
    """
    Module-scope worker for _cmp_cflw_m1.
    Args: (batch, cflw_keys, periods)
          where batch is a list of (i, tree) pairs
    Returns: list of (t, o, i, j, value)
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
    """
    Worker to compute (name, coeffs, sense, rhs) tuples for flow constraints Phase 3.
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

# def _worker_cmp_cflw_phase3_batch(batch):
def worker_cmp_cflw_phase3_batch(batch):
    """Worker that handles a batch of phase3 tasks."""
    batch_results = []
    for task in batch:
        batch_results.extend(worker_cmp_cflw_phase3(task))
    return batch_results

# ----------------------------
# Globals for _cmp_cgen_m1 parallel execution
# ----------------------------

def worker_cmp_cgen_batch(args):
    """
    Module-scope worker for _cmp_cgen_m1.
    Args: (batch, cgen_keys, periods)
          where batch is a list of (i, tree) pairs
    Returns: list of (t, o, i, j, val)
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
    Worker to compute (name, coeffs, sense, rhs) tuples for general constraints Phase 3.
    """
    t, o, mu_t_o, lb, ub, xnames = args
    results = []

    keys = list(mu_t_o.keys())
    x_keys = [xnames[k] for k in keys]
    mu_vals = [mu_t_o[k] for k in keys]

    # Lower bound row
    if lb is not None and t in lb:
        mu_lb = dict(zip(x_keys, mu_vals))
        results.append((f'gen-lb_{t:03d}_{o}', mu_lb, opt.SENSE_GEQ, lb[t]))

    # Upper bound row
    if ub is not None and t in ub:
        mu_ub = dict(zip(x_keys, mu_vals))
        results.append((f'gen-ub_{t:03d}_{o}', mu_ub, opt.SENSE_LEQ, ub[t]))

    return results

# def _worker_cmp_cgen_phase3_batch(batch):
def worker_cmp_cgen_phase3_batch(batch):
    """Worker that handles a batch of phase3 tasks."""
    batch_results = []
    for task in batch:
        batch_results.extend(worker_cmp_cgen_phase3(task))
    return batch_results

class PersistentWorkerPool:
    """
    Context manager for a persistent ProcessPoolExecutor that initializes
    workers with ForestModel and coeff_funcs.
    """

    def __init__(self, workers, blob_bytes=None, serialized_funcs=None):
        self.workers = workers
        self.blob_bytes = blob_bytes
        self.serialized_funcs = serialized_funcs
        self.executor = None

    def __enter__(self):
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
        if self.executor is not None:
            self.executor.shutdown()