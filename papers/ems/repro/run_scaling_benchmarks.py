#!/usr/bin/env python3
"""Scaling benchmarks for heuristic and LP scheduling workflows."""

from __future__ import annotations

import gc
import os
import sys
import time
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
import rasterio

import ws3
import ws3.forest
import ws3.opt
from ws3.common import hash_dt
from ws3.spatial import ForestRaster

from style import apply_fresh_style, ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from bootstrap_ws3 import bootstrap_forestmodel, compile_basecodes
from util import compile_scenario, gen_scenario, schedule_harvest_areacontrol


DATA_ROOT = REPO_ROOT / "papers" / "ems" / "repro" / "data" / "cccandies_demo_input"
TABLES_DIR = REPO_ROOT / "papers" / "ems" / "tables"
FIGS_DIR = REPO_ROOT / "papers" / "ems" / "figs"
SCALING_CACHE = REPO_ROOT / "papers" / "ems" / "repro" / "_scaling_cache"

BASE_YEAR = 2020
HORIZON = 10
PERIOD_LENGTH = 10
HEURISTIC_WORKERS = 1
LP_WORKER_MODES = [1, 16]
EVENFLOW_TOLERANCE = 0.05

VERBOSE = False

_PROCESS = psutil.Process(os.getpid())


def rss_gb() -> float:
    """Return current resident set size (parent + children) in GiB."""
    rss = 0
    try:
        rss += _PROCESS.memory_info().rss
    except psutil.NoSuchProcess:
        return 0.0
    for child in _PROCESS.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return rss / (1024 ** 3)


def init_metrics(combo: str, mode: str, workers: int, complexity: Optional[int]) -> Dict[str, Optional[float]]:
    """Initialize a metrics dictionary with consistent columns."""
    columns = [
        "combo",
        "mode",
        "workers",
        "n_periods",
        "n_dtypes",
        "n_dt_age",
        "init_s",
        "sched_s",
        "compile_s",
        "lp_build_s",
        "lp_solve_s",
        "spatial_s",
        "mem_init_gb",
        "mem_sched_gb",
        "mem_compile_gb",
        "mem_lp_build_gb",
        "mem_lp_solve_gb",
        "mem_spatial_gb",
        "mem_peak_gb",
        "lp_status",
        "obj_value",
        "ha_mean",
        "hv_mean",
        "gs_last",
        "status",
        "error",
    ]
    metrics = {col: None for col in columns}
    metrics["combo"] = combo
    metrics["mode"] = mode
    metrics["workers"] = workers
    metrics["n_dt_age"] = complexity
    metrics["status"] = "pending"
    return metrics


def finalize_mem_peak(metrics: Dict[str, Optional[float]]) -> None:
    """Populate mem_peak_gb from available stage measurements."""
    mem_fields = [
        metrics.get("mem_init_gb"),
        metrics.get("mem_sched_gb"),
        metrics.get("mem_compile_gb"),
        metrics.get("mem_lp_build_gb"),
        metrics.get("mem_lp_solve_gb"),
        metrics.get("mem_spatial_gb"),
    ]
    mem_values = [val for val in mem_fields if isinstance(val, (int, float))]
    if mem_values:
        metrics["mem_peak_gb"] = round(max(mem_values), 3)


def enumerate_basenames(dataset_root: Path) -> List[str]:
    hdt_dir = dataset_root / "hdt"
    if not hdt_dir.exists():
        return []
    return sorted(p.stem.replace("hdt_", "") for p in hdt_dir.glob("hdt_*.pkl"))


def load_hdt(dataset_root: Path, basenames: List[str]) -> Dict[str, Dict]:
    hdt: Dict[str, Dict] = {}
    for bn in basenames:
        pkl = dataset_root / "hdt" / f"hdt_{bn}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing hdt pickle: {pkl}")
        with open(pkl, "rb") as fh:
            hdt[bn] = pickle.load(fh)
    return hdt


def compute_complexity(dataset_root: Path, basenames: List[str]) -> int:
    total = 0
    for bn in basenames:
        tif_path = dataset_root / "tif" / bn / "inventory_init.tif"
        with rasterio.open(tif_path, "r") as src:
            bh = src.read(1)
            ba = src.read(2)
            nodata_age = src.nodatavals[1] if len(src.nodatavals) > 1 else None
        hdt = load_hdt(dataset_root, [bn])[bn]
        for h in hdt:
            ages = ba[bh == h]
            if ages.size == 0:
                continue
            uniq = np.unique(ages)
            if nodata_age is not None:
                uniq = uniq[uniq != nodata_age]
            uniq = uniq[uniq > 0]
            total += len(uniq)
    return int(total)


def sorted_combinations(dataset_root: Path) -> List[List[str]]:
    singles = enumerate_basenames(dataset_root)
    if not singles:
        return []
    complexities = {bn: compute_complexity(dataset_root, [bn]) for bn in singles}
    ranked = sorted(singles, key=lambda bn: complexities[bn])
    combos: List[List[str]] = []
    running: List[str] = []
    for bn in ranked:
        running.append(bn)
        combos.append(list(running))
    return combos


def build_forest_model(dataset_root: Path, basenames: List[str]):
    hdt = load_hdt(dataset_root, basenames)
    basecodes = compile_basecodes(hdt, basenames, ["theme0", "theme1", "theme2", "theme3"])
    tif_path_fn = lambda bn: str(dataset_root / "tif" / bn)
    action_params = {
        "harvest": {
            "oe": "_age >= 100 and _age <= 400",
            "mask": ("?", "1", "?", "?"),
            "is_harvest": True,
            "targetage": 0,
        }
    }
    fm = bootstrap_forestmodel(
        basenames=basenames,
        model_name="+".join(basenames),
        model_path=str(dataset_root),
        base_year=BASE_YEAR,
        yld_path=str(dataset_root),
        tif_path_fn=tif_path_fn,
        horizon=HORIZON,
        period_length=PERIOD_LENGTH,
        max_age=1000,
        basecodes=basecodes,
        action_params=action_params,
        hdt=hdt,
        tvy_name="totvol",
    )
    complexity = sum(len(dt._areas[0]) for dt in fm.dtypes.values())
    return fm, hdt, complexity


def build_target_masks(basenames: List[str], hdt: Dict[str, Dict]) -> List[str]:
    masks = []
    for bn in basenames:
        au_codes = sorted({dt[2] for dt in hdt[bn].values()})
        for au in au_codes:
            masks.append(f"{bn} 1 {au} ?")
    return masks


def ensure_transition_templates(fm: ws3.forest.ForestModel) -> None:
    """Fill missing (action, age) transitions using generic templates."""
    for dt in fm.dtypes.values():
        for acode in fm.actions:
            template = dt.transitions.get((acode, -1))
            if template is None:
                continue
            for age in range(fm.max_age + 1):
                dt.transitions.setdefault((acode, age), template)


def run_spatial_allocation(fm, basenames, hdt, dataset_root: Path) -> float:
    start = time.perf_counter()
    label = "+".join(basenames)
    cache_dir = SCALING_CACHE / label
    ensure_dir(cache_dir)

    def allocate_single(bn: str):
        bn_dir = cache_dir / bn
        ensure_dir(bn_dir)
        src_path = dataset_root / "tif" / bn / f"inventory_{fm.base_year}.tif"
        fr = ForestRaster(
            hdt_map=hdt[bn],
            hdt_func=hash_dt,
            src_path=str(src_path),
            snk_path=str(bn_dir),
            acode_map={"harvest": f"{bn}_harvested"},
            forestmodel=fm,
            base_year=fm.base_year,
            horizon=fm.horizon,
            period_length=fm.period_length,
            time_step=fm.period_length,
            piggyback_acodes={},
        )
        try:
            fr.allocate_schedule(mask=(bn, "?", "?", "?"), verbose=False, sda_mode="randblk", nthresh=10)
        finally:
            fr.cleanup()

    for bn in basenames:
        allocate_single(bn)

    shutil.rmtree(cache_dir, ignore_errors=True)
    return time.perf_counter() - start


def benchmark_heuristic(dataset_root: Path, basenames: List[str], complexity_hint: Optional[int]) -> Dict[str, Optional[float]]:
    label = "+".join(basenames)
    metrics = init_metrics(label, "heuristic", HEURISTIC_WORKERS, complexity_hint)
    fm = None
    try:
        t0 = time.perf_counter()
        fm, hdt, complexity = build_forest_model(dataset_root, basenames)
        metrics["init_s"] = round(time.perf_counter() - t0, 3)
        metrics["mem_init_gb"] = round(rss_gb(), 3)
        metrics["n_dt_age"] = complexity
        metrics["n_periods"] = len(fm.periods)
        metrics["n_dtypes"] = len(fm.dtypes)

        target_masks = build_target_masks(basenames, hdt)
        t1 = time.perf_counter()
        schedule_harvest_areacontrol(fm, target_masks=target_masks, verbose=0)
        metrics["sched_s"] = round(time.perf_counter() - t1, 3)
        metrics["mem_sched_gb"] = round(rss_gb(), 3)

        ensure_transition_templates(fm)
        t2 = time.perf_counter()
        df = compile_scenario(fm)
        metrics["compile_s"] = round(time.perf_counter() - t2, 3)
        metrics["mem_compile_gb"] = round(rss_gb(), 3)

        spatial_time = run_spatial_allocation(fm, basenames, hdt, dataset_root)
        metrics["spatial_s"] = round(spatial_time, 3)
        metrics["mem_spatial_gb"] = round(rss_gb(), 3)

        metrics["ha_mean"] = float(np.mean(df.oha))
        metrics["hv_mean"] = float(np.mean(df.ohv))
        metrics["gs_last"] = float(df.ogs.iloc[-1])
        metrics["status"] = "ok"
    except Exception as exc:  # pragma: no cover - defensive logging
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    finally:
        if fm is not None:
            fm.reset()
        finalize_mem_peak(metrics)
        gc.collect()
    return metrics


def benchmark_lp(
    dataset_root: Path,
    basenames: List[str],
    complexity_hint: Optional[int],
    workers: int,
    solver_threads: int,
    verbose: bool,
) -> Dict[str, Optional[float]]:
    label = "+".join(basenames)
    metrics = init_metrics(label, "lp", workers, complexity_hint)
    fm = None
    problem = None
    mem_init = None
    try:
        t0 = time.perf_counter()
        fm, hdt, complexity = build_forest_model(dataset_root, basenames)
        metrics["init_s"] = round(time.perf_counter() - t0, 3)
        mem_init = rss_gb()
        metrics["mem_init_gb"] = round(mem_init, 3)
        metrics["n_dt_age"] = complexity
        metrics["n_periods"] = len(fm.periods)
        metrics["n_dtypes"] = len(fm.dtypes)

        cflw = ({p: EVENFLOW_TOLERANCE for p in fm.periods}, 1)
        t_build = time.perf_counter()
        problem = gen_scenario(
            fm,
            name=f"lp_scaling_{workers}",
            cflw_ha=cflw,
            cflw_hv=cflw,
            workers=workers,
            verbose=verbose,
        )
        metrics["lp_build_s"] = round(time.perf_counter() - t_build, 3)
        metrics["mem_lp_build_gb"] = round(rss_gb(), 3)

        problem.solver(ws3.opt.SOLVER_HIGHS)
        t_solve = time.perf_counter()
        problem.solve(threads=solver_threads, verbose=False)
        metrics["lp_solve_s"] = round(time.perf_counter() - t_solve, 3)
        metrics["mem_lp_solve_gb"] = round(rss_gb(), 3)

        status = problem.status()
        metrics["lp_status"] = status
        if status == ws3.opt.STATUS_OPTIMAL:
            try:
                metrics["obj_value"] = float(problem.z())
            except Exception:  # pragma: no cover - defensive
                metrics["obj_value"] = None
        metrics["status"] = "ok"
    except Exception as exc:  # pragma: no cover - defensive logging
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    finally:
        if fm is not None:
            fm.reset()
        finalize_mem_peak(metrics)
        del problem
        gc.collect()
    return metrics


def bench_example_woodstock() -> List[dict]:
    example_root = REPO_ROOT / "examples" / "data" / "woodstock_model_files"
    if not example_root.exists():
        return []
    metrics = init_metrics("example", "heuristic", HEURISTIC_WORKERS, None)
    fm = ws3.forest.ForestModel(
        model_name="tsa24_clipped",
        model_path=str(example_root),
        base_year=BASE_YEAR,
        horizon=HORIZON,
        period_length=PERIOD_LENGTH,
        max_age=1000,
    )
    t0 = time.perf_counter()
    fm.import_landscape_section()
    fm.import_areas_section()
    fm.import_yields_section()
    fm.import_actions_section()
    fm.import_transitions_section()
    fm.initialize_areas()
    fm.add_null_action()
    fm.reset_actions()
    metrics["init_s"] = round(time.perf_counter() - t0, 3)
    metrics["mem_init_gb"] = round(rss_gb(), 3)
    metrics["n_periods"] = len(fm.periods)
    metrics["n_dtypes"] = len(fm.dtypes)
    metrics["n_dt_age"] = sum(len(dt._areas[0]) for dt in fm.dtypes.values())

    t1 = time.perf_counter()
    schedule_harvest_areacontrol(fm)
    metrics["sched_s"] = round(time.perf_counter() - t1, 3)
    metrics["mem_sched_gb"] = round(rss_gb(), 3)

    ensure_transition_templates(fm)
    t2 = time.perf_counter()
    df = compile_scenario(fm)
    metrics["compile_s"] = round(time.perf_counter() - t2, 3)
    metrics["mem_compile_gb"] = round(rss_gb(), 3)
    metrics["ha_mean"] = float(np.mean(df.oha))
    metrics["hv_mean"] = float(np.mean(df.ohv))
    metrics["gs_last"] = float(df.ogs.iloc[-1])
    metrics["status"] = "ok"
    finalize_mem_peak(metrics)
    return [metrics]


def main() -> None:
    apply_fresh_style()
    ensure_dir(TABLES_DIR)
    ensure_dir(FIGS_DIR)

    combos = sorted_combinations(DATA_ROOT)
    if not combos:
        print("No benchmark dataset found. Expected DataLad install under:", DATA_ROOT)
        print("Try: datalad install -r -g -s https://github.com/UBC-FRESH/cccandies_demo_input", DATA_ROOT)
        results = bench_example_woodstock()
        if results:
            out_csv = TABLES_DIR / "perf_scaling.csv"
            pd.DataFrame(results).to_csv(out_csv, index=False)
            print("Wrote", out_csv)
        return

    precomputed = {"+".join(combo): compute_complexity(DATA_ROOT, combo) for combo in combos}
    run_lp = os.environ.get("RUN_LP", "0").lower() in {"1", "true", "yes"}
    if not run_lp:
        print("RUN_LP not set; LP benchmarks will be skipped.")

    results: List[dict] = []
    for combo in combos:
        label = "+".join(combo)
        try:
            heur_metrics = benchmark_heuristic(DATA_ROOT, combo, precomputed[label])
            results.append(heur_metrics)
            print("Heuristic benchmark:", heur_metrics.get("combo"), heur_metrics.get("status"))
        except Exception as exc:  # pragma: no cover - defensive logging
            results.append({
                "combo": label,
                "mode": "heuristic",
                "workers": HEURISTIC_WORKERS,
                "status": "error",
                "error": str(exc),
            })

        if run_lp:
            for workers in LP_WORKER_MODES:
                solver_threads = workers
                lp_metrics = benchmark_lp(
                    DATA_ROOT,
                    combo,
                    precomputed[label],
                    workers,
                    solver_threads,
                    verbose=VERBOSE,
                )
                results.append(lp_metrics)
                print(
                    "LP benchmark:",
                    lp_metrics.get("combo"),
                    f"workers={workers}",
                    lp_metrics.get("status"),
                )

        gc.collect()

    if not results:
        print("No benchmarks executed; writing fallback example if available.")
        results = bench_example_woodstock()

    out_csv = TABLES_DIR / "perf_scaling.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
