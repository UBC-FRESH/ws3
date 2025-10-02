#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import shutil
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil
import rasterio
from concurrent.futures import ThreadPoolExecutor

import ws3
import ws3.forest
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


DATA_ROOT = REPO_ROOT / "papers" / "ems" / "repro" / "data" / "cccandies_demo_input"
TABLES_DIR = REPO_ROOT / "papers" / "ems" / "tables"
FIGS_DIR = REPO_ROOT / "papers" / "ems" / "figs"
SCALING_CACHE = REPO_ROOT / "papers" / "ems" / "repro" / "_scaling_cache"

BASE_YEAR = 2020
HORIZON = 10
PERIOD_LENGTH = 10
WORKER_MODES = [1, 16]


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
        hdt[bn] = pickle.load(open(pkl, "rb"))
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
        action_params=None,
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


def run_spatial_allocation(fm, basenames, hdt, dataset_root: Path, workers: int) -> float:
    start = time.perf_counter()
    label = "+".join(basenames)
    cache_dir = SCALING_CACHE / label / f"w{workers}"
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
            fr.allocate_schedule(mask=(bn, '?', '?', '?'), verbose=False, sda_mode='randblk', nthresh=10)
        finally:
            fr.cleanup()

    if workers > 1 and len(basenames) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(basenames))) as executor:
            list(executor.map(allocate_single, basenames))
    else:
        for bn in basenames:
            allocate_single(bn)

    shutil.rmtree(cache_dir, ignore_errors=True)
    return time.perf_counter() - start


def run_benchmark(dataset_root: Path, basenames: List[str], workers: int, precomputed_complexity: int | None = None) -> Dict[str, float]:
    process = psutil.Process(os.getpid())
    rss_gb = lambda: process.memory_info().rss / (1024 ** 3)

    t0 = time.perf_counter()
    fm, hdt, complexity = build_forest_model(dataset_root, basenames)
    init_time = time.perf_counter() - t0
    mem_init = rss_gb()
    if precomputed_complexity is not None:
        complexity = precomputed_complexity

    from util import schedule_harvest_areacontrol, compile_scenario

    target_masks = build_target_masks(basenames, hdt)

    t1 = time.perf_counter()
    schedule = schedule_harvest_areacontrol(fm, target_masks=target_masks, verbose=0)
    sched_time = time.perf_counter() - t1
    mem_sched = rss_gb()

    # ensure age-specific transition templates exist for spatial allocation
    for dtk, dt in fm.dtypes.items():
        for acode in fm.actions:
            if (acode, -1) in dt.transitions:
                template = dt.transitions[(acode, -1)]
                for age in range(fm.max_age + 1):
                    if (acode, age) not in dt.transitions:
                        dt.transitions[(acode, age)] = template

    t2 = time.perf_counter()
    df = compile_scenario(fm)
    compile_time = time.perf_counter() - t2
    mem_compile = rss_gb()

    spatial_time = run_spatial_allocation(fm, basenames, hdt, dataset_root, workers)
    mem_spatial = rss_gb()

    metrics = {
        "combo": "+".join(basenames),
        "workers": workers,
        "n_periods": len(fm.periods),
        "n_dtypes": len(fm.dtypes),
        "n_dt_age": complexity,
        "init_s": round(init_time, 3),
        "sched_s": round(sched_time, 3),
        "compile_s": round(compile_time, 3),
        "spatial_s": round(spatial_time, 3),
        "mem_init_gb": round(mem_init, 3),
        "mem_sched_gb": round(mem_sched, 3),
        "mem_compile_gb": round(mem_compile, 3),
        "mem_spatial_gb": round(mem_spatial, 3),
        "mem_peak_gb": round(max(mem_init, mem_sched, mem_compile, mem_spatial), 3),
        "ha_mean": float(np.mean(df.oha)),
        "hv_mean": float(np.mean(df.ohv)),
        "gs_last": float(df.ogs.iloc[-1]),
    }

    fm.reset()
    if schedule:
        fm.apply_schedule(
            schedule,
            force_integral_area=True,
            override_operability=True,
            fuzzy_age=True,
            recourse_enabled=True,
            verbose=False,
            compile_c_ycomps=True,
        )

    return metrics


def bench_example_woodstock() -> List[dict]:
    example_root = REPO_ROOT / "examples" / "data" / "woodstock_model_files"
    if not example_root.exists():
        return []
    t0 = time.perf_counter()
    fm = ws3.forest.ForestModel(
        model_name="tsa24_clipped",
        model_path=str(example_root),
        base_year=BASE_YEAR,
        horizon=HORIZON,
        period_length=PERIOD_LENGTH,
        max_age=1000,
    )
    fm.import_landscape_section()
    fm.import_areas_section()
    fm.import_yields_section()
    fm.import_actions_section()
    fm.import_transitions_section()
    fm.initialize_areas()
    fm.add_null_action()
    fm.reset_actions()
    init_time = time.perf_counter() - t0

    from util import schedule_harvest_areacontrol, compile_scenario

    t1 = time.perf_counter()
    schedule_harvest_areacontrol(fm)
    sched_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    df = compile_scenario(fm)
    compile_time = time.perf_counter() - t2

    return [
        {
            "combo": "example",
            "workers": 1,
            "n_periods": len(fm.periods),
            "n_dtypes": len(fm.dtypes),
            "n_dt_age": sum(len(dt._areas[0]) for dt in fm.dtypes.values()),
            "init_s": round(init_time, 3),
            "sched_s": round(sched_time, 3),
            "compile_s": round(compile_time, 3),
            "spatial_s": 0.0,
            "mem_init_gb": 0.0,
            "mem_sched_gb": 0.0,
            "mem_compile_gb": 0.0,
            "mem_spatial_gb": 0.0,
            "mem_peak_gb": 0.0,
            "ha_mean": float(np.mean(df.oha)),
            "hv_mean": float(np.mean(df.ohv)),
            "gs_last": float(df.ogs.iloc[-1]),
        }
    ]


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

    results: List[dict] = []
    for combo in combos:
        label = "+".join(combo)
        for workers in WORKER_MODES:
            try:
                metrics = run_benchmark(DATA_ROOT, combo, workers, precomputed[label])
                results.append(metrics)
                print("Benchmarked:", metrics)
            except Exception as exc:
                print(f"Skipping {label} (workers={workers}) due to: {exc}")

    if not results:
        print("No TSA produced metrics; running fallback example benchmark.")
        results = bench_example_woodstock()

    out_csv = TABLES_DIR / "perf_scaling.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Wrote", out_csv)


if __name__ == "__main__":
    main()
