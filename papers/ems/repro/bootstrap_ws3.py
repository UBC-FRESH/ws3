#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import rasterio

import ws3


def bootstrap_themes(fm, theme_cols: List[str] = None, basecodes: List[List[str]] = None, aggs=None, verbose=False):
    if theme_cols is None:
        theme_cols = ["theme0", "theme1", "theme2", "theme3"]
    if basecodes is None:
        basecodes = [[], [], [], []]
    if aggs is None:
        aggs = [{}, {}, {}, {}]
    for ti, t in enumerate(theme_cols):
        fm.add_theme(t, basecodes=basecodes[ti], aggs=aggs[ti])


def bootstrap_areas(
    fm,
    basenames: List[str],
    rst_path: Callable[[str], str],
    hdt: Dict[str, Dict],
    year: int | None = None,
    new_dts: bool = True,
    verbose: bool = False,
):
    if year is None:
        year = fm.base_year
    for bn in basenames:
        inv_dir = Path(rst_path(bn))
        inv_dir.mkdir(parents=True, exist_ok=True)
        inv_init = inv_dir / "inventory_init.tif"
        inv_year = inv_dir / f"inventory_{year}.tif"
        if not inv_year.exists():
            shutil.copyfile(inv_init, inv_year)
        with rasterio.open(inv_year, "r") as src:
            pxa = (src.transform.a ** 2) * 0.0001
            bh = src.read(1)
            ba = src.read(2)
            nodata_age = src.nodatavals[1] if len(src.nodatavals) > 1 else None
        total_area = 0.0
        for h, dt in hdt[bn].items():
            dt = tuple(str(x) for x in dt)
            mask = bh == h
            if not np.any(mask):
                continue
            ages = ba[mask]
            if new_dts and dt not in fm.dtypes:
                fm.dtypes[dt] = ws3.forest.DevelopmentType(dt, fm)
            for age_val in np.unique(ages):
                if nodata_age is not None and age_val == nodata_age:
                    continue
                if age_val <= 0:
                    continue
                area = float(np.sum(ages == age_val) * pxa)
                if area <= 0:
                    continue
                fm.dtypes[dt].area(0, int(age_val), area)
                total_area += area
        if verbose:
            print(f"bootstrap_areas: {bn} total area {total_area:.1f} ha")


def bootstrap_yields(fm, yld_path: str, tvy_name: str = "totvol", period_length: float = 10.0, x_max: int = 350):
    au_table = pd.read_csv(f"{yld_path}/au_table.csv").set_index("au_id")
    curve_points_table = pd.read_csv(f"{yld_path}/curve_points_table.csv").set_index("curve_id")
    for au_id, au_row in au_table.iterrows():
        yname = f"s{int(au_row.canfi_species):04d}"
        curve_id = au_row.unmanaged_curve_id
        mask = ("?", "?", str(curve_id), "?")
        dt_keys = fm.unmask(mask)
        if not dt_keys:
            continue
        points = [
            (r.x, r.y)
            for _, r in curve_points_table.loc[curve_id].iterrows()
            if (r.x % period_length) == 0 and r.x <= x_max
        ]
        c = fm.register_curve(
            ws3.core.Curve(
                yname,
                points=points,
                type="a",
                is_volume=True,
                xmax=fm.max_age,
                period_length=period_length,
            )
        )
        fm.yields.append((mask, "a", [(yname, c)]))
        fm.ynames.add(yname)
        for dtk in dt_keys:
            fm.dtypes[dtk].add_ycomp("a", yname, c)
    expr = "_SUM(%s)" % ", ".join(fm.ynames)
    fm.yields.append((("?", "?", "?", "?"), "c", [(tvy_name, expr)]))
    fm.ynames.add(tvy_name)
    for dtk in fm.dtypes.keys():
        fm.dtypes[dtk].add_ycomp("c", tvy_name, expr)


def bootstrap_actions(fm, action_params: Dict[str, Dict]):
    for acode, ap in action_params.items():
        mask, oe, is_harvest, targetage = ap["mask"], ap["oe"], ap["is_harvest"], ap["targetage"]
        target = [(mask, 1.0, None, None, None, None, None)]
        fm.actions[acode] = ws3.forest.Action(acode, targetage=targetage, is_harvest=is_harvest)
        fm.oper_expr[acode] = {mask: oe}
        fm.transitions[acode] = {mask: {"": target}}
        for dtk in fm.unmask(mask):
            dt = fm.dtypes[dtk]
            dt.oper_expr[acode] = [oe]
            for age in range(1, fm.max_age):
                if not dt.is_operable(acode, 1, age):
                    continue
                fm.dtypes[dtk].transitions[(acode, age)] = target


def compile_basecodes(hdt: Dict[str, Dict], basenames: List[str], theme_cols: List[str]) -> List[List[str]]:
    if not basenames:
        return [[] for _ in theme_cols]
    n_themes = len(theme_cols)
    agg = [set() for _ in range(n_themes)]
    for bn in basenames:
        for dt in hdt[bn].values():
            if len(dt) != n_themes:
                raise ValueError("Theme count mismatch between hdt entries and theme_cols")
            for i, val in enumerate(dt):
                agg[i].add(str(val))
    return [sorted(list(vals)) for vals in agg]


def bootstrap_forestmodel(
    basenames: List[str],
    model_name: str,
    model_path: str,
    base_year: int,
    yld_path: str,
    tif_path_fn: Callable[[str], str],
    horizon: int,
    period_length: int,
    max_age: int,
    basecodes: List[List[str]],
    action_params: Dict[str, Dict],
    hdt: Dict[str, Dict],
    tvy_name: str = "totvol",
):
    from ws3.forest import ForestModel

    fm = ForestModel(
        model_name=model_name,
        model_path=model_path,
        base_year=base_year,
        horizon=horizon,
        period_length=period_length,
        max_age=max_age,
    )
    bootstrap_themes(fm, basecodes=basecodes)
    bootstrap_areas(fm, basenames, tif_path_fn, hdt)
    bootstrap_yields(fm, yld_path, tvy_name=tvy_name, period_length=period_length)
    bootstrap_actions(fm, action_params)
    fm.add_null_action()
    fm.compile_actions()
    fm.reset_actions()
    fm.initialize_areas()
    fm.grow()
    return fm
