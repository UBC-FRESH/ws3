#!/usr/bin/env python3
"""Generate the spatial allocation figure used in the EMS manuscript."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from style import apply_fresh_style, ensure_dir

# Resolve repository paths regardless of invocation directory
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import ws3.forest
from ws3.common import rasterize_stands, hash_dt
from ws3.spatial import ForestRaster
from util import schedule_harvest_areacontrol, compile_scenario


DATA_DIR = EXAMPLES_DIR / "data"
WOODSTOCK_DIR = DATA_DIR / "woodstock_model_files"
SHAPE_PATH = DATA_DIR / "shp" / "tsa24_clipped.shp" / "stands.shp"

CACHE_DIR = REPO_ROOT / "papers" / "ems" / "repro" / "_spatial_cache"
RASTER_DIR = CACHE_DIR / "rasters"
SPATIAL_DIR = CACHE_DIR / "spatial_outputs"
INVENTORY_TIF = RASTER_DIR / "tsa24_inventory.tif"

FIGS_DIR = REPO_ROOT / "papers" / "ems" / "figs"
OUTPUT_FIG = FIGS_DIR / "f3_spatial_allocation.png"

BASE_YEAR = 2020
HORIZON = 5
PERIOD_LENGTH = 10
YEARS_TO_PLOT = 5
THEME_COLUMNS = ["theme0", "theme1", "theme2", "theme3", "curve1"]
COLOR_PALETTE = ["#2E7D32", "#1976D2", "#C2185B", "#F57C00", "#6D4C41"]
BACKGROUND_COLOR = "#e0e0e0"


def build_forest_model() -> ws3.forest.ForestModel:
    """Instantiate ForestModel with Woodstock inputs."""
    fm = ws3.forest.ForestModel(
        model_name="tsa24_clipped",
        model_path=str(WOODSTOCK_DIR),
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
    return fm


def ensure_transition_templates(fm: ws3.forest.ForestModel) -> None:
    """Fill missing (action, age) transitions using the fallback -1 template."""
    for period in fm.periods:
        for acode, by_dtype in fm.applied_actions[period].items():
            for dtk, by_age in by_dtype.items():
                dt = fm.dtypes[dtk]
                for age in list(by_age.keys()):
                    if (acode, age) not in dt.transitions:
                        if (acode, -1) in dt.transitions:
                            dt.transitions[(acode, age)] = dt.transitions[(acode, -1)]
                        else:
                            raise KeyError(f"Missing transition for {(acode, age)} in {dtk}")


def rasterize_inventory(fm: ws3.forest.ForestModel) -> Dict[Tuple[int, int, int, int, int], int]:
    """Rasterize the stand polygons into an age-class inventory grid."""
    ensure_dir(RASTER_DIR)
    return rasterize_stands(
        str(SHAPE_PATH),
        str(INVENTORY_TIF),
        theme_cols=THEME_COLUMNS,
        age_col="age",
        blk_col="curve2",
        age_divisor=1.0,
    )


def allocate_schedule(
    fm: ws3.forest.ForestModel,
    hdt_map: Dict[Tuple[int, int, int, int, int], int],
) -> None:
    """Allocate the heuristic schedule to per-year rasters."""
    ensure_dir(SPATIAL_DIR)

    for tif in SPATIAL_DIR.glob("harvested_*.tif"):
        tif.unlink()

    forest_raster = ForestRaster(
        hdt_map=hdt_map,
        hdt_func=hash_dt,
        src_path=str(INVENTORY_TIF),
        snk_path=str(SPATIAL_DIR),
        acode_map={"harvest": "harvested"},
        forestmodel=fm,
        base_year=BASE_YEAR,
        horizon=HORIZON,
        period_length=PERIOD_LENGTH,
        time_step=1,
        piggyback_acodes={},
    )

    forest_raster.allocate_schedule(verbose=False, sda_mode="randblk", nthresh=10)
    forest_raster.cleanup()


def harvest_rasters_by_year() -> Dict[int, Path]:
    """Return a mapping from calendar year to harvested raster path."""
    files = sorted(SPATIAL_DIR.glob("harvested_*.tif"))
    by_year: Dict[int, Path] = {}
    for path in files:
        try:
            year = int(path.stem.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unexpected raster naming: {path.name}") from exc
        by_year[year] = path
    return by_year


def render_overlay(by_year: Dict[int, Path]) -> Path:
    """Create the multi-year disturbance overlay and save to disk."""
    ensure_dir(FIGS_DIR)

    years = [BASE_YEAR + offset for offset in range(YEARS_TO_PLOT)]
    available = [(year, by_year[year]) for year in years if year in by_year]
    if not available:
        raise ValueError("No harvest rasters found for requested years.")

    with rasterio.open(INVENTORY_TIF) as src:
        inventory = src.read(1)
        inv_nodata = src.nodata
        template_shape = inventory.shape

    if inv_nodata is None:
        forest_mask = np.ma.array(np.ones(template_shape, dtype=int), mask=np.isnan(inventory))
    else:
        forest_mask = np.ma.array(
            np.ones(template_shape, dtype=int), mask=(inventory == inv_nodata)
        )

    display = np.full(template_shape, fill_value=-1, dtype=int)

    for idx, (year, tif_path) in enumerate(available):
        with rasterio.open(tif_path) as src:
            arr = src.read(1)
            nodata = src.nodata
            if nodata is None:
                harvested = np.isfinite(arr)
            else:
                harvested = arr != nodata
        update_mask = harvested & (display == -1)
        display = np.where(update_mask, idx, display)

    masked_display = np.ma.array(display, mask=display == -1)

    palette = COLOR_PALETTE[: len(available)]
    harvest_cmap = ListedColormap(palette)
    norm = BoundaryNorm(range(len(palette) + 1), harvest_cmap.N)
    background_cmap = ListedColormap([BACKGROUND_COLOR])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(forest_mask, cmap=background_cmap, interpolation="nearest", alpha=0.45)
    ax.imshow(masked_display, cmap=harvest_cmap, norm=norm, interpolation="nearest")
    ax.set_title(
        f"Spatial harvest allocation by year ({available[0][0]}-{available[-1][0]})"
    )
    ax.axis("off")

    handles = [
        Patch(facecolor=palette[i], label=str(year))
        for i, (year, _path) in enumerate(available)
    ]
    ax.legend(handles=handles, title="Harvest year", loc="lower left", frameon=False)

    fig.savefig(OUTPUT_FIG, dpi=300)
    plt.close(fig)
    return OUTPUT_FIG


def main() -> None:
    apply_fresh_style()

    ensure_dir(CACHE_DIR)
    fm = build_forest_model()
    schedule_harvest_areacontrol(fm)
    compile_scenario(fm)  # populates outputs for reporting if desired
    ensure_transition_templates(fm)
    hdt_map = rasterize_inventory(fm)
    allocate_schedule(fm, hdt_map)
    by_year = harvest_rasters_by_year()
    figure_path = render_overlay(by_year)
    print(f"Wrote spatial allocation figure to {figure_path}")


if __name__ == "__main__":
    main()
