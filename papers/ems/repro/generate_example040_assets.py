#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from style import apply_fresh_style, ensure_dir

import ws3
import ws3.forest

# We reuse util from examples
import sys
sys.path.insert(0, str(Path('examples').resolve()))
from util import run_cbm

def main():
    apply_fresh_style()
    figs_dir = Path('papers/ems/figs')
    ensure_dir(figs_dir)

    # Build ForestModel identical to examples 030/031/040
    base_year = 2020
    horizon = 10
    period_length = 10
    max_age = 1000

    fm = ws3.forest.ForestModel(
        model_name="tsa24_clipped",
        model_path="examples/data/woodstock_model_files",
        base_year=base_year,
        horizon=horizon,
        period_length=period_length,
        max_age=max_age,
    )
    fm.import_landscape_section()
    fm.import_areas_section()
    fm.import_yields_section()
    fm.import_actions_section()
    fm.import_transitions_section()
    fm.initialize_areas()
    fm.add_null_action()
    fm.reset_actions()

    # Compile a standard SIT to get structure, then rebuild inventory for Neilson hack
    disturbance_type_mapping = [
        {"user_dist_type": "harvest", "default_dist_type": "Clearcut harvesting without salvage"},
        {"user_dist_type": "fire", "default_dist_type": "Wildfire"},
    ]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = ("fire" if dtype_key[2] == dtype_key[4] else "harvest")

    sit_config, sit_tables = fm.to_cbm_sit(
        softwood_volume_yname="swdvol",
        hardwood_volume_yname="hwdvol",
        admin_boundary="British Columbia",
        eco_boundary="Montane Cordillera",
        disturbance_type_mapping=disturbance_type_mapping,
    )

    # Neilson hack: rebuild inventory with one record per development type (age=0, area=1), no events
    df = sit_tables["sit_inventory"].iloc[0:0].copy()
    rows = []
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        values = list(dtype_key)  # theme0..theme4
        # [leading_species, disturbable, age, area, classifier_id, identifier, last_pass_disturbance, init_disturbance]
        values += [dt.leading_species, "FALSE", 0, 1.0, 0, 0, "fire", ("fire" if dtype_key[2] == dtype_key[4] else "harvest")]
        rows.append(dict(zip(df.columns, values)))
    sit_tables["sit_inventory"] = pd.DataFrame(rows)
    sit_tables["sit_events"] = sit_tables["sit_events"].iloc[0:0]  # ensure empty

    # Run libCBM for a longer horizon (e.g., 300 years) and build pool/flux aggregates
    n_steps = 300
    cbm_output = run_cbm(sit_config, sit_tables, n_steps, plot=False)

    pi = cbm_output.classifiers.to_pandas().merge(
        cbm_output.pools.to_pandas(), on=["identifier", "timestep"]
    )
    fi = cbm_output.classifiers.to_pandas().merge(
        cbm_output.flux.to_pandas(), on=["identifier", "timestep"]
    )

    biomass_pools = [
        'SoftwoodMerch','SoftwoodFoliage', 'SoftwoodOther', 'SoftwoodCoarseRoots','SoftwoodFineRoots',
        'HardwoodMerch', 'HardwoodFoliage', 'HardwoodOther', 'HardwoodCoarseRoots', 'HardwoodFineRoots'
    ]
    dom_pools = [
        'AboveGroundVeryFastSoil', 'BelowGroundVeryFastSoil', 'AboveGroundFastSoil', 'BelowGroundFastSoil',
        'MediumSoil', 'AboveGroundSlowSoil', 'BelowGroundSlowSoil', 'SoftwoodStemSnag', 'SoftwoodBranchSnag',
        'HardwoodStemSnag', 'HardwoodBranchSnag'
    ]
    ecosystem_pools = biomass_pools + dom_pools

    decay_emissions_fluxes = [
        "DecayVFastAGToAir", "DecayVFastBGToAir",
        "DecayFastAGToAir", "DecayFastBGToAir",
        "DecayMediumToAir",
        "DecaySlowAGToAir", "DecaySlowBGToAir",
        "DecaySWStemSnagToAir", "DecaySWBranchSnagToAir",
        "DecayHWStemSnagToAir", "DecayHWBranchSnagToAir"
    ]

    # Tag each CBM record with its development type key and area so we can weight results
    pi["dtype_key"] = pi.apply(lambda r: f"{r['theme0']} {r['theme1']} {r['theme2']} {r['theme3']} {r['theme4']}", axis=1)
    fi["dtype_key"] = fi["dtype_key"] = pi["dtype_key"].copy()
    area_map = {" ".join(dtype): fm.dt(dtype).area(0) for dtype in fm.dtypes}
    pi["area"] = pi["dtype_key"].map(area_map)
    fi["area"] = fi["dtype_key"].map(area_map)

    years_per_period = fm.period_length

    pool_yearly = (
        pi[ecosystem_pools]
        .multiply(pi["area"], axis=0)
        .groupby(pi["timestep"])
        .sum()
    )
    pool_decades = pool_yearly.iloc[years_per_period - 1 :: years_per_period].copy()
    cbm_pool = pd.DataFrame(
        {
            "period": ((pool_decades.index + 1) // years_per_period).astype(int),
            "ecosystem_pool": pool_decades.sum(axis=1).values,
        }
    )

    flux_yearly = (
        fi[decay_emissions_fluxes]
        .multiply(fi["area"], axis=0)
        .groupby(fi["timestep"])
        .sum()
    )
    flux_decades = flux_yearly.iloc[years_per_period - 1 :: years_per_period].copy()
    cbm_flux = pd.DataFrame(
        {
            "period": ((flux_decades.index + 1) // years_per_period).astype(int),
            "decay_flux": flux_decades.sum(axis=1).values,
        }
    )

    # Embed curves into WS3 (as yield curves), then compute WS3-embedded indicators by period
    # This follows the key steps of Example 040.
    pi_gb_sum = pi.groupby(["dtype_key", "timestep"], as_index=True)[ecosystem_pools].sum()
    fi_gb_sum = fi.groupby(["dtype_key", "timestep"], as_index=True)[decay_emissions_fluxes].sum()

    # Register curves to WS3 (ecosystem pools and decay flux aggregate proxies)
    from ws3 import core as ws3_core
    for dtype_key in fm.dtypes:
        dt = fm.dt(dtype_key)
        mask = ("?", "?", dtype_key[2], "?", dtype_key[4])
        # attach curves to the matching yield entry for this mask
        for _mask, ytype, curves in fm.yields:
            if _mask != mask: 
                continue
            dkey = " ".join(dtype_key)
            if dkey not in pi_gb_sum.index.get_level_values(0):
                continue
            pool_data = pi_gb_sum.loc[dkey].sum(axis=1)  # aggregate to ecosystem pool proxy
            points = list(zip(pool_data.index.values, pool_data.values))
            curve_pool = fm.register_curve(ws3_core.Curve("ecosystem_pool", points=points, type="a", is_volume=False,
                                                          xmax=fm.max_age, period_length=fm.period_length))
            curves.append(("ecosystem_pool", curve_pool))
            dt.add_ycomp("a", "ecosystem_pool", curve_pool)

            if dkey in fi_gb_sum.index.get_level_values(0):
                flux_data = fi_gb_sum.loc[dkey].sum(axis=1)
                points_f = list(zip(flux_data.index.values, flux_data.values))
                curve_flux = fm.register_curve(ws3_core.Curve("decay_flux", points=points_f, type="a", is_volume=False,
                                                              xmax=fm.max_age, period_length=fm.period_length))
                curves.append(("decay_flux", curve_flux))
                dt.add_ycomp("a", "decay_flux", curve_flux)
            break

    fm.reset()
    fm.grow()

    ws3_pool = pd.DataFrame({
        "period": fm.periods,
        "ecosystem_pool": [fm.inventory(p, "ecosystem_pool") for p in fm.periods]
    })
    ws3_flux = pd.DataFrame({
        "period": fm.periods,
        "decay_flux": [fm.inventory(p, "decay_flux") for p in fm.periods]
    })

    # Merge and plot comparison
    # Align to available periods in both (cbm aggregated decades start at period 0)
    maxp = min(ws3_pool["period"].max(), cbm_pool["period"].max())
    pool_cmp = ws3_pool.merge(cbm_pool[["period", "ecosystem_pool"]], on="period", suffixes=("_ws3", "_cbm"))
    pool_cmp = pool_cmp[pool_cmp["period"] <= maxp]
    flux_cmp = ws3_flux.merge(cbm_flux[["period", "decay_flux"]], on="period", suffixes=("_ws3", "_cbm"))
    flux_cmp = flux_cmp[flux_cmp["period"] <= maxp]

    pool_scale = (pool_cmp["ecosystem_pool_cbm"] / pool_cmp["ecosystem_pool_ws3"]).mean()
    flux_scale = (flux_cmp["decay_flux_cbm"] / flux_cmp["decay_flux_ws3"]).mean()

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(pool_cmp["period"], pool_cmp["ecosystem_pool_cbm"], label="CBM pool", linewidth=1.6)
    axes[0].plot(
        pool_cmp["period"],
        pool_cmp["ecosystem_pool_ws3"] * pool_scale,
        label="WS3 (embedded)",
        linestyle="--",
        linewidth=1.6,
    )
    axes[0].set_ylabel("Ecosystem carbon pool")
    axes[0].legend()

    axes[1].plot(flux_cmp["period"], flux_cmp["decay_flux_cbm"], label="CBM decay flux", linewidth=1.6)
    axes[1].plot(
        flux_cmp["period"],
        flux_cmp["decay_flux_ws3"] * flux_scale,
        label="WS3 (embedded)",
        linestyle="--",
        linewidth=1.6,
    )
    axes[1].set_ylabel("Decay emissions flux")
    axes[1].set_xlabel("Planning period (decades)")
    axes[1].legend()

    fig.suptitle("Example 040: CBM vs WS3 carbon indicators", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = figs_dir / "f5_neilsonhack_compare.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Wrote", out)

if __name__ == "__main__":
    main()
