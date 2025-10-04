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

    # Pool and flux sets, mirroring Example 040
    biomass_pools = [
        "SoftwoodMerch","SoftwoodFoliage","SoftwoodOther","SoftwoodCoarseRoots","SoftwoodFineRoots",
        "HardwoodMerch","HardwoodFoliage","HardwoodOther","HardwoodCoarseRoots","HardwoodFineRoots",
    ]
    dom_pools = [
        "AboveGroundVeryFastSoil","BelowGroundVeryFastSoil","AboveGroundFastSoil","BelowGroundFastSoil",
        "MediumSoil","AboveGroundSlowSoil","BelowGroundSlowSoil","SoftwoodStemSnag","SoftwoodBranchSnag",
        "HardwoodStemSnag","HardwoodBranchSnag",
    ]
    emissions_pools = ["CO2","CH4","CO","NO2"]
    products_pools = ["Products"]
    ecosystem_pools = biomass_pools + dom_pools
    all_pools = biomass_pools + dom_pools + emissions_pools + products_pools

    decay_emissions_fluxes = [
        "DecayVFastAGToAir","DecayVFastBGToAir","DecayFastAGToAir","DecayFastBGToAir",
        "DecayMediumToAir","DecaySlowAGToAir","DecaySlowBGToAir",
        "DecaySWStemSnagToAir","DecaySWBranchSnagToAir","DecayHWStemSnagToAir","DecayHWBranchSnagToAir",
    ]
    annual_process_fluxes = [
        "DecayDOMCO2Emission","DeltaBiomass_AG","DeltaBiomass_BG","TurnoverMerchLitterInput",
        "TurnoverFolLitterInput","TurnoverOthLitterInput","TurnoverCoarseLitterInput","TurnoverFineLitterInput",
        "DecayVFastAGToAir","DecayVFastBGToAir","DecayFastAGToAir","DecayFastBGToAir","DecayMediumToAir",
        "DecaySlowAGToAir","DecaySlowBGToAir","DecaySWStemSnagToAir","DecaySWBranchSnagToAir",
        "DecayHWStemSnagToAir","DecayHWBranchSnagToAir",
    ]

    # Tag dtype key and aggregate per dtype,timestep (per-ha because SIT area=1)
    pi["dtype_key"] = pi.apply(lambda r: f"{r['theme0']} {r['theme1']} {r['theme2']} {r['theme3']} {r['theme4']}", axis=1)
    fi["dtype_key"] = fi.apply(lambda r: f"{r['theme0']} {r['theme1']} {r['theme2']} {r['theme3']} {r['theme4']}", axis=1)
    pi_gb_sum = pi.groupby(["dtype_key","timestep"], as_index=True)[all_pools].sum()
    fi_gb_sum = fi.groupby(["dtype_key","timestep"], as_index=True)[decay_emissions_fluxes].sum()

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
            # Register all pools as curves (per-ha), mark as special to avoid simplification
            if dkey in pi_gb_sum.index.get_level_values(0):
                pool_df = pi_gb_sum.loc[dkey]
                for yname in all_pools:
                    if yname not in pool_df.columns:
                        continue
                    pts = list(zip(pool_df.index.values, pool_df[yname].values))
                    curve = fm.register_curve(ws3_core.Curve(yname, points=pts, type="a", is_volume=False,
                                                             xmax=fm.max_age, period_length=fm.period_length,
                                                             is_special=True))
                    curves.append((yname, curve))
                    dt.add_ycomp("a", yname, curve)

            # Register decay-emissions flux curves
            if dkey in fi_gb_sum.index.get_level_values(0):
                flux_df = fi_gb_sum.loc[dkey]
                for yname in decay_emissions_fluxes:
                    if yname not in flux_df.columns:
                        continue
                    pts = list(zip(flux_df.index.values, flux_df[yname].values))
                    curve = fm.register_curve(ws3_core.Curve(yname, points=pts, type="a", is_volume=False,
                                                             xmax=fm.max_age, period_length=fm.period_length,
                                                             is_special=True))
                    curves.append((yname, curve))
                    dt.add_ycomp("a", yname, curve)
            break

    fm.reset()
    fm.grow()

    # Helper mirroring Example 040's comparison logic (no scaling, decade endpoints)
    def compare_ws3_cbm(pools, fluxes, cbm_x_shift=False):
        sit_config_cmp, sit_tables_cmp = fm.to_cbm_sit(
            softwood_volume_yname="swdvol",
            hardwood_volume_yname="hwdvol",
            admin_boundary="British Columbia",
            eco_boundary="Montane Cordillera",
            disturbance_type_mapping=disturbance_type_mapping,
        )
        cbm_output_cmp = run_cbm(sit_config_cmp, sit_tables_cmp, n_steps=100, plot=False)
        pi_cmp = cbm_output_cmp.classifiers.to_pandas().merge(
            cbm_output_cmp.pools.to_pandas(), on=["identifier", "timestep"]
        )
        fi_cmp = cbm_output_cmp.classifiers.to_pandas().merge(
            cbm_output_cmp.flux.to_pandas(), on=["identifier", "timestep"]
        )
        df_cbm = pd.DataFrame({
            "period": pi_cmp["timestep"] * 0.1,
            "pool": pi_cmp[pools].sum(axis=1),
            "flux": fi_cmp[fluxes].sum(axis=1),
        }).groupby("period").sum().reset_index()
        if cbm_x_shift:
            df_cbm = df_cbm.iloc[1::10, :].reset_index(drop=True)
            df_cbm["period"] = (df_cbm["period"] - 0.1 + 1.0).astype(int)
        else:
            df_cbm = df_cbm.iloc[10::10, :].reset_index(drop=True)
            df_cbm["period"] = df_cbm["period"].astype(int)

        df_ws3 = pd.DataFrame({
            "period": fm.periods,
            "pool": [sum(fm.inventory(period, pool) for pool in pools) for period in fm.periods],
            "flux": [sum(fm.inventory(period, flux) for flux in fluxes) for period in fm.periods],
        })
        return df_ws3, df_cbm

    pools_compare = biomass_pools + dom_pools + products_pools
    fluxes_compare = decay_emissions_fluxes
    df_ws3, df_cbm = compare_ws3_cbm(pools_compare, fluxes_compare, cbm_x_shift=True)

    pool_cmp = df_ws3[["period", "pool"]].merge(
        df_cbm[["period", "pool"]], on="period", suffixes=("_ws3", "_cbm")
    )
    flux_cmp = df_ws3[["period", "flux"]].merge(
        df_cbm[["period", "flux"]], on="period", suffixes=("_ws3", "_cbm")
    )

    debug_dir = figs_dir.parent / "repro" / "_debug"
    ensure_dir(debug_dir)
    pool_cmp.to_csv(debug_dir / "pool_cmp.csv", index=False)
    flux_cmp.to_csv(debug_dir / "flux_cmp.csv", index=False)

    pool_mape = ((pool_cmp["pool_ws3"] - pool_cmp["pool_cbm"]).abs() / pool_cmp["pool_cbm"]).mean()
    flux_mape = ((flux_cmp["flux_ws3"] - flux_cmp["flux_cbm"]).abs() / flux_cmp["flux_cbm"]).mean()
    print(f"Neilson hack validation: pool MAPE={pool_mape:.4%}, flux MAPE={flux_mape:.4%}")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    axes[0].plot(pool_cmp["period"], pool_cmp["pool_cbm"], label="CBM pool", linewidth=1.6)
    axes[0].plot(pool_cmp["period"], pool_cmp["pool_ws3"], label="WS3 (embedded)", linestyle="--", linewidth=1.6)
    axes[0].set_ylabel("Ecosystem carbon pool")
    axes[0].legend()

    axes[1].plot(flux_cmp["period"], flux_cmp["flux_cbm"], label="CBM decay flux", linewidth=1.6)
    axes[1].plot(flux_cmp["period"], flux_cmp["flux_ws3"], label="WS3 (embedded)", linestyle="--", linewidth=1.6)
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
