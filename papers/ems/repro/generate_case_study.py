#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from style import apply_fresh_style, ensure_dir

# Import ws3 and util functions from examples
import ws3
import ws3.forest

import sys
sys.path.insert(0, str(Path('examples').resolve()))
from util import schedule_harvest_areacontrol, compile_scenario, plot_scenario, run_cbm


def main():
    apply_fresh_style()

    figs_dir = Path('papers/ems/figs')
    tables_dir = Path('papers/ems/tables')
    ensure_dir(figs_dir)
    ensure_dir(tables_dir)

    # Build ForestModel using the same parameters/data as the 031 notebook
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

    # Schedule harvesting via heuristic to have content to pass to libCBM
    schedule = schedule_harvest_areacontrol(fm)

    # Export schedule for Woodstock parity (space-delimited)
    schedule_path = Path('examples/data/woodstock_model_files/tsa24_clipped.sch')
    schedule_lines = []
    for record in schedule:
        parts = []
        for value in record:
            if isinstance(value, tuple):
                parts.extend(str(v) for v in value)
            else:
                parts.append(str(value))
        schedule_lines.append(" ".join(parts))
    schedule_path.write_text("\n".join(schedule_lines) + "\n")

    # Compile scenario and save flows table
    df = compile_scenario(fm)
    df.to_csv(tables_dir / 'scenario_flows.csv', index=False)

    # Update Woodstock parity placeholder with WS3 totals
    parity_path = tables_dir / 'woodstock_parity_placeholder.csv'
    if parity_path.exists():
        parity_df = pd.read_csv(parity_path)
        totals = {
            'total_harvest_area_kha': df['oha'].sum(),
            'total_harvest_volume_Mm3': df['ohv'].sum(),
            'total_growing_stock_units': df['ogs'].sum(),
        }
        for metric, value in totals.items():
            mask = parity_df['metric'] == metric
            parity_df.loc[mask, 'ws3_value'] = value
        parity_df.to_csv(parity_path, index=False)

    # Plot F4a-like flows (harvest area/volume, stock)
    fig, ax = plot_scenario(df)
    fig.savefig(figs_dir / 'f4a_harvest_and_stock.png', dpi=300)
    plt.close(fig)

    # Prepare libCBM SIT
    disturbance_type_mapping = [
        {"user_dist_type": "harvest", "default_dist_type": "Clearcut harvesting without salvage"},
        {"user_dist_type": "fire", "default_dist_type": "Wildfire"},
    ]
    for dtype_key in fm.dtypes:
        fm.dt(dtype_key).last_pass_disturbance = (
            "fire" if dtype_key[2] == dtype_key[4] else "harvest"
        )

    sit_config, sit_tables = fm.to_cbm_sit(
        softwood_volume_yname="swdvol",
        hardwood_volume_yname="hwdvol",
        admin_boundary="British Columbia",
        eco_boundary="Montane Cordillera",
        disturbance_type_mapping=disturbance_type_mapping,
    )

    # Run libCBM for 200 years as in the notebook
    n_steps = 200
    cbm_output = run_cbm(sit_config, sit_tables, n_steps, plot=False)

    # Extract annual carbon stocks from util.run_cbm output
    # The util.run_cbm returns a CBMOutput object; we reconstruct totals similarly
    from libcbm.model.cbm.cbm_output import CBMOutput
    pi = cbm_output.classifiers.to_pandas().merge(
        cbm_output.pools.to_pandas(), left_on=["identifier", "timestep"], right_on=["identifier", "timestep"]
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

    annual_carbon_stocks = pd.DataFrame({
        'Year': pi['timestep'],
        'Biomass': pi[biomass_pools].sum(axis=1),
        'DOM': pi[dom_pools].sum(axis=1),
        'Total Ecosystem': pi[biomass_pools + dom_pools].sum(axis=1),
    }).groupby('Year').sum().reset_index()

    annual_carbon_stocks.to_csv(tables_dir / 'annual_carbon_stocks.csv', index=False)

    # Plot F4b-like carbon stocks
    fig, ax = plt.subplots()
    sns.lineplot(data=annual_carbon_stocks.melt(id_vars='Year', var_name='Pool', value_name='Stock'),
                 x='Year', y='Stock', hue='Pool')
    ax.set_xlim(0, n_steps)
    ax.set_ylabel('Carbon stock (tC equivalent)')
    ax.set_title('Annual carbon stocks (libCBM)')
    fig.savefig(figs_dir / 'f4b_carbon_stocks.png', dpi=300)
    plt.close(fig)

    print("Done. Wrote figures to:", figs_dir)
    print("Tables to:", tables_dir)


if __name__ == '__main__':
    main()
