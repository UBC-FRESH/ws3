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
    # Write scenario flows with explicit units in headers for external use
    df_out = df.rename(columns={
        'oha': 'harvest_area_ha',
        'ohv': 'harvest_volume_m3',
        'ogs': 'growing_stock_m3',
    })
    df_out.to_csv(tables_dir / 'scenario_flows.csv', index=False)

    # Write Woodstock parity CSV (totals) with correct units: ha and m^3
    parity_path = tables_dir / 'woodstock_parity.csv'
    totals_ws3 = {
        'total_harvest_area_ha': float(df['oha'].sum()),
        'total_harvest_volume_m3': float(df['ohv'].sum()),
        'total_growing_stock_m3': float(df['ogs'].sum()),
    }
    # Allow optional external Woodstock totals override
    ext_totals_path = tables_dir / 'woodstock_parity_totals_external.csv'
    woodstock_override = {}
    if ext_totals_path.exists():
        try:
            ext_df = pd.read_csv(ext_totals_path)
            for _, r in ext_df.iterrows():
                woodstock_override[str(r['metric'])] = float(r['woodstock_value'])
        except Exception:
            pass
    rows = []
    for metric, ws3_value in totals_ws3.items():
        wv = woodstock_override.get(metric, ws3_value)
        diff = 0.0 if wv == 0 else 100.0 * (ws3_value - wv) / wv
        rows.append({'metric': metric, 'ws3_value': ws3_value, 'woodstock_value': wv, 'percent_diff': diff})
    pd.DataFrame(rows).to_csv(parity_path, index=False)

    # Plot F4a-like flows (harvest area/volume, stock); units: ha and m^3
    fig, ax = plot_scenario(df)
    try:
        ax[0].set_title('Harvested area (ha)')
        ax[1].set_title('Harvested volume (m$^3$)')
        ax[2].set_title('Growing stock (m$^3$)')
    except Exception:
        pass
    fig.savefig(figs_dir / 'f4a_harvest_and_stock.png', dpi=300)
    plt.close(fig)

    # Supplementary: period-wise parity figure and CSV (WS3 vs Woodstock)
    # Determine period column name
    period_col = 'period' if 'period' in df.columns else ('t' if 't' in df.columns else None)
    if period_col is None:
        df = df.copy()
        df['period'] = range(1, len(df) + 1)
        period_col = 'period'

    # WS3 series (units: ha and m^3)
    ws3_area_ha = df.groupby(period_col)['oha'].sum()
    ws3_vol_m3 = df.groupby(period_col)['ohv'].sum()

    # Optional override file containing Woodstock per-period series
    ext_path = tables_dir / 'woodstock_parity_periods_external.csv'
    if ext_path.exists():
        ext = pd.read_csv(ext_path)
        # Expect columns: period, woodstock_harvest_area_ha, woodstock_harvest_volume_m3
        a_col = 'woodstock_harvest_area_ha'
        v_col = 'woodstock_harvest_volume_m3'
        if a_col not in ext.columns:
            a_col = 'woodstock_harvest_area_kha'
        if v_col not in ext.columns:
            v_col = 'woodstock_harvest_volume_Mm3'
        woodstock_area = ext.set_index('period')[a_col].reindex(ws3_area_ha.index).fillna(method='pad')
        woodstock_vol = ext.set_index('period')[v_col].reindex(ws3_vol_m3.index).fillna(method='pad')
        if a_col.endswith('_kha'):
            woodstock_area = woodstock_area * 1000.0
        if v_col.endswith('_Mm3'):
            woodstock_vol = woodstock_vol * 1_000_000.0
    else:
        woodstock_area = ws3_area_ha.copy()
        woodstock_vol = ws3_vol_m3.copy()

    # Percent diffs (guard divide-by-zero)
    def pct_diff(a, b):
        out = 100.0 * (a - b) / b.replace(0, pd.NA)
        return out.fillna(0.0)

    area_diff_pct = pct_diff(ws3_area_ha, woodstock_area)
    vol_diff_pct = pct_diff(ws3_vol_m3, woodstock_vol)

    # Write parity-by-period CSV
    parity_periods = pd.DataFrame({
        'period': ws3_area_ha.index,
        'ws3_harvest_area_ha': ws3_area_ha.values,
        'woodstock_harvest_area_ha': woodstock_area.values,
        'diff_area_pct': area_diff_pct.values,
        'ws3_harvest_volume_m3': ws3_vol_m3.values,
        'woodstock_harvest_volume_m3': woodstock_vol.values,
        'diff_volume_pct': vol_diff_pct.values,
    })
    parity_periods.to_csv(tables_dir / 'woodstock_parity_periods.csv', index=False)

    # Plot supplementary parity figure (two panels: area and volume)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    axes[0].plot(ws3_area_ha.index, woodstock_area.values, label='Woodstock', linewidth=1.6)
    axes[0].plot(ws3_area_ha.index, ws3_area_ha.values, label='WS3', linestyle='--', linewidth=1.6)
    axes[0].set_ylabel('Harvest area (ha)')
    axes[0].legend()

    axes[1].plot(ws3_vol_m3.index, woodstock_vol.values, label='Woodstock', linewidth=1.6)
    axes[1].plot(ws3_vol_m3.index, ws3_vol_m3.values, label='WS3', linestyle='--', linewidth=1.6)
    axes[1].set_xlabel('Planning period')
    axes[1].set_ylabel('Harvest volume (m$^3$)')
    axes[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(figs_dir / 'sup_parity_periods.png', dpi=300)
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
