#!/usr/bin/env python3
import os
import json
import re
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


def load_woodstock_everything(path: Path) -> pd.DataFrame:
    """Parse Woodstock ``everything.txt`` output for harvest area/volume by period."""
    period_re = re.compile(r"period\s*=\s*(\d+)", re.IGNORECASE)
    value_re = re.compile(r"([0-9][0-9,]*\.?[0-9]*)")
    records = {}
    current_period = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(';'):
                continue
            match = period_re.search(line)
            if match:
                current_period = int(match.group(1))
                records.setdefault(current_period, {})
                continue
            if current_period is None:
                continue
            if line.lower().startswith('harvested_volume'):
                match_value = value_re.search(line)
                if not match_value:
                    continue
                value = float(match_value.group(1).replace(',', ''))
                records[current_period]['harvested_volume'] = value
            elif line.lower().startswith('harvested_area'):
                match_value = value_re.search(line)
                if not match_value:
                    continue
                value = float(match_value.group(1).replace(',', ''))
                records[current_period]['harvested_area'] = value
            elif line.lower().startswith('growing_stock'):
                match_value = value_re.search(line)
                if not match_value:
                    continue
                value = float(match_value.group(1).replace(',', ''))
                records[current_period]['growing_stock'] = value
    if not records:
        raise ValueError(f"No harvest records parsed from {path}")
    df = (
        pd.DataFrame.from_dict(records, orient='index')
        .reset_index()
        .rename(columns={'index': 'period'})
        .sort_values('period')
    )
    expected_cols = {'harvested_area', 'harvested_volume', 'growing_stock'}
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in Woodstock output: {missing_cols}")
    return df


def load_woodstock_schedule(path: Path, nthemes: int, age_multiplier: int = 1):
    """Parse a Woodstock ``.seq`` schedule into ``ws3``'s schedule tuple format."""
    schedule = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(';'):
                continue
            tokens = re.split(r"\s+", line.lower())
            if len(tokens) < nthemes + 4:
                continue
            dtype_key = tuple(tokens[:nthemes])
            age = int(tokens[nthemes]) * age_multiplier
            area = float(tokens[nthemes + 1].replace(',', ''))
            acode = tokens[nthemes + 2]
            period = int(tokens[nthemes + 3])
            etype = tokens[nthemes + 4] if len(tokens) > nthemes + 4 else ''
            schedule.append((dtype_key, age, area, acode, period, etype))
    if not schedule:
        raise ValueError(f"No schedule entries parsed from {path}")
    schedule.sort(key=lambda item: item[4])
    return schedule


def apply_prescriptive_schedule(fm: ws3.forest.ForestModel, schedule):
    """Apply a prescriptive schedule to a ``ForestModel`` with strict error reporting."""
    fm.reset()
    current_period = None
    missing_total = 0.0
    for dtype_key, age, area, acode, period, _etype in schedule:
        if current_period is None:
            current_period = period
        if period != current_period:
            fm.commit_actions(current_period)
            current_period = period
        err, missing_area, _ = fm.apply_action(
            dtype_key,
            acode,
            period,
            age,
            area,
            override_operability=True,
            fuzzy_age=False,
            recourse_enabled=False,
            compile_t_ycomps=True,
            compile_c_ycomps=True,
        )
        if err:
            raise RuntimeError(
                f"Failed to apply schedule entry {dtype_key} period {period} age {age}: error {err}"
            )
        if missing_area:
            missing_total += missing_area
    if current_period is not None:
        fm.commit_actions(current_period)
    if missing_total > 0:
        raise RuntimeError(f"Schedule application lost {missing_total} area")


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
        model_path="examples/data/woodstock_model_files_tsa24_clipped",
        base_year=base_year,
        horizon=horizon,
        period_length=period_length,
        max_age=max_age,
    )
    period_factor = fm.period_length
    fm.import_landscape_section()
    fm.import_areas_section(convert_periods_to_years=period_factor)
    fm.import_yields_section(convert_periods_to_years=period_factor)
    fm.import_actions_section(convert_periods_to_years=period_factor)
    fm.import_transitions_section(convert_periods_to_years=period_factor)
    fm.initialize_areas()
    fm.add_null_action()
    fm.reset_actions()

    # Import prescriptive Woodstock schedule and apply it verbatim
    schedule_path = Path('examples/data/woodstock_model_files_tsa24_clipped/tsa24_clipped.seq')
    schedule = load_woodstock_schedule(
        schedule_path,
        nthemes=fm.nthemes(),
        age_multiplier=fm.period_length,
    )
    apply_prescriptive_schedule(fm, schedule)

    from util import compile_scenario, plot_scenario, run_cbm

    # Compile scenario and save flows table
    df = compile_scenario(fm)
    # Write scenario flows with explicit units in headers for external use
    df_out = df.rename(columns={
        'oha': 'harvest_area_ha',
        'ohv': 'harvest_volume_m3',
        'ogs': 'growing_stock_m3',
    })
    df_out.to_csv(tables_dir / 'scenario_flows.csv', index=False)

    woodstock_output_path = Path('examples/data/woodstock_model_files_tsa24_clipped/everything.txt')
    woodstock_df = load_woodstock_everything(woodstock_output_path).rename(columns={
        'harvested_area': 'woodstock_harvest_area_ha',
        'harvested_volume': 'woodstock_harvest_volume_m3',
        'growing_stock': 'woodstock_growing_stock_m3',
    })

    # Write Woodstock parity CSV (totals) with correct units: ha and m^3
    parity_path = tables_dir / 'woodstock_parity.csv'
    totals_ws3 = {
        'total_harvest_area_ha': float(df['oha'].sum()),
        'total_harvest_volume_m3': float(df['ohv'].sum()),
        'total_growing_stock_m3': float(df['ogs'].sum()),
    }
    totals_woodstock = {
        'total_harvest_area_ha': float(woodstock_df['woodstock_harvest_area_ha'].sum()),
        'total_harvest_volume_m3': float(woodstock_df['woodstock_harvest_volume_m3'].sum()),
        'total_growing_stock_m3': float(woodstock_df['woodstock_growing_stock_m3'].sum()),
    }
    rows = []
    def round_two(value):
        rounded = round(value, 2)
        return 0.0 if abs(rounded) < 1e-9 else rounded

    for metric, ws3_value in totals_ws3.items():
        woodstock_value = totals_woodstock[metric]
        diff = 0.0 if woodstock_value == 0 else 100.0 * (ws3_value - woodstock_value) / woodstock_value
        rows.append({
            'metric': metric,
            'ws3_value': round_two(ws3_value),
            'woodstock_value': round_two(woodstock_value),
            'percent_diff': round_two(diff),
        })
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
    ws3_stock_m3 = df.groupby(period_col)['ogs'].sum()

    woodstock_area = woodstock_df.set_index('period')['woodstock_harvest_area_ha'].reindex(ws3_area_ha.index)
    woodstock_vol = woodstock_df.set_index('period')['woodstock_harvest_volume_m3'].reindex(ws3_vol_m3.index)
    woodstock_stock = woodstock_df.set_index('period')['woodstock_growing_stock_m3'].reindex(ws3_stock_m3.index)
    if woodstock_area.isna().any() or woodstock_vol.isna().any() or woodstock_stock.isna().any():
        missing_periods = sorted({
            *woodstock_area[woodstock_area.isna()].index.tolist(),
            *woodstock_vol[woodstock_vol.isna()].index.tolist(),
            *woodstock_stock[woodstock_stock.isna()].index.tolist(),
        })
        raise ValueError(f"Woodstock outputs missing data for periods: {missing_periods}")

    # Percent diffs (guard divide-by-zero)
    def pct_diff(a, b):
        out = 100.0 * (a - b) / b.replace(0, pd.NA)
        return out.fillna(0.0)

    area_diff_pct = pct_diff(ws3_area_ha, woodstock_area)
    vol_diff_pct = pct_diff(ws3_vol_m3, woodstock_vol)
    stock_diff_pct = pct_diff(ws3_stock_m3, woodstock_stock)

    # Write parity-by-period CSV
    parity_periods = pd.DataFrame({
        'period': ws3_area_ha.index,
        'ws3_harvest_area_ha': ws3_area_ha.values,
        'woodstock_harvest_area_ha': woodstock_area.values,
        'diff_area_pct': area_diff_pct.values,
        'ws3_harvest_volume_m3': ws3_vol_m3.values,
        'woodstock_harvest_volume_m3': woodstock_vol.values,
        'diff_volume_pct': vol_diff_pct.values,
        'ws3_growing_stock_m3': ws3_stock_m3.values,
        'woodstock_growing_stock_m3': woodstock_stock.values,
        'diff_growing_stock_pct': stock_diff_pct.values,
    })
    numeric_cols = [col for col in parity_periods.columns if col != 'period']
    parity_periods[numeric_cols] = parity_periods[numeric_cols].apply(lambda col: col.map(round_two))
    parity_periods.to_csv(tables_dir / 'woodstock_parity_periods.csv', index=False)

    # Summary error statistics for reporting
    area_abs_diff = (ws3_area_ha - woodstock_area).abs()
    vol_abs_diff = (ws3_vol_m3 - woodstock_vol).abs()
    stats_rows = [
        {
            'metric': 'harvest_area_total_diff_ha',
            'value': float((ws3_area_ha - woodstock_area).sum()),
        },
        {
            'metric': 'harvest_volume_total_diff_m3',
            'value': float((ws3_vol_m3 - woodstock_vol).sum()),
        },
        {
            'metric': 'growing_stock_total_diff_m3',
            'value': float((ws3_stock_m3 - woodstock_stock).sum()),
        },
        {
            'metric': 'harvest_area_mae_ha',
            'value': float(area_abs_diff.mean()),
        },
        {
            'metric': 'harvest_volume_mae_m3',
            'value': float(vol_abs_diff.mean()),
        },
        {
            'metric': 'growing_stock_mae_m3',
            'value': float((ws3_stock_m3 - woodstock_stock).abs().mean()),
        },
        {
            'metric': 'harvest_area_mape_pct',
            'value': float(area_diff_pct.abs().mean()),
        },
        {
            'metric': 'harvest_volume_mape_pct',
            'value': float(vol_diff_pct.abs().mean()),
        },
        {
            'metric': 'growing_stock_mape_pct',
            'value': float(stock_diff_pct.abs().mean()),
        },
        {
            'metric': 'harvest_area_max_abs_pct_diff',
            'value': float(area_diff_pct.abs().max()),
        },
        {
            'metric': 'harvest_volume_max_abs_pct_diff',
            'value': float(vol_diff_pct.abs().max()),
        },
        {
            'metric': 'growing_stock_max_abs_pct_diff',
            'value': float(stock_diff_pct.abs().max()),
        },
    ]
    pd.DataFrame(stats_rows).to_csv(tables_dir / 'woodstock_parity_stats.csv', index=False)

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

    fair_rows = [
        {
            'principle': 'Findable',
            'checklist_focus': 'Persistent identifier; indexed repository',
            'evidence': 'Zenodo DOI 10.5281/zenodo.17219651; tagged GitHub releases; PyPI project metadata',
        },
        {
            'principle': 'Accessible',
            'checklist_focus': 'License and public access',
            'evidence': 'MIT license; public GitHub repository; PyPI wheels; bundled reproduction package assets',
        },
        {
            'principle': 'Interoperable',
            'checklist_focus': 'Open formats and connectors',
            'evidence': 'Woodstock LAN/ARE/YLD import; libCBM SIT export; CSV/GeoTIFF outputs; documented API',
        },
        {
            'principle': 'Reusable',
            'checklist_focus': 'Documentation, provenance, validation',
            'evidence': 'ReadTheDocs site; notebooks/examples; CI-tested pytest suite; deterministic repro scripts',
        },
    ]
    pd.DataFrame(fair_rows).to_csv(tables_dir / 'fair_checklist.csv', index=False)

    print("Done. Wrote figures to:", figs_dir)
    print("Tables to:", tables_dir)


if __name__ == '__main__':
    main()
