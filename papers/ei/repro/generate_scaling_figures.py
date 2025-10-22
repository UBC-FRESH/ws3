#!/usr/bin/env python3
"""Render scaling figures from perf_scaling.csv for the EMS manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from style import apply_fresh_style, ensure_dir


CSV_PATH = Path("papers/ems/tables/perf_scaling.csv")
FIGS_DIR = Path("papers/ems/figs")


def _load_perf() -> pd.DataFrame | None:
    if not CSV_PATH.exists():
        print(f"Scaling CSV not found at {CSV_PATH}; skipping figure generation.")
        return None
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("Scaling CSV is empty; skipping figure generation.")
        return None
    # Ensure combinations are ordered by complexity for plotting
    if "n_dt_age" in df.columns:
        df = df.sort_values("n_dt_age")
    return df


def _format_axis(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)


def plot_schedule_runtime(df: pd.DataFrame):
    heur = df[df["mode"] == "heuristic"]
    if heur.empty:
        print("No heuristic rows in perf_scaling.csv; skipping schedule runtime plot.")
        return

    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(
        heur["n_dt_age"],
        heur["sched_s"],
        marker="o",
        label="Heuristic (1 worker)",
    )

    lp = df[df["mode"] == "lp"].copy()
    if not lp.empty:
        for metric, linestyle in [("lp_build_s", "-"), ("lp_solve_s", "--")]:
            if metric not in lp or lp[metric].isna().all():
                continue
            for workers, marker in [(1, "s"), (16, "^")]:
                sub = lp[lp["workers"] == workers]
                if sub.empty or sub[metric].isna().all():
                    continue
                ax.plot(
                    sub["n_dt_age"],
                    sub[metric],
                    marker=marker,
                    linestyle=linestyle,
                    label=f"LP {'build' if metric=='lp_build_s' else 'solve'} ({workers} workers)",
                )

    _format_axis(ax, "dtype-age combinations", "Runtime (s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_dir(FIGS_DIR)
    out_path = FIGS_DIR / "scaling_schedule_runtime.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_spatial_runtime(df: pd.DataFrame):
    heur = df[df["mode"] == "heuristic"]
    if heur.empty or heur["spatial_s"].isna().all():
        print("No heuristic spatial rows; skipping spatial runtime plot.")
        return

    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(
        heur["n_dt_age"],
        heur["spatial_s"],
        marker="o",
        label="Spatial allocation (serial)",
    )
    _format_axis(ax, "dtype-age combinations", "Runtime (s)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = FIGS_DIR / "scaling_spatial_runtime.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_peak_memory(df: pd.DataFrame):
    heur = df[df["mode"] == "heuristic"]
    if heur.empty or heur["mem_peak_gb"].isna().all():
        print("No heuristic memory data; skipping peak memory plot.")
        return

    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    ax.plot(
        heur["n_dt_age"],
        heur["mem_peak_gb"],
        marker="o",
        label="Heuristic peak RSS",
    )

    lp = df[df["mode"] == "lp"].copy()
    if not lp.empty:
        for workers, marker in [(1, "s"), (16, "^")]:
            sub = lp[(lp["workers"] == workers)]
            if sub.empty or sub["mem_lp_build_gb"].isna().all():
                continue
            ax.plot(
                sub["n_dt_age"],
                sub["mem_lp_build_gb"],
                marker=marker,
                linestyle="-",
                label=f"LP build peak RSS ({workers} workers)",
            )

    _format_axis(ax, "dtype-age combinations", "Peak RSS (GB)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = FIGS_DIR / "scaling_peak_mem.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    df = _load_perf()
    if df is None:
        return
    plot_schedule_runtime(df)
    plot_spatial_runtime(df)
    plot_peak_memory(df)


if __name__ == "__main__":
    main()
