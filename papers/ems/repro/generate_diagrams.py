#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from pathlib import Path
from style import apply_fresh_style, ensure_dir


def draw_box(ax, xy, text, width=2.6, height=1.0):
    x, y = xy
    rect = Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10)
    return (x + width/2, y + height/2)


def draw_arrow(ax, start, end):
    ax.add_patch(FancyArrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                            width=0.02, length_includes_head=True, head_width=0.15, head_length=0.25,
                            color='black'))


def fig_architecture(out_path):
    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Left column: Inputs
    b_inventory = draw_box(ax, (0.5, 6.0), 'Inventory\n(Yields, Areas, Themes)')
    b_actions = draw_box(ax, (0.5, 4.5), 'Actions &\nTransitions')
    b_config = draw_box(ax, (0.5, 3.0), 'Scenario\nConfig (YAML/CSV)')

    # Middle: WS3 core
    b_forest = draw_box(ax, (4.0, 6.0), 'WS3 ForestModel')
    b_opt = draw_box(ax, (4.0, 4.5), 'Optimization\n(PuLP / HiGHS / Gurobi)')
    b_sim = draw_box(ax, (4.0, 3.0), 'Simulation &\nEvaluation')

    # Right: External couplings
    b_cbm = draw_box(ax, (7.5, 5.25), 'libCBM\n(carbon pools/flux)')
    b_spatial = draw_box(ax, (7.5, 3.75), 'Spatial allocation\n(rasterio/GeoTIFF)')
    b_outputs = draw_box(ax, (10.0, 4.5), 'Outputs\n(plots, tables)')

    # Arrows (inputs to forest)
    draw_arrow(ax, (1.8, 6.5), (4.0, 6.5))
    draw_arrow(ax, (1.8, 5.5), (4.0, 5.5))
    draw_arrow(ax, (1.8, 4.0), (4.0, 4.0))

    # Internal flow
    draw_arrow(ax, (5.3, 6.5), (5.3, 5.75))
    draw_arrow(ax, (5.3, 4.5), (5.3, 3.75))

    # Couplings
    draw_arrow(ax, (5.3, 5.25), (7.5, 5.75))  # Forest->libCBM
    draw_arrow(ax, (5.3, 3.75), (7.5, 4.25))  # Sim->Spatial

    # Outputs
    draw_arrow(ax, (8.8, 5.5), (10.0, 5.0))
    draw_arrow(ax, (8.8, 4.25), (10.0, 4.5))

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig_workflow(out_path):
    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)

    x = 0.5
    centers = []
    labels = [
        'Load inputs',
        'Build ForestModel',
        'Schedule (heuristic or LP)',
        'Compile SIT (to_cbm_sit)',
        'Run libCBM',
        'Spatial allocation',
        'Plots & Tables'
    ]
    for i, lab in enumerate(labels):
        c = draw_box(ax, (x, 1.1), lab, width=2.2, height=0.9)
        centers.append(c)
        x += 2.2 + 0.6

    for i in range(len(centers)-1):
        draw_arrow(ax, centers[i], centers[i+1])

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    figs_dir = Path('papers/ems/figs')
    ensure_dir(figs_dir)

    fig_architecture(figs_dir / 'f1_architecture.png')
    fig_workflow(figs_dir / 'f2_workflow.png')
    # Graphical abstract: reuse architecture but smaller
    fig_architecture(figs_dir / 'graphical_abstract.png')
    print('Wrote diagrams to', figs_dir)


if __name__ == '__main__':
    main()
