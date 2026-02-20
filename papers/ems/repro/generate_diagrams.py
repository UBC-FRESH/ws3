#!/usr/bin/env python3
"""Generate architecture and workflow diagrams for the EMS manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from style import apply_fresh_style, ensure_dir


def add_box(
    ax,
    center: tuple[float, float],
    text: str,
    width: float = 1.4,
    height: float = 0.58,
    facecolor: str = "#FFFFFF",
    edgecolor: str = "#333333",
    fontsize: int = 10,
):
    """Draw a rounded box centered at *center* and return the bounding box."""
    x0 = center[0] - width / 2
    y0 = center[1] - height / 2
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        linewidth=0.95,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return (x0, y0, width, height)


def edge_point_by_side(bbox: tuple[float, float, float, float], side: str | None) -> tuple[float, float]:
    x0, y0, w, h = bbox
    if side == "left":
        return x0, y0 + h / 2
    if side == "right":
        return x0 + w, y0 + h / 2
    if side == "top":
        return x0 + w / 2, y0 + h
    if side == "bottom":
        return x0 + w / 2, y0
    return x0 + w / 2, y0 + h / 2


def edge_point_direction(bbox: tuple[float, float, float, float], dx: float, dy: float) -> tuple[float, float]:
    x0, y0, w, h = bbox
    cx = x0 + w / 2
    cy = y0 + h / 2
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, cy
    tx = (w / 2) / abs(dx) if dx else float("inf")
    ty = (h / 2) / abs(dy) if dy else float("inf")
    t = min(tx, ty)
    return cx + dx * t, cy + dy * t


def connect_boxes(
    axis,
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    color: str = "#555555",
    curvature: float = 0.0,
    arrowstyle: str = "-|>",
    side_a: str | None = None,
    side_b: str | None = None,
):
    """Draw an arrow between two bounding boxes touching their edges."""
    axc, ayc = bbox_a[0] + bbox_a[2] / 2, bbox_a[1] + bbox_a[3] / 2
    bxc, byc = bbox_b[0] + bbox_b[2] / 2, bbox_b[1] + bbox_b[3] / 2
    dx, dy = bxc - axc, byc - ayc
    if side_a:
        start = edge_point_by_side(bbox_a, side_a)
    else:
        start = edge_point_direction(bbox_a, dx, dy)
    if side_b:
        end = edge_point_by_side(bbox_b, side_b)
    else:
        end = edge_point_direction(bbox_b, -dx, -dy)
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={curvature}",
        arrowstyle=arrowstyle,
        mutation_scale=9,
        linewidth=0.85,
        color=color,
    )
    axis.add_patch(arrow)


def fig_architecture(out_path: Path) -> None:
    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 4.8)

    columns = {
        "inputs": 0.9,
        "core": 2.9,
        "integration": 4.9,
        "outputs": 6.9,
    }

    palette = {
        "inputs": "#E8F1FA",
        "core": "#F4F1FA",
        "integration": "#F9F1E8",
        "outputs": "#E9F6F0",
    }

    header_y = 4.2
    ax.text(columns["inputs"], header_y, "Inputs", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(columns["core"], header_y, "WS3 core", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(columns["integration"], header_y, "Integration", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(columns["outputs"], header_y, "Outputs", ha="center", va="bottom", fontsize=11, fontweight="bold")

    y_positions = [3.2, 1.9, 0.7]

    nodes = {}
    nodes["inventory"] = add_box(
        ax,
        (columns["inputs"], y_positions[0]),
        "Inventory\n(yields, areas, themes)",
        width=1.35,
        height=0.55,
        facecolor=palette["inputs"],
    )
    nodes["actions"] = add_box(
        ax,
        (columns["inputs"], y_positions[1]),
        "Actions &\ntransitions",
        width=1.35,
        height=0.55,
        facecolor=palette["inputs"],
    )
    nodes["config"] = add_box(
        ax,
        (columns["inputs"], y_positions[2]),
        "Scenario config\n(CSV / YAML)",
        width=1.35,
        height=0.55,
        facecolor=palette["inputs"],
    )

    nodes["forest"] = add_box(
        ax,
        (columns["core"], y_positions[0]),
        "ForestModel\n(data API)",
        width=1.45,
        height=0.55,
        facecolor=palette["core"],
    )
    nodes["opt"] = add_box(
        ax,
        (columns["core"], y_positions[1]),
        "Optimization\n(PuLP / HiGHS / Gurobi)",
        width=1.65,
        height=0.58,
        facecolor=palette["core"],
    )
    nodes["sim"] = add_box(
        ax,
        (columns["core"], y_positions[2]),
        "Simulation &\nreporting",
        width=1.45,
        height=0.55,
        facecolor=palette["core"],
    )

    nodes["cbm"] = add_box(
        ax,
        (columns["integration"], 2.6),
        "libCBM\n(carbon pools / flux)",
        width=1.65,
        height=0.58,
        facecolor=palette["integration"],
    )
    nodes["spatial"] = add_box(
        ax,
        (columns["integration"], 1.0),
        "Spatial allocation\n(rasterio / GeoTIFF)",
        width=1.65,
        height=0.58,
        facecolor=palette["integration"],
    )

    nodes["outputs"] = add_box(
        ax,
        (columns["outputs"], 1.8),
        "Dashboards, reports\nAPIs, reproducible assets",
        width=1.75,
        height=0.62,
        facecolor=palette["outputs"],
    )

    connect_boxes(ax, nodes["inventory"], nodes["forest"], curvature=0.0, side_a="right", side_b="left")
    connect_boxes(ax, nodes["actions"], nodes["forest"], curvature=-0.02, side_a="right", side_b="top")
    connect_boxes(ax, nodes["config"], nodes["opt"], curvature=0.02, side_a="right", side_b="left")

    connect_boxes(ax, nodes["forest"], nodes["opt"], curvature=0.0, side_a="bottom", side_b="top")
    connect_boxes(ax, nodes["opt"], nodes["sim"], curvature=0.0, side_a="bottom", side_b="top")

    connect_boxes(ax, nodes["forest"], nodes["cbm"], curvature=0.02, side_a="right", side_b="left")
    connect_boxes(ax, nodes["sim"], nodes["cbm"], curvature=-0.04, side_a="right", side_b="bottom")
    connect_boxes(ax, nodes["sim"], nodes["spatial"], curvature=0.04, side_a="right", side_b="left")

    connect_boxes(ax, nodes["cbm"], nodes["outputs"], curvature=0.05, side_a="right", side_b="top")
    connect_boxes(ax, nodes["spatial"], nodes["outputs"], curvature=-0.05, side_a="right", side_b="bottom")

    ax.text(
        4.25,
        0.05,
        "WS3 coordinates data preparation, optimization, and analysis in a transparent pipeline.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def fig_workflow(out_path: Path) -> None:
    apply_fresh_style()
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.axis("off")
    ax.set_ylim(0, 3)

    steps = [
        "Load inputs",
        "Build ForestModel",
        "Schedule (heuristic or LP)",
        "Compile SIT (to_cbm_sit)",
        "Run libCBM",
        "Spatial allocation",
        "Plots & tables",
    ]

    box_width = 2.0
    box_height = 0.8
    spacing = 0.45
    margin = 0.8

    x = margin + box_width / 2
    centers = []
    for label in steps:
        bbox = add_box(
            ax,
            (x, 1.5),
            label,
            width=box_width,
            height=box_height,
            facecolor="#EDF2FA",
        )
        centers.append(bbox)
        x += box_width + spacing

    for left, right in zip(centers[:-1], centers[1:]):
        connect_boxes(ax, left, right, curvature=0.0, side_a="right", side_b="left")

    last_box = centers[-1]
    right_extent = last_box[0] + last_box[2] + margin
    ax.set_xlim(0, right_extent)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    figs_dir = Path("papers/ems/figs")
    ensure_dir(figs_dir)

    print("Wrote diagrams to", figs_dir)


if __name__ == "__main__":
    main()
