#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from style import apply_fresh_style, ensure_dir

def main():
    apply_fresh_style()
    figs_dir = Path('papers/ems/figs')
    ensure_dir(figs_dir)

    # Schematic grid: 100x100 cells, ~1% highlighted
    rng = np.random.default_rng(42)
    n = 100
    grid = np.zeros((n, n), dtype=int)

    # Select ~1% harvested cells to illustrate small footprint
    harvested_mask = rng.random((n, n)) < 0.01
    grid[harvested_mask] = 1

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(["#e5e5e5", "#1b4965"])  # light grey, FRESH deep blue
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest', origin='upper')
    ax.set_title('Spatial allocation (schematic stub): harvested cells in one period (~1%)')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(figs_dir / 'f3_spatial_stub.png', dpi=300)
    plt.close(fig)
    print('Wrote', figs_dir / 'f3_spatial_stub.png')

if __name__ == '__main__':
    main()