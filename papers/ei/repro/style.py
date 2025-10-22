import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Basic FRESH palette derived from the provided PNG (manually defined for reproducibility)
FRESH_PALETTE = [
    "#1b4965",  # deep blue
    "#5fa8d3",  # light blue
    "#62b6cb",  # teal-ish
    "#a7c957",  # green
    "#ffb703",  # amber
    "#e76f51",  # orange-red
]


def apply_fresh_style():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette(FRESH_PALETTE)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": True,
    })


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
