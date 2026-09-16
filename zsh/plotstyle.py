"""Shared figure style for the article (static, print, light surface)."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
FULL_W, HALF_W = 6.9, 3.4  # inches (MDPI text width ~17.5 cm)


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "legend.frameon": False,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "text.color": INK,
    })
