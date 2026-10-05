"""Shared figure style for the article (static, print, light surface).

Journal requirements this encodes: Times New Roman, nothing below 12 pt as printed,
600 dpi. "As printed" is the point that matters: the article embeds a full-width figure
at 6.5 in, so a figure authored wider than that is scaled down in the document and its
12 pt type arrives smaller. FULL_W therefore matches the embedded width exactly, and
MIN_PT is the floor for every label, tick and annotation in the figure scripts.
"""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"

# inches, equal to the width each figure is embedded at, so authored pt == printed pt
FULL_W, HALF_W = 6.5, 3.6
MIN_PT = 12.0          # nothing in a figure may be set smaller than this
FONT = "Times New Roman"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "serif", "font.serif": [FONT, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": MIN_PT,
        "axes.titlesize": MIN_PT, "axes.titleweight": "bold", "axes.labelsize": MIN_PT,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
        "legend.fontsize": MIN_PT, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "text.color": INK,
    })
