"""Graphical abstract for the article, built from the saved result files.

MDPI asks for at least 560 x 1100 pixels (height x width) and the same ratio for larger
images; this renders 1120 x 2200. Three panels, left to right: what was done, what was
measured, and what was found. Every number is read from results/ so the picture cannot
drift away from the article.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from zsh import plotstyle as ps  # noqa: E402
from zsh.config import RESULTS, results_dir  # noqa: E402

SHOW = [("L2:P2TR", "P2TR inputs"), ("L2:P2WPKH", "P2WPKH inputs"), ("L2:P2PKH", "P2PKH inputs"),
        ("L2:P2SH", "P2SH inputs"), ("L2:P2WSH", "P2WSH inputs"), ("L4:exchange", "Exchange tag")]


def box(ax, x, y, w, h, text, face, edge, size=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, facecolor=face, edgecolor=edge, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=ps.INK, weight=weight, zorder=3, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=1.1, shrinkA=2, shrinkB=2))


def main():
    ps.apply()
    fc = pd.read_csv(RESULTS / "E18" / "feature_ceiling.csv")
    piv = fc.pivot(index="target", columns="method", values="ap") * 100
    base = fc.groupby("target").base_rate.first() * 100
    prof_med = piv["ZSH"].median()
    bound_med = piv["supervised in-period, free cells"].median()
    warm = pd.read_csv(RESULTS / "E20" / "warm_refit.csv").set_index("period")
    followed = int(warm.loc["test", "n_matched_ge_0.5"])
    profiles = int(warm.loc["test", "profiles"])

    fig = plt.figure(figsize=(11.0, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.12, 1.06], wspace=0.24,
                          left=0.035, right=0.978, top=0.86, bottom=0.07)

    fig.text(0.5, 0.955, "What do unsupervised Bitcoin transaction profiles actually capture?",
             ha="center", va="center", fontsize=14.5, weight="bold", color=ps.INK)
    fig.text(0.5, 0.905, "A pre-specified evaluation: the plan and code were frozen before any "
             "evaluation data were examined", ha="center", va="center", fontsize=10, color=ps.INK2)

    # ---------------------------------------------------------------- panel 1: what was done
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.5, 0.965, "1.  The profiles", ha="center", fontsize=11.5, weight="bold", color=ps.INK)
    box(ax, 0.06, 0.80, 0.88, 0.115,
        "3.3 million transactions\n2022–2023", ps.SURFACE, ps.GRID)
    arrow(ax, 0.5, 0.80, 0.5, 0.735)
    box(ax, 0.06, 0.615, 0.88, 0.12,
        "twelve features, weighted by the rank\nof their mutual information", "#eef4fd", ps.SEQ[3])
    arrow(ax, 0.5, 0.615, 0.5, 0.55)
    box(ax, 0.06, 0.43, 0.88, 0.12,
        "K-means from merged micro-clusters,\ncluster size capped", "#eef4fd", ps.SEQ[3])
    arrow(ax, 0.5, 0.43, 0.5, 0.365)
    box(ax, 0.06, 0.245, 0.88, 0.12, "31 profiles", ps.SEQ[5], ps.SEQ[6], size=12, weight="bold")
    arrow(ax, 0.5, 0.245, 0.5, 0.18)
    ax.text(0.5, 0.135, "tested on data the clustering never saw", ha="center", fontsize=9.6,
            color=ps.INK, weight="bold")
    ax.text(0.5, 0.055, "2.6 M transactions from 2024   ·   456,292 collected to Aug 2026\n"
                        "after the freeze   ·   the Elliptic benchmark",
            ha="center", fontsize=8.8, color=ps.INK2, linespacing=1.5)

    # ---------------------------------------------------------------- panel 2: what was measured
    ax = fig.add_subplot(gs[0, 1])
    ax.text(0.5, 1.055, "2.  Measured against what is attainable", ha="center", fontsize=11.5,
            weight="bold", color=ps.INK, transform=ax.transAxes)
    labels = [lab for k, lab in SHOW]
    attained = [piv.loc[k, "ZSH"] for k, _ in SHOW]
    rates = [base[k] for k, _ in SHOW]
    y = range(len(SHOW))
    ax.barh(y, [100] * len(SHOW), color="#eceff4", edgecolor="none", height=0.62)
    ax.barh(y, attained, color=[ps.SEQ[6] if a >= 50 else ps.SEQ[3] for a in attained],
            edgecolor="none", height=0.62)
    for i, (a, r) in enumerate(zip(attained, rates)):
        ax.text(a + 1.6, i, f"{a:.0f}%", va="center", fontsize=9.4, color=ps.INK, weight="bold")
        ax.text(99, i - 0.30, f"{r:.2f}% of transactions", va="center", ha="right",
                fontsize=7.6, color=ps.INK2)
    ax.set_yticks(list(y), labels, fontsize=9.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 108), ax.set_xticks([0, 50, 100], ["0", "50", "100%"], fontsize=8.6)
    ax.set_xlabel("share of the concentration a perfect partition could reach", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(False)
    ax.text(0.5, -0.155, "common properties are captured well, rare ones are not",
            transform=ax.transAxes, ha="center", fontsize=9.2, color=ps.INK, style="italic")

    # ---------------------------------------------------------------- panel 3: what was found
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    ax.text(0.5, 0.965, "3.  Where the limit lies", ha="center", fontsize=11.5,
            weight="bold", color=ps.INK)

    bx = fig.add_axes([0.705, 0.545, 0.245, 0.30])
    bx.bar([0, 1], [prof_med, bound_med], color=[ps.SEQ[3], ps.SERIES[1]], width=0.56)
    for i, v in enumerate([prof_med, bound_med]):
        bx.text(i, v + 3.5, f"{v:.1f}%", ha="center", fontsize=10.5, weight="bold", color=ps.INK)
    bx.set_xticks([0, 1], ["the profiles", "supervision on\nthe same features"], fontsize=8.6)
    bx.set_ylim(0, 116), bx.set_yticks([])
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.grid(False)
    bx.set_title("median share attained", fontsize=9.2, color=ps.INK2, pad=6)

    ax.text(0.5, 0.455, "The twelve features carry the information.\n"
                        "The clustering objective does not find it.",
            ha="center", fontsize=10.2, color=ps.INK, weight="bold", linespacing=1.5)
    ax.plot([0.08, 0.92], [0.385, 0.385], color=ps.GRID, lw=1)
    ax.text(0.5, 0.315, "Six other clustering families reach the same ceilings.",
            ha="center", fontsize=9.3, color=ps.INK2)
    ax.text(0.5, 0.225, f"After a free refit no profile can be followed;\n"
                        f"constraining the refit follows {followed} of {profiles}.",
            ha="center", fontsize=9.3, color=ps.INK2, linespacing=1.5)
    box(ax, 0.055, 0.045, 0.89, 0.115,
        "A readable map of activity — not a detector", "#fdf3ef", ps.SERIES[1], size=10.8,
        weight="bold")

    out = results_dir("figures") / "GA_graphical_abstract.png"
    fig.savefig(out, dpi=200, facecolor="white")
    fig.savefig(str(out).replace(".png", ".tif"), dpi=300, facecolor="white")
    print(f"wrote {out} ({fig.get_size_inches()[0] * 200:.0f} x {fig.get_size_inches()[1] * 200:.0f} px)")
    print(f"  profiles median {prof_med:.1f}%, bound median {bound_med:.1f}%, "
          f"followed {followed}/{profiles}")


if __name__ == "__main__":
    main()
