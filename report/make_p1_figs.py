"""Figures and numbers for the Part 1 report, from the synced run outputs.

    python report/make_p1_figs.py [results_dir]     (default: colab_results/part1_vlm_dora)
"""
import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else "colab_results/part1_vlm_dora"
HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("full", "Full FT"), ("lora", "LoRA"), ("dora", "DoRA")]
COL = {"full": "#444444", "lora": "#2b7bba", "dora": "#d1495b"}


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def stats(run):
    p = os.path.join(RES, f"stats_{run}.json")
    return json.load(open(p))["stats"] if os.path.exists(p) else None


def final_rows(st, tower, proj):
    last = max(r["step"] for r in st)
    return sorted((r for r in st if r["step"] == last and r["tower"] == tower and r["proj"] == proj),
                  key=lambda r: r["layer"])


def figure():
    """Row = tower (q_proj, the module of DoRA Fig. 2), column = method; points = layers at the final checkpoint."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), gridspec_kw={"hspace": 0.32, "wspace": 0.28})
    for i, tower in enumerate(["vision", "text"]):
        for j, (m, label) in enumerate(METHODS):
            ax = axes[i, j]
            rows = final_rows(stats(m), tower, "q_proj")
            n = len(rows)
            x, y = [r["dD"] for r in rows], [r["dM"] for r in rows]
            ex, ey = math.floor(math.log10(max(x))), math.floor(math.log10(max(y)))
            xs_, ys_ = [v / 10 ** ex for v in x], [v / 10 ** ey for v in y]
            sc = ax.scatter(xs_, ys_, c=[r["layer"] / (n - 1) for r in rows], cmap="viridis", vmin=0, vmax=1,
                            s=22, edgecolor="k", lw=0.3)
            mx, my = sum(xs_) / n, sum(ys_) / n
            sl = sum((a - mx) * (b - my) for a, b in zip(xs_, ys_)) / sum((a - mx) ** 2 for a in xs_)
            xl = [min(xs_), max(xs_)]
            ax.plot(xl, [my + sl * (v - mx) for v in xl], "--", color=COL[m], lw=1.2)
            ax.text(0.97, 0.95, f"r = {pearson(x, y):+.2f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=COL[m], fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"ΔD  (×1e{ex})", fontsize=8)
            ax.set_ylabel(f"ΔM  (×1e{ey})", fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=10)
        axes[i, 0].annotate(f"{tower} q_proj\n({n} layers)", xy=(-0.42, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=9)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.015)
    cb.set_label("relative depth (0 = first layer)", fontsize=8)
    out = os.path.join(HERE, "fig_p1_final_ckpt.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


