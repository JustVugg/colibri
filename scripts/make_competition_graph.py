"""Graph the jar->hnmod competition: spec compliance + closeness to the user's
real port. Reads results/eval_jar_to_hnmod.json and
results/competition_vs_reference.json -> results/bench_competition.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
with open(os.path.join(RESULTS, "competition_vs_reference.json"), encoding="utf-8") as f:
    C = json.load(f)
with open(os.path.join(RESULTS, "eval_jar_to_hnmod.json"), encoding="utf-8") as f:
    S = json.load(f)

plt.rcParams.update({
    "figure.facecolor": "#0f1420", "axes.facecolor": "#161d2e",
    "axes.edgecolor": "#2a3550", "axes.labelcolor": "#dfe6f5",
    "text.color": "#dfe6f5", "xtick.color": "#aab6d0", "ytick.color": "#aab6d0",
    "grid.color": "#243049", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 130,
})
COL = {"agent": "#4f8ef7", "colibri": "#f7a24f"}
names = list(C["contestants"].keys())
winner = C["winner_by_closeness"]

fig, axs = plt.subplots(1, 3, figsize=(17, 5.6),
                        gridspec_kw={"width_ratios": [1, 1, 1.5]})

# Spec compliance
vals = [S.get(n, {}).get("passed", 0) for n in names]
tot = S.get(names[0], {}).get("total", 14)
b = axs[0].bar([n.capitalize() for n in names], vals, color=[COL[n] for n in names], width=0.5)
for r, v in zip(b, vals):
    axs[0].text(r.get_x() + r.get_width() / 2, v + 0.2, f"{v}/{tot}",
                ha="center", fontweight="bold", fontsize=12)
axs[0].set_ylim(0, tot + 1.5)
axs[0].set_ylabel("Checks passed")
axs[0].set_title("Shared-spec compliance (blind)")
axs[0].grid(axis="y", alpha=0.35)
axs[0].set_axisbelow(True)

# Closeness to real port
vals = [C["contestants"][n]["closeness"] for n in names]
b = axs[1].bar([n.capitalize() for n in names], vals, color=[COL[n] for n in names], width=0.5)
for r, v in zip(b, vals):
    axs[1].text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontweight="bold", fontsize=12)
axs[1].set_ylim(0, 1.1)
axs[1].set_ylabel("Closeness (0..1)")
axs[1].set_title(f"Closeness to real port — winner: {winner.capitalize()}")
axs[1].grid(axis="y", alpha=0.35)
axs[1].set_axisbelow(True)

# Per-metric closeness
metrics = list(C["contestants"][names[0]]["metrics"].keys()) or ["none"]
import numpy as np
x = np.arange(len(metrics))
w = 0.38
for i, n in enumerate(names):
    mv = C["contestants"][n]["metrics"]
    axs[2].bar(x + (i - 0.5) * w, [mv.get(m, 0) for m in metrics], w,
               color=COL[n], label=n.capitalize())
axs[2].set_xticks(x)
axs[2].set_xticklabels(metrics, rotation=35, ha="right", fontsize=8.5)
axs[2].set_ylim(0, 1.15)
axs[2].set_ylabel("Score")
axs[2].set_title("Structural similarity vs real .hnmod")
axs[2].legend(framealpha=0.2)
axs[2].grid(axis="y", alpha=0.35)
axs[2].set_axisbelow(True)

ref = C["reference"]
fig.suptitle(f"jar -> .hnmod competition vs user's port  (reference: {ref['n_code']} Rust mixins, "
             f"{ref['n_members']} members, id={ref['manifest'].get('id')})",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
p = os.path.join(RESULTS, "bench_competition.png")
fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", p)