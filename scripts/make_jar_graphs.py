"""Graph the jar->hnmod head-to-head (agent vs Colibri) from
results/eval_jar_to_hnmod.json. Writes results/bench_jar_to_hnmod.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
with open(os.path.join(RESULTS, "eval_jar_to_hnmod.json"), encoding="utf-8") as f:
    D = json.load(f)

plt.rcParams.update({
    "figure.facecolor": "#0f1420", "axes.facecolor": "#161d2e",
    "axes.edgecolor": "#2a3550", "axes.labelcolor": "#dfe6f5",
    "text.color": "#dfe6f5", "xtick.color": "#aab6d0", "ytick.color": "#aab6d0",
    "grid.color": "#243049", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 130,
})
AGENT, COLIBRI = "#4f8ef7", "#f7a24f"

names = [n for n in ("agent", "colibri") if n in D]
fig, axs = plt.subplots(1, 2, figsize=(15, 6),
                        gridspec_kw={"width_ratios": [1, 1.6]})

# Overall score
vals = [D[n]["passed"] for n in names]
tot = D[names[0]]["total"]
bars = axs[0].bar([n.capitalize() for n in names], vals,
                  color=[AGENT if n == "agent" else COLIBRI for n in names], width=0.5)
for b, v in zip(bars, vals):
    axs[0].text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}/{tot}",
                ha="center", fontweight="bold", fontsize=13)
axs[0].set_ylim(0, tot + 1.5)
axs[0].set_ylabel("Spec checks passed")
axs[0].set_title("jar -> .hnmod converter: overall")
axs[0].grid(axis="y", alpha=0.35)
axs[0].set_axisbelow(True)

# Per-check heatmap
checks = list(D[names[0]]["checks"].keys())
ys = range(len(checks))
for xi, n in enumerate(names):
    for yi, c in enumerate(checks):
        ok = D[n]["checks"].get(c, False)
        axs[1].add_patch(plt.Rectangle((xi - 0.4, yi - 0.4), 0.8, 0.8,
                                       color=("#3ecf8e" if ok else "#f76f6f")))
axs[1].set_yticks(list(ys))
axs[1].set_yticklabels(checks, fontsize=9)
axs[1].set_xticks(range(len(names)))
axs[1].set_xticklabels([n.capitalize() for n in names])
axs[1].invert_yaxis()
axs[1].set_xlim(-0.6, len(names) - 0.4)
axs[1].set_ylim(len(checks) - 0.5, -0.6)
axs[1].set_title("Per-check pass (green) / fail (red)")
for s in ("top", "right", "left", "bottom"):
    axs[1].spines[s].set_visible(False)

fig.suptitle("Harder head-to-head: convert any Minecraft .jar to .hnmod (shared spec, no reference .hnmod read)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
p = os.path.join(RESULTS, "bench_jar_to_hnmod.png")
fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", p)