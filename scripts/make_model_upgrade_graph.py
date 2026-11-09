"""Render the local-model upgrade benchmark (30B coder vs Qwen3.6-35B-A3B).

Reads results/bench_best_*.json and writes results/bench_model_upgrade.png.
Run: .venv/Scripts/python.exe scripts/make_model_upgrade_graph.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

BG = "#15120F"
PANEL = "#1B1714"
BORDER = "#3C322A"
TEXT = "#F2EEE8"
MUTED = "#A99D91"
ACCENT = "#E9865D"
GREEN = "#5EC98F"
RED = "#EF615C"
AMBER = "#E9B45D"


def load(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


D30 = load("bench_best_coder30b_hard.json")
D36 = load("bench_best_qwen36_hard.json")
B30 = load("bench_best_coder30b.json")
B36 = load("bench_best_qwen36_base.json")
R30 = load("bench_best_coder30b_repair3.json")
T3 = load("bench_best_qwen36_think3.json")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": BORDER, "axes.labelcolor": TEXT,
    "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": "#2A231D", "font.size": 10.5, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 150,
    "font.family": "DejaVu Sans",
})


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.86, bottom=0.09,
                      hspace=0.42, wspace=0.22)
fig.text(0.06, 0.955, "Colibri model upgrade — execution-verified",
         fontsize=19, fontweight="bold", color=TEXT)
fig.text(0.06, 0.915,
         "Same 12 hidden-test coding tasks · greedy decoding · pass = all unit tests executed green",
         fontsize=10.5, color=MUTED)

# --- A: hard-suite pass by difficulty -------------------------------------
ax = fig.add_subplot(gs[0, 0])
labels = ["30B coder", "Qwen3.6-35B"]
medium_pass = [D30["summary"]["by_difficulty"]["medium"]["passed"],
               D36["summary"]["by_difficulty"]["medium"]["passed"]]
hard_pass = [D30["summary"]["by_difficulty"]["hard"]["passed"],
             D36["summary"]["by_difficulty"]["hard"]["passed"]]
medium_fail = [6 - v for v in medium_pass]
hard_fail = [6 - v for v in hard_pass]
x = range(2)
w = 0.5
ax.bar(x, medium_pass, w, color=GREEN, label="medium passed")
ax.bar(x, hard_pass, w, bottom=medium_pass, color="#3E8F68", label="hard passed")
ax.bar(x, medium_fail, w, bottom=[a + b for a, b in zip(medium_pass, hard_pass)],
       color=RED, label="failed")
for i in x:
    total = medium_pass[i] + hard_pass[i]
    ax.text(i, total + 0.18, f"{total}/12", ha="center", fontsize=12,
            fontweight="bold", color=TEXT)
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 13.6)
ax.set_ylabel("tasks")
ax.set_title("A · Hard suite (12 tasks)")
ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=MUTED)
style(ax)

# --- B: the 3 former failures ----------------------------------------------
ax = fig.add_subplot(gs[0, 1])
names = ["30B\ninitial", "30B +\nrepair", "Qwen3.6\nno-think"]
vals = [sum(r["initial_passed"] for r in R30["rows"]), R30["summary"]["passed"],
        sum(1 for r in D36["rows"] if r["id"] in
            ("expr_eval", "parse_csv_line", "lexicographic_topo"))]
colors = [RED, AMBER, GREEN]
bars = ax.bar(names, vals, 0.5, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v}/3", ha="center",
            fontsize=12, fontweight="bold", color=TEXT)
ax.set_ylim(0, 3.6)
ax.set_ylabel("tasks passed")
ax.set_title("B · Tasks the 30B failed (expr_eval, CSV, lexicographic topo)")
style(ax)

# --- C: quality vs throughput ----------------------------------------------
ax = fig.add_subplot(gs[1, 0])
pts = [
    ("30B coder", D30["summary"]["pass_at_1"], D30["summary"]["tokens_per_s"], ACCENT),
    ("Qwen3.6-35B", D36["summary"]["pass_at_1"], D36["summary"]["tokens_per_s"], GREEN),
]
for name, pr, tps, c in pts:
    ax.scatter([tps], [pr * 100], s=180, color=c, edgecolor=BG, linewidth=2, zorder=3)
    ax.annotate(f"{name}\n{tps:.1f} tok/s · {pr*100:.0f}%",
                (tps, pr * 100), textcoords="offset points", xytext=(12, -6),
                fontsize=9.5, color=TEXT)
ax.set_xlim(6, 18)
ax.set_ylim(66, 106)
ax.set_xlabel("decode throughput (tokens/s)")
ax.set_ylabel("hard-suite pass rate (%)")
ax.set_title("C · Faster and more correct")
ax.axvspan(6, 7.5, color=RED, alpha=0.06)
ax.text(6.35, 69, "Qwen3.8-Flash-Next\n0.21 tok/s — 74× slower,\nquality run not practical",
        fontsize=8.5, color=RED)
style(ax, "both")

# --- D: time per solved task -----------------------------------------------
ax = fig.add_subplot(gs[1, 1])
groups = ["base suite", "hard suite"]
v30 = [B30["summary"]["wall_s"] / B30["summary"]["passed"],
       D30["summary"]["wall_s"] / D30["summary"]["passed"]]
v36 = [B36["summary"]["wall_s"] / B36["summary"]["passed"],
       D36["summary"]["wall_s"] / D36["summary"]["passed"]]
xg = range(2)
bw = 0.34
b1 = ax.bar([i - bw / 2 for i in xg], v30, bw, color=ACCENT, label="30B coder")
b2 = ax.bar([i + bw / 2 for i in xg], v36, bw, color=GREEN, label="Qwen3.6-35B")
for bars, vals in ((b1, v30), (b2, v36)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}s", ha="center",
                fontsize=9.5, color=TEXT)
ax.set_xticks(list(xg), groups)
ax.set_ylabel("seconds per solved task")
ax.set_title("D · Wall-clock cost of a verified solution")
ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
style(ax)

fig.text(0.06, 0.025,
         "Measured on RTX 4060 Laptop 8 GB + Intel Arc iGPU + 33.8 GB RAM · "
         "Qwen3.6-35B-A3B int4 gs64, all 10,240 experts resident · 2026-11-09",
         fontsize=8.5, color=MUTED)

out = os.path.join(RESULTS, "bench_model_upgrade.png")
fig.savefig(out, facecolor=BG)
print("saved", out)
