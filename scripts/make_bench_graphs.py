"""Render the agent-vs-Colibri + NPU-embedding benchmark into charts.

Reads results/bench_agent_vs_colibri.json and writes:
  results/bench_overview.png   (6-panel dashboard)
  results/bench_pass_at_1.png  (correctness comparison)
  results/bench_colibri_perf.png (latency + throughput)
  results/bench_npu_embed.png  (retrieval + similarity)

Run: .venv/Scripts/python.exe scripts/make_bench_graphs.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
with open(os.path.join(RESULTS, "bench_agent_vs_colibri.json"), encoding="utf-8") as f:
    D = json.load(f)

COD = D["coding"]
EMB = D["embedding_npu"]
PRIOR = D["prior"]
TASKS = COD["tasks"]

AGENT = "#4f8ef7"
COLIBRI = "#f7a24f"
STUDENT = "#9b8cf7"
NPU = "#3ecf8e"
DIFF_COLOR = {"easy": "#3ecf8e", "medium": "#f7c94f", "hard": "#f76f6f"}

plt.rcParams.update({
    "figure.facecolor": "#0f1420", "axes.facecolor": "#161d2e",
    "axes.edgecolor": "#2a3550", "axes.labelcolor": "#dfe6f5",
    "text.color": "#dfe6f5", "xtick.color": "#aab6d0", "ytick.color": "#aab6d0",
    "grid.color": "#243049", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "figure.dpi": 130,
})


def style(ax):
    ax.grid(axis="y", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pass_at_1(ax):
    a = COD["agent_summary"]["pass_at_1"] * 100
    c = COD["colibri_summary"]["pass_at_1"] * 100
    st = PRIOR.get("student_3b_summary") or {}
    s = (st.get("exec_passed", 0) / st.get("total", 1)) * 100 if st else None
    labels = ["Agent\n(reference)", "Colibri 30B\n(same 10 tasks)"]
    vals = [a, c]
    colors = [AGENT, COLIBRI]
    notes = [f"{COD['agent_summary']['passed']}/{COD['agent_summary']['total']}",
             f"{COD['colibri_summary']['passed']}/{COD['colibri_summary']['total']}"]
    if s is not None:
        labels.append("Student 3B base\n(5-task baseline)")
        vals.append(s)
        colors.append(STUDENT)
        notes.append(f"{st['exec_passed']}/{st['total']}")
    bars = ax.bar(labels, vals, color=colors, width=0.6, zorder=3)
    for b, v, n in zip(bars, vals, notes):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%\n({n})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Execution pass@1 (%)")
    ax.set_title("Coding correctness — same exec harness")
    style(ax)


def colibri_latency(ax):
    ids = [t["id"] for t in TASKS]
    times = [t["colibri"]["time_s"] for t in TASKS]
    colors = [DIFF_COLOR[t["difficulty"]] for t in TASKS]
    y = range(len(ids))
    ax.barh(list(y), times, color=colors, zorder=3, height=0.65)
    ax.set_yticks(list(y))
    ax.set_yticklabels(ids, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(times):
        ax.text(v + 0.3, i, f"{v:.1f}s", va="center", fontsize=8.5)
    avg = COD["colibri_perf"]["avg_latency_s"]
    ax.axvline(avg, color="#dfe6f5", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(avg + 0.3, -0.7, f"avg {avg:.1f}s", fontsize=8.5, color="#dfe6f5")
    ax.set_xlabel("Wall-clock latency (s)")
    ax.set_title("Colibri 30B latency per task")
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=DIFF_COLOR[d], label=d) for d in ("easy", "medium", "hard")],
              loc="lower right", fontsize=8, framealpha=0.2)


def colibri_throughput(ax):
    ids = [t["id"] for t in TASKS]
    tps = [t["colibri"]["tokens_per_s"] for t in TASKS]
    colors = [DIFF_COLOR[t["difficulty"]] for t in TASKS]
    bars = ax.bar(range(len(ids)), tps, color=colors, zorder=3, width=0.65)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=40, ha="right", fontsize=8)
    for b, v in zip(bars, tps):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.1f}",
                ha="center", fontsize=8)
    avg = COD["colibri_perf"]["avg_tokens_per_s"]
    ax.axhline(avg, color="#dfe6f5", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(ids) - 0.5, avg + 0.12, f"avg {avg:.2f}", ha="right", fontsize=8.5)
    ax.set_ylabel("Completion tokens / s")
    ax.set_title("Colibri 30B generation throughput")
    style(ax)


def npu_retrieval(ax):
    metrics = {
        "Retrieval\naccuracy@1": EMB["accuracy_at_1"],
        "Retrieval\nMRR": EMB["mrr"],
        "NPU vs CPU\ncosine": PRIOR.get("npu_min_cosine_vs_cpu") or 0,
    }
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=[NPU, NPU, "#4f8ef7"], width=0.55, zorder=3)
    for b, v in zip(bars, metrics.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (1.0 = perfect)")
    ax.set_title(f"Intel NPU embeddings — {EMB['n_queries']} queries / {EMB['n_docs']} docs")
    style(ax)


def npu_semantics(ax):
    sem = PRIOR.get("npu_semantics") or {}
    labels = list(sem.keys())
    vals = list(sem.values())
    colors = [NPU if v > 0.5 else "#f76f6f" for v in vals]
    bars = ax.barh(labels, vals, color=colors, zorder=3, height=0.55)
    ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Cosine similarity (NPU)")
    ax.set_title("NPU embedding semantic geometry")
    ax.grid(axis="x", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def summary_panel(ax):
    ax.axis("off")
    cp = COD["colibri_perf"]
    lines = [
        ("Hardware lanes", True),
        ("  NVIDIA GPU  →  Colibri dense trunk + KV", False),
        ("  Intel NPU   →  bge-base embeddings (RAG)", False),
        ("  NVMe SSD    →  MoE experts streamed on demand", False),
        ("", False),
        ("Coding (10 tasks, exec pass@1)", True),
        (f"  Agent reference : {COD['agent_summary']['passed']}/{COD['agent_summary']['total']}", False),
        (f"  Colibri 30B     : {COD['colibri_summary']['passed']}/{COD['colibri_summary']['total']}", False),
        (f"  Colibri avg     : {cp['avg_latency_s']:.1f}s/task, {cp['avg_tokens_per_s']:.2f} tok/s", False),
        ("", False),
        ("NPU embeddings", True),
        (f"  Retrieval acc@1 : {EMB['accuracy_at_1']:.2f}   MRR {EMB['mrr']:.2f}", False),
        (f"  Throughput      : {EMB['throughput_emb_per_s']:.2f} emb/s", False),
        (f"  Query latency   : {EMB['avg_query_latency_s']:.2f}s", False),
        (f"  NPU≈CPU cosine  : {PRIOR.get('npu_min_cosine_vs_cpu')}", False),
    ]
    y = 0.97
    for text, bold in lines:
        ax.text(0.02, y, text, va="top", fontsize=10.5,
                fontweight="bold" if bold else "normal",
                color="#3ecf8e" if bold else "#dfe6f5",
                family="monospace")
        y -= 0.058 if text else 0.03
    ax.set_title("Result summary", loc="left")


def save(fig, name):
    p = os.path.join(RESULTS, name)
    fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", p)


# 6-panel dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
fig.suptitle("Agent vs Colibri 30B  +  Intel-NPU Embeddings — local benchmark",
             fontsize=17, fontweight="bold", y=1.00)
pass_at_1(axes[0][0])
colibri_latency(axes[0][1])
colibri_throughput(axes[0][2])
npu_retrieval(axes[1][0])
npu_semantics(axes[1][1])
summary_panel(axes[1][2])
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "bench_overview.png")

# Individual charts
fig, ax = plt.subplots(figsize=(7.5, 5))
pass_at_1(ax)
fig.tight_layout()
save(fig, "bench_pass_at_1.png")

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
colibri_latency(axs[0])
colibri_throughput(axs[1])
fig.tight_layout()
save(fig, "bench_colibri_perf.png")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
npu_retrieval(axs[0])
npu_semantics(axs[1])
fig.tight_layout()
save(fig, "bench_npu_embed.png")

print("DONE")