"""Compare the 3B coder student before vs after QLoRA fine-tuning on the
held-out execution-verified benchmark (sum_even/fib/palindrome/factorial/flatten
— deliberately NOT in the SFT templates).

Reads results/eval_student_base.json and results/eval_student_tuned.json
(produced by scripts/evaluate.py --hf-model ...), writes
results/eval_student_comparison.json and results/eval_student_before_after.png.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
BASE = os.path.join(RESULTS, "eval_student_base.json")
TUNED = os.path.join(RESULTS, "eval_student_tuned.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def by_id(report):
    return {r["id"]: r for r in report["results"]}


def main():
    base, tuned = load(BASE), load(TUNED)
    b, t = by_id(base), by_id(tuned)
    ids = list(b.keys())

    per_task = []
    for i in ids:
        bp = bool(b[i]["score"].get("exec_passed"))
        tp = bool(t[i]["score"].get("exec_passed")) if i in t else False
        per_task.append({
            "id": i,
            "base_exec_passed": bp,
            "tuned_exec_passed": tp,
            "delta": int(tp) - int(bp),
            "base_time_s": b[i]["time_s"],
            "tuned_time_s": t[i]["time_s"] if i in t else None,
        })

    bs, ts = base["summary"], tuned["summary"]
    comparison = {
        "benchmark": "held-out execution-verified (not in SFT templates)",
        "student": "Qwen2.5-Coder-3B-Instruct",
        "base": bs,
        "tuned": ts,
        "per_task": per_task,
        "exec_passed_delta": ts["exec_passed"] - bs["exec_passed"],
        "winner": ("tuned" if ts["exec_passed"] > bs["exec_passed"]
                   else "base" if bs["exec_passed"] > ts["exec_passed"] else "tie"),
    }
    out_json = os.path.join(RESULTS, "eval_student_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    plt.rcParams.update({
        "figure.facecolor": "#0f1420", "axes.facecolor": "#161d2e",
        "axes.edgecolor": "#2a3550", "axes.labelcolor": "#dfe6f5",
        "text.color": "#dfe6f5", "xtick.color": "#aab6d0", "ytick.color": "#aab6d0",
        "grid.color": "#243049", "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "figure.dpi": 130,
    })
    COL = {"base": "#aab6d0", "tuned": "#5fd08a"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))

    # Panel 1: overall exec pass@1
    vals = [bs["exec_passed"], ts["exec_passed"]]
    tot = bs["total"]
    bars = axs[0].bar(["Base 3B", "Fine-tuned 3B"], vals,
                      color=[COL["base"], COL["tuned"]], width=0.5)
    for r, v in zip(bars, vals):
        axs[0].text(r.get_x() + r.get_width() / 2, v + 0.12, f"{v}/{tot}",
                    ha="center", fontweight="bold", fontsize=13)
    axs[0].set_ylim(0, tot + 1)
    axs[0].set_ylabel("Tasks passed (execution)")
    axs[0].set_title(f"Held-out pass@1 — winner: {comparison['winner']}")
    axs[0].grid(axis="y", alpha=0.35)
    axs[0].set_axisbelow(True)

    # Panel 2: per-task exec pass
    x = np.arange(len(ids))
    w = 0.38
    axs[1].bar(x - w / 2, [int(p["base_exec_passed"]) for p in per_task], w,
               color=COL["base"], label="Base")
    axs[1].bar(x + w / 2, [int(p["tuned_exec_passed"]) for p in per_task], w,
               color=COL["tuned"], label="Fine-tuned")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    axs[1].set_ylim(0, 1.25)
    axs[1].set_yticks([0, 1])
    axs[1].set_ylabel("Exec pass (0/1)")
    axs[1].set_title("Per-task generalization")
    axs[1].legend(framealpha=0.2)
    axs[1].grid(axis="y", alpha=0.35)
    axs[1].set_axisbelow(True)

    # Panel 3: latency
    bt = [p["base_time_s"] for p in per_task]
    tt = [p["tuned_time_s"] or 0 for p in per_task]
    axs[2].bar(x - w / 2, bt, w, color=COL["base"], label="Base")
    axs[2].bar(x + w / 2, tt, w, color=COL["tuned"], label="Fine-tuned")
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    axs[2].set_ylabel("Generation time (s)")
    axs[2].set_title("Latency per task (greedy, 256 tok)")
    axs[2].legend(framealpha=0.2)
    axs[2].grid(axis="y", alpha=0.35)
    axs[2].set_axisbelow(True)

    fig.suptitle("QLoRA distillation: Qwen2.5-Coder-3B student before vs after "
                 f"(110 verified rows, {ts['exec_passed']}/{tot} held-out exec pass)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(RESULTS, "eval_student_before_after.png")
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", out_json)
    print("wrote", out_png)
    print(json.dumps({"base": bs, "tuned": ts,
                      "winner": comparison["winner"],
                      "exec_passed_delta": comparison["exec_passed_delta"]}, indent=2))


if __name__ == "__main__":
    main()