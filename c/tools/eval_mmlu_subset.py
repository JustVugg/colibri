#!/usr/bin/env python3
"""Rough catastrophic-forgetting probe: MMLU-subset accuracy, base vs adapter.

Not a leaderboard run — a *delta* signal at streamed-CPU speeds. One question =
one forward: the prompt force-closes the think block and ends with "Answer:",
the engine generates 2 greedy tokens (DRAFT=0), and the first A–D letter in
them is the answer. Same protocol for base and adapter, so whatever gap appears
is the adapter's doing. Questions are sampled evenly across MMLU subjects with
a fixed seed (HF datasets-server, cached in /tmp, nothing committed).

  python3 tools/eval_mmlu_subset.py --model ~/models/glm52_i4 [--adapter DIR]
      [--n 25] [--seed 0] [--out results.json]

Prints per-question lines and a final accuracy; diff two runs for the delta.
"""
import argparse, json, os, random, re, subprocess, sys, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--binary", default="./colibri")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cache", default="/tmp/mmlu_subset_cache.json")
ap.add_argument("--out", default=None)
args = ap.parse_args()

# ---- fetch + cache a pool of questions (deterministic subset from it) ----
if not os.path.exists(args.cache):
    rows = []
    for off in (0, 2000, 4000, 6000, 8000):   # spread over the test split
        url = ("https://datasets-server.huggingface.co/rows?dataset=cais%2Fmmlu"
               f"&config=all&split=test&offset={off}&length=100")
        with urllib.request.urlopen(url, timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    json.dump(rows, open(args.cache, "w"))
pool = json.load(open(args.cache))

rng = random.Random(args.seed)
by_subj = {}
for q in pool:
    by_subj.setdefault(q["subject"], []).append(q)
subjects = sorted(by_subj)
rng.shuffle(subjects)
sample = []
while len(sample) < args.n:
    for s in subjects:
        if by_subj[s] and len(sample) < args.n:
            sample.append(by_subj[s].pop(rng.randrange(len(by_subj[s]))))

LET = "ABCD"
def make_prompt(q):
    ch = "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(q["choices"]))
    return ("[gMASK]<sop><|system|>Answer with a single letter only."
            f"<|user|>{q['question']}\n{ch}\nAnswer with A, B, C or D."
            "<|assistant|><think></think>Answer:")

def run_one(prompt):
    env = dict(os.environ, SNAP=os.path.expanduser(args.model), PROMPT=prompt,
               NGEN="2", TEMP="0", DRAFT="0")
    env.pop("ADAPTER", None)
    if args.adapter: env["ADAPTER"] = args.adapter
    out = subprocess.run([args.binary, "64"], env=env, capture_output=True,
                         text=True, timeout=1800).stdout
    # generated text = everything after the prompt echo, before the stats block
    m = re.search(r"Answer:(.*?)(?:\n---|\[t=|$)", out, re.S)
    tail = m.group(1) if m else out[-200:]
    lm = re.search(r"\b([ABCD])\b", tail)
    return (lm.group(1) if lm else None), tail.strip()[:40]

res, correct, answered = [], 0, 0
for i, q in enumerate(sample):
    letter, raw = run_one(make_prompt(q))
    gold = LET[q["answer"]]
    ok = letter == gold
    answered += letter is not None
    correct += ok
    print(f"[{i+1:2d}/{len(sample)}] {q['subject'][:28]:28s} gold={gold} "
          f"got={letter or '?'} {'OK ' if ok else 'no '} | {raw!r}", flush=True)
    res.append({"subject": q["subject"], "gold": gold, "got": letter, "raw": raw})

acc = correct / len(sample)
print(f"\naccuracy: {correct}/{len(sample)} = {acc:.1%} "
      f"(valid letter emitted: {answered}/{len(sample)}) | "
      f"adapter={args.adapter or 'OFF'} seed={args.seed}")
if args.out:
    json.dump({"adapter": args.adapter, "seed": args.seed, "acc": acc,
               "answered": answered, "n": len(sample), "results": res},
              open(args.out, "w"), indent=1)
