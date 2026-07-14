#!/usr/bin/env python3
"""Expert Atlas (#175): measure per-expert topic affinity by diffing .coli_usage
across themed probe batches, served through a running colibri API server.

Protocol per category: snapshot .coli_usage -> send probes -> snapshot again;
the delta is that category's expert-activation spectrum. One engine load total.

Output: experts.json — for every (layer, expert): counts per category,
normalized affinity, entropy, and a "specialist" label when one topic dominates.

Usage (server already running with the model):
    python3 tools/expert_atlas.py --api http://127.0.0.1:8000 \
        --usage /path/to/model/.coli_usage --out experts.json --ngen 64
"""
import argparse, json, math, time, urllib.request

PROBES = {
    "code": [
        "Write a Python function that parses a CSV file and returns a dict keyed by the first column.",
        "Explain the difference between a mutex and a semaphore, with a C example.",
        "Refactor this into idiomatic Rust: for i in range(len(xs)): total += xs[i] * 2",
    ],
    "math": [
        "Prove that the square root of 2 is irrational.",
        "Compute the derivative of x^3 * ln(x) and explain each step.",
        "A fair die is rolled 4 times. What is the probability of at least one six?",
    ],
    "chinese": [
        "请用中文解释一下什么是光合作用,以及它对地球生态系统的重要性。",
        "把这句话翻译成中文并解释语法:The early bird catches the worm.",
        "写一段关于秋天的短文,一百字左右。",
    ],
    "english_prose": [
        "Write a vivid paragraph describing an old lighthouse keeper watching a storm arrive.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
        "Continue this story: The last train left the station, and Maria realized her mistake.",
    ],
    "science": [
        "Explain how mRNA vaccines work at the cellular level.",
        "Why is the sky blue during the day but red at sunset?",
        "Describe the life cycle of a massive star, from formation to supernova.",
    ],
    "law": [
        "Explain the difference between a patent, a trademark, and a copyright.",
        "What are the key elements required to form a legally binding contract?",
        "Summarize what 'due process' means in constitutional law.",
    ],
    "poetry": [
        "Write a short poem about a hummingbird in the style of Emily Dickinson.",
        "Compose a haiku about winter rain, then explain its imagery.",
        "Write four rhyming lines about the sea at night.",
    ],
    "structured": [
        'Convert to JSON: name Alice, age 30, hobbies reading and chess, address 5 Oak St.',
        "Write a SQL query returning the top 5 customers by total order value, with the schema you assume.",
        "Write a regex that matches ISO-8601 dates and explain each part.",
    ],
    "translation": [
        "Translate into French, German and Spanish: 'Knowledge is the only treasure that grows when shared.'",
        "Translate this Italian sentence to English and comment on nuance: 'In bocca al lupo per domani.'",
        "Translate into Japanese: 'The meeting has been moved to next Tuesday afternoon.'",
    ],
    "casual": [
        "Hey! Any tips for staying awake during boring afternoon meetings?",
        "What should I cook tonight? I have eggs, rice, tomatoes and some cheese.",
        "My friend is always late. How do I tell them it bothers me without being rude?",
    ],
}


def read_usage(path):
    counts = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 3:
                    counts[(int(p[0]), int(p[1]))] = int(p[2])
    except FileNotFoundError:
        pass
    return counts


def diff(after, before):
    return {k: v - before.get(k, 0) for k, v in after.items() if v - before.get(k, 0) > 0}


def chat(api, prompt, ngen):
    body = json.dumps({"model": "glm-5.2-colibri", "stream": False, "max_tokens": ngen,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--usage", required=True, help="path to the model's .coli_usage")
    ap.add_argument("--out", default="experts.json")
    ap.add_argument("--ngen", type=int, default=64)
    a = ap.parse_args()

    spectra = {}
    for cat, prompts in PROBES.items():
        before = read_usage(a.usage)
        t0 = time.time()
        for p in prompts:
            chat(a.api, p, a.ngen)
        time.sleep(2)                      # let the engine flush .coli_usage
        spectra[cat] = diff(read_usage(a.usage), before)
        total = sum(spectra[cat].values())
        print(f"[{cat}] {len(spectra[cat])} experts touched, {total} selections, {time.time()-t0:.0f}s", flush=True)

    cats = list(PROBES.keys())
    experts = {}
    for cat, spec in spectra.items():
        for k, v in spec.items():
            experts.setdefault(k, {c: 0 for c in cats})[cat] = v

    atlas = {}
    for (layer, eid), counts in experts.items():
        total = sum(counts.values())
        if total < 8:
            continue                       # too few observations to characterise
        aff = {c: v / total for c, v in counts.items() if v}
        ent = -sum(p * math.log2(p) for p in aff.values())
        top = max(aff, key=aff.get)
        label = f"specialist: {top}" if aff[top] >= 0.45 and ent < 2.2 else "generalist"
        atlas[f"{layer}:{eid}"] = {"counts": counts, "affinity": {c: round(p, 3) for c, p in aff.items()},
                                   "entropy": round(ent, 2), "top": top, "label": label}

    spec_n = sum(1 for v in atlas.values() if v["label"].startswith("specialist"))
    with open(a.out, "w") as f:
        json.dump({"categories": cats, "ngen": a.ngen, "experts": atlas}, f)
    print(f"\natlas: {len(atlas)} experts characterised, {spec_n} specialists -> {a.out}")


if __name__ == "__main__":
    main()
