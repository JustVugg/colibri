#!/usr/bin/env python3
"""Leave-one-prompt-out validation of the Expert Atlas (#175).

"Replicates across the 3 prompts I picked" is not the same as "generalises". The atlas is
only real if a specialist set learned from SOME prompts predicts routing on a prompt it has
never seen.

Protocol, for every category c and every held-out prompt h of c:
  1. build c's top-K specialist set from c's OTHER prompts only (h excluded entirely)
  2. do the same for all other categories (using ALL their prompts — they never saw h either)
  3. on the held-out run h, measure what share of routing selections land in each set
  4. the atlas works if c's own set wins on h

If specialisation were an artifact of prompt wording, the held-out prompt would not prefer
its own category's set. Chance is 1/C.
"""
import glob, json, os, sys
from collections import defaultdict

STATS = sys.argv[1] if len(sys.argv) > 1 else "stats"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 200      # specialists per category

runs, tot = {}, {}
for path in sorted(glob.glob(os.path.join(STATS, "*.txt"))):
    cat, idx = os.path.basename(path)[:-4].rsplit("_", 1)
    d = {}
    t = 0
    for line in open(path):
        p = line.split()
        if len(p) == 3:
            d[(int(p[0]), int(p[1]))] = int(p[2])
            t += int(p[2])
    runs[(cat, idx)] = d
    tot[(cat, idx)] = t

cats = sorted({c for c, _ in runs})
idxs = {c: sorted(i for cc, i in runs if cc == c) for c in cats}
C = len(cats)


def specialists(cat, exclude):
    """Top-K experts by lift for `cat`, computed WITHOUT the excluded run."""
    share = defaultdict(lambda: defaultdict(float))
    for c in cats:
        used = [i for i in idxs[c] if not (c == cat and i == exclude)]
        for i in used:
            for k, n in runs[(c, i)].items():
                share[k][c] += n / max(1, tot[(c, i)]) / len(used)
    scored = []
    for k, per in share.items():
        s = sum(per.values())
        if s <= 0:
            continue
        p = per[cat] / s
        if per[cat] > 0:
            scored.append((p, k))
    scored.sort(reverse=True)
    return {k for _, k in scored[:K]}


print(f"leave-one-prompt-out, {C} categories, top-{K} specialists per category")
print(f"chance = {100.0/C:.1f}%\n")
hits = 0
trials = 0
for c in cats:
    for h in idxs[c]:
        sets = {cc: specialists(cc, h if cc == c else None) for cc in cats}
        held = runs[(c, h)]
        htot = max(1, tot[(c, h)])
        scores = {cc: sum(held.get(k, 0) for k in sets[cc]) / htot for cc in cats}
        win = max(scores, key=scores.get)
        ok = win == c
        hits += ok
        trials += 1
        own = 100 * scores[c]
        best_other = 100 * max(v for cc, v in scores.items() if cc != c)
        print(f"  {c:<12} prompt {h}  own-set {own:5.2f}%  best-other {best_other:5.2f}%  "
              f"-> {'HIT ' if ok else 'MISS'} (predicted {win})")
print(f"\naccuracy: {hits}/{trials} = {100*hits/trials:.1f}%   (chance {100.0/C:.1f}%)")
