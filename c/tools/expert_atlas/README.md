# Expert Atlas — what does each of the 19,456 experts actually do?

Probe harness for #175. Runs a set of topic-tagged prompts, dumps each run's expert-routing
histogram, and turns them into a per-expert topic-affinity vector.

```bash
cd c
export COLI_MODEL=/path/to/glm52_i4
./tools/expert_atlas/sweep.sh                             # 30 probes (10 topics x 3 prompts)
python3 tools/expert_atlas/analyze.py  --stats atlas_out/stats --out atlas_out/experts.json \
        --web web/dist/experts.json                       # optional: feed the web dashboard Atlas
python3 tools/expert_atlas/validate.py atlas_out/stats 200 # leave-one-prompt-out check
```

`--web` writes the same atlas in the shape the web dashboard consumes (the Atlas galaxy and the
Brain hover tooltips): keyed `"layer:expert"` with `affinity`/`entropy`/`top`/`label`. It replaces
the retired `tools/expert_atlas.py`, whose API-driven probing ran through a live server and was
exposed to exactly the traps above (server-side `--topp`, speculative drafts, shared `.coli_usage`).

## Read this before you trust any atlas

Four things silently corrupt this measurement. The sweep script controls all of them; if you
roll your own, don't skip them.

| trap | effect | control |
|---|---|---|
| **`--topp`** | prunes experts by cumulative probability — measured: it hides **38% of the distinct experts** (7,587 → 4,687). It is also the *recommended speed setting*. | `TOPP=0` |
| **speculative drafts** | `eusage` is incremented inside `moe()`, *before* verification, so **rejected** drafts count. Those are experts routed for text the model never emitted. | `MTP=0 DRAFT=0` |
| **`.coli_usage`** | is loaded at startup and accumulates, so a naive `STATS` dump contains **all prior history**, not this run. | remove per run (script backs it up and restores) |
| **autocorrelation** | routing inside one run is highly correlated — the same context routes to the same experts token after token. An expert firing 38 times during one prompt is **one** observation, not 38. Chi-square/entropy on raw selections will certify single-prompt flukes as perfect specialists. | `analyze.py` requires the affinity to **replicate across a category's independent prompts** |

The CUDA expert tier is also not run-to-run deterministic, so the sweep uses `--gpu none`. Tier
config only decides *where weights live*, not what the router picks, so this costs nothing.

## Method

`analyze.py`:

1. `n[e][c]` — selections of expert *e* while running category *c*
2. `f[e][c] = n[e][c] / N[c]` — normalise by **category size** (prefill routes the prompt too, so a
   verbose category would otherwise look busier)
3. `p(c|e)` — renormalise into a topic distribution per expert, i.e. base-rate corrected. Ranking
   by raw count instead just rediscovers which experts are popular in general.
4. `spec(e) = 1 − H(p(c|e)) / log C` — 0 = generalist, 1 = fires for exactly one topic
5. **replication gate** — an expert is only a candidate specialist for *c* if it fires in ≥2 of *c*'s
   independent prompts

`validate.py` — leave-one-prompt-out. Build each category's top-K specialist set from its *other*
prompts, then check which set the held-out prompt's routing actually lands in. If specialisation
were an artifact of prompt wording, the held-out prompt would not prefer its own category.

## Result on GLM-5.2 744B int4 (Zen5, CPU routing path)

- **Leave-one-prompt-out: 29/30 = 96.7%** (chance 10%). Specialisation is a property of the topic,
  not of prompt wording.
- The single miss is instructive: `写一首关于秋天的短诗` ("write a short poem about autumn") is
  classified **poetry**, not Chinese — routing follows the **task** over the **language**.
- **Only 7.9% of experts are strong specialists** (spec ≥ 0.5). The "one expert = one topic"
  picture is wrong for ~92% of them.
- Specialisation **rises with depth**: layer 3 ≈ 0.07 (generalist, token/syntax level) → layers
  18–58 ≈ 0.19–0.27.
- The replication gate removed **587** experts that looked like flawless specialists on one prompt.

## Extending it

`probes.json` is the whole probe set — add categories and prompts. Use **3+ prompts per category
with varied phrasing**, or the replication gate has nothing to check and you are back to measuring
one prompt. Keep prompt lengths in a similar band across categories.
