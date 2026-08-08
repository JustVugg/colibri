# CNRE Phase 0: offline residency-policy simulator

This experiment supports [Discussion #884](https://github.com/JustVugg/colibri/discussions/884).
It does not modify an engine or claim a runtime speedup. It replays existing
`ROUTE_TRACE` files through bounded expert caches so weak policies can be
rejected before changing inference code or renting hardware.

## Scope

`c/tools/residency_sim.py` currently models the main GLM CPU-routing path:

- first-seen expert unions for each contiguous `(call, layer)` batch;
- promotion between GLM's 64-expert working-set blocks;
- separate per-layer resident footprint, read bytes, and felt miss cost;
- a fixed expert-residency RAM budget, excluding the rest of process RAM;
- equal slots per layer versus policy-specific, trace-trained layer budgets;
- GLM LRU, synthetic half-capacity pins + LRU, decayed LFU, segmented LRU,
  and decayed-frequency admission.

The half-pinned and alternative admission policies are experiments, not exact
reproductions of current GLM startup pins or live `REPIN`.

The tool does **not** model new prefetch reads, routing substitutions, GPU
kernels, NUMA effects, page-cache interference, lock contention, startup read
cost, or actual tok/s. A predicted reduction in felt wait is a gate for a
runtime A/B, not the result of one.

Current GLM also read-aheads the next 64-expert block while computing the
current block. The simulator preserves block promotion semantics but does not
model this existing overlap. Long-prefill felt-wait estimates are therefore
conservative and must be checked with lower `felt_fraction` sensitivity.

Kimi K3 traces are not currently safe input because that engine emits routes
without advancing the trace call ID. Inkling and OLMoE initialize routing
telemetry but do not currently emit route records. Their cache traversal also
differs from GLM, so each needs an explicit engine profile before inclusion.

## Trace contract

The input is the engine's current `ROUTE_TRACE` stream:

```text
<call> <row> <layer> <expert>:<gate> ...
```

Rows sharing one contiguous `(call, layer)` group are one runtime batch. The
simulator unions their expert IDs in first-seen order for demand I/O, while
retaining per-row selection multiplicity to train startup residents like
`.coli_usage` counts. It rejects malformed fields, missing/reordered rows,
non-advancing call IDs, and repeated noncontiguous call groups rather than
silently merging a broken trace.

Use a clean environment and deterministic, placement-neutral GLM trace
collection. In particular, unset `EXPERT_BUDGET` and `ABLATE_*`; the former is
applied after tracing and would make the trace differ from executed demand.
Record `COLI_PREFILL_CHUNK`, because it changes batch boundaries.

For a large model, place the model and traces on the instance's local scratch
volume rather than the root disk. On DigitalOcean's H200 image this is commonly
`/dev/vdc1`, mounted at `/mnt/scratch`; verify with `lsblk` and `findmnt` before
using it. The scratch volume is ephemeral and must not be treated as durable
storage.

```sh
env -i PATH="$PATH" HOME="$HOME" \
  DRAFT=0 MTP=0 TOPP=0 TOPK=0 CACHE_ROUTE=0 \
  PILOT=0 PILOT_REAL=0 REPIN=0 TEMP=0 \
  ROUTE_TRACE=trace.txt \
  SNAP=/path/to/model PROMPT="..." NGEN=128 ./colibri 64 4 8
```

Training traces and held-out evaluation traces must be separate files. The CLI
rejects identical resolved paths. Use one file per cold process run; policy
state is intentionally reset between files.

## Manifest and cost calibration

A manifest with fixed model geometry is required. Inferring `max_experts` from
held-out IDs would leak evaluation data and understate dimensions when a valid
expert did not happen to route.

```json
{
  "layers": {
    "3": {
      "resident_bytes": 18915328,
      "read_bytes": 75661312,
      "felt_miss_us": 2400.0,
      "max_experts": 256
    }
  }
}
```

Manifest rows that are absent from all supplied traces are ignored. This keeps
disabled MTP rows from consuming only the uniform baseline's budget.

`resident_bytes` consumes the expert cache budget. `read_bytes` counts demand
I/O and may differ when a source tensor is transformed into a smaller resident
representation. If `felt_miss_us` is omitted, it defaults to:

```text
read_bytes / (read_gbps * 1e9) * 1e6 * felt_fraction
```

`felt_fraction` is the fraction of read service not hidden by concurrent
compute. Caching a 20 ms read that is fully overlapped does not remove 20 ms
from token latency.

## Commands

Run deterministic fixtures:

```sh
cd c
python3 tools/residency_sim.py demo
python3 -m unittest tests.test_residency_sim
```

Evaluate a fixed expert-cache budget:

```sh
python3 tools/residency_sim.py run \
  --train traces/chat_train.trace traces/code_train.trace \
  --eval traces/chat_heldout_1.trace traces/code_heldout_1.trace \
  --eval-category chat --eval-category code \
  --manifest glm-residency.json \
  --expert-budget-gb 32 \
  --read-gbps 3.0 \
  --felt-fraction 0.7 \
  --felt-scales 0.5 1.0 1.5 \
  --sensitivity-layer 3 --sensitivity-layer 30 --sensitivity-layer 60 \
  --prof-log /mnt/scratch/logs/trace/coding_eval.out \
  --json-out cnre-32gb.json
```

Sweep expert-cache capacities:

```sh
python3 tools/residency_sim.py sweep \
  --train traces/chat_train.trace traces/code_train.trace \
  --eval traces/chat_heldout_1.trace traces/code_heldout_1.trace \
  --eval-category chat --eval-category code \
  --manifest glm-residency.json \
  --expert-budgets 8 16 24 32 48 64 \
  --read-gbps 3.0 \
  --felt-fraction 0.7 \
  --json-out cnre-sweep.json
```

JSON preserves model specs, every allocation, per-trace source paths and
metrics, and each complete sweep result. This is enough to compute category
regressions outside the simulator without parsing human output.

`--policies` must include `lru`, because every decision is measured against the
uniform-LRU baseline. Both `run` and every point in `sweep` report nominal and
layer-cost-sensitivity verdicts.

`--prof-log` stores the aggregate `PROF=1` felt-wait calibration in the JSON.
Current GLM logs expose aggregate, not per-layer, felt wait; this is provenance
and a global calibration check, not a claim of layer-specific measurement.

`run` also applies the Phase-0 decision gate directly. Categories receive equal
weight regardless of their number or length of traces. A candidate passes only
when its category-mean felt-wait gain is at least 10% and its worst category is
not more than 3% slower than uniform LRU. If `--eval-category` is omitted, the
tool derives categories by stripping a final numeric replicate from file names,
for example `chat_2.trace` -> `chat`.

Sensitivity reruns the complete training allocation and held-out evaluation
after scaling each layer's felt miss cost independently. `--felt-scales 0.5 1
1.5` therefore tests the baseline plus low/high cost perturbations for every
active layer. Uniformly scaling every layer would leave policy rankings and
percentage gains unchanged and is not a useful robustness test.

For a large model, use a representative cost sample first, for example
`--sensitivity-layer 3 --sensitivity-layer 30 --sensitivity-layer 60`. The
default all-layer sensitivity is intentionally thorough but can be expensive
because it recomputes policy-specific capacity allocation for every perturbation.

## Deterministic fixtures

The built-in stationary workload is intentionally cacheable and demonstrates
scan resistance. The shifted held-out workload deliberately changes its hot
set and demonstrates that training-seeded residents and dynamic layer budgets
can overfit. Exact fixture numbers are printed by `demo` and asserted only where
they encode an invariant; they are not evidence about real Colibri workloads.

`demo` then evaluates both fixtures as equal-weight categories in one decision
gate. This is deliberately stricter than averaging all accesses: a large
stationary win cannot hide a shifted-category regression beyond 3%. In the
current fixture, uniform frequency admission fails that guard (`-4.76%` in the
shifted category). Dynamic frequency admission narrowly passes the nominal
gate (`+4.27%` worst category) but fails the layer-by-layer felt-cost
sensitivity check at that 8 MB budget.

The final demo table sweeps synthetic expert budgets from 2 to 32 MB. It shows
that robustness is an operating region rather than a policy-wide property. For
example, dynamic frequency admission is robust in the deliberately constrained
2-6 MB fixture, fragile at 8-12 MB, and robust again at larger toy budgets. These
transitions are artifacts of the fixture, but they demonstrate why real traces
must be swept across realistic RAM budgets rather than evaluated at one point.

## Phase-0 decision gate

Advance to runtime telemetry or an opt-in cache policy only if category-held-out
real traces show:

```text
at least 10% lower predicted felt wait
no principal held-out category worse by more than 3%
bounded churn and exact expert-cache byte-budget compliance
the result survives conservative miss-cost sensitivity
```

The expert-cache budget does not establish whole-process RAM compliance. A
runtime implementation must additionally account for dense weights, KV state,
working-set slabs, scratch, page cache, and OS reserve. A negative result is a
completed experiment and should be published.

## Data still needed

The repository does not currently contain complete GLM routing traces. The
first useful campaign needs:

1. GLM-5.2 training and held-out traces from coding, chat, multilingual,
   reasoning, and long-context prompts.
2. Matching `PROF=1` logs or a per-layer felt-cost manifest.
3. Exact commit, model/container identity, DRAFT/sampling settings, cache state,
   prefill chunk, and frozen `.coli_usage` snapshot.
4. Separate instrumentation fixes and engine profiles before Kimi K3, Inkling,
   or OLMoE data is interpreted by this simulator.

No server is required to run the simulator. A model-capable machine is needed
only to collect missing traces and later validate a policy end to end.

## H200 pilot result

The first real-trace campaign ran on a DigitalOcean H200 instance using the
gs64 GLM container at revision
`95bb5f03e3e0ca16b4c711394d0461fa86a3cfcb`. The model occupied 429.3 GB on the
5 TB local scratch volume. The machine had 24 physical CPU cores, 247 GB RAM,
and 143.8 GiB H200 VRAM.

The collection produced eight training traces and five held-out traces across
coding, chat, reasoning, multilingual, and long-context prompts. These are
small pilot traces, not a production benchmark corpus.

At expert-cache budgets of 120, 160, and 200 GB, the offline simulator reported:

```text
budget   candidate             mean held-out gain   worst category
120 GB   uniform frequency          +39.91%             +34.84%
120 GB   dynamic frequency          +40.45%             +35.41%
160 GB   uniform frequency          +53.35%             +49.92%
160 GB   dynamic frequency          +54.50%             +50.66%
200 GB   uniform frequency          +65.31%             +62.64%
200 GB   dynamic frequency          +66.29%             +63.07%
```

All frequency candidates passed the nominal category gate and the sampled
layer-cost sensitivity at layers 3, 30, and 60. LRU dynamic allocation did not
meet the 10% gate, with approximately 0.0-0.4% gain.

The runtime pilot is the necessary qualification. With `RAM_GB=120`, `PIPE=0`
measured 1.66 tok/s at 57.2% hit; with `RAM_GB=200`, it measured 1.31 tok/s at
73.5% hit. Enabling the existing `PIPE=1` overlap reached 2.00 tok/s at the
same 57.2% hit. This means hit rate alone is not a performance result: I/O
overlap, disk service, and compute contention materially change tok/s.

Across the five held-out `PROF=1` logs, the aggregate runtime instrumentation
reported 19.6 seconds of felt wait over 2,625 loads, or approximately 7.47 ms
per load. This is a whole-run calibration signal, not a per-layer cost: the
current GLM profile does not attribute felt wait to individual layers.

No runtime frequency-admission policy was implemented in this pilot. The
current GLM cache admits every demand miss in `moe()` and promotes the bounded
tail of each 64-expert block. An unrecognized `COLI_CACHE_ADMISSION` variable
therefore has no effect. This is intentional: the real-trace evidence is strong
enough to justify a narrowly scoped opt-in prototype, but not enough to claim a
runtime win or to make a misleading A/B number part of the RFC.
