> **Historical decision record (2026-10-19/22).** The F-number statuses below predate the
> 2026-10-23 parity campaign; several items are landed, falsified or re-attributed since.
> The live state is `HANDOVER-2026-10-23.md`; falsifications are its §2.

# PORT_PLAN — transplant `colibri-ds4-vulkan` onto a fresh upstream fork

**Status:** decisions taken 2026-10-19 (D3/D4/D5/D10/D11); baseline running · 2026-10-19
**Sources:** `docs/research/A-upstream-architecture.md`, `B-fork-inventory.md`,
`C-perf-evidence.md`, `D-risk-validation.md`
**Decision owner:** operator. Every "DECIDE" item needs an explicit call.

---

## 1. Why this is a transplant, not a rebase

| | fork `colibri-ds4-vulkan` | upstream `dev` |
|---|---|---|
| base | `dd7df2c` (v1.8.0, 2026-08-24) | `a0037a3a` (v2.0.0 + ~590, 2026-10-19) |
| commits past base | 353 non-merge / 356 total (HEAD `d11fba5b`) | **2,499** |
| files / lines | 318 files, **+124,595 / −3,653** | 1,103 files, **+361,069 / −8,586** |
| DSV4 Vulkan design | one hardwired **GPU-FOLD** stack, per-op shaders, `backend_vulkan_dsv4.c` (5,433 lines) | **dense chain** (`vk_chain.c`, `deepseek_v4_chain.h`, `chain_dsv4.comp`) + **shared routed-expert tier** (`vk_tier.c`, `tier.h`) + per-op fallback |
| failure model | GPU refusal fatal; no mid-run CPU fallback | silent per-op degradation; a lost device re-runs the forward on the CPU |
| cross-engine | DSV4 only | tier/chain shared by 9–11 engines |
| DSV4 real-checkpoint perf | measured (1.0 → 2.0 → 3.0 tok/s decode progression) | **never measured** (upstream docs say so) |

> Lanes A–D were researched at `5d56c05a`; upstream moved to `a0037a3a`
> during the analysis (PRs #2224–#2238, `vk_chain`/`vk_tier` perf). The
> structural conclusions are unchanged; the baseline is re-pinned to the new tip.

The fork's own `REVIEW.md` §F2 defined the trigger for a full re-evaluation:
*"upstream ships its own dsv4 Vulkan tier (converge)"*. That has happened.

### 1.1 The divergence is structural, not textual

| shared hot file | base | fork | upstream | fork ± | upstream ± |
|---|---:|---:|---:|---:|---:|
| `c/deepseek_v4.c` | 17,017 | 30,154 | 22,047 | **+15,354 / −2,217** | +5,568 / −538 |
| `c/backend_vulkan.c` | 2,166 | 18,237 | 7,144 | **+16,382 / −311** | +5,321 / −343 |
| `c/deepseek_v4_internal.h` | 998 | 2,196 | 1,249 | +1,216 / −18 | +251 / −0 |
| `c/backend_vulkan.h` | 134 | 1,581 | 508 | +1,448 / −1 | +375 / −1 |
| `c/Makefile.deepseek-v4` | 281 | 1,422 | 393 | +1,164 / −23 | +130 / −18 |

Both sides changed **49 of the same functions** in `deepseek_v4.c`
(`attention_token_impl`, `block_token_impl`, `moe_token_pipeline`,
`target_batch/token`, the whole `coli_v4_expert_store_*` family, …). There are
**217 `COLI_V4_GPU_TIER` sites in the fork vs 48 upstream**, **307
`dsv4_cuda_*` call sites vs 143**, and a **7,752-line GPU unit** upstream does
not have. `backend_vulkan.c` has 371 functions in the fork and 333 upstream,
but **only 68 names in common** — they are two programs sharing a filename.

A rebase does not conflict; it produces a plausible-looking binary that
misbehaves. That is the core reason for a capability transplant.

**Principle:** keep upstream's architecture and maintenance path. Port the
fork's *measured capabilities* into upstream's seams. Do not port the fork's
stack, its lever surface, or its history.

### 1.2 The fact that must be measured first

Upstream enables the dense chain on a discrete GPU by default, and its DSV4
chain + tier are correctness-tested but **unmeasured on a real checkpoint**
(no 167 GB model on the upstream author's box). The fork's numbers are the only
real-checkpoint Vulkan evidence for DSV4 on this GPU.

Therefore **Phase 0 is a mandatory baseline**: build `upstream/dev VK=1`, run
the real checkpoint at 2k/8k, and compare with the fork's recorded numbers
(§5). The result decides whether we port a pipeline or only deltas. Do not
start shader work before this gate.

---

## 2. Goal and non-goals

**Goal.** A new fork of `upstream/dev` that:
1. keeps upstream's chain/tier architecture and cross-engine fixes;
2. gains the fork's measured wins where they still beat upstream on this box;
3. passes a parity + perf gate that is reproducible on this host;
4. remains upstreamable in slices.

**Non-goals.**
- Preserving the fork's `COLI_DSV4_VK_*` lever surface or the GPU-FOLD
  hardwiring.
- Resurrecting `backend_vulkan_dsv4.c` as a permanent second device layer
  (see the strategy arbitration in §6.1).
- Porting experiments that measured neutral or negative (§4.5).
- Rewriting upstream's engines beyond DSV4 unless a generic fix belongs there.

---

## 3. Target architecture and port seams (lane A)

```
deepseek_v4.c (34 units)
  ├─ per-op hooks: coli_v4_vk_matmul / _fp8 / _host   ──► backend_vulkan.c
  │                                                        └─ vk_tier.c (vkt_*)
  └─ GENERATE_STATS unit: v4c_forward ──► deepseek_v4_chain.h ──► vk_chain.c
                                                             └─ chain_dsv4.comp / chain_mhc.comp
expert_store.h (vtable) ◄── expert_store_registry.c
```

Seams a ported capability must use:

| Seam | File | Use for |
|---|---|---|
| expert tier contract | `c/vk_tier.h` | VRAM expert policy, async issue/join, streaming |
| chain frame model | `c/vk_chain.h`, `deepseek_v4_chain.h` | any dense/attention op recording |
| per-op matmul hooks | `deepseek_v4.c` `v4_vk_open` | weight placement, fallback |
| engine store vtable | `c/expert_store.h` | RAM slots, lookup/release, prefetch |
| V4 prefix checkpoint | `deepseek_v4.c:15236+` | KV persistence upgrade |
| unit build | `c/Makefile.deepseek-v4.units` | new TU/object |
| shader build | `c/Makefile.deepseek-v4` `V4_VK_SPV` | new `.comp`/`.spv` |
| env docs | `docs/ENVIRONMENT.md` rows | every new knob |

Constraints:
- RDNA2/gfx1030: **no cooperative matrices, no native fp8 MMA, no fp64
  shaders**. `chain_gemm` and `COLI_VK_XB_GROUPED` are off here; `vkc_moe`
  (BDA + clustered) and fp32 GEMV/GEMM are available.
- Upstream has **no env registry**; a knob is a `getenv` in its owning unit
  plus an `ENVIRONMENT.md` row. The fork's ~157 fork-only env names must not
  migrate wholesale.
- GPU work must stay inside the owning `COLI_V4_UNIT_*`; a reference from a
  non-GPU unit breaks the `VK=0` build.
- **Shader names collide.** Both trees ship `qmatmul.comp`, `rmsnorm.comp`,
  `attention_absorb.comp`, `qmatmul_gate_up.comp`, but the fork's fp8 is
  **fmt 8 with 128×128 E8M0 block scales (`uint scale[]`)**, upstream's is
  **fmt 12 with per-row `float scale[]`**, and the fork adds `woff`/`soff`
  push constants. A copied `.comp` compiles cleanly and reinterprets the
  bindings at runtime. Ported shaders must be renamed (`dsv4_*`) and the
  push-constant size asserted at load.

---

## 4. Portfolio — what to port, what not, what to fix

Verdicts combine measured evidence (lane C) with upstream equivalence (lanes
A/B). "Port" means port the *capability*, re-expressed in the seam above.

### 4.1 Port — Phase 1: engine/storage (no shader work, lowest risk)

| # | Capability | Why (measured) | Integration seam | Size | Notes |
|---|---|---|---|---|---|
| **P1.1** | ✅ **landed** — **KV cold tier v2** — content-addressed `<model>/.coli_ckpt/kv_<fp>_<idhash>.bin`, RAM catalog (ids + 4 B/token), per-layer streaming store/restore, pinned system prefixes, LRU + byte GC, cancelled-prefill demote, fingerprint incl. weights identity | Kills the 4-agent slot ceiling; a miss costs 677 s re-prefill vs 25–80 ms restore / 70–250 ms write; survives restart | upgrade upstream `V4_PREFIX_CKPT_DISK` (already has prompt-end captures + id matching, max 8 slots, full snapshots resident) | M | Add payload checksum + Windows scan before landing. Decide interaction with `--kv-slots`. |
| **P1.2** | **Reactive prefill segment fit + OOM recovery** — size segments from live `MemAvailable`, shed pool for the segment, mid-prefill re-fit, 64-row floor | Eliminated two host OOM kills at 128k on a 38.9 GiB box | `V4_PREFILL_SEGMENT` path in `deepseek_v4.c` | S–M | Robustness port; independent of Vulkan. |
| **P1.3** | **KV usage budget + request admission** — project per-request KV growth, 503 instead of OOM, baseline-derived boot plan | Stops a 200/503 loop; honest capacity planning | V4 serve/generate unit | S–M | Coordinate with upstream `serve_budget.h` / `kv-slots`. |
| **P1.4** | **Serve framing hint + LCP capture triggers** | Makes cold-KV captures fire under interleaved agents | V4 generate + `openai_server.py` + `serve_protocol.md` | S | Only if P1.1 lands. |
| **P1.5** | ✅ **landed** — **Opt-in strict capability gate** — enumerate every Vulkan degradation/refusal point; when `COLI_VK_STRICT=1`, abort non-zero naming the op instead of silently staying on the CPU; default off. Our deployment and test gates enable it. | D5 decision. Upstream's silent degradation hides a failed tier; the fork's strict mode is the diagnostic that caught real regressions. | `backend_vulkan.c` degradation points + boot/op wrappers | S–M | Self-contained; must not change default behaviour for other engines. Pair with a boot line that says which mode is active. |

### 4.2 Port — Phase 2: decode I/O pipeline (the biggest measured decode win)

| # | Capability | Why (measured) | Integration seam | Size | Notes |
|---|---|---|---|---|---|
| **P2.1** | **Decode MoE demand lanes** — all k misses become read+compute jobs on a lane pool; single fail-closed join in selection order | 0.92 → 1.10–1.19 tok/s decode (+23 %), byte-identical | V4 `moe_token_pipeline` + lane pool around the store | M | Preserve upstream's rank-order accumulation and tier hand-back. |
| **P2.2** | **Async PILOT + warm-pool lane compute** | Blocking −35 %; lane compute 2.05 → 3.0 tok/s (−32 % wall) | `vk_tier` prefetch + store prefetch vtable | M | Extend `vkt_stream_prefetch` rather than add a second predictor. |
| **P2.3** | **Tier admission fixes** — demand misses honour the split direction; reserve floor at admission need; prompt-sized VRAM gap shed; boot-warm LRU re-stamp | Fixed a permanent 503 wedge; adapt 75.0 % hit vs 70.3/72.8; decode 1.20 vs 1.09/1.08 | `tier.h` policy + `vk_tier.c` budget code | S–M | Upstream's LFRU is the base. Port only deltas the baseline proves. |

### 4.3 Port — Phase 3: RDNA2 kernel wins (shader work, behind upstream chain/tier)

| # | Capability | Why (measured) | Seam | Size | Notes |
|---|---|---|---|---|---|
| **P3.1** | ✅ **landed** (`cc66bf61`, split-K `04aec2c7`) — **mm8 tiled weight-stationary projection GEMM** (+ split-K/ping-pong) | projections 87.1 → 18.4 ms/cell (4.73×), ~6.8 % e2e — the fork's numbers; ours: RX 6900 XT microbench 2,065–4,131 GMAC/s (best tile, split) against the GEMV's 185–243 at S 128, engine arm pending | `matmul_aligned`'s mm8 arm (`qmatmul_mm8.spv` + `_reduce`), opt-in `COLI_VK_MM8`/`COLI_VK_MM8_SPLITK`, default off | M | Upstream `chain_gemm` needs coopmat; this is the RDNA2 answer. |
| **P3.2** | **pf4 expert GEMM family + expert-major bank** | routed cell 335 → 114 ms (2.94×); 2.65× session GMAC/s | tier batch compute / shaders | M–L | Upstream has MXFP4 decode but not this kernel family. |
| **P3.3** | **Flash-attention kernel** | attention 5.0× @2k, 16.7× @8k; 2.1–2.4× decode @3k/8k | `chain_dsv4` attention mode | M | 1 bf16-ulp gate; big long-context win. |
| **P3.4** | **LDS-tiled idxscore + idxsel v2** | indexer 26.1 → 6.2 ms/cell (4.2×); TTFT −9.4 % @33k | `chain_dsv4` indexer modes | M | The measured long-context slope term. |
| **P3.5** | **Activation placement + KV-ring append + wo_b −55 %** | segment 14.2 → 9.0 ms; 256k prefill 2h16m | chain buffers/recording | M | Only if baseline shows host round trips upstream does not already remove. |
| **P3.6** | **mHC decode fusion / residency** | +19 % decode; prefill −5.5 % TTFT @32k | `chain_mhc` / decode ops | M | Upstream chain already records mHC; port only the decode-specific fusion if faster. |

### 4.4 Conditional — Phase 4: prefill pipeline (only if Phase 0 says upstream is behind)

| # | Capability | Why it is conditional | Size |
|---|---|---|---|
| **P4.1** | **GPU prefill expert bank, gate 1 token** + parallel ring→bank take | fork wins at every length (−2.2 s @11 tok → −708 s @4093), but upstream's tier streaming + chain chunk may already cover it | L |
| **P4.2** | **Host-RAM next-layer prefetch ring** | −5…−24 % TTFT at 0.5–8.8k; only meaningful with P4.1's bank | M |
| **P4.3** | **Cell/submit merge, cell-tail, queue-ahead** | **measured neutral** (cell-tail: submits −1/cell, TTFT in noise; queue-ahead: wall-neutral). Do **not** port as perf wins. Port the CB-ownership guard only if upstream's frame model lacks an equivalent (it has `vkc_split_layers`) | — |

### 4.5 Do NOT port

**Architecture / infrastructure**
- `backend_vulkan_dsv4.c/.h` and the same-named CUDA ABI shim as a *permanent*
  layer — it bypasses the arena, staged uploads, dev2, partial chain, and every
  upstream Vulkan fix. (Conditional exception: a temporary shim is allowed as a
  porting scaffold, retired by the end of Phase 3; see §6.1.)
- GPU-FOLD hardwiring, removal of composition levers, `COLI_V4_VK_ONLY`.
- `COLI_VK_STRICT` as a global default — upstream degrades per op. Keep an
  opt-in strict mode for tests only (DECIDE D5).
- The fork's ~157 fork-only env names; port only knobs that earn a place.
- The 19 composition levers the fork silently retired (`_DEC2*`, `_FFN_*`,
  `_COMP_*`, `_SHARED_DEFER`, `_MHC_CELL_DEC`, `_ATTN_CELL_DEC`, …); do not
  resurrect them as documented knobs.

**Experiments with a measured null or negative result**
- dec2-mhc (−10.5 %), SHARED_DEFER (−4.6 %), COMP_FOLD_DEC (−2.1 %),
  COMP_RESIDENT / FFN_TAIL / MHC_PREFUSED (washes), T5 next-token PILOT
  (waits collapsed, wall unchanged), per-device NVMe queues (noise),
  fp32-accumulator mm8 (+0.8 % for a parity-class break), PSI elastic RAM and
  device-only fp16 KV (no measurement exists).
- MFU/MRU as a *parallel cache* — port the policy deltas into `tier.h`, not the
  cache.

**Hygiene**
- Dead refactors (`b44a343f`, `d8d4d241`, `d009f149`, `262a8932`), stale plan
  docs, debug-only env levers, committed `.o`/binaries, `run_arm.sh`,
  `c/tools/keepalive_vk*`, `qd_probe`, `rans_measure`, one-off bench sources.

### 4.6 Fix — before or during the port

| # | Fix | Why |
|---|---|---|
| **F1** | Commit or park the fork's uncommitted WIP (`c/deepseek_v4.c`, `REVIEW.md`, `docs/deepseek-v4.md`, two untracked scripts). The WIP currently **deletes the 2026-10-18 split-controller decision record** while adding the KV floor. | The port source must be a clean, tagged commit. |
| **F2** | Resolve branch confusion: fork `main` is 85 commits past base; HEAD (`feat/kv-cold-tier`) is 353. README says the active branch is `feat/decode-gpu-residency`. | Prevents porting a stale tree. Tag the intended source. |
| **F3** | `c/Makefile` does not forward `VK=`; only `-f Makefile.deepseek-v4 VK=1` works. | Silent CPU rebuild trap that already invalidated one perf gate. |
| **F4** | `docs/ENVIRONMENT.md` duplicated row / missing table pipe (`COLI_DSV4_VK_CPU_BOOT_FILL`). | Doc drift guard misses it. |
| **F5** | Add a `.spv` ↔ `.comp` compile/staleness gate and a push-constant-size assertion at pipeline load. `idxsel2.spv` shipped byte-identical to `idxsel.spv` for weeks; the two trees' `qmatmul.spv` parse different bindings. | A "fix" that was never compiled; silent runtime misinterpretation. |
| **F6** | Known-red suites (`test_expert_cache` boot-decay, `test_prefill_cache`/`test_cache_parity` non-hermetic counts) are record-only; gate or fix before using them as evidence. | Cannot claim green if they are red. |
| **F7** | KV cold tier gaps: payload checksum, Windows directory scan, no bitwise restored-vs-prefilled gate. | Land these with P1.1. |
| **F8** | Numerical contract language: greedy output is expert-history dependent; the tier is thresholded, not bitwise; the envelope is prompt- and build-dependent. | Never claim "token-for-token identical" without the envelope. |
| **F9** | Add a `NOTICE` file; never commit checkpoint bytes, `.golden` fixtures or stage-oracle bundles. The fork has no `NOTICE`. | Apache-2.0 provenance hygiene. |
| **F10** | Re-pin `.golden/upstream` from `dd7df2c` to the new base and re-record. Full gate ≈ **4.1 h** per run (41 min/pinned leg). | Golden artifacts keyed by the old merge-base are invalid. |
| **F11** | Decide the `dsv4_cuda_*` export set if any shim survives: header declares 161, `dsv4.def` exports 83, `.cu` defines 118. | Windows DLL would not resolve ~78 symbols. |
| **F12** | Upstream fixes worth PR-ing independently: `COLI_NO_OMP_TUNE` value bug, `expert_cache_trim` 503 floor, VK=0 build restores. | Small, generic, reduces fork debt. |
| **F13** | **Tier host pool ignores `--memory-gb`.** `vk_tier.c:host_init` sizes `COLI_VK_TIER_HOST` from `mem_available() - 4 GiB`, not from the engine's RAM plan. On this host it allocated **25.75 GiB** beside a 15.5 GiB engine cache and OOM-killed the run; the knob is also missing from `docs/ENVIRONMENT.md`. | Real overcommit bug; disable (`COLI_VK_TIER_HOST=0`) for the baseline, then fix sizing to respect the engine budget and document it. |

---

## 5. Phase 0 — baseline and namespace gate (do this before anything else)

### 5.1 Build and run the target

```bash
cd c
make deepseek-v4                     # CPU
make deepseek-v4 VK=1                # Vulkan chain + tier
make deepseek-v4-tiny-check          # tiny fixture oracle
make vk-tier-check VK=1 && make vk-chain-check VK=1
```

Run on the real checkpoint (`/root/Projects/colibri-v4-docker/models/DeepSeek-V4-Flash`)
with `COLI_VULKAN=1`, first on the default chain, then with the chain off.
Use `--memory-gb 18`, `COLI_VK_TIER_HOST=0` (see F13), `COLI_V4_PREWARM=0`,
`OMP_NUM_THREADS=8`, and run under a 24 GiB cgroup:
- 2k code prompt / 64 decode tokens
- 8k prompt TTFT (one run, directional)
- record: decode tok/s, TTFT, prefill tok/s, disk-busy %, GPU %, tier hit rate

### 5.2 Compare with the fork's recorded baseline (lane C)

| operating point | fork record |
|---|---|
| 512-tok prefill, 12 GiB | 225.4 s CPU → 140.9 s VK dense; 111.7 s with D7 drop |
| decode, 2k class | 0.85–1.27 tok/s (1.27 on 2026-10-05); 3.0 tok/s with a warm MFU pool |
| GPU idle during decode | 85–88 % (disk/host-bound) |
| prefill 8.8k | fork ring −5.3 % TTFT |

### 5.3 Gate outcomes

- **A — upstream ≥ fork within 10 %**: port only P1.x + P3.x deltas that
  measure positive; skip P4 entirely. Fastest, safest.
- **B — upstream 10–40 % slower**: port P2 + P3, then re-measure; P4 only if
  prefill still trails.
- **C — upstream >40 % slower or refuses the checkpoint**: escalate before
  writing engine code. Re-check the chain fit (`COLI_VK_CHAIN_LAYERS`), tier
  budget, and whether `COLI_VK_CHAIN` is actually on. If the architecture
  cannot reach parity, the fallback is §6.1 Option B, which is a different
  project (continue the fork), not a transplant.

### 5.4 Namespace and evidence work (lane D's P0)

- Rename every ported shader to a `dsv4_*` namespace; never overwrite
  upstream's `qmatmul.comp`/`rmsnorm.comp`/`attention_absorb.comp`.
- One `fmt` table; assert `sizeof(push constants)` against the loaded SPIR-V.
- Re-pin `.golden/upstream` and re-record (budget two full runs).
- Land `scripts/env_doc_drift.sh --check` and a `.spv` byte-reproduction check
  as merge gates.
- Add `NOTICE`.

---

## 6. Delivery plan

### 6.1 Strategy arbitration — which Vulkan stack survives?

This is the central decision, and the lanes disagree. Both are defensible:

| | **Option A (recommended)** | **Option B (lane D's reading)** |
|---|---|---|
| base | `upstream/dev` | `upstream/dev` |
| device layer | upstream `backend_vulkan.c` + `vk_tier.c` + `vk_chain.c` | fork's `backend_vulkan_dsv4.c` per-op ABI |
| port direction | fork capabilities → upstream seams | upstream tier/chain gains → fork ABI ops |
| cross-engine fixes | inherited automatically | must be re-imported |
| maintenance | one shared stack | two stacks; DSV4-only layer |
| effort to first perf win | higher (re-express in chain/tier) | lower (fork already measured) |
| risk if upstream moves | low | high (permanent divergence) |
| upstreamability | high | low |

**Recommendation: Option A.** The fork's measured wins are valuable, but its
device layer is the part most likely to rot: 68/371 shared function names, a
7,752-line GPU unit upstream lacks, and a refusal/fatal contract incompatible
with every other engine. Porting the *capabilities* preserves the wins while
keeping upstream's tier/chain fixes.

**Decided 2026-10-19: Option A.** Upstream's chain/tier is the surviving
stack. Option B is no longer the fallback; if Phase 0 outcome C occurs, the
response is to fix or extend upstream's architecture, or to stop and re-plan —
not to resurrect the fork's per-op layer.

**When Option B becomes the right call:** Phase 0 outcome C, or a slice where
upstream's chain structurally cannot host the capability (e.g., upstream
declines the checkpoint geometry). Even then, prefer a **temporary shim** over
a permanent second stack: land the capability behind the ABI, pass the gate,
then retire the shim when the chain equivalent lands. Record the retirement
date in the same PR.

**Lane D's dissent is recorded:** D scores `(a) fresh branch, slice-by-slice`
highest (21/25) but assumes the per-op ABI is the survivor and upstream's
tier/chain gains are imported into it. Its divergence data is used throughout
this plan; only the survivor choice differs.

### 6.2 Repository and branch model

- New repo from `upstream/dev` (not from the fork): `port/baseline`.
- One branch per slice: `port/kv-cold-v2`, `port/moe-lanes`, `port/mm8`, …
  Each merges only after its gate.
- Keep `upstream/dev` as a remote; merge it every release boundary (the fork
  did not, and paid 2,399 commits of drift).
- Tag the fork source (`port-source-2026-10-19`) and freeze it bugfix-only.
- No merge of fork history into the new repo. Port by capability, with
  `Source: colibri-ds4-vulkan@<sha>` in each commit body.

### 6.3 Slice order and exit criteria

| Phase | Slices | Exit criteria |
|---|---|---|
| **0** | baseline + namespace/pin + tooling | CPU/VK/tiny green; real-checkpoint numbers recorded; `.golden` re-pinned; fork tagged |
| **1** | P1.1–P1.5 | parity green; interleaved-agent test restores from disk; restart survives; no OOM at 128k; strict mode aborts on a forced degradation |
| **2** | P2.1–P2.3 | decode ≥ baseline, byte-identical text class; tier hit rate + disk-busy improved; no other-engine regression |
| **3** | P3.1–P3.6 | each kernel bitwise/thresholded vs its CPU oracle; per-op gate; e2e A/B ≥3 pairs |
| **4** | P4.1–P4.2 (conditional) | TTFT improves at 2k and 8k; no decode regression; bank engagement proven by counters |
| **5** | tooling/docs | `env_doc_drift --check` green; golden parity reproducible; REVIEW/ADRs landed |

### 6.4 Estimated effort (T-shirt, not a promise)

| Phase | Effort | Risk |
|---|---|---|
| 0 baseline + namespace | M (3–5 d, incl. 2 × 4.1 h golden re-record) | Low |
| 1 storage/KV | M (3–5 d) | Low–Med |
| 2 decode I/O | M–L (5–8 d) | Med |
| 3 kernels | L (5–10 d) | Med |
| 4 prefill | L (5–10 d, conditional) | High |
| 5 docs | S–M | Low |

---

## 7. Validation matrix

Adopt upstream's gates first, then bring the fork's evidence harness as tooling.

| Gate | What it proves | Command / artifact | Pass threshold |
|---|---|---|---|
| **G0a build** | CPU and VK both link; no unit-boundary break | `make -C c check`, `test-c`, `make deepseek-v4`, `make deepseek-v4 VK=1` | exit 0; VK=0 output identical to baseline |
| **G0b exclusivity** | `VK=1 CUDA=1` fails loudly | Makefile error | must fail, never silently pick VK |
| **G0c boot/namespace** | no second shader owner; right SPIR-V | boot with a shader dir missing one `.spv`; push-constant size assert | abort non-zero naming the op; no silent reinterpretation |
| **G1 op parity** | each ported op matches its oracle | `make vk-tier-check`, `make vk-chain-check`, `tests/test_deepseek_v4`, fork stage-oracle bundles | bitwise where upstream is; thresholded where documented |
| **G2 engine parity** | tokens are right end-to-end | tiny fixture exact; real checkpoint greedy at 8/28/64 tokens | tiny exact; real within the recorded envelope |
| **G3 cross-engine** | tier/chain changes did not break other engines | `tests/vulkan_engines.sh deepseek` + one non-DSV4 family | no new failures |
| **G4 perf** | the slice is actually faster | interleaved A/B, ≥3 pairs, median, same process state | ≥ baseline − 5 %; target ≥ fork |
| **G5 robustness** | failures are graceful | forced VRAM refusal, device loss, torn KV file, restart, interleaved agents | no wedge, no corruption, no silent wrong answer |
| **G6 hygiene** | docs and artifacts are honest | `env_doc_drift.sh --check`, `.spv` reproduction, `vk_l2_targets_check.sh` | exit 0 |

**Minimum gate without the checkpoint** (the iteration loop):
`make -C c check` → `test-c` → `make -f Makefile.deepseek-v4 test-vk-all VK=1`
(or upstream's `make vk-tier-check vk-chain-check`) → `deepseek-v4-tiny-check`
→ boot-degrade → `env_doc_drift --check` → `.spv` reproduction.

**Release gate with the checkpoint:** the above plus real-checkpoint golden
parity, the cell-hash GPU-vs-CPU A/B, and G4 perf.

Perf discipline (from the fork's hard-won process findings):
- Pin clocks or report them; 500 MHz sclk dips produced false negatives.
- Gate on engagement counters, not on a flag's value (the G17 wrong-lever trap).
- Never cite parity from one binary; rebuild and re-run the bitwise gate.
- Report the disk/GPU split, not just tok/s.

---

## 8. Risk register

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| 1 | **Two Vulkan stacks coexist**, with opposite failure semantics | H | H | §6.1 decision; one permanent stack; a shim must have a retirement PR |
| 2 | `qmatmul`/`rmsnorm`/`attention_absorb` name + `fmt` + push-constant collision | H | H | `dsv4_*` rename; one `fmt` table; push-constant size assert |
| 3 | Engine text-merge infeasible (49 shared functions, 217 vs 48 ifdefs, 7,752-line GPU unit) | H | H | No rebase; port call sites in coalesced slices |
| 4 | Upstream DSV4 VK is slower than the fork / refuses the checkpoint | Med | High | Phase 0 gate; Option B escalation path |
| 5 | `.golden` keyed to the old merge-base; re-record ≈ 4.1 h | H | M | Re-pin in Phase 0; budget two runs; `--pinned-only` for iteration |
| 6 | Golden gate is CPU-only in practice (`target_batch(NULL,…)`); GPU divergence passes silently (already realised as #19/#40) | M | H | Keep the split contract; assert the GPU leg allocated; cell-hash A/B |
| 7 | Missing upstream ABI symbols if a shim survives | H | M | Implement `device_unified`; loud refusal stub for `expert_bank_upload_aux` |
| 8 | Rounding/fmt drift (fp64 removal, bf16 points, E8M0) | M | H | Bitwise only for same-computation pairs; envelope elsewhere; fresh stage-oracle bundles |
| 9 | RADV denormal-flush waiver is host-specific | M | M | Named waiver with measured floor; re-measure per ICD |
| 10 | Env surface: 234 shared names semantically unverified; 19 removed levers; strict vs refusal | M | M | Drift check as merge gate; per-name doc diff; publish the chosen semantics |
| 11 | `dsv4.def` (83) vs header (161) if a shim survives | M | M | Decide export set; stub or hide behind an ifdef |
| 12 | Licensing/provenance: no `NOTICE` in the fork; never commit weights/fixtures | M | M | Add `NOTICE`; keep Apache headers; re-run upstream's third-party inventory |
| 13 | Toolchain drift: no pinned `glslc`, 8 fork-specific byte recipes | M | L | Pin `glslc`; `.spv` byte-reproduction check; keep `qmatmul_coop` off gfx1030 |
| 14 | Upstream moves during the port | H | Med | Merge `upstream/dev` at every slice boundary; small branches |
| 15 | Fork WIP is the only copy of a fix | Low | Med | F1: commit/tag before touching anything |

---

## 9. Decisions required (owner)

| # | Decision | Status / recommendation |
|---|---|---|
| **D1** | Target branch: `upstream/dev` or `upstream/main`? | **Confirmed by D3 = upstream; use `dev`** (483 commits ahead, where Vulkan work lands). Flag if `main` was intended. |
| **D2** | New repo name/location and Gitea backup | Planning home is `/root/Projects/colibri-upstream`; repo to be cut from `upstream/dev` there, Gitea `pi/colibri-upstream`. Confirm or rename. |
| **D3** | **Which Vulkan stack survives** (§6.1) | **DECIDED: Option A** — upstream chain + tier; port capabilities. No permanent per-op shim. |
| **D4** | Serving model: `--kv-slots` concurrency, disk cold-KV, or both | **DECIDED: both** — kv-slots for live concurrency; cold-KV catalog for many/idle lineages and restarts. P1.1 must not fight kv-slots. |
| **D5** | Failure semantics | **DECIDED: upstream silent degradation is the default; strict-fatal is an opt-in mode, and our deployment + test gates enable it.** The strict mode must name the op and abort non-zero. |
| **D6** | Performance priority | **Decode first**, then long-context prefill (the wall is 85–88 % GPU-idle/disk-bound). |
| **D7** | Scope | **DSV4 only** for engine surgery; generic tier/kernel improvements go into shared code. |
| **D8** | Upstream PR intent | **Yes**, for generic slices (KV catalog, tier policy, kernels); keeps the fork small. |
| **D9** | Fork lifecycle | Freeze after `port-source-2026-10-19`; bugfix-only; no new features. |
| **D10** | **Phase 0 baseline go/no-go** — build `upstream/dev VK=1` and measure the real checkpoint at 2k/8k (~1–2 h) before any port work | **PENDING.** This is the measurement gate, not the 8 h golden re-record (D11). Recommendation: GO. |
| **D11** | Golden re-record budget | **DECIDED: approved, 8 h** for two full runs. |
| **D12** | `NOTICE` / artifact policy | Add `NOTICE`; never commit weights, `.golden`, or stage-oracle bundles. |

---

## 10. Immediate next actions

1. **F1**: commit or park the fork WIP; tag `port-source-2026-10-19` at the
   intended source commit (`d11fba5b` if the WIP belongs, else its parent).
2. **D10**: run the Phase 0 baseline on `upstream/dev VK=1` with the real
   checkpoint; write `docs/research/E-baseline.md`.
3. **D11**: re-pin `.golden/upstream` and start the first re-record.
4. Scaffold the new repo from `upstream/dev`; add the fork's benchmark harness
   (decode/prefill/AB scripts) as the first tooling slice.
5. Convert Phase 1–3 items into issues with acceptance gates.
6. Only then start P1.1 (KV cold v2) and P2.1 (demand lanes).
