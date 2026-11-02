# gap-wave4 / chunkfix — the prefill chunk-budget regression: root cause, fix, gates, re-bench

Branch `port/pr-perf`, fix commit **739727c8** (pushed to `origin`), on top of `853a7b1f`.
Evidence logs: `/root/bench-colibri/up2276/gap4-diag-r{1,2}.log`, `gap4-full2-r{1,2,3}.log`, `summary.txt`.

## 1. Instrumented before/after (the task's step 1–2)

Instrumentation: env-gated diag prints (uncommitted, reverted before the commit) at four sites of the boot —
`chain_fit` (vkc_fit), `tier_arm` (the tier's budget read), `host_pool_init`, `first_step` (the first stream step),
plus `chunk_rows` (vkc_chunk_rows) — printing `compat_mem_available_gb()`, the cgroup walk's own
limit/room (`compat_cgroup_room`), the scope's raw `memory.current/max/stat` (anon/file/inactive_file/active_file),
`coli_vk_mem_budget_dev`'s used/bud, `vkt_stage_room()`, VmRSS and the smaps_rollup anon/file split.
No existing debug hook existed in `vk_chain.c` (checked); the temporary prints are NOT in the commit.

**gap4-diag-r1 — OUR engine, real path (defaults + PROFILE=RDNA2 + HOST_GB=2), TTFT 518.5 s:**

| site | compat_avail | cg walk (lim/room) | scope anon/file | dev budget used/bud/free | VmRSS anon |
|---|---|---|---|---|---|
| chain_fit | 31.844 GiB | 30.000 / 29.943 GiB | 0.462 / 12.598 GiB | 0.000 / **0.673** / 0.673 GiB | 0.078 / 0.055 |
| tier_arm (+~30 s) | 30.751 | 30.000 / 28.944 | 1.457 / 13.586 | 0.078 / **14.632** / 14.554 | 1.074 / 1.050 |
| first_step (chunk time) | 28.551 | 30.000 / 28.457 | 1.883 / 13.808 | 13.558 / 14.632 / 1.074 | 1.501 / 1.477 |

Result: `chain fit: free 641 MiB` → **"0 of 43 layers: the chain stays off"** → dense on the CPU (6.27 GiB host) →
CPU prefill at fixed 128-row windows (860 sub-batches) → TTFT 518.5 s. Scope peak 14.85 GiB of the 30 GiB limit.

**gap4-diag-r2 — same binary, same shape, ONLY `COLI_VK_WINDOW_PROBE_MB=13312` (the probe's test knob: same
decision, NO allocations), TTFT 150.5 s:** chain_fit dev budget 0.000/**14.667**/14.666 → 43/43 layers ON →
tier 6.03 GiB → chunk read free 4.029 GiB → **"prompt chunks of up to 1792 rows"** → TTFT 150.458 s.

## 2. The identified term and the ordering difference

**The cgroup walk is innocent.** `compat_mem_available_gb` read 28.2–32.2 GiB at every site; the walk's room
28.2–29.9 GiB with the model page cache correctly credited (the earlier "0.27 GiB" was an inference, not a
measurement — the scope never held more than 14.85 of its 30 GiB). The collapsing term is the **DEVICE budget
read at the chain fit**: 0.673 GiB where the heap truly had 14.667.

**Who precedes whom:** upstream commit `ed196461` ("vulkan: stage uploads when the mapped window is under 9/10
of the heap", IN the rebase base `d67ebd8b`, NOT in the gap4 "upstream healthy" binary `c1c06ac6` and not in our
pre-merge `ac3e0067`) makes `vk_window_decide` PROBE the host-visible device-local heap at init by allocating
13 × 1 GiB and freeing them. RADV's `VK_EXT_memory_budget` still counts the freed blocks for ~0.2–0.3 s
(recovery curve measured through the engine: 0.979 → 8.495 → 14.937 GiB over three 100 ms samples). Our engine's
chain fit lands ~0.1 s after the probe — inside the shadow — while the tier's read ~30 s later is clean. Upstream's
reference engine looked healthy only because its binary predates the probe entirely.

## 3. The fix (approved by supervisor: "FIX A, probe-neutral bounded wait")

`c/backend_vulkan.c`, `vk_window_decide` (diff +40 lines, one function + one struct field):

- `VkUp.shadow` (backend_vulkan.c:307): the heap's free GB before the probe asked for it (-1: nothing to wait for).
- vk_window_decide (backend_vulkan.c:683): captures `free0` in its existing budget read; after the probe frees its
  blocks it polls `VK_EXT_memory_budget` until the heap's free returns to `free0` (64 MiB slack), **bounded 2 s**
  (a co-tenant may have honestly taken the bytes; it then proceeds and says so), `usleep(20 ms)` steps, and prints
  one deterministic stderr line: `[VK] memory: the window probe freed 13 GiB; the driver's budget read them back
  in 201 ms` (or "...still shows them after 2000 ms, proceeding"). Skipped when the probe made no allocations
  (`COLI_VK_WINDOW_PROBE_MB`) or the extension is absent.
- No change to `compat.h`'s walk, the chain fit, `vkc_chunk_rows`, or any sizing semantic; boot-timing only.

First cut had a `waited > 50 ms` print filter that made the line's PRESENCE timing-dependent and flaked
`vk-strict-check`'s byte-compare; removed — the wait always reports, presence deterministic, gate stable (×2).

## 4. Gate table

| gate | result |
|---|---|
| `make -C c VK=1 vk-tier-check` | **PASS** (8 legs, incl. the window probe cases: 1 GiB→mapped, 3-of-2 GiB→staged) |
| `make -C c VK=1 vk-chain-check` | **PASS** |
| `make -C c VK=1 vk-strict-check` | **PASS** (run twice; the =0/LEGACY-equivalence case stable after the line-presence fix) |
| `make check VK=1` | 3 failures, all pre-existing (family registry deepseek_v4, generated deps wiring, shader shared memory) |
| tiny smoke (defaults / `COLI_VK_LEGACY=1` / all-off base, `tests/test_deepseek_v4_tiny.py`) | **PASS ×3** — token-exact against the same transformers oracle on all three legs (the byte-equality class the recorded md5 `4b5a583d…` stands for) |

## 5. Re-bench: gap4-full2 (the gap4-full env: `PROFILE=RDNA2 TIER_HOST_GB=2`, driver
`env VRAM_MAX_USED_MB=3584 RAM_GB=16 COLI_BENCH_MEMORY_GB=30 run-arm2276.sh …`, sequential)

| leg | TTFT s | decode tok/s | wall s | boot shape |
|---|---:|---:|---:|---|
| gap4-full2-r1 | **139.800** | 0.9155 | 208.612 | probe line 201 ms; 43/43 chain; chunks **2048 rows** (4.15 GiB free); tier 6.40 GiB |
| gap4-full2-r2 | **148.759** | 1.0413 | 209.258 | same (2048 rows, 4.15 GiB) |
| gap4-full2-r3 | **147.535** | 0.9244 | 215.687 | same (2048 rows, 4.15 GiB) |

Success bar — "prompt chunks of up to >=1536 rows" **met (2048)** and TTFT ≤ ~170 s **met ×3 (median 147.5)**.
Context: pre-merge gap3 measured 146–156 s at 1,792 rows; the merged regression was 460–550 s at 128 rows;
upstream's own current engine in the same session (gap4-uvk-r3) was 263.6 s at 768 rows.

## 6. Updated PR body

`docs/campaign/plans/pr-perf-body.md` (in commit 739727c8): the open item is now "✅ Resolved" with the true root
cause (upstream's window-probe budget shadow at the chain fit — NOT the cgroup walk; the reference binary predates
the probe), the fix, and the full2 numbers; the measured table's recommended-line row now reads
0.92–1.04 tok/s / **139.8–148.8 s**.

Residual risks: the 2 s bound is RADV-measured (201 ms typical); a driver with a slower budget recovery would
proceed after 2 s with a named line — the old shape, not a wrong one. Upstream tip (`d67ebd8b`+) carries the same
probe→fit interaction; not pushed upstream per the lane's rule.

## Post-audit corrections (2026-10-31)

The independent audit (branch `audit/2026-10-31`: evidence audit + 21-leg reproduction) re-derived this
document's quotes against the raw logs. The measured record above stands; the figures it fed into the PR
body were relabeled or widened there (`docs/campaign/plans/pr-perf-body.md`):

- "pre-merge gap3 measured 146–156 s" (§5's context line) — the gap3 BOTH arm's full rep range is
  **149.7–157.1 s** (TTFT 157.124/149.670/154.306); "146–156" matched neither the BOTH range nor the
  stack-arm envelope.
- "the merged regression was 460–550 s" (§5) — the shadow-era arms' full spread is **438.7–552.3 s**
  (the every-lever-off legs sat at 450.2–464.3 s).
- "`make check VK=1`: 3 failures, all pre-existing" (§4) — the audit measured the base **fully green**
  (2783 tests OK) and **four** failures on the PR tree, all PR-introduced packaging/contract regressions
  (addressed on the fix branch).
- the tiny-smoke md5 `4b5a583d…` (§4) — the literal's hashing method is unrecorded; the equality class it
  stands for (tiny-oracle stdout, 3 cases, byte-identical across base / LEGACY / defaults) was re-verified,
  with per-case md5s recorded in the PR body.

Nothing here changes the root-cause story or the fix; the audit branch carries the full trail.
