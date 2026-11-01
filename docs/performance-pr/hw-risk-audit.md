I have all the evidence I need. Composing the audit artifact now from the verified citations collected above.

## Review

**Provenance/method.** All code citations are from the port-tree worktree `/root/bench-worktrees/imp3-stack`, verified to sit on `refs/heads/port/imp3-stack` (worktree `.git/HEAD` → the ref; `.git/packed-refs` resolves it to `e5d08bdf`, the tip named in the task). Docs cited from that tree's `docs/`; measured numbers from `/root/colibri-upstream-port/audits/{gap2,gap3}-bench-synthesis.md`, `run-2026-10-27-outcome.md`, `specs/CS-am1-mixed-split-repricing.md`, `plans/pr-draft-{cs-experts,grow-headroom}.md`, and CAMPAIGN.md. Caveat: upstream-side claims were checked against the local `upstream-dev` worktree (detached @ `5d56c05a`, **not** `d67ebd8b` — none of the lever envs exist there); re-verify absence at the rebase tip. I did not run anything; tests named below are what a supervisor must run.

---

# gap-wave4 / hw-risk — other-hardware risk audit for default-on levers + the default-policy spec

## A. Per-lever hardware audit

### (a) `COLI_VK_V4_DECODE_PATH=fork` — the slab registry

**Hardware assumptions actually encoded (all verified in code):**

1. **A coherent, non-device-local system-RAM memory type must exist.** `pick_memtype_sys` (backend_vulkan.c:568–581) skips every type whose *heap* is `DEVICE_LOCAL` and requires `HOST_VISIBLE|HOST_COHERENT`, preferring `HOST_CACHED` (a WC host type made CPU expert compute 25× slower — comment at :560-566). Consequences by hardware class:
   - **Strix Halo / any iGPU, Lavapipe, Intel iGPU**: every host-visible heap is device-local → `memtype_sys = -1` (:581, assigned at :1590) → `slabreg_arm` returns 0 before allocating anything (backend_vulkan.c:2644) → **the registry never arms on unified memory**. The fork lever is a self-disabling no-op there. (Env escape exists: `COLI_VK_TIER_HOST=1` forces `memtype_sys = G.memtype`, backend_vulkan.c:1594.)
   - **Discrete RDNA3/4, GCN/Polaris RADV, AMDVLK, NVIDIA, Intel Arc dGPU**: a system heap exists → registry can arm.
2. **VK_KHR_buffer_device_address is optional, not required.** The slab buffer usage is `STORAGE | (G.has_bda ? SHADER_DEVICE_ADDRESS : 0)` (backend_vulkan.c:2683–2686) and the alloc flag only when `has_bda` (:2704–2705). The alias binds the slab's buffer *whole* with descriptor offsets (`t->wbuf = t->sbuf = reg->buf`, :2820; offsets checked against `G.ssbo_align = limits.minStorageBufferOffsetAlignment`, :1567 + :2816) — descriptor-based, works without BDA. Device creation already degrades BDA (retry without it, backend_vulkan.c:1509–1544). So "probe-verified on RADV NAVI21" does **not** generalize as a hard requirement — NVIDIA/Arc/AMDVLK only need the system heap.
3. **Heap-room clamp (the gap3 overcommit lesson) is real and layered** (backend_vulkan.c:2620–2626, 2664–2668): room = `VK_EXT_memory_budget` `heapBudget−heapUsage` of that heap (`coli_vk_tier_host_heap_room`, backend_vulkan.c:4643–4659 — NVIDIA's Linux heap is ~3/4 of RAM, Windows ~1/2, #2314 cited in-code), clamped to min(heap size, **7/8 of heap**, `COLI_VK_V4_SLAB_ARENA_GB`). The gap3 failure mode (10×2 GiB over a 19.46 GiB heap: vkAllocateMemory "succeeds", every submit fails) is directly addressed; 9×2 GiB was probe-verified end-to-end (docs/V4-forked-decode-path.md:72–84).
4. **Block size ≤ `maxBufferSize`**: maintenance4's when present, else **4 GiB assumed** (backend_vulkan.c:2651–2660). On a driver without maintenance4 whose real max is 2 GiB, the default 2048 MiB block `vkAllocateMemory` fails → refusal → ladder (safe, lever disabled; one P2 below).

**Ladder — verified clean at every rung:**
- Slot alloc, base store: `vkt_slabreg_slot` refusal → `malloc` (deepseek_v4.c:9194–9198).
- Slot alloc, hot store: refusal → `posix_memalign(4096)` (deepseek_v4.c:10477–10490; slabs carved at `max(4096, bind alignment)`, backend_vulkan.c:2693–2694, so O_DIRECT reads keep working into mapped pages).
- Alias: registry lookup miss → the standing `coli_vk_slab_import` branch (vk_tier.c:3200–3239); per-expert refusal (fmt∉{7,15}, bounds, `ssbo_align` descriptor-offset misalignment, backend_vulkan.c:2811–2816) → `T.cs_refuse++` → NULL → **CPU compute in the finish loop** (vk_tier.c:3240–3242) — never dropped, never fatal.
- Standing-import rung itself refuses when `VK_EXT_external_memory_host` is absent **or staged uploads are on** (no-ReBAR card: `g_up[0].on` → `coli_vk_import_alignment()` returns 0, backend_vulkan.c:2324–2326) — i.e. on a no-ReBAR 8 GB card the whole CS alias layer self-disables and residents-only ride the group. Every rung is today's path — "clean fallback to today's path" is confirmed by construction, matching docs/V4-forked-decode-path.md:56–70.

**Findings on (a):**
- **P2 — silent registry refusal + overstated engagement line.** `slabreg_arm` prints only on *success* ("slab registry armed: …", backend_vulkan.c:2671–2673); refusal (APU, no room, alloc fail) prints nothing, while `vkt_forkdec_store_line` still prints `decode-path=fork arena=X slabs=N` over the *plan* (deepseek_v4.c:9575–9577, 20966–20968). On an APU with fork default-on, stderr would claim engagement that didn't happen (only the run-end `cs_fork` count, vk_tier.c:4593, tells the truth). Smallest fix: one stderr line in `slabreg_arm`'s refusal return + make the store line conditional on ≥1 slab.
- **P2 — maintenance4 fallback.** Default 4 GiB when the ext is absent (:2656–2659); on old drivers this silently disables the lever via the ladder. Acceptable; document it.
- **Correct (no fix):** heap clamps, BDA-optional usage, alignment refusal, malloc ladder, `slabreg_reset` at store destroy (backend_vulkan.c:2780–2796), Windows DLL-loader leg exists (c/Makefile:1432).

### (b) `COLI_VK_V4_S1GEMV` — the S=1 fmt-15 GEMV

**Subgroup assumptions — the kernel is subgroup-size-generic.** `qmatmul_s1_mx.comp` derives everything at runtime: `sgsize = gl_SubgroupSize`, `nsg = nth/sgsize`, lane→word `for (wi = lane; wi < words; wi += sgsize)` (shaders/qmatmul_s1_mx.comp:84–97; `local_size_x = 256`, :31). Wave64 (RDNA compute), wave32, GCN, Intel SIMD8/16/32 and Lavapipe (sgsize 8) all execute the same *assignment function* — and the bitwise contract is "verbatim the generic fmt-15 branch's" (:17–21), so both kernels share the order at any sgsize.

**Does the parity gate protect everywhere it dispatches?** Yes — structurally:
- **One dispatch site only**: the down-GEMV record, `… : bd_fmt(…) ? X->p_mv_bd : xb_s1gemv_pick(X, e->d, rows[c])` (backend_vulkan.c:6204), with the S=1 dispatch special-cased (:6210–6213). `xb_s1gemv_pick` refuses unless `p_s1 && s1_ok && rows==1 && fmt==15 && I≤6144 && I%8==0 && rowWords==I>>3 && gs && gs%8==0` (backend_vulkan.c:5581–5586).
- **`s1_ok` is only set past `xb_s1gemv_parity`** — a live-device, same-context bit-exact A/B of the generic dispatch vs the dedicated kernel (backend_vulkan.c:5592–5675), run at arm (:5777), pipeline destroyed on mismatch with one stderr line ("parity vs qmatmul NOT exact -- the generic GEMV stays", :5783–5785), absent/invalid shader gets its own line (:5786). Per-device: `s1_ok`/`p_s1` live in each `XbCtx`, so dev2 re-runs its own parity (:6405–6408, armed(dev)).
- **Resource ceilings fail clean**: `GL_KHR_shader_subgroup_arithmetic : require` (:29) or `shared float xsh[6144]` = **24,576 B LDS** (:82) exceeding a device's `maxComputeSharedMemorySize` → pipeline create fails → the "absent or invalid" line → generic GEMV. Residual: the parity leg fixes PI=2048/PO=256/gs=32 (:5594) while dispatch runs I≤6144; the in-tree A/B unit case (tests/test_vk_tier.c:4463–4556) covers the batch plumbing at case geometry — keep both legs.

**Finding on (b):** none blocking. **P2**: parity leg covers one geometry; if a new I/gs class is ever admitted by the pick, extend the leg (structural argument holds — identical loops — but the discipline is live-device proof).

### (c) `COLI_VK_CHAIN_ATTN_FA` — the flash tile

**Documented fallbacks — verified:** the shape gate at dispatch is `p->hd == 512 && !(p->H % 8) && !(q_off|q_row|o_off|o_row|w_off|c_off & 3)` (vk_chain.c:3673–3674); a pipeline that did not load (`fa4_pipe` lazy, tried-once, :3658–3665) or a refused f16-mirror scratch falls through to the **mode-0 dispatch at the same call** (:3707–3710) — "never a misread" (header comment :3648–3653). Numerics are thresholded (2e-3 / 8e-3 classes), **never bit-identical** (chain_dsv4_fa.comp:21–27; test_vk_chain.c:2098–2101 enforces rel-err + run-to-run bit stability, not mode-0 equality).

**Compile risk on small-LDS hardware — arithmetic:** the Br 8 build stages `Qf[Br*(VEC+1)]` = 8×129×16 = **16,512 B**, `ksh[Bc*VEC]` = 16×128×16 = **32,768 B**, `kent` 64 B, `tmpsh` 1,024 B, `tmpshv4` 4,096 B (chain_dsv4_fa.comp:134–138) → **54,464 B ≈ 53.2 KiB** total. The Br 2 decode twin: 4,128 + 32,768 + 64 + 1,024 + 4,096 = **42,080 B ≈ 41.1 KiB**. Consequences: 64 KB LDS (AMD GCN/RDNA, Intel ANV) compiles both; **48 KB devices (Pascal-class NVIDIA) compile the decode twin but not the prefill tile** → correct partial engagement; ≤32 KB devices compile neither → mode 0 everywhere. `GL_KHR_shader_subgroup_shuffle` (:33–35) with XOR masks ≤ D_split/2 = 8 stays inside 16-lane aligned blocks — safe at subgroup 16/32/64.

**Findings on (c):**
- **P1 — silent fallback + default numerics change.** `fa4_pipe` failure prints **nothing** (contrast s1gemv's two lines); a default-on FA on a 48/32 KB-LDS card or a shuffle-less driver silently never engages — the reviewer's "prove it fired" demand is unanswerable from stderr (only `KC.st.attn_fa` counts, vk_chain.h:885). Smallest fix: one line at first refusal (`[VK] chain: FA tile not taken (pipeline/shape) -- mode 0`). Separately, FA **changes shipped numerics wherever it engages** (D11 recorded "F/GF token-1 FA numerics", CAMPAIGN W1) — default-on is a deliberate output-contract change, not a perf-only flip (see policy §B).
- **P2**: `attn_fa_wanted()` reads env per call (vk_chain.c:3666–3670) — fine, but a future default must fold `COLI_VK_LEGACY` into this read.

### (d) EXPERTS (CS) + DECODE_GROW family + RING/RING_COLD

**CS (`COLI_VK_TIER_EXPERTS`), default-on VRAM/RAM risk — measured and bounded:**
- No new VRAM: residents already ride the boot-fit budget; aliases are device-visible *host* pages. Import ladder bounded at **1408 live `VkDeviceMemory`** wraps (`slab_cap`, backend_vulkan.c:2504–2515), deliberately under the 4096 spec-min `maxMemoryAllocationCount`; past the cap → per-expert CPU fallback (the lever's own ladder, ENVIRONMENT.md:272).
- **Low-RAM hosts actually get safer**: under a 34 GiB cgroup the standing arm peaked 37.1 GiB and OOM-killed (rc=137) ×2, the CS arms peaked **27.2 GiB and survived every leg** (pr-draft-cs-experts.md, cprof2). On 8–16 GB cards the no-ReBAR/no-import rungs degrade to residents-only — slow but alive.
- Numerics: the ON arm's text is a deterministic device-tolerance class — model-scale md5 differs from standing (`beb8028c` vs `83b8a8f7`, W-CS S9; texts fragment per fit-band under shared hosts). Token-exact at tiny held (deepseek-v4-tiny-check-cs green, c/Makefile:964–998).
- Composition guard: `EXPERTS=2` refuses with one line unless GROW+HEADROOM armed (vk_tier.c:4260–4266); `DECODE_PATH=fork` without `EXPERTS=1` says so and stays off (vk_tier.c:3151) — gap3's prcfa-forkdec legs demonstrated exactly this self-ignore (0 engagement lines, by design).

**DECODE_GROW family — probes and floors:**
- Free-VRAM probe: `COLI_VK_DEVICE_CAP_MB` else `VK_EXT_memory_budget` budget−usage (dg_free_now, vk_tier.c:3515–3521). **APU semantics are explicitly correct**: on an integrated GPU *all* heaps count as device memory and budgets are summed (vk_memtype.h:87–112, incl. the RADV system-heap-budget-capped-at-size case). Unreadable budget → 0 → "no room to grow into, the warm set stands" line (vk_tier.c:3760–3762) — never sheds.
- Floors: 0.5 GiB default reserve (`_RESERVE_GB`, :3705), byte cap `_GB`, **stream-slot floor of two** (`st_n > 2`, :3642), headroom live guard re-probes every 64 asks and makes a place only when real free ≥ reserve + one expert (:3656–3689). W1 measured it: 273→427 residents, 0 free VRAM left, installs one-time.
- Engine scope: needs `VktConfig.decode_heat`; only DeepSeek V4 supplies it (deepseek_v4.c:15114) — every other engine gets the refusal line and standing behavior (vk_tier.c:4466–4470).

**RING/RING_COLD:**
- Worker thread failure is fail-soft: `pthread_create` fail → one line, `ring_ok=0`, inline fill for the rest of the run (vk_tier.c:2361–2368); teardown bounded to the read in flight via `ring_stop` (:2316–2322); every miss/unpredicted expert covers through the inline fallback (:2273–2277). RING≡C0 measured **byte-identical** at model scale (W2).
- **P1 — the grow↔ring slot conflict is documented and unresolved**: PGRT's installs consumed the stream slots **256 → 2** — "slot accounting conflicts with the ring (PRC) and must be re-derived before any PGRT+PRC stack arm" (CS-am1 §4.1, :88–90). Default-on **both** families fires this by default. Also: RING/COLD trades decode for TTFT (gap2: the ring trades ~0.21 tok/s for ~30 s TTFT; PRC decode −5.2% at W3), and **the CS+forkdec+ring TTFT composition is explicitly unmeasured** (gap3 session note 1).

**The gap2 RAM lesson — floor guards exist now:**
- Operator RAM override can't exceed reality: `COLI_VK_TIER_HOST_GB` "wins over the plan, never over MemAvailable" — `if (gb > avail) gb = avail` (vk_tier.c:3958–3959); heap-room cut `heap − 2 GiB` (:3968–3976); one-expert floor refusal with a named line (:3984–3999); host pool off on UMA by default (`T.uma` guard, :3935).
- The engine plan itself starts from MemAvailable with system reserve (1/8, 512 MiB–4 GiB), a runtime reserve (2×layer + runtime_other) and `minimum_expert_bytes` = topk×record×layers (deepseek_v4.c:1366–1381).
- Thread creation is fail-soft everywhere it matters: ring (above), pilot pool partial-start (deepseek_v4.c:10717–10725), persistent/dual loaders degrade to sync (deepseek_v4.c:5311, 5424–5435).

### (e) fmt-15 (already default)

Settled: `tier_fmt15()` = packed unless `COLI_VK_TIER_FMT15=0` (vk_tier.c:476–478), default `1` in ENVIRONMENT.md:270, lossless (byte lands in the float exponent field), P0 root-caused and default restored only after model-scale proof (PW-Dfix, CAMPAIGN). One line; keep.

---

## B. Default-policy spec for the PR

**Rule 1 — default = probe-passed, not vendor-string.** Every lever keeps its *existing* probe and gains default-on *behind* it; no hardcoded RDNA2/Navi anywhere:

| lever | probe at device open (already in code) | default when probe passes | fallback + line |
|---|---|---|---|
| `V4_DECODE_PATH=fork` | `slabreg_arm`: memtype_sys≥0 + VIS\|COH flags + heap room ≥ 1 block (backend_vulkan.c:2640–2676) | **on** (with `EXPERTS≥1`, vk_tier.c:3148–3154) | malloc/import ladder; **add the missing refusal line** (P2a) |
| `V4_S1GEMV` | `xb_s1gemv_parity` live bit-exact A/B (backend_vulkan.c:5592–5675) | **on** | generic GEMV; lines exist (:5783–5787) |
| `TIER_EXPERTS` | import alignment available (not staged) or registry armed; refusal ladder | **on (=1)**; `=2` composition stays opt-in | per-expert CPU; **add a run-line `cs_refuse` count if >0** |
| `TIER_DECODE_GROW`(+HEADROOM) | `dg_free_now` readable + `decode_heat` supplied (vk_tier.c:3515–3521, 4466) | **on** for dsv4 only | "no room / no counts" lines exist (:3760, :4468) |
| `TIER_STREAM_RING`(+COLD) | `ring_batch` provided + `pthread_create` ok | **see P1-3: opt-in until slot-accounting resolved** | inline fill; line exists (:2362) |
| `CHAIN_ATTN_FA` | fa4 pipelines load + shape gate (vk_chain.c:3658–3674) | **on, conditional on P1-1 sign-off** (numerics class) | mode 0; **add the missing line** (P1-2) |
| `TIER_FMT15` | unconditional (lossless) | on (unchanged) | `=0` hatch |
| CS_ALIAS_CACHE+SCAN_SKIP, ATTN_CELL, STREAM_ROWS_*, TIER_QUEUE=0, DEVSUM, PILOT | — | **stay off** (measured −39% / neutral / negative; DEVSUM+PILOT ride arms explicitly) | — |

**Rule 2 — one `COLI_VK_LEGACY=1` escape hatch** (net-new; grep confirms no such env today): reads at each lever's existing env site (`lever_on() && !vk_legacy()`), semantics = every new-default lever forced off, stack byte-identical to the upstream base; per-lever `=0` continues to work individually. Boot prints **one** line per lever: `engaged / refused:<reason> / legacy`. Precedent for the "one line" contract: s1gemv's arm/refuse pair and the registry's armed line.

**Rule 3 — gate-suite flips (the PR must change these deliberately, not discover them):**
1. `c/Makefile:871–889 deepseek-v4-tiny-check` (tiny/prefix/brio, **no env**) — now runs at the new defaults; fork/s1gemv/grow are token-exact by design (tiny legs already green armed, Makefile:964–1053), but **FA and CS change the numerics class** — either pin the old base on a new `COLI_VK_LEGACY=1` leg or regenerate expectations in-tree; the fmt-15 flip is the precedent (its legs were rewritten: tests/test_vk_tier.c:3649, 3734, 4452, 4460, 4575).
2. `c/Makefile:2780–2790 vk-tier-check` — leg 1 (no env) contains the unset-invariants that **break by design**: tests/test_vk_tier.c:4474–4475 (`CHECK(!coli_vk_xb_s1gemv_armed(0))`, `st.s1_gemvs == 0`); leg 3's `COLI_VK_TIER_FMT15=1` is now redundant; the dgrow/forkdec/s1gemv explicit legs stand. Add `COLI_VK_V4_S1GEMV=0` (or LEGACY=1) to the arm-0 halves.
3. `c/Makefile:2797–2817 vk-strict-check` / tests/vulkan_strict.sh — the "unset and 0 give the same stderr (cmp)" case (:81–88) generalizes: add unset-vs-`LEGACY=1` equivalence legs per lever; the Lavapipe anti-fatal case (:100–107) now runs levers by default and must stay green (registry refuses on unified memory; FA tile may fail to load — *after* P1-2, observably).
4. `c/tests/test_vk_chain.c:2058–2110` + the plain dsv4-attn cases (~:2020–2052) — pin `COLI_VK_CHAIN_ATTN_FA=0` for the mode-0 reference legs or they silently test the tile; the FA case itself (arms 1/2 at :4966–4969) stands.
5. `c/Makefile:892–897 deepseek-v4-smoke` (tests/deepseek_v4_smoke.sh) — the model-scale default gate must gain a `COLI_VK_LEGACY=1` leg (the fmt-15 precedent: "COLI_VK_TIER_FMT15=0 runs the escape hatch", script :28–33) — this is the gate that caught the fmt-15 P0 (AI-pw-w1 lesson).
6. Windows runtime-DLL leg (c/Makefile:1432) re-run for the decode-path kernels.

**Rule 4 — minimal per-hardware-class matrix a reviewer will ask for** (each: smoke + tiny + strict + the lever boot lines; one decode/TTFT leg where the class has a GPU):
1. **Strix Halo-class APU (RADV, unified)** — expect: registry refuses (memtype_sys=−1, with the new line), CS aliases via host-import or refuses cleanly, grow bounded by summed-heap budget (vk_memtype.h:103–112), host pool off (`T.uma`), no double-accounting of the same RAM.
2. **8 GB discrete, ReBAR off AND on (RADV + one NVIDIA)** — staged leg: imports disabled (backend_vulkan.c:2326) → CS residents-only, `cs_refuse` counted; grow floor behavior at tight free; no boot OOM; NVIDIA heap-room clamp vs its ~3/4-RAM heap (#2314 class) and `ssbo_align`=16 descriptor-offset refusals ≈ 0.
3. **NVIDIA proprietary (BDA + external-memory-host)** — registry arms (STORAGE+SDA), parity gate passes, LDS 54.5 KB FA tile compiles or degrades with the new line.
4. **Lavapipe CPU (CI)** — vk-tier-check/vk-chain-check/strict green with defaults on (parity gate decides s1gemv at sgsize 8; FA may refuse on LDS; nothing fatal, nothing silent).

**Rule 5 — measurement debts that gate defaults:** (i) the CS+forkdec+RING TTFT composition is one env line and unmeasured (gap3 note 1); (ii) the grow↔ring slot-accounting re-derivation (CS-am1 §4.1); (iii) a D11-style golden re-pin for the CS+FA numerics classes before the flip merges (S9's own pre-registered requirement: "needs the D11/golden treatment before any PR").

### Findings summary
- **P1-1** Default-on `ATTN_FA` + `CS` change shipped outputs (deterministic numerics classes, W-CS md5 `beb8028c` vs `83b8a8f7`; D11 token-1 flips) — needs the owner's explicit sign-off + in-tree baseline regeneration + a LEGACY leg; token-exact-at-tiny is proven, model-scale text change is not a bug, it is the contract change.
- **P1-2** FA fallback is silent (vk_chain.c:3658–3665) — one stderr line owed.
- **P1-3** GROW-family × RING-family default co-enablement fires the documented 256→2 slot conflict (CS-am1 §4.1) — resolve or default only one (grow).
- **P2** registry refusal silent + store line overstates engagement (backend_vulkan.c:2640–2676 vs deepseek_v4.c:9575); maintenance4-absent 4 GiB default; parity leg single-geometry; per-expert `cs_refuse` has no boot-visible count on the standing path.
- **Correct** (no action): the entire degradation ladder for fork (both slot sites, alias, import, CPU); s1gemv single guarded dispatch site + per-device parity; grow probes/floors incl. iGPU budget semantics; ring pthread fail-soft + bounded teardown; RAM floor guards answering the gap2 lesson (MemAvailable clamp, heap−2 GiB, one-expert floor); fmt-15 standing default.

**Merge verdict (Phase A, read-only):** no code to merge; the audit supports the owner direction with the conditions above.

VERDICT: SAFE-DEFAULTS — for fmt-15 (already), `COLI_VK_V4_DECODE_PATH=fork`, `COLI_VK_V4_S1GEMV`, `COLI_VK_TIER_EXPERTS=1`, and the `COLI_VK_TIER_DECODE_GROW` family, each probe-gated with the P2 line-fixes; `COLI_VK_CHAIN_ATTN_FA` and `COLI_VK_TIER_STREAM_RING(+_COLD)` stay opt-in until P1-1..P1-3 close (numerics sign-off + FA stderr line + the grow↔ring slot-accounting re-derivation and the unmeasured composition leg).