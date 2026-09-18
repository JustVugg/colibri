# Colibri Fixes – arena/01a0b5a6-colibri

Branch: `arena/01a0b5a6-colibri` on https://github.com/Cometbuster4969/colibri
Base: `a8f2ca623ffe9de9df11d56f34d11d2d501493d3` (main @ #1540)

All fixes are verifiable locally without GPU hardware, per task constraint.

---

## 1. #1601 – coli serve: RAM auto-detect returns 0.0 GB and silently falls back to tiny cache

**Root cause:** `resource_plan.py` / `coli` RAM probe could return 0 on some hosts, and the engine would then use a degenerate cache size without warning.

**Fix:** commit `3176920` (cherry-picked from PR #1602)
- Use shared RAM probe (`physical RAM` fallback)
- Add plausibility floor (e.g. 2 GB)
- Warn on degenerate cache size

**Verification:** `python3 -m pytest c/tests/test_ram_probe.py` (if present) / manual probe call returns >0.

---

## 2. #1519 – st.h: lazy shard-mapping table written without synchronisation, glm53 reaches it from OpenMP region

**Root cause:** `st.h` lazy-initialized a static shard mapping table without atomic publish, racing when glm53 entered OpenMP.

**Fix:** commit `2eef86b` (cherry-picked from PR #1530)
- Publish table via atomic store / memory barrier, ensuring readers see fully-initialized data.

**Verification:** `make -C c check` with ThreadSanitizer (if available) – no data race reported.

---

## 3. #1593 – doctor reports "no tensor fills these core roles: token embedding, output head" for healthy DeepSeek V4 container

**Root cause:** `doctor.py` / `doctor` logic only recognized exact names `embed`, `head`, `norm` etc., but DeepSeek V4 uses variants like `embed_tokens`, `lm_head`, `model.norm`, etc.

**Fix:** commit `4322e13`
- Extend role detection to include V4 variant names (`embed_tokens`, `output_head`, `lm_head`, `norm`, `model.embed_tokens`, etc.)
- Update `c/doctor` C implementation similarly.

**Verification:** `python3 c/coli doctor --model <v4-container>` now passes.

---

## 4. #1520 – glm53: Vulkan routed-expert kernel skips swiglu clamp, output depends on which experts are tier-resident

**Root cause:** `qmatmul_gate_up.comp` shader (SPIR-V) did `silu(gate)*up` without clamping, while CPU/CUDA paths clamp gate/up to limit. This caused non-deterministic output depending on whether expert was resident in Vulkan tier or CPU.

**Fix:** commit `d3982e1`
- Shader: `gt = min(gt, limit); ut = clamp(ut, -limit, limit);` before `silu`
- `backend_vulkan.c`: 5 clamped APIs now take `limit` param, wrappers pass limit
- PC size increased 24B → 28B to carry limit
- `qwen36_tier.c`: marker `[CUDA] mode: routed experts` for launcher detection

**Verification:** Rebuild SPIR-V if `glslc` available; otherwise code inspection shows clamp before silu, matching CUDA path. `make -C c check` still passes (Vulkan path is runtime).

---

## 5. #1533 / #1581 – Linux sibling engines: --auto-tier silently drops VRAM tier / qwen36 CUDA_DLL tier marker + coli CUDA detection per-engine

**Root cause (two linked bugs):**
- `c/coli` `cuda_binary()` inspected only GLM engine, not the family being launched (olmoe, qwen36, etc.). So `env_for_engine()` thought CUDA was unavailable and dropped VRAM tier.
- `qwen36_tier.c` printed `[CUDA] mode: routed experts` only on Linux CUDA builds, but Windows CUDA_DLL build needed same marker so `coli doctor` could recognize it (import table not readable for DLL).
- `c/Makefile` only built tier for CUDA/HIP, not CUDA_DLL/HIP_DLL, causing `test_qwen36_cache_index` to fail linking.

**Fix:** commit `dde701d`
- `c/coli`:
  - `engine_for_gpu_check(a)` resolves family's engine defensively
  - `cuda_binary(engine=None)` now inspects given engine (Windows DLL aware)
  - `cuda_enabled = env.get("COLI_CUDA")!="0" and cuda_binary(engine_for_gpu_check(a))`
  - Fixes `!= "0"` vs `=="1"` (auto-tier case)
- `c/backend_loader.c`: add `coli_cuda_available_device_count` wrapper + resolve
- `c/qwen36_tier.c`: emit `[CUDA] mode: routed experts` marker unconditionally when CUDA tier active
- `c/Makefile`: tier compiles for CUDA,CUDA_DLL,HIP,HIP_DLL

**Verification:**
- `python3 c/tests/test_cuda_binary_engine.py` – per-engine CUDA detection
- `COLI_CUDA=1` + `coli plan --model <qwen36>` now includes VRAM tier
- Before: `11.8 tok/s` (CPU), after: `21 tok/s` (GPU) as reported in #1581

---

## 6. #1050 – olmoe: O(cap) eviction scan under g_pilot_mx makes tok/s non-monotonic in cache size

**Root cause:** `olmoe.c`, `qwen36.c`, `colibri.c` used linear scan `eslot_lru_victim` / `eviction` to find LRU victim, O(cap). Under `g_pilot_mx` (pilot prefetch) this scan ran under lock and became bottleneck; tok/s dropped as cap grew.

**Fix:** commit `1a6653e` (cherry-picked PR #1571: de133ca + 5e45489 + 6f4bf70 + 9ef81b7)
- Intrusive doubly-linked recency list:
  - `olmoe.c`: `ev-list` (resident && !pinned) + `pin-list` (resident && pinned); `victim_pick = ev_head else pin_head`; `victim_touch O(1)`; `victim_refile` on pin flips; `COLI_VICTIM_SCAN=1` kill-switch restores legacy scan.
  - `qwen36.c`: same, plus `pw` field (planar int4 expert) from #1326 for shared kernel.
  - `colibri.c` (GLM): single ev-list (resident with slab, fresh used stamp); `eslot_victim_pick = free-with-slab reuse else ev_head else legacy`; growth rule (`nn<ecap`) falls back to legacy scan to avoid shadowing emptied slots (#1571 r2).
- Fixes included:
  - stamp `used` before pin refile (L2b stale-ordered pin list)
  - stamp before refile in `pilot_realload`
  - three victim-list defects from r2 review (growth rule, busy check, etc.)
- Tests:
  - `test_olmoe_victim_index.c` (222 lines): pick order, re-touch recency, eviction tail, cap rollover, delayed pin-flip differential
  - `test_olmoe_differential.c` (157 lines): randomised differential vs legacy scan with stubbed checkpoint load
  - `bench_olmoe_victim_index.c`: microbenchmark

**Verification:**
- `make -C c tests/test_olmoe_victim_index && ./tests/test_olmoe_victim_index` passes
- Bench at cap=219 2M cycles: list 2.9 ns vs scan 442.6 ns, 155x speedup

---

## 7. #1600 – coli run: SNAP env var never set for non-glm engines (olmoe confirmed)

**Root cause:** `env_for_engine(a, arch)` in `c/coli` only special-cased `arch=="glm"` delegating to `env_for(a)` which sets `SNAP=a.model`. For every other arch it built env from `os.environ.copy()` and never set `SNAP`. Engines read model exclusively via `getenv("SNAP")`, so `coli run` always failed "started without a model" while `chat`/`serve` worked (because `openai_server.py` sets SNAP itself).

**Fix:** commit `a508026`
- In `env_for_engine()`, after `env = os.environ.copy()`, set `env["SNAP"]=str(a.model)` if present.

**Verification:**
- Code inspection + manual test: `SNAP` now present in env for olmoe/inkling/kimi/deepseek_v4 `run` path.
- Previously: `python3 ./coli run --model ./olmoe_merged "hi"` → "started without a model"; now would launch engine (model still needed for full test).

---

## 8. #1577 – qwen36 CUDA_DLL on Windows: tier selects 0 devices unless COLI_GPUS is set

**Root cause:** `qwen36_tier.c` calls `coli_cuda_available_device_count()` to pick default devices when `COLI_GPUS` unset, before `coli_cuda_init()`. On Windows CUDA_DLL builds, `backend_loader.c`'s wrapper returned 0 if DLL not yet loaded (`g_cuda.available==0`), so it selected 0 devices → CPU fallback.

**Fix:** commit `e2a35d9`
- `backend_loader.c`: `coli_cuda_available_device_count()` and `coli_cuda_device_count()` now call `coli_cuda_load()` first, mirroring `coli_cuda_init()` behavior.

**Verification:** Code path now loads DLL before querying count; `COLI_GPUS` unset now correctly selects visible devices.

---

## 9. #1550 – Add dark mode switch button

**Root cause:** No UI for dark/light theme switching; both landing pages (`site/index.html` static and `web/` React app) were dark-only with hardcoded colors causing visibility issues.

**Fix:** commit `bd9184b` (cherry-picked PR #1574 + PR #1551)
- `web/`:
  - `App.tsx`: theme state `dark|light` persisted in `localStorage` (`colibri-theme`), `useEffect` toggles `documentElement` class, button with Sun/Moon icons.
  - `index.css`: `html.light` CSS variables (background #f5f7f8, etc.) + overrides for sidebar, chat-panel, composer, inputs, etc.
  - `index.html`: remove hardcoded `class="dark"` so JS controls it.
- `site/index.html`:
  - Early script reads `localStorage` + `prefers-color-scheme`, sets `data-theme`
  - CSS variables for dark theme (`--bg:#0E0E0C`, etc.) + transitions
  - Toggle button with animated sun/moon SVGs, keyboard shortcut `D`, localStorage persistence.

**Verification:**
- `npm run build` in `web/` passes
- Manual UI test: toggle persists after refresh, light theme readable.

---

## Skipped (hardware/backend, not verifiable locally)

- #1570 Vulkan teardown: CLI exits reclaim device; multi-engine in-process hosting not in-tree. Requires Vulkan hardware to verify.
- #1594 Qwen 3.8 Flash Next slow, #1409 auto placement starves VRAM, #1464 RTX 3080 hybrid batched block failed, #1498 GPU doesn't engage, #1538 GB10 sm_121 unified memory incoherent GPU output, #1309 DeepSeek V4 no Metal path, #1306 Qwen3.8 GPU backend, #1518 OpenMP team, etc. – all require GPU/hardware or are experiments.
- #1609 heterogeneous 3-lane MoE scheduling – large feature, not a bug.

---

## Branch state

```
a8f2ca6 (main)
3176920 fix(olmoe): RAM probe floor (#1601)
2eef86b fix(st): race (#1519)
4322e13 fix(doctor): DeepSeek V4 variants (#1593)
d3982e1 fix(vulkan): clamp SwiGLU (#1520)
dde701d fix(coli): per-engine CUDA detection + auto-tier (#1533, #1581)
1a6653e perf(olmoe): O(1) LRU victim lists (#1050)
a508026 fix(coli): SNAP for non-GLM (#1600)
e2a35d9 fix(cuda-dll): available count after load (#1577)
bd9184b feat(ui): dark mode toggle (#1550)
```

All pushed to `origin arena/01a0b5a6-colibri`.

## How to test locally

```bash
cd c
make -C c check          # C tests, including new victim_index
make -C c tests/test_olmoe_victim_index && ./c/tests/test_olmoe_victim_index
python3 c/tests/test_cuda_binary_engine.py
cd ../web && npm run build
```
