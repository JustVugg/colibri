# The forked decode path (`COLI_VK_V4_DECODE_PATH=fork`)

The fork's decode expert path (`colibri-ds4-vulkan`) keeps its expert weight tensors as
zero-copy views over **persistent GPU-visible slabs** registered once, so a per-expert
alias is 3 callocs and zero driver objects, and the records themselves live in a RAM
arena for the slot's whole LRU lifetime. The port had the fork's *submit topology* (the
finish-site expert-batch group, `COLI_VK_TIER_EXPERTS`) but not that *tensor lifetime*:
it re-imported each store slot's malloc pages through `VK_EXT_external_memory_host`
every token and minted 2 `VkBuffer`s per matrix (`coli_vk_slab_import` + `slab_buf`).

This lever ports the tensor lifetime, not the kernels — the port's device kernels,
record layout and rounding points are untouched (port-parity by construction; the
token-exact harnesses transfer unchanged).

## Mechanism

1. **The slab registry** (`coli_vk_slabreg_*`, `backend_vulkan.c`): a small number of
   large HOST_VISIBLE|HOST_COHERENT, non-device-local arena blocks (`G.memtype_sys`,
   the fork's `memtype_sys` class; cached preferred), each block mapped once, slabs
   carved at `max(4096, bind alignment)` — one `STORAGE` (`+SHADER_DEVICE_ADDRESS`
   where the device has BDA) `VkBuffer` a slab, bound at its offset.
2. **Store slot routing** (`deepseek_v4.c`, the three slot-alloc sites + the destroys):
   under the arm, slot slabs come out of the registry; the loader's disk reads land in
   the mapped pages exactly as before (O_DIRECT included: slabs are 4096-aligned and
   sized `record_bytes + 8192`, the posix_memalign path's own capacity).
3. **The alias seam** (`cs_alias_expert`'s forked branch, `vk_tier.c`): a registry-backed
   slot aliases straight off the slab's own buffer — `coli_vk_slabreg_alias` reuses
   `coli_vk_tensor_alias_wide`'s geometry/bounds/wide-verdict rules minus the import:
   `t->wbuf = t->sbuf = slab->buf` (whole-buffer binding), the record's `woff`/`soff`
   carried on the tensor as the descriptor offsets, `pool = slab = NULL`, free is
   `free(t)`. `COLI_VK_TIER_CS_ALIAS_SCAN_SKIP`'s verdict cache composes unchanged
   (content-keyed); `COLI_VK_TIER_CS_ALIAS_CACHE`'s keepalive is superseded (one line).

The run line carries `N registry` beside the alias count; the boot lines carry
`decode-path=fork arena=..GiB slabs=N slab=..MiB` and the registry's own arming line.

## Degradation ladder (logged, never fatal)

1. No coherent host-visible type / block alloc fails / arena cap reached → that slot
   keeps the malloc path and its expert takes the standing import branch exactly as
   today.
2. A per-expert refusal (format, bounds, the descriptor-offset alignment) → that
   expert computes on the CPU in the finish loop (the standing `cs_fallback`).

## The heap overcommit guard (gap3 arena probe)

`vkAllocateMemory` past the host heap's reported size can **succeed silently** on RADV
and then every queue submit fails (`radv/amdgpu: Not enough memory for command
submission`; 10×2 GiB over a 19.46 GiB heap on the bench RX 6900 XT — allocations all
"succeed", the device is unusable). The registry therefore never asks past the room it
can see: the arena is capped at `COLI_VK_V4_SLAB_ARENA_GB` if set, else the heap's
budget-usage room, always clamped to 7/8 of the heap's size. At 9×2 GiB the same probe
verified the full mechanism: 2000×12.8 MiB STORAGE+SDA slabs across mapped blocks, one
dispatch reading 6 slabs spread across blocks byte-exact (0/65536 word mismatches,
re-verified after timing), and 1000 dispatch+submit+fence cycles at 82-93 µs/cycle
against 91-123 µs/cycle for the same cycle over a small host import (ratio
0.67-1.02×, gate 2×). Probe artifacts: `/tmp/gap3-arena/` (`run9.log`, `run9b.log`,
`run10.log`, `run13.log`).

## Gates

- `make deepseek-v4-tiny-check-forkdec` — the `-cs` mirror with the fork arm: tiny
  plain, prefix + brio forced-alias (`COLI_VK_TIER_GB=0 COLI_VK_TIER_WARM=0
  COLI_VK_TIER_PROMOTE=0`: no resident budget, every selection aliases off its
  registry slab), the growth-composition leg, and the engagement line asserted
  (`--expect-stderr`).
- `make -C c VK=1 vk-tier-check`'s seventh leg — `tests/test_deepseek_v4 forkdec`:
  registry-backed slots, an LRU victim's **in-place rewrite** serving its new expert's
  exact bytes through the *same slab pointer*, the registry alias's geometry + f16 wide
  verdict read off the slab's own scale bytes, `free(t)`'s own-nothing lifetime, and
  the malloc ladder when no device is available.
- Model-scale oracle (run where allowed): `make deepseek-v4-oracle` with the same env
  (teacher-forcing 32 + greedy 20), the `-cs` leg's class.

## Not ported from the fork (by measurement or design)

The `qdq.comp` host-side x-rounding (the port rounds on the host), `moe_swiglu` /
`moe_merge` (folded into `qmatmul_gate_up` act 2 and `expert_sum_dec`), the shared-
expert defer (the port's shared expert is a host path), the FFN opener stash, the
id-bank (measured ~8% slower in the fork and removed there), and the optional
S=1/grouped-GEMV adopters (spec §2.6 — separate levers, not this one).
