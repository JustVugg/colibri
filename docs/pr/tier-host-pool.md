# vulkan: the tier's host pool sizes itself from the engine's RAM plan

*Refreshed against `upstream/dev` @ b90a369e (2026-10-21). Upstream landed its
own host-pool RAM accounting in #2258 (`ram_floor`/`ram_given`, engines hand
cache bytes back) plus a `ram_reserve` fallback. This PR now extends that
mechanism — the plan is a cap on the same `want` — instead of shipping a
second sizing rule beside it.*

## The bug that remains at b90a369e

The pool's `want` still starts from `MemAvailable − 4 GiB`. Upstream's
`ram_floor`/`ram_given` makes the engine hand cache bytes back once the pool
is sized, and `ram_reserve` holds back cache growth — but the grant is still
bounded by *free RAM*, never by the *resolved budget* the operator asked for.
Reproduction (DeepSeek V4 Flash, 38 GiB host, `--memory-gb 18`):

```
ram_tiers available=18.00GiB ... target_cache=15.53GiB ... projected=17.65GiB
[VK] tier deepseek_v4: host pool on, 1742 experts in 25.75 GiB of system RAM ...
radv/amdgpu: Not enough memory for command submission.
[VK] chain: queue submit failed (-4): the device is lost
```

17.65 GiB committed by the plan + ~25 GiB granted to the pool (whatever the
floor/reserve left of free RAM) = the kernel kills the process at the first
big submission. The KV/workspace reserves, pinned rows and dense weights the
plan already spent are invisible to the pool. `deepseek_v4` wires neither
`ram_floor` nor `ram_given` upstream, so it takes the plain `ram_reserve`
branch.

## The fix

`VktConfig` gains `ram_planned`, `ram_budget`, `ram_spare`: the engine
resolved a budget and projects what it holds at capacity. When
`ram_planned` is set, the pool's `want` is capped at `ram_spare` (what the
plan leaves unclaimed) — composed with upstream's flow, not replacing it:

- engines with `ram_given` (colibri, glm53, qwen36/38, …) keep upstream's
  behaviour byte for byte: same `want`, same floor subtraction, same
  callback;
- engines with only `ram_reserve` keep upstream's rule exactly;
- a planned engine sizes beside its whole plan and the floor subtraction is
  skipped — the plan already accounted for the caches, subtracting again
  would double-count;
- `COLI_VK_TIER_HOST_GB` still wins over the plan (one line when it goes
  over), never over `MemAvailable`; `COLI_VK_TIER_HOST=0` still disables, and
  every off path prints why.

`deepseek_v4` is the one engine with a plan (its `--memory-gb` resolves
cache, dense, KV and head budgets before `vkt_init`). Via the `XT(init)`
dispatch the same cap governs the CUDA tier's host pool for free. The
`vkt_host_room()` helper stays as the pure-arithmetic core (unit-tested both
ways); it is mirrored into `cut_host_room` for the CUDA double-compile.

Same host after the fix: `host pool on, 23 experts in 512.0 MiB ... (the RAM
plan leaves 356.3 MiB of the 18.00 GiB budget)` — and the run completes.

## Tests

`test_vk_tier` gains `host room` (the arithmetic, no device) and
`host pool` (a plan's room bounds the pool; a plan that leaves nothing turns
it off with a line; the operator's GiB wins with one warning). All of
upstream's tier cases (promote, hybrid, pool copy, host pairs) run unchanged
beside them.
