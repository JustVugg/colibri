# CACHE_ROUTE — opt-in cache-aware MoE routing

**Default: OFF.** Stock full top-K router behavior unless `CACHE_ROUTE=1`.

Paper-style max-rank selection (arXiv:2412.00099): keep true top-`J` always;
fill remaining slots preferring experts already **resident** (pin ∪ LRU) that
still rank inside top-`M`.

This is **routing-side** (can change which experts run). Complementary to
**PILOT** (next-layer *prefetch* of weights; does not change expert IDs).

## Flags

| Env | Default | Meaning |
|-----|---------|---------|
| `CACHE_ROUTE` | `0` | `1` = max-rank cache-aware fill (pin∪LRU prefer) |
| `ROUTE_J` | `2` | Always take true top-J (even if uncached) |
| `ROUTE_M` | `12` | Max-rank window for resident preference |
| `ROUTE_P` | `0` | Optional cumulative mass window (`0` = use fixed M) |
| `ROUTE_ALPHA` | `1` | Scale gate mass of *substituted* experts before renorm (`1` = off) |
| `ROUTE_AGREE` | auto | Overlap% + KL vs true top-K; auto-on when `CACHE_ROUTE=1` |

## One-liners

```bash
# Stock full-K (leaderboard-comparable routing)
CACHE_ROUTE=0 ./coli chat

# Experimental CACHE_ROUTE
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12 ./coli chat

# Wider prefer window (more hit, more possible swap)
CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=16 ./coli chat
```

## Stats

Footer / serve `STAT` when enabled:

- `swap N%` / `swap_pct` — fraction of chosen slots not in true top-K
- `route_swaps` / `route_slots` — raw substitution counters
- `route_agree` — |chosen ∩ true top-K| / K
- `route_kl` — mass KL (true top-K vs chosen)
- `hit N%` — expert cache hit (disk residency)

## A/B vs PILOT

```bash
# A: cache-aware routing only
CACHE_ROUTE=1 PILOT=0 ...

# B: lookahead prefetch only (does not change expert IDs)
CACHE_ROUTE=0 PILOT=1 ...

# C: both
CACHE_ROUTE=1 PILOT=1 ...
```

## Scope of this PR

**Routing-only** + telemetry + this note. Does **not** require CUDA/fuse/device-tier
patches — CPU streaming + pin/LRU is enough to A/B the lever against PILOT / #119.

Treat as experimental until quality gates (e.g. `./coli bench`) pass; do not
default `CACHE_ROUTE=1`.
