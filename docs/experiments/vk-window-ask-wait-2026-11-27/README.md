# The window ask and the driver's budget (#3148, #3150)

At a start, `vk_window_decide` asks the host-visible device-local type for 9/10 of its heap in
1 GiB blocks and frees them. A tier that sizes itself from `VK_EXT_memory_budget` soon after reads
whatever part of those blocks the driver still counts. RX 9060 XT 16 GB, RADV (Mesa 26.2.5),
Linux 7.2.9.

## How fast RADV gives the blocks back

`vk_release_curve.c` does the ask as `vk_window_decide` does (13 x 1 GiB of the same memory
type, freed at once) and reads the heap's free memory every millisecond for 3 s. For each moment
it prints the budget colibri's tier would take there, `free - COLI_VK_TIER_RESERVE_GB (1) -
chain_bytes (2.44: Inkling-2-Mini UD-Q3_K_XL at CTX_MAX=73728)`.

```sh
cc -O2 vk_release_curve.c -lvulkan -o vk_release_curve
DEV_ID=0x7590 DENSE_GB=0 CHAIN_GB=2.44 TIER_MS=20,40,60,100,150,170 ./vk_release_curve 5 13
```

`curve.txt`: the blocks come back one GiB about every 11-12 ms (in some rounds two GiB every
12-13 ms), all of them 97-165 ms after the free:

| the tier sized at | budget |
|---|---|
| 20 ms | 0 (no room) |
| 40 ms | 1.37-3.37 GiB |
| 100 ms | 6.37-12.37 GiB |
| 150 ms | 11.37 GiB (the slower rounds), 12.37 GiB |
| 170 ms | 12.37 GiB |

`curve-gpu-load.txt`: the same with `tests/test_vk_tier` looping beside it (the GPU 14-33%
busy): the same 154-164 ms. Its budget lines are from the tool's default `DENSE_GB=0.89`, counted
twice beside the chain's 2.44; only its curve is meant here.

## What the engine does with it

`starts-timestamped.txt`: `./inkling --chat -f hi.txt -n 1` (default memory mode,
`CTX_MAX=73728`), stderr timestamped. Dev's tier sizes itself 151-172 ms after the ask's line and
takes 11.36 GiB, as the curve says at 150 ms, in 4 starts of 5. The one at 172 ms took 12.36 GiB,
as the curve says at 170 ms. With #3150 the ask's line reads `found in 0.2 s` (the wait) and the
tier takes 12.36 GiB in 5 of 5.

`starts-plain.txt`: the same without the timestamps, 21 starts each (`CTX_MAX` 16384, then 73728):
dev 1 short (10.36 GiB), #3150 none.

## The budget of 0 in #3148

Neither the curve nor about 80 starts gave one here. That includes starts right after a llama.cpp
server (stopped, killed, or not yet gone), after the router unloaded a 15.7 GB model, after 5
minutes of an idle GPU, and with every CPU busy. By the curve, a budget of 0 needs the tier to size
itself within about 20-30 ms of the free, or a driver that gives the blocks back about four times
slower (about 45 ms a GiB, over half a second for 13). The wait's 3 s bound covers about 230 ms a GiB.
