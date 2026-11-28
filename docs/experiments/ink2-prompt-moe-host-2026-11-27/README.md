# Inkling's prompt steps: the host's share of the expert tier (ink_vk_moe, moe_ex)

RX 9060 XT 16 GB (RADV, Mesa 26.2.5), Ryzen 7 7800X3D (`OMP_NUM_THREADS=8`), 32 GB.
unsloth Inkling-2-Mini UD-Q3_K_XL (every one of its 3072 experts on the device) and UD-Q4_K_XL
(2472 of them), `COLI_VK_STAGED=1`.

## Where a prompt step's host time went

A prompt chunk's layer step leaves the chain's frame for the host (the in-frame experts take
steps of up to `VKC_DMOE_ROWS` rows), and `moe_ex` / `ink_vk_moe` run the tier from there.
`handoff-timers.patch` (not part of the change) times their parts; the 8K prompt on Q3, 240 prompt
layer steps:

| part | dev | this change |
|---|---|---|
| the chain's frame (mode 0's attention, mostly) | 87.06 s | 87.46 s |
| `vkt_issue` + `vkt_join` (the experts on the device) | 15.49 s | 15.46 s |
| routing (`moe_ex` pass 1) | 1.73 s | 0.23 s |
| NVFP4's scaled pairs (and the ids' books) | 3.87 s | 1.44 s |
| the rows' rank-order sums | 0.93 s | 0.65 s |
| K/V rows down, the routed rows' copy | 0.49 s | 0.55 s |

Routing computed every expert's sigmoid and bias K times a row, on one thread; the NVFP4 layers'
(row, rank) pairs were scaled on one thread (S x K x D floats a step); the sums ran row after row.

## A/B

`ab.sh`: a fresh `coli serve --ctx 24576` on a fresh copy of one `.coli_usage` history, a warm-up,
then the 8K prompt with `max_tokens` 1; dev and the change interleaved, three rounds; medians:

| model | dev | this change |
|---|---|---|
| Q3_K_XL | 110.49 s (72.6 tok/s) | 105.65 s (75.9 tok/s) |
| Q4_K_XL | 114.28 s (70.2 tok/s) | 112.16 s (71.5 tok/s) |

Samples in `manifest-q3.json` and `manifest-q4.json`.

## The same text

Each row's sums keep their order and each pick sees the same values, so the bits do not move. On
Q3, where every expert is on the device and two runs give the same text, the greedy code answer (512
tokens) and the 8K needle answer are byte-identical to dev's (`raw.txt`). On Q4 two runs of dev
itself already give different text (which cold experts the device takes follows measured timings);
the needle is 5/5 in every run of both.
