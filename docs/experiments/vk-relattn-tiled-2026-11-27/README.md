# Inkling's prompt attention in tiles of rows (chain_relattn_t.comp, #3149)

RX 9060 XT 16 GB (RADV, Mesa 26.2.5), Ryzen 7 7800X3D, 32 GB. unsloth Inkling-2-Mini
UD-Q3_K_XL GGUF, every one of its 3072 experts on the device (`COLI_VK_STAGED=1`).

## Profile before

`COLI_VK_CHAIN_PROF=1`, mode 0 (each workgroup one row and head, every position's K and V read
for it): attention was 83.5 s of a 8K prompt's 87.4 s of device time, and 262 s of a 16K prompt's
270 s. The expert GEMMs were 0.1 s.

## A/B: the same build, COLI_VK_RELATTN_TILE=1 against 0

`ab.sh` (prof.sh per run): a fresh `coli serve --ctx 24576` on a fresh copy of one `.coli_usage`
history, a warm-up request, then the prompt (`max_tokens` 8); prefill is the time to the first
streamed token. Tiles on and off interleaved, three rounds; medians:

| prompt | tiles off | tiles on | attention, device time |
|---|---|---|---|
| 8K (8021 tokens) | 111.4 s, 72.0 tok/s | 45.3 s, 177.1 tok/s | 83.5 s -> 16.9 s |
| 16K (15769 tokens) | 320.1 s, 49.3 tok/s | 109.0 s, 144.6 tok/s | 261.9 s -> 52.4 s |

Samples in `manifest-8k.json` and `manifest-16k.json`, the runs' profile lines in `raw.txt`.

## Quality

- `tests/test_vk_chain`: every tiled case within 2e-5 of the double reference: one tile of rows,
  a partial last tile, a ring wrapped more than once, Inkling-2-Mini's head (128), window and
  ring (1024), bank (16) and extent (1024), sinks, a global layer past the extent, and slices of
  rows over submissions. Mode 0 keeps its cases with the tiles off; a bank past 32 keeps mode 0.
- `needle.sh` (thinking on): the five facts 5/5 at 8K, 32K and 64K with the tiles on. 32K's
  prompt took 321 s (919 s in mode 0); 64K's completed (1263 s) where mode 0's lost the device in
  its first chunk (#3149).

## Not this change

At 64K the chain's device memory reached 6.15 GiB beside the tier's 11.4 GiB on a 16 GB card, and
the decode after the prompt ran at 1.05 tok/s (11.1 at 32K). Decode steps do not take the tiled
shader, which allocates nothing; the chain's K/V mirrors outgrow its plan at that context.
