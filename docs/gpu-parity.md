# Vulkan and CUDA on the same models

One machine, one build of each engine with both backends (`VK=1 CUDA=1`), the same models
and the same requests on the CPU, Vulkan and CUDA. It shows where each backend stands for
each engine and what is left between them.

**The machine:** an RTX PRO 4000 Blackwell SFF (24 GB, PCIe 4.0 x8), an i5-13500 (6
performance and 8 efficiency cores) and 62 GB of RAM, on Linux.

**The method:**
- A fresh server for each cell, with `USAGE_SAVE=0` so that the expert history stays as it
  was (DeepSeek V4 on a copy of one history).
- A 120-word prompt (163 to 215 tokens, by tokenizer) and 128 tokens, after one warm-up
  request.
- Prefill and decode in tok/s, the range of three runs.
- The backends chosen with `COLI_CUDA=0 COLI_VULKAN=0` (CPU), `COLI_CUDA=0 COLI_VULKAN=1`
  (Vulkan) and `COLI_CUDA=1 COLI_VULKAN=0` (CUDA).

| model | CPU | Vulkan | CUDA |
|---|---|---|---|
| Qwen3.6-35B-A3B, colibri int4 group 64 (22 GB) | 108-112 / 13.9-14.0 | 1242-1250 / 84.3-84.7 | 1332-1464 / 100.7-102.8 |
| Qwen3.6-35B-A3B, GGUF Q4_K_M (21 GB) | 89-91 / 13.3-13.4 | 979-1067 / 73.9-74.4 | 882-941 / 97.8-98.4 |
| Qwen3.8 Flash Next, GGUF IQ2_XS (64 GB) | 27.5-28.0 / 9.3-9.9 | 95.9-97.4 / 58.3-61.0 | 98.1-99.2 / 59.2-63.4 |
| DeepSeek V4 Flash, GGUF UD-IQ1_S (82.5 GB) | 7.8 / 2.68 | 21.4-21.9 / 6.88-7.00 | 23.2-24.4 / 7.79-8.14 |
| OLMoE-1B-7B, GGUF Q4_K_M (4 GB) | 94-97 / 41.0-42.7 | 2835-3170 / 205.9-206.3 | no CUDA path |

Each cell is prefill / decode in tok/s, over runs 2 to 4 (the first run is left out because
some engines still fill their caches then). On CUDA, Qwen3.6 and Qwen3.8 run their dense
chain (`cuda_chain.cu`) beside their expert tiers. DeepSeek V4 runs the shared CUDA expert
tier (`cu_tier`), with its dense matrices on the card a matrix at a time (`cu_dense.h`) and
its attention core there too (`COLI_CUDA_ATTN_BATCH`).

## What the table says

- **Qwen3.6** runs faster on CUDA than on Vulkan in decode, with both containers: 101-103
  against 84-85 tok/s with colibri's int4, and 98 against 74 with the GGUF. In prefill CUDA
  leads with colibri's int4 (1332-1464 against 1242-1250) and trails with the GGUF (882-941
  against 979-1067).
- **Qwen3.8 Flash Next**, larger than the card (64 GB), comes out even: prefill 98-99 against
  96-97, decode 59-63 against 58-61.
- **DeepSeek V4 Flash**, also larger than the card, runs faster on CUDA: decode 7.8-8.1 against
  6.9-7.0, prompt 23-24 against 21-22. The attention core, the indexer and the
  hyper-connection mixes run on the card there too (`COLI_CUDA_ATTN_BATCH`, on by default for a
  GGUF whose dense part the card holds); without them CUDA read a prompt at 18-19.
- **OLMoE** has no CUDA path; Vulkan's chain runs it at 206 tok/s.
- **Kimi K3** UD-IQ1_S (594 GB) was measured in two other sessions on this machine, with the
  same method: 0.4-0.5 / 0.18-0.22 on the CPU, 1.0 / 0.28 on Vulkan and 0.6 / 0.26 on CUDA.
  Most of its experts come from the disk, which sets the pace. The experts' shards now read
  without the kernel's readahead, which had tripled what a token read from this disk: on a
  96-token answer, Vulkan's decode went from 0.269 to 0.349 tok/s, and the table's method
  gives 1.1 / 0.35-0.36 on Vulkan.

## Which backend a build with both picks

A build with both backends runs Qwen3.6 and Qwen3.8 on CUDA when `COLI_CUDA=1`, except a GGUF
whose expert types the CUDA chain does not take, and Qwen3.8 with colibri's int4-g64 or
low-bit experts, which the CUDA tier does not stream: those run on Vulkan. The Qwen3.8
release with the int4-g64 sidecar runs there at 100-102 / 23.7-24.9 tok/s, against 9-18 /
6.3-6.8 with its experts on the CPU and 7.2-7.4 / 3.1-3.2 with its FP8 experts on the CUDA
tier (176 GB, most of them from the disk). It runs the engines
whose CUDA path lacks Vulkan's dense chain on Vulkan unless `COLI_VULKAN=0`: a DeepSeek V4
GGUF, Kimi K3, Inkling, GLM-5.3, DeepSeek V4.1 and MiMo. `coli setup` builds both backends
for them on an NVIDIA card.

On the numbers above, DeepSeek V4 is the one place where that choice leaves speed behind: on
this 24 GB card `COLI_VULKAN=0` decodes 14% faster and reads a prompt 10% faster. The build
still picks Vulkan for it, since CUDA keeps its dense part (6 GiB for V4 Flash) on the CPU
where half of the card's free memory less the reserve does not hold it (a 12 GB card), and Vulkan's chain
then still puts part of it on the card.
