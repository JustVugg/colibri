---
title: "GGUF models"
description: "Run a GGUF you already have, read in place: no conversion, no second quantization."
---

# GGUF models

colibri runs the GGUF files people already have on disk. The ggml blocks are computed as the file holds them, from IQ1_S to Q8_0 and MXFP4: nothing is converted or quantized a second time.

```bash
./coli import ~/models/GLM-5.3-UD-IQ1_S-00001-of-00006.gguf   # first part of a split model
./coli chat --model ~/models/GLM-5.3-UD-IQ1_S.colibri         # or coli serve / coli web
```

`coli import` writes a model directory next to the GGUF, named after it, or wherever `--out` says. The directory holds `config.json`, the tokenizer, the chat template and `gguf.json`, which points at the file. From there `coli chat`, `coli serve`, `coli web`, `coli plan` and `coli doctor` treat it like any other model.

## Supported architectures

| GGUF architecture (`general.architecture`) | Models | Engine |
|---|---|---|
| `qwen35moe` | Qwen3.5 / Qwen3.6 MoE | qwen36 |
| `qwen35` | Qwen3.5 / Qwen3.8 dense | qwen36 |
| `qwen3moe` | Qwen3 MoE, Qwen3-Coder | qwen36 |
| `qwen3` | Qwen3 dense | qwen36 |
| `flint` | Flint (RecursiveSI/flint; [qwen36.md](qwen36.md#flint)) | qwen36 |
| `muse-spark` | Muse Spark Open, text ([qwen36.md](qwen36.md#muse-spark-open)) | qwen36 |
| `qwen4exp` | Qwen3.8-Flash-Next (and its MTP head) | qwen38 |
| `deepseek4` | DeepSeek V4 Flash | deepseek_v4 |
| `deepseek41` | DeepSeek V4.1 Flash (files with e4m3 engram rows, and the published K-quant and IQ files; `dflash`, its DSpark head, with `--mtp`) | deepseek_v41 |
| `inkling` | Inkling | inkling |
| `glm5-next` | GLM-5.3-Flash | glm53 |
| `glm-dsa` | GLM-5.2, GLM-5.3 | colibri |
| `mimo2` | MiMo-V2.6 | mimo |
| `olmoe` | OLMoE | olmoe |
| `kimi-k3` | Kimi K3 | kimi_k3 |

A GGUF of another architecture is refused by `coli import` with the list above.

## Two modes

- **`--mode link`** (the default): the engine reads the GGUF where it is. Nothing is copied.
- **`--mode copy`**: also writes `model.colibri.gguf`, with every dense tensor byte for byte and each layer's routed experts re-laid out one expert per 4 KiB-aligned slot (gate, up and down contiguous), so that an expert is one aligned read. It works for every architecture with routed experts but `qwen4exp`, whose engine reads in place only.

Both are lossless: the same blocks, the same tokens.

## What the engines do with the blocks

- **The CPU interleaves rows by 16.** Matrices whose type has the form (Q4_K, Q5_K, Q6_K, Q3_K, IQ4_XS, Q8_0, Q4_0, IQ4_NL) are held with their rows interleaved by 16, which speeds up prompts and speculative verification. Experts are packed when they are read, dense matrices once at load. Each engine's switch keeps the file's rows, as `GQ_EXACT=1` does:

  | Engine | Switch |
  |---|---|
  | qwen36 | `Q36_ILV16=0` |
  | qwen38 | `Q38_ILV16=0` |
  | deepseek_v4 | `COLI_V4_ROWS16=0` |
  | deepseek_v41 | `V41_ILV16=0` |
  | inkling | `INK_ILV16=0` |
  | glm53 | `G53_ILV16=0` |
  | colibri | `GLM_ILV16=0` |
  | olmoe | `OLM_ILV16=0` (dense matrices; its experts keep the file's rows) |

- **The Vulkan expert tier** (`COLI_VULKAN=1`) takes a GGUF's experts as the file has them. Each layer keeps its own types, since published files mix them. The layers of the types most layers have keep their experts at their own size on the device and in the host pool, and only the few other layers take the room of the biggest of theirs ([vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc): 25-31% more experts held, GLM-5.3 Flash UD-IQ2_XXS from 2.50-3.14 to 2.92-3.40 tok/s of decode, Inkling-Small from 7.87-8.11 to 10.92-11.53).
- **MXFP4 experts** (DeepSeek V4 and V4.1, MiMo) are laid out as their checkpoints' FP4, so they run on those engines' own kernels and device forms.
- **MiMo's other types.** A MiMo GGUF of K-quants or IQ types (ggml-org's MiMo-V2.6-Flash Q2_K_S, 106 GB) keeps its blocks: the trunk on the CPU's integer kernels and on the Vulkan chain's block formats, the experts each layer its types, on the CPU and in the Vulkan tier. On an RTX PRO 4000 with 62 GB of RAM: 7.8 tok/s of decode and 8.2 of prompt, the CPU's greedy text. The engine used to refuse every type but MXFP4 experts and a BF16 trunk.
- **OLMoE's experts keep the file's rows**: they fit in RAM, and interleaving one at its read cost more than the 16-row kernels gave back on its matrices. llama.cpp's `olmoe` files carry no `norm_topk_prob`; the import writes false, as the official checkpoints have it and as llama.cpp computes them.
- **DeepSeek V4.1's engram tables** are read a row at a time, as from the checkpoint: a `deepseek41` file holds each row's e4m3 values and their scales together. The published K-quant and IQ files keep every table in one tensor of a ggml type (`per_layer_token_embd`, IQ4_NL), read the same way, a row of 256 values in 144 bytes. They carry the config as keys rather than the checkpoint's `config.json`: the importer writes it from them, and the engram's sidecar (`dsv41_engram.json`) from the `ple` keys, whose primes, hash multipliers and offsets are the ones `tools/prepare_dsv41.py` derives from the checkpoint. Their `engram_gate` is the engram's two norm weights multiplied, which is all the gate reads of them.
- **DeepSeek V4.1's DSpark head** ships as a GGUF of its own (arch `dflash`, `dspark-*.gguf`: three stages under V4.1's layer names, MXFP4 experts, Q8_0 or BF16 projections, and the head's `fc`, norms, confidence and Markov tensors). `--mtp dspark-....gguf` links it: the importer checks it against the model's `dspark` keys and names it in `gguf.json`, the engine reads its stages as `mtp.<s>.*`, their MXFP4 experts as the checkpoint's FP4 in slots of their own, and the planner prices them as the checkpoint's. A head next to the model is only named, because its drafts measured slower: UD-IQ1_S with the Q8_0 head on an RTX PRO 4000 (24 GB) with 62 GB of RAM, the Vulkan chain, a 163-token prompt and 128 tokens, accepted 22-33% of its drafts, and decode went from 5.01-5.29 tok/s without the head to 4.29-4.64 with it (1.72-1.89 with `V41_DSPARK_MINACC=0`). The head was trained on the full model, and a 1-bit one agrees with it less; the verify rows also bring more of the experts the RAM cannot hold. The greedy text is the same bytes with and without it.
- **Kimi K3** (`kimi-k3`). The importer writes the checkpoint's config from the keys: a layer whose `head_count_kv` is 0 is a KDA layer, the others MLA. The engine takes the file's tensors as their checkpoint counterparts:
  - `attn_res_score`, `ffn_res_score` and `output_res_score` are the AttnRes norm and projection already multiplied, which is all the engine reads of them.
  - `ssm_a` holds -exp(A_log): the engine's decay rate A = exp(A_log) is its negative.
  - MLA's `attn_k_b` (transposed) and `attn_v_b` are put back into `kv_b_proj`.
  - The routed experts sit in the latent (`ffn_routed_down` and `ffn_routed_up` are the latent projections) and keep the file's blocks on the CPU and in the Vulkan tier. A GGUF's `lm_head` stays on the host.
  - unsloth's UD-IQ1_S (594 GB: F32, Q6_K, Q8_0, IQ2_XXS and IQ1_S) runs. On an RTX PRO 4000 (24 GB) with 62 GB of RAM, a 181-token prompt and 128 tokens: the CPU 0.4-0.5 tok/s of prompt and 0.18-0.22 of decode, Vulkan 1.0 and 0.28. The chain holds 45 of the 93 layers (21.3 GiB) and the tier 106 experts on the card and 2839 in the host pool; most experts still come from the disk. UD-TQ1_0 and UD-Q1_0 use ggml types 65 and 66, which colibri does not know: `coli import` refuses them ("unsupported ggml type 65").
  - The experts' shards are read without the kernel's readahead (`POSIX_FADV_RANDOM`), three slices an expert. With it, on a two-NVMe md array whose readahead is 2 MiB, decode read 17.3 GB a token for 6.7 GB of experts. Without it the read is 5.8 GB, and a 96-token answer on Vulkan decodes at 0.349 tok/s against 0.269, its 31-token prompt in 46.5 s against 54-57 s.
- **A GGUF's experts read without the kernel's readahead.** A layer's experts sit in three tensors, so an expert is three slices read apart through the page cache, and the readahead read past each one. The files that hold them take `POSIX_FADV_RANDOM` on every drive, the mirrors too (`st_no_readahead` in `c/st.h`): colibri (GLM-5.2), glm53, mimo and kimi_k3. The engines that read their experts with `O_DIRECT` (deepseek_v4, deepseek_v41) never had it. GLM-5.2 UD-IQ1_S (202 GB) on Vulkan, an RTX PRO 4000 with 62 GB of RAM and a two-NVMe md array whose readahead is 2 MiB, a 96-token answer, two runs without readahead and three with it, alternated: 2.3-3.6 GB read a decoded token against 4.7-6.6 with the same number of experts loaded, 0.633-0.658 tok/s against 0.567-0.602, and 41 GB read for the prompt against 64-67.
- **GLM's MLA**: `k_b` / `v_b` (llama.cpp splits `kv_b_proj`) stay the file's blocks on the CPU; the Vulkan chain (glm53 and colibri) takes them as floats, since its MLA core reads them a head at a time.
- **The Vulkan dense chain** (qwen36, qwen38, glm53, colibri, deepseek_v4, deepseek_v41, olmoe, inkling, mimo and kimi_k3) takes a GGUF's dense matrices as the file has them too. Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S, IQ4_NL and IQ4_XS run on a GEMV with one decoder per type (`shaders/chain_gemv_bd.comp`), and the other types on the general block decoder. V4.1's grouped output projection (`wo_a`) reads no blocks, so it goes up as int8 from Q8_0 and as floats from the other types. The routed experts run inside the same frame when the tier holds every expert of a layer. A decode step or a verify of up to 16 rows runs there; a prompt's longer chunks go through the CPU and the expert tier.
- **The CUDA backend takes a GGUF's blocks too** (`cuda_bd.cuh`): every ggml block type of 32 weights or more, each row as the file has it. A decode step and a speculative verify (up to 8 rows, sharing each weight load, each row with a decode step's bits) run in the CPU's integer form on Q8_0, Q2_K to Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and Q2_0 (the CPU's codes, scales and integer sums: the two differ only in the order of the float sums), the other types in floats. A prompt chunk goes through the f16 tensor-core tiles (`COLI_CUDA_BD_TILES`).
- **qwen36's CUDA chain runs a GGUF whole on the card**: its experts in the CUDA tier as the file has them (gate/up and down types by layer, one arena a device: `QT_GGUF_ARENA`), computed in the chain's frame (without the chain, through the tier's expert groups, the misses on the CPU), and its dense matrices, lm_head and DeltaNet projections as their blocks. Qwen3.6-35B-A3B on an RTX PRO 4000 (24 GB), every expert on the card:

  | | CUDA chain | Vulkan chain |
  |---|---|---|
  | UD-IQ2_XXS: decode, prefill of 467 tokens | 118 tok/s, 1356 tok/s | 100 tok/s, 1600 tok/s |
  | UD-IQ2_XXS: a code edit of 400 tokens (prompt lookup) | 2.38 s | 3.3 s |
  | UD-Q4_K_M: decode, prefill | 96 tok/s, 1388 tok/s | 75 tok/s, 1450 tok/s |
  | UD-Q4_K_M: the code edit | 2.67 s | 3.1-3.7 s |

  The greedy text is the CPU's (the same integers). Before, a CUDA run of a GGUF computed every routed expert on the CPU (UD-IQ2_XXS: 13.7 tok/s of decode).
- **qwen38's CUDA chain runs a GGUF too**: the trunk and the experts on the CUDA tier as the file has them (warm from the history), the experts the card leaves in a host pool of page-locked RAM the card reads over the bus (`COLI_CUDA_CHAIN_POOL`: a decode step's staged on the card first, a prompt chunk's layer copied over DMA while the layers before it run), routed and computed in the chain's frame with no host step. Qwen3.8-Flash-Next IQ2_XS (68 GB) on an RTX PRO 4000 (24 GB): prompt 209-213 tok/s, decode 49.6-50.8 (before: 19-21 and 11.6; the Vulkan chain 194-197 and 44-51). The greedy text is the CPU's and the Vulkan chain's. A CUDA run of such a GGUF stays on CUDA in a build with both backends.
- **The other engines' dense matrices stay on the CPU** on CUDA, and their CUDA tiers take no ggml block types yet.
- **A GGUF those engines' CUDA path does not take runs on Vulkan on an NVIDIA card** (qwen38 for a file whose expert types the CUDA chain's frame lacks, colibri for GLM-5.2, whose CUDA path keeps a GGUF's matrices and experts on the CPU, deepseek_v4, whose CUDA tier mirrors native FP4 experts only, and qwen36 for a file whose expert types the CUDA chain's frame lacks). A CUDA run of such a GGUF computed every routed expert on the CPU (Qwen3.8 IQ2_S/Q2_0 on an RTX PRO 4000: 28 tok/s of prompt, 9 of decode). A build with both backends switches such a run to Vulkan on the same card when CUDA was asked for and `COLI_VULKAN` was not set (114-118 and 46-49 tok/s); `COLI_VULKAN=0` keeps CUDA. `coli setup` builds these engines' CUDA engine with Vulkan too where the Vulkan toolchain is there (on Linux; the Windows CUDA package has both).

## Ollama's files

The blobs Ollama pulls for Qwen3.5 / Qwen3.6 (`qwen3.6:35b`: 23.9 GB, Q4_K_M, under `/usr/share/ollama/.ollama/models/blobs/sha256-...` on Linux) come from Ollama's own converter, not llama.cpp's, and differ in three places `coli import` and the qwen36 engine know: `attention.head_count_kv` is an array per layer (0 on the DeltaNet layers), whose maximum is the config's; the DeltaNet's dt bias is `blk.N.ssm_dt` (llama.cpp's `ssm_dt.bias`, the same values); and the MTP head (`mtp.*`) and the vision tower (`v.*`, `mm.*`) sit in the model's file, which the engine leaves unread. `ssm.v_head_reordered` and `rope.mrope_interleaved` carry no change of layout. A symlink with a `.gguf` suffix on the blob serves as the import's file:

```bash
ln -s /usr/share/ollama/.ollama/models/blobs/sha256-<digest> ~/models/qwen3.6-35b-ollama.gguf
./coli import --out ~/models/qwen3.6-35b-ollama ~/models/qwen3.6-35b-ollama.gguf
```

Qwen3.6-35B-A3B from that blob on an RTX 3070 (8 GB) with the CUDA chain, 2619 experts on the card: 73-78 tok/s of decode where Ollama 0.40.3 decodes the same file at 38 on the same card.

## Checking a GGUF before downloading it

`tools/gguf_header.py` fetches only the header of a model on the Hugging Face Hub (a few MB). With `GGUF_CHECK=1` the importer writes the model directory from that header alone, so you see the config it would get. The qwen36, deepseek_v4 and deepseek_v41 engines then check the folder against every tensor they would load, without reading one:

```bash
python3 tools/gguf_header.py unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr
GGUF_CHECK=1 python3 tools/gguf_import.py /tmp/hdr/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf /tmp/hdr/imp
GGUF_CHECK=1 SNAP=/tmp/hdr/imp ./qwen36
```
