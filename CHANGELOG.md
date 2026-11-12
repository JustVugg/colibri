# Changelog

All notable changes to colibrì are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### New

- **The packed KV cache is the default on DeepSeek V4.1 Flash** (`COLI_VK_KV_FP32` unset,
  `1` is the exact fp32 opt-out): the chain's device KV mirrors *and* the host's own KV
  arrays stay in the checkpoint's cache formats — MXFP8 window rings (528 B a row), FP4
  compressed rows (288 B), FP4 index keys (68 B) — so the host state costs 890 B a position
  against fp32's 6,400 (86.1% less at the released geometry), read paths dequantizing on
  read and the boundaries (pull, park, reset, rollback) moving code rows. An approximate
  mode: the logits drift a little and the `kv-quant-deepseek` family gates that documented
  tolerance instead of bit-identity. The planner prices it — the host's window rings and
  the `(dspark_block_size + 1)` rollback rows a layer included, 3,032,832 B packed against
  11,763,712 B fp32, and 17,614,592 B at 16K, the engine's own `host kv` figure — reading
  the stage count the way the released `config.json` spells it
  (`num_nextn_predict_layers`, which it carries where `n_mtp_layers` is absent). New CI leg:
  `kv-quant-deepseek` in the `vulkan-engines` job runs the codec bit-for-bit, the six chain
  arms, the KV split's give-way and the CPU host
  ([deepseek-v41.md](docs/deepseek-v41.md#quantized-kv-mirrors-coli_vk_kv_fp32)).

- **Kimi K3.1 Pro** (`coli pull kimi-k3.1-pro`): the checkpoint as it ships, on the Kimi K3
  engine. K3.1 is K3's weights plus DeepSeek Sparse Attention on its 24 gated MLA layers: a
  token whose context passes 2048 positions attends to the 2048 its indexer picks, and up to
  2048 the attention is K3's, bit for bit. The indexer follows GLM-5.2's definition with no
  rotary part; Moonshot ships no K3.1 modeling code, so `tools/k3_ref.py` implements it apart
  from the engine and `make kimi-k3-tiny-check` checks a tiny K3.1 against it. The chat and the
  tokenizer are K3's, and a thinking request's `reasoning_effort` now reaches both models as
  their template's thinking effort (low, high, max). Both GPU chains run the indexer: the
  Vulkan chain with GLM-5.2's selection on the device, and a new CUDA chain for Kimi K3 and
  K3.1 (`kimi_k3_cuchain.h`, `K3_CUDA=1`): the first layers the card holds, the rest on the CPU.

- **K1b's grouped planar IDOT on pre-AVX2 x86 CPUs** (`IDOT_GS=1`, #2694): a build with SSSE3 and
  without AVX2 (Sandy and Ivy Bridge) now planarizes gs64 `fmt=4` tensors and takes the integer
  kernel's SSSE3 arms; `fmt=2` keeps the pair layout there, and with `IDOT_GS` unset nothing
  changes. Such a CPU has no FMA, and libm's software fmaf took 85 ns a call with glibc 2.39: the
  per-group step is now computed with fmaf's bits in double, rounded to odd. A 4096 x 512 gs64
  matrix on a Sandy Bridge build, two threads: 0.087 ms against the float grouped kernel's 0.22
  (S=1), 0.60 against 1.86 (S=8), the same bits as an AVX2 build's.
- **A start profile for a model's first answers** (`tools/usage_start_profile.py`): forty
  ordinary prompts through `coli serve`, their expert history and draft counts written beside
  the model as `.coli_usage.start` and `.coli_vocab.start`. Every engine with an expert history
  starts from the first where it has no history of its own (Qwen3.8 from the draft counts too),
  so its warm start puts on the device the experts ordinary text routes to (`COLI_USAGE_START`,
  `COLI_VOCAB_START`). Qwen3.8-Flash-Next IQ2_XS on an RTX PRO
  4000, the first answer after the start: its first 32 tokens 36.5-38.7 -> 60.7-63.2 tok/s.
- **Watts and joules per token on the Profiling page** (`c/power.py`): with
  [joules](https://pypi.org/project/landauer-gap/) installed next to the gateway
  (`pip install landauer-gap`), `/power` gives the machine's draw per device (NVIDIA, AMD and
  Intel GPUs, the CPU, Macs) and each `/profile` turn carries `energy_j`, what the devices used
  over it; the page shows the watts and the last turn's joules per token. Optional: without
  joules nothing runs and the page is as before. The meters start on the first `/power`.
- **The terminal UI, `coli tui`** (`c/tui.c`, `make colibri-tui`): the chat in a terminal, one
  C file with no libraries, a client of `coli serve`. The conversations, the answer streamed
  with its reasoning folded, the monitor (Overview and Profiling, as on the chat page, plus the
  GPU), the expert map, and the chat page's hummingbird with its moves, drawn as a sixel picture
  where the terminal can. It attaches to a running server or starts one for `--model`; the
  release archives carry it next to `coli` ([docs/tui.md](docs/tui.md)).
- **The desktop window** (`desktop/`, Tauri): the chat page in a window of its own. It finds
  `coli` and Python, uses the colibri server that runs or starts the configured one with
  `coli start` and waits for its model, serves the page on a fixed local port, and on close
  stops what it started. `--backend URL` points it at another server. The release archives
  carry it next to `coli` (`colibri-desktop`, `colibri-desktop.exe`)
  ([desktop/README.md](desktop/README.md)).
- **`/health` reports the NVIDIA cards** (`system.gpus`: use, memory, temperature, power, from
  `nvidia-smi` at most once a second), for the monitors.

### Performance

- **CPU without FMA: FP8 and int4-g64 experts read once for a prompt's rows** (#2715). On a
  CPU without AVX2 or FMA (Sandy and Ivy Bridge, a build for the x86-64 baseline or v2),
  quant.h's FP8 kernel decoded every weight again for each row and waited on one chain of adds:
  a Xeon E5-2690 v1 spent ~150 ms a prompt token in Qwen3.8's FP8 experts, about a decode step.
  Two rows or more now go into the lanes of a vector (8 with AVX, 4 with SSE2) with four output
  rows a pass, and a single row runs four output rows a pass. expert_ffn.h's int4-g64 f32
  kernel, the `experts-int4g64/` sidecar's, made its double-rounded multiply-adds one element
  at a time; it now makes them in double vectors, a block's codes widened once for every row and
  four blocks in flight. Every result keeps its bits: each row's sums are the scalar kernels'
  own operations in their order (`tests/test_nofma_kernels.c`, built for the x86-64 baseline
  and for Sandy Bridge). Qwen3.8's expert shapes on a Ryzen 7 PRO 8700GE built for Sandy Bridge,
  one thread: FP8 2.4x for a decode row, 7.7x for 4 rows, 15x for 8 or more; int4-g64 2.6-2.9x
  for a row, 3.8-4.1x for 4 or more (16 threads: FP8 1.5x and 7.5-9.6x, int4-g64 2.9-3.1x and
  4.3-6.1x). DeepSeek V4's and GLM's FP8 matrices take the same kernel.
- **CPU: qwen36 holds a GGUF's dense IQ3_S matrices as 4-bit codes** (#2704). IQ3_S packs four
  values into a 9-bit index of a 512-row grid, and the CPU gathered a grid row for every four
  values: those matrices read at a third of the RAM's speed on a Zen 4, about half of a token's
  time for Qwen3.8-27B UD-Q3_K_XL (27% of its bytes IQ3_S). qwen36 now holds them as one code a
  value (sign and level) in IQ4_XS's 16-row interleaved layout (`GQ_IQ3_S_R16`), read by its
  kernels with the codes' values for the IQ4 table: a quarter more bytes for those matrices, the
  same integer products. On a Ryzen 7 PRO 8700GE's CPU the 27B decodes at 2.82 tok/s instead of
  2.35 with the release's x86-64-v3 build (3.04 instead of 2.38 built native), RSS 12.3 -> 13.1 GB;
  an i5-13500, whose P-cores read the file's rows near their RAM's speed already, 2.4 -> 2.5. Only
  dense matrices, each at its first product on the CPU (a matrix a device holds keeps the file's
  rows: the 27B on a card that holds it all starts and takes RAM as before), only when the RAM free
  then holds that quarter and 2 GB to spare (the engine says which), `Q36_IQ3S_CODES=0` to keep the
  file's rows; experts and every other engine keep theirs. `tests/test_gq_r16` checks the form
  (round trip exact, the kernels to the bit against their reference), and
  `tests/test_qwen36_iq3s_codes.py` a small qwen35 GGUF with IQ3_S matrices: the same tokens, the
  layers' residuals within 1e-4.
- **CPU: the grid IQ types' dot product with the signs on the activation.** IQ2_XXS, IQ2_XS,
  IQ2_S, IQ3_XXS and IQ3_S applied their sign bits to the decoded weights, and the integer dot
  product took them off again to multiply by the activation. The grid rows now go in as they are
  and the signs onto the activation: the same two operands for every product, so the same bits
  (`tests/test_gq_signs` against the kernels it replaced). The indices of a sub-block come from
  one load, and IQ1_M's decoder reads its scales and index bits a word at a time. One Zen 4
  core, x86-64-v3, a 27B gate matrix: IQ2_XXS 3.58 -> 4.37 GB/s, IQ3_XXS 3.45 -> 5.22, IQ2_XS
  3.08 -> 3.53, IQ1_M 1.16 -> 1.90; sixteen threads IQ2_XXS 20.7 -> 25.5. Qwen3.8-27B UD-IQ2_XXS
  on the CPU, the same greedy answers: a Ryzen 7 8700GE prompt / decode 2.7 / 2.4 -> 2.8 / 2.8
  tok/s, a Core i5-13500 1.9 / 1.9 -> 2.0 / 2.2.
- **Qwen3.8-27B on Vulkan: its attention on the matrix units.** `chain_attn_flash` takes 64 query
  rows a workgroup of one kv head's heads, so it ran where the heads a kv head divide 64; the 27B's
  six took the blocked shader, and its decode, which follows the prompt's path, read every
  position a workgroup a head. A kv head's heads now split over several workgroups (`fa_group`:
  three pairs for the 27B), each staging its kv head's keys. RTX PRO 4000, served: UD-IQ2_XXS
  prompt 1K / 4K / 8K 404 / 292 / 205 -> 436 / 423 / 425 tok/s, decode 16.6 -> 17.8; UD-Q4_K_XL
  431 / 308 / 203 -> 488 / 474 / 471, decode 14.0 -> 14.8.
- **Qwen3.8-27B on CUDA: its attention on the tensor cores and in blocks of positions.** Its 24
  query heads over 4 kv heads (six a kv head) fell outside the groupings the chain's prompt
  attention took on the tensor cores (dividing 64), so its prompt rows ran on the CUDA cores and
  its decode read every position a block a head. `cc_attn_f2`'s three- and six-warp blocks now
  take six and twelve heads a kv head without a selection list, and the decode in blocks of
  positions takes them too, a kv head's heads a block. RTX PRO 4000, served: UD-Q4_K_XL prompt
  1K / 4K / 8K 556 / 290 / 160 -> 762 / 804 / 810 tok/s, decode after 8K 12.1 -> 19.8; UD-IQ2_XXS
  411 / 244 / 144 -> 494 / 495 / 492, decode after 8K 14.7 -> 28.3
  ([docs/qwen36.md](docs/qwen36.md#from-a-gguf-on-a-cuda-card)).
- **Qwen3.8-27B with its trunk in int4 on the CPU: the prompt 2.4x to 4.5x faster.** The grouped
  kernel (`COLI_DENSE_BITS=4`) summed every group of 64 inputs across the vector for every row and
  every prompt row, more work than the products themselves. The matrices now load interleaved by 16
  rows (`idot.h` `matmul_i4p_gidot_r16`): one integer dot product advances sixteen rows, and a
  group's scale goes into each row's sum with the fused multiply-add the kernel had, so the same
  bits (`tests/test_i4r16.c`; `make i4r16-check` adds the AVX-512 VNNI and AVX-VNNI kernels). Ryzen
  7 8700GE, a 1000-token prompt: 9.5 -> 22.4 tok/s at the release's x86-64-v3, 9.2 -> 41.1 built
  native; decode 3.8 -> 4.0, the same answers. `Q36_ILV16=0` keeps the grouped layout.
- **Qwen3.8-27B from unsloth's GGUFs on CUDA: the chain runs the trunk.** The CUDA placement
  fused a DeltaNet layer's qkv and z rows into one tensor of one type, and these mixes give the
  two different types in about half the layers, so those layers' in_proj stayed on the CPU and
  the dense chain stayed off. Such a layer now holds two tensors in the file's two types
  (`qt_dnproj_z_init_fmt`), and the chain reads them with their own strides. RTX PRO 4000,
  served, prompt 1K / 4K tokens and decode: UD-IQ2_XXS 19.5 / 18.2 and 10.0 -> 408.0 / 242.6
  and 29.5 tok/s, UD-Q4_K_XL 18.9 / 17.4 and 9.6 -> 580.7 / 300.0 and 19.5 tok/s
  ([docs/qwen36.md](docs/qwen36.md#from-a-gguf-on-a-cuda-card)).
- **DeepSeek V4.1 on the Vulkan chain: a prompt of more than one chunk layer by layer.** Each
  chunk went through every layer before the next, so the expert tier streamed each layer's
  cold experts once a chunk: a 4104-token prompt at 512-row chunks copied 150 GiB of the host
  pool's experts and read 98.9 GB from disk. Now each layer runs its chunks in order through
  the attention, the FFN's mix and the shared expert, with every row's streams on the device,
  then its routed experts for all the rows in one step, then each chunk's FFN sum
  (`COLI_VK_V41_LAYERWISE=0`: chunk by chunk). Each row keeps from one layer to the next the
  selection its last index source made and the candidate source's mask, as a chunk keeps them
  through its layers chunk by chunk. unsloth's UD-IQ1_S on an RTX PRO 4000, the 4104-token
  prompt: 34.9 GiB copied, 44.3 GB read, the tier's device time 81.7 -> 33.6 s, the prompt
  39.1 -> 65.1 tok/s.
- **DeepSeek V4.1 and V4 on the Vulkan chain: a prompt chunk's sparse attention in three
  passes, the same bits** (`chain_dsv4_pa.comp`). The attention gave each (head, row) a
  workgroup that read every listed row from memory, a thread a row, so the 64 heads of a row
  (one KV row a position) read each of its 640 rows 64 times. From 8 rows a chunk's rows go
  through the scores of a row's heads on tiles of its listed rows, the weights and
  denominators, and the output, each sum in the old kernel's order (`COLI_VK_DSV4_PA=0`: the
  old kernel). unsloth's V4.1-Flash UD-IQ1_S on an RTX PRO 4000, a 4104-token prompt: the
  attention's device time 31.5 -> 4.4 s; the text the same with it on and off.
- **DeepSeek V4.1 on the Vulkan chain: a GGUF's wo_a as its output groups' GEMMs.** The
  chain multiplied a GGUF's wo_a with the per-head reader, which has no block form: the matrix
  went to the device as floats (8192 x 4096 a layer of V4.1-Flash, 128 MiB, 5 GiB over its 40
  layers) and each row of a prompt chunk read all of it again. Its eight output groups now go
  up as matrices of their own in the file's blocks, as wo_b does: a chunk's rows go to
  group-major order on the device, a GEMM a group, and back; a decode row multiplies its
  groups' slices in place (`COLI_VK_V41_WOA=0`: the reader). unsloth's UD-IQ1_S on an RTX PRO
  4000: the chain's matrices 9.45 -> 5.42 GiB, the room going to resident experts; a 4104-token
  prompt's 44.9 s of device time in wo_a gone, the prompt 27.8 -> 39.3 tok/s; the decode
  5.40-5.42 -> 5.96-6.10 tok/s.
- **GLM-5.3 on CUDA: every layer on the card.** On the CUDA tier the CPU ran GLM-5.3's layers
  and multiplied each dense matrix on the card a call at a time, about 530 synchronous calls a
  decoded token of GLM-5.3-Flash. With the routed experts on the CUDA tier, a prompt chunk's and
  a decode step's layers now run on the CUDA chain (`glm53_cuchain.h`), the Vulkan chain's
  layout on new ops (`cuda_glm53c.cuh`: the KDA convolution and recurrence, the MLA core, RoPE,
  LayerNorm and per-head matmuls, the k-pooled indexer): the KDA state on the card while the
  chain runs, the host's MLA caches mirrored, the routed experts on the host between two frames
  while the card computes the shared expert. unsloth's GLM-5.3-Flash UD-IQ2_XXS on an RTX PRO
  4000: decode 2.93-2.95 -> 3.57-3.65 tok/s on a first turn and 3.85-4.11 -> 4.50-4.68 on a
  second, a 523-token prompt 5.92-6.57 -> 10.92-15.16 tok/s. `COLI_CUDA_CHAIN=0`: the CPU's
  layers.
- **The CUDA expert tier on two cards** (`COLI_VK_DEV2`, as on Vulkan). The CUDA tier kept every
  expert on one card: a machine with two (two V100s, #2608) gave the routed experts one card's
  memory. The second card now holds the experts after the primary's, in its own memory with its
  own block table, scratch, streams and events, and runs the batch of its residents beside the
  primary's at every step; a batch that fails there gives its experts back to the CPU and the
  tier goes on with the primary. `COLI_VK_DEV2=auto` takes the first CUDA device that is not the
  primary; its index takes that one (the primary's own: the same card twice, the test mode).
  Tested with the same card twice against the CPU (glm53 and DeepSeek V4.1's tiny fixtures:
  decode, eviction, a failing batch); not measured on two real cards.
- **GLM-5.2 and GLM-5.3 on CUDA: every layer on the card.** A glm-dsa GGUF under `COLI_CUDA=1`
  put its routed experts on the CUDA tier and left the attention to the CPU, and its dense
  matrices too where half of the card's free memory did not hold them (GLM-5.3 UD-IQ1_S's 13
  GiB): the CPU read every dense matrix every token. Every layer now runs on the CUDA chain
  (`colibri_cuchain.h`), the Vulkan chain's layout on the CUDA chain's ops, with two new ones:
  the DSA selection (`cc_dsa_select`, colibri.c's position for position, up to top-2048) and the
  per-head matmul on a GGUF's Q8_0 rows (`k_b` and `v_b` as the file has them). Every dense
  matrix goes to the card when it fits beside the chain, as Vulkan's chain takes it, and the
  routed experts keep the rest; a CUDA build also runs a GGUF's dense matrices on the card where
  the chain does not (the head among them), as only a build with Vulkan did. unsloth's GLM-5.3
  UD-IQ1_S on an RTX PRO 4000: decode 0.66-0.68 -> 0.77-0.79 tok/s on a first turn and
  0.76 -> 0.88-0.98 on a second, a 528-token prompt 2.23-2.24 -> 4.40-4.51 tok/s (Vulkan's chain
  0.77-0.78, 0.85-0.90, 3.68-4.29). `COLI_CUDA_CHAIN=0`: the CPU's attention.
- **DeepSeek V4.1 with a GGUF on CUDA: a prompt's layers on the card.** On the CUDA tier the CPU
  ran a prompt's layers, the sparse attention, the hyper-connections and the router three fifths
  of a 4104-token prompt. A prompt of more than one chunk now runs layer by layer on the CUDA
  chain (`deepseek_v41_cuchain.h`), op for op the Vulkan chain's: each layer's chunks through the
  attention, the FFN's mix and the shared expert with every row's streams, list and candidate
  mask on the card, then the layer's routed experts in one step of the tier, then each chunk's
  FFN sum. The host's state comes back as the CPU would leave it, for the decode and the next
  prompt. Up to 4096 rows go through the layers at once (`COLI_CUDA_V41_SEG`), and the tier
  leaves the chain that room. unsloth's UD-IQ1_S on an RTX PRO 4000: the 4104-token prompt
  7.42-7.57 -> 91.3-97.0 tok/s, the 523-token one 23.2 -> 58.3-59.8 tok/s, the decode
  5.97-6.03 -> 5.65-5.98 tok/s. `COLI_CUDA_CHAIN=0`: the CPU's layers.
- **The CUDA expert tier streams a prompt's cold experts, and the CUDA dense matrices take a
  prompt 512 rows a call.** The CUDA tier (`cu_xb.cu`) declined the Vulkan tier's streaming, so
  a long prompt's cold experts ran on the CPU 32 rows at a time and came from the disk again for
  each block: a 4104-token prompt of DeepSeek V4.1-Flash UD-IQ1_S read 333 GB of experts. Its
  sub-batches now run on two halves with their own streams, the host pool's experts copied into
  the streaming slots ahead of each, an expert's rows from 16 on the dense matrices' f16 tiles
  (`coli_cuda_bd_gemm_raw`) and the GEMV below (`COLI_CUDA_TIER_STREAM=0`: none). The dense
  matrices (`cu_dense.h`) took a prompt's rows in one call, whose scratch (538 MB for wq_b's
  output at 4104 rows) the card did not have: 51 matrices stayed on the CPU. unsloth's UD-IQ1_S
  on an RTX PRO 4000: 21 GB of experts read, none left to the CPU, the 4104-token prompt
  5.75 -> 7.42-7.57 tok/s, the 523-token prompt 23.2 tok/s, the decode 5.97-6.03 tok/s.
- **GLM-5.3: the routed experts read from the disk with `O_DIRECT`** (`G53_DIRECT`, on by
  default; `0` reads them through the page cache as before). An expert read from the disk is
  wanted once, and the expert tier's host pool leaves the page cache no room to keep it:
  unsloth's GLM-5.3-Flash UD-IQ2_XXS on an RTX PRO 4000 with 62 GB of RAM, two 128-token
  turns refaulted 54-55 GiB, and the compaction kswapd's reclaim kept up migrated the engine's
  pages under its threads, a TLB shootdown on every core each time. The reads now land in the
  slots straight from the drive. The same machine on the CUDA tier, two serves a side: decode
  3.11-3.18 -> 3.54-3.71 tok/s on the first turn and 2.87-3.17 -> 4.40-4.55 on the second, the
  523-token prompt 5.35-5.64 -> 5.89-6.14 tok/s, the host pool's pairs on the CPU 556-584 ->
  439-450 us each, the warm start 23-30 -> 18 s.
- **CUDA chain: a hyper-connection read's up projection and mix in one pass.** Qwen3.8's gated
  residuals read the four streams through a low-rank mix: the up projection's products (four times
  the hidden size a row, 84 MB a 2048-row chunk) went to memory and the mix read them back with the
  normalized streams. `matmul_hcmix` (`coli_cuda_tensor_gemm_hcmix_async`, a new chain op) computes
  the projection on the f16 tiles with a block's weight rows four streams' 32 outputs, and its
  epilogue gives each output the four streams' sigmoids times the normalized streams in the mix's
  order, so the same bits; the products are never written. `COLI_CUDA_HCMIX=0`: the two ops.
  unsloth's UD-Q2_0 on an RTX PRO 4000, two serves a side, a 4180-token prompt 1364-1386 -> 1396-1398
  tok/s, a second one 1446-1455 -> 1475-1476; the answers the same.
- **CUDA chain: a long prompt's QSA attention on the tensor cores.** A prompt chunk's rows with a
  selection list (Qwen3.8's QSA) ran `cc_attn_gk` on the CUDA cores, 15.9 ms of a 2048-row chunk
  a layer on an RTX PRO 4000. `cc_attn_f2` (f16 operands, one pass) now takes them too: each
  token's ascending list of positions gives it a mask a block of keys, a block no token selects
  from is not read, and six warps a block take eight tokens of Qwen3.8's 12 query heads a kv head
  (`COLI_CUDA_ATTN_F2_WARPS=3`: three warps, four tokens). `COLI_CUDA_ATTN_SEL_TC=0` keeps
  `cc_attn_gk`; the rows' bits are not those of `cc_attn_gk` (against the reference within 8e-5 in
  `test_cuda_chain`). unsloth's sets with their start profiles, two serves a side, a 4180-token
  prompt: UD-Q2_0 1231-1235 -> 1357-1364 tok/s, UD-IQ2_XS 1164-1179 -> 1271-1294; the shorter
  prompts and the decode the same.
- **CUDA chain: a long prompt's QSA scores in tiles.** `cc_qsa_score_k` gave each (row, block of
  keys) a thread that read its block's 128 dims alone, 2 ms of a 2048-row chunk a layer.
  `cc_qsa_score_t` takes 64 rows by 64 blocks a block of 256 threads, each thread 4 by 4, the
  index queries and keys staged in shared memory; each score's sums are in the old order, so the
  same bits and the same selections (`COLI_CUDA_QSA_TILES=0`: the old kernel). UD-Q2_0, a 4180-token
  prompt 1357-1361 -> 1387-1392 tok/s, UD-IQ2_XS 1277-1294 -> 1315-1316; the answers the same.
- **CUDA: a prompt chunk's f32 matrices of few outputs a row at a time.** Qwen3.8's gated
  residuals take four outputs of 10,240 inputs (`inject`), which ran on `quant_matmul`'s grid of a
  block an output and a row, each block reading the row again: 105 ms of a 4180-token prompt under
  nsys on an RTX PRO 4000. `f32_few_k` takes a row a block for up to eight outputs, each output's
  sums in `quant_matmul`'s order, so the same bits. unsloth's UD-IQ2_XS, a 4180-token prompt
  1167-1170 -> 1181 tok/s, UD-Q2_0 1215-1221 -> 1219-1229.
- **Qwen3.8 on the CUDA chain: a long prompt's next layer copied while this layer's experts run.**
  A long prompt run layer by layer copies each layer's experts that are off the card to a staging
  on it, and every layer's places started at the staging's start, so the next layer's copy waited
  until this layer's experts had read theirs: on a PCIe 4.0 x8 card the bus then ran alone for 321
  ms of a 4180-token prompt. The places now run on as a ring, each layer's after the last one's, and
  the part of the next layer's copy that falls outside this layer's places starts beside its
  experts (11.2 of UD-Q2_0's 20.8 GB). `COLI_CUDA_CHAIN_STAGE_RING=0`: every layer from the
  staging's start, after the layer before. unsloth's sets on an RTX PRO 4000, two serves a side, a
  4180-token prompt: UD-Q2_0 1176-1179 -> 1222-1225 tok/s, UD-IQ2_XS 1066-1069 -> 1167-1168; the
  answers the same, the shorter prompts the same.
- **Qwen3.8 on the CUDA chain: a long prompt's streams read and written where they wait.** Run layer
  by layer, every chunk copied its rows' streams from the segment's (`hyper_all`) into the chain's
  own and back after each pass, 80 MiB a 2048-row chunk each way: 204 ms of a 4180-token prompt's
  4.1 s under nsys with no kernel beside them. The hyper-connections' reads and writes and the PLE
  now take the chunk's rows of the segment's streams in place. unsloth's sets on an RTX PRO 4000,
  two serves a side, a 4180-token prompt: UD-Q2_0 1132-1137 -> 1178-1180 tok/s, UD-IQ2_XS
  1041-1043 -> 1068-1073; the answers the same, the shorter prompts the same.
- **Qwen3.8 on the CUDA chain: a long prompt's routed experts in one call a layer.** Run layer by
  layer, a prompt's chunks each ran a layer's routed experts on their own, and a chunk's tiles
  decode each expert's weights for its rows: a 4180-token prompt's chunks of 2048, 2048 and 84
  rows decoded every layer's experts three times. The chunks of a segment now route first and then
  run their experts in one call (`cc_moe_cols`, a new chain op), down's outputs a slice at a time so
  that the chunk's scratch holds every row's gate and up, and each slice's sums go out before the
  next; every row has the bits one call of all the outputs gives it. As many chunks as leave a slice
  of 512 outputs (a 4180-token prompt's three, 896 a slice); `COLI_CUDA_CHAIN_MOE_GROUP=0` runs them
  chunk by chunk. unsloth's sets on an RTX PRO 4000, with their start profiles, two serves a side:
  a 4180-token prompt UD-Q2_0 1007-1017 -> 1138-1141 tok/s, UD-IQ2_XS 914-917 -> 1045-1051, UD-IQ1_M
  1041-1050 -> 1155-1161; the 545-token and one-line prompts the same, a decode step's time the
  same.
- **Qwen3.8 on the CUDA chain: a server's first long prompt as fast as the next one.** The first
  long prompt after a start grew the chain's buffers to a chunk's 2048 rows on its way, in
  page-locked host memory at 0.2 ms a MB (the K/V rows down, the streams down, a prompt-end
  snapshot of the state), and a growth frees the buffer it replaces, which waits for the stream;
  the runtime loaded the chain's kernels and the backend's at their first launch (40 and 17 ms).
  They are reserved and loaded before the first request now, at the sizes that prompt left (110-120
  ms of the start), and the recurrent state's pages are touched there too. A prompt read from its
  first token no longer copies the DeltaNet state for a stop to put back (it is the empty state, and
  a stop resets it again), and a prompt's embedding rows go into the streams on the CPU's threads.
  unsloth's UD-Q2_0 on an RTX PRO 4000, three serves a side: a 4180-token prompt as the server's
  first request 954-978 -> 992-1002 tok/s, another as long after it 1012-1025 -> 1025-1033; the
  545-token prompt, the one-line one and the decode the same.
- **Qwen3.8 on the CUDA chain: a short prompt after a long one keeps its two lanes.** A chunk of 384
  rows or more with the host pool on the CPU runs as two lanes, and the second lane's buffers came
  from the rows the chain's reserve counted or, past them, where the card had their bytes and a GiB
  besides. After a long prompt the chain's own buffers hold all 2048 rows of its reserve, and the
  card had 1.1-1.4 GB free, so the second lane was refused, and the prompt, which had chosen the CPU
  for its pooled experts on the lanes' account, ran in one lane with every pooled slot on the CPU.
  The second lane now needs half a GiB besides its bytes (the tier's allowance keeps a GiB of the
  card out of its own), and the choice of the CPU asks whether the lanes fit. unsloth's UD-Q2_0 on
  an RTX PRO 4000, a 545-token prompt after a 4180-token one, two serves a side: 334-359 -> 582-593
  tok/s (UD-IQ2_XS 337-361 -> 371-485, where the CPU's time a slot varied from serve to serve); the
  decode the same.
- **Qwen3.8 on the CUDA chain: a long prompt's layer in two passes.** Run layer by layer, a
  layer's experts that are off the card are copied to the prompt's staging while the layers before
  it run, but the copy could start only once the layer before had read its own, so it ran beside
  the first chunk's attention alone. Each layer now runs every chunk's attention (or DeltaNet)
  first and every chunk's experts after it, the same ops on the same values, so the copy runs beside
  the whole first pass. UD-IQ2_XS on an RTX PRO 4000, two serves a side: a 4220-token prompt
  737.8-744.8 -> 810.4 tok/s, the answer the same, the other prompts and the decode the same.
- **Qwen3.8 on the CUDA chain: each expert at its own layer's size, on the card and in the host
  pool.** unsloth's Dynamic V3.1 GGUFs mix expert types by layer (UD-IQ2_XS: five sizes from 1.26
  to 2.09 MB), and the chain kept every expert in a slot of the largest, on the card and in the
  page-locked pool. Qwen3.8 now asks the tier for slots of each layer's own size, as Qwen3.6
  does, and the chain takes that arena over (`qt_chain_arena_exact`): its promotions draw within
  the layer's size class and carve new slots from the arena's room before they evict anyone. The
  host pool packs each layer's experts at their own size, and a prompt's staging counts its places
  per layer. UD-IQ2_XS on an RTX PRO 4000, two serves a side: 6845 -> 8647 experts on the card,
  the pool 49.8 -> 39.4 GiB, a 4220-token prompt 126 -> 370 tok/s, a 585-token one 190-210 ->
  361-365, the decode after them 53-55 -> 60-61 and 62.7 -> 74.1-74.6 tok/s; Qwen3.8-Flash-Next
  IQ2_XS (three sizes) 10807 -> 11328 experts and its decode the same or faster. With the chain
  only: the tier on its own (`COLI_CUDA_CHAIN=0`) swaps an expert only for one of its size, and its
  decode came 5-12% slower, so it keeps one size there unless `COLI_CUDA_EXACT_SLOTS=1`.
  `COLI_CUDA_EXACT_SLOTS=0` and `COLI_CUDA_CHAIN_POOL_EXACT=0` restore the largest layer's slot.
- **Qwen3.8 on the CUDA chain: a prompt's staging sized by the expert budget.** A long prompt's
  layer copies its experts that are off the card into a staging buffer of 512 MiB, which held 246
  to 411 of a UD-IQ2_XS layer's on a 24 GB card; the rest the kernels read over the bus at a third
  of its rate. Where the tier sizes the expert budget it now gives the staging what the largest
  layer's experts off the card would take, a fifth more, at most a sixteenth of the budget and
  never under the 512 MiB (`qt_reserve_stage`; `COLI_CUDA_CHAIN_PROMPT_STAGE_MB` sets it by hand).
  UD-IQ2_XS on an RTX PRO 4000 with the exact slots, a 4220-token prompt: 318-371 tok/s with 512
  MiB, 719-729 with 800 (826 the rule's), 726-728 with 1100; the decode the same with 800 and 2-4
  tok/s down with 1100.
- **CUDA: a prompt chunk's routed experts in tiles of 128 rows** where the f16 tiles take them and an
  expert has 32 rows or more on average (`COLI_CUDA_MOE_TILE128`: 1 always, 0 never). A tile decodes
  its expert's weights once for its rows, which is most of its time, and 64-row tiles left an expert of
  more rows decoding them again. Qwen3.8's 512 experts on an RTX PRO 4000 (`test_cuda_bd`'s new
  `BENCH_MOE`): 4096 rows 26.5 -> 16.2 ms a call; a 4180-token prompt of IQ2_XS 935 -> 966-977 tok/s,
  of unsloth's UD-Q2_0 927-929 -> 960-962, a 545-token one the same.
- **CUDA: a prompt's IQ2_XXS, IQ2_XS, IQ3_XXS, Q5_0 and Q5_1 routed experts on the mma.sync
  tiles.** A layer whose gate or down type had no decoder in the f16 tiles fell back to hcg_tile's
  for its prompt chunks. unsloth's Dynamic V3.1 Qwen3.8 files put IQ2_XXS or IQ2_XS gates on up to
  36 of their 48 layers (UD-IQ1_S), IQ3_XXS gates on up to 38 (UD-Q3_K_XL) and Q5_0 downs on up to
  33 (UD-IQ4_XS). UD-IQ1_M on an RTX PRO 4000: a 4220-token prompt 756-769 -> 837-839 tok/s.
- **CUDA chain: a QSA row's scores over a grid of the device** (`cc_qsa_score_k`). A query row's
  scores of its blocks of keys took one block of 128 threads, so a decode step's 1-4 rows used as many
  of the card's multiprocessors: 197 us a call at 4400 positions on an RTX PRO 4000 after the radix
  select. A thread a block of keys over a grid of the rows' blocks, the dot in the CPU's order, the
  same scores: 41 us, and the select's 20. Qwen3.8-Flash-Next IQ2_XS, the decode after a 4180-token
  prompt 68.7-68.9 -> 76.1-76.7 tok/s (unsloth's UD-Q2_0 59.4-59.9 -> 64.8-65.6).
- **CUDA chain: a QSA row's selection by a radix select** (`cc_qsa_k`). Picking a query row's
  blocks of positions counted each block's rank against every other on one block of 128 threads, then
  one thread wrote the list: 298 us a call at 4400 positions on an RTX PRO 4000, 7 ms of a 31 ms
  Qwen3.8 decode step, and four times that a doubling of the context. A radix select over the
  scores' bytes and prefix sums give the same list. Qwen3.8-Flash-Next IQ2_XS, the decode after a
  4180-token prompt: 58.3-60.8 -> 66.6-70.3 tok/s (unsloth's UD-IQ2_XS 58.4-59.0 -> 61.4-61.5).
- **CPU: Q2_0, IQ4_NL, Q5_0 and Q5_1 matrices eight rows at a time** (`gq_rows_f8`, ggml_quants.h).
  The kernels that take a GGUF matrix's file rows by eight, each block decoded once for four
  activation rows, had none of the four, and those went row by row. They are the down matrices of
  every unsloth Dynamic V3.1 Qwen3.8 file (Q5_1 UD-Q4_K_XL's, Q5_0 most of UD-IQ4_XS's), where the
  CPU computes the experts the card does not hold. One core of an i7-1355U (AVX-VNNI), 512 x 2048
  rows in cache, one activation row: Q2_0 4.8 -> 7.1 GB/s, IQ4_NL 13.8 -> 23.6, Q5_0 13.9 -> 19.5,
  Q5_1 10.2 -> 18.0. The same bits as the scalar reference and the R16 kernels (`test_rows_f8`, with
  and without VNNI). UD-Q2_0 on an RTX PRO 4000 and an i5-13500: a 545-token prompt 588 -> 640-648
  tok/s.
- **CUDA chain: a prompt's selected keys staged in shared memory** (`cc_attn_gk`). Each thread read
  its own key's elements in turn, so a warp's loads fell 32 lines apart and the kernel was bound by
  L1 (82% of it under Nsight Compute, the multiprocessors at 43%). A tile's selected keys now come in
  32 elements at a time by 16-byte loads, four threads a key, and each position's dot runs over d in
  the order it did: each row the bits of that row alone. Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000,
  a 4180-token prompt: the attention 628 -> 424 ms, the prompt 892-893 -> 920-934 tok/s (unsloth's
  UD-Q2_0 875-883 -> 924-925).
- **Qwen3.8: a long prompt layer by layer with the MTP head too** (CUDA and Vulkan chains). A prompt of
  several chunks runs layer by layer, each layer's host pool experts staged once for all its chunks, but
  not when the caller wanted the rows' final streams back, and the MTP head's prefill always does: with
  the head (on by default where the file has it) every chunk staged every layer again. The layer-by-layer
  path now gives the streams back as the chunks do, and picks the promotions at the prompt's end as the
  chunks did after each one. Qwen3.8-Flash-Next IQ2_XS with its MTP head on an RTX PRO 4000, two serves a
  side: a 4220-token prompt 580-583 -> 798-807 tok/s on CUDA (56 GB -> 19 GB over the bus), 154-173 ->
  305-308 on Vulkan, a 585-token one 115-134 -> 193-196 on Vulkan; the same text, the decode after it
  and in the turns after the same.
- **The CUDA chains' promotions in one call a frame.** An expert promoted to the card went over the
  bus in three copies (Qwen3.6's six), each `cc_host_up` five calls of the runtime (a mark's wait,
  the frame's event, the copy stream's wait on it, the copy, its mark): 20-30 experts after a
  Qwen3.8-Flash-Next verify took 0.3 ms of the host between the verify and the next draft on an
  RTX PRO 4000. The frame's copies now go in one `host_up_n` call (`cudaMemcpyBatchAsync` from CUDA
  12.8, else a copy at a time) with one wait for the frame and one ticket: 303 -> 87 us of runtime
  calls in that gap. `COLI_CUDA_UP_BATCH=0`: a `cudaMemcpyAsync` a copy, still one ticket.
- **qwen36: a retry, an edited answer and a client gone mid-prompt no longer read the whole
  conversation again** (#2589). Three of every four layers are DeltaNet, whose recurrence
  cannot step back, so a request that does not extend the held state started from its first
  token: a retry of a cancelled reply (the state holds the reply's first tokens too), a last
  answer edited or rendered otherwise, a prompt a client left while it was read (the engine
  zeroed everything). qwen36 now keeps the state its last chat prompt read to its end left, as
  qwen38 does (`COLI_PROMPT_SNAPSHOT`, `KV_SLOTS=1`): the retry reads nothing, the edited
  answer reads from the last prompt's end, the stopped read keeps that end for the next request.
- **CUDA decode: a selection list's attention split over its positions, int8 rows four outputs a
  warp.** A decode step's or a verify's rows with a selection list per row (Qwen3.8's QSA) went
  through one block per (head, row) walking every selected position; they now take the blocks
  of positions the plain attention takes (1.56 -> 0.41 ms of a Qwen3.8-Flash-Next verify on an
  RTX PRO 4000). The int8 GEMV of a decode step's or a verify's rows reads each lane's slice of x
  once for four outputs, the inputs split where the outputs are few (the hyper-connection read,
  10240 x 320 at four rows: 15.1 -> 9.0 us alone). `COLI_CUDA_ATTN_DEC_SEL`, `COLI_CUDA_GEMV_W4`,
  `COLI_CUDA_GEMV_SK`.
- **Qwen3.8's CUDA chain: a greedy answer's tokens picked on the card.** At temperature 0 and
  without logprobs a decode step's or a verify's rows only need their argmax: the card picks
  them (`pick_rows`, the lowest place on a tie as before) and two words a row come down, where
  the host copied every row's 248,320 logits (4 MB a verify of four rows) and scanned each again
  between a verify and the next draft. Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000, MTP head on
  the card, a 512-token answer served: 74.6-78.1 -> 78.3-82.4 tok/s. The text is the same.
  `COLI_CUDA_CHAIN_GREEDY_CARD=0`: the logits down.
- **Qwen3.8's promotion brake at 97 % (it was 90).** The CUDA chain stops promoting experts to the card while the last 64 frames found this share of their routes there. With the MTP head on the card, a verify's layer with a missing expert waits for the CPU, and at 90 those waits were what the brake kept. Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000, served 512-token answers, six runs: 96.1-96.7 % of the routes on the card instead of 92-94 %, the wait 5.1 -> 1.9 ms a verify, decode 79.3 -> 82.4 tok/s, the first answer 75.1 -> 79.3, the prompt unchanged. `COLI_CUDA_CHAIN_PROMOTE_HITS=90` restores it.
- **Qwen3.8 served on the CUDA chain: the state's copies off a prompt's first token.** A served
  prompt copied the DeltaNet state from the card before its first layer (so a stop could put it
  back) and again after its last (the prompt-end snapshot), 118 MB each way read layer by layer
  and copied twice: 10 and 25 ms of a 470-token prompt. A prompt the chain takes in one chunk
  needs no copy to come back to (a stop finds it before it starts or not at all), and the
  snapshot is copied down by the stream into mapped memory without waiting, taken from there
  when a later request needs it. Served on an RTX PRO 4000: 452-token prompts 682-702 -> 744-757
  tok/s, 92 tokens 297-324 -> 327-367.
- **Qwen3.8's CUDA chain: a prompt in two lanes, the CPU's pool slots beside the card's work.**
  A prompt chunk with the host pool on the CPU left the card waiting at every layer while the
  CPU computed the layer's pooled slots. A chunk of 384 rows or more now runs as two lanes, its
  two halves a layer apart: the card runs one lane's layer while the CPU computes the other's
  slots. The card's slots of a layer are one call for both lanes, and each layer's pooled
  experts that the first lane routes at least twice go over the bus for the card while the
  second lane runs, so the CPU keeps the experts with few slots. The chain's prompt GEMMs on the
  f16 tiles split their inputs when a call has few blocks, and a prompt's PLE rows are asked of
  the disk together. Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000 with an i5-13500, served:
  452-token prompts 585-615 -> 685-704 tok/s, 1512 tokens 644-647 (staged) -> 959-966, the
  decode the same (`COLI_CUDA_CHAIN_LANES`, `COLI_CUDA_CHAIN_LANE_STAGE`, `COLI_CUDA_HCG_SPLIT`,
  `Q38_PLE_ADVISE`).
- **CUDA: a prompt's IQ2_S and Q2_0 routed experts through the `mma.sync` tiles.** A GGUF's
  routed experts of these types went through the older f16 tiles, which decode the weights into
  shared memory as their words come in: Qwen3.8-Flash-Next IQ2_XS (IQ2_S gate and up, Q2_0
  down) took 4.52 ms a layer for a 470-row prompt chunk on an RTX PRO 4000, 30 GB/s of the
  88 experts it read. They now take the tiles the K-quants have (the loads apart from the
  decoding, a step ahead): 3.43 ms a layer, and the engine's 470-token prompts 751-819 ->
  881-922 tok/s (with the two-lane prompt chunks). IQ4_XS, IQ4_NL and IQ3_S take them too,
  dense and routed: Qwen3.8's trunk matrices of those types, 913-986 tok/s.
- **CUDA GEMVs of a GGUF's blocks, a decode step's and a verify's rows.** IQ4_NL and IQ4_XS take their values by byte permutes instead of table reads (Qwen3.8's IQ4_XS lm_head at two rows 845 -> 713 us on an RTX PRO 4000); a wide row's x is quantized once for all the GEMV's blocks instead of by each (IQ4_XS 2048 x 8192 at two rows 44 -> 28 us, `COLI_CUDA_BD_PREQ`), and from three rows on each block copies its rows of it into shared memory, where the steps read it instead of beside the weights (four rows: IQ4_XS 10240 x 2560 306 -> 323 GB/s, IQ3_S 244 -> 277, Q6_K 216 -> 246; Qwen3.8-Flash-Next's trunk 6.93 -> 6.55 ms a verify, `COLI_CUDA_BD_XSM`). The bits are the same.
- **A decode step's shared expert in one GEMV pass on CUDA.** The gate and up of a SwiGLU of one GGUF type, the rows of a decode step or a verify, now go through one kernel that writes silu(gate) * up, the bits of the two GEMVs and the elementwise product (three kernels into one). Qwen3.6-35B-A3B Q4_K_M on an RTX PRO 4000, 400-token answers, three runs: 104.4-104.8 -> 105.9-106.3 tok/s. Qwen3.8's chain takes it too where its shared expert's gate and up share a type (half of Qwen3.8-Flash-Next IQ2_XS's layers). `COLI_CUDA_GLU=0` turns it off.
- **DeepSeek V4.1: a prompt's engram rows asked of the disk together.** The rows the engram
  cache did not hold came one pread at a time; a block's rows are now asked of the OS first, then
  read. DeepSeek-V4.1-Flash UD-IQ1_S on an RTX PRO 4000, a 444-token served prompt: 28.7-28.8 ->
  32.4-32.7 tok/s (`V41_ENGRAM_ADVISE`).
- **Qwen3.8: a new conversation with a known system prompt reads only its own text.** An agent's conversations begin with the same long system prompt (Hermes: 14,500 tokens of instructions and tools) and part right after it; the live state and the prompt-end snapshot run past that point, so every new conversation read the whole system prompt again (25-30 s on Qwen3.8-Flash-Next IQ2_XS). A prompt whose first message is a system prompt of `COLI_HEAD_PIN_MIN` tokens or more (1024) now leaves a pin at its end, in the pool the client's pins use, and the next conversation with it restores that pin.
- **Qwen3.8's CUDA and Vulkan chains: a short prompt's host pool experts on the CPU.** A prompt chunk copied
  every pooled expert of every layer over the bus, routed or not (18 GB for any prompt up to
  2048 tokens on Qwen3.8-Flash-Next IQ2_XS with an RTX PRO 4000). A prompt of one chunk now has
  the CPU compute its pooled slots, grouped by expert, while the card computes its own, where a
  measure of the bus and of the CPU at startup says it is faster
  (`COLI_CUDA_CHAIN_POOL_CPU_ROWS`, 0 off). 452 tokens 213-215 -> 480-491 tok/s, 1012 tokens
  357-367 -> 500-518, 76 tokens 42.7 -> 206, 1912 tokens 418 -> 433-439; the decode the same (48.5
  against 48.6 tok/s after the 1012-token prompt, the history held fixed). The Vulkan chain the same
  way (`COLI_VK_CHAIN_POOL_CPU_ROWS`): 452 tokens 206-207 -> 471-477, 1012 tokens 308-310 -> 476-482.
  An attention layer's index keys come down in one copy, not one a row (a staged 2612-token prompt
  +3 %).
- **The CUDA chains' selection attention, a kv head's query heads a block** (`COLI_CUDA_ATTN_GROUPED`):
  a prompt chunk's QSA rows read their selected keys and values once for the 12 query heads that share
  them on Qwen3.8, not once a head, with each head's sums in the same order (the same bits). A
  1012-token prompt on Qwen3.8-Flash-Next IQ2_XS 522-528 -> 641-645 tok/s, 452 tokens 493-504 ->
  532-538; the decode untouched.
- **Qwen3.8's CUDA chain: the PLE projections as BF16 rows** (`COLI_CUDA_CHAIN_PLE_BF16`): the
  key and value projections, BF16 in the GGUF, went up as f32 and a prompt chunk took them through
  the three-part TF32 GEMM (140 ms of a 470-token prompt's 1,000). They go up as BF16 rows, half the
  bytes: a prompt chunk's through the f16 tiles, a decode step's through a BF16 GEMV that reads 16
  bytes a lane (every CUDA BF16 row's GEMV takes it). Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000,
  with the grouped attention: 452 tokens 538-540 -> 639-656 tok/s, 1012 tokens 638-648 -> 772-805,
  the greedy answers to four prompts the same. The PLE rows go to the card once before the
  projections: they were read from host memory over the bus again for every output row (a decode
  step's key and value 424 and 408 us); decode 50.3-52.8 -> 52.5-54.5 tok/s.
- **Qwen3.8's CUDA chain: a decode step's pooled experts on threads of their own**
  (`COLI_CUDA_CHAIN_SERVE_THREADS`). The recording thread computed a hybrid layer's pooled slots
  itself, through OpenMP's team (woken a layer at a time), once it had recorded the next layer. The
  pool's server threads wait for the device's routes themselves, take the gate and up rows and then
  the down rows eight at a time and write the rows' sums where the device reads them, the same bits.
  Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000 with an i5-13500, after a 452-token prompt, the
  history held fixed: decode 49.7-50.2 -> 52.3-52.9 tok/s (16 threads, 64-68 us a slot against
  83-89).
- **Qwen3.8's CUDA chain: the MTP head on the card, on by default.** The head's layer runs on the
  card (`Q38_MTP_CHAIN`), its 512 routed experts too, quantized at the start from the head's Q8_0
  file to Q2_0 (0.66 GiB, `COLI_CUDA_CHAIN_MTP_CARD`; in RAM they take the pool's server,
  `COLI_CUDA_CHAIN_MTP_POOL`), and a draft's list of tokens is scored on the card from the
  lm_head's rows there, the pick too (`coli_cuda_tensor_gemv_rows_async`, each row the bits of the
  whole GEMV). A draft then costs 0.6 ms of the card, so where the head's experts reach the card
  the head drafts by default, three drafts a verify cut by their probabilities, over a list of
  64,000 tokens. A draft is only a guess the verify judges: the text is that of plain decoding.
  Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000, served, nothing set, the history held fixed:
  512-token answers 53.3-54.0 -> 71.7-72.3 tok/s, 128-token answers 54.4-55.4 -> 64.4-65.0; the
  452-token prompt 635-661 -> 574-624 (the card holds 494 experts fewer for the head's).
- **The CUDA chains' small copies by kernel, off the copy engine** (`COLI_CUDA_COPY_K`). A copy of up
  to 1 MiB, mapped host memory on either side or none, and a copy of scattered regions now run as a
  kernel on the chain's stream: as `cudaMemcpyAsync` they took the copy engine and queued behind
  the megabytes of the experts the chain promotes on its copy stream, so a draft's frame started
  up to 12.5 ms late. Qwen3.8's chain writes a step's streams through mapped memory, its expert
  table's changed entries from mapped copies of the tables, waits for a decode step's frame alone
  (not for the promotions' copies, which land on their own), and the device's memory pool keeps 64
  MiB of the kernels' scratch between synchronizations (a `cudaMallocAsync` went to the driver
  each time, 300 us). Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000 with the MTP head, 256-token
  answers on the engine: 73.9-76.4 -> 77.4-80.1 tok/s (the cycle of a verify and its drafts 34.9
  -> 31.9 ms before the waits' and the pool's part).
- **Qwen3.8: a speculative step's PLE rows asked of the disk ahead** (`Q38_PLE_ADVISE`): a verify read
  its rows of the PLE table (a GGUF's, on the disk) at its start, 0.75 ms a verify on the main
  thread. Each row's token is known one draft before: its rows are now asked of the OS then
  (`POSIX_FADV_WILLNEED`), and the verify finds them read.
- **Vulkan: DeepSeek V4.1 Flash's and DeepSeek V4's indexer scores in tiles of rows**
  (`chain_dsv4_idx.comp`, `COLI_VK_DSA_TILED`). A prompt chunk's scores went through mode 10's
  column grid, a workgroup a row and eight columns: every row staged its own queries and walked
  its own copy of the keys. A workgroup now holds a tile of 16 rows x 32 compressed columns,
  stages the tile's keys in shared memory once (and a head pair's queries after it, 32 KB in
  all, which is why it is a shader of its own), and every thread keeps two rows' dots in
  registers. The sums are in mode 10's order and the scores are equal bit for bit
  (`tests/test_vk_chain.c` holds the two paths to `memcmp`, the selection comparing raw bits),
  so answers do not change. RX 6900 XT, 32 heads of 128: a 256-row chunk over 8192 unmasked
  46.70 -> 3.56 ms (13.10x), 256 over 65536 unmasked 445.42 -> 31.93 ms (13.95x), 2048 over
  65536 with a candidate mask 1633.09 -> 212.61 ms (7.68x). Below 16 rows, or on a device with
  under 32 KiB of workgroup memory, the
  grid keeps the work. `COLI_VK_DSA_TILED=0` keeps it everywhere ([docs/vulkan.md](docs/vulkan.md),
  [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)).

### Fixed

- **CUDA on Windows: the answer stopped a few tokens in with the card at 100%** (#2696). Windows'
  WDDM driver model, which WSL2 drives the card through too, holds CUDA launches back and sends
  them in batches. A hybrid step (a layer whose experts are partly in RAM) has the card signal the
  host and wait for the host's rows; the host spun on the signal with no CUDA call that would send
  the batch, so the signal never left it. After 60 s the host gave up and synchronised, which sent
  the batch, and the card then spun on the wait for rows that would not come. A hybrid step's
  signal now sends the frame recorded so far at once (a stream query), on Windows without the TCC
  driver and in WSL2 (`COLI_CUDA_FLUSH`, the engine says so at start), in qwen36's and qwen38's
  chains. On Linux nothing changes; forced on there, the tokens and the speed are the same.
- **A new conversation shorter than the last read its whole history at every turn** (#2703).
  With one slot, a request that does not continue the held conversation and fits below it parks
  that conversation (park.h), and the conversation came back when the request ended or first
  thing at the next one. An agent running one task after another sends a new conversation
  shorter than the last: the old one took the state back after every turn, and each of the new
  one's turns read everything again (115 s a turn instead of 8 in the report). The next request
  now decides: a prompt that follows the parked conversation further than the state held now
  brings it back, one whose rows fit the rows kept apart leaves it parked, any other drops it.
  qwen36 and qwen38 came first; olmoe, MiMo, DeepSeek V4.1, Inkling, Kimi K3 and K3.1, GLM-5.2
  (colibri) and DeepSeek V4 decide the same way now, each with the scenario in its tests, and a
  second short request beside a parked conversation resets only the rows kept apart (MiMo's and
  V4.1's windows wider than them stopped on a lost ring position or answered otherwise).
- **Vulkan: a turn stalled in `vkt_give_back` while the uploader waited for room** (#2705). A
  give-back waited for the uploader's queue to empty, and an upload the pool refused for room
  waited for a free only the engine thread makes, the thread waiting in the give-back. An upload
  or promotion refused for room now gives up while a give-back waits: the expert stays where it
  was, nothing is counted failed and the budget stands. The tier test's case: 0.00 s against
  179.7 s and three failed uploads.
- **Vulkan: `COLI_VK_STAGED=0` on a card without Resizable BAR planned with the whole card.**
  The resident data stays in the host-visible window there (246 of 24467 MiB on an RTX PRO
  4000), and nothing past it spills. The dense chain placed 40 of 40 layers and had its first
  upload refused, the expert tier warmed 10240 experts into a budget of 22.6 GiB and held
  none, and the pools' 256 MiB blocks did not fit, so no tensor reached either device, while
  the CPU read the experts the warm start had counted as placed from the disk: Qwen3.6 Q4_K_M
  answered at 12.9 tok/s. The window is now the device's memory for the engines and the pools,
  what does not fit stays on the CPU, and the answer comes at 13.1, the CPU's own speed. Left
  unset, the card stages its uploads as before (90 tok/s).

- **CUDA: a hybrid step's wait for the host has a bound** (#2696). Where the card holds only part
  of the experts, a decode layer's experts split between the card and the CPU, and the card waited
  for the CPU's rows in a loop with no end: rows that never came left the GPU at 100% and the answer
  stopped. The wait now ends after `COLI_CUDA_HOST_WAIT_S` (30 s): the engine says so and the CPU
  computes from there, in qwen36's and qwen38's chains. With the host's word of one step withheld on
  purpose (`COLI_CUDA_HOST_WAIT_FAULT`), Qwen3.6-35B int4 at `--vram 11` and Qwen3.8-Flash-Next
  UD-IQ2_XS at `--vram 8` on an RTX PRO 4000 finished their answers; with the bound out of reach the
  process hung until killed. `tests/cuda_tier_engines.sh` runs a small qwen35moe GGUF's hybrid steps
  and the lost word against the CPU's tokens.

- **A list or a dict written the way Python prints it reaches the client as one.** Asked to
  run a query with the parameters 7, 'x,y', null and true, Flint wrote `[7, 'x,y', null, true]`
  for the declared array, two calls in 24 of a tool-call bench over seven models. The GLM call
  form (GLM, Flint) passed that on as a string, the Qwen form (Qwen3.6, Qwen3.8, LFM3, MiMo)
  dropped such a call, Muse and Kimi K3 kept the text, and the DeepSeek V4 and V4.1 decoders put
  it into `arguments` as written, so the client got arguments that were not JSON. A declared
  `array` or `object` parameter whose text is not JSON is now read as a Python literal, JSON's
  `null`/`true`/`false` accepted too (`_literal_container`), and kept only when it is that
  container of JSON values throughout; DeepSeek's arguments are read the same way when they are
  not JSON. The text is parsed, never run. Tuples, sets, non-string keys, non-finite numbers and
  cut-off text are not read, and `strict_tool_calls` reads JSON only, as before
  (`tests/test_tool_args_literal.py`).

- **DeepSeek V4.1: a long prompt's answer could start with `<｜begin▁of▁sentence｜>`.** Past
  layer 20, the first with compress ratio 1, a prompt of 2478 tokens or more runs the later
  layers on its last 2414 rows (`v41_prefill_tail`). Layers 21 to 23 run no indexer: they read
  the list layer 20 published for every row, whose entries count past the window half of a
  prompt read from position 0, all its rows, while the tail starts past position 0 and reads the
  window ring. `v41_take_tail` moved the streams, their mix and the candidate mask but not that
  list: each of the tail's rows read the list of a row from the start of the prompt, its entries
  off the rows the attention holds, and the last row could turn to NaN at layer 21. The list now
  moves with the tail, its entries onto the ring's rows. Wherever the CPU runs a prompt's
  layers (without a GPU, on the CUDA tier, on Vulkan with the chain off): unsloth's UD-IQ1_S on
  the CUDA tier answered a 4104-token prompt with `<｜begin▁of▁sentence｜>` and the document's
  title in 4 fresh serves of 9, and answers it in 3 of 3 now. The Vulkan chain kept the lists
  right.
- **GLM-5.3 from unsloth's GGUF: the shared indexer layers reuse the full layer's selection, as
  the model has them.** GLM-5.3 computes the DSA indexer's top-k on 21 of its 78 layers (zai-org's
  config.json: `indexer_types`, one full layer every `index_topk_freq`) and the shared layers
  between them reuse its selection. unsloth's files hold a shared layer's indexer weights as a
  copy of the full layer's, and `tools/gguf_import.py` read every layer whose weights are in the
  file as full: past `index_topk` (2048) context tokens each layer selected again from its own
  rows. The import now reads a layer whose indexer is the layer before's byte for byte as shared,
  and the engine finds such copies in a folder imported before, which needs no new import
  (`[DSA] N layers hold the indexer of the layer before as a copy`). The 57 copies are no longer
  loaded either.

- **`chat_template_kwargs` reaches the chat template** (`coli serve`). Many OpenAI-compatible
  clients turn a Qwen model's thinking off with `"chat_template_kwargs": {"enable_thinking":
  false}`, and the gateway ignored the field: a Qwen3.8 request that asked for no thinking this
  way reasoned at xhigh, its prompt 40 tokens longer. `enable_thinking`, `preserve_thinking` and
  `reasoning_effort` there now count as they do at the top of the request, where the request
  does not set them itself, with the same checks; a `chat_template_kwargs` that is not an object
  gets a 400.
- **CUDA: unsloth's Dynamic GGUFs with Q5_0 or Q5_1 experts kept every expert on the CPU.**
  The tier's integer step had neither type, and one such tensor left the whole tier off:
  Qwen3.8-Flash-Next UD-IQ2_XS has one Q5_0 down, UD-Q2_K_XL one, UD-Q3_K_XL 20, UD-IQ4_XS 33,
  UD-Q4_K_XL 43 Q5_1. Both are now on the card as the CPU's integer form computes them.
  UD-IQ2_XS on an RTX PRO 4000: decode 8.3-8.6 -> 51.7-68.2 tok/s, a 585-token prompt
  24.6-25.1 -> 206-325 tok/s.
- **Qwen3.8 on a CPU without FMA: the DeltaNet recurrence 4x faster** (#2606). Its fused
  multiply-adds (`fmaf`) are one instruction where the CPU fuses (FMA3, arm64) and a call into
  libm for every element where it does not (Sandy and Ivy Bridge, the x86-64 baseline), which
  no loop vectorizes and glibc computes in software: a Xeon E5-2690 v1 spent 898 ms of a 1248 ms
  forward there. `coli_fmaf` (`fma_compat.h`) is `fmaf` where the CPU fuses, the same
  instructions, and elsewhere the product in double and the sum rounded to float, which AVX
  vectorizes. Built for Sandy Bridge with glibc's software `fmaf`, Qwen3.8 IQ2_XS: the recurrence
  208 -> 52 ms a forward, the same perplexity (2.2402 nats a token over 258 tokens); the planar
  expert kernel's f32 mode (`QWEN_EXPERT_ACT=f32`) on Qwen3.6 0.59 -> 2.9 tok/s.
- `make install` ships the chat UI and `coli ui` finds it in libexec. The
  Models page resolves the installed support modules and the launcher in bin,
  so it can import GGUFs and start a gateway outside a source checkout.
- `make install` includes `tools/iq3xxs_grid.json`, required by the IQ3/E8
  codec when a conversion uses the published lattice table.
- **Qwen3.8: memory that grew with every answer** (#2592). The expert loader reads a batch of
  missing experts in parallel (on by default) with a team as large as the batch, and libgomp
  ends the threads a smaller team leaves out and starts new ones for a larger. Each read
  buffer was a thread's own (`static __thread`, an expert record each) and stayed behind
  with the thread that ended: on Qwen3.8-Flash-Next IQ2_XS on the CPU, 3.4 GiB over a
  512-token answer at 48 slots a layer, 3.5 GiB at 96, until the system ran out. The batch
  now runs on the default team and the buffers go with their thread; the resident memory
  stops at the expert cache (+3.33 GiB at 48 slots, +6.65 at 96) and the answers are the same.
  With a card (CUDA, RTX PRO 4000) the threads it no longer starts at every batch count too:
  decode 10.8 to 11.3 tok/s before, 14.7 to 15.3 now, the first token 6.0 s before, 4.9 now.
- **`coli tune` on Qwen3-Coder: "engine reported no decode speed"** (#2590). Tune put every
  qwen36-family checkpoint's prompts in one form, ending in an open `<think>`; Qwen3-Coder has
  no thinking mode and ended its turn on the first token there (0 tokens, measured on
  Qwen3-Coder-30B-A3B with CUDA), so the baseline had no speed and the tune stopped. The
  family's prompts are now rendered by the gateway (`openai_server.tune_chat_prompt`), each
  checkpoint with its own template and the default reasoning a chat request gets: the request
  tune times is the one `coli serve` is sent. Qwen3-Coder tunes to the end; Qwen3.6 is timed
  without thinking, as it is served unless asked.
- **Qwen3.8 with a card on an ARM host (DGX Spark, GH200, Jetson): the build.** The CUDA chain
  groups a verify's pooled slots of one expert by `GQ_FUSED_ROWS`, which only the x86 AVX2
  kernels defined, so `qwen38` with `CUDA=1` did not compile without them. Every build has it
  now (the paths without those kernels give a row its own bits as well), and CI compiles the
  CUDA engines' host side without AVX2 (`make cuda-host-syntax`).
- **Vulkan: answers in loops with every expert of Qwen3.6 resident on a Radeon** (#2583). With
  experts in the host pool, a decode step's pool experts are computed by the CPU inside the
  chain's frame (qwen36, qwen38, OLMoE), the frame waiting on an event the host sets once they
  are done. The specification now wants such an event set before the frame is submitted
  (VUID-vkSetEvent-event-09543), and a driver may let the frame through: the step's sum then
  reads the CPU's rows before they are written. A Radeon RX 9060 XT on AMD's own driver fell
  into repetition; an RTX PRO 4000 made to skip the wait did the same ("1 1 1 1", broken words,
  speeds that only loops reach). The chain now probes the wait once at its start (a frame
  writes, waits, writes again; the host checks the second write has not happened 20 ms on) and,
  where the frame does not hold, runs the steps on the timelines instead, which the
  specification allows; a device that holds keeps the events and its speed. AMD's own driver
  leaves the events off without the probe: on that Radeon the steps on events answered in loops
  and `COLI_VK_HYBRID_EVENTS=0` answered soundly, while a frame that waits and still reads the
  host's rows as they were before would pass the probe (`COLI_VK_HYBRID_EVENTS=1` tries them
  there, the probe deciding). Two barriers of
  the same steps were wrong as well (the sum read the device's rows with none after the
  dispatch that wrote them, and the event's stage mask held a stage the specification forbids
  there). CI runs the steps under the validation layer's synchronization check (`sync_gate`)
  and with the wait dropped (`COLI_VK_TEST_EV_NOWAIT=1`: the probe must see it, the tokens stay
  the CPU's), and as AMD's own driver (`COLI_VK_TEST_AMD_OWN=1`: the timelines, the CPU's tokens).
- **The UI's coding agents with the UI on Windows itself** (`coli ui` from a Windows prompt, the
  desktop window). The agents of WSL were never looked for there, and every folder was "not
  found" (its check asked `wslpath`, which Windows has not). They are now found and opened
  through `wsl.exe` in its default distribution, the folder handed to the agent as WSL has it,
  its files written through the distribution's share; where WSL does not reach the colibri
  server on 127.0.0.1 (WSL's network without `networkingMode=mirrored`) the page says so
  before any window. An agent an installer left as an alias (Claude Code's `~/.claude/local`)
  is found too.
- **`tools/convert_fp8_to_int4.py`: the token as an unredirected header.** Its downloads set `HF_TOKEN` as a plain urllib header, which urllib copies into a redirect's request; it now goes as `setup_download.py` sends it, to the hub only. `tests/test_convert_hf_token.py` checks it.
- **The chat page's live expert map lights up on Qwen3.8.** The gateway serves the experts each decode step routed (`/experts?since=N`) from the engine's `HITL` lines, and no engine wrote them, so the map stayed dark while an answer was written and filled in only after the turn. qwen38 now writes one after the tokens of each step, the experts routed since the previous one (the CUDA chain's too). `COLI_HITS_LIVE=0` turns them off.
- The chat page's first-run dialog waits for the models service to finish measuring the
  machine; on a fresh state it showed "- GB di RAM" and no model that fits.

### Docs

- Qwen3.8-Flash-Next's long-context table measured again on 2.1.0: 65,536 tokens 208 / 52.2
  tok/s, 131,072 190 / 51.9, 262,144 151 / 42.9 (prompt / decode).
### Fixed

- **#2612**: on MSYS2 UCRT64 with gcc 16, the static OpenMP library (`libgomp.a`)
  now comes from its own package, so a fresh install failed at the link with
  `cannot find -lgomp`. `make` warns up front and names
  `mingw-w64-ucrt-x86_64-libgomp`; `setup.sh` probes the way the Makefile links
  (`-static`) and names the right package; `coli setup` treats a gcc without
  `libgomp.a` as "build packages missing"; the Windows and quickstart docs and the
  CI/release MSYS2 install lists now include the package.

## [2.1.0] - 2026-10-31

387 pull requests since v2.0.0, 208 of them from 35 contributors. GGUF files run
natively on every engine, on the CPU, on Vulkan and on CUDA. CUDA gets device
chains of its own and a routed-expert tier for every engine. New models:
Mistral-Large-4 (text and pictures), LFM3-VL, Flint and Muse Spark Open, the
published GGUFs of DeepSeek V4.1 Flash and Kimi K3. Qwen3.8 Flash Next runs a
256k context at the speed of 64k. A new chat UI (`coli ui`). An agent's short
requests between two turns no longer make the conversation read again, on every
engine.

### New models

- **#2493**, **#2500**, **#2503**: Mistral-Large-4, the text model from every
  published GGUF and from the FP8 release, on the CPU, the Vulkan chain and CUDA,
  and its pictures (the Pixtral tower, from the GGUF's mmproj or our containers).
- **#2489**, **#2491**: LFM3-VL's text model (from its GGUFs or the converted
  checkpoint) and its pictures.
- **#2292**, **#2297**, **#2310**, **#2316**: Flint, its safetensors and every
  unsloth GGUF, on the CPU, Vulkan and CUDA; its sliding layers keep their K/V in
  a ring.
- **#2293**, **#2324**: Muse Spark Open from the unsloth GGUFs, CPU, Vulkan and
  CUDA.
- **#2143**, **#2269**: DeepSeek-V4.1-Flash-1017 as it is, and the published
  DeepSeek V4.1 GGUFs with their DSpark head.
- **#2274**: the published Kimi K3 GGUFs, on the CPU, the Vulkan chain and the
  tier.
- **#2102**: Qwen-Image gets a single-frame CPU VAE encoder.

### GGUF on every engine

- **#2130**: GGUF models read natively, the CPU kernels interleaved by 16 rows.
- **#2140**, **#2141**: DeepSeek V4, Inkling, GLM-5.3 Flash, GLM-5.2, MiMo,
  DeepSeek V4.1 and OLMoE read GGUFs too.
- **#2182**, **#2185**, **#2195**, **#2228**, **#2237**, **#2238**: a GGUF's
  dense matrices and routed experts on the Vulkan chain, its prompt on the
  device's tiled and grouped GEMMs and on the matrix units; IQ1, IQ2, IQ3 and
  K-quant block decoders.
- **#2252**, **#2254**, **#2255**, **#2491**: a GGUF's block formats on CUDA,
  Qwen3.6's and Qwen3.8's CUDA chains on a GGUF, the prompt GEMM on f16
  `mma.sync`.
- **#2239**, **#2248**: on an NVIDIA card a GGUF runs on Vulkan where CUDA does
  not cover its work (GLM-5.2, DeepSeek V4, Kimi K3, Inkling).
- **#2503**, **#2515**: colibri.c's GGUF experts and dense matrices on CUDA, and
  F16 and BF16 matrices on the GPU as block formats (Qwen3-0.6B F16 on CUDA:
  prompt 137 to 15,921 tok/s).
- **#2198**, **#2280**: IQ2_S and IQ3_S rows in one gather; expert reads without
  readahead (Kimi K3 decode +30 %).

### CUDA

- **#1992**, **#2089**: every layer of Qwen3.6 and of Qwen3.8 as one chain on the
  CUDA device (`COLI_CUDA_CHAIN`).
- **#2209**, **#2231**, **#2232**, **#2233**: the routed experts in the chain's
  frame, a prompt chunk's attention and GEMMs on the tensor cores, the experts
  outside VRAM staged from the RAM cache.
- **#2276**, **#2277**, **#2278**, **#2279**: a routed-expert tier on a CUDA
  device for every engine (`cu_tier`), and the dense matrices on the card
  (`cu_dense.h`) for DeepSeek V4, V4.1, Kimi K3, GLM-5.3 and MiMo.
- **#2257**, **#2332**, **#2377**, **#2395**, **#2416**, **#2417**, **#2423**:
  the card's residents follow the conversation, promoted from the host pool.
- **#2335**: the Vulkan and CUDA chains serve several conversations at once, each
  one's state on the device.
- **#2459**, **#2461**: the decision engines (Laya, GLiNER2.5-Decide) and OLMoE,
  every layer with its experts, on a CUDA card.
- **#2462**, **#2497**, **#2385**, **#2392**, **#2408**, **#2418**, **#2421**,
  **#2453**: programmatic dependent launch, add and norm in one pass, the MTP
  head's draft on the device, one-pass prompt attention on sm_75.

### Vulkan

- **#2093**, **#2206**, **#2430**, **#2432**, **#2490**: a layer's routing and
  routed experts in the dense chain's frame; the host pool's experts on the CPU
  beside the device's, measured against the device's reads and kept where they
  win, inside the frame on events.
- **#2184**, **#2258**, **#2318**: a host pool in system RAM the device reads, its
  experts moving up to the device as a conversation routes them.
- **#2225**, **#2435**, **#2437**, **#2499**, **#2253**: a prompt's GEMMs,
  experts and attention on the matrix units at subgroups of 32, flash attention
  with windows and sinks, vec4 reads (`COLI_VK_CHAIN_GEMM_ROWS` for DeepSeek
  V4's chain).
- **#2440**, **#2441**, **#2442**, **#2447**, **#2454**, **#2484**: decode after
  a long prompt, float16 KV mirrors, polling through a step, wider prompt tiles,
  OLMoE's device KV cache (prompt +51 %).
- **#2046**, **#2328**, **#2404**: the pipelines at the pinned subgroup size, the
  cooperative matrix off by default on AMD's own driver.

### Speculative decoding

- **#2250**, **#2251**, **#2256**: Qwen3.6's MTP head on Vulkan, CUDA and the
  CPU; a verify's rows share the decode GEMV's weight loads and get its bits.
- **#2187**, **#2200**, **#2118**: Qwen3.8's MTP head on the device.
- **#2376**, **#2407**: DSpark's verify depth from its confidence; the gate skips
  drafts that cannot pay.

### Long contexts, conversations and agents

- **#1973**: a conversation longer than the context goes on from a summary.
- **#2113**, **#2114**: a chat's next turn continues from what the engine holds,
  its reasoning included.
- **#2326**, **#2327**: the KV caches in float16 (GLM-5.3 at 131,072 tokens:
  11.9 GB instead of 23.9).
- **#2517**: Qwen3.8 Flash Next at a 256k context at the speed of 64k (prompt
  15.6 to 149 tok/s, decode 9.6 to 45.7 on an RTX PRO 4000).
- **#2517**, **#2521**: an agent's short request between two turns (a title, a
  summary) parks the conversation instead of taking its state, on every engine
  (`COLI_PARK`); Hermes on Qwen3.8 86 s to 4.9 s a turn, Qwen3.6-35B on the CPU
  51.6 s to 3.3 s.
- **#2485**, **#2309**: a served prompt stops on CANCEL while it is read, on every
  engine; a dead engine starts again.
- **#2025**: agents get the context they need.

### The CPU

- **#2042**: SSSE3 integer-dot kernels for x86-64 CPUs without AVX2, bit-identical
  to the scalar reference.
- **#2486**, **#2502**, **#2495**, **#2199**: prompt reading on the CPU (OLMoE
  2.5x, Qwen3.6 +15 %), the hot OpenMP team (decode up to +13 %), one place per
  core, a hybrid CPU's team on its performance cores.
- **#2009**, **#2069**, **#2511**: NEON paths (the int4 expert repack, the sparse
  attention row) and faster DeepSeek V4 CPU decode kernels.

### Interface

- **#2505**: a new chat UI (`coli ui`): reasoning, pictures, web search, the
  expert map, a live monitor, models, coding agents.
- **#2087**, **#2094**, **#2004**, **#2005**, **#2457**: the web chat sends a past
  turn's reasoning back and follows a streaming answer; Brazilian Portuguese;
  `coli chat` reads documents (`/file`, `--file`).
- **#2144**: `residency_sim.py --engine olmoe` replays OLMoE traces with OLMoE's
  own cache traversal.

### Setup and platforms

- **#1981**, **#1990**: an engine on a processor without AVX2 says so; a clear
  stop on an MSYS or Cygwin gcc.
- **#2026**, **#2095**, **#2099**, **#2100**, **#2197**: setup applies the plan
  off CUDA and takes `--ctx`, CUDA on Windows, models split across drives,
  `coli stop`, GGUF import on Windows.
- **#2315**, **#2325**, **#2037**, **#2509**: Windows heap and memory fixes, a
  shard's readers overlapping.
- **#2396**, **#2397**: a container's cgroup limit bounds the RAM.

### Cluster

- **#1962**, **#1963**, **#1964**, **#1967**, **#1969**, **#1970**, **#2120**,
  **#2388**: protocol v2, q8 activations on the wire, DeepSeek V4.1 expert
  workers, bit-exact sums for any worker count.

### Fixes and hardening

- **#2522**: a grammar whose alternates fan out into the same stacks no longer
  holds up the engine (a fixed budget of walker steps a byte).
- **#2524**: `/v1/systemone` answers on GLM-5.2 and Mistral-Large-4 (the header
  colibri.c reads).
- **#2523**: a cold Qwen3.6 engine on the CUDA chain no longer hangs on a first
  prompt of a few tokens.
- **#2457**: the `/v1/systemone` error body, and question ids up to 256
  characters.
- Over a hundred fixes across the API, setup, benchmarks, the MCP server, images,
  audio, JSON schemas and LoRA adapters; 92 pull requests from rudycelekli alone.

## [2.0.0] - 2026-10-06

169 pull requests since v1.12.1, 88 of them from contributors. Every MoE
engine now runs on any GPU a Vulkan driver can see: the routed experts on a
shared device tier, the dense layers on a device chain (all of them, or the
first N that fit), and a second GPU for more experts. Three model families
arrive (MiMo-V2.6, Qwen3-Coder, Qwen-Image-2.1) and one dense checkpoint
(Qwen3.8-27B). Qwen3.8 Flash Next gets int4 experts and speculative decoding
that is on by default. Every text engine serves up to 16 conversations at once.
The release archives carry the GPU backends. Setup is one step, and there is one
decision API (System One).

### Several conversations at once

- **#1955**: every text engine serves several conversations at once (`coli serve
  --kv-slots N`, up to 16), as GLM-5.2 and GLM-5.3 Flash did: qwen36 (Qwen3.6,
  Qwen3-Coder, Qwen3.8-27B), qwen38, OLMoE, Inkling, Kimi K3, MiMo-V2.6,
  DeepSeek V4 and V4.1. Each slot keeps its own conversation's state, and every
  decode step takes the next token of each active request as one batch, so the
  weights and experts a step reads serve all of them. A request gets the tokens
  it gets alone, checked frame for frame on every engine, on the CPU and with
  the expert tier. With more than one slot nothing drafts and the dense chain
  stays off.

### Vulkan on every engine

- **#1830**: every engine's resident matrices run on any Vulkan GPU (fmt 10, 11
  and 12 added), each configuration checked against the engine's own CPU run on
  Lavapipe (`tests/vulkan_engines.sh`).
- **#1834**: tiled GEMMs for prefill in every weight format, and a
  cooperative-matrix GEMM where the device has one.
- **#1840**, **#1843**: a shared routed-expert tier (`vk_tier.c`). It holds a
  cache of experts in device memory, filled from the expert history and adapting
  while you chat, and computes the resident experts of each step as one async
  batch while the CPU computes the rest. First on qwen38 and qwen36, then on
  every MoE engine: GLM-5.2, GLM-5.3 Flash, Inkling, OLMoE, Kimi K3, MiMo-V2.6,
  DeepSeek V4 and V4.1.
- **#1844**, **#1848**, **#1849**, **#1851**, **#1857**: the dense chain
  (`vk_chain.c`), a layer's dense part in one submission, on qwen36, qwen38,
  MiMo-V2.6 (sliding window and sinks on the device), OLMoE, Inkling (and
  OLMoE's PILOT race fixed), GLM-5.2 and GLM-5.3 Flash (shared MLA, DSA, KDA and
  mHC ops), Kimi K3, DeepSeek V4.1 and DeepSeek V4.
- **#1871**: staged device-local uploads for cards without Resizable BAR, where
  the mapped path failed or spilled to system RAM.
- **#1901**: expert streaming for big prompt steps, memory budgets and
  speculative verification on the device.
- **#1909**, **#1913**, **#1916**, **#1918**: a partial dense chain. The first N
  layers that fit go to the device, the CPU runs the rest, and nothing is
  uploaded that the chain will not use, so an 8 GB card with a larger trunk keeps
  the chain and its expert tier instead of losing both. `COLI_VK_CHAIN_LAYERS`
  forces N, `coli plan` predicts it. DeepSeek V4, qwen36 and olmoe, qwen38,
  GLM-5.2 and GLM-5.3 Flash.
- **#1920**: a second GPU for the expert tier (`COLI_VK_DEV2=auto|<index>`) on
  every MoE engine: the experts after the primary device's, each step one batch
  per device, both in flight at once. A failing second device gives its experts
  back to the CPU and the tier goes on with the first.
- **#1942**: the dense chain's layers on two GPUs. With `COLI_VK_DEV2`, the
  layers a partial chain leaves go to the second device instead of the CPU, on
  every chain engine (qwen36, qwen38, OLMoE, MiMo-V2.6, Inkling, GLM-5.2,
  GLM-5.3 Flash, Kimi K3, DeepSeek V4.1 and V4). `COLI_VK_CHAIN_LAYERS2` forces
  how many, `COLI_VK_CHAIN_DEV2=0` keeps them off. A lost second device leaves
  the CPU to run its layers, the recurrent state rebuilt where it was held.
  Checked on Lavapipe opened twice; two real GPUs not measured yet.
- **#1933**: `COLI_VK_KV_COLD=device` (opt-in): with the KV cache split past the
  device's budget, the host's part of the attention runs on the device too, from
  host memory it reads in place, so a step runs in one frame. The prompt gains
  4-6% on a Radeon 780M, decode is slower there (hence off by default); a
  dedicated GPU has not been measured.
- **#1948**: several conversations at once on the dense chain. colibri's
  multiplexed decode (`KV_SLOTS`, one row from each active conversation) runs on
  the device: the batch's matrices as one, each row's attention over its own
  conversation's KV mirror (up to `COLI_VK_CHAIN_MUX`, default 16, beside the
  chain's own) with its own DSA list; two devices and the partial chain too.
  Every frame checked against the CPU's on Lavapipe; not measured with a real
  checkpoint yet.
- **#1934**, **#1935**, **#1936**, **#1937**, **#1938**, **#1939** (@huppiflupp):
  the matrix units for prompt attention and int8/int4 prompt GEMMs on the
  chain, a step's experts as one grouped GEMM (decode: one grouped GEMV per
  phase), qwen36's routing in parallel, and int8 decode GEMVs sized by the
  matrix. On a Radeon 780M with Qwen3.6-35B-A3B the first token of a
  1000-token prompt comes in 9.3 s instead of 12.6, decode 6.1 tok/s instead
  of 5.7, perplexity unchanged.
- **#1954**: on an Intel GPU the dense chain decoded garbage (Qwen3.6 on an Iris
  Xe, the default there). Its pipelines now run at the subgroup size the device
  reports; the device compiles at 8 to 32 lanes, and the chain's GEMVs had run
  at another width than the one they read. On Windows the chain also finds its
  optional shaders (MLA, KDA, mHC, DeepSeek V4, the split KV cache), and the
  split cache frees its host memory with the matching call.
- **#1911**: `COLI_VK_DEV=<index>` picks the device on a machine with two GPUs of
  the same kind; the cooperative GEMM's epilogue stride is aligned to 16 bytes, a
  validation error on some drivers (#1908).
- **#1846** (@crichalchemist): correct expert-batch results under MoltenVK's
  default mode.
- **#1762**, **#1763** (@Kenneth-Javier): the Vulkan loader linked on Windows
  under `-static`, and the Vulkan objects linked into every engine-including
  test rule.

### Models

- **#1813**, **#1819**, **#1820**, **#1827**, **#1835**: MiMo-V2.6 (Flash and
  Pro): the engine, the vision tower, tool calling and Brio mode with logprobs,
  read as released; Pro verified against Xiaomi's reference on the real
  checkpoint, and `coli serve` names the loaded variant.
- **#1828**, **#1829**: Qwen3-Coder-30B-A3B (`qwen3_moe`) on the qwen36 engine.
- **#1777**, **#1779**: dense checkpoints of the Qwen3.6 family on the qwen36
  engine (Qwen3.8-27B), text and images.
- **#1780**: Qwen-Image-2.1, text to image: a C engine, pictures in the
  terminal, the web app and the OpenAI images API.
- **#1951**: Qwen-Image on the GPU and a faster start. The weights load in
  7.5 s instead of 34 (four layers at once, quantized straight from bf16, the
  same int8 bytes). With Vulkan every block of a denoising step runs on the
  device, its attention included, and the VAE decodes there too; a card that
  does not hold the 7.1 GB transformer keeps the blocks that fit and streams
  the others each step. On a Radeon 780M a 512x512 image takes 61 s instead of
  112 and a 1024x1024 step 38 s instead of 67; every stage checked against the
  diffusers reference.
- **#1831**: Qwen3.8 Flash Next's routed experts as int4-g64: 1.4 to 1.56x
  decode.
- **#1832**, **#1917**: speculative decoding with the checkpoint's MTP head,
  lossless, +12-14% on the release; with #1917 it is on by default with prompt
  lookup on every engine that has it (`Q38_MTP=0`, `COLI_LOOKUP=0` turn them
  off), and `coli plan` counts the MTP head's memory.
- **#1814**: Inkling reads the embedding norm under the name transformers 5.18
  writes.

### System One: one decision API

- **#1856**: `POST /v1/systemone`, one decision API, and the Laya decision
  engine; `/v1/brio` is removed.
- **#1868**: GLiNER2.5-Decide as a native decision engine (DeBERTa-v3 and
  GLiNER2's head).
- **#1872**: Clef as a native decision model on qwen36, a joint head over
  Qwen3.8-27B, with a 16-bit dense mode.
- **#1743**: the brio serve gates run in CI against the converted fixture.
- **#1753** (@tarazum): brio normalizes to sum by default and warns when a mean
  mixes option token counts.

### Setup and the CLI

- **#1953**: the release archives carry the GPU backends: every Linux and
  Windows engine built with Vulkan and its shaders, the macOS engines with
  Metal, and a Windows CUDA package (`coli_cuda.dll` for compute 8.0 and newer,
  with colibri, qwen36 and kimi_k3 built to load it). The Vulkan loader is
  opened at run time, so these engines still start on a machine without one,
  on the CPU. `coli setup` uses them without a Vulkan toolchain.

- **#1842**: one-step install and start (`start-here`, `coli setup`), the
  `AI_SETUP.md` guide and an MCP server.
- **#1850**, **#1896**: setup picks the backend that works on the machine: on an
  integrated GPU, Vulkan only where it was measured faster; CUDA only when the
  toolkit builds for the card, Vulkan otherwise, and a failed build falls back.
- **#1903**: an engine already built is brought up to date with `make` before it
  is used (a `git pull` used to keep running the old binary), and `coli logs`
  says where a foreground server prints (#1852).
- **#1919**: a fresh install of Qwen3.6-35B-A3B or Qwen3.8 Flash Next gets a
  starting expert history, so its first run fills the GPU tier from the start:
  on a Radeon 780M Qwen3.6's first run went from about 6 to 11 tok/s.
- **#1817**: in `coli chat`, a line that starts with a picture's path is a
  message, not a command.
- **#1959** (@rudycelekli): `coli setup` counts a server on its port as a running
  colibri only when `/health` answers as colibri does; another program's HTTP 200
  no longer passes for one, and the setup starts its own on the next free port.
- **#1315** (@Avicennasis): resumable, hash-verified qpack installers for
  Hugging Face and static mirrors.
- **#1316** (@Avicennasis): the planner honours cgroup memory limits.
- **#1949**: on Windows every engine has its bare `make <engine>` target (six
  fell to make's built-in rule and linked without CUDA or Vulkan, #1945,
  #1900); `coli setup` prefers an MSYS2 that can build Vulkan over a PATH
  compiler that cannot, and installs libgomp; `COLI_VK_DENSE=0` keeps the dense
  chain off too (on an 8 GB laptop it had taken the expert tier's budget,
  #1900); DeepSeek V4 uses the physical cores on an SMT machine (+10% decode on
  a Threadripper, #1906); `coli plan` leaves a card below an engine's CUDA
  floor out; the server says why its engine stopped (#1941); clearer messages
  for a missing model and a CUDA DLL that does not load.

### The serve contract

- **#1353** (@monotophic): `logprobs`, `top_logprobs` and `echo` on the OpenAI
  endpoints, aligned by raw-stream span.
- **#1794**, **#1767**: Qwen3.6 tool calling, and the template's
  `preserve_thinking` rendered so standard clients get KV prefix reuse.
- **#1802** (@bherald): opt-in strict GLM tool-call parsing.
- **#1765** (@SulimanAbdulrazzaq): GLM-5.3's reasoning stays out of the
  `/v1/messages` answer.
- **#1791**, **#1793** (@Avicennasis): Anthropic image blocks are served and a
  tool result's picture is not dropped; the modalities a model card claims are
  the ones the engine loaded.
- **#1787**, **#1804** (@benmaster82): unsupported OpenAI result shaping is
  refused; the completions keepalive streams as a text chunk.
- **#1782**, **#1783**, **#1785**, **#1786** (@kevin9327): malformed tool calls,
  text parts, tools, messages and image URLs answer 400 instead of 500.
- **#1809** (@Stamina9): non-string brio message text is refused.
- **#1749**, **#1751**, **#1789**, **#1803** (@enitimeago): glm53 reports REUSE
  in detail and logs CANCEL, does not restore a pin over rows another branch
  rewrote, handles cancellation during prompt processing, and continues from the
  cache instead of prefilling again.

### Correctness and security

- **#1930**: hardening of three input paths. The files the engines keep beside a
  model (`.coli_kv`, `.coli_usage.tmp`, `hot_pinned.bin`, `.coli_ckpt/`) are
  opened without following a symlink and only as regular files, so a link
  planted in a downloaded model directory no longer redirects the engine's
  writes. Request grammars are limited to 32767 alternates and symbols per rule
  (past that the walker's index wrapped) and are refused with a message.
  colibri, kimi_k3 and deepseek_v41 check the indexer's and RoPE's head
  dimensions in the config.
- **#1739**, **#1740**, **#1754**, **#1755**, **#1854**, **#1855**
  (@namespaceMarcello): the published advisory fixes carried to glm53, qwen36 and
  deepseek_v41; glm53 refuses f32 tensors shorter than the config reads and
  matrices whose shape differs from it; deepseek_v41 keeps engram lookups inside
  their table; qwen36 gives the same logits at every expert-cache capacity on FMA
  builds and never evicts an expert a MoE run is still reading.
- **#1775**: the DeepSeek V4/V4.1 pre-tokenizer in `tok.h`, so Turkish
  tokenizes as HF does.
- **#1799** (@RayanR000): the DeepSeek V4.1 reference sends top-k ties to the
  lowest index and its indexer never keeps -inf.
- **#1812** (@ZacharyZcR): the shared-expert bindings of DeepSeek V4's fused MoE.
- **#1611** (@Cometbuster4969): OLMoE uses the shared RAM probe, with a
  plausibility floor.
- **#1744**, **#1745** (@kevin9327): a numeric `node_id` no longer kills the
  cluster's topology and health requests, and a manifest that is not a JSON
  object no longer crashes the experiments validator.
- **#1356** (@monotophic): the ablation scoring mode replaced by a checked,
  digest-bound one.
- **#1904**: DeepSeek V4's `PROF` measures the expert wait instead of printing 0
  and folding it into `attention_s` (#1852).
- **#1902**: Windows expert prefetch is asynchronous and its queued work bounded.

### Speed on the CPU

- **#1837**: every engine's per-token projections run per block of rows in
  prefill.
- **#1761** (@mfethe1): a NEON arm for batched `matmul_fp8`, bit-exact, 6 to 10x
  at prefill shapes.
- **#1790**, **#1798** (@RayanR000): the f32 matmul runs eight `fmaf` chains at
  once with the same bits on every FMA build, and one vectorized f32 matmul
  serves quant.h, olmoe, inkling and qwen36 (#442).
- **#1773** (@JackKnifeAI): glm53's dense prefill batched, bit-identical per
  token.
- **#1784** (@kreuzzelg): qwen36's DeltaNet layer on the card
  (`Q36_DN_GPU=1`), expert homes per layer range (`QT_HOME=layer`), the shared
  expert offered on request.
- **#1323**, **#1343** (@Avicennasis): qwen36 reads routed experts from a qpack
  container through bounded Metal slots, and MLX-affine dense weights and norms.
- **#1261** (@Kenneth-Javier): an optional explicit AMD XDNA2 lane for the GLM
  shared expert.

### Build, tests and CI

- **#1758** (@Kenneth-Javier): header prerequisites generated with `-MMD -MP`
  (#1741).
- **#1845**, **#1912**: the sanitizer runs and the bf16 gates hold on AVX-512
  runners, whose CPUs keep bf16 on the CPU.
- **#1914**: the Windows UCRT64 job's limit from 35 to 60 minutes.
- **#1795**, **#1821**, **#1836**, **#1839**: four flaky tests made
  deterministic.

### Docs

- **#1897**, **#1833**: the README rewritten for the release (one-step setup,
  models per machine, GPUs, System One) in five languages, with the site.
- **#1768**, **#1776**: every engine in the environment reference, DeepSeek V4.1
  in its own section, and the weights-from-disk options.
- **#1788** (@jamshaid120): duplicate entries removed from the DeepSeek V4 docs.
- **#1801** (@benmaster82): the web app's image alt count and German image
  labels.
- **#1738**: the 1.12.1 entry's pull request count corrected.

## [1.12.1] — 2026-09-24

96 pull requests since v1.12.0, 80 of them from contributors. Two tokenizers
brought back to the reference, brio on the ninth engine, `coli chat` working
again at the default context on two families, and a placement decision that
is now measured on the card in front of it instead of predicted.

### Tokenizers, measured against the reference

- **#1654**: qwen36 tokenized differently from HF `tokenizers` in two ways.
  An added token right after punctuation was encoded as text (`X.<|im_end|>`
  was 7 tokens instead of 3, every chat turn ending in punctuation paid +4,
  #1653), and a whitespace run followed by a non-space was one piece where
  the regex's `\s+(?!\S)` leaves the last char to the next one, so every
  indented line of code tokenized differently. Measured on the real
  vocabulary: 2,803 lines and blocks of code, Markdown, Chinese and
  Japanese went from 757 identical to 2,803, with 5.4% fewer tokens.
- **#1656**: OLMoE's `tokenizer.json` has no Split, a bare ByteLevel with
  `use_regex`, for which HF runs the original GPT-2 pattern; `tok.h` applied
  cl100k. A GPT-2 family in `tok.h`: 1,560/1,708 identical before, 1,708/1,708
  after. The same measurement on GLM-5.2/5.3/5.3-Flash, DeepSeek V4 and V4.1,
  Inkling and Qwen3.8 came back identical on every case.

### Brio and the serve contract

- **#1662**: `POST /v1/systemone`, the request and the reply of TypeSafe's
  Jev API, served by the brio channel: a client written for it points at
  colibri and changes the base URL. `noul` is a yes/no question, `choice`
  scores the labels with their descriptions in the text, `score` the level
  numbers with the expected value and the legend; `confidence` by their
  documented formula. Any `model` name is accepted on that route. Measured on
  the real Qwen3.6: the three-question example of the docs in 1m46 with the
  state read once.
- **#1655**: the DeepSeek V4 engine speaks the numeric channel (`logprobs=k`,
  `pin=1`, `max_tokens=0`), so `/v1/brio` works on the ninth engine instead
  of answering 500 (#1648). The head that used to keep only its argmax now
  returns the whole row; `ECHO` per prompt position during prefill, the
  prompt-end scores kept with a state snapshot on `pin`, and a logprob tail
  on every `DATA` frame during generation. The tiny fixture pins that the
  best `ECHO` token equals the greedy token from the same prefix, and that
  the pinned predictor equals a cold prefill's.
- **#1659**: qwen36 and qwen38 refused a request when `prompt + max_tokens`
  exceeded the context, and the gateway's default budget for these two
  families is 8192, the whole default context: every request without
  `max_tokens` and every `coli chat` message answered 400 on a two-token
  prompt (#1641). `max_tokens` is now a ceiling, clamped to the room the
  prompt leaves, as GLM and DeepSeek V4 already did; only a prompt that does
  not fit is refused. `docs/api.md` states the rule, `docs/qwen38.md` names
  `Q38_MAXT` as the variable `--ctx` becomes.

### The dense trunk in VRAM, measured before it is placed

- **#1657**: qwen36 offers the rest of its dense trunk to the VRAM placer:
  the DeltaNet out_proj (`dnout`), the attention q/k/v/o (`attnproj`) and
  the shared expert (`shexp`), about 650 MB more of int8 on the 35B beside
  `lmhead` and `dnproj`. Measured on four Tesla M10 by the reporter of
  #1652, every placed component ran slower than the CPU (lm_head 68.8 ms
  against 41.7), so the engine now times one GEMV both ways at startup and
  withdraws the whole automatic placement when the GPU loses, giving the
  VRAM back to the experts: `auto` equals `off` on that box, byte-identical
  output. A hand-written `COLI_PLACE` stands; `COLI_TRUNK_PROBE=0` trusts
  the placer.

### Performance

- **#1664**: qwen36's dense trunk and routed experts multiply with integer
  dot products. The activation is quantized to int8 once per call and the
  weights, int8 rows or int4 planar blocks, meet it with maddubs / vpdpbusd
  instead of a float conversion per weight; the integer kernels move from
  `quant.h` into `idot.h`, shared by every engine. Measured on the 35B, 8
  threads, every expert resident: decode 6.71 to 8.23 tok/s (+22.6%), lm_head
  12.6 to 10.1 ms/token, the expert compute 22.7 to 15.6, for +1.3%
  perplexity on 4 x 512 tokens. Both are the default (`COLI_DENSE_IDOT=0`,
  `QWEN_EXPERT_ACT=f32` restore the f32 kernels). `COLI_DENSE_BITS=4` with
  `COLI_DENSE_INT4=<components>` stores part of the trunk as int4 in blocks
  of 64: opt-in, with the perplexity it costs per component in the docs
  (lm_head alone +2.4%, everything +10%).
- **#1668**: qwen38's dense trunk (553 matrices, 3.6 G weights, 8 GiB of
  BF16 read on every token, more than the ten routed experts) is kept on the
  CPU as int8 rows with the BF16 copy released, and multiplied with the same
  integer kernels; the routed experts' e4m3 blocks are decoded eight at a
  time in registers and multiplied with FMA instead of one table lookup per
  weight. Measured on the released Qwen3.8-Flash-Next-FP8, 8 threads, RAM
  LRU 96 per layer: decode 0.61 to 1.42 tok/s, the trunk 434 to 85 ms/token,
  lm_head 76 to 13, the expert GEMVs 388 to 140, peak RSS 32.2 to 28.5 GB;
  prefill of 512 tokens 495 to 149 s. Perplexity on 4 x 512 tokens +0.5%
  (two chunks lower, two higher); the vector FP8 kernel alone reproduces the
  BF16 run to four decimals. Both are the default (`Q38_TRUNK_CPU_INT8=0`
  keeps the BF16 trunk, `Q38_FP8_KERNEL=scalar` the table kernel).

### Performance, from contributors

- **#1606**: the K1b grouped int4 family gets a multi-row tile and AVX-512
  and AMX arms, and is no longer switched off on AVX-512 builds; exact on
  all eight engines on a 16-core AVX-512 host.
- **#1239**: an SSE4.1 tier for the olmoe and qwen36 int8 GEMV, for hosts
  with SSE4.1 but no AVX2; on the Sandy Bridge of #1652 decode went from
  2.45 to 3.54 tok/s.
- **#1313**: `matmul_fp8` computes four output rows per pass under clang,
  where the contraction makes it bit-exact; GCC keeps the one-row kernel.
- **#1612**: qwen36 gains the GLM engine's `CACHE_ROUTE` lever, with the
  VRAM tier as the first residency level, opt-in.
- **#906**: `DEGRADE_ZERO`, an opt-in policy that zero-fills a missed
  expert slot below a gate-weight threshold instead of blocking on the
  load (#865).
- **#1677**: qwen36 projects a prompt's DeltaNet inputs (qkv and z) on the
  card in blocks of up to 256 rows instead of one row at a time; a 259-row
  prefill makes 2 projection calls instead of 259, with the convolution
  history and the recurrent state checked against the CPU run. The paired
  microbenchmark on an RTX 4070 read 4 to 10x per projection.
- **#1674**: qwen36's attention projections the tier placed in VRAM answer
  a whole prompt batch with one call per matrix; a failed call turns only
  that handle off and the prompt continues on the CPU.
- **#1676**: Kimi K3's streaming CUDA expert keeps the gate, up and SiTU
  intermediates on the device and applies down there (one fused entry
  point, optional in the DLL: an older backend keeps the three-call path).
- **#1673**: the streaming MXFP4 matmul reuses one grow-only device scratch
  per card instead of allocating and freeing weights and scales on every
  call.
- **#1559** (kreuzzelg): `convert_qwen36.py --down-bits 8` writes the mixed
  expert layout, int4 gs64 gate/up and int8 down in one slab (5.7 bits per
  weight against gs64's 4.5); the engine tells it apart by size and reads
  each matrix in its own format on the CPU path, and refuses the VRAM tier
  with a line. It is the knob behind the #1370 numbers: on wikitext-2 the
  int8 down alone recovers a quarter of the gap between gs64 and all-int8,
  the rest sits in gate/up. A measurement tool and a middle step, not the
  answer to the gap.
- **#1730** (mfethe1): DeepSeek V4's FP4 expert kernels, the prefill batch
  and the decode matvec, get a NEON arm; arm64 used to take the scalar
  arm, which is why Apple Silicon prefilled at decode speed (#1696). Bit
  identical to the scalar arm, and the ARM CI job now checks that on every
  change; 17 to 22x on the kernel at the V4 expert shapes on an M-series
  Mac, as measured by the author.
- **#1716** (jtinbergen): qwen36 quantizes its dense weights to int8 while
  loading instead of keeping an f32 copy first, and converts f16/bf16 with
  SIMD. On the 35B the resident set after load goes from 9.2 to 4.8 GB;
  the generated text and the perplexity are identical to before.
- **#1286** (cameron): the grouped int4 GEMV and the fused gate/up GEMV get
  an SSE4.1 arm for CPUs without AVX2 (Ivy Bridge and older). It vectorizes
  across output rows, so each lane runs the scalar row's exact sequence and
  the result is bit-identical; forced-SSE4.1 tests at -O1, -O3 and without
  FP contraction pin it. 2.2 to 2.7x on the isolated kernel on a dual
  E5-2680 v2.
- **#1686** (DebugSultan): qwen38's prefill chunk (`Q38_PREFILL_BATCH_ROWS`)
  and workspace (`Q38_PREFILL_WORKSPACE_MIB`) are runtime knobs, the expert
  load batch is no longer capped at top-k, and the QSA ranking and
  attention run per position in parallel at prefill. Measured on the
  released checkpoint on top of the int8 trunk: output byte-identical at
  every chunk width, no speed change on our 16-core server; the knobs are
  there for hardware where the chunk binds.

### Fixed

- **#1650** (bokiko): a Qwen3.8 pin snapshots the recurrent and PLE state
  but reuses the live attention and indexer rows; after an unrelated prompt
  overwrote those rows, returning to the pin could change brio logprobs
  without a warning. The engine now records the token identity of the live
  rows (`kv_prefix.h`) and refuses a stale pin or prefix restore; image rows
  are tainted. Wire regressions run on the BF16 and FP8 fixtures.
- **#1626**: `SNAP` is the model directory for every non-GLM engine, so
  `coli run` stops handing them a leftover environment (#1600).
- **#1604**: glm53 honours `Mat.resident` in the Vulkan gate and frees the
  Vulkan handle in `mat_release`.
- **#1321**: glm53 sizes its expert cache around the model rather than
  around `MemAvailable`, which the page cache had been inflating.
- **#1588**: qwen36 refuses loudly on a failed encode-buffer realloc instead
  of writing through NULL.
- **#1546**: `coli convert` routes OLMoE to `convert_olmoe_merged.py`.
- **#1630**: olmoe emits the `ROUTE_TRACE` records it announced; the stream
  used to be a zero-byte file.
- **#1610**: `v41_dsml.py` is staged during installation (and the nix flake
  bumped).
- **#1511**: the GPU test suite builds under HIP on gfx1151.
- **#1658**: `test_mem_available` compared two reads of available memory
  with `==` and failed on a busy Windows runner; a quarter of a GB of
  tolerance.
- **#1670**: DeepSeek V4.1 read only its argv cache cap, so `RAM_GB=120`
  on a 128 GB box left the engine at eight expert slots per layer and
  23.8 GB of RSS (#1666). With `--cap` omitted, `coli chat`, `coli serve`
  and `coli web` now size the cache from the resource plan, with `RAM_GB`
  or `--ram` as the budget; an explicit `--cap`, a measured profile and an
  auto-tier plan keep precedence.
- **#1671**: DeepSeek V4.1 treats `max_tokens` as a ceiling like the other
  engines (#1641): a fitting prompt with a large request generates what
  the context leaves, a score-only prompt may fill the context, and a
  prompt one token over it is refused instead of silently truncated.
- **#1675**: resident MXFP4 tensors on CUDA carried O float scales where
  the kernel reads O x ceil(I/32) exponent bytes: short buffers were
  over-read and long ones truncated. One format-aware size for upload,
  refresh, accounting and release.
- **#1678**, **#1679**, **#1680**, **#1682**, **#1683**, **#1684**: the
  Qwen CUDA tier's lifecycle, end to end. Shutdown releases every resident
  expert, projection handle and host table after parked callers resume;
  a failed gate, up or down upload frees what it had already allocated;
  the expert budget charges the three scale buffers at their own sizes
  (two experts used to be admitted where one fit); a failed result
  collection stops inference instead of publishing a partial MoE sum; a
  failed or explicitly disabled tier start unwinds its storage and
  synchronization objects, and a second init cannot overwrite a running
  tier; the CUDA backend validates the whole device list before touching
  state and keeps live contexts on a repeated init. Fault-injected on the
  fake backend, then run together on an RTX 4070 under compute-sanitizer
  with zero errors and zero bytes leaked.
- **#1669**: `test_systemone_api` imports its scoring engine relative to
  its package, so an installed `tests` package no longer breaks discovery.
- **#1697** (kevin9327): the dashboard redesign had dropped the reasoning
  stream: thinking tokens arrived on `delta.reasoning_content` and vanished,
  the bubble stayed empty until the answer and a stop during thinking lost
  the turn. The stream is read again, rendered as its own folding block,
  counted in the rate and the time to first token, and a unit test pins the
  split.
- **#1693** (namespaceMarcello): with `PILOT` on, OLMoE could read the same
  expert twice, once from the prefetcher and once from the forward pass,
  into two slots; a slot being read now keeps a reservation in the index
  (the `colibri.c` pattern) and the second caller waits for the first read
  to publish. Three model-free scenarios pin it.
- **#1695** (namespaceMarcello): the prefill echo state and `serve_echo` sit
  under the same `QWEN36_NO_MAIN` guard, so the segment build no longer
  warns about a function it never gets; the full build is byte-identical.
- **#1724** (GenericRikka): Qwen3.6 decoded `<think>`, `</think>` and the
  tool tags to nothing, because they live only in the tokenizer's
  `added_tokens`; with thinking on, the closing tag never reached the
  gateway and the whole answer came back as `reasoning_content`. The
  non-special added tokens are decoded now; special ones such as
  `<|im_start|>` still decode to nothing.
- **#1734** (tarazum): stopping `coli serve` closes the engine's stdin and
  waits for it to exit on its own before the hard-stop ladder, so the
  engine's teardown runs; qwen36 never saved its `HEAT_FILE` under `coli
  serve` (#1733). On Windows the gateway handles SIGBREAK and the engine
  runs in its own process group.
- **#1726** (kevin9327): Inkling measured no RAM on Windows and sized its
  expert cache to 16 per layer; it uses the shared probe now, which on
  Linux and macOS reads the same numbers as before.
- **#1735**: on GNU Make 3.81, the system make on macOS, `.build-config`
  was never written and every build relinked (#1732).
- **#1731** (bokiko): the DeepSeek V4 CUDA object rebuilds when the nvcc
  command changes, so a new `CUDA_ARCH` no longer links the old object.
- **#1728** (crichalchemist): `make test-c VK=1` built 43 test binaries
  without the Vulkan object; they link it now.
- **#1712** (kevin9327): Kimi K3, Inkling and OLMoE now treat `max_tokens`
  as a ceiling like the other engines; `coli chat`'s default of 16384
  answered 400 on every Kimi and Inkling message against their 8192-token
  window.
- **#1713** (kevin9327): `coli plan`, `doctor` and `--auto-tier` export the
  variable that actually sizes the expert cache on Kimi K3
  (`K3_EXPERT_GB`) and GLM-5.3 (`GLM53_EXPERT_GB`); only `RAM_GB` was
  exported, which neither engine reads as the cache size.
- **#1711** (kevin9327): on Windows, a Kimi K3 `CUDA_DLL` build and a HIP
  host were refused by `--gpu` as CPU-only; the probe reads the backend DLL
  name the host was built with, and the launcher maps `--gpu` onto Kimi's
  `K3_CUDA`.
- **#1710** (kevin9327): a `tools[]` entry whose `function` is not an object
  answered HTTP 500 from the GLM and DeepSeek renderers; it is the 400 that
  `generation_options` already had.
- **#1721** (monotophic): every frame the gateway writes to the engine is
  checked, short writes are completed, and a failed `CANCEL` or `STOP`
  drops the request's pending entry and answers a named 500 instead of a
  silent close.
- **#1714**, **#1719** (benmaster82): the brio options form pins the shared
  state, so per-question requests on one document read it once; the web
  page can stop a scoring run, and duplicate options are removed on both
  clients.
- **#1709** (kevin9327): regenerating a turn with pictures sends them again
  and leaves the composer alone.
- **#1646** (Stamina9): qwen38 says once, on stderr, why the parallel expert
  read path is not taken (disabled, cache smaller than the route, no FP8
  scale bank, repeated expert, converted layout).
- **#1708** (wittchen): every `VK=1` build of glm53 failed to compile on a
  misplaced parenthesis.
- **#1707** (namespaceMarcello): the DeepSeek V4 unit objects rebuild when
  the build flags change, so a CUDA engine build followed by `make test-c`
  no longer links the wrong objects (#1702).
- **#1715** (namespaceMarcello): seven GLM-5.3 harnesses matched the
  unittest glob and counted as zero tests; they are renamed, a skip exits 2,
  the two tiny oracles run in CI, and a discovery test catches the next
  empty module (#1700).
- **#1622**: DeepSeek V4's REAP checkpoints store each expert as six
  per-matrix records; the engine read them through buffered pread and
  counted every one as a direct-I/O fallback (36% of expert reads on the
  150B, #1615). Each segment now goes through the aligned direct window,
  with a regression on a generated per-matrix fixture.
- **#1597**: a replayed tool call whose `arguments` parsed as JSON but was
  not an object (`"[1, 2]"`, `"5"`) answered HTTP 500 from the GLM renderers
  before the engine was asked anything; it renders the call without
  arguments, as every other renderer already did.
- **#1624**: the five gcc 13 warnings left in `make check` are gone, and
  `st_index_load` refuses an index path that would not fit its buffer
  instead of opening a truncated one, with a long-path case in the tests.
- **#1651**: a `pyflakes` pass over the launcher, autotune, the family
  registry and the qwen36 converter: a `measure()` defined twice, a
  `readline` import without a fallback, a stray f-string, dead variables.
- **#1580**: `make qwen36 CUDA_DLL=1` on Windows reached GNU make's implicit
  rule and built a CPU-only binary; a bare `qwen36` alias, a `.build-config`
  prerequisite so a CUDA_DLL change rebuilds, a loader-against-header parity
  test, and the Windows CUDA tier documented.
- **#1556**: `coli plan` on macOS said "no supported GPU detected" on every
  Mac; it now lists the Metal device by name, as identity only, without
  pretending unified memory is a VRAM budget.
- **#1691**: the installed launcher invoked as `/bin/coli` or `/sbin/coli`
  on a merged-/usr system derived `/libexec/colibri` instead of
  `/usr/libexec/colibri`, because `abspath` kept the alias (#1689,
  florin65's patch): `realpath` first. A test runs the launcher through
  such an alias, and another checks that every root module the launcher
  reaches is in the `make install` list, the gap #1610 closed by hand.

### Tools and the gateway

- **#1425**: a general GGUF reader, pure stdlib, and a converter from GGUF
  OLMoE checkpoints to a colibri container, with the numerical evidence in
  its own CI job.
- **#1497**: opt-in prompt-injected tool calling for the families without
  native tool tokens (OLMoE, Qwen3.6), behind `COLI_TOOL_FALLBACK=1`, with a
  two-turn end-to-end test.
- **#1355**, **#1357**: durable per-request results and strict stdout
  classification in the eval harness, and the logprob-gap check gated on
  the engine preamble.
- **#1687**: `GET /metrics` in Prometheus text format, behind the API key:
  four gauges, six outcome counters and four histograms (queue wait, slot
  occupancy, first output, engine call), no request labels, no new
  dependency. The admission scheduler distinguishes completion, failure
  and cancellation, lets a request use a free slot that no earlier waiter
  reserved, and joins the keepalive pump before the slot is released.
- **#1717** (enitimeago): the web chat offers Continue on the last
  assistant message when it stopped at the token limit, by hand or on an
  error, and only when `/health` says the server continues assistant
  turns (#1699).
- **#1402** (enitimeago): a request whose last message is a non-empty
  `assistant` turn continues that turn instead of answering in a new one, on
  `/v1/chat/completions` and `/v1/messages`, for all nine families (Kimi K3
  frames the open turn engine-side); the prompt ends inside the turn as the
  official template renders it without a generation cue. On by default,
  `COLI_CONTINUE_ASSISTANT=0` restores the old behaviour; refused together
  with tools or a turn ending in whitespace, with a 400 that says why. Each
  renderer is pinned against the vendored template (#1401).
- **#1102** (monotophic): checkpoint-faithful FP8 containers that store
  `kv_b_proj` as fmt=8 could load but not decode attention; the absorb path
  now decodes fmt=8 on CPU (bit-exact against the reference) and CUDA
  (within the documented tolerance), and the kv_b sharding refuses by name
  the formats it cannot serve, which also closes two silent misreads of
  fmt=5 and fmt=6.
- **#1395** (rybruscoe): `COLI_EXACT_VERIFY=1` makes the speculative verify
  batch token-exact against sequential decode, at a measured cost on the
  dot itself; off by default, the default path is unchanged.
- **#1720** (monotophic): a request carrying `seed` is accepted and the seed
  ignored, as `docs/api.md` now says, instead of a 400; no determinism is
  implied.
- **#1605**: `ORACLE_STRICT=1` makes a GLM oracle comparison exit non-zero
  when it fails, token-exact by default with `ORACLE_TF_MAX_MISMATCHES` for
  the documented teacher-forcing allowance; references are validated before
  the comparison and non-finite logits cannot pass. Both oracle CI jobs run
  real-process regressions against it.
- **#1705**: `tools/benchmark_baseline.py`, a collection protocol on top of
  the HTTP harness for a repeated three-engine serving baseline: one frozen
  manifest (hardware, model and template identity, per-engine launch
  settings, cache and speculation policy), a rotating plan over a
  concurrency matrix, one collector per engine and round that manages no
  server, and a comparison that keeps failed and missing cells visible and
  distinguishes matched artifacts from deployment comparisons. No results
  are bundled and no ranking is emitted.
- **#1688**: `tools/benchmark_http_serving.py`, a stdlib HTTP streaming
  benchmark over fixed JSONL conversations: closed-loop or paced arrivals
  (periodic or Poisson, seeded), warmup separated from measurement,
  first-output and duration SLOs, latency percentiles and usage-based
  token throughput; a truncated or malformed stream is a failure, not a
  sample.

### Docs

- **#1639**, **#1644**: the README shows brio mode and the dashboard as it
  is: the workspace, the Brain page (the measured expert atlas as a cortex,
  and a region inside it) and the Profiling page, in four languages;
  `docs/api.md` describes the four pages instead of the old console.
- **#1492**, **#1643**: a Japanese README, and its banner at the shipping
  version, which the banner test now checks in every language.
- **#1634**: the multi-disk guide states measured gains and limits instead
  of "twice the bandwidth", with Bash and PowerShell examples.
- **#1617**: connecting the pi coding agent to `coli serve`.
- **#1619**: `expected_bytes` identity versus physical extent for
  int4-rans256-g0 (#1273).
- **#1618** (bherald): `docs/qwen38.md` no longer calls the engine text-only;
  the vision tower and the gateway image path shipped in 1.12.0.
- **#1649**, **#1647** (Suraj2105-1): the musl CI job runs on Alpine 3.24
  and the release pipeline on Node.js 22, ahead of the 3.21 and Node 20
  end of life.
- **#1568** (Yoruxyv): an Indonesian translation of the dashboard.
- **#1681** (XBold): the README and `docs/qwen38.md` no longer say Qwen3.8
  has no GPU backend; the CUDA VRAM expert tier and the int8 trunk in VRAM
  shipped in 1.12.0.

## [1.12.0] — 2026-09-20

81 pull requests since v1.11.0. A new way to ask a model a closed question,
a redesigned dashboard and landing page, and a long run of small failures
that used to answer 500 or die on a locale.

### Brio mode: score a closed set instead of generating

- **#1632**: brio mode, on all nine engines. Hand the engine the options an
  answer is allowed to take and it reports the probability of each one
  instead of writing the answer: the option tokens are read, not sampled,
  so `completion_tokens` is 0 and no reply can fall outside the list. It is
  opt-in per request through two `SUBMIT` keys, `logprobs=k` and `pin=1`;
  without them every frame is byte-identical to before, which is asserted
  per engine rather than claimed. `max_tokens=0` became legal, but only
  together with `logprobs>0`.
- The shared contract is three headers: `decode_batch.h` for the logprob
  tail, `serve_codec.h` for the wire, and `pin_pool.h` for nested state
  snapshots, so a request can keep one photograph of the shared
  instructions and a deeper one of the instructions plus the question. With
  one level the question is re-read once per option; with two, four items
  cost 176 tokens instead of 496.
- `kv_prefix_holds()` is the guard that makes the snapshots safe: a
  snapshot is reused only when the prefix record still holds the same ids,
  because a K/V bank can be dropped between two options.
- Exactness is measured, not asserted: the logprobs read from a snapshot
  match a cold recomputation to 0.00e+00 on all nine tiny fixtures, and on
  the two real checkpoints available (qwen36 at 22 GB and DeepSeek V4.1
  Flash at 476 GB).
- Three clients: `POST /v1/brio` on the gateway, `/brio` in `coli chat`,
  and a page in the dashboard. `docs/brio.md` has the request, the reply,
  the normalisation choice and the cases where the mode does not help.
- The endpoint takes three forms. `options` is one closed question.
  `questions` is many questions on one state, each with its own options,
  with the state photographed once and the snapshots ordered by the server.
  `schema` is an object of field to allowed values: the server writes the
  JSON skeleton and fills it one field at a time, so the object is valid by
  construction and every value is one the caller allowed, each with its own
  probability and entropy. Both were measured before they were exposed: the
  5.7x and the 2.4x above are these two forms.

### The dashboard and the landing page

- **#1633**: the web dashboard is redesigned around a workspace and a
  navigation dock, with a brio page that reads a document once and asks it
  several questions, each with its own option set, and a light and dark
  theme.
- **#1635**: the landing page lists the models as a searchable list instead
  of a card grid, gains a brio section, and gets a light and dark theme
  built on the mechanism proposed in #1551. The shipping version and the
  family names are generated from `c/version.py` and `c/family_registry.py`,
  so the page cannot claim eight families again while nine are running.
  The hero plays a recorded session rather than a mock-up.

### Performance

- **#1477**: one routed-expert kernel, `expert_ffn.h`, shared by the MoE
  engines and used by qwen36 first: int4 kept planar in RAM instead of
  unpacked to int8, and two OpenMP regions per layer instead of three per
  expert. 12.8 to 15.7 tok/s with the resident set down from 29 to 17 GB,
  output identical.
- **#1424**: a CUDA VRAM tier for Qwen3.8-Flash-Next, 43 % on one 8 GB card
  and 60 % on two.
- **#1553** and **#1558**: the previous turn's KV is reused instead of
  re-prefilling the transcript, on qwen36, olmoe and DeepSeek V4.1, with a
  resumed prefill that is exact.
- **#1521** and **#1524**: the mHC mix and the KDA convolution and head
  loops run in parallel, bit-identical, for 3.9 % and 3.0 %.
- **#1517**: the four engines that never sized their OpenMP team now do.
- **#1522**: the DeepSeek V4.1 expert stream can be read from more than one
  drive; **#1455** weights glm53 expert reads by disk and honours
  `COLI_MODEL_MIRROR`; **#1543** fixes the V4 mirror cache on unified memory.
- **#1495**: less overhead in the sparse router's top-K selection.

### Fixed

- Requests that answered `500 engine failed` and now answer properly:
  a non-object `json_schema` (**#1587**), an unpaired surrogate escape
  (**#1589**), a non-object `tool_choice.function` (**#1598**), and a tool
  call whose arguments are not an object.
- **#1595**: the grammar draft walker switched itself off after about sixty
  repetitions. **#1596**: a nested array schema doubled the compiled
  grammar at every level.
- **#1483** and **#1514**: olmoe, inkling and glm53 refuse an over-long
  prompt with `CONTEXT_EXCEEDED` instead of failing late.
- **#1549**: the lazy shard-mapping table in `st.h` was published without
  synchronisation and glm53 reached it from an OpenMP region.
- **#1557**: a truncated multibyte tail made the qwen36 tokenizer read past
  the prompt. **#1530**: an integer overflow check in `kimi_k3.c`.
- **#1631**: `coli serve` reaches the expert-history save when it is stopped
  with SIGTERM, not only on Ctrl-C. **#1628**: the expert grid is resent
  after every turn, so the dashboard stops showing the cold snapshot
  forever.
- **#1608**: `coli doctor` matches core tensor roles by component instead of
  by one family's spelling.
- **#1526**, **#1584**, **#1585**, **#1536**: the planner, the auto-tuner
  and the registry agree about which engine is running, and stop advising
  GLM-only knobs on every family.
- **#1489**: `coli bench` ran the GLM engine on any model. **#1503**: every
  engine is handed its model directory through `SNAP`. **#1486** and
  **#1488**: a directory without `config.json` names no engine, and
  `coli convert` refuses an output directory that already holds one.

### Windows, macOS and locales

- **#1515**, **#1560**, **#1561**, **#1562**, **#1516**: the chat, `coli
  run`, `coli tune` and the datapoint harness no longer die when an engine
  line, a prompt or a dash does not fit the console code page.
- **#1537**, **#1579**, **#1535**: the qwen36 CUDA_DLL build on Windows gets
  its tier, is recognised, and the device probe loads the backend before
  counting devices. **#1563**: the Inkling dense int4 converter finishes.
- **#1599**: MacPorts libomp built every macOS engine single-threaded.
  **#1548**: Metal falls back to `MTLCopyAllDevices` when there is no
  system default device. **#1504**: olmoe measures available RAM on macOS
  and Windows too.

### Diagnostics

- **#1494**, **#1506**, **#1512**: `PROF` reports block time, head time,
  matmul, attention and lm_head, at microsecond resolution, on deepseek_v4,
  olmoe and qwen36. **#1555**: `CONSIST=1` checks prefill against decode.
- **#1482**: configurable generation statistics in `coli chat`, with exact
  token throughput when the server reports it.
- **#1468**, **#1487**, **#1480**: a failure in the MoE step, a missing core
  tensor and duplicate tensor names are diagnosed by name instead of by a
  generic message.

### Docs and licence

- **#1540** and **#1541**: the copyright holder is named and a `NOTICE` file
  is added.
- **#1294**: a reproducible benchmarking protocol; **#1508**: a GPU-backend
  datapoint needs a correctness line to be accepted.
- **#1471** and **#1473**: the expired Discord invite is replaced in all
  four READMEs and on the site.

## [1.11.0] — 2026-09-13

56 pull requests since v1.10.2. A ninth model family, five real bugs closed
across four engines, and the two platforms the C tests never built on now
building them in CI.

### A ninth engine: DeepSeek V4.1 Flash

- **#1453**: DeepSeek V4.1 Flash (552B, 510 GB on disk) runs on a CPU box
  streaming experts from an SSD, with no conversion: the released checkpoint
  is read natively, fp8 dense with 32x32 ue8m0 tiles and fp4 experts whose
  layout is byte-identical to the mxfp4 the Kimi K3 engine already reads.
  Everything in the architecture that is not V4 is in: Engram (two n-gram
  memories of 384M rows, 203 GB, that never enter RAM), the DSA indexer with
  its two-level candidate source, hyper-connections, the compressor, the
  32-layer vision tower with its aligner, DSpark speculative decoding (3
  stages, blocks of 5, verified in one batched forward with a rollback for
  the rejected rows) and tool calling in the checkpoint's own DSML format.
- Held token-exact in CI against a torch-only CPU reference
  (`c/tools/dsv41_ref.py`, written because the vendor's own forward needs
  tilelang GPU kernels), at three cache capacities, on a short and a
  40-token prompt, under all three speculative modes; the vision tower is
  matched to 5e-06. The CI also asserts that the vendor's index-key
  republication policy and the tidy reading disagree, so the shipped default
  cannot be "corrected" by accident (`V41_INDEX_OWNER`).
- Measured on the released checkpoint, cold, caches dropped before every run,
  same prompt and seed: a turn went from 78.7 s to 25.1 s during the work
  (0.305 to 0.957 tok/s) through batched expert reads (`V41_READ_DEPTH`,
  default 8 is the measured knee), attention matrices read once per block of
  positions instead of once per token, and an expert-major MoE. Every step
  is bit-exact against what it replaced. A five-turn chat session runs at
  1.14 to 1.58 tok/s; an `image_url` part goes through the gateway into the
  vision tower on the same engine.
- Two ways of hiding the expert reads behind the matmuls were built,
  measured worse, and removed; both are written up in `docs/deepseek-v41.md`
  with the numbers, so the next attempt starts from them.
- `coli chat`, `coli serve` and `coli web` work; `coli run` is deliberately
  unwired for this family, as for qwen36 and qwen38. Per-turn accounting is
  behind `V41_STATS`, off by default, so the chat stays clean.

### Fixed

- **#1390** (@crichalchemist): the qwen36 GPU tier's `qt_fill_wait` returned
  when the upload queue was empty, but the uploader frees the ring slot at
  dequeue, before the backend copies anything, so the warmstart freed the RAM
  int8 copy of experts still in flight. An in-flight count that the uploader
  decrements only once the slot is resident (#1360).
- **#1434** (@njloof): `fmt=0` is raw f32 and has no scale array, but the
  Metal sizing helper returned a per-row scale size for it, so
  `coli_metal_matmul` wrapped a NULL pointer and segfaulted inside
  `newBufferWithBytes`, or silently produced all-zero output when the size
  happened to be a page multiple. The kernel no longer applies a scale for
  `fmt=0`, and a fail-closed guard sized off the same helper falls back to
  the CPU path for any format that needs scales and got none.
- **#1389** (@bherald): GLM-5.3 teardown leaked the lazily allocated dashboard
  HITS table.
- **#1461**: `coli cluster worker` could never start without an explicit
  `--cap`: the default is `None`, and `str(None)` reached the engine's
  argument check as the literal `"None"` (#1452). The worker now resolves the
  cap the way every other launcher does.
- **#1460**: olmoe's serve-mode `PROF` line published five literal zeros, so
  `/profile` reported `expert_disk_s = 0.0` on turns where the expert reads
  were the workload. The disk phase is measured now; the other four fields
  stay zero because they are unmeasured, not because a guess would look
  better (#1449, half of it).
- **#1445**: olmoe's `--ram` was inert and "no `--cap`" meant a constant eight
  slots per layer regardless of memory; the cap is derived from the RAM budget
  (#1443). On the reporter's box a 2,291-token prompt went from 675 s to 353 s
  with no flags (#1442).
- **#1377**: available RAM is measured on every platform (glm53 read
  `/proc/meminfo` unconditionally, so on Windows and macOS it read 0 and the
  expert budget collapsed to 1 GB, which is how `coli tune` went OOM in
  #1375), and the Windows commit limit is respected.
- **#1381**: the context ceiling is announced, a request that cannot be
  honoured is refused up front, and the number reported is the right one
  (#1376).
- **#1393**: GLM-5.3 replies always start inside the think block (#1278).
- **#1414** (@dajiaohuang): the GLM-5.3 serve loop honours `CANCEL` while the
  turn is still running (#1332).
- **#1423** (@kreuzzelg): the lazy HITS table is built privately and published
  under a lock in qwen36, kimi_k3, glm53 and qwen38; the parallel warmstart
  segfaulted on first touch about one run in twelve (#1422).
- **#1404** (@Petsku01): the qwen36 tier offers int8 experts on the decode
  path (#1391). **#1388** (@crichalchemist): `QT_MAX_ROWS` replaces seven
  literal 32s that had to agree, and the int8 copy is freed by ownership
  rather than by format.
- **#1421** (@bherald): `coli` no longer overwrites an explicit
  `CUDA_EXPERT_GB` when it decides to place the dense trunk on the card.
- **#1244** (@Unknown-Findout): the CUDA expert tier is charged real VRAM, not
  logical bytes (#687). **#1394**: the VRAM prefix is priced per row, not at
  the container's widest (#1351). **#1410**, **#1411**: on a single GPU the
  VRAM prefix is sized from the VRAM budget and the measured headroom, not
  from the RAM pin plan capped by the LRU reserve (#1409, #1405).
- **#1447** (@trigger2k20): `coli mirror verify` accepts a complete mirror
  copied by other means, with no receipt, and fails closed on a short or
  corrupt shard.
- **#1456** (@trigger2k20): the planner reports `memory.unified` from the
  host, not from whether the selected engine has a placement-capable GPU, so
  a CPU-only engine on Apple Silicon no longer reads as non-unified; and
  `coli tune` stops suggesting `DRAFT`/`PIPE`/`PIN`/`NUMA` to glm53, which
  ignores them.
- **#1396** (@dmoraesrs): the web reasoning selector drops "Medium" for
  GLM 5.3, which the engine renders identically to "High".
- **#1428**, **#1435** (@iiEliJas), **#1433** and **#1432** (@texasich),
  **#1440**, **#1459**, **#1458** (@trigger2k20), **#1463**: `setenv` visible
  to `getenv` in-process on Windows, `malloc_trim` guarded to glibc so the
  engines build on musl, a missing bench recipe restored, the per-test
  `_putenv_s` helpers dropped, tests including `compat.h` directly, and
  Clang warnings cleaned up.

### Added

- **#1462** (@trigger2k20): an opt-in Metal path for GLM-5.3-Flash's routed
  experts on Apple Silicon (`COLI_METAL=1`): gate, up, clamped SwiGLU, down
  and the route-weighted scatter on the GPU, CPU path unchanged when Metal
  is off. Validated against the CPU implementation to 2.8e-09 and measured
  on an M4 Max: 0.469 to 0.621 tok/s at 32 tokens, up to 2.15 tok/s with
  the expert cache tuned. Also fixes `coli run --cap N` being silently
  ignored on the glm53 one-shot path.
- **#1457** (@trigger2k20): GLM-5.3-Flash feeds the shared routing telemetry
  and writes `.coli_usage`, so usage-driven placement and partial-mirror
  planning have data for it.
- **#1454** (@yuripourre): the qwen36 GPU tier compiles with `HIP=1`.
- **#1399** (@SebaWag): `TRUNK_RESIDENT_LAYERS=N` streams the dense trunk
  through mmap (#826).
- **#1361**, **#1374** (@kreuzzelg): the qwen36 dense trunk places itself
  (`COLI_PLACE=auto`, priced in bytes saved per byte of VRAM), fp8 streaming
  mode, VRAM accounting at allocator granularity, thread affinity, `COLI_GPU`.
- **#1383**, **#1385**, **#1386**, **#1387**: Brain and Profile tabs (EMAP,
  HITS, PROF) on every engine. **#1382** (@dmoraesrs): a reasoning depth
  selector in the web UI (#1311).
- **#1407**, **#1408**, **#1406**: CI builds the portable Windows CUDA DLL and
  publishes it as an artifact; the Makefile refuses the 32-bit `cl.exe` before
  nvcc runs (#1405).
- **#1436**: CI builds against musl (Alpine) and compiles the environment
  tests on Windows, the two holes that let #1430 and #1420 ship.
- **#1360** (@kreuzzelg): tier invariants on the fake backend, and the
  reservation leak they found. **#1419**, **#1415**, **#1418**
  (@dajiaohuang): the V4 and fp8 test harnesses build on Windows.
- **#1439** (@bherald): the planner tests no longer materialise four 3 GB
  zero-filled payloads. **#1358** (@monotophic): a raw-evidence adapter for
  the scoring corpus. **#1238** (@ZacharyZcR): reproducible performance
  records are validated.

### Docs

- **#1446**: the qwen36 `--ram` section said the engine does not stream at
  all. **#1392**: Azure GLM-5.2 benchmark rows (#1379, #1380, #1384), the rANS
  cross-reference (#1273), and the int8 probe pinned in CI (#1331).
- **#1427** (@ZH1995): README.zh-CN formatting. **#1438** (@iiEliJas): the
  `malloc_trim` comment.

## [1.10.2] — 2026-09-06

Patch release. Three of these fixes answer reports made against 1.10.1 in the
days after it shipped.

### Fixed

- `coli convert` picks the converter from the checkpoint's `config.json`. It ran
  GLM-5.2's converter on everything; on GLM-5.3-Flash that quantized the nested
  embedding and the engine refused the result hours later inside `coli web`.
  An option the target converter does not take is refused, not dropped (#1368,
  #1369). The converter also refuses a checkpoint it cannot serve, with a
  per-family pointer (#1305).
- `coli doctor` no longer reports two missing core tensors on a healthy
  GLM-5.3-Flash: it matches roles, prefix-agnostically, instead of GLM-5.2's
  literal names (#1365, #1366).
- The release archive ships every file `coli` reaches: `iq3_pack.py`, its grid
  data file, and `tools/convert_glm53.py`, which had never been packaged
  (#1359, #1364).
- The RSS guard counts anonymous memory, not reclaimable page cache, so a mapped
  container no longer evicts experts to free memory it was not using (#1350).
- `serve` honors CANCEL while a turn is still running (#1336), and the
  disconnect scenario is built rather than hoped for (#1329).
- Metal: bit-exact fp8-e4m3 decode (#1346). Qwen3.6: tokenizer merges in both
  spellings (#1319). macOS: Homebrew prefixes found when `brew` is off the PATH
  (#1320). DeepSeek V4 on macOS: real CPU and memory in HWINFO (#1308).
- `coli doctor` omits the GPU plan for a CPU-only engine (#1322); a missing core
  tensor explains itself on every fatal path (#1318); GLM-5.3-Flash `--no-think`
  is the template's lowest effort level, not a shape of ours (#1327).
- GLM-5.3-Flash `serve` can use the 16 KV slots the engine has; the registry
  declared 1 (#1283).
- `image_url` local reads: `..` is refused, and with `COLI_IMAGE_ROOT` set a
  path must resolve inside it, so an authenticated client of a non-loopback
  server cannot read arbitrary files through the image API. Error messages no
  longer confirm a path or its permissions (#1354).

### Changed

- The GLM family is named `GLM-5.2/5.3`: the two checkpoints share the base
  model and cannot be told apart from their configuration (#1367).
- DeepSeek V4 Flash REAP-150B (85 GB, 132 of 256 experts) loads with the same
  engine (#1310), is documented, and is announced by its measured geometry
  rather than the official checkpoint's 284B.
- The qwen36 VRAM tier promotes int8 experts instead of reserving for nothing
  (#1334), and the three tier bugs that surfaced with it are fixed: an `is_x`
  overrun with two or more GPUs, a shutdown that could hang, and a
  use-after-free on int8 containers (#1339, #1340, #1341, #1344). Kimi K3
  stops paying for a DSA indexer nothing reads (#1335).
- Opt-in: `COLI_MAP_EXPERTS=1` serves experts through a per-shard file mapping
  (#1325). Off by default; output is byte-identical either way.

### Docs

- Windows DeepSeek V4 users are led to the release launcher (`coli.cmd`)
  instead of a source build (#1291).

### Build and CI

- CI workflows run with `contents: read`, and third-party actions are pinned
  by commit SHA (#1354).
- Makefile lists the headers each engine includes as prerequisites (#1284,
  #1349). Site and READMEs carry the eight families with real RAM figures under
  a contract test (#1302).

## [1.10.1] — 2026-08-31

Packaging repair for the prebuilt archives; no engine changes.

## [1.10.0] — 2026-08-31

### A seventh engine: Qwen3.8-Flash-Next

- Added complete text-only inference for the official Qwen3.8-Flash-Next FP8
  checkpoint: four-stream Gated Residual, Gated DeltaNet, Qwen Sparse Attention,
  pageable hashed n-gram embeddings, top-10 routed MoE, and the shared expert.
- The original 131 safetensors shards run directly. Vision and MTP are not
  loaded or advertised; tools and non-text gateway content are refused.
- Added family/planner/doctor/build/release integration and a pinned upstream
  tiny oracle covering sparse selection, cached decode, LRU eviction, and
  sanitizer runs.
- Kept native FP8 expert payloads and normalized scale banks bounded, added
  cache-sized parallel demand loading, and made prompt MoE execution
  expert-major with bounded shared-expert and causal DeltaNet batching.
- Added exact single-slot hybrid prompt-prefix reuse across QSA, DeltaNet and
  PLE state, with persistent state and bounded workspace reflected explicitly
  in planner RAM accounting.
- Added the boundary-only Qwen3.8 Edge adapter and the seventh-family real
  Edge -> Segment -> Edge oracle gate without loading transformer, vision or
  MTP tensors into the Edge process.

## [1.7.0] — 2026-08-19
71 pull requests since v1.6.2. A sixth model family with its GPU tier, a
rebuilt expert-matmul path, and the CI that would have caught the class of bug
we shipped twice.

### A sixth engine: Qwen3.6-35B-A3B — CPU and GPU
- **#712** (@kreuzzelg) — hybrid Gated Attention + Gated DeltaNet + streaming
  MoE, in `c/qwen36.c`. Pre-converted containers published (int4-gs64
  recommended: cosine to the int8 anchor 0.98777 → 0.99313, KL 0.109 → 0.080
  vs per-row). The engine takes any architecture-identical checkpoint
  unchanged — KAT-Coder v2.5 runs on it with no code path of its own.
- **#713** (@kreuzzelg) — CUDA VRAM expert tier with heat-based placement
  across GPUs: **1.44 → 10.05 tok/s (7.0×) on 2× 8 GB cards, output
  bit-identical to CPU** (`cmp` over the full 200-token generation), measured
  cold with no heat table. A `qt_ready()` gate keeps CPU-only builds from
  allocating the packed int4 buffers they never read: **7.33 GB saved**.

### The expert matmul path, rebuilt (all bit-identical)
- **#1071 / #1075 / #1076 / #1077** — activation quantization hoisted to layer
  level across GLM, Kimi K3 and DeepSeek V4: the same vector was being
  re-quantized ~16× per layer, serially. Removes ~5.2 ms/token of serial time
  and every per-call `malloc` from the hot path.
- **#1079 / #1086** — **K1: plane-nibble int4 layout + unsigned-VNNI dot.**
  Storing element *k* and *k+32* in one byte deletes the unpack, and since
  nibbles are stored unsigned, `dot(v,x) = dot(u,x) − 8·Σx` feeds `vpdpbusd`
  natively. 8 uops per 64 MACs instead of 32: **1.45–2.65× on the IDOT
  kernels**, zero bytes added.
- **#1088** — **K2: 1×4 union tile.** The prefill union hands each expert
  2–16 rows; the weight block's load+mask is now paid once per four rows
  instead of once per row: **2.67–3.10× at S=4** (peak 253 GMAC/s).
- **#1093** — parallel silu and the down-side activation hoist.
- **#1094** — **K1b: grouped planar IDOT for gs64 containers** (`IDOT_GS=1`,
  opt-in): the recommended container format could not reach the integer
  kernels at any batch size before this.
- **#1082** (@outtodata) — `FUSED3=1` opt-in fused AVX2 expert matmul.

### Streaming and I/O
- **#1097** — DeepSeek V4 expert-loader pool default 3 → 9 lanes:
  **1.41× decode** on the real V4-Flash checkpoint (8 interleaved runs on a
  quiet 25 GB box). `V4_LOADER_LANES` still overrides.
- **#1056** (@dcutugno) — DeepGEMM sm120 headers fetched at a pinned commit on
  first build: 2.5× prefill on sm120 with nothing vendored in-tree.
- **#988 / #1054 / #1055** — DeepSeek V4 CUDA tier and dual-SSD mirror.

### Correctness and CI
- **#1083** — **ARM CI job** (`ubuntu-24.04-arm`) plus an integer-kernel
  bit-exactness gate that runs on both ISAs. Every tiny-oracle job ran on x86
  before this, so NEON-divergent paths were invisible — which is how the IDOT
  defaults below shipped. Closes #1081.
- **#1044 / #1080** — IDOT made opt-in in olmoe and inkling: the fast path is
  x86-only and quantizes activations, so the same model produced different
  tokens on x86 and ARM by default.
- **#1109** (@SebaWag) — ARM64 dotprod probed by *compiling* the intrinsic:
  GCC 11 defines `__ARM_FEATURE_DOTPROD` for a base it cannot emit
  `vdotq_s32` for. Fixes #1104.
- **#1111** — `__syncwarp()` after `grouped_s4_wmma`'s store (reported by
  @monotophic with `compute-sanitizer` evidence). Fixes #1099.
- **#1073 / #1074** (@bherald) — Kimi K3 cancels prefill between layers
  instead of holding the engine for a minutes-long prompt; cancelled requests
  no longer count as completed.

### Apple Silicon
- **#790 → #1113** (@RDouglasSharp) — **Metal backend for Kimi K3**: KDA state and
  window buffers aligned, wrap-once buffer cache, CPU-side MLA KV cache. 1.7×/2.4×
  on the compute-bound phases (KDA attention + projections dispatched to the GPU);
  MoE experts stay on the CPU. Kimi K3's first GPU backend.

### More correctness fixes
- **#1098** (@monotophic) — `__syncthreads()` missing from the absorb softmax
  reduction, with a determinism test that reproduces the hazard.
- **#1101** (@monotophic) — allocation and `snprintf` results checked on the
  checkpoint-load path (#798).
- **#1100 / #1108** (@monotophic) — fmt=8/fmt=6 scale-byte accounting in
  `tensor_bytes`/`tensor_free`, and `weights_owned` set before the host-to-device
  copy so a failed upload frees its buffer. Each ships with its own regression
  test; all four of this contributor's CUDA fixes landed in this release.
- **#1122** (@ZacharyZcR) — `USAGE_SAVE=0` honoured in every engine (#1039): the
  history was loaded but written back anyway, which quietly contaminated any A/B
  that shared a usage file between arms.
- **#1121** (@ZacharyZcR) — LRU victim selection now respects a lowered `ecap`
  (#1034): after an RSS-guard reduction the cache kept evicting against the old
  capacity.
- **#1123** (@ZacharyZcR) — the v1.6.2 warning-cleanup patches landed (#1032).
- **#1106** (@monotophic) — duplicate tensor names across indexed shards are now
  refused rather than silently resolved to one of them (untrusted containers).

### Interfaces
- **#829** (@aaristov) — GPU-vs-fallback counters and chat status made visible in
  `coli serve`: the tier's behaviour is now observable instead of inferred.
- **#1095** (@benmaster82) — OLMoE planner geometry adapter, and **#1103**
  (@SebaWag) — Kimi K3, Inkling and DeepSeek V4 adapters with 23 tests: every
  family now has real planner geometry, so `coli plan` stops guessing (#1066).
- **#1096** (@terrizoaguimor) — DeepSeek V4 serve framing on the shared codec,
  completing the codec migration across OLMoE, Kimi K3 and V4.
- **#1063 / #1068** (@terrizoaguimor) — model families are registry-owned:
  `coli`, the gateway, `doctor` and the planner read one descriptor table.
- **#1087 / #1090 / #1096 / #1116** (@terrizoaguimor) — a shared serve framing
  codec, now adopted by **every engine**: OLMoE, Kimi K3, DeepSeek V4 and
  Inkling (whose audio payload rides as an opaque extension). Each migration
  landed behind a byte-exact wire-transcript freeze, so the gateway contract is
  provably unchanged. Byte framing had been duplicated five times, which is how
  Windows binary mode silently disappeared from sibling engines (#748).
- **#1036** (@lineape) — distributed expert workers (LAN, opt-in via
  `CLUSTER_WORKERS`).
- Planner: DeepSeek V4 expert naming now recognized, so `coli plan` and
  `coli doctor` stop counting every routed expert as dense (fixes #1110).

## [1.6.2] — 2026-08-14
Security release: **six privately-reported memory-safety issues fixed**, all reachable
from attacker-controlled input (malicious model file / `config.json`, or the kimi_k3
SERVE stdin). Every fix validates at the trust boundary — no behavioural change on
well-formed models or requests. Advisories: GHSA-gf38-c8fx-ppvv (kimi_k3 SERVE OOB
write), GHSA-2qrj-xjmh-mv74 (json.h OOB read), GHSA-w696-h9p7-6rgc (inkling audio
OOB), GHSA-7654-r78q-vc3r (deepseek_v4 indexer OOB R/W), and two more in the same
class. See the GitHub Security Advisories for details.

## [1.6.1] — 2026-08-13
- `--allowed-host '*'` lets an operator reach a public bind deliberately (#990, #993)
- OLMoE streaming no longer drops the answer into the reasoning channel (#984, #985)
- chat reasoning-channel fixes and pty test hardening (#980)

## [1.6.0] — 2026-08-12
**If you are on v1.5.0, update:** it shipped a performance regression on GLM-5.2
(#856), left up to ~60 GB of RAM unused with a 13-point expert hit-rate loss (#885),
and broke Kimi K3 outright on some machines (#888).
- #869 — the planner priced every expert row at the container's *widest* width;
  mixed-width containers had their cache silently halved
- #914 (bherald) — pin budgets accounted correctly
- prefill batch-union: each distinct expert is read **once** per chunk

## [1.5.0] — 2026-08-05
51 pull requests from 15 contributors.
- **Fifth engine: DeepSeek V4 Flash** (@DrewZt, #165) — MLA + DSA sparse attention,
  43 layers, 256 routed experts + 1 shared, top-6; official checkpoint streams with
  no conversion (fp4 experts, fp8-e4m3 dense with UE8M0 block scales)
- #839 — the rows16 fp4 fast path no longer requires AVX-512: consumer Intel/AMD
  CPUs since Alder Lake get the fast path

## [1.4.0] — 2026-08-01
162 commits, 29 pull requests (25 from contributors).
- **Release archives now contain every engine** — v1.3.0 archives shipped `c/colibri`
  alone while the README promised four families (#720); `inkling` and `kimi_k3` are
  built, packaged and smoke-tested per platform
- Third GPU backend lands

## [1.3.0] — 2026-07-29
Three MoE families on one engine, 744B → 2.8T.
- **Kimi K3** (#676) — 2.8T/104B active: KDA + gated-NoPE-MLA + AttnRes + LatentMoE,
  streams Moonshot's QAT MXFP4 experts straight from the original HF shards
- **Inkling** — a 975B model answers on a 25 GB machine

## [1.2.0] — 2026-07-28
- GB10 / DGX Spark unified-memory OOM fix (#653); AMD/ROCm recognized by `doctor`
  and `resource_plan` (#662, #663)
- AVX2 `matmul_e8` — fmt=6 was 92% of decode on the scalar kernel (#654); native
  SIMD fmt=6 encoder, 15× over numpy
- chat stop-set fix (#633/#381), async packed-int4 parity (#632), stable KV slot
  per conversation (#634, #639)

## [1.1.1] — 2026-07-23

A same-day patch release. **Windows users on v1.1.0 should upgrade**: Microsoft
Defender flags the v1.1.0 Windows binary, and the cause was ours.

### Fixed

- **107 KB of zeros were shipped inside every binary** (#527, #532) — and that is
  what antivirus ML heuristics were reacting to. `static GrDraft g_grd={.max=24};`
  looks harmless, but `GrDraft` is ~107 KB (the grammar's 1024 static rules plus the
  PDA walker) and **any** initializer moves the whole struct out of `.bss` and into
  `.data`, writing 106,848 bytes of near-zero-entropy data into the file — in a
  *writable* section, which is the classic shape of an unpacking buffer for a packed
  payload. Section forensics against v1.0.0 (clean on the same Defender definitions)
  isolated it: identical toolchain, identical PE layout, `.data` 1,840 → 108,752
  bytes. A Windows build with the fix scans clean where v1.1.0 does not. Every
  platform's binary also gets smaller: the Linux engine drops 474,904 → 368,016
  bytes, **-22.5%**.
- **`python3 openai_server.py` was broken on a clean checkout** (#526) — the gateway
  still looked for an engine named `glm` after the #391 rename. It resolves
  `colibri`/`colibri.exe` first now, falling back to `glm` for older trees.
  `coli serve` was unaffected. Spotted by @RDouglasSharp while debugging #488.

### Added

- **Anthropic Messages API** on `/v1/messages` (#343, #525) — clients that only speak
  to Anthropic endpoints, Claude Code above all, now work against colibri with no
  shim: same port, nothing to enable. It is a translation layer over the same
  generation path, so tools, streaming and the KV cache behave exactly as on
  `/v1/chat/completions`. Covers system prompts, `text`/`tool_use`/`tool_result`
  blocks, `input_schema` tools, every `tool_choice` mode, the full named-event SSE
  sequence with protocol `ping` keepalives, `stop_reason` mapping, extended thinking,
  and `x-api-key` auth (`Bearer` still works). `stop_sequences`, `top_k` and non-text
  blocks are refused explicitly rather than silently ignored.
- **`SHA256SUMS.txt` published with every release** (#530) — verify a download is
  exactly what CI built from the tagged source.
- **The Windows engine is uploaded as a CI artifact** (#532) — an antivirus report can
  now be verified on a pull request instead of only after a release is published.

### Changed

- **Docs: "Get started" now starts by getting the program** (#521). The README told
  newcomers to download the 372 GB model *before* it told them how to obtain colibri —
  and for Linux/macOS it never told them at all. New order: get colibri (prebuilt
  archive or build from source) → get the model (372 GB stated up front) → run it.
  The obsolete "rename the engine to `glm.exe`" step is gone; archives have shipped a
  plainly-named `colibri.exe` since #508. Applied in all four languages.

## [1.1.0] — 2026-07-22

A community release. 27 pull requests from more than 20 contributors, 216 commits since
v1.0.0. Most of what follows was found, measured, or fixed by people who do not work on
this project and had nothing to gain from it.

### Added

- **AMD GPU support (HIP/ROCm)** (#339) — single-source `backend_gpu_compat.h` with a WMMA
  dispatch gate, so one codebase builds for CUDA and HIP. Validated on an RX 9070 XT
  (RDNA4, ROCm 7.2): token-exact against CPU on a real fmt=4 gs64 container, with resident
  dense *and* with routed experts in VRAM, plus a fail-injection control proving the GPU
  actually executed the work.
- **Dual-SSD streaming** (`COLI_MODEL_MIRROR`, #421) — read the model from two drives at
  once, roughly doubling streaming bandwidth on a disk-bound host.
- **N-drive shard split** (`COLI_MODEL_DIRS`, #469) — capacity aggregation: run a container
  no single drive can hold, spread across several with no duplication.
- **fmt=5 (int3-g64)** (#168) — 3-bit weights with per-64 group scales: measured 3.3x lower
  outlier-row error than per-row int4 at 25% fewer bytes.
- **fmt=6 (E8/IQ3 lattice)** (#465) — CPU decode kernel and dispatch; index codec tooling (#458).
- **`tools/try_tool_calling.py`** — dependency-free two-turn tool-calling probe that doubles
  as a smoke test.

### Fixed

- **Tool calling in coding clients** (#401), root cause found, in two parts:
  - **#506** — the engine capped prompt encoding at `CTX-2`, and the tokenizer stops dead at
    its limit *without reporting anything*. A prompt longer than the context was therefore
    silently truncated to its first `CTX-2` tokens and answered anyway. With the 4096 default
    that is 4094 — exactly the `prefill 4094` in the field report. The dropped tail was the
    tool instructions and the user's actual turn, so the model emitted a bare `<` and stopped;
    and because clients append to the *end* while truncation keeps the *head*, every retry
    re-sent a byte-identical prompt. Now refused with a 400 `context_length_exceeded`.
  - **#505** — a tool call whose closing `</tool_call>` never arrived was dropped whole,
    because the parser required both tags. Now recovered when unambiguous, on both the
    streamed and non-streamed paths.
  - **#437** — non-EOS role markers were armed as hard stops in serve mode and cut generation
    the instant a tool block started.
- **Grouped-int4 (fmt=4) produced garbage output on CUDA** with `CUDA_DENSE=1` (#298) — the
  dense and attention kernels applied per-group scales as if they were per-row. Hardware-verified.
- **OpenMP tuning re-exec preserved the CPU affinity mask** (#476), jailing every thread onto
  one core when `OMP_PROC_BIND`/`OMP_PLACES` were set: roughly a 20x slowdown.
- **Pilot eviction guard dropped ~100% of speculations** once the cache filled (#497),
  collapsing `PILOT_REAL` to a hint-only path.
- **Silent budget clamp** capped the CUDA expert tier at ~109 experts regardless of
  `CUDA_EXPERT_GB` (#495).
- fmt=4 guard at the per-row-only CUDA entry points (#464/#470); `COLI_CUDA_MTP=1` and
  `COLI_CUDA=0` are now honoured over implicit defaults (#468).

### Security

Threat model: model files come from mirrors that are not trusted.

- **#368** — server hardening, JSON and tokenizer parser hardening, build flags, downloader
  and dependency pinning.
- **#413** — the quant layout is resolved *and* validated against the on-disk byte counts
  (unknown layouts are refused rather than falling through to int2), shape-product overflow
  is rejected, and the olmoe dtype-3 path no longer trusts a crafted `nbytes` (heap overflow).

### Performance — all byte-identical

- **#481** +4.7x on the MLA-absorb score and value-mix reductions
- **#477** +13% decode on AVX-512 (`qt_addrow` / `qt_matvec_rows`)
- **#475** +11.6% with opt-in `XEXP=1` (one OpenMP region per expert block at S=1 full residency)
- **#473** +5.5% int4 IDOT at S=1 on AVX-512 VNNI

### Changed

- `glm.c` is now `colibri.c` plus header modules (#391); `make glm` remains as an alias.
- Serve stage 2 (#192): `response_format`, per-request grammars, grammar-forced drafts.

### Upgrade notes

- **`CTX` still defaults to 4096.** Coding clients send far more than that in a single system
  prompt. Use `CTX=32768`. Before this release an over-long prompt was silently truncated;
  now you get a clear 400 instead.

## [1.0.0] — 2026-07-19

First tagged release. The engine has been running in production since late June
2026; this tag marks the baseline for semantic versioning going forward.

### Highlights

- **GLM-5.2 (744B MoE)** runs on ~25 GB RAM in pure C, streaming experts from disk
- **Three-tier placement**: VRAM (hot) / RAM (warm) / NVMe (cold), with a learning
  cache that pins your workload's hottest experts automatically
- **CUDA backend**: multi-GPU expert tier, dense tensor distribution, batched
  ragged attention, resident pipeline (`COLI_CUDA_PIPE=2`)
- **Metal backend** (Apple Silicon): batched expert SwiGLU + fused decode attention
  on unified memory GPU
- **MTP speculation**: native GLM-5.2 draft heads, grammar-forced drafts, kernel-
  pinned verification (`SPEC_PIN=1`)
- **OpenAI-compatible API**: `coli serve` with SSE streaming, KV slots, bounded
  queue, web dashboard (`coli web`)
- **Web UI**: chat with live metrics, expert cortex brain page, profiling breakdown,
  expert atlas 3-D galaxy
- **Cross-platform**: Linux, macOS, Windows 11 (native MinGW), PowerPC; CI on all three
- **Auto-tune**: `coli plan --auto-tier` classifies the bottleneck and derives
  MTP/PIPE/NUMA/PIN settings with explanations

### Engine

- Token-exact validation against `transformers` oracle (teacher-forcing 32/32)
- Compressed MLA KV cache (576 floats/token, 57× smaller), persisted across
  restarts (`.coli_kv`, zero re-prefill)
- DSA sparse attention (lightning indexer), faithfully implemented
- Router-lookahead prefetch (`PILOT=1`, 71.6% predictive)
- Async expert I/O pool (`PIPE=1`), io_uring batching (`URING=1`)
- NUMA-aware expert placement (`COLI_NUMA=1`, +13–40% on multi-socket)
- AVX2 / AVX-512 / AVX-VNNI / ARM NEON / NEON-i8mm / POWER VSX kernels
- int4 / int8 / int2 / grouped-int4 (fmt=4) quantization formats

### Tools

- `coli convert` — FP8→int4 one-shard-at-a-time converter
- `coli doctor` — read-only setup diagnostics
- `coli plan` — resource planner with auto-tune prescription
- `coli bench` — MMLU / HellaSwag / ARC quality benchmarks
- Expert atlas (`tools/analyze.py --web`) — measured topic affinity for 19,456 experts

### Community

- 30+ hardware datapoints in the benchmark tracker
- Contributions from 20+ authors across engine, docs, tooling, and ports
