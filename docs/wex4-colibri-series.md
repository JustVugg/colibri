# WEX4 expert containers in colibri: the whole series in one pull request

WEX4 expert containers in colibri, with the reader, the WNR3 codec (CPU, fast CPU, CUDA), the I/O
switches, the bench and GLM-5.3. The series brings together the 20 stacked pull requests that were
opened against JustVugg/colibri (#2861-#2897) and tracked in JustVugg/colibri#2860, plus a build
fix and this page; `dev` was merged into it later. The commits of the parts are kept as they were,
so the series can still be reviewed commit by commit. The PR numbers below are those of
JustVugg/colibri.

Those 20 drafts were closed on 2026-11-17, and their discussion stays there for reference.

Tracking issue: JustVugg/colibri#2860. Its "Tests wanted" section is the list of tests still
missing.

## What WEX4 and WNR3 are

- **WEX4** is a container for a MoE model's routed experts: one file, one record per (layer,
  expert), each record on a 4096-byte page and already in the layout the expert kernels use. A miss
  is one aligned read of one record instead of six tensor reads across the safetensors shards. The
  first page holds the SHA-256 of the index, and the index holds the SHA-256 of every record. The
  hash of the first page (the *root*) names the whole file, so an engine can refuse a file that is
  not the one it was told to use, and can check each record the first time it reads it.
- **WNR3** is a lossless codec for an MXFP4 expert record: rANS with a small trained model, and a
  23-bit check on every stream. A WEX4 record is either RAW or WNR3.

The specifications, the conformance vectors and the measurements are in Wex-WRN2 (WEX4-SPEC,
WNR3-SPEC, BENCHMARKS).

What they give, as measured so far (each with its conditions):

- In a downstream fork of colibri (not on `dev`), DeepSeek V4.1 Flash REAP-192 MXFP4 on Windows 11,
  one desktop:
  - A RAW container against the safetensors, CPU-only engine, 16 GB RAM, the same NVMe drive: decode
    1.11-1.14 against 0.98-1.01 tok/s (about +13%). TTFT for 30 tokens: 11-12 s against 13-14 s.
  - WNR3 makes the 7,680 expert records 0.854 of their RAW size (123.3 GB against 144.4 GB),
    losslessly. Decoded on the GPU with the container on two drives, one 1,886-token turn gave
    2.92-2.96 tok/s against 2.86 for the earlier WNR2 codec, with the same text. On one drive alone
    WNR3 was slower (2.15 tok/s).
  - A byte copy on a second drive, with reads split by a hash of (layer, expert): +49% decode (2.02
    to 2.99 tok/s, one pair, RTX 5090, 7 prompts x 64 tokens), bit-identical logits.
- On the former PR branches of the series, DeepSeek V4.1 with the published WNR3 container on an
  i7-7700K with an RTX 3070:
  the same tokens as the shards, at the shards' speed with the fast CPU decoder, and about twice it
  with the CUDA helper (see [End-to-end results](#end-to-end-results)).
- Per record (16 records of the published V4.1 container, 18,800,640 decoded bytes each):
  - The reference CPU decoder takes 741.9 ms on one i7-7700K thread, and the fast decoder takes
    101.2 ms (7.3x).
  - 8 records at once: 125.5 -> 23.9 ms a record.
  - The CUDA helper on an RTX 3070 takes 5.9-9.4 ms one record a call, and 4.0-4.4 ms in a host
    batch of 8.

## The parts, in commit order

Short SHAs are those of the series' commits; the former PR numbers in the last column stay valid
whatever form the merge takes.

| # | Part | What it adds | Variables (default) | Commits | Former PR |
|---|---|---|---|---|---|
| 1 | WEX format headers | `wex_format.h`, `wex_neural_rans.h`, `wex_palette.h` and their test; nothing includes them yet | none | `4e8eae24` | #2861 |
| 2 | WEX4 reader | `wex4.h` (header page, index, root, per-record SHA-256) and `tools/wex4_verify` | none | `fac22c0d` | #2862 |
| 3 | DeepSeek V4 from WEX4 | routed experts from RAW records, with the root pin and first-read hash; the shards stay the fallback | `COLI_WEX_PATH` (unset), `COLI_WEX_ROOT` (unset), `COLI_WEX_VERIFY` (on with a root, else off) | `78bf3df2` | #2863 |
| 4 | Qwen3.8 from WEX4 | MXFP4 experts from RAW records, in the record layout of #2825; implies `Q38_EXPERT_LOWBIT=mxfp4` | the same three | `93c3d624` | #2864 |
| 5 | DeepSeek V4.1 from WEX4 | routed experts from RAW records; the shards stay the fallback | the same three | `b6755724` | #2865 |
| 6 | WNR3 on the CPU | `wex_decode.c`/`.h`: the WNM2 model registry, the WNR2/WNR3 record decode (the reference), `coli_wex_container_refused` | none | `ef92de8a` | #2876 |
| 7 | DeepSeek V4: WNR3 | WNR3 records decoded on the CPU by the reading thread | no new one | `13930baf` | #2877 |
| 8 | Qwen3.8: WNR3 | the same for qwen38 | no new one | `345653c7` | #2878 |
| 9 | DeepSeek V4.1: WNR3 | the same for deepseek_v41; the published REAP-192 container runs end to end | no new one | `728c9e33` | #2879 |
| 10 | WNR3 encoder | `tools/wex_nr3_tools.c`, `tools/build_nr3.py` (writes WEX3; Wex-WRN2's `wex4_convert.py upgrade` makes WEX4) | none | `069ecad5` | #2880 |
| 11 | CUDA helper | `tools/wex_cuda_bridge.cu` over `wex_cuda.cuh`, `make wex-cuda-helper` (`kaira_wex_cuda.dll` / `.so`), `make wex-cuda-test`; workspaces pooled per call (the VRAM-leak fix); the host batch writes a refused record nowhere | none | `8d7fdf45` | #2883 |
| 12 | CUDA loader | `wex_decode.c` opens the helper and sends it the neural records; CPU fallback on any status but 0; `coli_wex_cuda_*` device and batch calls | `COLI_WEX_CUDA_LIB` (unset: nothing is opened or searched for) | `a2c61a77` | #2884 |
| 13 | Mirrors and split reads | qwen38 and deepseek_v41 read the container from `COLI_MODEL_MIRROR` copies, split by `COLI_DISK_WEIGHTS` | `COLI_MODEL_MIRROR` (unset), `COLI_DISK_WEIGHTS` (V4.1: probed; qwen38: equal shares) | `0f0698e9` | #2885 |
| 14 | Hash on the team | deepseek_v41 hashes a batch's RAW records at their first read on the OpenMP team | `COLI_WEX_VERIFY` (as above) | `a6bdd4e5` | #2886 |
| 15 | Fast CPU decoder | `wex_fast_decode.h` (kernel 0 and kernel 1: AVX2 by CPUID, NEON) and its test against the reference in three builds; then the `--seed` fix | none | `496b9e96`, `614e4c9e` | #2888 |
| 16 | Fast decoder in use | `coli_wex_decode_record` uses it on the CPU, one thread a record | `COLI_WEX_FAST_DECODE` (`1`, on; `0`: the reference) | `2177e679` | #2889 |
| 17 | Bench | `COLI_WEX_STATS`, `tools/wex_bench.py`, `tools/wex_decode_bench.c`, `docs/wex-bench.md`, then the report scrub fix | `COLI_WEX_STATS` (unset, off) | `60c27c52`..`a0ac6751` (3) | #2894 |
| 18 | GLM-5.3 on MXFP4 | glm53 reads AMD's MXFP4 checkpoint as published (`amd/GLM-5.3-Flash-Quark-MXFP4`): routed experts as fmt 7 through `matmul_mxfp4`; tiers given `VKT_SRC_MXFP4_E8M0` | none | `eb91d039` | #2895 |
| 19 | GLM-5.3 from WEX4 | routed experts from a WEX4 container written by Wex-WRN2's `pack --names qwen38`, RAW or WNR3, through `expert_read` | `COLI_WEX_PATH`, `COLI_WEX_ROOT`, `COLI_WEX_VERIFY`, `COLI_WEX_CUDA_LIB`, `COLI_WEX_FAST_DECODE`, `COLI_WEX_STATS` | `eff6bfa0` | #2896 |
| 20 | `make check` and direct decode | `make check` green on the stack (`test_wex_nr3_gpu` in `AUTODEP_OWN_FLAGS`, the fast decoder's test rules from `-MMD`); `coli_wex_decode_record_to` (decode straight into the expert's buffers, for V4.1 and Qwen3.8); the stream count is measured at run time | `COLI_WEX_FAST_LANES` (unset: measured on the first 16 records; `1`/`2`/`3`/`4`/`6`/`8` fixes it) | `d0dcb413`, `830525f0` | #2897 |
| 21 | `make check` on the whole series | `tests/test_glm53_wex4`'s rule takes its headers from `-MMD` (`tests.test_makefile_deps` failed on it once #2896 and #2897 were stacked together) | none | `86a09fc1` | - (new) |
| - | This page | the description of the series | - | the last commit | - |

The series order of JustVugg/colibri#2860 is followed: series 1 (rows 1-5), 2 (6-10), 3 (11-12),
4 (13-14) and 5 (15-16), then the bench (17, "series 7"), then GLM-5.3 (18-19, "series 6"), then
#2897.

### What changed in the consolidation

- **#2897 was not on top of the stack.** It had been stacked on #2889 (`bceab880`), beside #2894,
  #2895 and #2896. Its two commits were cherry-picked onto #2896's head. The first commit applied
  cleanly. The second conflicted with #2894 in four files, and both sides were kept:
  - `CHANGELOG.md`: both entries.
  - `docs/ENVIRONMENT.md`, three tables: both the `COLI_WEX_STATS` and the `COLI_WEX_FAST_LANES`
    rows.
  - `c/coli_env.h`: `COLI_WEX_FAST_LANES` was added beside the others, and it is also registered
    for glm53, which decodes through the same `coli_wex_decode_record`.
  - `c/wex_decode.c`: both the `COLI_WEX_STATS` counters and the stream-count measurement.
    `coli_wex_decode_record_to`, which #2897 wrote before the counters existed, now counts its
    decodes for `COLI_WEX_STATS` as `coli_wex_decode_record` does (fast, reference, failed).
- The whole linear stack was then rebased from `dev` @ c7752bf5 onto `dev` @ 7f6a8ab2 without a
  conflict.
- The two #2897 commits were committed under the series author. Their content is the PR's, apart
  from the conflict resolution above.
- One build fix was added (part 21). #2897's `make check` fix predates #2896 being under it, so
  `tests.test_makefile_deps` flagged the headers that #2896's `tests/test_glm53_wex4` rule lists by
  hand.

## How to review

Review part by part, in the order of the table, and commit by commit within a part. Each part was a
pull request of its own on JustVugg/colibri and built and passed `make test-c` there on top of the
parts before it. Its discussion and its full description (measurements, the exact commands) stay
on that PR. Nothing changes for a run that sets none of the variables above. Without
`COLI_WEX_PATH`, every engine reads the shards exactly as on `dev`.

## End-to-end results

DeepSeek V4.1 Flash REAP-192 with the published WNR3 container
([Kaira666/Kaira-WEX4-Experts](https://huggingface.co/Kaira666/Kaira-WEX4-Experts),
`v41-reap/v4_flash_experts.nr3.wex`, root
`f0f78a84d040a3b336f3abef3cf8b0a36c649deb38586c813e94d35dea3beeb1`).

Conditions:

- Windows 10, Intel i7-7700K (4 cores / 8 threads), RTX 3070 8 GB.
- 8 expert slots, prompt "The three primary colors of light are", 8 greedy tokens.
- Engram off, DSpark on (as the checkpoint says).
- The checkpoint on a SATA SSD, the container on NVMe (Samsung 970 EVO Plus).
- Root pinned, so every record is hashed at its first read.
- One binary of the branch for every arm.

| experts from | WNR3 decoded by | decode 8 tokens | tok/s |
|---|---|---:|---:|
| the shards (3 runs) | - | 62.5 / 78.8 / 91.6 s | 0.13 / 0.10 / 0.09 |
| container | the reference (`COLI_WEX_FAST_DECODE=0`) | 293.2 s | 0.03 |
| container | the fast CPU decoder (default) | 71.1 s | 0.11 |
| container | the CUDA helper (`COLI_WEX_CUDA_LIB`), after the VRAM-pool fix | 35.2 s | 0.23 |
| container, Vulkan backend (30 of 40 layers on the card) | the CUDA helper, after the fix | 48.0 s | 0.17 |

- Every arm gave the same tokens, `6247 14 9214 14 14817 343 2869 288` (" Red, Green, Blue
  (noting").
- The container arms read 37.5 GB against 43.7 GB from the shards (2,257 WNR3 records, 0 refused).
- After the fix, every record went to the helper: 2,257 / 0 on the CPU engine, 2,198 / 0 with
  Vulkan.
- Before the fix, the same two helper arms took 57.9 s and 48.7 s, because a third of the records
  went back to the CPU.
- In the same session as the post-fix rows, the fast CPU decoder with no helper gave 69.1 s (CPU
  engine) and 76.8 s (Vulkan).
- With the reference decoder (series 2 alone), an earlier session gave the shards 74.6 s against
  293.6 s from the container.

Per record:

- The direct decode and the measured stream count of part 20, on real V4.1 records decoded from
  memory on a 4-vCPU Xeon VM (median of 5 runs): 8 readers 31.3 -> 29.0 -> 25.6 ms a record, 4
  readers 31.7 -> 29.0 -> 25.8 ms. 6 streams were picked in 10 of 10 runs.
- With 8 readers, the V4.1 readers' transient buffers go from about 301 MB to about 129 MB.

## Review history kept

- **Apple M4, native, by mfethe1** on #2888
  ([comment](https://github.com/JustVugg/colibri/pull/2888#issuecomment-6929957696)): the three
  builds of `test_wex_fast_decode` pass with 0 mismatches. On synthetic V4.1-size records, NEON
  kernel 1 is 5.4x the reference on one thread, and 6 streams run 1.39x the default 4. This was the
  first native ARM run of this code; part 20's measured stream count follows from it.
- **The `--seed` fix** (`614e4c9e`), found in that run: a nonzero seed is taken as it is (2 and 3 are
  different streams), and 0 maps to 1 because the xorshift state must not be zero.
- **The helper's VRAM leak**, found in the end-to-end runs and fixed in part 11. With
  `COLI_WEX_CUDA_LIB` set, about a third of the WNR3 records came back with status 1 (1,500 / 757
  on the CPU engine).
  - The cause: each calling thread kept its own workspace, and the engine's reader team retires and
    respawns threads. On Windows a thread-exit `cudaFree` fails, so the VRAM of exited threads
    leaked until the 8 GB card was full.
  - The fix: the workspaces come from process-wide pools, taken for one call and given back.
  - `test_wex_nr3_gpu` has a new check: 64 rounds of 4 fresh threads, and the device memory the
    helper holds must not grow. The old helper fails it (1654 MiB more), and the fixed one passes
    (2-4 MiB).

## Decisions taken

- No licence header on the files brought from Wex-WRN2 (colibri's files carry none).
- `COLI_WEX_PATH` has no default: unset, no container is looked for and nothing of the series runs.
- The CUDA helper is opened only when `COLI_WEX_CUDA_LIB` names it: unset or `0`, nothing is opened
  or searched for, and every record decodes on the CPU.
- `COLI_WEX_VERIFY` is on when `COLI_WEX_ROOT` pins the container, and off otherwise.
- A bad or refused record:
  - DeepSeek V4 and V4.1 read that expert from the shards, and it stays on them for the run.
  - Qwen3.8 stops with the reason, because its store has no second copy of the MXFP4 experts.
  - GLM-5.3 stops too. Its MXFP4 shards do hold the same experts, but it does not fall back to
    them.
- A container whose WNR3 records are partly value-exact (canonical zero) and partly not is refused
  as a bad file.
- Under `DSV4_CUDA_MOE_MASKED=1`, DeepSeek V4 refuses a canonical-zero (value-exact) container
  and reads the safetensors. The masked MoE path is not deterministic and emits -0 itself, so the
  value-exact claim is not proven there.
- `wex_decode.c` is included in each engine, as series 2 does, until the maintainer asks for a
  linked object.
- Kernel 2 of Wex-WRN2's fast decoder (AVX2 / SVE gathers) is left out: it was slower here.
- The fast CPU decoder is on by default (`COLI_WEX_FAST_DECODE=1`), because it is byte-identical to
  the reference, refuses the same records, and is tested.
- One thread per record in the engines. The decoder can run 8 threads on one record (27 ms a V4.1
  record), but the engines already decode several records at once.

## What ran where

At the head of this branch, before this page's commit, on one Windows 10 desktop (i7-7700K, RTX
3070) and its WSL2, on 2026-11-17. The machine was otherwise idle: a background job was suspended.

The full suite and the WEX subset ran at `830525f0`, and part 21 was checked again at `86a09fc1`. The
WEX subset is `test_wex_format`, `test_wex4_verify`, `test_wex_decode`, `test_wex_fast_decode`
(default, `_x86_64` and `_scalar` builds), `test_wex_cuda_loader`, `test_v4_wex4`,
`test_qwen38_wex4`, `test_dsv41_wex4`, `test_glm53_wex4` and `test_glm53_mxfp4`.

| Where | What | Result |
|---|---|---|
| Windows, MSYS2 UCRT64 gcc 16.2 | `make -j8 test-c` (the whole C suite) | exit 0 |
| Windows | `make check-env` | 868 variables, registry matches the sources |
| Windows | the WEX subset built with `-fsanitize=undefined -fsanitize-undefined-trap-on-error` (MinGW has no ASan), with `COLI_WEX_FAST_DECODE` unset and `0`, and with `COLI_WEX_FAST_LANES=8` | all pass, except the known case below |
| Windows | `tests.test_makefile_deps` after part 21 | OK |
| Windows, CUDA 13.5, RTX 3070 (driver 621.01) | `make wex-cuda-test` | `ALL OK`: both synthetic models on 4 entry points, crafted records rejected with destinations untouched; thread churn 64 rounds x 4 threads, device memory grew 0 MiB |
| WSL2 Ubuntu 26.04, gcc 15.2 | the WEX subset with `COLI_WEX_FAST_DECODE` unset and `0` | all pass |
| WSL2 | the same subset under ASan + UBSan (`make test-asan`'s flags and options) | all pass, no report |
| WSL2 | `make check-env` | 868 variables, registry matches |
| WSL2 | `tests.test_wex_nr3_tools`, `tests.test_makefile_deps` | OK at `86a09fc1` (12 tests, 1 skipped); one failure at `830525f0`, fixed by part 21 |

The known case: `test_wex_fast_decode` fails its last check when `COLI_WEX_FAST_LANES` is already
set in the environment it is started from. With the variable set to 8, the check reported
"stream count measured here: 0". This is a limit of the test, not of the decoder:

- The test sets and unsets the variable itself.
- At its end, it expects the count to have been measured on the earlier records, and with a fixed
  count nothing is measured.
- Started with the variable unset, it passes everywhere above. On the i7-7700K it measured 4
  streams (CPU time per byte against 4: 3: 1.08, 6: 1.32, 8: 1.24).

Before the consolidation, each part ran its own tests, listed in its PR. Highlights:

- Windows 10 (i7-7700K, MSYS2 UCRT64 gcc) and WSL2 Ubuntu gcc, plain and under ASan + UBSan.
- NEON under `qemu-aarch64`, and the AVX2-less path under `qemu-x86_64 -cpu Westmere`.
- `make wex-cuda-test` on the RTX 3070.
- The end-to-end runs above.
- #2897's measurements on a 4-vCPU Xeon VM.

Not run on this branch:

- `make check` and `make test-python` as a whole (only the two Python tests above).
- Any whole-model run: the end-to-end numbers above come from the separate PR branches.

Not run on any machine:

- A whole-model run of Qwen3.8, DeepSeek V4 or GLM-5.3 with a WNR3 container.
- GLM-5.3 shards against the container.
- A native ARM Linux machine, macOS end to end, and another NVIDIA GPU.
- MSVC builds of the decoder files.
- The engram tables.

## Tests wanted

See the "Tests wanted (help welcome)" section of JustVugg/colibri#2860. It lists each missing test
with the hardware, the `c/tools/wex_bench.py` command, and what to expect.

## What is left

- A WEX4 writer in colibri. Today the encoder of part 10 writes WEX3, and Wex-WRN2's
  `wex4_convert.py upgrade` makes WEX4.
- Speed numbers measured on colibri `dev` itself, per engine and per backend.
- The Vulkan and Metal paths for WNR3 records. WNR3 on Vulkan exists as a prototype in Wex-WRN2.
- The engines' use of the helper's batch and device calls (decoding straight into VRAM, a batch of
  misses in one call).
- More than one thread per record, where an engine decodes a single record at a time.
- Not proposed, from measurements in the downstream fork:
  - Splitting one record across two drives (-8.7% decode).
  - One read per V4.1 record instead of six (+10% on NVMe, -11% on a SATA SSD).
  - Sub-reads and the batched tier prewarm.

