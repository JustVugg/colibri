# Avoid allocating streaming slots for an empty prefetch

`vkt_stream_prefetch()` allocated its staging slots before looking for eligible
experts. Even if all experts were resident, or no expert met the prefetch
threshold, the empty call consumed the entire reserved slot allocation.

The change finds candidates first, then allocates slots only when at least one
candidate exists. The resident budget, expert placement and existing streaming
behavior are unchanged. It does not reclaim slots that an earlier real streaming
request already allocated.

A regression uses a small native-MXFP4 model with all 20 experts resident and
four reserved streaming slots. The same test is run against the original and
candidate tier, using CPU Lavapipe with no GPU devices exposed.

| Empty-prefetch result | Original | Candidate |
| --- | ---: | ---: |
| Tier bytes before | 921,600 | 921,600 |
| Tier bytes after | 1,105,920 | 921,600 |
| Extra tier allocations | 24 | 0 |
| Full Vulkan tier suite | Only new regression fails | Pass |

The original's exact diagnostic is:

```text
FAIL tests/test_vk_tier.c:970: empty prefetch allocated slots: used 921600 -> 1105920, allocations 8422 -> 8446, live 120 -> 144
```

The existing streaming tests cover actual cold candidates, mixed resident and
streamed experts, changing formats, staged copies, and failure fallback. They
pass with the deferred allocation. The test verifies live/used allocation
counters, not a platform-specific physical heap size.

Reproduce on original/candidate trees with the same new test:

```sh
make -C c tests/test_vk_tier VK=1 ARCH=x86-64-v3 -j8
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  ./c/tests/test_vk_tier c/shaders/qmatmul.spv
```

The recorded containers use Mesa 26.2.6 / llvmpipe LLVM 21.1.8, CPUs 16–31,
and no passed GPU devices. Exact container commands and exit codes are in
`commands.jsonl`; compressed full build/test logs and their hashes are retained.
The original exits 1 with exactly the new assertion; the candidate exits 0.
This is an allocation regression test, not a throughput benchmark.

## Model reproduction

DeepSeek V4 Flash REAP uses 64 slots of 15 MiB each: 960 MiB of unused payload. The model logs show 58 tier heap blocks before and 54 after (256 MiB each), a 1 GiB heap-allocation reduction. Both keep 913 resident experts, a 14.37 GiB budget and automatic 1,280-row chunks. Resident placement and shader binaries are unchanged.

| Arm/run | Long prompts completed | Peak sampled GPU MiB | Outcome |
| --- | ---: | ---: | --- |
| Original `20261108-134038` | 1/3 | 24308.137 | second long prompt: execbuf ENOMEM |
| Deferred slots `20261108-140146` | 3/3 | 24297.160 | full workload completed |

All three fixed answers match (452, 632 and 771 characters); every common completed benchmark request and output is byte-identical. The original stops after one warmup, one long prompt, one short prompt and two decode samples. The candidate completes one warmup and all nine measured samples. The peak VRAM figures cover different completed durations; they are not a claim of a fixed VRAM delta.

| Arm | Workload | Repetition | TTFT s | Decode tok/s |
| --- | --- | ---: | ---: | ---: |
| Original | warmup | 1 | 12.045928 | — |
| Original | prefill_long | 1 | 585.746125 | — |
| Original | prefill_short | 1 | 12.905747 | — |
| Original | decode_short | 1 | 20.776576 | 1.517708 |
| Original | decode_short | 2 | 21.252173 | 1.669518 |
| Deferred slots | warmup | 1 | 11.873396 | — |
| Deferred slots | prefill_long | 1 | 593.102427 | — |
| Deferred slots | prefill_short | 1 | 12.877480 | — |
| Deferred slots | decode_short | 1 | 20.738703 | 1.587910 |
| Deferred slots | decode_short | 2 | 21.185653 | 1.846439 |
| Deferred slots | prefill_long | 2 | 525.966514 | — |
| Deferred slots | prefill_short | 2 | 12.502379 | — |
| Deferred slots | prefill_long | 3 | 528.000271 | — |
| Deferred slots | decode_short | 3 | 20.885073 | 1.663905 |
| Deferred slots | prefill_short | 3 | 12.676769 | — |

This is an allocation bug fix. The incomplete original arm does not support a three-sample throughput comparison, so no speedup or throughput manifest is claimed. All partial samples and the exact failure remain in the raw evidence. This 3,047-token prompt workload does not validate the full configured 64K context.

Hardware: Ubuntu 26.04.1 guest, kernel 7.0.0-42, Xeon Gold 6254 (32 vCPUs, experiment CPUs 16–31, OMP 16), 247.37 GiB RAM. One Arc Pro B60 24 GB SR-IOV VF, PCIe Gen3 x8, Mesa 26.2.6. Model SHA-verified locally on ext4 /dev/sda2, a 1 TiB QEMU disk through virtio-scsi. The other GPU served unchanged production throughout.

Runtime base `e24cd51afd44b480c1e41ba82e20a28bac76c34f` plus the common Xe2 patch in `shared-runtime.patch.gz`; the candidate adds only this allocation change. The entire runtime `vk_tier.c` matches this PR, and all 69 shaders match the control. The CPU SIMD change from PR #2708 is absent from both arms. Image and binary identities, effective Compose configurations, model file hashes, logs and checksums are included.

Build each runtime tree with GCC 15.2.0-16ubuntu1 and the retained shaderc 2026.1-1 /
glslang 16.1.0-1 toolchain:

```sh
make -C c deepseek-v4 VK=1 ARCH=x86-64-v3 -j8
```

The measured build ran in `colibri-builder:xe2`, user 1001:1001, CPUs 16–31, with
the source mounted at `/src` and no GPU devices. `runtime-build.log.gz` preserves
the build output; `compiler.json` identifies the toolchain. `Dockerfile.models`
copies that executable and the shaders over the retained Mesa/gateway runtime.
The local container image tags are not a claim of published downloadable images.

Timing uses a fresh engine for each arm, deterministic placement `COLI_VK_TIER_SYNC=1` / `COLI_VK_TIER_BALANCE=0`, request tag `f6734185f687174a`, temperature 0, reasoning disabled, one warmup and three repetitions of each workload. Original adaptive answers are preserved separately; this deterministic reference is not an adopted serving configuration.

To reproduce the requests, start the corresponding runtime with the configuration in `original-session.json` or `candidate-session.json` in an isolated GPU session, then run:

```sh
python3 replay.py --url http://127.0.0.1:8002/v1/chat/completions --output /tmp/reap-replay.json
```

`replay.py` uses the measured HTTP timing function and saved smoke/reference/workload sequence. It was packaged after the measurements; it does not manage containers or collect VRAM telemetry. The original is expected to fail during the second long request. The local lifecycle runner restored Qwen3.8 and verified production after each arm.

Full `make check -j16` also passed on the PR base `f5d65566de468c8e7ac285a5e8c7257bf4d06856`: 3,089 Python tests, 397 skips, plus the C/portable checks. The full log includes 66 warning lines from unchanged code/test fixtures; none names the modified tier files. CPU-only containers exposed no GPU devices. Raw command and full log are retained.
