# Stage 0 — prepare (2026-11-01 amend bench)

Clone: /tmp/bench-amend from /tmp/colibri-perf, branch performance-pr @ f06f9630bd221e4de1c7538438eaadeddb80b46c (amended PR tip; base main @ d67ebd8b).

## Binaries

| role | path | sha256 | notes |
|---|---|---|---|
| PR (fresh build) | /tmp/bench-amend/c/deepseek_v4 | 0e820833a162e16d7f430673957a50538c1c989a3f94578f87f5e633dbcccb14 | make -j18 VK=1 deepseek_v4, rc=0, 99 coli_vk symbols |
| BASE (audit, not rebuilt) | /root/audit-wt/base-tests/c/deepseek_v4 | cad05bb456dc3b66b53bdf973815402295c386da1e0d26036bd0f7c3bd397800 | built @ d67ebd8b in the 2026-10-31 audit; --help boots rc=0; 91 coli_vk symbols |

PR boot-engagement check (tiny invocation, harness env, GPU-locked, --max-tokens 1, rc=0, TTFT 13.384 s):
"tier deepseek_v4: on,", "chain: 43 of 43 layers on the device", "attention FA engaged",
"decode-path=fork ... engaged", "s1gemv ... armed", "cs experts armed", "decode grow armed".
Log: /tmp/amend-boot.log (not an evidence artifact).

## Driver and harness (path-only adaptations of the protocol of record)

- driver: /tmp/bench-amend/bench-amend-driver.sh (sha256 33d5c17e...) — from docs/experiments/dsv4-perf-2026-10-31-audit2/driver.sh; changed lines 20/22/26 only: BENCH=/tmp/bench-amend/bench-out, RUN=/tmp/bench-amend/bench-amend-run.sh, PR=/tmp/bench-amend/c.
- harness: /tmp/bench-amend/bench-amend-run.sh (sha256 b22cdbcb...) — from the same dir's run-audit.sh; changed line 16 only: OUT=/tmp/bench-amend/bench-out.
- BASE, MODEL, PROMPT, regime env, leg list, interleave+shuffle, VRAM gate, retry, resume, device-lost stop: all unchanged.
- Launch protocol: `flock /tmp/gpu.lock /tmp/bench-amend/bench-amend-driver.sh` (GPU lock applied at launch; the audit2 driver of record has no embedded GPU lock).

## Prompt

/root/Projects/colibri-upstream/bench/prompt_2k.txt — sha256 2e06e22974e445d49e623405d10b4f065bcb88369f63aeff8c9cba0aa904cc3d, 8,944 bytes — byte-identical to the audit2 manifest's fixed 2,515-token prompt. (Searched /root/bench-colibri/up2276: it references prompt_2k in audit-report.md but holds no local copy; both protocol scripts point at the path above.)

## Leg plan (12 legs)

- rep 1 (shuffled): uvk-r1 (BASE, env-free), legacy-r1 (PR, COLI_VK_LEGACY=1), defaults-r1 (PR, env-free), oneline-r1 (PR, COLI_VK_PROFILE=RDNA2 COLI_VK_TIER_HOST_GB=2)
- rep 2 (shuffled): uvk-r2, legacy-r2, defaults-r2, oneline-r2 (same arm/env mapping)
- rep 3 (shuffled): legacy-r3, defaults-r3, oneline-r3
- tail: mm8-r1 (PR, oneline env + COLI_VK_CHAIN_MM8_NDBUF=2)

Per leg: VRAM gate (7x10 s: band<=64 MiB, max<=3544 MiB, avail>=2 GiB; blocked >10 min -> SKIPPED-GATE), systemd-run scope MemoryMax=22G, env COLI_TEMP=0 KVSAVE=0 V4_PREFIX_CKPT=0 COLI_VULKAN=1 COLI_V4_PREWARM=0 OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores, --prompt-file prompt_2k --max-tokens 64 --memory-gb 16, one retry per failed leg, resumable status files in bench-out/legs/.

## Deviations from the letter of the task brief (none to the protocol of record)

- The brief wrote "V4_PREWARM=0"; the harness of record passes COLI_V4_PREWARM=0 — script kept verbatim.
- Driver-internal leg tags keep the audit2- prefix (protocol verbatim); committed raw artifacts will be renamed <leg>.raw.txt as in the audit2 evidence dir.
- systemd-run present on the rig -> MemoryMax scope path active; no plain-run fallback needed.
- Base binary verified working; the conditional rebuild was not triggered.
