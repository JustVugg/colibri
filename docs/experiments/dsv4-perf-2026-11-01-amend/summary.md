# dsv4-perf-2026-11-01-amend — summary

The amended PR tip f06f9630, re-measured 2026-11-01 (18:04–20:09 AEDT) under
the audit2 protocol of record (same rig, same regime, same prompt, same
driver/harness logic — path-only adaptations; `samples.txt` / `manifest.json`
/ `window.raw.txt` in this directory carry the full evidence). 12 legs, all
GREEN; every number recomputable from the raw logs (decode = 63/(wall−ttft)).

## Arm table vs the audit2 reference (2026-10-31, cb0a0a91)

| arm | n | decode med (range) this | TTFT med s (range) this | decode med (range) audit2 | TTFT med s (range) audit2 |
|---|---|---|---|---|---|
| LEGACY (COLI_VK_LEGACY=1) | 3 | 0.6898 (0.6839–0.8350) | 258.334 (248.161–269.972) | 0.7882 (0.6208–0.8708) | 253.953 (252.481–261.163) |
| defaults (env-free) | 3 | 0.8786 (0.8663–0.9259) | 259.386 (245.870–269.755) | 0.9037 (0.8987–0.9193) | 251.805 (245.498–260.897) |
| oneline (RDNA2+HOST_GB=2) | 3 | 0.8887 (0.8085–0.9265) | 151.606 (147.327–159.439) | 0.8911 (0.8740–0.9330) | 155.972 (149.080–160.839) |
| uvk / base (d67ebd8b) | 2 | 0.6754 (0.5503–0.8005) | 468.322 (432.921–503.724) | 0.5187 (0.5135–0.5239) | 424.675 (414.880–434.471) |
| mm8 opt-in (n=1 tail) | 1 | 0.9115 | 150.492 | 0.9570 | 162.178 |

Mem peaks this session: LEGACY 15.33–15.81 GiB; defaults/oneline/mm8
2.24–2.42 GiB (the slab registry's host-RAM drop reproduced); base legs
16.18/16.19 GiB (the sampler caught them this time — audit2's D2 miss).

## Derived claims (med/med; audit2 reference in brackets)

- defaults vs LEGACY, decode: **+27.37%** (0.8786 vs 0.6898; per-rep
  +10.9/+26.7/+27.4%) [audit2 +14.65%]; TTFT **+0.4%** = parity [−0.8%].
  Sits between audit2's +14.7% and the gap4 record's +41.8% — LEGACY is
  again the noisy arm (0.68–0.84 this session; 0.62–0.87 in audit2).
- oneline vs LEGACY, TTFT: **−41.31%** (151.606 vs 258.334 s; per-rep
  −40.6/−41.3/−40.9%) [audit2 −38.58%]; decode +28.8% [+13.1%].
- oneline vs defaults, TTFT: **−41.55%** [audit2 −38.06%], decode +1.1%
  [−1.4%] — parity kept.
- oneline vs this session's base legs: decode +31.58%, TTFT −67.63%
  [audit2 +71.79% / −63.27%] — see disclosure (a): the base arm moved.
- mm8 opt-in vs oneline (n=1): decode +2.6%, TTFT −0.7% [audit2 +7.4% /
  +4.0%].

## Honest disclosures

(a) **The base-engine legs ran chain 2/43 this session (audit2's ran
9/43)** — both are the base tip's co-tenant stranding, not a code delta: the
same d67ebd8b binary (not rebuilt) maps the 13.28–13.31 GiB fmt-7 expert
tier onto the device budget before the chain fit, and how many dense layers
survive depends on the co-tenant's VRAM at open time. The base arm is a
cross-session reference only; the PR's claims rest on the within-tree arms
(LEGACY vs defaults vs oneline, all 43/43 via the PR's window probe in every
PR leg).

(b) **The uvk (base) decode spread is 0.5503–0.8005** across its two reps
(0.8005 @ TTFT 503.724, 0.5503 @ TTFT 432.921) — a 1.46x spread, consistent
with 41 of 43 dense layers running on the CPU beside a full expert tier; its
medians (0.6754 / 468.322 s) carry that spread and straddle LEGACY's. Do not
read the base-vs-PR deltas as a stable engine comparison; use audit2's and
gap4's base records alongside.

(c) **legacy-r3's first attempt was killed by an external timeout at 14:19**
(14 min 19 s elapsed; rc=143, PARSE-FAIL — the log shows the mid-prompt
device loss `queue submit failed (-4): the device is lost` and the CPU
fallback, then the kill; no timing lines). Its **recorded retry is the leg
of record** (audit2-legacy-r3-r2, launch 19:58:29, rc=0, 269.972 / 0.6898 /
361.299). Both raw logs ship (`legacy-r3-attempt1.raw.txt` is the killed
attempt, not a sample; `legacy-r3.raw.txt` is the retry).

Also disclosed (quality gates): no fresh `make check` ran this session. The
fresh build is rc=0 with 99 coli_vk symbols; every shipped leg is rc=0 with
its engagement boot lines; the audit2 gate matrix ran green at cb0a0a91
(2026-10-31), and the only engine change cb0a0a91 → f06f9630 is 9ba76c9e
(mm8-NDBUF boot line + registry-refusal reorder; byte-silent unset, arming
path byte-identical), witnessed both ways this session (zero mm8-ndbuf
lines in every arm that sets no env; the armed boot line + rc=0 in mm8-r1).

Validator: `python3 c/experiment_manifest.py` →
`manifest.json: ok (COLI_VK_LEGACY, improvement)`.
