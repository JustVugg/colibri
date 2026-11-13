# GGUF on qwen36's Metal graph, measured on an M1

The fast mode of the GGUF kernels before and after this change, the expert cache's budget for a GGUF, the model's own
format before and after the three GGUF changes, and the GGUF against the model's own format, with where the time goes.
One harness, `bench.c`, built with the command in the manifests against the code as measured: 08293922 (this change),
435651fb (the change before) and 085b5fc4 (the base of the three GGUF changes), local commits of 18 and 19 October on
upstream dev e68e0cc9 (085b5fc4 was this series' PR 14 then). This branch holds this change and the one before on a
newer dev and on the series as it is now: this change patch for patch, the one before with a few lines adapted where
it met newer code of dev. An M1 MacBook Air with 8 GB (no fan), macOS 26.6.2. Two files of Qwen3.6-35B-A3B: unsloth's
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.1 GB: the dense matrices in Q8_0, the head in Q6_K, the routers and the
DeltaNet's alpha and beta projections in F32, the routed experts' gate and up in Q4_K and their down in Q5_K, or Q6_K
in three layers), in the directory `coli import --mode link` makes for it (`<GGUF model>` in the logs); and the model
in Colibri's qwen36 container with `canonical.qc` and `experts.qc` beside it (`<model>`).

## How it was measured

- Texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a), code
  ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`) and chat ([chat.txt](chat.txt): an English
  conversation written for these measurements, plain User:/Assistant: text). The harness reads (longest prompt +
  forced tokens) × 32 + 4,096 bytes of a text: 21,504 in the speed runs (all of the chat's 6,900). Each manifest's
  `prompt_hash` is its text's SHA-256. The two files give the same token ids for the three texts: the hash of the
  teacher-forced tokens is the same in every run of both.
- The harness `bench.c` includes `qwen36.c`, opens a GGUF directory as the engine's `main` does (its `gguf.json`
  names the file), and decodes with `q36_spec_step`, as `generate()` does (no drafts here: the GGUF has no MTP head,
  and the model's own directory has no `mtp.safetensors`). It times `model_init`, reads each prompt from an empty
  state, then runs the decode steps and hashes every logit (FNV-1a) and the tokens. `TEACHER=1` feeds each step the
  text's next token, `BENCH_NLL=<file>` writes each token's negative log-likelihood and whether it was ranked first,
  `BENCH_DUMP=<file>` the last logits; `compare_nll.py` compares two `BENCH_NLL` files.
- Build: the command in the manifests (`git archive`, `make qwen36_metal.o backend_metal.o METAL=1`, then the harness
  with the flags of `make -C c qwen36 METAL=1` except its qpack path). This change's harness was built from the `c/`
  files of an earlier local version of the commit, which differed only in comments of `qwen36_metal.m`, the kernel
  test and the documentation; the command builds the same binary, byte for byte, from `git archive 08293922` (sha256
  dbc7581d…, with the same output name, which the linker's ad hoc signature includes). The launchers in the commands
  are one-line scripts
  that set the model and the variables of a variant and run one of the binaries: `gguf-435651fb` (the change before)
  and `gguf` (this change) set `SNAP=<GGUF model>`, `gguf-264-slots` also `QWEN_EXPERT_CACHE_BYTES=538509312`, `own`
  and `own-085b5fc4` set `SNAP=<model>`, `gguf-guessed` and `own-guessed` add `QWEN36_FAST_EXPERTS=5
  QWEN36_FAST_GUESSED=4`, `gguf-exact` and `own-exact` `QWEN36_METAL_EXACT=1`, `gguf-cpu-exact` `COLI_METAL=0
  GQ_EXACT=1` (the CPU engine's exact mode).
- Speed runs: the first 64 and 512 tokens of each text, each from an empty state, then 32 decode steps, each fed the
  text's own next token (teacher forcing: both versions read the same words); a warm-up run of the baseline (in the
  comparisons within one file both versions read the same file; in those of the two files the trial had one too,
  before the baseline's, so that the order is symmetric: each measured run of one version has one of the other version
  with the same pattern of files, its own or the other, in the two runs before it), then 4 runs per version in the
  order baseline, trial, trial, baseline, baseline, trial, trial, baseline, each a new process. Decode speed is steps
  2 to 32 (the first step after the prompt is timed apart). Before every run the script waited for the nominal thermal
  state (`NSProcessInfo.thermalState`), rested 60 s and read the swap counters (`vm.swapusage`, the Swapouts of
  `vm_stat`); each run's times, thermal state, counters, engine log and `/usr/bin/time -l` output are in the raw logs
  (`baseline-<comparison>-<text>.txt`, `trial-<comparison>-<text>.txt`), the summaries in
  `summary-<comparison>-<text>.txt`, the manifests in `manifest-<comparison>[-prompt]-<text>.json` (decode, or the
  prompt's speed, after the 512-token prompt; the outcome compares every run of one version with every run of the
  other).
- Files: the GGUF was downloaded on 18 October (finished at 23:34); `experts.qc` was last written on 7 October,
  `canonical.qc` on 1 October. A freshly written file reads faster on this SSD for a while, which could only favour
  the newer file, the GGUF: the comparisons within one file ran from 19 October 12:31, and those of the two files from
  23:47, more than a day after the GGUF was written.

## The kernels: the change before against this one, on the GGUF

The same file and the same defaults (the expert cache of a sixteenth of the RAM, 263 slots of 2.04 MB), the harness
of 435651fb against this change's, fast mode:

| text | prompt | decode, before | decode, this change | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 5.197 tok/s (5.136–5.204) | 7.119 tok/s (6.983–7.260) | +37.0% | 1033 → 921 ms | 9.20 (9.12–9.56) → 15.69 (14.92–15.97) tok/s, +70.5% |
| prose | 512 | 5.221 tok/s (5.193–5.278) | 7.482 tok/s (7.462–7.550) | +43.3% | 1310 → 933 ms | 12.77 (12.59–12.81) → 69.07 (66.06–70.16) tok/s, +440.7% |
| code | 64 | 4.893 tok/s (4.796–4.932) | 6.348 tok/s (6.275–6.383) | +29.7% | 1183 → 988 ms | 8.98 (8.66–9.42) → 15.30 (13.89–15.53) tok/s, +70.4% |
| code | 512 | 4.803 tok/s (4.765–4.846) | 6.151 tok/s (6.028–6.232) | +28.1% | 1373 → 1126 ms | 12.66 (12.50–12.69) → 60.56 (58.97–63.60) tok/s, +378.4% |
| chat | 64 | 4.828 tok/s (4.703–4.868) | 6.056 tok/s (5.782–6.134) | +25.4% | 1445 → 947 ms | 9.39 (9.30–9.42) → 13.95 (12.30–14.44) tok/s, +48.4% |
| chat | 512 | 4.652 tok/s (4.583–4.663) | 5.794 tok/s (5.648–5.842) | +24.6% | 1389 → 1130 ms | 12.54 (12.51–12.64) → 63.57 (59.74–65.08) tok/s, +406.9% |

Every run of this change is faster than every run of the change before, in decode after the 512-token prompt and in
the 512-token prompt, on the three texts (the six manifests' outcome: improvement). The words read are the same; the
logits differ in their last bits (the two fast modes group their sums differently), and the 32 tokens' mean negative
log-likelihood moves by 0.0011 at most. The model was ready in 0.14–0.25 s in both versions, the peak memory footprint
was 1090–1095 MiB, and no page was written to swap during the runs.

## The expert cache: one slot more for the layer-by-layer prompt

A slot holds the file's largest expert, 2.04 MB here against 1.77 MB in the model's own format, so the 8 GB Mac's
default budget (a sixteenth of the RAM, 512 MiB) holds 263 slots, of which 255 are retained: one short of the 256 the
layer-by-layer prompt (`qwen36_prefill_stream.h`) needs, so a prompt loads its experts eight at a time (the engine's
log gives the slot in MiB: "1.95 MB per expert at most", 2,039,808 bytes). With
`QWEN_EXPERT_CACHE_BYTES=538509312`, one slot (2.04 MB) more, the cache holds 264 slots and a prompt runs layer by
layer. This change's harness, fast mode:

| text | prompt | decode, 263 slots | decode, 264 slots | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 7.101 tok/s (6.956–7.141) | 7.137 tok/s (7.092–7.222) | +0.5% | 995 → 876 ms | 16.68 (15.32–16.97) → 16.22 (16.03–17.58) tok/s, -2.8% |
| prose | 512 | 7.380 tok/s (7.359–7.394) | 7.362 tok/s (7.292–7.546) | -0.2% | 1094 → 816 ms | 68.79 (67.97–71.52) → 85.50 (79.46–92.85) tok/s, +24.3% |
| code | 64 | 6.111 tok/s (5.974–6.165) | 6.046 tok/s (5.874–6.174) | -1.0% | 1165 → 1155 ms | 14.13 (13.38–15.28) → 14.40 (14.10–16.40) tok/s, +1.9% |
| code | 512 | 6.287 tok/s (5.652–6.431) | 5.986 tok/s (5.893–6.049) | -4.8% | 1120 → 1014 ms | 62.19 (59.48–64.18) → 71.72 (71.03–81.90) tok/s, +15.3% |
| chat | 64 | 6.096 tok/s (5.990–6.182) | 6.012 tok/s (5.776–6.273) | -1.4% | 970 → 1026 ms | 12.94 (12.63–14.72) → 14.57 (13.26–14.92) tok/s, +12.6% |
| chat | 512 | 6.014 tok/s (5.900–6.052) | 5.909 tok/s (5.750–5.992) | -1.7% | 862 → 932 ms | 60.97 (59.77–64.97) → 79.72 (75.10–84.33) tok/s, +30.8% |

The 512-token prompt is faster in every run with the larger cache (the three prompt manifests: improvement). Decode
after it: the ranges overlap on the three texts (no-change), but on code three of the four default runs (6.24–6.43
tok/s) are faster than every run with the larger cache (5.89–6.05; the fourth default run 5.65): the layer-by-layer
prompt may leave the cache a little worse for the decode that follows. The logits are the same, bit for bit, in both
versions (the same fingerprints); the peak memory footprint rose from 1094 to 1097 MiB.

## The model's own format before and after the three GGUF changes

The same model directory and the same defaults, the harness of 085b5fc4 (the base of the three GGUF changes) against
this change's, fast mode:

| text | prompt | decode, 085b5fc4 | decode, this change | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 9.139 tok/s (9.067–9.251) | 9.158 tok/s (9.108–9.231) | +0.2% | 258 → 262 ms | 26.70 (26.07–27.41) → 26.95 (26.04–27.51) tok/s, +0.9% |
| prose | 512 | 9.372 tok/s (9.366–9.500) | 9.462 tok/s (9.340–9.517) | +1.0% | 262 → 254 ms | 137.48 (130.18–139.85) → 136.48 (134.34–140.04) tok/s, -0.7% |
| code | 64 | 8.273 tok/s (8.132–8.351) | 8.251 tok/s (8.222–8.366) | -0.3% | 302 → 303 ms | 24.17 (23.51–24.54) → 23.70 (23.16–24.18) tok/s, -1.9% |
| code | 512 | 8.024 tok/s (7.989–8.037) | 8.034 tok/s (8.021–8.039) | +0.1% | 314 → 291 ms | 115.09 (111.50–116.78) → 114.55 (113.06–116.45) tok/s, -0.5% |
| chat | 64 | 7.950 tok/s (7.844–8.015) | 8.005 tok/s (7.987–8.034) | +0.7% | 313 → 312 ms | 21.75 (20.68–22.19) → 21.73 (21.43–21.75) tok/s, -0.1% |
| chat | 512 | 7.969 tok/s (7.913–8.053) | 8.044 tok/s (8.033–8.096) | +0.9% | 274 → 290 ms | 114.94 (112.90–118.75) → 117.59 (115.88–119.01) tok/s, +2.3% |

The logits are the same, bit for bit, in every run of both versions (the same fingerprints), and the speeds are
within the noise (the six manifests' outcome: no-change). The model was ready in 0.14–0.20 s in both versions. The
peak memory footprint was 983–986 MiB before and 984–987 MiB after, 1.0 to 1.4 MiB more in the medians (the graph
makes every kernel's pipeline at start, the GGUF ones too). No page was written to swap during the runs.

## The GGUF against the model's own format

The same harness (this change's) and defaults, the model's own format (the baseline, `own`) against the GGUF (the
trial, `gguf`): two quantizations of the same checkpoint, so the words read are the same and the logits differ (their
likelihoods are under [Quality](#quality)). Three modes: the default fast mode, fast mode with fast and guessed
experts (`QWEN36_FAST_EXPERTS=5 QWEN36_FAST_GUESSED=4`, see [qwen36-metal.md](../../qwen36-metal.md)) and exact mode
(`QWEN36_METAL_EXACT=1`). Each version had a warm-up run, the GGUF's first.

Fast mode, the default:

| text | prompt | decode, own format | decode, GGUF | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 8.975 tok/s (8.625–9.154) | 7.034 tok/s (6.804–7.199) | -21.6% | 339 → 1179 ms | 22.09 (18.35–26.38) → 14.41 (14.00–15.43) tok/s, -34.8% |
| prose | 512 | 9.443 tok/s (9.400–9.468) | 7.409 tok/s (7.267–7.552) | -21.5% | 237 → 972 ms | 131.93 (127.95–137.22) → 66.06 (63.34–71.55) tok/s, -49.9% |
| code | 64 | 8.100 tok/s (7.914–8.167) | 6.027 tok/s (5.944–6.076) | -25.6% | 365 → 1278 ms | 20.06 (16.98–23.07) → 14.22 (13.24–15.53) tok/s, -29.1% |
| code | 512 | 7.825 tok/s (7.801–8.025) | 6.143 tok/s (5.980–6.233) | -21.5% | 270 → 1171 ms | 111.23 (106.07–114.14) → 60.94 (59.39–63.09) tok/s, -45.2% |
| chat | 64 | 7.848 tok/s (7.684–7.900) | 5.864 tok/s (5.598–5.957) | -25.3% | 368 → 1113 ms | 19.26 (16.47–22.21) → 12.15 (11.90–12.62) tok/s, -36.9% |
| chat | 512 | 7.779 tok/s (7.568–7.983) | 5.821 tok/s (5.792–5.836) | -25.2% | 285 → 1021 ms | 114.10 (112.81–115.88) → 60.69 (59.21–65.13) tok/s, -46.8% |

Fast mode with fast and guessed experts:

| text | prompt | decode, own format | decode, GGUF | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 12.008 tok/s (11.782–12.182) | 10.039 tok/s (9.929–10.153) | -16.4% | 332 → 1087 ms | 21.97 (18.01–26.25) → 14.23 (13.96–17.07) tok/s, -35.2% |
| prose | 512 | 12.463 tok/s (12.443–12.479) | 10.591 tok/s (10.406–10.630) | -15.0% | 206 → 1067 ms | 128.88 (124.67–135.00) → 70.32 (69.56–70.53) tok/s, -45.4% |
| code | 64 | 11.401 tok/s (11.237–11.560) | 9.126 tok/s (9.116–9.159) | -20.0% | 295 → 1153 ms | 20.59 (17.41–24.17) → 14.21 (13.20–15.22) tok/s, -30.9% |
| code | 512 | 11.301 tok/s (11.218–11.411) | 9.159 tok/s (9.052–9.245) | -19.0% | 212 → 1028 ms | 113.45 (106.73–114.86) → 61.95 (58.54–62.05) tok/s, -45.4% |
| chat | 64 | 11.215 tok/s (11.120–11.302) | 8.877 tok/s (8.812–8.993) | -20.8% | 307 → 1118 ms | 19.33 (16.50–22.27) → 12.26 (12.15–12.68) tok/s, -36.6% |
| chat | 512 | 11.102 tok/s (11.030–11.216) | 8.714 tok/s (8.509–8.860) | -21.5% | 199 → 1044 ms | 114.12 (112.97–119.05) → 61.44 (59.36–64.20) tok/s, -46.2% |

Exact mode:

| text | prompt | decode, own format | decode, GGUF | change | first step | prompt |
|---|---|---|---|---|---|---|
| prose | 64 | 8.260 tok/s (8.109–8.359) | 4.823 tok/s (4.800–4.898) | -41.6% | 437 → 1496 ms | 14.32 (12.46–15.88) → 7.16 (7.11–7.79) tok/s, -50.0% |
| prose | 512 | 8.437 tok/s (8.401–8.484) | 4.934 tok/s (4.878–4.962) | -41.5% | 307 → 1309 ms | 24.48 (24.34–24.84) → 9.46 (9.39–9.48) tok/s, -61.3% |
| code | 64 | 7.642 tok/s (7.515–7.751) | 4.513 tok/s (4.468–4.559) | -40.9% | 476 → 1622 ms | 13.61 (12.27–14.93) → 7.21 (7.04–7.46) tok/s, -47.0% |
| code | 512 | 7.377 tok/s (7.372–7.380) | 4.505 tok/s (4.457–4.513) | -38.9% | 320 → 1430 ms | 24.11 (23.96–24.39) → 9.33 (9.30–9.42) tok/s, -61.3% |
| chat | 64 | 7.412 tok/s (7.311–7.483) | 4.448 tok/s (4.355–4.495) | -40.0% | 471 → 1434 ms | 13.00 (11.71–14.34) → 6.98 (6.86–7.09) tok/s, -46.2% |
| chat | 512 | 7.308 tok/s (7.170–7.412) | 4.390 tok/s (4.352–4.410) | -39.9% | 310 → 1392 ms | 24.41 (24.02–24.44) → 9.32 (9.25–9.38) tok/s, -61.8% |

Every run on the GGUF is slower than every run on the model's own format, in decode after the 512-token prompt and in
the 512-token prompt, on the three texts and in the three modes (the 18 manifests' outcome: regression; their
hypothesis, that the GGUF runs as fast, does not hold). In fast mode the GGUF decodes 21.5–25.6% slower, reads the
prompts 29–50% slower and takes 0.7–0.9 s more for the first step after a prompt; with fast and guessed experts it
decodes 15.0–21.5% slower; in exact mode, whose kernels for the file's blocks give each output a thread that adds in
the CPU engine's order, 38.9–41.6% slower, and a 512-token prompt takes 2.6 times as long. At these defaults the two
files do not read a fast prompt the same way: the model's own format reads it layer by layer and gives part of its
products and experts to the CPU's matrix unit (`QWEN36_CPU_MATMUL`), while the GGUF, one slot short of the
layer-by-layer prompt (one slot more made its 512-token prompts 15–31% faster, above), loads its experts eight at a
time on the GPU alone; in exact mode both read a prompt the GGUF's way. The medians of the peak memory footprint were
1094 MiB on the GGUF against 985–988 MiB on the model's own format in fast mode and 1169 against 999 MiB in exact
mode, and those of the time to a ready model 0.17–0.21 s against 0.24–0.29 s. No page was written to swap during the
runs, except 12 (16 KB each) during those of fast mode with fast and guessed experts on prose.

Where the time goes, from timed runs, two per file and text ([diagnosis.txt](diagnosis.txt): `COLI_TIMERS=1`, fast
mode, the 512-token prompt and 32 steps, and again with one step; the phases, not a speed). Steps 2 to 32 are the 32
steps less the first, whose phases come from the run with one step: the two runs' prompts took the same DeltaNet,
attention and MoE times within 3.1%, but the first step took longer in the runs with one step, 755–822 against 280–619
ms on the model's own format and 1352–1575 against 1227–1382 ms on the GGUF. The phases add the GPU's time of each and
the wall time spent waiting for the experts read from disk; during that wait the GPU also runs the shared expert and,
untimed, the routed experts the cache held. In decode steps 2 to 32 the phases of a step added up to 32–43 ms more on
the GGUF (prose 103.4 → 135.2 ms, code 123.1 → 165.7, chat 131.5 → 174.0; the same runs' measured steps 110.1 → 141.9,
128.9 → 173.6 and 138.4 → 183.6 ms), and the wait grew by 69–74% of that, 65–74% of the measured difference
(46.5 → 70.0, 64.3 → 95.1 and 72.5 → 101.9 ms a step). The wait grew 40–50% while the decode read 1.6–6.1% more
experts (the GGUF's cache holds 263 slots against 303), each 14% larger (1,900,544 bytes, or 2,039,808 in the three
layers whose down is Q6_K, against 1,671,168 for a record of `experts.qc`). A GGUF expert is also three reads from
three places in the file (its gate, up and down are three tensors), a record one: a likely cause of the rest, not
measured apart. The GPU time of the routed experts computed after the wait grew from 9.0–11.6 to 11.9–16.6 ms a step,
the router's from 2.2 to 4.4–4.7 (the file keeps the routers in F32), the DeltaNet's by 1.9–2.5 ms. The GGUF's extra
time in the first step after a prompt is mostly outside the phases, and not diagnosed: in the runs with one step the
first step's phases took 289–327 ms on the GGUF and 190–196 ms on the model's own format, of 1352–1575 and 755–822 ms
(in the speed runs above the first step after the 512-token prompt took medians of 972–1171 and 237–285 ms). In the
512-token prompt every phase but lm_head (one row, 8–10 ms on both) took longer on the GGUF (prose: DeltaNet 1236
against 226 ms, attention 289 against 70, the MoE with its wait for the disk 3882 against 2607), but the two files do
not read a fast prompt the same way (above), so the phases do not compare one to one.

## Correctness

The exact graph on the GGUF against the CPU engine's exact mode on the same file (`COLI_METAL=0 GQ_EXACT=1`), with
this change's harness, one run each ([correctness.txt](correctness.txt), the per-word files in `correctness/`):

- 32 words chosen by the model after a 64-token prompt: the same words on the three texts; the 248,320 logits after
  the last word within 2.4e-5 to 2.8e-5 of each other, about 1.3e-6 of the largest logit, with the same ten most
  likely tokens in the same order.
- 256 teacher-forced words after a 64-token prompt: each word's negative log-likelihood within 3.5e-5 of the CPU
  engine's (the files have six decimals), the same first choice for every word; mean negative log-likelihood
  3.018107 / 3.018106 (prose), 2.413106 / 2.413106 (code), 1.376022 / 1.376022 (chat).

The CPU engine's peak memory footprint was 3,788–3,923 MiB against the exact graph's 979–990 MiB, and it decoded
0.61–0.76 tok/s against the exact graph's 4.45–4.89 (single runs, not the speed runs above). The graph's footprint
does not count the 1.91 GB of dense weights it reads from the mapped file (`dense=mapped` in its log).

## Quality

Each text read with its true tokens, 1024 of them after a 512-token prompt (prose, code) or a 64-token one (chat), one
run per variant (the same binary and settings give the same numbers in every run), compared token by token with
`compare_nll.py` ([quality.txt](quality.txt), the per-token files in `quality/`): the change of the mean negative
log-likelihood with its standard error, and of the share of tokens the model ranks first, in points. A change under
twice its standard error is within the noise.

Fast mode against exact mode on each file: what fast mode costs.

| text | own format: mean NLL | own format: top-1 | GGUF: mean NLL | GGUF: top-1 |
|---|---|---|---|---|
| prose | −0.04% ± 0.06% (within the noise) | +0.29 ± 0.26 points (within the noise) | −0.10% ± 0.07% (within the noise) | +0.39 ± 0.24 points (within the noise) |
| code | +0.07% ± 0.07% (within the noise) | +0.39 ± 0.20 points (2.00 standard errors) | +0.24% ± 0.13% (within the noise) | +0.00 ± 0.28 points (within the noise) |
| chat | −0.04% ± 0.06% (within the noise) | +0.20 ± 0.14 points (within the noise) | +0.06% ± 0.07% (within the noise) | −0.39 ± 0.20 points (2.00 standard errors) |

Exact mode's mean negative log-likelihood: 2.67841, 2.41721 and 1.32548 on the model's own format, 2.65682, 2.45180
and 1.31565 on the GGUF (prose, code, chat). On both files a few tokens move more than the rest: on the GGUF the
largest single change is 2.67 nats (a code token the exact run gives probability 0.994), on the model's own format
0.83; 72, 46 and 19 tokens move by more than 0.1 on the GGUF, 63, 41 and 13 on the model's own format.

The GGUF against the model's own format, the same tokens: two quantizations of one checkpoint, not a change of the
engine.

| text | exact mode: mean NLL | exact mode: top-1 | fast mode: mean NLL | fast mode: top-1 |
|---|---|---|---|---|
| prose | −0.81% ± 0.42% (within the noise) | +0.29 ± 0.67 points (within the noise) | −0.86% ± 0.41% (2.12 standard errors) | +0.39 ± 0.68 points (within the noise) |
| code | +1.43% ± 0.74% (within the noise) | −0.39 ± 0.60 points (within the noise) | +1.61% ± 0.74% (2.17 standard errors) | −0.78 ± 0.60 points (within the noise) |
| chat | −0.74% ± 0.79% (within the noise) | +0.20 ± 0.72 points (within the noise) | −0.64% ± 0.78% (within the noise) | −0.39 ± 0.72 points (within the noise) |
