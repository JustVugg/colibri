# A GGUF's dense matrices kept in memory through a prompt, measured on an M1

The change before (08293922) against this change (627409db), which locks a GGUF's dense matrices in memory (`mlock`)
from the start of a prompt to the end of the decode step after it: the speed in fast and exact mode, the memory, and
the diagnosis runs that found why a GGUF's first decode step after a prompt was slow and what the lock does to the
decode after it. One harness, `bench.c`, built with the command in the manifests against the code as measured, local
commits of 19 and 20 October on upstream dev e68e0cc9 and this series as it was then. This branch holds both changes,
patch for patch, on a newer dev and on the series as it is now. An M1 MacBook Air with 8 GB (no fan), macOS 26.6.2.
The file: unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.1 GB: the dense matrices in Q8_0, the head in Q6_K, the
routers and the DeltaNet's alpha and beta projections in F32, the routed experts' gate and up in Q4_K and their down
in Q5_K, or Q6_K in three layers), in the directory `coli import --mode link` makes for it (`<GGUF model>` in the
logs).

## How it was measured

- Texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a), code
  ([code.txt](../qwen36-gguf-metal-2026-10-19/code.txt): an earlier version of `c/qwen36_graph.h`) and chat
  ([chat.txt](../qwen36-gguf-metal-2026-10-19/chat.txt): an English conversation written for these measurements),
  the texts of [the GGUF record](../qwen36-gguf-metal-2026-10-19/README.md). Each manifest's `prompt_hash` is its
  text's SHA-256.
- The harness `bench.c` is that record's: it includes `qwen36.c`, opens a GGUF directory as the engine's `main`
  does, reads each prompt from an empty state, then runs the decode steps with `q36_spec_step` and hashes every logit
  (FNV-1a) and the tokens; `TEACHER=1` feeds each step the text's next token.
- Build: the command in the manifests, from `git archive` of each commit (`bin-08293922/` for the change before,
  `bin-627409db/` for this change) with this record's `bench.c`. The launchers in the commands are one-line scripts
  that set the model and the variables of a variant and run one of the binaries: `gguf-08293922` (the change before)
  and `gguf` (this change) set `SNAP=<GGUF model>`, `gguf-08293922-exact` and `gguf-exact` also
  `QWEN36_METAL_EXACT=1`.
- Speed runs: the first 64 and 512 tokens of each text, each from an empty state, then 32 decode steps, each fed the
  text's own next token; a warm-up run of the baseline (both versions read the same file), then 4 runs per version in
  the order baseline, trial, trial, baseline, baseline, trial, trial, baseline, each a new process. Decode speed is
  steps 2 to 32; the first step after the prompt is timed apart. The first new word comes from the prompt's logits, at
  the prompt's end; the time to the second new word is the prompt's time plus that step's, and to the 32nd that and 30
  steps at the decode speed. Before every run the script waited for the nominal thermal state, rested 60 s and read
  the swap counters; each run's times, thermal state, counters, engine log and `/usr/bin/time -l` output are in the
  raw logs (`baseline-<mode>-<text>.txt`, `trial-<mode>-<text>.txt`), the summaries in `summary-<mode>-<text>.txt`,
  the manifests in `manifest-<mode>[-first|-prompt]-<text>.json` (decode, the first decode step as tokens per second,
  or the prompt's speed, after the 512-token prompt, or after the 32-token turn in the turn runs; the outcome compares
  every run of one version with every run of the other). The turn runs (`turn` in the file names, prose and chat) read
  the first 64 tokens from an empty state, then, from the state the harness saved after them (`PHOTO=1`), the 32
  tokens after (a short turn of a chat), each followed by 32 decode steps as above.
- Diagnosis runs ([diagnosis.txt](diagnosis.txt)): the harness with lines that say how many pages of the file's dense
  matrices are in memory (`mincore`), the system's page-ins and the experts read in each phase
  ([bench_diag.c](bench_diag.c)), twelve sets of runs, sets 3 to 12 in alternating order, every run after the nominal
  thermal state and a 60 s rest; from set 9 with a sample of the system's memory every second.

## Fast mode

The same file and the same defaults (the expert cache of a sixteenth of the RAM, 263 slots of 2.04 MB), the harness
of 08293922 against this change's:

| text | prompt | prompt speed | first decode step | decode |
|---|---|---|---|---|
| prose | 64 | 15.20 (13.75–15.74) → 18.36 (18.25–18.63) tok/s, +20.8% | 971 (799–1032) → 366 (365–378) ms | 6.915 (6.907–7.051) → 6.620 (6.585–6.702) tok/s, -4.3% |
| prose | 512 | 66.76 (63.01–69.63) → 70.12 (68.58–71.76) tok/s, +5.0% | 882 (650–1182) → 303 (288–315) ms | 7.418 (7.406–7.454) → 7.105 (7.029–7.227) tok/s, -4.2% |
| code | 64 | 13.00 (12.72–13.42) → 16.64 (16.27–16.93) tok/s, +28.0% | 777 (730–858) → 383 (373–388) ms | 6.093 (6.007–6.273) → 5.869 (5.763–5.977) tok/s, -3.7% |
| code | 512 | 59.88 (58.46–61.79) → 63.35 (61.44–64.02) tok/s, +5.8% | 1172 (1013–1258) → 330 (327–342) ms | 6.069 (5.994–6.191) → 5.985 (5.724–6.040) tok/s, -1.4% |
| chat | 64 | 12.80 (12.14–14.53) → 15.57 (15.41–15.70) tok/s, +21.6% | 1001 (935–1074) → 377 (375–383) ms | 5.986 (5.972–6.121) → 5.724 (5.660–5.860) tok/s, -4.4% |
| chat | 512 | 62.56 (60.35–64.69) → 65.09 (64.10–66.14) tok/s, +4.1% | 1074 (882–1135) → 313 (307–321) ms | 5.870 (5.700–5.899) → 5.741 (5.674–5.819) tok/s, -2.2% |

The time to the second new word (the prompt and the first decode step) and to the 32nd (each run's second new word
and 30 steps at its decode speed); the first new word comes from the prompt's logits, at the prompt's end:

| text | prompt | to the second new word | to the 32nd new word |
|---|---|---|---|
| prose | 64 | 5.20 (5.06–5.46) → 3.85 (3.80–3.88) s, -25.9% | 9.54 (9.40–9.71) → 8.37 (8.31–8.44) s, -1.17 s |
| prose | 512 | 8.59 (8.47–8.78) → 7.61 (7.42–7.77) s, -11.4% | 12.64 (12.50–12.82) → 11.83 (11.57–12.04) s, -0.81 s |
| code | 64 | 5.72 (5.58–5.77) → 4.23 (4.15–4.31) s, -25.9% | 10.60 (10.45–10.77) → 9.35 (9.17–9.52) s, -1.25 s |
| code | 512 | 9.72 (9.54–9.77) → 8.41 (8.33–8.68) s, -13.5% | 14.61 (14.53–14.76) → 13.41 (13.32–13.92) s, -1.19 s |
| chat | 64 | 6.06 (5.37–6.21) → 4.49 (4.46–4.53) s, -25.9% | 11.07 (10.27–11.22) → 9.74 (9.62–9.78) s, -1.34 s |
| chat | 512 | 9.24 (9.05–9.41) → 8.18 (8.06–8.30) s, -11.5% | 14.34 (14.16–14.67) → 13.40 (13.22–13.58) s, -0.94 s |

Every run of this change is faster than every run of the change before in the first decode step (2.0 to 3.6 times in
the medians) and to the second and the 32nd new word, on the three texts and both lengths (the first-step manifests:
improvement), and in the 64-token prompt; the 512-token prompt is 4.1 to 5.8% faster in the medians, with the runs
overlapping (the prompt manifests: no-change). The first new word, the prompt's, comes 0.32 to 1.08 s sooner in the
medians (3.9 to 22%), every run sooner after the 64-token prompt. The decode after the prompt is 1.4% to 4.4% slower
in the medians; after the three 64-token prompts and the 512-token one on prose, every run of this change is slower
than every run of the change before (the decode manifests: regression on prose, no-change on code and chat; see [The
decode after a locked prompt](#the-decode-after-a-locked-prompt)). The logits are the same, bit for bit, in every run
of both versions (the same fingerprints), and no page was written to swap during the runs.

## Exact mode

`QWEN36_METAL_EXACT=1`, otherwise the same:

| text | prompt | prompt speed | first decode step | decode |
|---|---|---|---|---|
| prose | 64 | 7.77 (7.66–7.80) → 8.07 (8.05–8.13) tok/s, +3.9% | 1526 (1356–1567) → 475 (460–486) ms | 4.720 (4.711–4.781) → 4.761 (4.717–4.780) tok/s, +0.9% |
| prose | 512 | 9.38 (9.30–9.47) → 9.44 (9.43–9.45) tok/s, +0.6% | 1443 (1182–1596) → 420 (416–423) ms | 4.871 (4.836–4.881) → 4.853 (4.847–4.883) tok/s, -0.4% |
| code | 64 | 7.46 (7.11–7.71) → 7.99 (7.93–8.01) tok/s, +7.1% | 1463 (955–1513) → 480 (476–481) ms | 4.547 (4.458–4.602) → 4.557 (4.467–4.581) tok/s, +0.2% |
| code | 512 | 9.37 (9.27–9.41) → 9.39 (9.38–9.40) tok/s, +0.2% | 1451 (1436–1554) → 435 (428–444) ms | 4.476 (4.421–4.524) → 4.454 (4.450–4.468) tok/s, -0.5% |
| chat | 64 | 6.93 (6.88–7.59) → 7.87 (7.82–7.87) tok/s, +13.5% | 1007 (935–1351) → 472 (467–472) ms | 4.446 (4.425–4.519) → 4.401 (4.379–4.452) tok/s, -1.0% |
| chat | 512 | 9.38 (9.32–9.42) → 9.40 (9.38–9.42) tok/s, +0.2% | 1398 (1363–1429) → 429 (424–435) ms | 4.367 (4.356–4.374) → 4.366 (4.301–4.388) tok/s, -0.0% |

| text | prompt | to the second new word | to the 32nd new word |
|---|---|---|---|
| prose | 64 | 9.76 (9.71–9.79) → 8.40 (8.34–8.43) s, -14.0% | 16.11 (15.99–16.15) → 14.69 (14.63–14.79) s, -1.42 s |
| prose | 512 | 55.94 (55.47–56.53) → 54.65 (54.59–54.73) s, -2.3% | 62.11 (61.63–62.69) → 60.82 (60.77–60.92) s, -1.30 s |
| code | 64 | 9.90 (9.81–10.28) → 8.49 (8.47–8.55) s, -14.2% | 16.51 (16.44–16.87) → 15.08 (15.02–15.27) s, -1.43 s |
| code | 512 | 56.13 (55.84–56.67) → 54.99 (54.87–55.02) s, -2.0% | 62.82 (62.49–63.46) → 61.71 (61.61–61.76) s, -1.11 s |
| chat | 64 | 10.22 (9.79–10.30) → 8.61 (8.60–8.66) s, -15.7% | 16.91 (16.53–17.07) → 15.42 (15.35–15.51) s, -1.50 s |
| chat | 512 | 55.98 (55.76–56.31) → 54.91 (54.76–54.99) s, -1.9% | 62.86 (62.62–63.17) → 61.77 (61.61–61.96) s, -1.09 s |

The exact kernels read a prompt more slowly (a 512-token prompt in about 54 s), so the time saved in the first decode
step is a smaller share of the time to the second new word: 1.4 to 1.6 s sooner after 64 tokens (14 to 16%), 1.1 to
1.3 s after 512 (1.9 to 2.3%). Every run of this change is faster than every run of the change before in the first
decode step (2.1 to 3.4 times in the medians) and to the second and the 32nd new word, on the three texts and both
lengths; the first new word, the prompt's, comes 0.3 to 1.1 s sooner after 64 tokens (4 to 12%, every run) and 0.1 to
0.3 s after 512 (0.2 to 0.6%; there the runs overlap). The 64-token prompt is 4 to 14% faster, the 512-token one
within 1% (the prompt manifests: no-change), and the decode 1.0% slower to 0.9% faster in the medians (the decode
manifests: no-change), with the runs of the two versions overlapping on every text. The logits are the same, bit for
bit, in every run of both versions, and no page was written to swap.

## A short turn resumed from a saved state

A prompt of 32 tokens added to the state saved after 64 (the harness's `PHOTO=1`), fast mode, prose and chat: the lock
is taken for every prompt, so a short turn pays its `mlock` and `munlock` (about 0.1 s) too.

| text | prompt | prompt speed | first decode step | decode |
|---|---|---|---|---|
| prose | 64 | 16.55 (14.50–16.90) → 18.81 (18.66–18.91) tok/s, +13.7% | 1116 (756–1175) → 381 (378–400) ms | 6.973 (6.940–7.015) → 6.798 (6.758–6.850) tok/s, -2.5% |
| prose | 96 | 12.48 (11.57–13.28) → 14.15 (14.07–14.24) tok/s, +13.4% | 382 (349–605) → 258 (233–265) ms | 7.429 (7.354–7.498) → 7.283 (7.260–7.322) tok/s, -2.0% |
| chat | 64 | 12.71 (12.55–13.11) → 15.54 (15.27–15.71) tok/s, +22.2% | 1041 (960–1125) → 395 (388–413) ms | 5.835 (5.812–5.894) → 5.585 (5.486–5.687) tok/s, -4.3% |
| chat | 96 | 10.59 (8.95–10.95) → 11.61 (11.55–12.23) tok/s, +9.6% | 478 (422–925) → 310 (297–316) ms | 6.124 (5.981–6.252) → 5.835 (5.788–5.958) tok/s, -4.7% |

| text | prompt | to the second new word | to the 32nd new word |
|---|---|---|---|
| prose | 64 | 5.02 (4.89–5.17) → 3.79 (3.76–3.81) s, -24.5% | 9.34 (9.16–9.45) → 8.19 (8.16–8.25) s, -1.15 s |
| prose | 96 | 3.07 (2.76–3.14) → 2.52 (2.49–2.53) s, -18.0% | 7.13 (6.76–7.18) → 6.64 (6.59–6.66) s, -0.49 s |
| chat | 64 | 6.04 (5.99–6.15) → 4.51 (4.49–4.58) s, -25.2% | 11.19 (11.13–11.24) → 9.91 (9.81–9.96) s, -1.28 s |
| chat | 96 | 3.73 (3.38–4.00) → 3.06 (2.93–3.08) s, -17.9% | 8.64 (8.21–8.97) → 8.20 (7.96–8.27) s, -0.43 s |

On the 32-token turn, every run of this change is faster than every run of the change before in the prompt (13.4 and
9.6% in the medians: the first new word 0.31 and 0.27 s sooner), in the first decode step (about 1.5 times: 382 and
478 ms without the lock, since a 32-token prompt's expert reads push part of the dense matrices out too, against 258
and 310 ms with it) and to the second new word (0.55 and 0.67 s sooner, 18% on both texts); the 32nd new word comes
0.49 and 0.43 s sooner in the medians, every run of this change first on prose, with the runs overlapping on chat.
The decode after the turn is 2.0% and 4.7% slower, every run of this change slower than every run of the change
before (the manifests after the turn: prompt and first step improvement, decode regression). So the lock pays on a
short turn too, and its lead would last about 200 and 80 decode steps at the median rates. The 64-token prompt of
these runs, from an empty state, is the speed runs' and gave the same picture.

## Memory

The medians of the speed runs: the peak memory footprint and the maximum resident set size of `/usr/bin/time -l`, in
MiB, the engine's time to a ready model, and the pages the system wrote to swap during all the runs of the text:

| text, mode | peak memory footprint | resident size | model ready | pages written to swap |
|---|---|---|---|---|
| prose, fast | 1094 → 1097 | 763 → 2542 | 0.23 → 0.22 s | 0 |
| code, fast | 1094 → 1097 | 762 → 2535 | 0.26 → 0.21 s | 0 |
| chat, fast | 1094 → 1097 | 775 → 2683 | 0.23 → 0.21 s | 0 |
| prose, exact | 1169 → 1171 | 827 → 2589 | 0.23 → 0.24 s | 0 |
| code, exact | 1169 → 1171 | 827 → 2609 | 0.24 → 0.22 s | 0 |
| chat, exact | 1169 → 1171 | 844 → 2650 | 0.20 → 0.18 s | 0 |
| prose, turn | 1050 → 1053 | 724 → 2643 | 0.17 → 0.18 s | 0 |
| chat, turn | 1050 → 1053 | 734 → 2692 | 0.18 → 0.18 s | 0 |

- The resident size is 1762 to 1958 MiB larger (724 to 844 MiB → 2535 to 2692 MiB in the medians), but not the memory
  in use: `mlock` maps the pages of the dense matrices (1914 MB) into the process, and the resident size counts them
  from then on. With the change before only the GPU read them, and the resident size did not count them: in the
  diagnosis runs of sets 9 to 11, 0.52 to 0.72 GB during its decodes, which had all of them in memory (set 10),
  against 2.16 to 2.50 GB with the lock taken in the graph's own mapping. They are pages of the file, which the peak
  memory footprint does not count: it is the same (2 to 3 MiB more). What the lock holds shows in the wired memory.
- The system's wired memory ([memory.txt](memory.txt): `vm_stat` every 0.5 s, one run of each version twice, prose
  64,512 32, fast mode, with the harness's lines timed) is larger during the prompts and the same during the decodes:
  the medians of the samples during the prompts were 1.77 to 1.86 GB with the change before and 3.27 to 3.49 GB with
  this change, about 1.4 to 1.5 GB more during the 64-token prompt and 1.7 GB during the 512-token one (the GPU holds
  part of the dense matrices during a prompt anyway); during the decodes 3.91 to 3.92 GB with the change before and
  3.91 to 3.93 GB with this change, since the GPU holds them all then, and the largest samples were 4.03 and 4.05 GB
  with the change before, 4.02 and 3.98 GB with this change.

## The decode after a locked prompt

The lock changes nothing that is computed, and the decode after a locked prompt reads the same experts (the counts
of the experts read are the same in every run of both versions), but it found fewer of them in the system's file
cache in most sets of runs, and read more of them from disk. The diagnosis sets on prose, with the decode steps
after the first:

| set | the lock | decode steps | prompt | decode, unlocked (tok/s) | decode, locked (tok/s) | change | read from disk: unlocked → locked |
|---|---|---|---|---|---|---|---|
| 7 | this change's code | 2–32 | 64 | 7.081 (7.018–7.272) | 6.979 (6.783–7.014) | -1.4% | 6.58 → 6.96 GB (+5.9%), steps 2–32 |
| 7 | this change's code | 2–32 | 512 | 7.281 (7.165–7.419) | 7.136 (7.102–7.225) | -2.0% | 5.87 → 6.16 GB (+4.9%), steps 2–32 |
| 8 | this change's code | 2–160 | 64 | 7.408 (7.209–7.463) | 7.155 (7.059–7.188) | -3.4% | 23.03 → 24.91 GB (+8.2%), steps 33–160 |
| 8 | this change's code | 2–160 | 512 | 7.373 (7.325–7.383) | 7.265 (7.165–7.294) | -1.5% | 23.96 → 24.85 GB (+3.7%), steps 33–160 |
| 9 | as this change, by the harness | 2–160 | 64 | 7.428 (7.161–7.537) | 7.181 (6.992–7.201) | -3.3% | 22.88 → 24.50 GB (+7.1%), steps 33–160 |
| 9 | through a second mapping | 2–160 | 64 | 7.428 (7.161–7.537) | 7.156 (7.118–7.223) | -3.7% | 22.88 → 24.60 GB (+7.5%), steps 33–160 |
| 9 | as this change, by the harness | 2–160 | 512 | 7.433 (7.351–7.479) | 7.253 (7.129–7.422) | -2.4% | 23.38 → 24.73 GB (+5.8%), steps 33–160 |
| 9 | through a second mapping | 2–160 | 512 | 7.433 (7.351–7.479) | 7.250 (7.240–7.357) | -2.4% | 23.38 → 24.67 GB (+5.5%), steps 33–160 |
| 10 | the harness's lock | 2–160 | 64 | 7.274 (7.253–7.294) | 6.975 (6.944–7.006) | -4.1% | 23.05 → 25.09 GB (+8.9%), steps 33–160 |
| 10 | the harness's lock | 2–160 | 512 | 7.201 (7.185–7.217) | 7.183 (7.128–7.237) | -0.3% | 24.55 → 24.66 GB (+0.5%), steps 33–160 |
| 11 | the harness's lock | 2–160 | 64 | 7.130 (7.048–7.198) | 7.040 (6.998–7.191) | -1.3% | 23.80 → 24.74 GB (+4.0%), steps 33–160 |
| 11 | the harness's lock | 2–160 | 512 | 7.072 (6.977–7.144) | 7.085 (7.062–7.179) | +0.2% | 24.73 → 24.86 GB (+0.5%), steps 33–160 |
| 12 | the harness's lock | 2–160 | 64 | 7.306 (7.225–7.356) | 7.120 (7.042–7.226) | -2.5% | 24.24 → 24.68 GB (+1.8%), steps 33–160 |
| 12 | the harness's lock, to the end of the decode | 2–160 | 64 | 7.306 (7.225–7.356) | 7.108 (7.063–7.154) | -2.7% | 24.24 → 24.56 GB (+1.3%), steps 33–160 |
| 12 | no lock, the dense tensors in a file of their own | 2–160 | 64 | 7.306 (7.225–7.356) | 7.359 (7.250–7.445) | +0.7% | 24.24 → 23.33 GB (-3.7%), steps 33–160 |
| 12 | the harness's lock | 2–160 | 512 | 7.353 (7.240–7.437) | 7.240 (7.221–7.277) | -1.5% | 24.48 → 24.99 GB (+2.1%), steps 33–160 |
| 12 | the harness's lock, to the end of the decode | 2–160 | 512 | 7.353 (7.240–7.437) | 7.218 (7.194–7.290) | -1.8% | 24.48 → 24.88 GB (+1.6%), steps 33–160 |
| 12 | no lock, the dense tensors in a file of their own | 2–160 | 512 | 7.353 (7.240–7.437) | 7.429 (7.357–7.490) | +1.0% | 24.48 → 23.47 GB (-4.1%), steps 33–160 |

Medians of four runs of each (two in set 10), min–max in brackets. In sets 7 and 8 the lock is this change's code at
its earlier revision (the same lock and unlock after the first step; the unlock at a request's end and the counts in
the report came after, and neither changes what these runs do); in sets 9 to 12 the change before's code with the
lock taken and released by the harness at the same points, outside its timed regions (about 0.1 s per prompt not
counted). Read from disk: the system's page-ins during those decode steps, from step 33 on in the sets of 160 steps
(the system sometimes counted a first step's reads late, in the window after it).

- Not heat: the thermal state was nominal before and after every run (the script waited for it before each, and the
  logs give it after each). An 800 ms pause after the first step (set 7) made the decode slower, not faster, but it
  also lets the GPU's clocks drop (the engine keeps them up for 0.25 s after its last renewal), so it says nothing
  about heat by itself.
- Not this change's binary nor the graph's mapping: the change before's code with the same lock done by the harness
  (set 9, B) gave the same difference, and so did the lock done through a second mapping of the file, unmapped after
  the first decode step, so that the harness's own pages were not left mapped (set 9, C: its resident size as the
  change before's).
- Not the dense matrices themselves: their pages were all in memory at decode steps 32, 96 and 160, with and without
  the lock (set 10, 24 checks of 24). The extra reads are experts.
- The system's memory during the decode (the medians of the samples, [diagnosis.txt](diagnosis.txt)'s `[memory]`
  lines): across the runs of sets 9 to 11, 27 for each prompt length, the decode was faster the more pages the file
  cache held during it (correlations of 0.73 and 0.75, after the 64- and the 512-token prompt, between its file-backed
  pages and its speed, and of -0.88 and -0.89 between those pages and its reads from disk over steps 33 to 160).
  Within the change before's runs alone the link holds (0.86 and 0.86), within the locked runs it is weak (0.19 and
  0.49), and in set 11 alone there is none (0.27 and -0.01): the pooled value comes mostly from the change before's
  runs of sets 9 and 10, which had the most file cache and the fastest decodes. The three runs of each length with the
  most (2.14 to 2.23 GB of file-backed pages, against 1.96 to 2.13 GB in the others) were the change before's, with
  little anonymous memory (0.55 to 0.72 GB, against 0.65 to 0.90 GB); its other runs had about the memory of the
  locked ones (in set 11, 1.99 to 2.11 GB against 1.96 to 2.05 GB), and there the difference was within the noise. A
  reading that set 12 then refuted: that without the lock the first decode step's read of the pages pushed out makes
  the system compress other memory, leaving a larger file cache for the decode after it.
- The cost comes with the lock (set 12, four runs each): locked to the end of the decode instead (F), the decode was
  as slow as with this change's lock (2.7% and 1.8% slower than the change before's after 64 and 512 tokens, against
  2.5% and 1.5%), so the unlock before the decode is not the cause. With the dense tensors in a file of their own and
  no lock (K: the mapping taken from the sparse copy of set 4, the experts still read from the file), the first step
  after the 512-token prompt was fast too (316 ms; after the 64-token one 504 ms against 338 ms locked, since in two
  of its four runs the copy lost 340 and 365 MB in that prompt) and the decode as fast as the change before's or
  faster (0.7% and 1.0%), with 3.7% and 4.1% less read from disk: the first step's burst is not what helps the change
  before. K also maps the dense pages from another file, and a lock on that copy was not tried. With the pages locked
  through the prompt (B) or through the decode (F), the system kept a little less file cache during the decode, in
  the medians: file-backed pages 1.99 to 2.03 GB in B and F against 2.04 to 2.14 GB in A and K, the compressor 1.12
  to 1.13 GB against 1.15 to 1.19, the anonymous pages 0.82 to 0.85 GB against 0.65 to 0.78; the runs overlap (the
  compressor 1.10 to 1.16 against 1.10 to 1.30 GB, the anonymous pages 0.79 to 0.87 against 0.45 to 0.86). Why the
  lock has that effect was not found.
- When to unlock (set 11): locked for the prompt only (D), the decode was within the noise of the change before's, and
  the first decode step 20 to 33 ms slower than with this change's lock. Locked from the model's loading on and never
  unlocked (E), the decode was as fast as the change before's after the 64-token prompt and 1.9% faster after the
  512-token one (every run of E faster than every run of the change before), with no more file cache than the other
  locked runs, and after the 64-token prompt it read 3.8% more from disk than the change before yet decoded as fast:
  the file cache does not explain everything. But the same lock in sets 2, 5 and 6 (of 32 steps; one, two and four
  runs each) decoded 0.4% to 3.7% slower than the change before after the 512-token prompt and 0.5% slower to 2.8%
  faster after the 64-token one, so E's advantage is not established. E keeps the 1.9 GB locked while the engine waits
  for a request.
- Longer answers: the lead of the second new word is eaten by the slower decode. At the medians of the speed runs it
  would last about 160 to 570 decode steps in fast mode, 200 and 80 after the 32-token turn, and in exact mode about
  700 after the 64-token prompt on chat and 1100 and 1700 after the 512-token prompts on code and prose, where the
  decode is 0.4 to 1.0% slower (in the other three cases it is within 0.1% or faster, and the lead stays); in set 8
  (160 steps, this change's code at its earlier revision), 280 after 64 tokens and 680 after 512. Not measured beyond
  160 steps.

## Why the first decode step was slow

Diagnosis sets 1 to 6, prose, with the change before's code:

- Set 1: before the first decode step after a 512-token prompt in a new process, the model's own format had 711 MB
  of its dense matrices out of memory (those of layers 0 to 19 but 5 MB), the GGUF 1.17 GB (layers 0 to 29 and part
  of 30 to 39); the first step read them back (page-ins of 1305 and 2090 MB) and took 806 and 1534 ms, or 317 and 546
  ms with the pages asked for before it (`madvise` WILLNEED, outside its time). As in the speed runs (64, then 512
  tokens in one process), the model's own format's 512-token prompt pushed nothing out (a first step of 252 ms), the
  GGUF's 1.1 GB (layers 0 to 29 but 20 MB; a first step of 1087 ms). Over the diagnosis runs of the change before on
  the GGUF, 0.3 to 1.2 GB of the 1.9 GB were out after the 512-token prompt, and the first step after it took 0.7 to
  1.5 s.
- Not the windows: the model's own format with each of its matrices in a window of its own, as a GGUF's (set 2),
  lost about 80 MB in the 512-token prompt.
- Not the CPU's share of a prompt: the model's own format with `QWEN36_CPU_MATMUL=0`, its whole prompt on the GPU as
  a GGUF's, kept all of `canonical.qc` in memory (set 3: first steps after 512 tokens of 238 and 260 ms).
- The file: with the GGUF's mapping taken from a sparse copy of all its tensors but the routed experts, at the same
  offsets, and the experts still read from the file itself, the dense matrices stayed in memory (set 4: after the
  512-token prompt 45 and 2 MB out against 607 and 874 MB, first steps of 336 and 331 ms against 927 and 1089 ms),
  with the same logits.
- `MADV_RANDOM` on the mapping did not help (set 5: first steps of 1083 and 1159 ms against 881 and 939 ms).
- Locked (set 6, four runs each): locked from the model's loading, the first step after 512 tokens took 251 ms
  against 1019 ms, and the 512-token prompt 7.05 s against 7.69 s; locked for each prompt and unlocked before the
  first step, 275 ms and 7.20 s. In all the diagnosis runs, `mlock` of the 1914 MB took 46 to 93 ms when the pages
  were in memory, and 1.08 and 1.09 s in the two runs of set 12 that followed a run of K, which had read the dense
  tensors from the copy and left the file's pages out (the lock read them in, as the prompt would have); `munlock`
  took 46 to 57 ms, or 93 to 105 ms where it also unmapped the second mapping (set 9, C).

## Files

- `README.md`: this file.
- `manifest-<mode>-<text>.json`, `manifest-<mode>-first-<text>.json`, `manifest-<mode>-prompt-<text>.json`: the
  manifests, mode `fast`, `exact` or `turn` (after the 32-token turn; prose and chat), text `prose`, `code` or
  `chat`; they pass `python3 c/experiment_manifest.py`.
- `baseline-<mode>-<text>.txt`, `trial-<mode>-<text>.txt`: the raw logs of every run, the warm-up's included.
- `summary-<mode>-<text>.txt`: the summary the measurement script printed.
- `diagnosis.txt`: the diagnosis runs, their lines, the memory medians and the raw logs.
- `memory.txt`: the system's wired memory every 0.5 s during two runs of each version, with the times of the
  harness's lines.
- `bench.c`: the harness; `bench_diag.c`: the harness of the diagnosis runs; `dense-copy.patch` (set 4: the GGUF's
  mapping replaced by a sparse copy without the routed experts) and `window-per-matrix.patch` (set 2: each matrix of
  the model's own format in a window of its own), both against the change before's code.
