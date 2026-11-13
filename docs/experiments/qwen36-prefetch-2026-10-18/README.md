# Copies of the guessed experts for qwen36 (Metal graph, decode), measured on an M1

`QWEN36_PREFETCH=1` against unset, both with `QWEN36_FAST_EXPERTS=5 QWEN36_FAST_GUESSED=4` and the MTP head's drafts,
with the same binary: the harness `bench.c` built against 2a2c5bc6, a local commit with this change on dev 6cf80dbc
with the Metal kernels and the expert cache merged (this branch holds the same change to the code, patch for patch, on
a newer dev), with the command in the manifests. An M1 MacBook Air with 8 GB (no fan), macOS 26.6.2, Qwen3.6-35B-A3B
in Colibri's qwen36 container with `canonical.qc`, `experts.qc` and the checkpoint's MTP head
(`model-mtp.safetensors`) beside it, the Metal graph's default expert cache (512 MiB, 303 slots), and with the copies
their eight slots beside it (14.2 MB).

## Speed

Three texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a, of which
the harness reads the first 20,480 bytes), code ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`, first
20,480 bytes read) and chat ([chat.txt](chat.txt): an English conversation written for these measurements, plain
User:/Assistant: text, read whole). Of each, the first 64 and the first 512 tokens, each from an empty state, then 32
decode steps, each fed the text's own next token (teacher forcing). One warm-up run, then 4 runs per version in the
order without, with, with, without, without, with, with, without, each a new process after the nominal thermal state
and a 60 s rest. Decode speed counts all 32 steps, the first included (steps 2 to 32 in their own column; the first
step after a prompt is a verify when the head's first draft is sure enough, and a verify copies nothing). Medians,
min–max in brackets; every run's raw log in `baseline-<text>.txt` and `trial-<text>.txt`, the summaries in
`summary-<text>.txt`, one manifest per text (decode after the 512-token prompt: prose no-change, code no-change, chat
no-change).

| text | prompt | decode, without the copies | decode, with the copies | change | steps 2–32 | first step | prompt |
|---|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 11.707 tok/s (11.502–11.904) | 11.974 tok/s (11.972–12.233) | +2.3% | +1.4% | 256 → 224 ms | +0.5% |
|  | 512 | 11.560 tok/s (11.138–12.103) | 11.847 tok/s (11.779–12.073) | +2.5% | +2.5% | 339 → 337 ms | +0.3% |
| code ([code.txt](code.txt)) | 64 | 10.180 tok/s (9.988–10.382) | 10.416 tok/s (10.339–10.457) | +2.3% | +1.9% | 285 → 271 ms | −1.2% |
|  | 512 | 11.592 tok/s (11.451–11.794) | 11.607 tok/s (11.481–11.837) | +0.1% | +0.9% | 357 → 393 ms | +3.1% |
| chat ([chat.txt](chat.txt)) | 64 | 11.177 tok/s (11.021–11.448) | 11.391 tok/s (11.188–11.618) | +1.9% | +0.4% | 282 → 254 ms | −1.3% |
|  | 512 | 10.398 tok/s (9.966–10.603) | 10.584 tok/s (9.865–10.780) | +1.8% | +3.1% | 336 → 358 ms | +1.2% |

### Clean re-runs

The benchmark guide re-runs from clean a case that slows by more than 5% (rule 6): code, where the first step after
the 512-token prompt was 10.1% slower with the copies, though the decode of all 32 steps was not (the script that
picks the re-runs looks only at the prompt and the whole decode, so this one was started by hand). The re-runs, with
the same procedure (`baseline-<text>-rerun.txt`, `trial-<text>-rerun.txt`, `summary-<text>-rerun.txt`, one manifest
each, of the decode after the prompt where the case slowed: code after the 512-token prompt, no-change):

| text | prompt | decode, without the copies | decode, with the copies | change | steps 2–32 | first step | prompt |
|---|---|---|---|---|---|---|---|
| code ([code.txt](code.txt)) | 64 | 10.528 tok/s (10.491–10.613) | 10.643 tok/s (10.406–10.739) | +1.1% | +0.7% | 254 → 249 ms | −1.8% |
|  | 512 | 12.066 tok/s (11.877–12.144) | 11.953 tok/s (11.896–12.226) | −0.9% | −0.3% | 320 → 329 ms | −3.2% |

In the re-run the first step after the 512-token prompt took 320 → 329 ms (+2.7%), the two versions' runs overlapping
(311–360 and 318–350 ms): the 10.1% of the first measurement did not repeat at that size, though a small cost there is
not ruled out. The decode of all 32 steps changed by +1.1% and −0.9% after the two prompts, with overlapping ranges,
and no run of the re-run wrote to swap.

The output is the same: in each text the nine runs (the warm-up, four without the copies, four with them) give the
same fingerprints of every logit and the same tokens. The model was ready in 0.26–0.34 s, and the peak memory
footprint (`/usr/bin/time -l`) was 1120–1123 MiB without the copies and 1132–1136 MiB with them. Two runs wrote a few
pages to swap, where a measurement should write none (the summaries and raw logs say so): a run with the copies on
code (v1-run4) wrote 20 pages of 16 KiB (0.3 MiB): without it, decode changes by +2.2% and +1.0% after the 64- and
512-token prompts (+2.3% and +0.1% with it); the warm-up on chat wrote 24 pages of 16 KiB (0.4 MiB), and it is not in
the medians. No other run of this record wrote any.

## The copies

From the line at exit of each run with the copies (both prompts of a run and their decode steps; the range of the four
runs where they differ; each run's line in [copies.txt](copies.txt)):

| text | copied | taken by their layer | after waiting for the copy | computed early | not in the file cache | held by the cache | too late | failed |
|---|---|---|---|---|---|---|---|---|
| prose | 2,818–2,840 | 2,003–2,031 | 192–220 | 672–695 | 1,115–1,139 | 4,809 | 9–11 | 0 |
| code | 2,441–2,515 | 1,713–1,793 | 318–374 | 726–759 | 1,206–1,275 | 2,699–2,701 | 15–21 | 0 |
| chat | 3,257–3,328 | 2,307–2,377 | 464–494 | 987–1,021 | 1,285–1,360 | 2,381–2,386 | 17–23 | 0 |

## Results unchanged

In exact mode, with the drafts and without them, the copies leave the fingerprints of every logit and the tokens as
they are, and this commit without them gives those of the guessed-experts commit below it; in fast mode with fast and
guessed experts, this record's runs with and without the copies give the fingerprints of the guessed-experts record's
runs with its rule ([fingerprints.txt](fingerprints.txt)).
