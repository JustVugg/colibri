# Fast experts for qwen36 (Metal graph, fast mode), measured on an M1

`QWEN36_FAST_EXPERTS=5` against unset, with the same binary: the harness `bench.c` built against f2aa1165, a local
commit with this change on dev 6cf80dbc with the Metal kernels and the expert cache merged (this branch holds the same
change to the code, patch for patch, on a newer dev), with the command in the manifests. An M1 MacBook Air with 8 GB
(no fan), macOS 26.6.2, Qwen3.6-35B-A3B in Colibri's qwen36 container with `canonical.qc` and `experts.qc` beside it,
the Metal graph's default expert cache (512 MiB, 303 slots).

## Speed

Three texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a, of which
the harness reads the first 20,480 bytes), code ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`, first
20,480 bytes read) and chat ([chat.txt](chat.txt): an English conversation written for these measurements, plain
User:/Assistant: text, read whole). Of each, the first 64 and the first 512 tokens, each from an empty state, then 32
decode steps, each fed the text's own next token (teacher forcing: both versions read the same tokens, so they compute
the same text). One warm-up run, then 4 runs per version in the order default, J = 5, J = 5, default, default, J = 5,
J = 5, default, each a new process after the nominal thermal state and a 60 s rest. Decode speed is steps 2 to 32 (the
first step after the prompt is timed apart). Medians, min–max in brackets; every run's raw log in
`baseline-<text>.txt` and `trial-<text>.txt`, the summaries in `summary-<text>.txt`, one manifest per text (decode
after the 512-token prompt: prose improvement, code improvement, chat improvement).

| text | prompt | decode, default | decode, J = 5 | change | first step, default → J = 5 | prompt change |
|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 9.058 tok/s (8.949–9.144) | 11.439 tok/s (11.375–11.474) | +26.3% | 258 → 216 ms | −0.3% |
|  | 512 | 9.514 tok/s (9.471–9.533) | 11.582 tok/s (11.438–11.650) | +21.7% | 252 → 215 ms | +0.5% |
| code ([code.txt](code.txt)) | 64 | 8.091 tok/s (7.894–8.193) | 10.643 tok/s (10.575–10.719) | +31.6% | 300 → 235 ms | +3.2% |
|  | 512 | 7.981 tok/s (7.888–8.054) | 10.228 tok/s (10.125–10.429) | +28.2% | 324 → 249 ms | +0.1% |
| chat ([chat.txt](chat.txt)) | 64 | 7.935 tok/s (7.817–8.020) | 10.203 tok/s (10.128–10.291) | +28.6% | 327 → 294 ms | −0.2% |
|  | 512 | 7.725 tok/s (7.575–7.792) | 10.069 tok/s (9.817–10.174) | +30.3% | 312 → 243 ms | −0.3% |

The prompt does not use the option (only decode does). In both versions the model was ready in 0.18–0.23 s and the
peak memory footprint (`/usr/bin/time -l`) was 983–986 MiB; a run of the default on code (v0-run1) wrote 27 MiB to
swap, and no other run wrote any.

## Quality

Each text read with its true tokens, 1024 of them after a 512-token prompt (prose, code) or a 64-token one (chat), one
run per version (the same binary and settings give the same numbers in every run), compared token by token with
`compare_nll.py` ([quality.txt](quality.txt), the per-token files in `quality/`): the change of the mean negative
log-likelihood with its standard error, the same change as perplexity, and the change of the share of tokens the model
ranks first, in points. A change under twice its standard error is within the noise. The decode speeds are of these
single runs of 1023 steps; experts read include the prompt's.

| text | decode, default | decode, J = 5 | experts read | mean NLL | perplexity | top-1 |
|---|---|---|---|---|---|---|
| prose | 9.339 tok/s | 11.799 tok/s | 169,388 → 93,061 | +1.21% ± 0.48% | +3.3% | −1.0 ± 0.7 points (within the noise) |
| code | 8.140 tok/s | 10.491 tok/s | 208,512 → 119,274 | −0.67% ± 0.39% (within the noise) | −1.6% | −0.7 ± 0.5 points (within the noise) |
| chat | 8.124 tok/s | 10.664 tok/s | 208,811 → 125,056 | +0.40% ± 0.96% (within the noise) | +0.5% | +0.8 ± 0.8 points (within the noise) |

Averaged over the three texts: mean NLL +0.31% ± 0.38%, top-1 −0.3 ± 0.4 points.

## Results unchanged without the option

Exact mode (`QWEN36_METAL_EXACT=1`), the 64- and 512-token prompts of `promessi_sposi.txt` and 32 greedy tokens: the
commit before (6914b80e) and f2aa1165 give the same fingerprints ([fingerprints.txt](fingerprints.txt)).