# Six guessed experts announced instead of eight (qwen36, Metal graph), measured on an M1

The change before (in decode the Metal graph announces to the kernel the next layer's eight guessed experts that are
not in the cache) against this change (only the six likeliest of them). The runs used a367db0c and b16c23ca, local
commits with those changes on dev 6cf80dbc with the Metal kernels and the expert cache merged; this branch holds the
same changes to the code, patch for patch, on a newer dev. The harness `bench.c` built against each commit with the
command in the manifests. An M1 MacBook Air with 8 GB (no fan), macOS 26.6.2, Qwen3.6-35B-A3B in Colibri's qwen36
container with `canonical.qc` and `experts.qc` beside it, the Metal graph's default expert cache (512 MiB, 303 slots).

## Speed

Three texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a, of which
the harness reads the first 20,480 bytes), code ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`, first
20,480 bytes read) and chat ([chat.txt](chat.txt): an English conversation written for these measurements, plain
User:/Assistant: text, read whole). Of each, the first 64 and the first 512 tokens, each from an empty state, then 32
greedy decode steps. One warm-up run, then 4 runs per version in the order parent, this, this, parent, parent, this,
this, parent, each a new process after the nominal thermal state and a 60 s rest. Decode speed is steps 2 to 32 (the
first step after the prompt is timed apart, in the first-step column). Medians, min–max in brackets; every run's raw
log in `baseline-<text>.txt` and `trial-<text>.txt`, the summaries in `summary-<text>.txt`, one manifest per text
(decode after the 512-token prompt: prose no-change, code improvement, chat improvement).

| text | prompt | decode, eight | decode, six | change | first step, eight → six | prompt change |
|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 9.214 tok/s (9.155–9.280) | 9.213 tok/s (9.207–9.271) | −0.0% | 265 → 252 ms | +0.8% |
|  | 512 | 9.559 tok/s (9.420–9.574) | 9.502 tok/s (9.469–9.556) | −0.6% | 269 → 261 ms | −1.1% |
| code ([code.txt](code.txt)) | 64 | 7.858 tok/s (7.822–7.918) | 8.109 tok/s (7.966–8.154) | +3.2% | 307 → 301 ms | −0.6% |
|  | 512 | 7.314 tok/s (7.314–7.317) | 7.712 tok/s (7.583–7.749) | +5.4% | 323 → 308 ms | −0.0% |
| chat ([chat.txt](chat.txt)) | 64 | 7.725 tok/s (7.669–7.802) | 7.903 tok/s (7.659–7.938) | +2.3% | 310 → 328 ms | −1.4% |
|  | 512 | 7.617 tok/s (7.597–7.626) | 7.897 tok/s (7.744–7.906) | +3.7% | 285 → 266 ms | −2.5% |

The prompt does not use the guesses (its experts are read layer by layer); its medians moved by −2.5% to +0.8%, within
the runs' spread except chat's 512-token prompt, where every run of six was slower than every run of eight. The logits
and the generated tokens are the same in every run of both versions. In both versions the model was ready in 0.14–0.19
s and the peak memory footprint (`/usr/bin/time -l`) was 983–986 MiB; no page was written to swap during the runs.

Why six: [guesses.txt](guesses.txt), exploratory runs of development builds, one run each.
