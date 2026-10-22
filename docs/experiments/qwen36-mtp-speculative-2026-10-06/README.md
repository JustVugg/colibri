# Speculative decoding with the MTP head for qwen36 (Metal graph), measured on an M1

The checkpoint's MTP head against `QWEN36_MTP=0`, with the same binary: the harness `bench.c` built with the command
in the manifests against c8fc51fc, a local commit with this change on dev 6cf80dbc with the Metal kernels and the
expert cache merged (this branch holds the same change to the code, patch for patch, on a newer dev). An M1 MacBook
Air with 8 GB (no fan), macOS 26.6.2, Qwen3.6-35B-A3B in Colibri's qwen36 container with `canonical.qc` and
`experts.qc` beside it, and the head in `model-mtp.safetensors` (in a directory of links to the model's files,
`<model+head>` in the logs); the Metal graph's default expert cache (512 MiB, 303 slots), which the head's experts
share.

## How it was measured

- Texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a), code
  ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`), chat ([chat.txt](chat.txt): an English
  conversation written for these measurements, plain User:/Assistant: text) and Chinese ([guxiang.txt](guxiang.txt):
  故乡, My Old Home, Lu Xun's short story of 1921, in the public domain, from the Chinese Wikisource's page 故鄉 in
  simplified characters, the page's HTML and licence notes taken out). No Chinese character is among the first 32768
  token ids the head drafts from (the text's punctuation and line breaks are), so on that text the head stays idle.
  The harness reads the first 20,480 bytes of a text when its longest prompt is 512 tokens (all of the chat's 6,900
  and of the Chinese text's 15,073) and the first 6,144 when it is 64 tokens. Each manifest's `prompt_hash` is its
  text's SHA-256.
- The harness `bench.c` includes `qwen36.c` and decodes with `q36_spec_step`, as `generate()` does: with the head in
  the container it drafts and verifies, with `QWEN36_MTP=0` it takes plain steps. It times `model_init`, reads each
  prompt from an empty state, then runs the decode steps and hashes every logit (FNV-1a) and the tokens. `TEACHER=1`
  feeds each step the text's next token, `BENCH_NLL=<file>` writes each token's negative log-likelihood and whether it
  was ranked first, `BENCH_TEXT=<file>` the generated text; `compare_nll.py` compares two `BENCH_NLL` files.
- Build: the command in the manifests (`git archive`, `make qwen36_metal.o backend_metal.o METAL=1`, then the harness
  with the flags of `make -C c qwen36 METAL=1`). The launchers in the commands are one-line scripts that set the
  variables of a variant and run the same binary: `exact-mtp0` sets `QWEN36_METAL_EXACT=1 QWEN36_MTP=0`, `exact-mtp`
  `QWEN36_METAL_EXACT=1`, `exact-mtp-p1` `QWEN36_METAL_EXACT=1 QWEN36_MTP_MIN_P=1`, `fast-mtp0` `QWEN36_MTP=0`,
  `fast-mtp` nothing, `fast5-mtp0` `QWEN36_FAST_EXPERTS=5 QWEN36_MTP=0`, `fast5-mtp` `QWEN36_FAST_EXPERTS=5`.
- Before every run the script waited for the nominal thermal state (`NSProcessInfo.thermalState`), rested 60 s and
  read the swap counters (`vm.swapusage`, the Swapouts of `vm_stat`); each run's times, thermal state, counters,
  engine log and `/usr/bin/time -l` output are in the raw logs.
- Files: the model's safetensors were written on 16 September, `experts.qc` on 5 October (its 8-byte stamp on 7
  October), the head's file on 6 October. Both versions read the same files, the head only with the drafts on.
- The head: `tools/convert_qwen36.py --mtp-only` writes it as `model-mtp.safetensors`. The runs read a file with the
  same 529 tensors, byte for byte, written on 6 October from the same checkpoint tensors: on this SSD a file written a
  few hours before reads faster for a while, so the runs read the older file, as old as the model's own
  ([head.txt](head.txt): both files' SHA-256, how each was made, how they compare).

## Speed

The first 64 and 512 tokens of each text, each from an empty state, then 32 decode steps; one warm-up run, then 4 runs
per version in the order plain, drafts, drafts, plain, plain, drafts, drafts, plain, each a new process. Decode speed
counts all 32 steps, the first included: 32 / (gen_first_ms / 1000 + 31 / gen_steady_tps), from the harness's two
columns. The first step after the prompt includes the head's first draft and, when the draft is sure enough, a verify,
and a verify whose draft is right makes the next step free: steps 2 to 32 alone would count that free step without the
verify that paid for it, so they favour the drafts (their change is in the column "steps 2–32 (favour the drafts)",
for comparison). The head's pass over the prompt's last positions is in the prompt time; the Chinese prompts skip it.
Medians, min–max in brackets; the drafts right of those verified and the tokens per forward, the same in every run of
a version, and the steps without the head; the first step's time; the change of the prompt's speed. Every run's raw
log is in `baseline-<mode>-<text>.txt` and `trial-<mode>-<text>.txt`, the summaries in `summary-<mode>-<text>.txt`,
one manifest per mode and text (decode after the 512-token prompt).

Exact mode (`QWEN36_METAL_EXACT=1`), greedy: both versions generate the same tokens, in every run, with the same
fingerprints of every logit:

| text | prompt | decode, plain | decode, drafts | change | steps 2–32 (favour the drafts) | drafts right (tokens per forward) | first step | prompt |
|---|---|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 7.919 tok/s (7.854–7.975) | 7.955 tok/s (7.868–8.025) | +0.5% | +1.3% | 4 of 5 (1.14) | 337 → 357 ms | −6.2% |
|  | 512 | 8.076 tok/s (7.769–8.097) | 7.553 tok/s (6.884–7.590) | −6.5% | −3.5% | 1 of 4 (1.03) | 332 → 468 ms | −1.6% |
| code ([code.txt](code.txt)) | 64 | 7.162 tok/s (6.856–7.205) | 7.188 tok/s (6.810–7.219) | +0.4% | +0.7% | 4 of 4 (1.14) | 353 → 371 ms | −1.8% |
|  | 512 | 6.647 tok/s (6.358–6.733) | 7.371 tok/s (7.324–7.497) | +10.9% | +17.8% | 11 of 11 (1.52) | 368 → 546 ms | −1.1% |
| chat ([chat.txt](chat.txt)) | 64 | 6.653 tok/s (6.570–6.692) | 6.768 tok/s (6.034–6.866) | +1.7% | +2.8% | 8 of 11 (1.33) | 386 → 420 ms | −4.8% |
|  | 512 | 6.536 tok/s (6.459–6.656) | 6.998 tok/s (6.925–7.063) | +7.1% | +13.5% | 10 of 10 (1.45) | 392 → 584 ms | −0.7% |
| Chinese ([guxiang.txt](guxiang.txt)) | 64 | 7.437 tok/s (7.309–7.657) | 7.335 tok/s (7.172–7.524) | −1.4% | −1.7% | none; 30 steps without the head | 365 → 369 ms | −1.9% |
|  | 512 | 7.564 tok/s (7.450–7.862) | 7.584 tok/s (7.277–7.595) | +0.3% | −0.2% | none; 30 steps without the head | 367 → 372 ms | −0.6% |

Fast mode, every chosen expert used, teacher-forced: each of the 32 tokens is the text's own next token, so both
versions read the same words, and a draft counts as right only when it is the text's token, not the model's own
choice; this can misstate the drafts either way (free generation, below, counts them on the model's own words):

| text | prompt | decode, plain | decode, drafts | change | steps 2–32 (favour the drafts) | drafts right (tokens per forward) | first step | prompt |
|---|---|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 8.277 tok/s (8.219–8.407) | 8.412 tok/s (8.385–8.452) | +1.6% | +2.3% | 4 of 5 (1.14) | 273 → 290 ms | −3.1% |
|  | 512 | 8.477 tok/s (8.251–8.600) | 8.355 tok/s (8.214–8.471) | −1.4% | +2.2% | 3 of 4 (1.10) | 298 → 434 ms | −2.0% |
| code ([code.txt](code.txt)) | 64 | 7.454 tok/s (7.320–7.678) | 6.747 tok/s (6.631–7.062) | −9.5% | −10.3% | 2 of 8 (1.07) | 343 → 335 ms | −4.6% |
|  | 512 | 7.206 tok/s (7.104–7.600) | 7.568 tok/s (7.192–7.785) | +5.0% | +10.4% | 8 of 9 (1.33) | 367 → 539 ms | −1.7% |
| chat ([chat.txt](chat.txt)) | 64 | 7.517 tok/s (7.313–7.569) | 7.715 tok/s (7.043–7.828) | +2.6% | +3.4% | 8 of 10 (1.33) | 315 → 346 ms | −0.8% |
|  | 512 | 7.424 tok/s (6.977–7.566) | 6.858 tok/s (6.553–7.048) | −7.6% | −4.2% | 5 of 9 (1.19) | 295 → 487 ms | −1.0% |
| Chinese ([guxiang.txt](guxiang.txt)) | 64 | 8.519 tok/s (8.410–8.585) | 8.427 tok/s (8.368–8.507) | −1.1% | −0.9% | none; 30 steps without the head | 254 → 258 ms | +0.4% |
|  | 512 | 9.033 tok/s (8.962–9.083) | 8.926 tok/s (8.904–8.949) | −1.2% | −0.9% | none; 30 steps without the head | 257 → 273 ms | −0.8% |

Fast mode with `QWEN36_FAST_EXPERTS=5`, teacher-forced:

| text | prompt | decode, plain | decode, drafts | change | steps 2–32 (favour the drafts) | drafts right (tokens per forward) | first step | prompt |
|---|---|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 10.901 tok/s (10.739–11.003) | 11.169 tok/s (11.054–11.325) | +2.5% | +5.3% | 4 of 5 (1.14) | 216 → 259 ms | −3.2% |
|  | 512 | 11.037 tok/s (10.912–11.275) | 11.352 tok/s (11.144–11.589) | +2.9% | +5.9% | 4 of 6 (1.14) | 228 → 302 ms | −2.4% |
| code ([code.txt](code.txt)) | 64 | 10.147 tok/s (10.109–10.193) | 9.878 tok/s (9.800–9.961) | −2.7% | −1.6% | 5 of 11 (1.19) | 256 → 298 ms | −3.7% |
|  | 512 | 9.787 tok/s (9.675–9.867) | 10.987 tok/s (10.980–11.446) | +12.3% | +19.4% | 8 of 8 (1.33) | 258 → 353 ms | −3.9% |
| chat ([chat.txt](chat.txt)) | 64 | 9.668 tok/s (9.608–9.789) | 10.845 tok/s (10.631–11.038) | +12.2% | +14.0% | 8 of 9 (1.33) | 273 → 291 ms | −2.6% |
|  | 512 | 9.627 tok/s (9.569–9.818) | 9.704 tok/s (9.498–9.796) | +0.8% | +4.1% | 5 of 9 (1.19) | 232 → 337 ms | −1.2% |
| Chinese ([guxiang.txt](guxiang.txt)) | 64 | 10.833 tok/s (10.758–10.907) | 10.726 tok/s (10.644–10.819) | −1.0% | −0.5% | none; 30 steps without the head | 227 → 244 ms | +1.0% |
|  | 512 | 11.058 tok/s (11.034–11.089) | 10.987 tok/s (10.934–11.048) | −0.6% | −1.1% | none; 30 steps without the head | 218 → 205 ms | −1.6% |

Outcomes of the manifests (decode after the 512-token prompt, all 32 steps, every run of one version against every run
of the other): exact mode prose regression, code improvement, chat improvement, Chinese no-change; fast mode prose
no-change, code no-change, chat no-change, Chinese regression; with fast experts prose no-change, code improvement,
chat no-change, Chinese no-change.

The spread of the four runs of a version, (max − min) / median of the decode speed, was 0.5–12.3% over the 48 groups
of four above, more than the 1% the benchmark guide expects in 44 of them; the "in every run" outcomes above hold
against that spread.

On the Chinese text the head stays idle at every step it could run (30 after each prompt) and the prompts skip its
pass, so the drafts change little: decode −1.4% and +0.3% in exact mode, −1.1% and −1.2% in fast mode (after the
512-token prompt slower in every run, the manifest's regression), −1.0% and −0.6% with fast experts; the prompt −1.9%
to +1.0%. What is left is the head's presence: with it the expert cache keeps room for a verify's two top-k groups, so
it holds 8 fewer experts between steps (287 of the 303 slots instead of 295) and reads a few more (50 to 60 more a run
in exact and fast mode, of about 9,100 to 11,300, and 31 and 36 of about 6,600 and 8,800 with fast experts), besides
the memory below. The runs of the head's own cost below measure the Chinese text again in exact mode with a setting
that matters only where the head drafts, so the same work as here: after the 512-token prompt they gave −4.7% where
this table gives +0.3%, three hours apart, so changes of a few percent between such runs are within their noise.

Rule 6 of the benchmark guide: a case more than 5% slower is measured again from clean, with a new warm-up and the
same ABBA order (3 of the 12; `baseline-<mode>-<text>-rerun.txt`, `trial-...`, `summary-...` and a manifest each, of
the decode that was slower): exact mode, prose: prompt after 64 tokens −6.2% (re-run −2.0%), decode after 512 tokens
−6.5% (re-run −7.4%, its manifest: regression); fast mode, code: decode after 64 tokens −9.5% (re-run −9.9%, its
manifest: regression); fast mode, chat: decode after 512 tokens −7.6% (re-run −6.5%, its manifest: regression). The
re-runs confirm the three decode slowdowns. Each is where few drafts were sure enough for a verify (4, 8 and 9 in 32
steps) and not all of those were right (1, 2 and 5): the head's pass at every step and the wrong verifies then cost
more than the right drafts save. The prompt's −6.2% after 64 tokens came out at −2.0%.

The head's own cost, in exact mode: with `QWEN36_MTP_MIN_P=1` only a draft of confidence 1 goes to a verify (the
launcher `exact-mtp-p1`; 0 drafts did on the prose), so the head runs after every token and the rest stays as in plain
decoding. Against `QWEN36_MTP=0` the 32 decode steps took −4.8% after 64 tokens and −5.7% after 512 on the prose
(regression); on the Chinese text, where the head stays idle (30 and 30 steps without it) and the prompts skip it,
−1.2% and −4.7% (no-change), the prompts −0.8% and −1.3% (`manifest-exact-p1-<text>.json` and the files beside it).

The costs. The model was ready in 0.17–0.20 s without the head's drafts and 0.26–0.40 s with them in exact mode
(0.18–0.20 and 0.28–0.30 s in fast mode, 0.17–0.19 and 0.26–0.30 s with fast experts). Peak memory footprint
(`/usr/bin/time -l`): exact mode 990–999 → 1205–1215 MiB, fast mode 983–987 → 1110–1123 MiB, with fast experts 983–986
→ 1111–1122 MiB; the medians grew by 213–214 MiB in exact mode, 136–137 in fast mode and 136 with fast experts on
prose, code and chat, and by 207, 127 and 125 MiB on the Chinese text. What the head adds by design comes to about 122
MiB in fast mode and 184 MiB in exact mode with these runs' KV capacity of 1024 positions: the copy of the DeltaNet
state a verify keeps (63 / 123 MiB), the head's dense weights in INT8 (38 MiB), a verify's logits for eight rows (8
MiB), the head's input rows for a 512-token prompt batch (8 MiB), its KV cache (2 / 4 MiB), the eight-row decode
tensors (3 MiB); the rest, 14–15 MiB in fast mode and 29–30 MiB in exact mode on those three texts, was not traced.
The head's KV cache is sized to the context like the model's, and the attention scores of the eight-row tensors grow
with it: about 2.3 / 4.3 MiB more per further 1024 positions (at 8,192 positions the head's KV cache alone takes 16 /
32 MiB). No page was written to swap during the runs of the three speed tables, nor during those of the quality, of
free generation and of the fingerprints (below). Two other runs wrote a few, where a measurement should write none, as
their summaries say: one with `QWEN36_MTP=0` in the fast-mode code re-run (128 pages of 16 KiB, 2 MiB) and one with
`QWEN36_MTP=0` in the head's own cost on the Chinese text (24 pages, 0.4 MiB). Without each of them, the re-run's
decode changes by −8.9% and +3.9% after the 64- and 512-token prompts (−9.9% and +4.3% with it), and the head's own
cost on the Chinese text by −1.6% and −4.6% (−1.2% and −4.7%).

## Quality

Fast mode, each text read with its true tokens, 1024 of them after a 512-token prompt (prose, code, Chinese) or a
64-token one (chat, as in [the fast-experts record](../qwen36-fast-experts-2026-10-06/README.md), whose per-token
files the plain and fast-experts runs of prose, code and chat here reproduce byte for byte), one run per variant (the
same binary and settings give the same numbers in every run), compared token by token with `compare_nll.py`
([quality.txt](quality.txt), the per-token files in `quality/`): the change of the mean negative log-likelihood with
its standard error, and of the share of tokens the model ranks first, in points. A change under twice its standard
error is within the noise. The drafts alone and fast experts alone are compared with plain decoding; fast experts with
the drafts with fast experts alone. The decode speeds are of these single runs, over all 1024 steps; experts read
include the prompt's.

| text | variant | decode | experts read | drafts right (tokens per forward) | mean NLL | top-1 |
|---|---|---|---|---|---|---|
| prose | plain (`QWEN36_MTP=0`) | 9.050 tok/s | 169,388 |  |  |  |
|  | drafts | 8.891 tok/s | 178,291 | 120 of 190 (1.13); 31 steps without the head | −0.03% ± 0.02% (within the noise) | −0.1 ± 0.1 points (within the noise) |
|  | fast experts (`QWEN36_FAST_EXPERTS=5`) | 11.694 tok/s | 93,061 |  | +1.21% ± 0.48% | −1.0 ± 0.7 points (within the noise) |
|  | fast experts, drafts | 12.055 tok/s | 92,667 | 123 of 188 (1.14); 31 steps without the head | +0.14% ± 0.22% (within the noise) | +0.7 ± 0.4 points (within the noise) |
| code | plain (`QWEN36_MTP=0`) | 7.876 tok/s | 208,512 |  |  |  |
|  | drafts | 8.069 tok/s | 213,422 | 229 of 297 (1.29) | −0.02% ± 0.02% (within the noise) | 0.0 ± 0.0 points |
|  | fast experts (`QWEN36_FAST_EXPERTS=5`) | 10.346 tok/s | 119,274 |  | −0.67% ± 0.39% (within the noise) | −0.7 ± 0.5 points (within the noise) |
|  | fast experts, drafts | 11.514 tok/s | 113,415 | 235 of 301 (1.30) | +0.07% ± 0.28% (within the noise) | +0.2 ± 0.4 points (within the noise) |
| chat | plain (`QWEN36_MTP=0`) | 7.918 tok/s | 208,811 |  |  |  |
|  | drafts | 7.961 tok/s | 214,775 | 218 of 299 (1.27) | 0.00% ± 0.00% | 0.0 ± 0.0 points |
|  | fast experts (`QWEN36_FAST_EXPERTS=5`) | 10.446 tok/s | 125,056 |  | +0.40% ± 0.96% (within the noise) | +0.8 ± 0.8 points (within the noise) |
|  | fast experts, drafts | 11.317 tok/s | 120,401 | 215 of 288 (1.27) | +0.94% ± 0.56% (within the noise) | −0.9 ± 0.5 points (within the noise) |
| Chinese | plain (`QWEN36_MTP=0`) | 9.222 tok/s | 179,313 |  |  |  |
|  | drafts | 8.949 tok/s | 181,326 | none; 1022 steps without the head | 0.00% ± 0.00% | 0.0 ± 0.0 points |
|  | fast experts (`QWEN36_FAST_EXPERTS=5`) | 11.627 tok/s | 101,182 |  | +2.61% ± 6.14% (within the noise) | +1.2 ± 0.4 points |
|  | fast experts, drafts | 11.515 tok/s | 102,408 | none; 1022 steps without the head | +0.27% ± 1.33% (within the noise) | −0.3 ± 0.3 points (within the noise) |

Averaged over prose, code and chat (the benchmark guide's three texts): the drafts alone mean NLL −0.02% ± 0.01%,
top-1 0.0 ± 0.0 points; with fast experts, the drafts mean NLL +0.38% ± 0.22%, top-1 0.0 ± 0.3 points; fast experts
alone mean NLL +0.31% ± 0.38%, top-1 −0.3 ± 0.4 points; both together, against plain decoding, mean NLL +0.70% ±
0.42%, top-1 −0.3 ± 0.4 points. On the Chinese text, where the head stays idle: the drafts alone the same per-token
file; with fast experts, the drafts mean NLL +0.27% ± 1.33%, top-1 −0.3 ± 0.3 points; fast experts alone mean NLL
+2.61% ± 6.14%, top-1 +1.2 ± 0.4 points.

## Free generation

256 greedy tokens in fast mode after the first 512 tokens of each text, one run per variant
([generation.txt](generation.txt), the texts in `generated/`). Decode over all 256 steps, the first included; in
brackets the drafts right of those verified and the tokens per forward, counted on the model's own words.

| text | plain | drafts | change | drafts right (tokens per forward) | with fast experts: drafts right (tokens per forward) |
|---|---|---|---|---|---|
| prose | 9.557 tok/s | 9.894 tok/s | +3.5% | 49 of 57 (1.24) | 83 of 87 (1.48) |
| code | 7.254 tok/s | 8.093 tok/s | +11.6% | 100 of 111 (1.64) | 85 of 105 (1.50) |
| chat | 7.986 tok/s | 8.255 tok/s | +3.4% | 73 of 84 (1.40) | 78 of 92 (1.44) |
| Chinese | 8.388 tok/s | 8.295 tok/s | −1.1% | none; 254 steps without the head | none; 254 steps without the head |

Without fast experts the drafts gave the same words as plain decoding on all four texts (fast mode does not promise
it). With fast experts the words differed on prose, code, chat, Chinese: a draft row keeps fewer sure experts, and the
experts a step leaves out depend on what the cache holds, which the verifies change; on the Chinese text, where the
head stays idle, the difference is in the cache: with the head it holds 8 fewer experts between steps (287 of the 303
slots instead of 295); their speeds are not compared here (generation.txt has them).

With fast experts and the drafts, the code continuation writes a comment rule of box-drawing characters in which the
model picked some byte tokens that leave a character incomplete:
[generated/code-fast5-mtp.txt](generated/code-fast5-mtp.txt) is not valid UTF-8 there.

## Results unchanged

Exact mode, the 64- and 512-token prompts of `promessi_sposi.txt` and 32 greedy tokens
([fingerprints.txt](fingerprints.txt)): the commit before (f86d6ecb) with the model without the head, c8fc51fc with
the head and `QWEN36_MTP=0`, and with the drafts give the same fingerprints.