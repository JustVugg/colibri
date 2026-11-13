# Guessed experts for qwen36 (Metal graph, fast experts), measured on an M1

`QWEN36_FAST_GUESSED=4` against unset, both with `QWEN36_FAST_EXPERTS=5` and the MTP head's drafts, with the same
binary: the harness `bench.c` built against 082ce6c2, a local commit with this change on dev 6cf80dbc with the Metal
kernels and the expert cache merged (this branch holds the same change to the code, patch for patch, on a newer dev),
with the command in the manifests. An M1 MacBook Air with 8 GB (no fan), macOS 26.6.2, Qwen3.6-35B-A3B in Colibri's
qwen36 container with `canonical.qc`, `experts.qc` and the checkpoint's MTP head (`model-mtp.safetensors`) beside it,
the Metal graph's default expert cache (512 MiB, 303 slots).

## Speed

Three texts: prose (`promessi_sposi.txt`: speed-bench/promessi_sposi.txt of antirez's ds4 at commit 0aaea5a, of which
the harness reads the first 20,480 bytes), code ([code.txt](code.txt): an earlier version of `c/qwen36_graph.h`, first
20,480 bytes read) and chat ([chat.txt](chat.txt): an English conversation written for these measurements, plain
User:/Assistant: text, read whole). Of each, the first 64 and the first 512 tokens, each from an empty state, then 32
decode steps, each fed the text's own next token (teacher forcing: both versions read the same tokens). One warm-up
run, then 4 runs per version in the order J = 5, R = 4, R = 4, J = 5, J = 5, R = 4, R = 4, J = 5, each a new process
after the nominal thermal state and a 60 s rest. Decode speed counts all 32 steps, the first included: the first step
is a verify when the head's first draft is sure enough, and a right draft makes the second step free (steps 2 to 32 in
their own column). Medians, min–max in brackets; every run's raw log in `baseline-<text>.txt` and `trial-<text>.txt`,
the summaries in `summary-<text>.txt`, one manifest per text (decode after the 512-token prompt: prose improvement,
code improvement, chat improvement).

| text | prompt | decode, J = 5 | decode, J = 5 and R = 4 | change | steps 2–32 | drafts right, J = 5 → R = 4 | first step | prompt |
|---|---|---|---|---|---|---|---|---|
| prose (`promessi_sposi.txt`) | 64 | 11.302 tok/s (11.266–11.303) | 12.022 tok/s (11.957–12.033) | +6.4% | +6.1% | 4 of 5 → 4 of 5 | 240 → 219 ms | −0.7% |
|  | 512 | 11.212 tok/s (11.141–11.366) | 12.043 tok/s (11.908–12.231) | +7.4% | +7.5% | 4 of 6 → 4 of 6 | 302 → 288 ms | 0.0% |
| code ([code.txt](code.txt)) | 64 | 9.884 tok/s (9.723–9.943) | 10.485 tok/s (10.402–10.539) | +6.1% | +5.5% | 5 of 11 → 4 of 10 | 279 → 258 ms | −0.1% |
|  | 512 | 10.999 tok/s (10.874–11.129) | 11.866 tok/s (11.659–11.958) | +7.9% | +7.9% | 8 of 8 → 8 of 9 | 364 → 350 ms | −1.0% |
| chat ([chat.txt](chat.txt)) | 64 | 10.611 tok/s (10.547–10.798) | 11.468 tok/s (11.370–11.597) | +8.1% | +8.3% | 8 of 9 → 7 of 8 | 279 → 266 ms | +2.4% |
|  | 512 | 9.372 tok/s (9.103–9.540) | 10.419 tok/s (10.397–10.729) | +11.2% | +11.9% | 5 of 9 → 5 of 8 | 392 → 345 ms | +2.8% |

The prompt does not use the option (only decode does). The model was ready in 0.26–0.30 s, and the peak memory
footprint (`/usr/bin/time -l`) was 1119–1122 MiB with J = 5 alone and 1119–1124 MiB with R = 4; no page was written to
swap during any run of this record (speed, quality, free generation and fingerprints).

## Quality

Each text read with its true tokens, 1024 of them after a 512-token prompt (prose, code, and the Chinese text of the
speculative-decoding record, [guxiang.txt](guxiang.txt), none of whose Chinese characters is among the ids the head
drafts from, so that the head stays idle and every step is a plain decode step) or a 64-token one (chat), one run per
version (the same binary and settings give the same numbers in every run), compared token by token with
`compare_nll.py` ([quality.txt](quality.txt), the per-token files in `quality/`): the change of the mean negative
log-likelihood with its standard error, the same change as perplexity, and the change of the share of tokens the model
ranks first, in points. A change under twice its standard error is within the noise. The decode speeds are of these
single runs, all 1024 steps; experts read include the prompt's; "kept as guessed" is from the line at exit: the
choices that stayed only because the layer before guessed them, in the rows the rule ran for (a token at a layer).

| text | decode, J = 5 | decode, J = 5 and R = 4 | experts read | kept as guessed | mean NLL | perplexity | top-1 |
|---|---|---|---|---|---|---|---|
| prose | 11.736 tok/s | 12.552 tok/s | 92,667 → 78,357 | 20,408 choices in 35,256 rows | +0.98% ± 0.33% | +2.7% | −0.6 ± 0.5 points (within the noise) |
| code | 11.397 tok/s | 12.590 tok/s | 113,415 → 96,945 | 21,517 choices in 30,849 rows | −0.12% ± 0.32% (within the noise) | −0.3% | −0.8 ± 0.4 points (within the noise) |
| chat | 11.061 tok/s | 12.035 tok/s | 120,401 → 104,347 | 24,568 choices in 31,512 rows | −0.23% ± 0.70% (within the noise) | −0.3% | −0.1 ± 0.6 points (within the noise) |
| Chinese | 11.069 tok/s | 12.269 tok/s | 102,408 → 85,310 | 26,349 choices in 39,936 rows | +18.65% ± 5.43% | +2.9% | −0.8 ± 0.4 points (within the noise) |

Averaged over prose, code and chat: mean NLL +0.21% ± 0.28%, top-1 −0.5 ± 0.3 points. Against plain fast decoding
(every chosen expert, without the drafts: the speculative-decoding record's runs, `quality/<text>-plain.nll`), J = 5
and R = 4 with the drafts gave +0.91% ± 0.47% of mean NLL and −0.8 ± 0.4 points of top-1 on average over the three
texts.

With the drafts the rule does not reach every token: when a draft is right, the token after it is scored on the
verify's row of that draft, and a verify's draft rows keep the rule of fast experts in both versions (with R = 4, 120,
233 and 216 of the 1024 tokens on prose, code and chat; with J = 5, 123, 235 and 215; for each of the other tokens the
line at exit counts the rule at every layer but the first). Without the head (`QWEN36_MTP=0`, or no
`model-mtp.safetensors` beside the model) the rule runs at every decode step, as on the Chinese text, where the head
stays idle; on prose, code and chat that case was not measured.

## Free generation

After the 512-token prompt of each text, 256 greedy tokens, two runs per version in the order J = 5, R = 4, R = 4, J =
5 ([generation.txt](generation.txt), the texts in `generated/`). Where the words differ, the speeds compare different
texts.

| text | decode, J = 5 (two runs) | decode, J = 5 and R = 4 (two runs) | words | drafts right, J = 5 → R = 4 |
|---|---|---|---|---|
| prose | 13.033–13.082 tok/s | 13.099–13.121 tok/s | each version the same in its two runs; the versions differ | 83 of 87 → 34 of 45 |
| code | 13.351–13.593 tok/s | 15.018–15.266 tok/s | each version the same in its two runs; the versions differ | 85 of 105 → 91 of 103 |
| chat | 11.632–11.706 tok/s | 13.225–13.295 tok/s | each version the same in its two runs; the versions differ | 78 of 92 → 81 of 89 |
| Chinese | 11.367–11.392 tok/s | 12.455–12.461 tok/s | each version the same in its two runs; the versions differ | none → none |

On prose the J = 5 words fall, about a fifth of the way in, into a list that repeats to the end; the R = 4 words do
not loop that way (towards the end they turn around a few short phrases). On code the R = 4 continuation repeats the
fields of the struct before it in the next one. The J = 5 texts are byte for byte those of the speculative-decoding
record's runs with the drafts and J = 5; as there, the code continuation with J = 5 writes a comment rule of
box-drawing characters in which the model picked some byte tokens that leave a character incomplete:
[generated/code-fast5.txt](generated/code-fast5.txt) is not valid UTF-8 there.

## Results unchanged without the option

Exact mode refuses the option. In exact mode, with the drafts, the 64- and 512-token prompts of `promessi_sposi.txt`
and 32 greedy tokens: the speculative-decoding change below (c8fc51fc, as its record measured it) and this one
(082ce6c2) give the same fingerprints; in fast mode with J = 5, this record's runs without the option give the
fingerprints of that commit's runs in the speculative-decoding record ([fingerprints.txt](fingerprints.txt)).