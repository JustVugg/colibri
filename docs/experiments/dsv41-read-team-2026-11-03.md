# DeepSeek V4.1 expert reads: team width and decode

One variable changes here: the width of the OpenMP team in `expert_fetch`. It used to ask for `min(read depth, tensors to read)` threads while every other region runs at the full team width, so on a layer with few experts missing from the cache each call made libgomp retire pool threads and create new ones. It now runs at the team width and a counter keeps the concurrent readers at the read depth. This follows the team sizing in `dsv41-omp-team-2026-09-15.md`, which fixed the width of the team; this note is about the width of one region inside it.

The machine is an i7-13700KF (coli sets `OMP_NUM_THREADS=8`, the P-core count) with 31 GiB of RAM, an RTX 4070 Ti 12 GB and an NVMe drive. The model is the DeepSeek V4.1 Flash UD-Q2_K_XL GGUF (311 GiB) on the CUDA tier, greedy, `--ram 18`. Both arms have the scratch pool; the first commit's binary is the baseline and the second commit's binary is the trial.

| request (128 tokens) | pool only, tok/s | + team width, tok/s | change |
|---|---|---|---|
| decode_a | 2.04 | 2.46 | +21% |
| decode_b | 1.97 | 2.34 | +19% |
| decode_a_again | 2.14 | 2.62 | +22% |
| decode_c | 1.81 | 2.04 | +13% |

These are medians of three runs per arm, alternated T B T B T B, each on a fresh server. The manifest samples `decode_c`: its text is the same in five of the six runs, and the sixth has the same length and read the same bytes. Prefill is unchanged (the 650-token prompt takes about 315 s in both) and resident memory is flat in both (12.4 GiB peak, no swap).

The kernel's pid counter gives the threads created per 128-token request: 6,557 and 7,522 for the original engine (51 to 59 per token) and 325 and 334 for the second commit (2.5 to 2.6 per token). That is one run per arm, and the "before" arm is the original engine, not the pool alone.

The CUDA tier's text is not reproducible from run to run even for one binary, so identity is judged on the CPU path (three prompts, full-text SHA-256, identical to the original engine for both commits) and on the V4.1 fixtures, which stay token-exact at cap 1, 2, 8 and with the 40-token prompt.

It is not known whether the thread churn is what costs the time. `perf` is blocked on the machine, and creating a thread costs far less than the gain, so the rest would come from elsewhere (cold caches and the placement of the new threads are candidates, not findings). It is one machine and one model, the numbers are specific to a disk-bound regime with a team of 8, and the GPU tier's warm-start history (`.coli_usage`) was not restored between runs.

The raw files are in `dsv41-read-team-2026-11-03-raw.txt` (engine lines and requests of every run, plus the RSS series of the memory runs every 30 s), and the manifest is `dsv41-read-team-2026-11-03.json`.
