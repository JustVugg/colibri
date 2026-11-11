# dsv41-kvq-2026-11-11 — the packed KV cache on the released checkpoint

The packed host KV (`COLI_VK_KV_FP32` unset, the default) against fp32
(`COLI_VK_KV_FP32=1`) on the real **DeepSeek-V4.1-Flash-1017** checkpoint: twelve fixed
prompts, a 16K needle, the planner's price at 16K. The throughput record is its own
[manifest](../dsv41-kvq-throughput-2026-11-11.json); every file here is hashed in
[`sha256sums.txt`](sha256sums.txt). The four run logs are byte-identical copies of
`E14-compare.log`, `E14-drift-detail.log`, `E15-A-needle.log` and `E15-B-needle.log` from
the run directory, renamed `.txt` because this repository ignores `*.log`.

**Machine.** AMD Radeon RX 6900 XT 16 GB (RADV NAVI21, Mesa 26.2.1-arch1.1, Vulkan
1.4.354); 12th Gen Intel i9-12900K, 16 cores / 24 threads, `OMP_NUM_THREADS=2
COLI_NO_OMP_TUNE=1`; 38.9 GiB RAM; the 476 GiB checkpoint on `/dev/nvme0n1p2` (f2fs);
Arch Linux, kernel 7.1.11-arch1-1.

**Commit under test: `382dfb45`.** These runs were made at **`d10c0a5a`**, the merge the PR
was audited at; part 1 changed the planner's fixed term, its unit test and the CI leg and
touches no inference path, so the numbers hold at `382dfb45` — a changed *printout* is noted.

## Commands

Serve, once an arm (`--ram 24` because MemAvailable was under 26 GB — 24,958,726,144 B at
plan time; the planner A/B used `--ram 28`, as the run plan says):

```sh
cd /root/Projects/colibri-pr2712
env -u COLI_VK_KV_QUANT [-u COLI_VK_KV_FP32 | COLI_VK_KV_FP32=1] \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json COLI_VULKAN=1 COLI_VK_CHAIN=1 \
  COLI_VK_LEDGER=1 V41_STATS=1 OMP_NUM_THREADS=2 COLI_NO_OMP_TUNE=1 COLI_USAGE=<usage> \
  ./c/coli serve --model /root/Models/DeepSeek-V4.1-Flash-1017 \
                 --ram 24 --ctx 16384 --gpu 0 --port 8061
python3 scripts/battery.py --out E12-answers.json     # packed; E13 = the fp32 arm
python3 scripts/compare.py E12-answers.json E13-answers.json
env [-u COLI_VK_KV_FP32 | COLI_VK_KV_FP32=1] VK_ICD_FILENAMES=$ICD ./c/coli plan \
  --model /root/Models/DeepSeek-V4.1-Flash-1017 --ram 28 --gpu 0 --vram 16 --ctx 16384 --json
```

`battery.py` is the 12 fixed prompts on `/v1/chat/completions` (temperature 0, `max_tokens` 110); `needle.py` plants a passphrase mid-prompt and asks for it back.

## The 12-prompt A/B — 10 of 12 byte-identical

[`E14-compare.txt`](E14-compare.txt): `SUMMARY identical=10 differ=2`. The two that differ
are the two long free-form answers, and both stay **correct but different** — the declared
approximation, not a parity claim ([`E14-drift-detail.txt`](E14-drift-detail.txt)):

- **Prompt 1** (17 bolts for $4.25): both reach $0.25 by different arithmetic — A writes
  "425 cents ÷ 17", B writes "4.25 ÷ 17 = 0.25" — both cut at the cap (A 340 chars, B 308).
- **Prompt 6** (fix `if x[i] = 0`): the same fix and explanation, differing in headings and
  note order (A 336 chars, B 353).

The other ten — passive rewrite, 47·19, the two capitals, `is_prime`, the two-sentence
summary, primary colours, missing word, weekday, German translation, letter count — are
identical byte for byte: [`E12-answers.json`](E12-answers.json) (packed),
[`E13-answers.json`](E13-answers.json) (fp32).

## The 16K needle

Packed default, 16,000-token prompt: **16,046 prompt tokens, passphrase returned, 978.7 s**
([`E15-A-needle.txt`](E15-A-needle.txt), [`E15-A-needle.json`](E15-A-needle.json)). The
fp32 control is the **4K** arm (4,046 prompt tokens, passphrase returned, 381.3 s,
[`E15-B-needle.txt`](E15-B-needle.txt)) — a control that the instrument works, not a 16K
fp32 comparison, which was not run.

## The host KV, and the planner/engine closed form

The ledger line ([`ledger.txt`](ledger.txt), one representative turn an arm): packed
`kv mirror 13.1 MiB (quantized) | host kv 16.8 MiB (packed)`; fp32 `kv mirror 51.6 MiB` and
no `host kv` field (its host rows are f32). The closed form:

```
packed: 890·ctx + (43·128 + 40·6)·528   = 17,614,592 B at 16K = 16.8 MiB
fp32:  6400·ctx + (43·128 + 40·6)·2048  = 116,621,312 B at 16K
```

43 rings = 40 layers + 3 DSpark stages; 40·6 = one rollback row a layer per
(`dspark_block_size` 5 + 1). The copied plans ([`plan-16384-packed.json`](plan-16384-packed.json), [`plan-16384-fp32.json`](plan-16384-fp32.json)) are the **`d10c0a5a` planner's** output: 40
rings, no rollback rows, `fixed_state_bytes` 2,703,360 / 10,485,760 B, totals 17,285,120 /
115,343,360 B. Part 1 fixed exactly that: at `382dfb45` the planner prints 3,032,832 /
11,763,712 B and totals **17,614,592 / 116,621,312 B**, the engine's own numbers;
`sequence_state_bytes` was already exact (`890·ctx` / `6400·ctx`).
