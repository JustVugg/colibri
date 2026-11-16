# WEX bench

`c/tools/wex_bench.py` runs the measurements the WEX series asks helpers for (#2860) and writes a
report to paste there. Python 3, standard library only; it builds the two C tools it needs
(`tools/wex4_verify`, `tools/wex_decode_bench`) with `make -C c` when they are missing.

```sh
python3 c/tools/wex_bench.py all --container v4_flash_experts.nr3.wex --out bench-out
# then paste bench-out/wex-bench-report.md as a comment on #2860
```

## Containers

- The published ones (#2860 links the Hugging Face repository):
  `v41-reap/v4_flash_experts.nr3.wex` (DeepSeek V4.1 Flash REAP-192, 7,680 WNR3 records, 123.3 GB) and
  `qwen38-mxfp4/q38_experts.wex` (Qwen3.8-Flash-Next, 24,576 RAW records, 64.3 GB).
- Your own: a RAW WEX4 file with Wex-WRN2's `wex4/wex4_convert.py pack`, a WNR3 one with
  `c/tools/build_nr3.py` (docs/wex.md, "Making a WNR3 container") and `wex4_convert.py upgrade`.

`decode` needs WNR3 records; `verify` and `io` take any WEX4 container.

## Stages

Each stage runs alone and adds its result to `OUT/wex-bench-results.json`; `report` turns what is
there into `OUT/wex-bench-report.md` and `.json`. Run the stages of one machine and one container
with the same `--out`.

| Stage | What it runs | Needs |
|---|---|---|
| `env` | OS, CPU model and flags (AVX2, AVX-512F, NEON, SVE), cores, RAM, GPUs (`nvidia-smi`, `vulkaninfo`), compilers, the bus / media / model of the container's drive where the OS tells it | nothing |
| `verify` | `tools/wex4_verify FILE 8`: every record against its SHA-256; time, GB/s, the root | `--container` |
| `decode` | `tools/wex_decode_bench decode`: N seeded WNR3 records in memory, ms per record for the reference decoder, the fast decoder at each `--lanes` stream count (one thread), both with T records at once, and the CUDA helper (one record a call and its host batch) when `--helper` or `COLI_WEX_CUDA_LIB` names it. Every arm's bytes are compared with the reference's | `--container` (WNR3) |
| `io` | `tools/wex_decode_bench io`: N seeded records read whole, uncached (Windows `FILE_FLAG_NO_BUFFERING`, Linux `O_DIRECT`, macOS `F_NOCACHE`) and warm (OS cache), on 1 and T threads | `--container` |
| `engine` | one engine, the same prompt, greedy, `--tokens` tokens, one run per arm: `shards` (no container), `container-fast`, `container-reference` (`COLI_WEX_FAST_DECODE=0`), `container-helper` (`COLI_WEX_CUDA_LIB`). Prefill and decode time, tok/s, the token ids (they must match), container reads, and which decoder took the records (`COLI_WEX_STATS=1`) | `--snap`, `--container`, the engine built |
| `report` | the Markdown and JSON report, with the commit, the variables set, the container's root and the record count | some stage before |
| `all` | `env verify decode io` (and `engine` with `--snap`), then `report` | |

Options that matter: `--records N` (16), `--seed S` (1: the same records on every machine),
`--lanes 2,4,6,8`, `--threads T` (records at once and read threads; default the OpenMP team, at
most 8), `--passes P`, `--helper PATH`, `--out DIR`.

### The engine stage

```sh
python3 c/tools/wex_bench.py engine --engine deepseek_v41 \
    --snap DeepSeek-V4.1-Flash-REAP192-MXFP4 --container v4_flash_experts.nr3.wex --tokens 8 \
    [--helper c/kaira_wex_cuda.dll] [--arms shards,container-fast] --out bench-out
```

`deepseek_v41` (`make -C c deepseek_v41`): the prompt is BOS + "The three primary colors of light
are" (`--prompt-ids` for another), 8 expert slots a layer (`--cap`). The root comes from the
`verify` stage of the same `--out` (or `--root`), so every record is hashed at its first read, as in
the series' measurements. `qwen38`: give `--prompt-ids` (Qwen3.8 tokenizer ids); it runs
`qwen38 CAP 8 prompt.json`. Any other engine or build (`deepseek_v4`, a Vulkan or CUDA build):
`--engine custom --cmd "./deepseek_v4 MODEL --prompt-file p.txt --max-tokens 8 --no-dspark"`;
the bench sets `SNAP` and the `COLI_WEX_*` variables per arm and compares the arms' stdout.
The logs of each arm are written to `OUT/wex-bench-engine-ARM.log`. The `deepseek_v41` driver has
run on Windows; the `qwen38` and `custom` drivers have not yet (no Qwen3.8 trunk or V4 checkpoint
where the bench was written): a report from them is welcome, a fix too.

A run reads the experts from the drive (32 GB of RAM against 144 GB of experts on a V4.1
checkpoint): each arm is one run, in the order given; the page cache is not dropped between arms.

## What the report holds, and what it does not

The report and the results file hold no host name, user name or path: paths are replaced by
`<container dir>`, `<checkpoint>`, `<repo>`, `<path>`, the container is named by its file name,
and variables that hold paths are shown as `<set>`. Read the report before pasting it.

## Example (Windows 10, i7-7700K, RTX 3070, the V4.1 container on NVMe)

```
| arm                                  | threads | ms/record, best |
| reference (wex_decode_nr2_record)    |       1 |          857.72 |
| fast AVX2, 4 streams (default)       |       1 |          117.00 |
| fast (default streams), 8 at once    |       8 |           29.25 |
| CUDA helper, coli_wex_decode_record  |       1 |            6.35 |
```

The sample reports of the PR that added the bench have the whole tables.
