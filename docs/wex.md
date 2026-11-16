# WEX expert containers

A WEX container holds a MoE model's routed experts in one file, one record per (layer, expert),
each record starting on a 4096-byte page. An engine reads a missed expert with one aligned read
of one record instead of six tensor reads spread over the safetensors shards, and the record lands
in the slab in the layout the expert kernels already use.

WEX4 is the current version of the container. Its first 4096 bytes (the header page) hold the
SHA-256 of the index, and its last 32 bytes hold the SHA-256 of the rest of the page: that hash
is the container's **root**. The index holds the SHA-256 of every record. So one 64-digit root
names every byte of the file, and a reader can check the file, or each record as it reads it,
against a root it got from somewhere it trusts.

The format and its tools come from [Wex-WRN2](https://github.com/KairaCode/Wex-WRN2), which holds
the specifications, the conformance vectors and the measurements. This page says what colibri
carries and how to use it. The bytes are specified there, not here:

- [WEX4-SPEC.md](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WEX4-SPEC.md): the WEX4
  container (header page, index, records, hashes).
- [WNR3-SPEC.md](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WNR3-SPEC.md): the WNR3
  lossless codec of an MXFP4 expert record.

## Files

| File | What it is |
|---|---|
| `c/wex_format.h` | the WEX1-WEX3 wire format: header, the 72-byte record entry, validation, and the lossless record codecs (scale tiles, palette3, the WNR rANS decoder). Header only, no engine dependency |
| `c/wex_palette.h`, `c/wex_neural_rans.h` | the palette and rANS primitives `wex_format.h` includes |
| `c/wex4.h` | the WEX4 reader: SHA-256, `wex4_parse_header` (the header page), `wex4_open` (the index, every rule of the spec, in place, no allocation), `wex4_expert_entry` (an expert record as the 72-byte entry above), `wex4_decode_record` (RAW, WBP1 and WNR3 records) |
| `c/tools/wex4_verify.c` | checks a WEX4 file and prints its root |
| `c/tests/test_wex_format.c` | the codecs and validators on synthetic records, with corruption cases |
| `c/tests/test_wex4_verify.c`, `c/tests/wex4_fixture.h` | the tool on synthetic WEX4 files; the fixture writes a WEX4 file in memory the way the in-place upgrade of a WEX1 file does |
| `c/wex_decode.c`, `c/wex_decode.h` | WNR3 records on the CPU: the process's WNM2 model registry (`coli_wex_register_model`, and `coli_wex4_register_model` for a WEX4 container's blob, checked against its index hash) and the in-place decode of one record (`coli_wex_decode_record`), and its decode straight into the caller's six buffers (`coli_wex_decode_record_to`) |
| `c/tests/test_wex_decode.c`, `c/tests/wnr3_fixture.h` | the registry and the decode on records made by the reference encoder; the fixture trains a model and writes the WNR3 form of a WEX4 container |
| `c/wex_fast_decode.h` | a fast CPU decoder of WNR3 records, the reference's bytes and verdicts ([below](#the-fast-cpu-decoder)) |
| `c/tests/test_wex_fast_decode.c` | the fast decoder against the reference of `wex_decode.c`: synthetic records, mutated mini-records, and real records of a container named on its command line; `--bench` times both |
| `c/tools/wex_nr3_tools.c`, `c/tools/build_nr3.py`, `c/tests/test_wex_nr3_tools.py` | the WNR3 encoder: a C library and the Python builder that drives it ([below](#making-a-wnr3-container)), and their test |

The files are kept byte for byte in step with Wex-WRN2's `c/` folder (only the line endings,
that repository's licence header, and the include guards and two comments, reworded here, differ); a change to the format goes there first, with its
specification and its conformance vectors. Comments in them that name a `docs/` or `wex4/` path
mean a path of that repository. `wex_decode.c` and `wex_decode.h` are the exception: Wex-WRN2's
copy also loads an optional CUDA helper that decodes on the GPU, which is not here (the GPU decode
comes later), and colibri's adds `coli_wex4_register_model` and the `COLI_WEX_DECODE_API` macro.
`wex_fast_decode.h` is the other exception: it comes from that repository's research folder,
without its gather kernel, with the AVX2 kernel chosen at run time, and with
`wex_fast_decode_record` added ([below](#the-fast-cpu-decoder)).

### Linking `wex_decode.c`

An engine that is one translation unit includes it:

```c
#define COLI_WEX_DECODE_API static inline
#include "wex_decode.c"
```

so every engine has its own model registry and no two engines linked into one library (the
segment adapters link qwen38 with DeepSeek V4) define the same symbol. A container registers its
model at open and decodes in the same unit. Compiled on its own (`cc -c wex_decode.c`) the
functions are ordinary external ones.

### WNR3 on the CPU

A WNR3 record (codec 5) is an MXFP4 expert record coded with rANS against a small model shared
by the container (a WNM2 blob, 59,408 bytes). Every stream carries a 23-bit check, so a damaged
record is refused rather than decoded to other bytes; a canonical-zero (`CZ`) record stores the
FP4 value -0 as +0, so it decodes to the same values, not the same bytes (the container's records
say so with their value-exact flag). `coli_wex_decode_record` decodes a WNR3 record on the CPU
with the fast decoder below, one record per thread, where the CPU has its kernel, and with the
reference one of `wex_neural_rans.h` with `COLI_WEX_FAST_DECODE=0` or on other CPUs: on an Intel
i7-7700K, one 18.8 MB DeepSeek V4.1 Flash expert takes about 100 ms on one thread with the fast
decoder and 740-760 ms with the reference (24.7 MB/s), and the first WNR3 record an engine decodes
prints which one it is:

```
[wex] WNR3 records decode on the CPU with the fast decoder (AVX2; COLI_WEX_FAST_DECODE=0: the reference)
```

How many streams the fast decoder interleaves in a record is a matter of speed only, and the best
count depends on the core (4 on the x86 desktops of Wex-WRN2's measurements, 6 on a Neoverse-N2
and on a 4-vCPU Xeon VM, 8 on Apple M1 VMs). Unless `COLI_WEX_FAST_LANES` fixes it, the first 16
records the fast decoder decodes try 3, 4, 6 and 8 in turn on whichever threads decode them,
each timed on its thread's CPU clock (so other readers on a busy machine do not skew it), and the
count with the lowest time per decoded byte is used from then on (DeepSeek V4.1 REAP-192 records,
8 readers on a 4-vCPU Xeon VM, from memory: 6 streams, 25.6 ms a record against 29.0 with 4):

```
[wex] fast decoder: 6 streams a record (measured on the first 16 records, CPU time per byte against it: 3: 1.41 4: 1.17 6: 1.00 8: 1.31; COLI_WEX_FAST_LANES=N fixes it)
```

DeepSeek V4.1 Flash REAP-192 with the published WNR3 container on an NVMe drive (the trunk
from the shards on a SATA SSD), CPU only, i7-7700K, 8 expert slots, 8 greedy tokens: decode
in 71.1 s with the fast decoder against 293.2 s with the reference, the same tokens; the
shards alone took 62.5-91.6 s in the same session. The CUDA helper, when
`COLI_WEX_CUDA_LIB` names it, still comes first.

### The fast CPU decoder

`c/wex_fast_decode.h` decodes the same WNR3 records to the same bytes, and refuses the same
ones, several times faster than the reference: per-context lookup tables built once per model
(224 KiB), a branchless symbol search (one AVX2 compare, or two NEON ones), four streams
interleaved in one loop, the stream hash and the ceil inverse fused into it. It comes from
Wex-WRN2's `wex4/research/fast-cpu-decode/` (whose RESULTS.md has its validation and
measurements) with its kernels 0 and 1; the gather kernel (AVX2 and SVE), slower than kernel 1
where it was measured, is not carried.

| Target | What runs |
|---|---|
| x86-64, GCC or clang | the AVX2 kernel when CPUID says the CPU has AVX2, BMI1, BMI2 and LZCNT and the OS saves the YMM state; it is compiled with a target attribute, so a build without `-mavx2` (`ARCH=x86-64`) has it too. Otherwise the scalar kernel |
| little-endian AArch64 | the NEON kernel (NEON is in the base instruction set) |
| anything else, or `-DWEX_FAST_SCALAR` | the scalar kernel: the reference's loop, stream by stream |

`coli_wex_decode_record_to` (wex_decode.c) decodes a record held in a buffer of its stored
size straight into the caller's six buffers (the three matrices' scales and weights): the same
decoders, checks and bytes, without the page of the decoded size the in-place call needs and
without the copy out of it; DeepSeek V4.1 and Qwen3.8 read WNR3 records through it. On -1 the
buffers may hold part of a decode, so the caller must not use them (both engines then read the
expert again, or stop). `wex_fast_decode_record` takes a record in its slab as `wex_decode_nr2_record` does (same
checks, scratch, slab untouched on -1). `tests/test_wex_fast_decode` checks it in three builds
(default flags, the x86-64 baseline, scalar) against the reference: synthetic records with six
chunk sizes, plain and canonical zero, every variant on 1 and 8 threads; 512 single-bit changes;
mini-records cut from decoded records, re-encoded, then damaged byte by byte (the same status,
the same bytes when both accept), every buffer against a guard page. With a container it adds
real records:

```sh
make -C c tests/test_wex_fast_decode
c/tests/test_wex_fast_decode --mutate 3000 experts.nr3.wex 8   # 8 real records, mini-records cut from them
c/tests/test_wex_fast_decode --bench experts.nr3.wex 16        # ms per record, reference against fast
```

### Counting the decodes

`COLI_WEX_STATS=1` makes `wex_decode.c` print one line on stderr when the engine exits:

```
[wex] stats: helper=2257 device=0 fast=0 reference=0 failed=0 cpu=AVX2
```

the neural records decoded by the CUDA helper into host memory (`helper`) and into device
buffers (`device`), by the fast CPU decoder, by the reference decoder, the ones refused or not
decoded (`failed`), and the CPU decoder at exit. It is the same for the three engines (the line
comes from the one call they make); a RAW container prints no line. Unset or `0`: nothing.

### Measuring

`c/tools/wex_bench.py` runs the decode, read, verify and engine measurements on one machine and
writes a report for #2860; `c/tools/wex_decode_bench` is its C part ([wex-bench.md](wex-bench.md)).

## Checking a container

```sh
make -C c tools/wex4_verify
c/tools/wex4_verify experts.wex4 [THREADS]
```

It reads the file once, read only, on THREADS threads (default 8): the header page and the index
are checked as an engine checks them at open, then every blob and every record against its
SHA-256 in the index. Exit 0 and a last line `COLI_WEX_ROOT=<64 hex digits>` when everything
matches; 1 when some record or blob does not (each one is listed); 2 when the file does not open
as WEX4 (a WEX1-WEX3 file, a damaged header page or index, a truncated copy).

Two copies of a container (a mirror on another drive) are the same file when they print the same
root.

## Making a container

colibri does not write WEX4 files yet. Wex-WRN2's `wex4/wex4_convert.py` (Python, standard
library only) writes them: `pack` from safetensors files, `upgrade` in place from a WEX1 file.

### Making a WNR3 container

The WNR3 encoder is here: `c/tools/wex_nr3_tools.c` (a small C library: prepare, train, encode
and verify one expert record with the reference codec of `wex_neural_rans.h`) and
`c/tools/build_nr3.py` (Python, standard library only, which drives it). Both are Wex-WRN2's
copies.

```sh
make -C c wex-nr3-tools                       # tools/wex_nr3_tools.dll, libwex_nr3_tools.so or .dylib
python3 -I c/tools/build_nr3.py c/tools/libwex_nr3_tools.so experts.wex container experts.nr3.wex \
        --layers all --train-lab80 [--cz] [--save-model model.wnm2]
```

`build_nr3.py` reads the RAW records of a WEX1 container, trains one WNM2 model on the experts
`--train L:E,...` lists (or `--train-lab80`: 80 seeded picks over every layer), encodes every
record as WNR3 on 8 threads, decodes each one again on the CPU and compares it with its source
before writing it, and writes a WEX3 container (header flags 33, 97 with `--cz`) with a JSON
report beside it. A record that would not be smaller stays RAW. `--cz` (canonical zero) stores the
FP4 code -0 as +0: smaller, value-exact, but the RAW bytes can then only be rebuilt from the
checkpoint. Then make it WEX4, which the engines read:

```sh
python3 -I wex4/wex4_convert.py upgrade experts.nr3.wex       # Wex-WRN2: in place, records do not move
c/tools/wex4_verify experts.nr3.wex                            # prints COLI_WEX_ROOT=<root>
```

or, with a model saved by `--save-model`, write the WEX4 file straight from the safetensors
with Wex-WRN2's `wex4/wex4_convert.py pack OUT.wex4 SHARDS... --experts nr3 --nr3-dll <library>
--model model.wnm2 [--cz]`. The WEX1 input comes from Wex-WRN2's `c/tools/repack_v4_wex.py`
(DeepSeek V4/V4.1) or `repack_q38_mxfp4_wex.py` (Qwen3.8). On an i7-7700K, 8 DeepSeek V4.1
records (150 MB) took 3.6 s, the training and the check decode included; a whole V4.1 REAP-192
container is 7,680 such records.

## DeepSeek V4 (`deepseek_v4`)

```sh
c/tools/wex4_verify experts.wex4                    # once: prints COLI_WEX_ROOT=<root>
COLI_WEX_PATH=experts.wex4 COLI_WEX_ROOT=<root> ./deepseek_v4 ...
```

| Variable | Default | Effect |
|---|---|---|
| `COLI_WEX_PATH` | unset | the container; unset, the store reads the safetensors as before |
| `COLI_WEX_ROOT` | unset | 64 hex digits: a container with another root is not used; turns `COLI_WEX_VERIFY` on |
| `COLI_WEX_VERIFY` | on with `COLI_WEX_ROOT`, else off | `1`: each record's SHA-256 against the index at its first read |

What the expert store does with it (`deepseek_v4.c`, the `v4_wex_*` functions):

```
open      v4_wex_open    read page 0 -> wex4_parse_header -> grid = the model's? root = COLI_WEX_ROOT?
                         read the index -> wex4_open -> for every expert record: wex4_expert_entry
                         WNR3 records: their one WNM2 blob read, checked against the index's SHA-256
                         and registered (coli_wex4_register_model); all or none value-exact
                         (any failure: a [v4_wex] line, and the safetensors serve every expert)
build     build_record   every expert record in the per-matrix slab layout
                         (w1 w2 w3 scales, then w1 w2 w3 weights: a WEX record's layout)
bind      v4_wex_bind    every container record must have its expert's shapes and sizes, else none is used
miss      v4_wex_read    RAW: one read of the record (O_DIRECT where there is one) into the slot's slab;
                         padding must be zero; first read: SHA-256 against the index (COLI_WEX_VERIFY)
                         failed read, bad padding or bad hash -> that expert from the safetensors
          v4_wex_read_nr3  WNR3: the stored bytes read at the end of the decoded page, first read:
                         SHA-256 (COLI_WEX_VERIFY); the record's owner and canonical-zero flag
                         (wex_nr3_owner_ok); decoded in place (coli_wex_decode_record) on the
                         thread that missed. Bad hash, another owner or a failed stream check ->
                         that expert from the safetensors for the rest of the run
prefetch                 the advice goes to the container's record instead of the shard ranges
```

An expert the container does not hold, or one it refused, comes from the safetensors, so the
checkpoint must stay where it is. The store prints one `[v4_wex]` line at open (the file, how
many experts it holds, how many are WNR3 and their model, its root, whether the hash check is
on) and one at the end (records read, WNR3 records decoded, bytes, fallbacks, refusals). A
canonical-zero container is refused under `DSV4_CUDA_MOE_MASKED=1`, whose masked MoE path is
not proven value-exact.

## Qwen3.8 (`qwen38`)

The container takes the place of the `experts-mxfp4/` sidecar ([qwen38.md](qwen38.md)): its
records are the same bytes (gate, down, up scales, then their e2m1 codes, 2,611,200 bytes on the
release), one 4096-aligned record per expert in one file. The trunk, the shared expert and the
MTP head stay the FP8 snapshot's.

```sh
COLI_WEX_PATH=experts-mxfp4.wex4 COLI_WEX_ROOT=<root> ./qwen38 ...
```

| Variable | Default | Effect |
|---|---|---|
| `COLI_WEX_PATH` | unset | the container, read as the MXFP4 experts (implies `Q38_EXPERT_LOWBIT=mxfp4`; any other low-bit format is refused); unset, nothing changes |
| `COLI_WEX_ROOT` | unset | 64 hex digits: a container with another root stops the engine; turns `COLI_WEX_VERIFY` on |
| `COLI_WEX_VERIFY` | on with `COLI_WEX_ROOT`, else off | `1`: each record's SHA-256 against the index at its first use, read or mapped (`COLI_MAP_EXPERTS=1`) |

`q38_wex4_attach` (`qwen38_core.h`) checks the header page, the index, the grid, the root, and
that every expert of the loaded layers is a record of exactly that layout; then the records
are read and bound by the sidecar's own code (`q38_x4_read`, `q38_bind_int4_record`), so the
CPU kernels, the batch loader and the mapped path are the sidecar's. As with a sidecar that was
asked for, anything that does not hold stops the engine with the reason; the qwen38 engine has
no second copy of the MXFP4 experts to fall back to. A copy of the container on another drive
splits the reads ([below](#the-container-on-several-drives)).

A container of WNR3 records: at attach its one WNM2 blob is read, checked against the index's
SHA-256 and registered (all or none of the records value-exact). `q38_x4_read` sends each WNR3
record to `q38_x4_read_nr3`, which reads the stored bytes into a buffer of their size, checks
its SHA-256 at the first use (`COLI_WEX_VERIFY`), its owner and canonical-zero flag, and decodes it
on the reading thread straight into the expert's buffer (`coli_wex_decode_record_to`); the mapped path (`COLI_MAP_EXPERTS=1`)
reads and decodes a WNR3 record instead of mapping it. A record that names another expert or does
not decode stops the engine, as a changed RAW record does under the check. Whether it pays:
Wex-WRN2 measured WNR3 slowing Qwen3.8 down on its test machine (the records are 2.6 MB and
are read often); the published Qwen3.8 container holds RAW records.

A container for Qwen3.8 is written by Wex-WRN2's `wex4/wex4_convert.py pack --experts raw
--names qwen38` from the sidecar's layer files or from AMD's Quark checkpoint. Its record layout
needs the experts' widths to be multiples of 32 (true of the release: hidden 2560,
moe_intermediate 640), so the tiny fixture (`qwen38_tiny_mxfp4`, moe_intermediate 8) has no
container form; `tests/test_qwen38_wex4.c` uses a synthetic grid instead.

## GLM-5.3-Flash (`glm53`)

glm53 reads AMD's MXFP4 checkpoint, `amd/GLM-5.3-Flash-Quark-MXFP4`, as it is published: each
routed expert's three matrices are `name.weight` (U8 e2m1 codes, two per byte, low nibble first)
and `name.weight_scale` (U8 ue8m0 exponents, one per 32 columns), streamed like the int4
container's and computed by `matmul_mxfp4` (fmt 7); the shared experts, also MXFP4 there, are
decoded at load and held at `GLM53_BITS` like the other dense matrices
([glm53-flash.md](glm53-flash.md)). A WEX4 container of that checkpoint, written by Wex-WRN2's
`wex4/wex4_convert.py pack --experts raw|nr3 --names qwen38`, then serves the routed experts:

```sh
COLI_WEX_PATH=glm53-wnr3.wex4 COLI_WEX_ROOT=<root> ./glm53 --model <checkpoint> --prompt "..."
```

| Variable | Default | Effect |
|---|---|---|
| `COLI_WEX_PATH` | unset | the container, read for the routed experts (an MXFP4 checkpoint's only; with the int4 container or a GGUF the engine stops); unset, nothing changes |
| `COLI_WEX_ROOT` | unset | 64 hex digits: a container with another root stops the engine; turns `COLI_WEX_VERIFY` on |
| `COLI_WEX_VERIFY` | on with `COLI_WEX_ROOT`, else off | `1`: each record's SHA-256 against the index at its first read |
| `COLI_WEX_CUDA_LIB` | unset | the CUDA helper for the WNR3 records ([below](#the-cuda-helper-wnr3-records-on-an-nvidia-gpu)) |

`g53_wex4_attach` (`glm53.c`) checks the header page, the index, the grid, the root, and that
every expert of the loaded sparse layers is a record of the slab layout (gate, down, up scales,
then their codes: 13,369,344 bytes on the release). `expert_read`, the one place every path takes
an expert from (the CPU's MoE, the Vulkan and CUDA expert tiers and their warm start, an expert
worker), then reads the record whole into the slot (`g53_wex_read`), hashes it at its first read
when asked, decodes a WNR3 record in place through `coli_wex_decode_record` (the helper first when
`COLI_WEX_CUDA_LIB` names it, the CPU otherwise) and points the six pieces into it. A record that
fails its hash, names another expert or does not decode stops the engine: glm53 has no second
copy of the experts to fall back to. The run's last line counts the records:
`[glm53] WEX4 records: N read, M WNR3 decoded (CUDA helper H, CPU C)`. `COLI_MODEL_MIRROR`
copies are for the shards; the container is read from its one path. `tests/test_glm53_wex4.c`
covers the shards' MXFP4 experts, RAW and WNR3 containers and the refusals.

## DeepSeek V4.1 (`deepseek_v41`)

```sh
COLI_WEX_PATH=v4_flash_experts.wex4 COLI_WEX_ROOT=<root> ./deepseek_v41 ...
```

The variables and their defaults are DeepSeek V4's (above). The engine keeps an expert in six
windows, one per tensor, and reads them on its reader threads (`V41_READ_DEPTH`); with a
container, `v41_wex_open` turns each expert record's six tensors into six entries over the
container's one file, so `expert_fetch` reads them exactly as it reads the shards' tensors, only
from one contiguous record:

```
open      v41_wex_open      header page, index, grid (n_layers x n_routed), COLI_WEX_ROOT,
                            every expert record RAW or WNR3 with the checkpoint's shapes and part
                            sizes; WNR3: the WNM2 blob checked against the index and registered
                            (any failure: a [v41_wex] line, and the shards serve every expert)
miss      expert_read_list  a backbone expert the container holds -> its six tensors in the container
          expert_fetch      RAW: the same windows, the same readers, O_DIRECT where there is one
          v41_wex_check     COLI_WEX_VERIFY, first read: the six parts in record order and the zero
                            padding against the index's SHA-256, the experts of a batch on the
                            default OpenMP team; a mismatch -> that expert read again from the
                            shards, and from the shards for the rest of the run
          v41_wex_fetch_nr3 WNR3: one reader takes the expert's first part and reads the whole stored
                            record into a pooled buffer of its size (one O_DIRECT read), hashes it at
                            the first read (COLI_WEX_VERIFY), checks its owner and canonical-zero flag
                            and decodes it straight into the six windows (coli_wex_decode_record_to);
                            a failure -> that expert read again from the shards, and from them for
                            the rest of the run
```

The DSpark stages, a GGUF's experts and every other tensor stay on the shards; so do the
Vulkan tier's prewarm reads. With `V41_READ_DEPTH` readers (8 by default) up to that many WNR3
records decode at once; each holds only the stored record (about 16 MB on V4.1 Flash REAP-192)
from the reader pool while it runs: the decode writes the windows directly, with no page of the
decoded size and no decoder scratch (two 18.8 MB buffers a reader before). The run prints a last line
`[v41_wex] container: N records read (M WNR3 decoded), K refused`.

## The container on several drives

`COLI_MODEL_MIRROR` (qwen38 and deepseek_v41) takes byte copies of the container too: a
directory in the list that holds a file with the container's name, its size and its header
page becomes a read replica of it, and each expert's record is read from the drive a hash of
its (layer, expert) picks, in the shares `COLI_DISK_WEIGHTS` sets ([ENVIRONMENT.md](ENVIRONMENT.md)).
The same expert always comes from the same drive; the bytes are the same on every copy, so the
tokens do not change. The copy does not need the checkpoint's shards beside it.

```sh
cp experts.wex4 /mnt/second/                       # byte copy, same file name
COLI_WEX_PATH=/mnt/first/experts.wex4 COLI_WEX_ROOT=<root> \
COLI_MODEL_MIRROR=/mnt/second COLI_DISK_WEIGHTS=1,1 ./deepseek_v41 ...
```

A copy whose size or header page differs is skipped with a `[MIRROR]` line. The header page
holds the index's hash, so two copies with the same page have the same index and the same
record offsets; a record damaged on the copy only is caught by `COLI_WEX_VERIFY` (on with a
pinned root) at its first read from that copy, and then handled as on the primary
(deepseek_v41: read from the shards; qwen38: the engine stops).

deepseek_v41 registers each directory once for both its shards and the container, so a
directory may hold either or both, and an expert's tensors and its record are routed to the
same replica. Without `COLI_DISK_WEIGHTS` it probes each drive at start, on the container's
copy where there is one. qwen38 does not probe: equal shares, or the weights given.

Pick the drives with care: the split helps when the drives are of comparable speed, or to add
a fast drive to a container on a slow one. Reading the 192 experts of one DeepSeek V4.1 layer
through `deepseek_v41`'s expert reader on a desktop with a SATA SSD and an NVMe drive (batches
of 6 experts, O_DIRECT, Windows 10): about 0.5 GB/s from the SATA copy alone, 1.8-1.9 GB/s
with the NVMe copy added (split 15/85 by the probe), and 2.5-2.7 GB/s from the NVMe copy alone,
so there the NVMe drive alone was the better choice.

## The CUDA helper (WNR3 records on an NVIDIA GPU)

A WNR3 record is decoded on the CPU at tens of MB/s a thread (Wex-WRN2's
[BENCHMARKS](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/BENCHMARKS.md)); the CUDA
helper decodes it on the GPU, into a host slab or straight into device buffers. It is an
optional shared library, never linked into an engine:

```sh
make -C c wex-cuda-helper      # kaira_wex_cuda.dll (Windows, from a vcvars64 shell) / kaira_wex_cuda.so
make -C c wex-cuda-test        # the helper and tests/test_wex_nr3_gpu, run on synthetic records
```

| File | What it is |
|---|---|
| `c/tools/wex_cuda_bridge.cu` | the helper: the exported functions below, the workspace pools, the model registry, the batch lanes |
| `c/wex_cuda.cuh` | its kernels (WNR1, WNR2 and WNR3 records; the commit kernel) |
| `c/tests/test_wex_nr3_gpu.cu` | every entry point against the CPU decoder of `wex_neural_rans.h`, on synthetic, crafted and (given a records file) real and mutated records |

The ABI (`wex_cuda_helper_abi()` returns 1) is `wex_cuda_decode_to_host`; everything else is
announced by a bit of `wex_cuda_helper_features()` and resolved by name, so a caller works with
an older helper that lacks a bit:

| Bit | Function | What it does |
|---|---|---|
| 1 | `wex_cuda_decode_to_device` | one record into six caller-owned device buffers (three matrices' scales and weights) |
| 2 | `wex_cuda_host_register` / `_unregister` | page-lock a caller's staging buffer |
| 4 | `wex_cuda_register_model` | WNR2 records (codec 4) after their WNM2 model is registered |
| 8 | `wex_cuda_decode_batch_to_device` | n records into device buffers, pipelined over 8 lanes, one host wait |
| 16 | (no new function) | WNR3 records (codec 5) in every decode entry point, with their per-stream hash, ceil-form and canonical-zero checks |
| 32 | `wex_cuda_decode_batch_to_host` | n records into caller host buffers (each record's canonical slab) |

Every entry point returns 0 (decoded and written), -1 (an invalid record or one that fails an
integrity check: the destination is untouched) or 1 (CUDA or memory unavailable, or the model
not registered: the caller decodes on the CPU). The device entry points write through a commit
kernel that runs only when the decode's error word is 0; the host batch copies a record back
only after the host has seen its error word. Each call takes a workspace from a pool and gives it
back, so the helper holds one per call in flight at once, never one per thread that ever called
it (VRAM outside any backend's accounting, kept until the process exits: on DeepSeek V4.1
records, about 36 MiB per single-record call and 290 MiB per batch call in flight); engines whose
reader threads are retired and respawned stay within that. The bytes and the order of the checks:
[WNR3-SPEC](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WNR3-SPEC.md), sections 4.7
and 4.8.

### How an engine finds it

Only when asked: `wex_decode.c` opens the helper `COLI_WEX_CUDA_LIB` names, once per process,
the first time a WNR3 container's model is registered or a neural record decoded (so a RAW
container, or no container, never loads it). Nothing is searched for:

| `COLI_WEX_CUDA_LIB` | What is loaded |
|---|---|
| unset, or `0` | nothing: every record on the CPU |
| a path | that file; one that does not load, or is not ABI 1, is not used, with a `[wex]` line |

```sh
make -C c wex-cuda-helper          # c/kaira_wex_cuda.so (Windows: c/kaira_wex_cuda.dll)
COLI_WEX_CUDA_LIB=$PWD/c/kaira_wex_cuda.so COLI_WEX_PATH=experts.wex4 ./c/deepseek_v41 ...
# [wex] CUDA helper .../c/kaira_wex_cuda.so: ABI 1, features 63
```

`coli_wex_decode_record` then sends each WNR2/WNR3 record whose model the helper holds to
`wex_cuda_decode_to_host`, and decodes it on the CPU when the helper does not (status 1: no
GPU, no memory; the helper wrote nothing). The bytes are the same either way, so tokens do not
depend on the helper. On 16 records of the published DeepSeek V4.1 container, an RTX 3070
decoded a record in 6-9 ms where the CPU reference decoder took about a second. The helper's
other calls are `coli_wex_cuda_decode_to_device`, `_decode_batch_to_device`,
`_decode_batch_to_host` and `_host_register` (`wex_decode.h`). Windows and Linux only.

## Status in colibri

DeepSeek V4, DeepSeek V4.1, Qwen3.8 and GLM-5.3-Flash read RAW and WNR3 records from a WEX4 container; WNR3
records decode on the GPU when `COLI_WEX_CUDA_LIB` names the CUDA helper, on the CPU
otherwise (the fast decoder where the CPU has its kernel). qwen38 and deepseek_v41 also read copies of the container on other drives. A
container mirror for `deepseek_v4` and the engines' use of the helper's device and batch
calls are the next PRs.
