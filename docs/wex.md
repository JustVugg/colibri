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
| `c/wex_decode.c`, `c/wex_decode.h` | WNR3 records on the CPU: the process's WNM2 model registry (`coli_wex_register_model`, and `coli_wex4_register_model` for a WEX4 container's blob, checked against its index hash) and the in-place decode of one record (`coli_wex_decode_record`) |
| `c/tests/test_wex_decode.c`, `c/tests/wnr3_fixture.h` | the registry and the decode on records made by the reference encoder; the fixture trains a model and writes the WNR3 form of a WEX4 container |
| `c/tools/wex_nr3_tools.c`, `c/tools/build_nr3.py`, `c/tests/test_wex_nr3_tools.py` | the WNR3 encoder: a C library and the Python builder that drives it ([below](#making-a-wnr3-container)), and their test |

The files are kept byte for byte in step with Wex-WRN2's `c/` folder (only the line endings,
that repository's licence header, and the include guards and two comments, reworded here, differ); a change to the format goes there first, with its
specification and its conformance vectors. Comments in them that name a `docs/` or `wex4/` path
mean a path of that repository. `wex_decode.c` and `wex_decode.h` are the exception: Wex-WRN2's
copy also loads an optional CUDA helper that decodes on the GPU, which is not here (the GPU decode
comes later), and colibri's adds `coli_wex4_register_model` and the `COLI_WEX_DECODE_API` macro.

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
say so with their value-exact flag). The decoder here is the reference one of
`wex_neural_rans.h`, one record per thread: on an Intel i7-7700K, 24.7 MB/s on one thread (761 ms
for one 18.8 MB DeepSeek V4.1 Flash expert) and about 155 MB/s with 8 records at once.

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
no second copy of the MXFP4 experts to fall back to. `COLI_MODEL_MIRROR` copies are not applied
to a container yet (one drive).

A container of WNR3 records: at attach its one WNM2 blob is read, checked against the index's
SHA-256 and registered (all or none of the records value-exact). `q38_x4_read` sends each WNR3
record to `q38_x4_read_nr3`, which reads the stored bytes at the end of the decoded page, checks
its SHA-256 at the first use (`COLI_WEX_VERIFY`), its owner and canonical-zero flag, decodes it in
place on the reading thread and copies the record out; the mapped path (`COLI_MAP_EXPERTS=1`)
reads and decodes a WNR3 record instead of mapping it. A record that names another expert or does
not decode stops the engine, as a changed RAW record does under the check. Whether it pays:
Wex-WRN2 measured WNR3 slowing Qwen3.8 down on its test machine (the records are 2.6 MB and
are read often); the published Qwen3.8 container holds RAW records.

A container for Qwen3.8 is written by Wex-WRN2's `wex4/wex4_convert.py pack --experts raw
--names qwen38` from the sidecar's layer files or from AMD's Quark checkpoint. Its record layout
needs the experts' widths to be multiples of 32 (true of the release: hidden 2560,
moe_intermediate 640), so the tiny fixture (`qwen38_tiny_mxfp4`, moe_intermediate 8) has no
container form; `tests/test_qwen38_wex4.c` uses a synthetic grid instead.

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
                            padding against the index's SHA-256; a mismatch -> that expert read again
                            from the shards, and from the shards for the rest of the run
          v41_wex_fetch_nr3 WNR3: one reader takes the expert's first part and reads the whole stored
                            record into a pooled page (one O_DIRECT read), hashes it at the first read
                            (COLI_WEX_VERIFY), checks its owner and canonical-zero flag, decodes it in
                            place (coli_wex_decode_record) and copies the six parts into the windows;
                            a failure -> that expert read again from the shards, and from them for
                            the rest of the run
```

The DSpark stages, a GGUF's experts and every other tensor stay on the shards; so do the
Vulkan tier's prewarm reads. With `V41_READ_DEPTH` readers (8 by default) up to that many WNR3
records decode at once; each holds a page of the decoded size from the reader pool and the
decoder's own scratch of the same size while it runs. The run prints a last line
`[v41_wex] container: N records read (M WNR3 decoded), K refused`.

## Status in colibri

DeepSeek V4, DeepSeek V4.1 and Qwen3.8 read RAW and WNR3 records from a WEX4 container, WNR3
decoded on the CPU (`wex_decode.c`). Their GPU decode and the I/O switches (several drives,
split reads) are the next series.
