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

The files are kept byte for byte in step with Wex-WRN2's `c/` folder (only the line endings,
that repository's licence header, and the include guards and two comments, reworded here, differ); a change to the format goes there first, with its
specification and its conformance vectors. Comments in them that name a `docs/` or `wex4/` path
mean a path of that repository.

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
                         (any failure: a [v4_wex] line, and the safetensors serve every expert)
build     build_record   every expert record in the per-matrix slab layout
                         (w1 w2 w3 scales, then w1 w2 w3 weights: a WEX record's layout)
bind      v4_wex_bind    every container record must have its expert's shapes and sizes, else none is used
miss      v4_wex_read    one read of the record (O_DIRECT where there is one) into the slot's slab;
                         padding must be zero; first read: SHA-256 against the index (COLI_WEX_VERIFY)
                         failed read, bad padding or bad hash -> that expert from the safetensors
prefetch                 the advice goes to the container's record instead of the shard ranges
```

An expert the container does not hold, or one it refused, comes from the safetensors, so the
checkpoint must stay where it is. The store prints one `[v4_wex]` line at open (the file, how
many experts it holds, its root, whether the hash check is on) and one at the end (records read,
bytes, fallbacks, refusals). Only RAW records are read; a container with WNR3 records is not used.

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
that every expert of the loaded layers is a RAW record of exactly that layout; then the records
are read and bound by the sidecar's own code (`q38_x4_read`, `q38_bind_int4_record`), so the
CPU kernels, the batch loader and the mapped path are the sidecar's. As with a sidecar that was
asked for, anything that does not hold stops the engine with the reason; the qwen38 engine has
no second copy of the MXFP4 experts to fall back to. `COLI_MODEL_MIRROR` copies are not applied
to a container yet (one drive).

A container for Qwen3.8 is written by Wex-WRN2's `wex4/wex4_convert.py pack --experts raw
--names qwen38` from the sidecar's layer files or from AMD's Quark checkpoint. Its record layout
needs the experts' widths to be multiples of 32 (true of the release: hidden 2560,
moe_intermediate 640), so the tiny fixture (`qwen38_tiny_mxfp4`, moe_intermediate 8) has no
container form; `tests/test_qwen38_wex4.c` uses a synthetic grid instead.

## Status in colibri

DeepSeek V4 and Qwen3.8 read RAW records from a WEX4 container. DeepSeek V4.1 comes in the next
pull request of the series; WNR3 records after it.
