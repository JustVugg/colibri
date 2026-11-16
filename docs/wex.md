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

## Status in colibri

No engine opens a WEX4 container yet. The engine hooks come in the next pull requests of the series.
