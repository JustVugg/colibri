# WEX expert containers

A WEX container holds a MoE model's routed experts in one file, one record per (layer, expert),
each record starting on a 4096-byte page. An engine reads a missed expert with one aligned read
of one record instead of six tensor reads spread over the safetensors shards, and the record lands
in the slab in the layout the expert kernels already use.

The format and its tools come from [Wex-WRN2](https://github.com/KairaCode/Wex-WRN2), which holds
the specifications, the conformance vectors and the measurements. This page says what colibri
carries and how the engines use it. The bytes are specified there, not here:

- [WEX4-SPEC.md](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WEX4-SPEC.md): the WEX4
  container (header page, index, records, hashes).
- [WNR3-SPEC.md](https://github.com/KairaCode/Wex-WRN2/blob/main/docs/WNR3-SPEC.md): the WNR3
  lossless codec of an MXFP4 expert record.

## Files

| File | What it is |
|---|---|
| `c/wex_format.h` | the WEX1-WEX3 wire format: header, the 72-byte record entry, validation, and the lossless record codecs (scale tiles, palette3, the WNR rANS decoder). Header only, no engine dependency |
| `c/wex_palette.h`, `c/wex_neural_rans.h` | the palette and rANS primitives `wex_format.h` includes |
| `c/tests/test_wex_format.c` | the codecs and validators on synthetic records, with corruption cases |

The files are kept in step with Wex-WRN2's `c/` folder (one comment reworded); a change to the format goes
there first, with its specification and its vectors.

## Status in colibri

The headers above are not used by any engine yet. The WEX4 reader, the integrity tool and the
engine hooks come in the next pull requests of the series.
