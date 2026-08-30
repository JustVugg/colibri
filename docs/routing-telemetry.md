# Routing telemetry: the expert history and the trace stream

*The canonical reference for `.coli_usage` and `ROUTE_TRACE=`, and for `route_trace.h`,
the header both are implemented in. Background and design discussion in
[#700](https://github.com/JustVugg/colibri/issues/700).*

## Why one header

Six of colibrì's seven engines use this header: `colibri.c` (GLM-5.2),
`kimi_k3.c`, `inkling.c`, `olmoe.c`, `deepseek_v4.c`, and `qwen38.c`.
Qwen3.6 is the one existing exception. The learning cache described in
[tuning.md](tuning.md) is only as good as the history it reads. Before
`route_trace.h`, that history was per-engine: `colibri.c` wrote sparse text,
`inkling.c` wrote a dense binary block with an `IKU1` magic, and the other
engines wrote nothing at all, so `PIN=auto` had nothing to read on them. Both writers defaulted to the
same filename, so a model directory shared between two engines ended up with one history
in a format the other refuses.

`route_trace.h` holds the format and nothing else. It depends on the C library plus
[`compat.h`](../c/compat.h) — no `Model`, no `Cfg`, no `st.h` — because an engine already
knows its layer index, the expert ids it selected and their gates, and that is the entire
input. `compat.h` is not optional and not incidental: on Windows the CRT `rename()` fails
when the destination exists, so without its shim the history stops updating after the first
write, silently. An engine gets the history and the trace stream from a handful
of calls — `qwen38.c` needs the same route calls as the other integrated MoE engines.

## The history file (`.coli_usage`, `stats.txt`, `PIN=<file>`)

Text, one record per line, sparse — only experts with a non-zero count appear:

```
-1 <n_layers>       <n_experts>
-2 <format_version> <engine_id>
<layer> <expert> <count>
<layer> <expert> <count>
...
```

The two leading records carry the dimensions and the writing engine's identity so a
history from another model is **refused by name** rather than misread:

```
[USAGE] /models/glm/.coli_usage: written by kimi_k3, this engine is glm_moe_dsa
        — refusing (pass PIN=<path> to use it anyway)
```

### Why the header records look like that

The layer field is negative on purpose, and the field order is not arbitrary. Every reader
written against the original format is

```c
while(fscanf(f, "%d %d %u", &l, &e, &cnt) == 3)
    if(l >= 0 && ...)          /* out-of-range records are discarded */
```

so a record with a negative layer is parsed, dropped, and the loop continues into the
data. That is what lets a history written today still load in a binary built before the
header existed — which matters, because a `.coli_usage` accumulated over weeks *is* the
value of `PIN=auto`.

Two consequences worth knowing before editing the format:

- **No field may be non-numeric.** A string makes `fscanf` return < 3, which ends the loop
  and silently drops every record after it. That is why the engine is identified by a hash
  rather than by name, with the names kept in `route_trace.h` so a mismatch can still be
  reported in words.
- **The hash must stay in the third field.** Old readers parse the second field with `%d`,
  and a 32-bit hash routinely exceeds `INT_MAX` — undefined behaviour there. The version
  (small) goes second, the hash third. `glm_moe_dsa` hashes to `3815245270`, which is above
  `INT_MAX`, so this is not hypothetical.

`tests/test_route_trace.c` asserts the contract against a literal copy of the old reader
loop, so breaking either rule fails CI rather than corrupting a user's history.

### Adding a header record later

The negative layer is a general mechanism, not two special cases. Every reader — including
the two that predate this header, still shipped in older binaries — advances through a record
whose first field is negative and ignores it, because each is a
`while(fscanf(f,"%d %d %u",...)==3)` guarded by `l>=0`. So a new record type can be added at
any time and older builds skip it instead of breaking, as long as it obeys the two rules above
and one more: **exactly three numeric fields, never more.**

That third rule is absolute and the failure if it is broken is silent. A four-field record does
not get skipped — it leaves a field unconsumed, desynchronising the reader, which then
fabricates a record from that leftover plus the start of the next line:

```
file:                3 2 5  |  -3 7 12 99  |  4 1 8
an old reader sees:   3 2 5     admitted   (correct)
                     -3 7 12    skipped    (correct — the l>=0 guard works)
                     99 4 1     ADMITTED   <- invented: leftover 99, then two fields
                                              of the NEXT line
```

The real `4 1 8` is consumed into the garbage and a count lands on layer 99. Nothing reports
it. Measured against a literal copy of the old loop, not inferred from reading it.

What is *not* defined yet is any specific record beyond `-1` and `-2`. If a future policy needs
data the per-expert counts cannot express — a per-*transition* count for an order-1 model over
`(layer, expert)` pairs, for instance — its record number and field meanings should be agreed
before it is written, so the format grows once rather than twice. Three numeric fields is the
whole constraint; everything else is open.

### An empty history is a zero-byte file

`PIN=auto` decides whether a history is usable by testing the file size, falling back to
`stats.txt` when it is empty. A run that routed nothing therefore writes no header at all,
keeping that fallback intact.

### Older layouts that are still read

- **Legacy text**: the same triples with no header records. Accepted as-is — it cannot be
  validated, which is precisely why the header was added.
- **`IKU1`**: `inkling`'s dense binary layout (`uint32 {magic, n_layers, n_experts}` then
  `uint32[n_layers][n_experts]`). Read by `inkling` itself; any other engine refuses it by
  name rather than loading another model's ranking.

## The trace stream (`ROUTE_TRACE=<path>`)

One line per (moe call, batch row), written as the layer routes:

```
<call> <row> <layer> <expert>:<gate> <expert>:<gate> ...
```

`call` increments once per `moe()` invocation, `row` is the position within the forward
batch, and the gates are the values the layer actually applies (post-normalisation). This
is the input to `tools/route_pairs.py`, which builds the `.coli_pairs` table used by
`COUPLE=` cross-layer prefetch. Measurement only: enabling it never changes what the model
computes, and it does disable the device-side router (which would otherwise bypass the CPU
ranking the trace records).

## Using it from a new engine

```c
#include "route_trace.h"

rt_init("my_engine", n_layers, n_experts);   /* counters + identity + ROUTE_TRACE */
...
for(i) if(!layer_routes(i)) rt_drop_row(i);   /* once sparsity is known — see below */
...
rt_route(layer, row, ids, gates, k);         /* per routing decision: traces and counts */
...
rt_save(usage_path, 1);                      /* where the engine already persists */
```

### A layer with no counter row is a layer that cannot be credited

`rt_init` allocates a row per layer because it is told dimensions, not shape. An engine
usually learns which layers actually route later, while it builds them, and must then call
`rt_drop_row(layer)` for the ones that do not — dense layers, and the MTP row on a model
without MTP.

This is the load admission rule, not housekeeping. Every reader treats a NULL row as "there
is no expert here to attribute a count to". Skip the drop and a history record naming a
dense layer is silently accumulated, added to the reported total, and written back out on
the next save — a record no engine would ever emit. `colibri.c` drops its rows at the end
of `model_init`, which is the first point where `L[i].sparse` and `has_mtp` are both final
and still before any history is read.

### Trusted reads: trust follows the user, not the file

`rt_read(path, cb, ud)` applies every check. `rt_read_ex(path, cb, ud, 1)` marks the read
trusted, which relaxes exactly one of them. The checks fall into two kinds and only the
first kind bends:

- **Identity** — *whose* history this is. The refusal says *"pass `PIN=<path>` to use it
  anyway"*, so a trusted read must honour the file, or that sentence sends the user in a
  circle. It is also the only way to try one engine's placement priors on another, which is
  a thing worth being able to try on purpose.
- **Parse geometry** — the format version, and the dimensions of an `IKU1` file, where a
  record's layer and expert come from its *position* rather than from the record. Never
  relaxed. Trust says which history the user accepts, not that this build can cut the bytes
  up correctly, and a layout read as the wrong version is misread silently. Declared text
  dimensions that do not match are refused for the same reason, and neither refusal ever
  offered a way past it.

What makes a read trusted is **how the path was reached, not what the file contains**.
`PIN=<file>` was typed by the user, so it is trusted. `PIN=auto` and the `AUTOPIN` history
are paths the engine found on its own — nobody vouched for those, so they get the full
check. `colibri.c` passes that distinction into `pin_load` as an argument for this reason:
the same function reads both kinds of path, and the file cannot tell you which it was.

The callback's own bounds always apply, so no read of any kind can write outside the
counters.

An engine whose router needs the two at different points (`colibri.c` counts before gate
normalisation and traces after) can call `rt_count()` and `rt_trace()` separately, with
`rt_trace_end()` once per `moe()` call to advance the call counter. `rt_load(path)` reads a
history back into the counters, and `rt_read(path, cb, ud)` streams records to a callback for
consumers that rank rather than accumulate. A callback returns 1 when it accepts a record,
and `rt_read` totals only accepted ones — the readers this replaced counted inside the same
test that admitted, and a history from a differently-shaped model otherwise inflates the
number reported to the user. Add the engine's name to `rt_engine_names[]` so mismatches
involving it can be named.
