# Grammar-forced speculative drafts

*The canonical reference for the `GRAMMAR=` draft source (method F). History: idea in
[#48](https://github.com/JustVugg/colibri/issues/48), implementation in
[#70](https://github.com/JustVugg/colibri/pull/70), consolidated write-up with A/B
measurements and corrections in [#146](https://github.com/JustVugg/colibri/issues/146).*

## The mechanism in one paragraph

On constrained-output workloads (JSON/NDJSON, function calling, structured extraction)
the grammar itself is a draft source. A byte-level GBNF subset is compiled into a
set-of-stacks PDA; wherever the grammar admits **exactly one** legal next byte
(braces, quotes, key names, enum bodies, fixed separators), the engine walks that
forced span, tokenizes it, and injects it as pre-accepted draft tokens into the
**same batch-union verify forward** used by MTP and n-gram drafts. It never
constrains sampling: forced spans are *verified by the target model*, so a wrong,
stale, or desynced grammar cannot change the output — worst case is rejected drafts,
and an adaptive guard disables the source below 50% acceptance. It composes with
`DRAFT`/MTP, which fill the free-text gaps between forced spans.

## Why it pays disproportionately in this engine

In a disk-streaming MoE, the marginal cost of a decode step is **unique expert bytes
off disk**, not FLOPs. The batch-union forward loads each unique expert once for
*all* positions in the verify batch, and structural tokens route heavily into experts
the neighboring free-text positions already pulled in. So every accepted forced span
converts almost directly into **disk reads avoided per emitted token**. On a dense
CPU engine the same trick saves only compute; here it saves the scarcest resource.
Fewer forwards also means less LRU churn, which compounds with cache hit rate.

## Usage

```bash
GRAMMAR=fit.gbnf COLI_TEMP=0 PROMPT="..." ./colibri 64 4 8
```

- `GRAMMAR=<file.gbnf>` — arm the draft source with a grammar. Byte-level GBNF
  subset: literals (escapes `\" \\ \n \r \t \xHH`), byte classes `[...]`/`[^...]`,
  rule refs, groups, postfix `? * +`, `|`, `#` comments. Root rule must be `root`.
- `GRAMMAR_DRAFT=n` — cap the forced span per forward (default 24, max 48).
- Arming is lazy (the walker starts at the first byte the root admits, skipping
  preambles) and desync-tolerant (a non-conforming byte kills the walker for the
  current span; it re-arms at the next opportunity).
- The engine reports at end of run: `grammar: NN% acceptance (a/b forced drafts)`.

## What to expect (measured, current `main`, M3 Max — details in #146)

| workload shape | tok/forward | end-to-end |
|---|---|---|
| conforming compact NDJSON (PR #70 era) | 1.60 | large |
| current-main NDJSON with sloppy spacing / long free-text fields | 1.21–1.22, 87% acceptance | ~+5% |
| prose-dominated windows (preambles, long `reason` strings) | ~1.0 | ~nil |

**The win tracks structural-span density.** Free-text content forces nothing;
keys, separators, quotes and enum bodies force everything. Two practical levers:

1. **Prompt for compact output** (short enum-like fields, no markdown fences).
2. **Whitespace-tolerant grammars**: a compact-only grammar desyncs at the first
   stray space and forfeits every span after it. Emitting an optional-whitespace
   rule at separators costs those single bytes (two legal bytes → not forced) but
   keeps the walker alive, so every multi-byte span after them still drafts.
   Because drafts are verified, tolerance is strictly acceptance-positive.

## Guarantees and limits

- **Lossless by construction**: greedy output is byte-identical with and without
  a grammar (verified in the #70 A/B); under sampling the rejection step preserves
  the sampling distribution. The tokenization boundary of a forced byte span is
  not guaranteed to coincide with the model's — verification absorbs the
  difference (worst case the last draft of a span is rejected).
- Adaptive shut-off below 50% acceptance, so a mismatched grammar degrades to
  baseline speed, never below it for long.
- Bounded work: a grammar the walker cannot follow cheaply switches it off
  instead of holding up the engine. Past 64 live stacks (`GR_MAX_STACKS`), a stack
  deeper than 64 frames (`GR_MAX_DEPTH`, left recursion) or 65,536 steps for one
  byte (`GR_MAX_WORK`: alternates that fan out into the same stacks, a few hundred
  bytes of grammar that would walk 2^N paths; a JSON schema's grammar takes at most
  35) the request runs without grammar drafts, its output unchanged.
- Benchmarking note (learned the hard way, see #146): hold the PIN hot-store and
  page-cache warmth constant across A/B rungs and report the expert hit-rate
  column next to any tok/s claim.

## Prior art

The umbrella idea is not colibri's: **jump-forward decoding** (SGLang;
[XGrammar](https://arxiv.org/abs/2411.15100)) skips grammar-deterministic tokens by
*constraining output*, and Outlines' coalescence does the analogous FSM move. What
this implementation adds: forced spans as a **draft source verified in the target's
own forward** (lossless even under a wrong grammar, composes with MTP/n-gram in one
union batch), deployed where the win is denominated in expert I/O rather than
forward passes.

## Server usage: `response_format` (OpenAI API)

The gateway (`openai_server.py`) turns `response_format` into a **per-request**
grammar carried to the engine over the `SUBMIT` protocol (optional 7th header
field, `gbytes`; 6-field headers from older clients remain valid — the field is
additive and back-compatible in both directions):

```jsonc
{"response_format": {"type": "json_object"}}                       // generic JSON grammar
{"response_format": {"type": "json_schema",
                     "json_schema": {"schema": { ... }}}}          // compiled by schema_gbnf.h
{"response_format": {"type": "gbnf", "grammar": "root ::= ..."}}   // raw GBNF (extension)
```

Semantics are identical to `GRAMMAR=`/`SCHEMA=`: a **draft source, never a
sampling constraint**. A schema outside the supported subset, or malformed GBNF,
costs the speedup — never the request, never the output. Drafting engages for
greedy requests (`temperature: 0`); sampled requests run undrafted. Compile
overhead is negligible: ~8 µs/request for a typical schema, ~18 µs at the
32-level nesting cap (measured, M3 Max). Grammar payloads are capped at 1 MiB.
As with MTP (#100), a drafted greedy run may differ from an undrafted one in
near-tie tokens (the verify forward has a different batch shape); each output
is a valid greedy stream of its own forward shapes.
