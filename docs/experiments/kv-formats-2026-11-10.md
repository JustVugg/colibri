# KV rows in block formats: what they cost Qwen3.6 (2026-11-10)

DeepSeek's deployment notes for V4.1 store its KV cache in block formats: the main rows as e2m1
(fp4) with an e4m3 scale every 16 values, the sliding window's as e4m3 with a power-of-two scale
every 32 (MXFP8, 528 bytes a token), the indexer's keys as MXFP4. V4.1 was trained for them. The
question here: what would the same formats cost a model that was not, against the float16 KV
cache qwen36 keeps today.

## Method

Qwen3.6-35B-A3B (int4 gs64 container) on the CPU (ds, 16 Zen 4 cores), teacher-forced over 8,192
tokens of prose (colibri's docs, code blocks and tables dropped), one decode step a token, so every
token's attention reads the cache as stored. Each K and V row was round-tripped through the format
as it was written (an experiment build, not in the tree): the reader sees exactly the dequantized
values. Per position, the target's log probability and the top token against the float32 run;
float32 run twice gives byte-identical dumps, so the differences are the format's.

| KV rows | bytes a value | dNLL | mean \|d log p\| | top token as float32's |
|---|---|---|---|---|
| float32 | 4 | 0 | 0 | 100% |
| float16 (the default) | 2 | -0.0056 | 0.156 | 91.6% |
| e4m3, one scale a row | 1 | -0.0057 | 0.162 | 91.5% |
| MXFP8: e4m3, a power-of-two scale every 32 | 1.03 | -0.0008 | 0.159 | 91.5% |
| e2m1, an e4m3 scale every 16 | 0.56 | +0.0069 | 0.182 | 90.3% |
| MXFP4: e2m1, a power-of-two scale every 32 | 0.53 | +0.0129 | 0.213 | 89.1% |

Over 1,000 tokens of English and of Italian prose every format but MXFP4 stayed within the
float16 run's own distance from float32 (0.012 nats either way).

## Reading

Qwen3.6 moves a lot under any rounding of its KV rows: float16 alone changes a token's log
probability by 0.156 nats on average and the top token at one position in twelve, because three
of every four layers carry a recurrence that propagates every difference. MXFP8 lands where
float16 does, at half its bytes; the e2m1 formats add a fifth to a third more on top. So an 8-bit
block format is a candidate for halving the KV cache again where a model is measured the same way;
4-bit rows are not, outside a model trained for them.

The larger lever in those notes is not the format but the context: every prompt read is kept by
content and any later request that starts with it reads only what follows (qwen36's prefix store,
[qwen36.md](../qwen36.md#the-prefix-store)).
