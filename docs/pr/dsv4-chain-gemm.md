# vulkan: DeepSeek V4's chain prompt rows may take the tiled GEMM (PR #2253)

*Status @ 2026-10-21: open, positively validated by mfethe1 on Apple M4 +
MoltenVK (same no-cooperative-matrix class as the campaign's RX 6900 XT):
default byte-identical (0 tiled GEMMs), the `-1` arm engages on MoltenVK
(49–73 tiled GEMMs, ids exact), wiring confirmed by mutation, deterministic.
One build-system note from that review — fixed below.*

## Follow-ups folded into the rebase onto b90a369e

1. **`make deepseek_v4` (bare) was a silent no-op.** `c/Makefile` had no rule
   for the binary name — only the hyphenated `deepseek-v4` — so the stale file
   read "up to date" and editing `deepseek_v4_chain.h` relinked nothing. The
   bare name is now a `.PHONY` alias that recurses into `Makefile.deepseek-v4`
   (whose own dependency graph is complete and always was). This is the trap
   behind the maintainer's note.
2. **ENVIRONMENT.md path nit:** the row pointed at
   `docs/experiments/gemm-rows-2026-10-20.json`; the record is the directory
   (`manifest.json` inside it). Fixed.
3. Two behavioural notes for the PR body, both new at b90a369e: with `-1`,
   GGUF dense matrices (upstream `b77dbd87`) become GEMM-eligible too, and a
   `use_prefill_pool == -1` step counts as prompt. Neither changes the
   RDNA2 campaign; both are worth one sentence each upstream.

## The change itself (unchanged)

`COLI_VK_CHAIN_GEMM_ROWS`: `0` (default) keeps every prompt-row matrix on the
per-row GEMV so a row's bits never depend on the chunking; `-1` follows the
backend's own `gemm_min_s` rule; `N` from N rows. Decode and a verify's rows
keep the GEMV's bits. Measured on the RX 6900 XT, 2,515-token prompt, three
interleaved runs per arm: first token 189.6 s median → 152.1 s with `-1`
(prefill 13.3 → 16.5 tok/s; 1,106 GEMMs vs 0), decode unchanged. Upstream's
own V4.1 chain adopted the same `vkc_gemm_rows()` pattern in this range, and
the merged tree rides the same seam.
