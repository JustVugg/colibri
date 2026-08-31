---
tags:
  - colibri
  - qwen3.6
  - qwen3.6-35b-a3b
  - moe
library_name: colibri
---

colibri container for Qwen3.6-35B-A3B (Phase 2: all layers, incl. Gated DeltaNet).
Every layer (Gated-Attention + Gated DeltaNet linear_attention) carries its
MoE/MLP block. DeltaNet weights live under model.layers.{i}.linear_attn.* and are
run by the recurrent gated-delta-rule in the colibri `qwen36` engine.

Engine: https://github.com/JustVugg/colibri (c/qwen36.c)
