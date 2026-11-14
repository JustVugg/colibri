# GLM-4.7-Flash parity runs

Use this directory for reproducible same-prompt comparisons between the local llama.cpp reference and Colibrì. Keep each run's prompt, sampling parameters, engine versions, token IDs, and logits/top-candidate evidence together. Do not overwrite prior runs. The first successful comparison data is held in `/tmp/glm47-*.json` and `/tmp/glm47-coli-*`; promote a run here only after its settings are normalized.

Current status (2026-11-09): one 16-token English prompt and one 32-token English prompt matched exactly. The Chinese prompt's first divergence was token 15: llama.cpp chose 17114 with logprob -0.797; Colibrì chose 1467, whose llama.cpp logprob was -0.836. Prompt token IDs match. This is a near-tie logit mismatch, not yet explained.
