# Why this directory ships an evidence.json, not a validator manifest

The 2026-10-30 gap4 post-fix session recorded n=2 green legs for the LEGACY and
defaults arms (uvk n=1; the one-line shape n=3) — the session's VRAM gate
refused the missing reps beside the display co-tenant (recorded in the campaign
window log). `c/experiment_manifest.py` requires >=3 samples per arm by
contract, so this evidence cannot be expressed as a validator-passing
manifest.json without inventing data.

- `evidence.json` carries the same structured record (arms, env, regime, n
  disclosure, quoted deltas, cross-session provenance) for human review.
- The validator-compliant manifest-of-record for the same comparison
  (defaults vs COLI_VK_LEGACY, decode improvement) is
  `docs/experiments/dsv4-perf-2026-10-31-audit2/` (n=3/3, `ok
  (COLI_VK_LEGACY, improvement)`), reproduced independently on 2026-10-31
  from fresh builds of the PR tip and the base tip.
- Raw legs are shipped verbatim; every number in the PR body is recomputable
  from them (decode = 63/(wall - ttft)).
