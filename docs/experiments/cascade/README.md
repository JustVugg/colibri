# Colibri optimisation Cascade

Bounded transfer proposal on `feat/colibri-optimisation-cascade`, based on
`upstream/dev` `ab98f29f`, not on unmerged Dory code.

- [Results and limitations](RESULTS.md)
- [Frozen qualifying plan](PLAN.md)
- [Recipient landscape and dependencies](LANDSCAPE.md)
- [Existing transfer registry, extended under `cascade`](../dory-native/TRANSFER_REGISTRY.json)
- [Raw synthetic CPU observations](sample-screen.json)
- [Same-build paired contrasts](contrasts.json)
- [Complete-rank/edge/fallback checks](sample-edges.json)
- [Dory feedback and parent admission request](DORY_FEEDBACK.md)
- [Concrete Q36 serving admission request](SERVING_REQUEST.md)
- [Verified recipient inventory](recipient-inventory.json) and [other local Qwen candidates](local-model-candidates.json)

Draft PR: https://github.com/JustVugg/colibri/pull/3161 (human review pending).

Dory owns source discoveries; Cascade owns these trials/proposals. Parent
(Dory coordinating session) is the **single shared-code integration and GPU
admission owner**. No edits to Dory's working tree, no installed change and no
GPU job. Parent has not adopted this interface extension yet: it is a reviewable
proposal, not a claim of two running producers sharing a deployed database.

The existing Dory registry's original top-level values remain structurally
identical to the frozen Git object. IDs are preserved. The additive `cascade`
section links donor mechanisms, target compatibility, predictions, bounded
measurements, negatives, gate statuses, costs and next actions. Do not maintain
a second registry. Referenced donor artifacts remain at their immutable Git
commit; original relative links belong to that donor snapshot, not this base.

## Reproduce

Python >=3.10, C11 GCC, and the immutable donor Git object are needed for the
CPU screen. No model, GPU or Python dependency install is needed.
Fetch the donor explicitly if it is not already in the local object database:

```sh
git fetch https://github.com/g-damilano/colibri.git feature/dory-native
python c/tools/cascade/run_screen.py --cc gcc --output docs/experiments/cascade/sample-screen.json
python c/tools/cascade/check_edges.py --cc gcc
python c/tools/cascade/summarize.py
python c/tools/cascade/publish_registry.py
python c/tools/cascade/test_registry.py -v
```

`run_screen.py` pins donor `f84e14b1`, not FETCH_HEAD. Build failures print
diagnostics and abort. Put the compiler runtime DLL directory on PATH on
Windows. This run used GCC 16.2.0 from the existing Dory tools directory and
`C:/ProgramData/anaconda3/python.exe`; TEMP/TMP pointed at the approved
`C:/Users/giaco/AppData/Local/Temp/opencode` directory.
The screen source commit and executable/header hashes are in the raw result.
The later helper refactor yields the same generated-header hash, checked by
the edge runner. To reproduce the originally timed runner byte-for-byte, use
the `source_revision` recorded in `sample-screen.json`.

Evidence-only CI needs neither donor Git history nor compiler:

```sh
python c/tools/cascade/test_registry.py -v
```

No automation authorizes model/GPU timing. Parent must admit a subsequent
matched Q36 serving test before execution; the next plan is in RESULTS.md.
The concrete bounded request is now `serving-admission.json`, supported by a
full-file recipient hash and metadata inventory. Its status is
`REQUESTED_NOT_ADMITTED`; none of its model measurements exists yet.
No G2–G5, canonical CALR/AST compliance, 10/10/5 streak, deployment, model
speedup or global convergence is claimed.

## Feedback to Dory / rollback

`cascade.feedback` is the machine-readable inbox: sort-heavy Q36 opportunity,
no useful peaked-nucleus gain, private-sampler contract exclusions and the
unresolved historical external-source locator. Preserve those failures.
Review this isolated PR; parent can selectively apply the additive registry
section and test-only tools. Do not merge upstream history into Dory merely
to copy evidence. Rollback is reverting isolated Cascade commits; no runtime
rollback is needed because no runtime was changed.
