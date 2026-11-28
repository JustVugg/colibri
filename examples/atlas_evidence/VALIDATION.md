# Validation record

Verified on **13 November 2026**, Linux x86-64, Python 3.12.14, Node.js 24.22.1.
Source dependencies are pinned in [integrations.lock.json](integrations.lock.json).

## Completed

- Installed the pinned Atlas Reweave dependency from GitHub into a fresh virtual
  environment using the documented requirements file. Its wheel built and
  installed successfully.
- **48 tests passed, zero skips** in that environment, including the actual
  RegNav CLI and Reweave implementation. See [tests.txt](evidence/tests.txt).
- Three tests traverse Colibri's actual `APIServer` and its existing scripted
  engine double. They verify successful structured capture, authentication
  before engine work, and rejection of truncated output. No real model weights
  are involved in these tests.
- The deterministic demo passes all twelve assertions. Original evidence is
  accepted; changed policy and changed source each block the dependent report.
  Both rechecks record zero pure producer calls and zero model requests while
  preserving work identity. See [summary](evidence/demo-summary.json),
  [replay records](evidence/demo-replay.json), [capture](evidence/demo-capture.json)
  and [offline review page](evidence/demo-review.html).
- RegNav interoperability creates a two-source temporary Git repository and
  verifies `UNCHANGED` before an amendment and `RECHECK_REQUIRED` afterwards,
  affecting only the claim linked to the changed source.
- An independent code review identified and corrected export-policy consistency,
  URL/Unicode interoperability, missing-dependency handling and the distinction
  between baseline evidence and current acceptance. Regression checks cover the
  corrections.

## Scope still unverified

- Real-model extraction quality, multilingual accuracy, inference latency,
  throughput, model-memory use and comparative value over other workflows have
  not been measured. This is an application integration, not an inference-speed
  claim.
- The full upstream `make -j4 check` was started on the unchanged Colibri base.
  The portable CPU engine compiled; the wider C/Python gate remains incomplete
  at this publication checkpoint. Upstream test translation units emitted
  existing unused-variable warnings. No inference-engine source is changed by
  this contribution. The upstream contribution is therefore presented as a
  draft pending that broader gate and maintainer feedback.
- The HTML renderer's escaping, rejection display and offline restrictions are
  tested. A visual browser inspection was unavailable because this environment
  refused local-file URLs. No claim of visual QA is made.
- Hash consistency is not proof of source authenticity or authorship. Exact
  quotation checks do not establish semantic support or legal accuracy.

The demonstration records contain synthetic data only. `source-manifest.json`
records hashes of the code and integration contracts used for this validation.
