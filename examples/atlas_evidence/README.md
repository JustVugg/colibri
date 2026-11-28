# Evidence Replay

**Run a local model once. Recheck its evidence when the source or review rule changes.**

An experimental Colibri integration for researchers and maintainers reviewing
source-linked claims. It captures a local model response and the exact source
text supplied to it, checks quoted passages and offsets, and produces an offline
human review page. Optional Atlas Reweave integration preserves computation while
updating acceptance. RegNav Source Watch can identify the linked claims affected
by a later committed source change.

Version **0.1.0a1**, a public prototype. This directory adds an optional application
example. It does not change Colibri's inference engine or default dependencies.
Original example code is Apache-2.0, copyright 2026 Albert Tan Sin Hon. Colibri
remains the work of Vincenzo Fornaro and its contributors. Reweave is a separately
installed MIT-licensed dependency. Model weights retain their own licences.

## A concrete use case

A researcher records a candidate claim from a policy containing a 30-day review
period. The tool checks whether the quoted passage occurs at the recorded source
offsets. Later, a supplied snapshot changes the period to 60 days. The earlier
candidate is flagged, and its dependent report cannot be exported as currently
accepted. The captured response remains available for inspection.

A stricter quotation-length rule can also reassess that saved response without
another model request. The example demonstrates this mechanism with explicitly
invented policy text. It makes no claim about current law.

## Try the offline example

From the repository root, Python **3.11 or later**:

```sh
python3 -m examples.atlas_evidence demo --out /tmp/evidence-demo
```

Choose a new output directory each time. On Windows, replace `/tmp/evidence-demo`
with a new local directory and use `python` or `py -3`.

Open `review.html` in that directory. The JSON records show five checks:

| Case | Expected result |
| --- | --- |
| Exact quotation from the supplied snapshot | Passage check passes; semantic support remains unreviewed |
| Fabricated quotation | Rejected |
| Changed source text | Rejected pending renewed review |
| Stricter quotation-length policy | Rejected under that policy |
| Altered capture without matching hashes | Integrity failure |

This demonstration makes **zero model requests** and needs no model download,
account or API key. Its success tests software behaviour, not model quality.

## Capture from your running Colibri server

Start Colibri with an already prepared model using the upstream
[serving instructions](../../docs/api.md). The default API address is
`http://127.0.0.1:8000/v1`. This example never installs or downloads models.

Create an explicit snapshot of a permitted UTF-8 source file:

```sh
python3 -m examples.atlas_evidence snapshot \
  --id policy --file ./policy.txt --path sources/policy.txt \
  --title "Policy supplied for review" --out ./sources.json
```

Capture a response, replacing the model and revision placeholders with the actual
served model ID, model revision or digest, and Colibri commit:

```sh
python3 -m examples.atlas_evidence capture \
  --sources ./sources.json --model YOUR_SERVED_MODEL_ID \
  --model-revision YOUR_MODEL_REVISION --engine-revision YOUR_COLIBRI_COMMIT \
  --out ./capture.json
python3 -m examples.atlas_evidence check ./capture.json --out ./checks.json
python3 -m examples.atlas_evidence render ./capture.json --out ./review.html
```

If your server requires authentication, set `COLI_API_KEY` in your environment.
The key is sent only as an HTTP header and is never deliberately added to the
capture. Revision labels are operator declarations; this client does not hash
large model files or authenticate the server's implementation.

The client accepts literal loopback IPs only, refuses redirects, ignores proxy
environment variables and makes one bounded non-streaming request. Its default
60-second timeout applies to socket operations, not a hard overall runtime limit.
The prompt asks for structured claims and exact Unicode codepoint offsets.
Malformed, incomplete, invented or incorrectly located answers fail checking.
Successful transport alone does not imply usable evidence.

## Recheck without generating again

Install the optional pinned dependency in your own virtual environment:

```sh
python3 -m pip install -r examples/atlas_evidence/requirements.txt
python3 -m examples.atlas_evidence demo --with-reweave --out /tmp/evidence-replay-demo
```

For your own capture:

```sh
python3 -m examples.atlas_evidence replay ./capture.json \
  --store ./reviews.sqlite3 --out ./review-v1.json
python3 -m examples.atlas_evidence replay ./capture.json \
  --store ./reviews.sqlite3 --min-quote-chars 80 --out ./review-v2.json
```

The second command may return exit status 1 because the stricter rule rejects a
short quotation. The JSON still records the outcome. The same capture and store
retain work identities while the judgement changes. For a changed source,
create a new snapshot with the same source ID and pass `--sources ./new-sources.json`.
An empty source list represents deletion of every current source.

Reweave callbacks are pure. The model request happens before capture and outside
the engine. The replay result separates the baseline evidence from
`current_review`, current receipts and an accepted-only portable bundle.
`stats.producer_calls` counts local pure callbacks, **not** model calls, tokens,
money saved or model throughput. `model_requests` is zero for replay by design.

## Track affected claims across Git commits

```sh
python3 -m examples.atlas_evidence export-source-watch ./capture.json \
  --out ./.regnav/manifest.json
```

Commit the manifest and source files at the paths recorded in its source entries.
Exporting the manifest does not write your source files or commit them. Use
[RegNav Source Watch](https://github.com/Albertito-AI/regnav-source-watch) to compare
two commits containing the manifest. The demo writes its synthetic source file
for convenience.

Every automated evidence relation is `context`, and no human review declaration
is invented. A passed literal check does not establish that the quotation
supports the proposition. The real interoperability test creates temporary Git
commits, exports two claims, changes one source and checks that RegNav flags only
the linked claim.

## Boundaries

- Matching text and hashes do not prove truth, legal validity, authenticity,
  entailment, completeness or human authorship. A faithfully quoted false source
  can pass. Human review remains necessary.
- Current sources are explicitly supplied snapshots. No website or filesystem is
  monitored automatically. Changes outside those snapshots are invisible.
- Captures, reports and SQLite records contain source/model text in plaintext.
  Only provide material you intend to process locally. Review files before
  publishing them. Nothing in this example uploads or publishes your records.
- Hashes detect inconsistent bytes. Someone able to replace content and hashes
  can construct a different consistent capture. This is not signed evidence.
- Limits include 16 sources, 256 KiB per source, 100 claims and a 1 MiB capture.
  Actual usable source size is smaller because the capture includes both the
  sources and the request containing them. Inputs are rejected rather than
  silently truncated. Offsets count Unicode codepoints, not UTF-8 bytes.
- Existing output files are refused. The trusted local operator controls paths
  and stores; this example is not an untrusted multi-user file service.

## Testing and contribution

```sh
python3 -m unittest discover -s examples/atlas_evidence/tests -v
```

Replay tests run when the optional dependency is installed. To include real
RegNav interoperability, check out the revision in `integrations.lock.json`,
install Node.js 24, and set `REGNAV_SOURCE_WATCH` to that checkout before running
the same command. The dedicated workflow runs all of these checks.

The contribution is this inspectable integration of local inference, literal
evidence checks, changed-policy replay, dependent-report blocking and Git source
tracking. These are established ideas combined for a concrete research workflow.
No priority claim, comparative accuracy gain, speedup or user-adoption claim is
made. Useful next contributions are measured live-model trials, multilingual
quotation cases, source extraction adapters and independently labelled semantic
review datasets. Keep model inference tests clearly separate from fixtures.

See [VALIDATION.md](VALIDATION.md) for the tested scope and open limitations, and
[integrations.lock.json](integrations.lock.json) for source identities.
