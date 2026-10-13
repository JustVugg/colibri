# Assemble chat SSE data before decoding

## Context

The chat client passes every `data:` line to JSON.parse separately. A legal multiline SSE event containing one JSON response therefore interrupts an otherwise valid streamed reply. The image-event parser already assembles data lines per frame and preserves whitespace after the optional single separator space.

## Decision

Reuse the existing event parser for chat and project each complete event's data. Keep incomplete frames buffered and preserve existing CRLF and separate-event behavior. No network API, engine or inference behavior changes.

## Alternatives

Maintain a second frame parser, increasing divergence risk, or require single-line server payloads, rejecting legal compatible-backend responses. Neither provides the existing shared parser's simple ownership.

## Consequences

Multiline JSON chat completions can stream successfully. Data whitespace beyond the optional first space is retained. Existing named-event parsing is unchanged. This does not add support for lone-CR framing or other unrelated SSE features.

## Validation

Against frozen dev 88f25a60776036414f62f087905a634890ca5a61, the same API suite produced 29 passes and three failures, including the actual streamChat entry point with a fixture fetch response. Candidate full web suite passed101 tests in9 files; TypeScript and Vite production build passed. Diff whitespace check passed. WHATWG SSE field processing specifies joining data fields with line feeds and removing only one optional leading space: https://html.spec.whatwg.org/multipage/server-sent-events.html. No model downloads, provider API calls, externally reachable server, credentials or live engine execution were used. Native C/oracle checks are separate gates and not claimed here.

The unchanged native C tree77eba454ebed606fec844f0dddaf98cb1043d535 was additionally exercised by make check in this worktree: exit0; portable CPU test runner reported no failures; Python2364 tests passed with345 skips in292.803s. Full log retains warning output and skipped capability/model cases. No Torch-generated glm_tiny teacher-forcing/greedy model oracle was run, so the contribution remains a draft with that gate unverified.
