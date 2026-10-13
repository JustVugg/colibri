# End chat reads at the terminal marker

## Context

The chat client ignores the SSE [DONE] marker and waits for network EOF. A backend that sends its completed reply but keeps the connection open therefore leaves the chat request unresolved. Unlike the existing image client, chat also leaves its body reader active if parsing or a delta callback throws.

## Decision

Stop the chat read loop at [DONE] and cancel the body reader using the image client's existing best-effort cleanup pattern. Cancel on a thrown processing/read error while preserving that original error. Keep ordinary EOF handling and returned usage/finish metadata unchanged.

## Alternatives

Require every compatible backend to close immediately, or add a timeout. Neither treats the protocol's explicit terminal message as the completion owner. A timeout additionally delays a completed answer.

## Consequences

Completed chat requests settle without a second read and release the transport. Cancellation rejection is handled without replacing the completion or original error. This does not retry requests or change generation settings. SSE framing behavior remains unchanged and is covered by a separate correction.

## Validation

The frozen dev base88f25a60776036414f62f087905a634890ca5a61 makes the actual streamChat fixture read beyond its terminal message and reject. Candidate full web suite passed100 tests across9 files; TypeScript/Vite build and whitespace checks passed. An additional actual-entry regression confirms a throwing delta handler cancels the body and a failed cancellation preserves the original error. The fixture controls the reader deterministically without real timers, model execution or network traffic. Native C/oracle checks remain separate; no such success is claimed here.

Native evidence is reused only for unchanged source: this branch and the first SSE branch share C tree77eba454ebed606fec844f0dddaf98cb1043d535. The first branch make check exited0, with CPU runner no failures and Python2364 tests/345 skips in292.803s. No second native build/check was run at this branch head. Source equivalence is recorded separately from exact-head web tests. Torch-generated teacher-forcing/greedy oracle remains unverified, so draft publication must disclose that limitation.
