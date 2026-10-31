# Remove the legacy credential slot independently of preference writes

## Context

App mount and public preference changes call persistPublicSettings. Its one try block writes endpoint/model preferences before removing the legacy colibri.apiKey entry. Standard Storage.setItem can throw QuotaExceededError while removeItem remains usable, so a full quota skips the intended cleanup. The current app does not read that old key; its active credential is initialized in memory. No authentication bypass or new exploit is demonstrated.

## Decision

Give legacy-entry removal its own best-effort operation before public preference writes. Independently preserve preference persistence when removal is unavailable. Remove only the known old slot; keep other stored settings.

## Alternatives

Reorder operations within one try block, which lets failed removal block otherwise valid public writes, or clear all storage, which loses unrelated user settings. Separate operations match the distinct cleanup and persistence responsibilities without retries.

## Consequences

Preference quota failure no longer prevents legacy cleanup. Storage denial stays nonfatal and current credentials remain memory-only. This does not rotate a credential, alter authorization, or access any real user's browser state.

## Validation

Frozen dev88f25a60776036414f62f087905a634890ca5a61 with quota fixtures:2 failures/4 passes. Candidate full web suite101 passed across9 files; TypeScript/Vite build and whitespace checks passed. Controlled Storage fixtures reproduce either public write failing and removal failing independently; synthetic values only. WHATWG Storage.setItem/removeItem contract: https://html.spec.whatwg.org/multipage/webstorage.html. The unchanged C tree77eba454ebed606fec844f0dddaf98cb1043d535 shares first-SSE native evidence (make check exit0, CPU no failures, Python2364 tests/345 skips); no native check was rerun at this head. Torch model oracle and real browser/backend account path remain unverified.
