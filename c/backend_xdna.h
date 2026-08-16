#ifndef COLIBRI_BACKEND_XDNA_H
#define COLIBRI_BACKEND_XDNA_H

/* Optional AMD XDNA2 (Ryzen AI NPU) compute lane -- host-side binding only.
 *
 * XDNA is an OPTIONAL lane, never an engine. Colibri keeps model semantics,
 * routing, expert identity, weight ownership, scheduling and fallback; the
 * helper only ever executes one already-selected, already-qualified operation.
 *
 * This header and backend_xdna.c are the C-side owners and they NEVER include
 * or link XRT. XRT lives exclusively behind an optional native helper DLL
 * (coli_xdna.dll), resolved at runtime the way backend_loader.c resolves the
 * GPU backend. An ordinary Colibri build has no XRT dependency at all, and a
 * machine with no helper, no XRT and no NPU is a normal machine, not a broken
 * one.
 *
 * Scope of THIS slice (W2-N7-I1): binding only. Nothing here opens a device,
 * reads an artifact, prepares a weight or executes anything -- see the state
 * comment on ColiXdnaBinding.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Host-side ABI generation. A helper reporting a different value is refused
 * outright: there is no partial acceptance and no speculative forward
 * compatibility, because neither could be tested here. Bump this when the
 * required entry-point set or any signature changes. */
#define COLI_XDNA_ABI_VERSION 1u

/* Symbols the helper must export, all prefixed coli_xdna_helper_ so they can
 * never collide with the host-side coli_xdna_ names in this header. */
#define COLI_XDNA_HELPER_DLL "coli_xdna.dll"

/* Loader verdict. This describes the HELPER BINDING and nothing else.
 *
 * COLI_XDNA_AVAILABLE means HELPER_ABI_AVAILABLE: a compatible helper is loaded
 * and every entry point this host requires resolved. It does NOT mean a device
 * is present, an artifact is usable, or any operation is eligible -- those are
 * separate concerns owned by later slices, deliberately not conflated here. */
typedef enum {
    COLI_XDNA_UNPROBED = 0,      /* no probe attempted yet */
    COLI_XDNA_AVAILABLE,         /* compatible helper bound (ABI only) */
    COLI_XDNA_ABSENT,            /* no helper file where we looked */
    COLI_XDNA_LOAD_FAILED,       /* file present, but it would not load */
    COLI_XDNA_ABI_INCOMPATIBLE,  /* loaded, reported a different ABI generation */
    COLI_XDNA_SYMBOL_INCOMPLETE  /* loaded and compatible, but an entry point is missing */
} ColiXdnaBinding;

/* Probe once, then answer from the cached verdict. The probe is LAZY: an
 * ordinary run that never asks pays nothing, and a run that asks pays one
 * LoadLibrary at most. Every outcome other than COLI_XDNA_AVAILABLE simply
 * means the lane is unavailable; none of them is an error the caller must
 * handle beyond continuing on its current path. */
ColiXdnaBinding coli_xdna_binding(void);

/* Stable, allocation-free label for diagnostics and tests. */
const char *coli_xdna_binding_text(ColiXdnaBinding state);

/* Release the helper. Safe when never probed, safe after a failed probe, and
 * safe to call repeatedly. Callable pointers are cleared before the module is
 * released, so no entry point can outlive the module that provided it. */
void coli_xdna_shutdown(void);

/* -- test seams -----------------------------------------------------------
 * Mirrors the coli_loader_test_* convention in backend_loader.c: exercised by
 * c/tests/test_backend_loader.py against synthetic helpers that contain no
 * XRT, so the whole binding contract is qualified without an NPU. */

/* Point the loader at a specific helper file instead of the default lookup.
 * Resets any cached verdict so the next query re-probes. */
void coli_xdna_test_set_helper_path(const char *path);

/* Cached verdict WITHOUT triggering a probe. */
ColiXdnaBinding coli_xdna_test_state(void);

/* How many times a module load has actually been attempted. The sticky verdict
 * contract is that this never exceeds 1 for a given path. */
int coli_xdna_test_load_attempts(void);

/* 1 only when a complete, compatible entry-point set is currently callable. A
 * rejected helper must leave this 0 -- binding is all-or-nothing. */
int coli_xdna_test_entry_points_bound(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_XDNA_H */
