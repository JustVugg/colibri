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

/* -- artifact registry ----------------------------------------------------
 *
 * Colibri owns which artifact answers which semantic operation, and whether
 * that artifact is trustworthy. The helper owns none of it: it will eventually
 * be handed an already-selected, already-verified artifact and nothing more.
 *
 * "Qualified" here is a research fact, recovered from the sealed N6 evidence
 * chain, not a runtime observation. A program that compiled, loaded and
 * dispatched successfully can still be numerically wrong -- N6 measured exactly
 * that -- so compilation and dispatch success are deliberately NOT inputs. */

typedef enum {
    COLI_XDNA_DT_NONE = 0,
    COLI_XDNA_DT_BF16 = 1,
    COLI_XDNA_DT_F32  = 2
} ColiXdnaDtype;

typedef enum {
    COLI_XDNA_TARGET_NONE  = 0,
    COLI_XDNA_TARGET_XDNA2 = 1      /* AMD XDNA2 / Strix Halo class NPU */
} ColiXdnaTarget;

/* Engine-owned semantic identity of an operation. This is what makes artifact
 * selection a decision the engine makes rather than one inferred from a shape:
 * two operations with identical M/K/N are still different operations, and one
 * may never inherit the other's qualification. */
typedef enum {
    COLI_XDNA_FAMILY_NONE = 0,
    COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP = 1   /* MoE shared-expert gate/up projection */
} ColiXdnaFamily;

/* One qualified artifact. Every field is a fact about the artifact; none is a
 * fact about the device, the weight, the request, or whether running here would
 * be a good idea. */
typedef struct {
    ColiXdnaFamily  family;
    const char     *research_family;   /* N6 shape-family label, e.g. "F3" */
    unsigned        artifact_m;        /* the program's M, exactly as qualified */
    unsigned        k, n;
    ColiXdnaDtype   in_dtype;          /* activation dtype the program expects */
    ColiXdnaDtype   weight_dtype;      /* prepared-weight dtype */
    ColiXdnaDtype   out_dtype;
    ColiXdnaTarget  target;
    const char     *xclbin_name;       /* logical name, resolved under a root */
    const char     *xclbin_sha256;     /* 64 lowercase hex chars */
    const char     *insts_name;
    const char     *insts_sha256;
    /* Research qualifications. All four must hold; see the enum comment. */
    unsigned char   runtime_weight_qualified;
    unsigned char   correctness_qualified;
    unsigned char   userptr_qualified;
    unsigned char   structural_qualified;
} ColiXdnaArtifact;

/* What the engine asks about. Deliberately free of expert id, layer index,
 * model name, router state and any reuse or economic term: those would
 * over-specialize artifact identity, and economics is a separate decision. */
typedef struct {
    ColiXdnaFamily family;
    unsigned       m, k, n;
    ColiXdnaDtype  in_dtype, weight_dtype, out_dtype;
} ColiXdnaRequest;

/* STATIC verdict. The strongest value here is COLI_XDNA_STATIC_QUALIFIED, which
 * means: Colibri knows this operation, holds a matching qualified artifact
 * definition, and the bytes on disk are the bytes the research programme
 * qualified.
 *
 * It does NOT mean a device exists, a context can be created, a weight has been
 * prepared, a pointer is aligned, memory is available, or that running here
 * would be preferable. Those gates do not exist yet, and no enumerator here
 * may be read as implying them. */
typedef enum {
    COLI_XDNA_STATIC_QUALIFIED = 0,
    COLI_XDNA_STATIC_HELPER_UNAVAILABLE,
    COLI_XDNA_STATIC_UNKNOWN_FAMILY,
    COLI_XDNA_STATIC_UNSUPPORTED_SHAPE_OR_BUCKET,
    COLI_XDNA_STATIC_UNSUPPORTED_FORMAT,
    COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE,
    COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
    COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED,
    COLI_XDNA_STATIC_REGISTRY_INVALID
} ColiXdnaStatic;

const char *coli_xdna_static_text(ColiXdnaStatic state);

/* Resolve a request to a qualified artifact definition, or NULL. Matching is on
 * the full key -- family, bucket, shape and dtypes -- never on shape alone. */
const ColiXdnaArtifact *coli_xdna_registry_lookup(const ColiXdnaRequest *req);

/* 1 when every row is well formed and no two rows share a lookup key. Callers
 * must treat 0 as "the registry may not be used". */
int coli_xdna_registry_validate(void);

/* Artifact-only verdict: does a trustworthy artifact exist for this request?
 * Independent of the helper on purpose -- artifact trust is a property of the
 * artifact. artifact_root is the directory logical names resolve under; NULL
 * yields COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE rather than a search. */
ColiXdnaStatic coli_xdna_artifact_status(const ColiXdnaRequest *req,
                                         const char *artifact_root);

/* Combined static verdict: helper binding AND artifact status. Still static;
 * see the ColiXdnaStatic comment for what it does not mean. */
ColiXdnaStatic coli_xdna_static_eligibility(const ColiXdnaRequest *req,
                                            const char *artifact_root);

/* SHA-256 (FIPS 180-4). Self-contained so the registry stays portable and the
 * host acquires no new link dependency; see the audit in this slice's evidence. */
void coli_xdna_sha256(const void *data, size_t len, unsigned char out[32]);
/* 1 on success, 0 when the file cannot be read. */
int  coli_xdna_sha256_file(const char *path, unsigned char out[32]);

/* Install a registry for tests; NULL restores the production table. */
void coli_xdna_test_set_registry(const ColiXdnaArtifact *rows, int count);
/* Both must stay 0 for the whole of this slice. */
int coli_xdna_test_device_opens(void);
int coli_xdna_test_helper_calls(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_XDNA_H */
