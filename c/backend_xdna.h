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
 * required entry-point set or any signature changes.
 *
 * Generation 2 (W2-N7-I5) is a DELIBERATE COMPATIBILITY BREAK. Generation 1
 * required two entry points and could only answer "a helper exists"; generation
 * 2 requires seven and can open a device, wrap a weight and execute. A
 * generation-1 helper is refused outright rather than partially used: it cannot
 * perform any of the new work, and silently binding it would produce a helper
 * that reports availability and then cannot execute. */
#define COLI_XDNA_ABI_VERSION 2u

/* Symbols the helper must export, all prefixed coli_xdna_helper_ so they can
 * never collide with the host-side coli_xdna_ names in this header. */
#define COLI_XDNA_HELPER_DLL "coli_xdna.dll"

/* Prepared-host pointer alignment required by the qualified Windows XRT
 * userptr path. Scoped to that path deliberately: it is not a claim about
 * every XRT implementation. The payload SIZE need not be a page multiple. */
#define COLI_XDNA_PREPARED_ALIGNMENT 4096u

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

/* -- prepared host state --------------------------------------------------
 *
 * The prepared BF16 image is DERIVED, DISPOSABLE, accelerator-specific HOST
 * state. The stored fmt=4 tensor remains authoritative and is never touched,
 * replaced or freed by anything here.
 *
 * Three properties vary independently and must never be collapsed into one
 * flag: whether memory is ALLOCATED, whether its contents are VALID, and how
 * many host bytes are consumed. An invalid buffer still costs memory, and no
 * amount of successful allocation makes contents valid. */

typedef enum {
    COLI_XDNA_PREP_UNPREPARED = 0,  /* no contents; may or may not hold capacity */
    COLI_XDNA_PREP_PREPARING,       /* a producer owns the destination */
    COLI_XDNA_PREP_VALID,           /* a producer published complete success */
    COLI_XDNA_PREP_INVALID          /* preparation failed, or contents revoked */
    /* A future IN_FLIGHT / PINNED value belongs here; the state is a field
     * rather than a boolean precisely so it can be added. */
} ColiXdnaPrepState;

/* Opaque: the layout is an implementation detail of backend_xdna.c, so callers
 * cannot reach past the transition helpers and set a state directly. */
typedef struct ColiXdnaPrepared ColiXdnaPrepared;

const char *coli_xdna_prep_text(ColiXdnaPrepState state);

/* Create an empty object. Allocates no payload: creation is not preparation. */
ColiXdnaPrepared *coli_xdna_prepared_create(void);

/* Free payload and object, clear the caller's handle. Safe on NULL, on a
 * never-allocated object, mid-PREPARING, on INVALID, on VALID, and repeatedly. */
void coli_xdna_prepared_release(ColiXdnaPrepared **p);

/* Begin a complete preparation cycle for a BF16 B[K,N] image. Allocates or
 * reuses capacity, guarantees 4096-byte pointer alignment, and moves to
 * PREPARING. Returns 0 without changing state on a rejected request.
 *
 * Beginning is the ONLY way to reach PREPARING, and publication is the only way
 * out, so a caller cannot arrive at VALID by allocating. */
int coli_xdna_prepare_begin(ColiXdnaPrepared *p, unsigned k, unsigned n,
                            ColiXdnaDtype prepared_dtype);

/* The writable destination, and only while PREPARING. NULL in every other
 * state, so a published image cannot be rewritten behind its own back. */
void *coli_xdna_prepare_dest(ColiXdnaPrepared *p);

/* PREPARING -> VALID. Refused from any other state: an INVALID image can never
 * shortcut to VALID, it must go through a complete new cycle. */
int coli_xdna_prepare_publish_success(ColiXdnaPrepared *p);

/* PREPARING -> INVALID. Capacity is retained; the bytes stay accounted. */
void coli_xdna_prepare_publish_failure(ColiXdnaPrepared *p);

/* VALID -> INVALID, keeping capacity. For a later weight replacement, format
 * change or eviction; this slice implements no trigger for it. */
int coli_xdna_prepared_invalidate(ColiXdnaPrepared *p);

/* Drop retained capacity, returning the object to UNPREPARED with zero bytes.
 * Validity and allocation lifetime are separate, so this is separate too. */
void coli_xdna_prepared_free_buffer(ColiXdnaPrepared *p);

/* -- fmt4 -> BF16 preparation ---------------------------------------------
 *
 * Turns the authoritative fmt=4 grouped-int4 weight into the prepared BF16
 * B[K,N] image, writing straight into the aligned destination above. The source
 * is read and never modified.
 *
 * This is engine representation logic and stays on the C side: the helper knows
 * nothing about nibble packing, group scales or tensor semantics. */

typedef enum {
    COLI_XDNA_PREP_OK = 0,
    COLI_XDNA_PREP_ERR_UNSUPPORTED_FORMAT,  /* not fmt=4 */
    COLI_XDNA_PREP_ERR_INVALID_SOURCE,      /* NULL pointers, zero dims, bad gs */
    COLI_XDNA_PREP_ERR_SIZE,                /* dimensions not representable */
    COLI_XDNA_PREP_ERR_ALLOC,               /* destination could not be obtained */
    COLI_XDNA_PREP_ERR_FAILED,              /* conversion failed after it began */
    COLI_XDNA_PREP_ERR_STATE                /* no object, or it cannot begin */
} ColiXdnaPrepResult;

const char *coli_xdna_prep_result_text(ColiXdnaPrepResult r);

/* Prepared dimensions derive from the tensor: K = I, N = O. They are not
 * caller-supplied, so they cannot silently disagree with the source.
 *
 * Publishes VALID only after the whole conversion succeeds. Any failure after
 * the cycle opens leaves PREPARED_INVALID with whatever partial bytes were
 * written -- which carry no authority, because state decides validity, not
 * contents. Never returns while still PREPARING. */
ColiXdnaPrepResult coli_xdna_prepare_from_fmt4(ColiXdnaPrepared *p,
                                               int fmt,
                                               const unsigned char *q4,
                                               const float *scale,
                                               int I, int O, int gs);

/* The prepared image, readable only when the state is VALID. NULL otherwise, so
 * an unpublished or revoked image cannot be consumed by accident. */
const void *coli_xdna_prepared_image(const ColiXdnaPrepared *p);

ColiXdnaPrepState coli_xdna_prepared_state(const ColiXdnaPrepared *p);
/* Logical payload bytes currently allocated: K*N*sizeof(bf16), not a rounded
 * allocator reservation. Non-zero for a retained INVALID buffer. */
size_t   coli_xdna_prepared_bytes(const ColiXdnaPrepared *p);
unsigned coli_xdna_prepared_k(const ColiXdnaPrepared *p);
unsigned coli_xdna_prepared_n(const ColiXdnaPrepared *p);

/* Publication generation: incremented every time this object publishes a VALID
 * image. Two images can occupy the same address -- retained capacity is reused
 * in place -- so an address alone does not identify contents. Anything that
 * caches a view of the prepared bytes must key on (pointer, generation).
 *
 * W2-N7-I6 added this because the I5 lane keyed its userptr wrapper on the
 * pointer alone. Re-preparing a different weight into the same buffer therefore
 * skipped the re-wrap, and the device -- which snapshots at wrap time -- kept
 * computing against a view that no longer matched the engine's image. Confirmed
 * on real XDNA2 hardware, not merely reasoned about. */
unsigned long long coli_xdna_prepared_generation(const ColiXdnaPrepared *p);

/* Defensive validator. The allocator guarantees alignment, but a buffer that
 * arrives from elsewhere, from a pool, or at an offset does not -- and a
 * misaligned pointer fails at the XRT boundary with a message about video
 * memory, which points at entirely the wrong subsystem. */
int coli_xdna_pointer_alignment_ok(const void *p);

/* Engine-side host accounting. These are HOST bytes: not VRAM, not NPU memory,
 * not an XRT device allocation. */
size_t coli_xdna_prepared_total_bytes(void);
int    coli_xdna_prepared_live_objects(void);

/* -- full hard eligibility and native execution ---------------------------
 *
 * I2 could only answer STATIC_ARTIFACT_QUALIFIED: does a trustworthy artifact
 * exist? This is the first slice that can answer the whole question:
 *
 *     CAN this operation safely run on XDNA, right now, on this machine?
 *
 * It still does NOT answer "should it". Hard eligibility and economic
 * preference are separate decisions with separate owners, and nothing here
 * consults, computes or caches a cost. Automatic selection belongs to a later
 * slice; this one executes only when an internal test control asks it to. */

typedef enum {
    COLI_XDNA_HARD_ELIGIBLE = 0,
    /* engine-side semantic gates, evaluated before anything is opened */
    COLI_XDNA_HARD_FAMILY_UNSUPPORTED,
    COLI_XDNA_HARD_M_OUT_OF_RANGE,
    COLI_XDNA_HARD_SHAPE_UNSUPPORTED,
    COLI_XDNA_HARD_FORMAT_UNSUPPORTED,      /* stored tensor is not fmt=4 */
    COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED,  /* fmt=4 but gs is not the qualified 64 */
    COLI_XDNA_HARD_LAYOUT_UNSUPPORTED,      /* fmt=4 pair layout expected; bytes are planar */
    /* artifact gates -- kept distinct because they mean different things
     * operationally: "this build does not ship that artifact", "the bytes are
     * not the bytes that were qualified", and "we never qualified this one"
     * lead to different actions, and collapsing them loses the difference
     * exactly where an operator needs it most. */
    COLI_XDNA_HARD_ARTIFACT_UNAVAILABLE,
    COLI_XDNA_HARD_ARTIFACT_INTEGRITY_FAILED,
    COLI_XDNA_HARD_ARTIFACT_UNQUALIFIED,
    COLI_XDNA_HARD_REGISTRY_INVALID,
    /* representation gates */
    COLI_XDNA_HARD_PREPARED_INVALID,
    COLI_XDNA_HARD_ALIGNMENT_INVALID,
    /* runtime gates */
    COLI_XDNA_HARD_HELPER_UNAVAILABLE,       /* absent, or would not load */
    COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE,  /* loaded, wrong generation or incomplete */
    COLI_XDNA_HARD_DEVICE_UNAVAILABLE,
    COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE,
    COLI_XDNA_HARD_WEIGHT_WRAP_UNAVAILABLE
} ColiXdnaHard;

const char *coli_xdna_hard_text(ColiXdnaHard h);

/* The qualified source group size. I4's converter accepts any gs>0 as a generic
 * representation transform; that breadth is a property of the CONVERTER, not a
 * statement about what the device was qualified on. A PREPARED_VALID image
 * built from another gs is a perfectly good host image and is still refused
 * here, because no artifact was ever correctness-qualified against one. */
#define COLI_XDNA_QUALIFIED_GROUP_SIZE 64

/* The logical-M range this lane implements, and the artifact buckets serving it.
 *
 * Row padding is legitimate because C = A x B is row-independent: output row i
 * depends only on input row i and on B. N6-A2-A1A qualified the strategy
 * physically -- 13 logical M values against an unmodified fixed-M artifact, all
 * bit-exact, with zeros written into every padded row. That argument is a
 * property of PURE GEMM and must never be extended to attention, normalisation
 * or routing.
 *
 * Row TILING (logical M above the largest bucket) is equally qualified as a
 * strategy and deliberately NOT implemented: one dispatch, one artifact.
 * Anything above the largest bucket declines to the current path.
 *
 * TWO buckets are served, and they are BUCKETS, not a range to interpolate
 * across. An M bucket is a separately compiled program: N6 saw `wa F6 M256`
 * fail to compile where its M64 sibling compiled, so a bucket that was never
 * built cannot be synthesised by rounding. Selection is therefore an exact
 * lookup of the smallest qualified bucket that holds the logical rows, and a
 * request between the buckets uses the larger one with zero padding -- never an
 * unbuilt M128. */
#define COLI_XDNA_LOGICAL_M_MIN 1
#define COLI_XDNA_BUCKET_M_SMALL 64
#define COLI_XDNA_BUCKET_M_LARGE 256
#define COLI_XDNA_LOGICAL_M_MAX COLI_XDNA_BUCKET_M_LARGE

/* Kept as the historical I5 names so the qualified-range vocabulary of the
 * earlier slices still resolves. _MAX was BOTH the range end and the bucket
 * when only one bucket existed; M1 split those meanings, and the name that
 * used to mean both now means only the small bucket. */
#define COLI_XDNA_I5_LOGICAL_M_MIN COLI_XDNA_LOGICAL_M_MIN
#define COLI_XDNA_I5_LOGICAL_M_MAX COLI_XDNA_BUCKET_M_SMALL

typedef enum {
    COLI_XDNA_EXEC_OK = 0,
    COLI_XDNA_EXEC_DECLINED,             /* not hard-eligible; not an error */
    COLI_XDNA_EXEC_DEVICE_INIT_FAILED,
    COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED,
    COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED,
    COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED,
    COLI_XDNA_EXEC_ACTIVATION_FAILED,
    COLI_XDNA_EXEC_EXECUTE_FAILED,       /* dispatch refused or threw */
    COLI_XDNA_EXEC_COMPLETION_FAILED     /* dispatched, did not complete cleanly */
} ColiXdnaExec;

/* Lane health. The narrowest model the observed failures justify, and no retry,
 * backoff or quarantine policy beyond it.
 *
 * Most failures are SINGLE-OPERATION: a wrap, a dispatch or a completion can
 * fail without saying anything about the next operation. A device that will not
 * initialise is different -- the sealed A5 taxonomy classifies it PROCESS-scoped,
 * and re-attempting it per operation is pure cost -- so it marks the lane
 * unavailable for the remainder of the process. Loader verdicts are already
 * sticky and are reported through ColiXdnaBinding, not here. */
typedef enum {
    COLI_XDNA_LANE_UNINITIALIZED = 0,  /* nothing has been attempted yet */
    COLI_XDNA_LANE_HEALTHY,            /* usable, or at worst failed per-operation */
    COLI_XDNA_LANE_UNAVAILABLE         /* device unusable for this process */
} ColiXdnaLaneHealth;

ColiXdnaLaneHealth coli_xdna_lane_health(void);
const char *coli_xdna_lane_health_text(ColiXdnaLaneHealth h);

const char *coli_xdna_exec_text(ColiXdnaExec e);

/* The production candidate, called from the GLM shared gate/up sites and
 * nowhere else. Deliberately shaped like the existing optional-lane idiom
 * (vk_matmul_qt): returns 1 when it has fully written y, 0 when the caller must
 * run its current path.
 *
 *     if(!coli_xdna_try_matmul(...)) matmul_qt(y, x, w, S);
 *
 * It NEVER calls matmul_qt itself, so there is no recursion, no double dispatch
 * and no path on which a failure here suppresses the current path.
 *
 * `family` is passed explicitly by the call site. It is never inferred from the
 * shape: two operations with identical M/K/N are different operations, and one
 * may not inherit the other's qualification.
 *
 * `slot` is the engine-owned prepared-state handle (QT::xdna), created lazily on
 * first use. The tensor's authoritative fmt=4 bytes are read and never modified.
 *
 * y is written ONLY after the helper reports successful completion, so a failure
 * at any stage leaves the caller's output buffer untouched for matmul_qt to
 * overwrite. */
int coli_xdna_try_matmul(ColiXdnaFamily family,
                         ColiXdnaPrepared **slot,
                         int fmt, const unsigned char *q4, const float *scale,
                         int I, int O, int gs, int planar,
                         float *y, const float *x, int S);

/* Release helper-owned runtime state and the transient staging buffers. Safe
 * when nothing was ever initialised, and safe to repeat. */
void coli_xdna_execution_shutdown(void);

/* -- the two internal execution modes -------------------------------------
 *
 * AF0 froze two conceptual modes and I6 defines both mechanically. Neither is
 * public: there is no --xdna flag and no COLI_XDNA variable, and the AUTO-like
 * mode is additionally inert unless the internal force control is set.
 *
 *   AUTO-LIKE   coli_xdna_try_matmul(). Any decline or failure returns 0 and
 *               the caller runs its current path. This is the production seam.
 *
 *   EXPLICIT    coli_xdna_test_attempt(). Returns the failure class instead of
 *               a handled flag, and implies NO fallback: a caller in this mode
 *               asked for XDNA specifically and gets a classified failure
 *               rather than a silent substitution. Deliberately a separate
 *               entry point rather than a mode flag, so no global state can
 *               leave the production seam in a no-fallback configuration.
 *
 * Both share one implementation, so their gate order and classification cannot
 * drift apart. */
ColiXdnaExec coli_xdna_test_attempt(ColiXdnaFamily family,
                                    ColiXdnaPrepared **slot,
                                    int fmt, const unsigned char *q4, const float *scale,
                                    int I, int O, int gs, int planar,
                                    float *y, const float *x, int S);

/* 1 only after an attempt reported successful completion. Any failure -- at any
 * stage, including one that occurs after the helper has already written output
 * bytes -- leaves this 0. No property of the bytes themselves (finite, non-NaN,
 * non-poison, plausible) can raise it: validity is a state, not a measurement. */
int coli_xdna_test_output_valid(void);

/* The lane's output staging buffer, so a test can observe that a failing helper
 * really did write real bytes into it before the failure was reported. Never
 * read by production code, which copies from it only after success. */
const float *coli_xdna_test_output_staging(size_t *floats);

/* Install a registry for tests; NULL restores the production table. */
void coli_xdna_test_set_registry(const ColiXdnaArtifact *rows, int count);

/* -- internal execution controls ------------------------------------------
 *
 * These are the ONLY way to reach the native path, and they are deliberately
 * test seams rather than user-facing controls: there is no --xdna flag, no
 * COLI_XDNA environment variable and no automatic policy anywhere in this
 * slice, so an ordinary build with a helper, a device and valid artifacts
 * present still runs exactly the path it runs today.
 *
 * Forcing bypasses ECONOMIC preference only -- which does not exist yet, so
 * today it bypasses nothing at all. It cannot bypass any hard gate: family,
 * format, group size, shape, bucket, artifact integrity, qualification,
 * prepared validity, alignment, helper ABI or device availability. */
void coli_xdna_test_set_force_execution(int on);
/* Where logical artifact names resolve. NULL (the default, and the only
 * production value) means no artifact is reachable and every request declines.
 * Artifact distribution is an open question this slice does not close. */
void coli_xdna_test_set_artifact_root(const char *root);

/* The last hard-eligibility verdict, for tests and diagnostics. */
ColiXdnaHard coli_xdna_test_last_hard(void);
/* The last execution verdict. */
ColiXdnaExec coli_xdna_test_last_exec(void);
int coli_xdna_test_dispatches(void);      /* successful helper executions */
int coli_xdna_test_completions(void);     /* of those, cleanly completed */
int coli_xdna_test_artifact_opens(void);
int coli_xdna_test_activation_preparations(void);
int coli_xdna_test_padded_operations(void);
int coli_xdna_test_fallbacks(void);       /* candidate returned 0 */

/* PER-BUCKET accounting.
 *
 * With one bucket, a single padded_ops total was unambiguous. With two it is
 * not: an S=200 operation padded to 256 and an S=40 operation padded to 64 are
 * both "padded" and waste 56 rows versus 24. Worse, the aggregate cannot say
 * WHICH artifact ran. Every accessor below takes the artifact M so the caller
 * names the bucket it is asking about; an unknown bucket returns 0 rather than
 * silently aggregating.
 *
 * padded_ROWS is reported as well as padded_OPS because the two answer
 * different questions: how many operations padded at all, and how much padding
 * was actually computed. Only the second scales with the waste.
 *
 * These are qualification diagnostics, not a product interface. */
int coli_xdna_test_bucket_dispatches(unsigned artifact_m);
int coli_xdna_test_bucket_completions(unsigned artifact_m);
int coli_xdna_test_bucket_padded_operations(unsigned artifact_m);
long long coli_xdna_test_bucket_padded_rows(unsigned artifact_m);
int coli_xdna_test_bucket_artifact_opens(unsigned artifact_m);
int coli_xdna_test_bucket_hard_eligible(unsigned artifact_m);
/* Logical M above the largest qualified bucket: the new first-decline class. */
int coli_xdna_test_m_above_range_declines(void);
/* Bucket transitions actually taken by the lane, counted on artifact reopen. */
int coli_xdna_test_bucket_switches(unsigned from_m, unsigned to_m);
/* The bucket the lane would select for a logical M, or 0 if none serves it.
 * Pure function of the compiled bucket set; touches no device and no registry. */
unsigned coli_xdna_test_bucket_for(int logical_m);
/* Zero every per-bucket counter. Tests measure one scenario at a time, and a
 * cumulative total silently turns "this scenario dispatched once" into "some
 * earlier scenario also dispatched". */
void coli_xdna_test_reset_bucket_counters(void);
/* Wrapper accounting. stale_rejects counts the case that made I6 necessary:
 * the same address carrying a NEW image, which must force a re-wrap rather
 * than be reused. */
int coli_xdna_test_wrapper_reuses(void);
int coli_xdna_test_wrapper_releases(void);
int coli_xdna_test_stale_wrapper_rejects(void);

/* Deterministic mid-conversion fault injection, for tests only. 0 disables it;
 * 1..99 fails after approximately that percentage of the conversion's rows.
 * Named coli_xdna_test_* like every other seam here, and never referenced by
 * production paths -- there is no environment variable or CLI flag that can
 * reach it. */
void coli_xdna_test_set_convert_fail_pct(int pct);
/* The destination regardless of state, so a test can inspect the partial bytes
 * a failed conversion left behind. Production code uses
 * coli_xdna_prepared_image(), which returns NULL unless the image is VALID. */
const void *coli_xdna_prepared_image_unchecked(const ColiXdnaPrepared *p);
/* Device opens, helper calls and userptr wraps became reachable in W2-N7-I5 and
 * are 0 only on paths that never reach the native lane. Conversions became
 * reachable in I4. */
int coli_xdna_test_device_opens(void);
int coli_xdna_test_helper_calls(void);
int coli_xdna_test_userptr_wraps(void);
int coli_xdna_test_conversions(void);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_XDNA_H */
