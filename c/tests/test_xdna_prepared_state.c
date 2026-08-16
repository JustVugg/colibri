/* Engine-owned XDNA prepared-host state: allocation, validity and lifetime.
 *
 * The prepared BF16 image is DERIVED state. The authoritative weight is the
 * stored fmt=4 tensor and nothing here may touch it. These tests own three
 * things that must never collapse into one another: whether memory is
 * ALLOCATED, whether its contents are VALID, and how many host bytes are
 * consumed. A buffer can be allocated and invalid; an invalid buffer still
 * costs memory; and no amount of successful allocation makes contents valid.
 *
 * Nothing here converts anything, opens a device, wraps a pointer or calls the
 * helper. Contents are deterministic filler used only to prove that STATE, not
 * content, decides validity.
 */

#include "../backend_xdna.c"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

static int g_fail = 0;

static void ck(int cond, const char *what){
    printf("  %-60s %s\n", what, cond ? "ok" : "FAIL");
    if(!cond) g_fail = 1;
}

#define K_ 6144u
#define N_ 2048u

/* ---- allocation and alignment ------------------------------------------ */
static void test_alloc_alignment(void){
    printf("aligned allocation\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    ck(p != NULL, "prepared object created");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED,
       "a new object is UNPREPARED");
    ck(coli_xdna_prepared_bytes(p) == 0, "a new object owns no bytes");

    ck(coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_BF16) == 1, "begin succeeds");
    void *d = coli_xdna_prepare_dest(p);
    ck(d != NULL, "a writable destination exists while PREPARING");
    ck(((uintptr_t)d % 4096u) == 0, "destination pointer is 4096-aligned");
    ck(coli_xdna_pointer_alignment_ok(d) == 1, "defensive validator accepts it");
    ck(coli_xdna_prepared_bytes(p) == (size_t)K_ * N_ * 2u,
       "logical bytes are K*N*sizeof(bf16)");
    coli_xdna_prepared_release(&p);
    ck(p == NULL, "release nulls the caller's handle");
}

/* A logical payload that is deliberately NOT a page multiple. The pointer must
 * still be aligned; the SIZE must not be rounded up behind the caller's back. */
static void test_non_page_multiple_size(void){
    printf("non-page-multiple payload\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    /* 3 * 5 * 2 = 30 bytes */
    ck(coli_xdna_prepare_begin(p, 3, 5, COLI_XDNA_DT_BF16) == 1,
       "30-byte payload accepted");
    ck(coli_xdna_prepared_bytes(p) == 30, "logical size reported exactly, not rounded");
    ck(((uintptr_t)coli_xdna_prepare_dest(p) % 4096u) == 0,
       "pointer still 4096-aligned for a 30-byte payload");
    coli_xdna_prepared_release(&p);
}

static void test_misaligned_validator(void){
    printf("defensive alignment validation\n");
    /* A non-owning validator: the offset pointer is inside memory we own, and
     * is never freed through it. */
    unsigned char *base = NULL;
    if(posix_memalign((void**)&base, 4096, 8192) != 0 || !base){
        ck(0, "fixture allocation"); return;
    }
    ck(coli_xdna_pointer_alignment_ok(base) == 1, "aligned base accepted");
    ck(coli_xdna_pointer_alignment_ok(base + 2) == 0, "offset 2 rejected");
    ck(coli_xdna_pointer_alignment_ok(base + 64) == 0, "offset 64 rejected");
    ck(coli_xdna_pointer_alignment_ok(base + 2048) == 0, "offset 2048 rejected");
    ck(coli_xdna_pointer_alignment_ok(base + 4096) == 1, "offset 4096 accepted");
    ck(coli_xdna_pointer_alignment_ok(NULL) == 0, "NULL rejected");
    compat_aligned_free(base);
}

/* ---- size safety -------------------------------------------------------- */
static void test_size_safety(void){
    printf("size and overflow safety\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    ck(coli_xdna_prepare_begin(p, 0, N_, COLI_XDNA_DT_BF16) == 0, "K=0 rejected");
    ck(coli_xdna_prepare_begin(p, K_, 0, COLI_XDNA_DT_BF16) == 0, "N=0 rejected");
    ck(coli_xdna_prepare_begin(p, 0, 0, COLI_XDNA_DT_BF16) == 0, "zero size rejected");

    /* Genuine size_t overflow: 0xFFFFFFFF^2 * 2 exceeds SIZE_MAX on 64-bit.
     * Unchecked, this wraps to a small plausible number and the allocator hands
     * back a buffer far too small for what the caller will write -- the worst
     * possible outcome, because it succeeds. */
    ck(coli_xdna_prepare_begin(p, 0xFFFFFFFFu, 0xFFFFFFFFu, COLI_XDNA_DT_BF16) == 0,
       "product exceeding SIZE_MAX rejected");
    ck(coli_xdna_prepare_begin(p, 0xFFFFFFFFu, 0x80000000u, COLI_XDNA_DT_BF16) == 0,
       "product whose doubling exceeds SIZE_MAX rejected");

    /* Representable but unsatisfiable. Not an arithmetic error, so it must not
     * be reported as one: the allocator refuses and the object stays clean.
     * (No arbitrary size cap is imposed here -- a ceiling would be policy, and
     * policy is not this slice's to invent.) */
    ck(coli_xdna_prepare_begin(p, 0xFFFFFFFu, 0xFFFFFFFu, COLI_XDNA_DT_BF16) == 0,
       "unsatisfiable allocation fails cleanly");
    ck(coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_F32) == 0,
       "non-BF16 prepared dtype rejected");
    ck(coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_NONE) == 0,
       "invalid dtype rejected");

    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED,
       "every rejection leaves the object UNPREPARED");
    ck(coli_xdna_prepared_bytes(p) == 0, "no bytes were accounted for a rejection");
    coli_xdna_prepared_release(&p);
}

/* ---- the state machine -------------------------------------------------- */
static void test_transitions(void){
    printf("state transitions\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    /* Allocation is not validity. This is the load-bearing separation. */
    coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_BF16);
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_PREPARING,
       "begin moves UNPREPARED -> PREPARING");
    ck(coli_xdna_prepared_state(p) != COLI_XDNA_PREP_VALID,
       "a successful allocation is NOT valid");

    ck(coli_xdna_prepare_publish_success(p) == 1, "PREPARING -> VALID on publish");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID, "state is VALID");

    /* No API may hand out a writable destination for a published image. */
    ck(coli_xdna_prepare_dest(p) == NULL, "no writable destination once VALID");

    ck(coli_xdna_prepared_invalidate(p) == 1, "VALID -> INVALID");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID, "state is INVALID");

    /* THE invariant: no shortcut back to valid. */
    ck(coli_xdna_prepare_publish_success(p) == 0,
       "INVALID cannot be published straight to VALID");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID, "still INVALID after refusal");

    /* A complete new cycle is the only way back. */
    ck(coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_BF16) == 1,
       "INVALID -> PREPARING via a new cycle");
    ck(coli_xdna_prepare_publish_success(p) == 1, "PREPARING -> VALID again");

    coli_xdna_prepared_release(&p);
}

static void test_illegal_transitions(void){
    printf("illegal transitions fail closed\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    ck(coli_xdna_prepare_publish_success(p) == 0,
       "UNPREPARED cannot publish success");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED, "still UNPREPARED");

    ck(coli_xdna_prepared_invalidate(p) == 0, "UNPREPARED cannot be invalidated");

    coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_BF16);
    coli_xdna_prepare_publish_success(p);
    /* Beginning again on a VALID image is a full re-preparation, and must not
     * leave the old image reachable as valid while it is being overwritten. */
    ck(coli_xdna_prepare_begin(p, K_, N_, COLI_XDNA_DT_BF16) == 1,
       "re-begin on a VALID image is allowed");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_PREPARING,
       "and it is PREPARING, not VALID, during the rewrite");

    coli_xdna_prepared_release(&p);
}

/* ---- failure publication ------------------------------------------------ */
static void test_failure_publication(void){
    printf("failure publication\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16);

    /* Poison the destination, then fail. Content is deliberately garbage: the
     * point is that STATE decides validity, not what the bytes happen to be. */
    unsigned char *d = (unsigned char*)coli_xdna_prepare_dest(p);
    memset(d, 0xAB, coli_xdna_prepared_bytes(p));

    coli_xdna_prepare_publish_failure(p);
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID,
       "PREPARING -> INVALID on failure");
    ck(coli_xdna_prepare_publish_success(p) == 0,
       "poisoned INVALID buffer cannot be republished as VALID");
    ck(coli_xdna_prepare_dest(p) == NULL, "INVALID exposes no writable destination");

    /* Capacity may be retained; the bytes are still consumed. */
    ck(coli_xdna_prepared_bytes(p) > 0,
       "an INVALID but retained allocation still consumes host bytes");

    coli_xdna_prepared_release(&p);
}

/* ---- allocation vs validity vs consumption ------------------------------ */
static void test_three_independent_axes(void){
    printf("allocation, validity and consumption are independent\n");
    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    ck(coli_xdna_prepared_bytes(p) == 0 &&
       coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED,
       "no allocation, not valid, no bytes");

    coli_xdna_prepare_begin(p, 128, 128, COLI_XDNA_DT_BF16);
    ck(coli_xdna_prepared_bytes(p) == 128*128*2 &&
       coli_xdna_prepared_state(p) == COLI_XDNA_PREP_PREPARING,
       "allocated, not valid, bytes consumed");

    coli_xdna_prepare_publish_failure(p);
    ck(coli_xdna_prepared_bytes(p) == 128*128*2 &&
       coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID,
       "allocated, INVALID, bytes still consumed");

    coli_xdna_prepared_free_buffer(p);
    ck(coli_xdna_prepared_bytes(p) == 0 &&
       coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED,
       "capacity dropped: no bytes, back to UNPREPARED");

    coli_xdna_prepared_release(&p);
}

/* ---- release ------------------------------------------------------------ */
static void test_release_lifetime(void){
    printf("release lifetime\n");
    ColiXdnaPrepared *p = NULL;

    coli_xdna_prepared_release(&p);           /* NULL handle */
    ck(p == NULL, "release on NULL is safe");

    p = coli_xdna_prepared_create();
    coli_xdna_prepared_release(&p);
    ck(p == NULL, "release of a never-allocated object is safe");

    p = coli_xdna_prepared_create();
    coli_xdna_prepare_begin(p, 32, 32, COLI_XDNA_DT_BF16);
    coli_xdna_prepared_release(&p);           /* released mid-PREPARING */
    ck(p == NULL, "release during PREPARING is safe");

    p = coli_xdna_prepared_create();
    coli_xdna_prepare_begin(p, 32, 32, COLI_XDNA_DT_BF16);
    coli_xdna_prepare_publish_failure(p);
    coli_xdna_prepared_release(&p);
    ck(p == NULL, "release of an INVALID object is safe");

    p = coli_xdna_prepared_create();
    coli_xdna_prepare_begin(p, 32, 32, COLI_XDNA_DT_BF16);
    coli_xdna_prepare_publish_success(p);
    coli_xdna_prepared_release(&p);
    coli_xdna_prepared_release(&p);           /* repeated */
    coli_xdna_prepared_release(&p);
    ck(p == NULL, "repeated release is safe and idempotent");

    ck(coli_xdna_prepared_live_objects() == 0, "no prepared object leaked");
    ck(coli_xdna_prepared_total_bytes() == 0, "no prepared bytes leaked");
}

/* ---- bounded stress ----------------------------------------------------- */
static void test_stress(void){
    printf("bounded allocation lifecycle stress\n");
    size_t peak = 0;
    for(int i = 0; i < 200; i++){
        ColiXdnaPrepared *p = coli_xdna_prepared_create();
        coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16);
        if(i & 1) coli_xdna_prepare_publish_failure(p);
        else      coli_xdna_prepare_publish_success(p);
        if(coli_xdna_prepared_total_bytes() > peak) peak = coli_xdna_prepared_total_bytes();
        /* reuse the retained capacity through a complete new cycle */
        coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16);
        coli_xdna_prepare_publish_success(p);
        coli_xdna_prepared_invalidate(p);
        coli_xdna_prepared_release(&p);
    }
    printf("    peak accounted bytes %zu, live objects %d, final bytes %zu\n",
           peak, coli_xdna_prepared_live_objects(), coli_xdna_prepared_total_bytes());
    ck(peak > 0, "stress actually allocated");
    ck(coli_xdna_prepared_live_objects() == 0, "no object leaked across 200 cycles");
    ck(coli_xdna_prepared_total_bytes() == 0, "accounting returns to zero");
}

/* ---- the authoritative weight is untouched ------------------------------ */
static void test_fmt4_immutability(void){
    printf("authoritative fmt4 immutability\n");
    /* A synthetic stand-in for the authoritative tensor: the prepared-state API
     * must never see, take or free these. */
    unsigned char q4[512]; float scales[16];
    for(int i = 0; i < 512; i++) q4[i] = (unsigned char)(i * 7 + 1);
    for(int i = 0; i < 16; i++)  scales[i] = 0.25f * (float)(i + 1);
    unsigned char q4_before[512]; float sc_before[16];
    memcpy(q4_before, q4, sizeof q4); memcpy(sc_before, scales, sizeof scales);
    const void *q4_ptr = q4; const void *sc_ptr = scales;

    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16);
    memset(coli_xdna_prepare_dest(p), 0x5A, coli_xdna_prepared_bytes(p));
    coli_xdna_prepare_publish_success(p);
    coli_xdna_prepared_invalidate(p);
    coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16);
    coli_xdna_prepare_publish_failure(p);
    coli_xdna_prepared_free_buffer(p);
    coli_xdna_prepared_release(&p);

    ck(memcmp(q4, q4_before, sizeof q4) == 0, "fmt4 packed bytes unchanged");
    ck(memcmp(scales, sc_before, sizeof scales) == 0, "scale block unchanged");
    ck(q4_ptr == (const void*)q4 && sc_ptr == (const void*)scales,
       "authoritative pointers unchanged");
}

/* ---- independence ------------------------------------------------------- */
static void test_independence(void){
    printf("helper and registry independence\n");
    coli_xdna_test_set_helper_path("C:/nonexistent/coli_xdna.dll");
    ck(coli_xdna_binding() == COLI_XDNA_ABSENT, "helper absent");

    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    ck(coli_xdna_prepare_begin(p, 64, 64, COLI_XDNA_DT_BF16) == 1,
       "allocation works with no helper");
    ck(coli_xdna_prepare_publish_success(p) == 1, "publication works with no helper");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID, "state is VALID");
    coli_xdna_prepared_release(&p);

    /* A statically qualified artifact does not create prepared state, and a
     * prepared image does not qualify an artifact. */
    ck(coli_xdna_prepared_live_objects() == 0,
       "registry lookup created no prepared object");

    coli_xdna_test_set_helper_path(NULL);
    ck(coli_xdna_test_helper_calls() == 0, "no helper entry point was called");
    ck(coli_xdna_test_device_opens() == 0, "no device was opened");
}

/* ---- what PREPARED_VALID must not be mistaken for ----------------------- */
static void test_no_runtime_readiness(void){
    printf("PREPARED_VALID is not runtime readiness\n");
    ck(!strcmp(coli_xdna_prep_text(COLI_XDNA_PREP_VALID), "PREPARED_VALID"),
       "the label says PREPARED, not READY");
    ck(coli_xdna_test_userptr_wraps() == 0, "no userptr wrap exists");
    /* Conversions DO happen from I4 onward -- the counter now reports real work
     * rather than being pinned at zero. What must stay zero is device work. */
    ck(coli_xdna_test_conversions() > 0, "conversions were performed and counted");
    ck(coli_xdna_test_device_opens() == 0, "still no device opened");
}


/* ---- fmt4 -> BF16 conversion ------------------------------------------- */

/* Independent reference. Deliberately written from the fmt=4 contract rather
 * than by calling the production converter, so an indexing or layout mistake in
 * the converter cannot hide by being reproduced here. */
static unsigned short ref_f2b(float f){
    unsigned int u; memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}
static void ref_convert(unsigned short *dst, const unsigned char *q4,
                        const float *scale, int I, int O, int gs){
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    for(int o = 0; o < O; o++){
        const unsigned char *w = q4 + (size_t)o * rb;
        const float *scl = scale + (size_t)o * ng;
        for(int i = 0; i < I; i++){
            unsigned char byte = w[i >> 1];
            int nib = (i & 1) ? (int)(byte >> 4) : (int)(byte & 0x0F);
            dst[(size_t)i * O + o] = ref_f2b((float)(nib - 8) * scl[i / gs]);
        }
    }
}

static void fill_src(unsigned char *q4, float *scale, int I, int O, int gs, unsigned seed){
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned s = seed ? seed : 1u;
    for(size_t k = 0; k < (size_t)O * rb; k++){ s = s*1664525u+1013904223u; q4[k] = (unsigned char)(s >> 24); }
    for(size_t k = 0; k < (size_t)O * ng; k++){ s = s*1664525u+1013904223u;
        scale[k] = 0.25f * (float)(1 + ((s >> 26) & 3)); }
}

static void test_bf16_rounding(void){
    printf("BF16 rounding\n");
    /* The sealed contract is round-to-nearest-even on the 16-bit boundary, not
     * truncation. These are the cases where the two differ. */
    union { unsigned int u; float f; } v;

    v.u = 0x00000000u; ck(ref_f2b(v.f) == 0x0000, "+0 -> 0x0000");
    v.u = 0x80000000u; ck(ref_f2b(v.f) == 0x8000, "-0 -> 0x8000 (sign preserved)");
    ck(ref_f2b(1.0f) == 0x3F80, "1.0 -> 0x3F80");
    ck(ref_f2b(-1.0f) == 0xBF80, "-1.0 -> 0xBF80");
    ck(ref_f2b(2.0f) == 0x4000, "2.0 -> 0x4000");

    /* Exactly halfway: 0x3F808000 rounds to even -> 0x3F80, not up. */
    v.u = 0x3F808000u; ck(ref_f2b(v.f) == 0x3F80, "tie rounds to even (down)");
    /* Halfway with an odd LSB rounds up. */
    v.u = 0x3F818000u; ck(ref_f2b(v.f) == 0x3F82, "tie with odd LSB rounds up");
    /* Just above halfway always rounds up. */
    v.u = 0x3F808001u; ck(ref_f2b(v.f) == 0x3F81, "above tie rounds up");
    /* Truncation would give 0x3F80 for all three; it does not here. */

    /* Every value this converter can actually produce is (nib-8)*scale with
     * nib-8 in [-8,7] and a positive scale, so it is exactly representable and
     * rounding never fires in practice -- but the rule must still be correct. */
    ck(ref_f2b(-8.0f * 0.25f) == ref_f2b(-2.0f), "representative decoded value");
}

static void test_conversion_matches_reference(void){
    printf("conversion vs independent reference\n");
    struct { int I, O, gs; const char *what; } cases[] = {
        {  64,   4, 64, "one full group, 4 rows" },
        {  63,   3, 64, "odd I, partial group" },
        {  65,   2, 64, "I just past a group boundary" },
        { 128,   5, 64, "two full groups" },
        {   1,   1, 64, "single element" },
        {   7,   3, 64, "odd I, several rows" },
        { 130,   3, 64, "partial trailing group" },
    };
    for(size_t c = 0; c < sizeof cases / sizeof cases[0]; c++){
        int I = cases[c].I, O = cases[c].O, gs = cases[c].gs;
        int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
        unsigned char *q4 = (unsigned char*)malloc((size_t)O * rb);
        float *scale = (float*)malloc((size_t)O * ng * sizeof(float));
        fill_src(q4, scale, I, O, gs, 1000u + (unsigned)c);

        unsigned short *want = (unsigned short*)malloc((size_t)I * O * 2);
        ref_convert(want, q4, scale, I, O, gs);

        ColiXdnaPrepared *p = coli_xdna_prepared_create();
        ColiXdnaPrepResult r = coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs);

        int ok = (r == COLI_XDNA_PREP_OK)
              && coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID
              && coli_xdna_prepared_bytes(p) == (size_t)I * O * 2
              && coli_xdna_prepared_k(p) == (unsigned)I
              && coli_xdna_prepared_n(p) == (unsigned)O;
        size_t mism = 0;
        if(ok){
            const unsigned short *got = (const unsigned short*)coli_xdna_prepared_image(p);
            for(size_t e = 0; e < (size_t)I * O; e++) if(got[e] != want[e]) mism++;
        }
        char msg[128];
        snprintf(msg, sizeof msg, "%s (I=%d O=%d)", cases[c].what, I, O);
        ck(ok && mism == 0, msg);

        coli_xdna_prepared_release(&p);
        free(q4); free(scale); free(want);
    }
}

static void test_layout_transpose(void){
    printf("layout transformation\n");
    /* A hand-checkable 2x3: source is [O][I], destination must be [I][O]. */
    int I = 2, O = 3, gs = 64;
    unsigned char q4[3];        /* rb = 1 byte per row */
    float scale[3];
    /* row o: low nibble = element i=0, high nibble = element i=1 */
    q4[0] = (unsigned char)((10u << 4) | 9u);   /* o=0: i0 nib 9, i1 nib 10 */
    q4[1] = (unsigned char)(( 8u << 4) | 7u);   /* o=1: i0 nib 7, i1 nib 8  */
    q4[2] = (unsigned char)(( 0u << 4) | 15u);  /* o=2: i0 nib 15, i1 nib 0 */
    scale[0] = 1.0f; scale[1] = 2.0f; scale[2] = 0.5f;

    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK, "convert");
    const unsigned short *b = (const unsigned short*)coli_xdna_prepared_image(p);

    /* dst[i*O + o] */
    ck(b[0*O + 0] == ref_f2b((9.0f  - 8.0f) * 1.0f), "B[0][0] = (nib 9  - 8) * 1.0");
    ck(b[0*O + 1] == ref_f2b((7.0f  - 8.0f) * 2.0f), "B[0][1] = (nib 7  - 8) * 2.0");
    ck(b[0*O + 2] == ref_f2b((15.0f - 8.0f) * 0.5f), "B[0][2] = (nib 15 - 8) * 0.5");
    ck(b[1*O + 0] == ref_f2b((10.0f - 8.0f) * 1.0f), "B[1][0] = (nib 10 - 8) * 1.0");
    ck(b[1*O + 1] == ref_f2b(( 8.0f - 8.0f) * 2.0f), "B[1][1] = (nib 8  - 8) * 2.0 = +0");
    ck(b[1*O + 2] == ref_f2b(( 0.0f - 8.0f) * 0.5f), "B[1][2] = (nib 0  - 8) * 0.5");
    coli_xdna_prepared_release(&p);
}

static void test_scale_group_boundary(void){
    printf("scale group indexing\n");
    /* gs=64 with I=130: groups are [0,64), [64,128), [128,130). Element 63 must
     * use scale 0, element 64 scale 1, element 128 scale 2. */
    int I = 130, O = 1, gs = 64;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned char *q4 = (unsigned char*)calloc(1, (size_t)O * rb);
    float scale[3] = { 1.0f, 4.0f, 16.0f };
    ck(ng == 3, "three groups for I=130, gs=64");
    for(int k = 0; k < rb; k++) q4[k] = 0x99;    /* every nibble = 9 -> value +1 */

    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK, "convert");
    const unsigned short *b = (const unsigned short*)coli_xdna_prepared_image(p);
    ck(b[63]  == ref_f2b(1.0f),  "element 63 uses group 0 scale");
    ck(b[64]  == ref_f2b(4.0f),  "element 64 uses group 1 scale");
    ck(b[127] == ref_f2b(4.0f),  "element 127 still group 1");
    ck(b[128] == ref_f2b(16.0f), "element 128 uses group 2 scale");
    ck(b[129] == ref_f2b(16.0f), "element 129 in the short trailing group");
    coli_xdna_prepared_release(&p);
    free(q4);
}

static void test_source_rejection(void){
    printf("source validation\n");
    int I = 64, O = 2, gs = 64;
    unsigned char q4[64]; float scale[2];
    fill_src(q4, scale, I, O, gs, 7u);
    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    ck(coli_xdna_prepare_from_fmt4(p, 2, q4, scale, I, O, gs)
       == COLI_XDNA_PREP_ERR_UNSUPPORTED_FORMAT, "fmt=2 rejected");
    ck(coli_xdna_prepare_from_fmt4(p, 4, NULL, scale, I, O, gs)
       == COLI_XDNA_PREP_ERR_INVALID_SOURCE, "missing q4 rejected");
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, NULL, I, O, gs)
       == COLI_XDNA_PREP_ERR_INVALID_SOURCE, "missing scales rejected");
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, 0, O, gs)
       == COLI_XDNA_PREP_ERR_INVALID_SOURCE, "I=0 rejected");
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, 0, gs)
       == COLI_XDNA_PREP_ERR_INVALID_SOURCE, "O=0 rejected");
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, 0)
       == COLI_XDNA_PREP_ERR_INVALID_SOURCE, "gs=0 rejected");
    ck(coli_xdna_prepare_from_fmt4(NULL, 4, q4, scale, I, O, gs)
       == COLI_XDNA_PREP_ERR_STATE, "NULL prepared object rejected");
    /* Dimensions are int, so on a 64-bit host the largest reachable product is
     * about 4.6e18 elements -- below SIZE_MAX/2. A genuine size_t overflow is
     * therefore UNREACHABLE through this API here, and the ERR_SIZE guard is
     * defence for platforms with a narrower size_t. What is reachable is an
     * enormous but representable request, and that must fail as an allocation
     * failure rather than be misreported as an arithmetic one. */
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, 0x40000000, 0x40000000, gs)
       == COLI_XDNA_PREP_ERR_ALLOC, "unsatisfiable request fails as ALLOC, not SIZE");

    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_UNPREPARED,
       "every rejection leaves the object UNPREPARED");
    ck(coli_xdna_prepared_bytes(p) == 0, "and accounts no bytes");
    coli_xdna_prepared_release(&p);
}

/* ---- mid-conversion failure -------------------------------------------- */
#define POISON16 0xDEADu

static void test_injected_failure(void){
    printf("mid-conversion failure\n");
    int I = 128, O = 8, gs = 64;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned char *q4 = (unsigned char*)malloc((size_t)O * rb);
    float *scale = (float*)malloc((size_t)O * ng * sizeof(float));
    fill_src(q4, scale, I, O, gs, 31u);
    size_t total = (size_t)I * O;

    /* The gold image, and proof the poison value cannot occur in it. */
    unsigned short *gold = (unsigned short*)malloc(total * 2);
    ref_convert(gold, q4, scale, I, O, gs);
    size_t collide = 0;
    for(size_t e = 0; e < total; e++) if(gold[e] == POISON16) collide++;
    ck(collide == 0, "poison value cannot occur in a valid image");

    unsigned char q4_before[4096]; float sc_before[64];
    memcpy(q4_before, q4, (size_t)O * rb);
    memcpy(sc_before, scale, (size_t)O * ng * sizeof(float));

    const int pct[3] = { 25, 50, 75 };
    for(int c = 0; c < 3; c++){
        ColiXdnaPrepared *p = coli_xdna_prepared_create();
        /* Allocate first so the destination can be poisoned before conversion. */
        ck(coli_xdna_prepare_begin(p, (unsigned)I, (unsigned)O, COLI_XDNA_DT_BF16) == 1, "begin");
        unsigned short *d = (unsigned short*)coli_xdna_prepare_dest(p);
        for(size_t e = 0; e < total; e++) d[e] = POISON16;
        coli_xdna_prepare_publish_failure(p);      /* park it INVALID with capacity */

        coli_xdna_test_set_convert_fail_pct(pct[c]);
        ColiXdnaPrepResult r = coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs);
        coli_xdna_test_set_convert_fail_pct(0);

        char m[96];
        snprintf(m, sizeof m, "%d%%: converter reports failure", pct[c]);
        ck(r == COLI_XDNA_PREP_ERR_FAILED, m);
        snprintf(m, sizeof m, "%d%%: state is PREPARED_INVALID", pct[c]);
        ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID, m);

        /* The destination is inspected through the test-only accessor: an
         * INVALID image exposes no writable destination by contract. */
        const unsigned short *b = (const unsigned short*)coli_xdna_prepared_image_unchecked(p);
        size_t written = 0, poison = 0;
        for(size_t e = 0; e < total; e++){
            if(b[e] == POISON16) poison++;
            else if(b[e] == gold[e]) written++;
        }
        snprintf(m, sizeof m, "%d%%: %zu converted, %zu poison remain", pct[c], written, poison);
        ck(written > 0 && poison > 0, m);

        snprintf(m, sizeof m, "%d%%: INVALID cannot publish VALID", pct[c]);
        ck(coli_xdna_prepare_publish_success(p) == 0, m);

        snprintf(m, sizeof m, "%d%%: fmt4 source unchanged", pct[c]);
        ck(memcmp(q4, q4_before, (size_t)O * rb) == 0
           && memcmp(scale, sc_before, (size_t)O * ng * sizeof(float)) == 0, m);

        /* Complete re-preparation over the same allocation. */
        ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK,
           "  complete reprepare succeeds");
        ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID, "  state VALID again");
        const unsigned short *g2 = (const unsigned short*)coli_xdna_prepared_image(p);
        size_t bad = 0, left = 0;
        for(size_t e = 0; e < total; e++){ if(g2[e] != gold[e]) bad++; if(g2[e] == POISON16) left++; }
        ck(bad == 0 && left == 0, "  reprepared image bit-exact, no residual poison");

        coli_xdna_prepared_release(&p);
    }
    free(q4); free(scale); free(gold);
}

static void test_success_failure_success(void){
    printf("success -> failure -> success\n");
    int I = 64, O = 4, gs = 64;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned char *q4 = (unsigned char*)malloc((size_t)O * rb);
    float *scale = (float*)malloc((size_t)O * ng * sizeof(float));
    fill_src(q4, scale, I, O, gs, 77u);
    size_t total = (size_t)I * O;
    unsigned short *gold = (unsigned short*)malloc(total * 2);
    ref_convert(gold, q4, scale, I, O, gs);

    ColiXdnaPrepared *p = coli_xdna_prepared_create();

    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK, "first prepare");
    ck(memcmp(coli_xdna_prepared_image(p), gold, total * 2) == 0, "first image bit-exact");

    coli_xdna_test_set_convert_fail_pct(50);
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_ERR_FAILED,
       "injected failure on the second attempt");
    coli_xdna_test_set_convert_fail_pct(0);
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_INVALID, "state INVALID");
    ck(coli_xdna_prepared_image(p) == NULL, "an INVALID image is not readable as prepared");

    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK, "third prepare");
    ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID, "state VALID");
    ck(memcmp(coli_xdna_prepared_image(p), gold, total * 2) == 0,
       "third image bit-exact, identical to the first");

    coli_xdna_prepared_release(&p);
    free(q4); free(scale); free(gold);
}

static void test_no_fp32_image(void){
    printf("no full FP32 intermediate\n");
    /* The only allocation a conversion may make is the BF16 destination itself.
     * Engine-side accounting reports exactly that and nothing more. */
    int I = 256, O = 16, gs = 64;
    int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
    unsigned char *q4 = (unsigned char*)malloc((size_t)O * rb);
    float *scale = (float*)malloc((size_t)O * ng * sizeof(float));
    fill_src(q4, scale, I, O, gs, 5u);

    ColiXdnaPrepared *p = coli_xdna_prepared_create();
    ck(coli_xdna_prepare_from_fmt4(p, 4, q4, scale, I, O, gs) == COLI_XDNA_PREP_OK, "convert");
    ck(coli_xdna_prepared_total_bytes() == (size_t)I * O * 2,
       "accounted bytes are exactly the BF16 destination");
    ck(coli_xdna_prepared_bytes(p) == (size_t)I * O * 2, "and nothing else was retained");
    ck(coli_xdna_pointer_alignment_ok(coli_xdna_prepared_image(p)) == 1,
       "destination still 4096-aligned after conversion");
    coli_xdna_prepared_release(&p);
    ck(coli_xdna_prepared_total_bytes() == 0, "released cleanly");
    free(q4); free(scale);
}

int main(void){
    test_alloc_alignment();
    test_non_page_multiple_size();
    test_misaligned_validator();
    test_size_safety();
    test_transitions();
    test_illegal_transitions();
    test_failure_publication();
    test_three_independent_axes();
    test_release_lifetime();
    test_stress();
    test_fmt4_immutability();
    test_independence();
    test_bf16_rounding();
    test_conversion_matches_reference();
    test_layout_transpose();
    test_scale_group_boundary();
    test_source_rejection();
    test_injected_failure();
    test_success_failure_success();
    test_no_fp32_image();
    test_no_runtime_readiness();

    printf("test_xdna_prepared_state: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
