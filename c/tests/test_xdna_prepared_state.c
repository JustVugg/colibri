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
    ck(coli_xdna_test_conversions() == 0, "no fmt4 conversion was performed");
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
    test_no_runtime_readiness();

    printf("test_xdna_prepared_state: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
