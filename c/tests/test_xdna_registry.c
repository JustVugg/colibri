/* Engine-owned XDNA artifact registry, integrity and STATIC eligibility.
 *
 * Colibri owns which artifact answers which semantic operation, and whether that
 * artifact is trustworthy. The helper owns none of it. These tests run entirely
 * without XRT, without an NPU and without a real .xclbin: integrity here means
 * byte identity against a hash the research programme sealed, not semantic XDNA
 * validity, so tiny deterministic fixtures are exactly the right instrument.
 *
 * The contract these tests defend is that STATIC qualification is a long way
 * short of "this can run": no device has been opened, no weight prepared, no
 * pointer aligned and no economics consulted.
 */

#include "../backend_xdna.c"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static int g_fail = 0;

static void ck(int cond, const char *what){
    printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
    if(!cond) g_fail = 1;
}

/* ---- SHA256 known vectors (FIPS 180-4 / NIST examples) ------------------ */
static const char *hex32(const unsigned char d[32], char out[65]){
    static const char *H = "0123456789abcdef";
    for(int i = 0; i < 32; i++){ out[i*2] = H[d[i]>>4]; out[i*2+1] = H[d[i]&15]; }
    out[64] = '\0';
    return out;
}

static void test_sha256_vectors(void){
    unsigned char d[32]; char h[65];
    printf("sha256 known vectors\n");

    coli_xdna_sha256("", 0, d);
    ck(!strcmp(hex32(d,h), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
       "empty string");

    coli_xdna_sha256("abc", 3, d);
    ck(!strcmp(hex32(d,h), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
       "abc");

    coli_xdna_sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56, d);
    ck(!strcmp(hex32(d,h), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
       "448-bit multi-block message");

    /* One million 'a': exercises the length encoding past 2^32 bits/8 and the
     * multi-block loop far beyond the padding boundary. */
    {
        char *buf = (char*)malloc(1000000);
        memset(buf, 'a', 1000000);
        coli_xdna_sha256(buf, 1000000, d);
        free(buf);
        ck(!strcmp(hex32(d,h), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
           "one million 'a'");
    }

    /* Exactly at the padding boundary: 55, 56 and 64 bytes are where naive
     * implementations lose a block. */
    {
        char b[64]; memset(b, 'x', sizeof b);
        coli_xdna_sha256(b, 55, d); (void)hex32(d,h);
        char h55[65]; strcpy(h55, h);
        coli_xdna_sha256(b, 56, d); (void)hex32(d,h);
        ck(strcmp(h55, h) != 0, "55 and 56 byte inputs differ");
        coli_xdna_sha256(b, 64, d); (void)hex32(d,h);
        ck(strcmp(h55, h) != 0, "64 byte input distinct");
    }
}

/* ---- fixtures ----------------------------------------------------------- */
static char g_root[512];

static int write_file(const char *rel, const char *bytes, size_t n){
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", g_root, rel);
    FILE *f = fopen(path, "wb");
    if(!f) return 0;
    size_t w = n ? fwrite(bytes, 1, n, f) : 0;
    fclose(f);
    return w == n;
}

static void remove_file(const char *rel){
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", g_root, rel);
    remove(path);
}

/* Deterministic fixture bodies. These are NOT XDNA programs. */
#define FAKE_XCLBIN "colibri-xdna-test-xclbin-body"
#define FAKE_INSTS  "colibri-xdna-test-insts-body"

/* SHA256 of the two bodies, computed by the implementation under test and
 * pinned here so a change in either the fixture or the hash is visible. */
static char g_xclbin_sha[65], g_insts_sha[65];

static ColiXdnaArtifact g_test_rows[4];

static void build_registry(void){
    unsigned char d[32];
    coli_xdna_sha256(FAKE_XCLBIN, strlen(FAKE_XCLBIN), d); hex32(d, g_xclbin_sha);
    coli_xdna_sha256(FAKE_INSTS,  strlen(FAKE_INSTS),  d); hex32(d, g_insts_sha);

    /* row 0: fully qualified */
    ColiXdnaArtifact *r = &g_test_rows[0];
    memset(r, 0, sizeof *r);
    r->family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    r->research_family = "T1";
    r->artifact_m = 64; r->k = 6144; r->n = 2048;
    r->in_dtype = COLI_XDNA_DT_BF16;
    r->weight_dtype = COLI_XDNA_DT_BF16;
    r->out_dtype = COLI_XDNA_DT_F32;
    r->target = COLI_XDNA_TARGET_XDNA2;
    r->xclbin_name = "t1_m64.xclbin"; r->xclbin_sha256 = g_xclbin_sha;
    r->insts_name  = "t1_m64_insts.bin"; r->insts_sha256 = g_insts_sha;
    r->runtime_weight_qualified = 1;
    r->correctness_qualified = 1;
    r->userptr_qualified = 1;
    r->structural_qualified = 1;

    /* row 1: identical shape, correctness NOT qualified */
    g_test_rows[1] = g_test_rows[0];
    g_test_rows[1].artifact_m = 256;
    g_test_rows[1].xclbin_name = "t1_m256.xclbin";
    g_test_rows[1].insts_name  = "t1_m256_insts.bin";
    g_test_rows[1].correctness_qualified = 0;
}

static ColiXdnaRequest req_of(ColiXdnaFamily fam, unsigned m, unsigned k, unsigned n){
    ColiXdnaRequest q;
    memset(&q, 0, sizeof q);
    q.family = fam; q.m = m; q.k = k; q.n = n;
    q.in_dtype = COLI_XDNA_DT_BF16;
    q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;
    return q;
}

/* ---- registry identity -------------------------------------------------- */
static void test_lookup(void){
    printf("registry lookup\n");
    coli_xdna_test_set_registry(g_test_rows, 2);

    ColiXdnaRequest q = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);
    const ColiXdnaArtifact *a = coli_xdna_registry_lookup(&q);
    ck(a != NULL, "known family + bucket + shape resolves");
    ck(a && a->artifact_m == 64, "resolves to the requested bucket");

    /* THE invariant: shape alone is not eligibility. */
    ColiXdnaRequest other = req_of(COLI_XDNA_FAMILY_NONE, 64, 6144, 2048);
    ck(coli_xdna_registry_lookup(&other) == NULL,
       "same M/K/N, unknown family declines");

    ColiXdnaRequest wrong_k = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 4096, 2048);
    ck(coli_xdna_registry_lookup(&wrong_k) == NULL, "wrong K declines");

    ColiXdnaRequest wrong_n = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 1024);
    ck(coli_xdna_registry_lookup(&wrong_n) == NULL, "wrong N declines");

    /* Buckets are exact: research qualified M=64 and M=256, nothing between. */
    ColiXdnaRequest between = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 128, 6144, 2048);
    ck(coli_xdna_registry_lookup(&between) == NULL,
       "unqualified M bucket declines (no extrapolation)");

    ColiXdnaRequest bad_dt = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);
    bad_dt.weight_dtype = COLI_XDNA_DT_F32;
    ck(coli_xdna_registry_lookup(&bad_dt) == NULL, "wrong prepared dtype declines");

    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- self validation ---------------------------------------------------- */
static void test_registry_validation(void){
    printf("registry self-validation\n");

    coli_xdna_test_set_registry(NULL, 0);
    ck(coli_xdna_registry_validate() == 1, "production registry self-validates");

    ColiXdnaArtifact bad[2];
    bad[0] = g_test_rows[0];
    bad[1] = g_test_rows[0];                       /* exact duplicate key */
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "duplicate key rejected");

    bad[1] = g_test_rows[0]; bad[1].k = 0;
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "zero dimension rejected");

    bad[1] = g_test_rows[0]; bad[1].family = COLI_XDNA_FAMILY_NONE;
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "empty family rejected");

    bad[1] = g_test_rows[0]; bad[1].xclbin_sha256 = "not-a-sha";
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "malformed sha rejected");

    bad[1] = g_test_rows[0]; bad[1].xclbin_name = "";
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "empty artifact name rejected");

    bad[1] = g_test_rows[0]; bad[1].artifact_m = 256; bad[1].in_dtype = (ColiXdnaDtype)99;
    coli_xdna_test_set_registry(bad, 2);
    ck(coli_xdna_registry_validate() == 0, "invalid dtype rejected");

    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- presence + integrity ----------------------------------------------- */
static void test_artifact_status(void){
    printf("artifact presence and integrity\n");
    coli_xdna_test_set_registry(g_test_rows, 2);
    ColiXdnaRequest q = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);

    write_file("t1_m64.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));
    write_file("t1_m64_insts.bin", FAKE_INSTS, strlen(FAKE_INSTS));
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_QUALIFIED,
       "both present, both hashes correct -> STATIC_QUALIFIED");

    remove_file("t1_m64.xclbin");
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE,
       "xclbin missing -> ARTIFACT_UNAVAILABLE");
    write_file("t1_m64.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));

    remove_file("t1_m64_insts.bin");
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE,
       "insts missing -> ARTIFACT_UNAVAILABLE");
    write_file("t1_m64_insts.bin", FAKE_INSTS, strlen(FAKE_INSTS));

    write_file("t1_m64.xclbin", "corrupted", 9);
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
       "corrupt xclbin -> INTEGRITY_FAILED");
    write_file("t1_m64.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));

    write_file("t1_m64_insts.bin", "corrupted", 9);
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
       "corrupt insts -> INTEGRITY_FAILED");

    write_file("t1_m64.xclbin", "corrupted", 9);
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
       "both corrupt -> INTEGRITY_FAILED");

    write_file("t1_m64.xclbin", "", 0);
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
       "empty file where content expected -> INTEGRITY_FAILED");

    /* Byte-identical content but a registry hash that does not match it. */
    write_file("t1_m64.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));
    write_file("t1_m64_insts.bin", FAKE_INSTS, strlen(FAKE_INSTS));
    {
        ColiXdnaArtifact wrong[1];
        wrong[0] = g_test_rows[0];
        wrong[0].xclbin_sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000";
        coli_xdna_test_set_registry(wrong, 1);
        ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_ARTIFACT_INTEGRITY_FAILED,
           "wrong expected hash in registry -> INTEGRITY_FAILED");
        coli_xdna_test_set_registry(g_test_rows, 2);
    }

    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- qualification flags ------------------------------------------------ */
static void test_qualification_gate(void){
    printf("qualification gates\n");
    coli_xdna_test_set_registry(g_test_rows, 2);

    /* Row 1 has valid files and correct hashes but correctness_qualified = 0.
     * This is COMPILE_PASS/DISPATCH_PASS != ELIGIBLE, executable rather than
     * asserted in prose. */
    write_file("t1_m256.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));
    write_file("t1_m256_insts.bin", FAKE_INSTS, strlen(FAKE_INSTS));
    ColiXdnaRequest q256 = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 256, 6144, 2048);
    ck(coli_xdna_artifact_status(&q256, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED,
       "valid hashes + correctness=false -> ARTIFACT_UNQUALIFIED");

    ColiXdnaArtifact one[1];
    ColiXdnaRequest q64 = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);

    one[0] = g_test_rows[0]; one[0].runtime_weight_qualified = 0;
    coli_xdna_test_set_registry(one, 1);
    ck(coli_xdna_artifact_status(&q64, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED,
       "runtime-weight qualification required");

    one[0] = g_test_rows[0]; one[0].userptr_qualified = 0;
    coli_xdna_test_set_registry(one, 1);
    ck(coli_xdna_artifact_status(&q64, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED,
       "userptr qualification required");

    one[0] = g_test_rows[0]; one[0].structural_qualified = 0;
    coli_xdna_test_set_registry(one, 1);
    ck(coli_xdna_artifact_status(&q64, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNQUALIFIED,
       "structural qualification required");

    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- path contract ------------------------------------------------------ */
static void test_path_contract(void){
    printf("artifact path contract\n");
    ColiXdnaArtifact esc[1];
    esc[0] = g_test_rows[0];
    esc[0].xclbin_name = "../escape.xclbin";
    coli_xdna_test_set_registry(esc, 1);
    ck(coli_xdna_registry_validate() == 0, "traversal in artifact name rejected by validation");

    ColiXdnaRequest q = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_REGISTRY_INVALID,
       "traversal never reaches the filesystem");

    esc[0].xclbin_name = "sub\\..\\escape.xclbin";
    coli_xdna_test_set_registry(esc, 1);
    ck(coli_xdna_registry_validate() == 0, "backslash traversal rejected");

    esc[0].xclbin_name = "C:/absolute.xclbin";
    coli_xdna_test_set_registry(esc, 1);
    ck(coli_xdna_registry_validate() == 0, "absolute artifact name rejected");

    coli_xdna_test_set_registry(g_test_rows, 2);
    ck(coli_xdna_artifact_status(&q, NULL) == COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE,
       "no artifact root supplied -> ARTIFACT_UNAVAILABLE, never CWD");

    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- helper state and artifact state are independent -------------------- */
static void test_state_separation(void){
    printf("helper / artifact state separation\n");
    coli_xdna_test_set_registry(g_test_rows, 2);
    write_file("t1_m64.xclbin", FAKE_XCLBIN, strlen(FAKE_XCLBIN));
    write_file("t1_m64_insts.bin", FAKE_INSTS, strlen(FAKE_INSTS));
    ColiXdnaRequest q = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);

    /* No helper anywhere: the artifact is still statically qualified, because
     * artifact trust is a property of the artifact, not of the runtime. */
    coli_xdna_test_set_helper_path("C:/nonexistent/coli_xdna.dll");
    ck(coli_xdna_binding() == COLI_XDNA_ABSENT, "helper absent");
    ck(coli_xdna_artifact_status(&q, g_root) == COLI_XDNA_STATIC_QUALIFIED,
       "artifact qualification independent of helper absence");

    /* ...but combined static eligibility still declines, because the lane as a
     * whole is unusable without a helper. */
    ck(coli_xdna_static_eligibility(&q, g_root) == COLI_XDNA_STATIC_HELPER_UNAVAILABLE,
       "combined eligibility declines when the helper is absent");

    coli_xdna_test_set_helper_path(NULL);
    coli_xdna_test_set_registry(NULL, 0);
}

/* ---- the states this slice must NOT be able to produce ------------------ */
static void test_no_runtime_readiness(void){
    printf("no runtime readiness is claimed\n");
    /* The strongest verdict reachable in I2 is STATIC_QUALIFIED. There is no
     * enumerator for device, prepared, aligned or dispatch readiness, and the
     * label says "STATIC" so it cannot be misread at a call site. */
    ck(!strcmp(coli_xdna_static_text(COLI_XDNA_STATIC_QUALIFIED),
               "STATIC_ARTIFACT_QUALIFIED"),
       "best verdict is explicitly STATIC");
    ck(coli_xdna_test_device_opens() == 0, "no device was opened");
    ck(coli_xdna_test_helper_calls() == 0, "no helper entry point was called");
}

/* ---- production rows ---------------------------------------------------- */
static void test_production_rows(void){
    printf("production registry rows\n");
    coli_xdna_test_set_registry(NULL, 0);
    ck(coli_xdna_registry_validate() == 1, "production rows self-validate");

    ColiXdnaRequest q64  = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 2048);
    ColiXdnaRequest q256 = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 256, 6144, 2048);
    const ColiXdnaArtifact *a64  = coli_xdna_registry_lookup(&q64);
    const ColiXdnaArtifact *a256 = coli_xdna_registry_lookup(&q256);
    ck(a64 && a256, "both qualified F3 buckets present");
    ck(a64 && !strcmp(a64->research_family, "F3"), "research family recorded");
    ck(a64 && a64->correctness_qualified && a64->userptr_qualified
           && a64->runtime_weight_qualified && a64->structural_qualified,
       "all four qualifications asserted");

    /* Scope discipline: V1 is one family. Other research families exist as
     * artifacts but must not be reachable here. */
    ColiXdnaRequest f1 = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 2048, 1024);
    ck(coli_xdna_registry_lookup(&f1) == NULL, "F1 shape not in the V1 registry");
    ColiXdnaRequest f6 = req_of(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 64, 6144, 12288);
    ck(coli_xdna_registry_lookup(&f6) == NULL, "F6 shape not in the V1 registry");

    /* Artifacts are not shipped: absence is the normal, non-fatal state. */
    ck(coli_xdna_artifact_status(&q64, g_root) == COLI_XDNA_STATIC_ARTIFACT_UNAVAILABLE,
       "unshipped production artifact -> ARTIFACT_UNAVAILABLE");
}

/* ---- the compiled-in production registry -------------------------------
 *
 * REGRESSION for a defect W2-N7-I5 found in this file: g_nrows was initialised
 * to 0 with no lazy initialisation, so the production table was installed but
 * EMPTY and every production lookup answered UNKNOWN_FAMILY. It stayed
 * invisible for two slices because every caller was a test that installed its
 * own registry first, and it surfaced only when I5 became the first production
 * consumer.
 *
 * So this runs FIRST, before anything calls coli_xdna_test_set_registry, and it
 * asks the production table what it actually contains. */
static void test_production_registry_default(void){
    printf("compiled-in production registry, with no test rows installed\n");

    ck(coli_xdna_registry_validate(), "the production registry validates as compiled");

    struct { unsigned m, k, n; const char *label; } want[] = {
        { 64,  6144, 2048, "F3 M64  K6144 N2048" },
        { 256, 6144, 2048, "F3 M256 K6144 N2048" }
    };
    for(size_t i = 0; i < sizeof want/sizeof want[0]; i++){
        ColiXdnaRequest q;
        q.family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
        q.m = want[i].m; q.k = want[i].k; q.n = want[i].n;
        q.in_dtype = COLI_XDNA_DT_BF16;
        q.weight_dtype = COLI_XDNA_DT_BF16;
        q.out_dtype = COLI_XDNA_DT_F32;
        const ColiXdnaArtifact *a = coli_xdna_registry_lookup(&q);
        char m[96];
        snprintf(m, sizeof m, "%s is visible without any test setup", want[i].label);
        ck(a != NULL, m);
        if(a){
            snprintf(m, sizeof m, "%s carries its research family", want[i].label);
            ck(a->research_family && !strcmp(a->research_family, "F3"), m);
            snprintf(m, sizeof m, "%s carries a 64-hex xclbin hash", want[i].label);
            ck(a->xclbin_sha256 && strlen(a->xclbin_sha256) == 64, m);
            snprintf(m, sizeof m, "%s is fully research-qualified", want[i].label);
            ck(a->runtime_weight_qualified && a->correctness_qualified
               && a->userptr_qualified && a->structural_qualified, m);
        }
    }

    /* Exactly those rows, and nothing that was never qualified. */
    ColiXdnaRequest q;
    q.family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    q.m = 128; q.k = 6144; q.n = 2048;
    q.in_dtype = COLI_XDNA_DT_BF16; q.weight_dtype = COLI_XDNA_DT_BF16;
    q.out_dtype = COLI_XDNA_DT_F32;
    ck(coli_xdna_registry_lookup(&q) == NULL,
       "an M bucket that was never qualified is absent");

    /* And a test registry stays a TEST registry: installing one must not be how
     * the production table becomes visible, and restoring must bring it back. */
    ColiXdnaArtifact none[1];
    memset(none, 0, sizeof none);
    none[0].family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    none[0].research_family = "TEST";
    none[0].artifact_m = 8; none[0].k = 8; none[0].n = 8;
    none[0].in_dtype = COLI_XDNA_DT_BF16;
    none[0].weight_dtype = COLI_XDNA_DT_BF16;
    none[0].out_dtype = COLI_XDNA_DT_F32;
    none[0].target = COLI_XDNA_TARGET_XDNA2;
    none[0].xclbin_name = "t.xclbin";
    none[0].xclbin_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    none[0].insts_name = "t.bin";
    none[0].insts_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    none[0].runtime_weight_qualified = none[0].correctness_qualified = 1;
    none[0].userptr_qualified = none[0].structural_qualified = 1;
    coli_xdna_test_set_registry(none, 1);
    q.m = 64;
    ck(coli_xdna_registry_lookup(&q) == NULL,
       "a test registry replaces the production one rather than adding to it");
    coli_xdna_test_set_registry(NULL, 0);
    ck(coli_xdna_registry_lookup(&q) != NULL,
       "and restoring brings the production rows back");
}

int main(void){
    /* Every fixture lives in a private directory beside the test binary; the
     * registry never reads the current directory of its own accord. */
    snprintf(g_root, sizeof g_root, "xdna_registry_fixtures");
#ifdef _WIN32
    _mkdir(g_root);
#else
    mkdir(g_root, 0777);
#endif

    /* FIRST, before any test registry is installed. */
    test_production_registry_default();

    build_registry();
    test_sha256_vectors();
    test_lookup();
    test_registry_validation();
    test_artifact_status();
    test_qualification_gate();
    test_path_contract();
    test_state_separation();
    test_production_rows();
    test_no_runtime_readiness();

    /* Leave nothing behind: the fixtures are written beside the test binary,
     * and a stray directory in the source tree would show up as untracked
     * noise for every developer who runs the suite. */
    remove_file("t1_m64.xclbin");
    remove_file("t1_m64_insts.bin");
    remove_file("t1_m256.xclbin");
    remove_file("t1_m256_insts.bin");
#ifdef _WIN32
    _rmdir(g_root);
#else
    rmdir(g_root);
#endif

    printf("test_xdna_registry: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
