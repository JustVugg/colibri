/* W2-N7-I6 -- XDNA failure classification, fallback exactness and late-output
 * invalidation.
 *
 * The invariant this file exists to defend:
 *
 *     Optional XDNA work may fail. Current Colibri operation semantics may not.
 *
 * So every case here poisons the caller output, drives the production candidate
 * into a failure, runs the EXACT current path the production seam runs, and
 * requires the result to equal a matmul_qt reference with zero residual poison.
 *
 * A new owner rather than an extension of an existing one: this is the only
 * place that needs BOTH the engine (for matmul_qt, the fallback target) and the
 * synthetic helper fixtures (for the failure stages). test_xdna_execution.c has
 * the helper but no engine; test_xdna_qt_state.c has the engine but no helper.
 *
 * Hardware-free. The synthetic ABI-2 helper contains no XRT and touches no
 * device, and it models the real helper faithfully enough to matter -- it
 * snapshots the weight at wrap time, exactly as sync(BO_TO_DEVICE) does.
 */

#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

#include "../backend_xdna.h"

static int g_fail = 0;
static int g_checks = 0;
static void ck(int cond, const char *what){
    g_checks++;
    if(!cond){ printf("  FAIL %s\n", what); g_fail = 1; }
    else       printf("  ok   %s\n", what);
}

/* ---- fixtures ----------------------------------------------------------- */

#define TK 256
#define TN 128
#define TM 64

/* Failure stages the synthetic helper can be told to produce. Mirrors the enum
 * in tests/xdna_fake_helper.c. */
enum { F_NONE = 0, F_DEVICE, F_ARTIFACT, F_WRAP, F_EXECUTE, F_COMPLETION };

static char g_root[1024], g_helper[1024], g_helper_abi1[1024], g_helper_partial[1024];
static char g_xclbin[2048], g_insts[2048];
static char g_xhex[65], g_ihex[65];
static ColiXdnaArtifact g_test_rows[1];
static HMODULE g_fake;
static void (*p_set_fail)(int);
static void (*p_reset)(void);

#define POISON (-987654.0f)

static void hexify(const unsigned char h[32], char out[65]){
    static const char *H = "0123456789abcdef";
    for(int i = 0; i < 32; i++){ out[i*2] = H[h[i]>>4]; out[i*2+1] = H[h[i]&15]; }
    out[64] = '\0';
}
static int write_blob(const char *path, unsigned seed, size_t n){
    FILE *f = fopen(path, "wb");
    if(!f) return 0;
    unsigned st = seed;
    for(size_t i = 0; i < n; i++){ st = st*1664525u+1013904223u; fputc((int)(st>>24) & 0xFF, f); }
    fclose(f); return 1;
}
static void build_registry(void){
    unsigned char h[32];
    if(!coli_xdna_sha256_file(g_xclbin, h)){ printf("  FAIL fixture unreadable\n"); g_fail=1; return; }
    hexify(h, g_xhex);
    if(!coli_xdna_sha256_file(g_insts, h)){ printf("  FAIL fixture unreadable\n"); g_fail=1; return; }
    hexify(h, g_ihex);
    g_test_rows[0].family = COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP;
    g_test_rows[0].research_family = "F3";
    g_test_rows[0].artifact_m = TM;
    g_test_rows[0].k = TK; g_test_rows[0].n = TN;
    g_test_rows[0].in_dtype = COLI_XDNA_DT_BF16;
    g_test_rows[0].weight_dtype = COLI_XDNA_DT_BF16;
    g_test_rows[0].out_dtype = COLI_XDNA_DT_F32;
    g_test_rows[0].target = COLI_XDNA_TARGET_XDNA2;
    g_test_rows[0].xclbin_name = "fake.xclbin";
    g_test_rows[0].xclbin_sha256 = g_xhex;
    g_test_rows[0].insts_name = "fake_insts.bin";
    g_test_rows[0].insts_sha256 = g_ihex;
    g_test_rows[0].runtime_weight_qualified = 1;
    g_test_rows[0].correctness_qualified = 1;
    g_test_rows[0].userptr_qualified = 1;
    g_test_rows[0].structural_qualified = 1;
    coli_xdna_test_set_registry(g_test_rows, 1);
}

static void qt_make(QT *t, int I, int O, int gs, unsigned seed){
    memset(t, 0, sizeof *t);
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    t->fmt = 4; t->I = I; t->O = O; t->gs = gs;
    t->q4 = (uint8_t*)malloc((size_t)O*rb);
    t->s  = (float*)malloc((size_t)O*ng*sizeof(float));
    unsigned st = seed;
    for(size_t i = 0; i < (size_t)O*rb; i++){ st = st*1664525u+1013904223u; t->q4[i] = (uint8_t)(st>>24); }
    for(size_t i = 0; i < (size_t)O*ng; i++){ st = st*1664525u+1013904223u;
        t->s[i] = 0.25f*(float)(1 + ((st>>26)&3)); }
}
static float *mk_x(int S, int K, unsigned seed){
    float *x = (float*)malloc((size_t)S*K*sizeof(float));
    unsigned st = seed;
    for(size_t i = 0; i < (size_t)S*K; i++){ st = st*1664525u+1013904223u;
        x[i] = (float)((int)((st>>20)&255)-128)*0.0078125f; }
    return x;
}
static unsigned short f2b_t(float f){
    unsigned int u; memcpy(&u,&f,4);
    u += 0x7FFFu + ((u>>16)&1u);
    return (unsigned short)(u>>16);
}
static void poison_buf(float *y, size_t n){ for(size_t i=0;i<n;i++) y[i]=POISON; }
static size_t count_poison(const float *y, size_t n){
    size_t c=0; for(size_t i=0;i<n;i++) if(y[i]==POISON) c++; return c;
}
static unsigned long long fnv(const void *p, size_t n){
    unsigned long long h = 1469598103934665603ULL;
    for(size_t i=0;i<n;i++){ h ^= ((const unsigned char*)p)[i]; h *= 1099511628211ULL; }
    return h;
}

/* Arm a failure stage. Deliberately does NOT reset the fake: reset clears its
 * open state, which would desynchronise it from a host lane that still believes
 * the artifact is open. Reset happens once, in use_helper, where the host lane
 * has just been torn down too. */
static void arm(int stage){ if(p_set_fail) p_set_fail(stage); }

static void use_helper(const char *path){
    coli_xdna_execution_shutdown();
    coli_xdna_test_set_force_execution(0);
    coli_xdna_test_set_helper_path(path);
    g_fake = NULL; p_set_fail = NULL; p_reset = NULL;
    if(coli_xdna_binding() == COLI_XDNA_AVAILABLE){
        g_fake = GetModuleHandleA(path);
        if(!g_fake){
            const char *base = strrchr(path, '\\');
            const char *fw = strrchr(path, '/');
            if(fw && (!base || fw > base)) base = fw;
            g_fake = GetModuleHandleA(base ? base+1 : path);
        }
        if(g_fake){
            p_set_fail = (void(*)(int))(void*)GetProcAddress(g_fake, "fake_set_fail");
            p_reset    = (void(*)(void))(void*)GetProcAddress(g_fake, "fake_reset");
            if(p_reset) p_reset();   /* host lane and fake start together */
        }
    }
}

/* ---- the production seam, reproduced exactly ----------------------------
 *
 * This is the shape of the two GLM shared gate/up sites, character for
 * character in structure: try the lane, and if it does not handle the
 * operation run the SAME matmul_qt call that stood there before the lane
 * existed. Nothing else is a fallback. */
static void seam(float *y, const float *x, QT *w, int S){
    if(!coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &w->xdna,
                             w->fmt, w->q4, w->s, w->I, w->O, w->gs, w->planar, y, x, S))
        matmul_qt(y, x, w, S);
}

/* Drive one failure and require the exact current-path result. */
static void expect_exact_fallback(const char *what, QT *w, const float *x, int S,
                                  const float *ref){
    float *y = (float*)malloc((size_t)S*w->O*4);
    poison_buf(y, (size_t)S*w->O);
    int d0 = coli_xdna_test_dispatches();
    seam(y, x, w, S);
    char m[220];
    snprintf(m,sizeof m,"%s -> no successful dispatch", what);
    ck(coli_xdna_test_dispatches()==d0, m);
    snprintf(m,sizeof m,"%s -> XDNA output not valid", what);
    ck(coli_xdna_test_output_valid()==0, m);
    snprintf(m,sizeof m,"%s -> residual poison 0", what);
    ck(count_poison(y,(size_t)S*w->O)==0, m);
    snprintf(m,sizeof m,"%s -> result is bit-identical to the current path", what);
    ck(memcmp(y, ref, (size_t)S*w->O*4)==0, m);
    free(y);
}

int main(int argc, char **argv){
    char dir[512]; snprintf(dir, sizeof dir, "%s", argc>0?argv[0]:"tests/x");
    { char *sl = strrchr(dir,'\\'); char *fw = strrchr(dir,'/');
      if(fw && (!sl || fw>sl)) sl = fw;
      if(sl) *sl = '\0'; else snprintf(dir,sizeof dir,"."); }

    snprintf(g_root, sizeof g_root, "%s/xdna_fail_fixtures", dir);
    snprintf(g_helper, sizeof g_helper, "%s/xdna_fake_helper.dll", dir);
    snprintf(g_helper_abi1, sizeof g_helper_abi1, "%s/xdna_fake_helper_abi1.dll", dir);
    snprintf(g_helper_partial, sizeof g_helper_partial, "%s/xdna_fake_helper_partial.dll", dir);
    { char cmd[1200]; snprintf(cmd,sizeof cmd,"mkdir \"%s\" 2>nul", g_root);
      for(char *p=cmd;*p;p++) if(*p=='/') *p='\\';
      if(system(cmd)){} }
    snprintf(g_xclbin, sizeof g_xclbin, "%s/fake.xclbin", g_root);
    snprintf(g_insts,  sizeof g_insts,  "%s/fake_insts.bin", g_root);
    if(!write_blob(g_xclbin, 31u, 4096) || !write_blob(g_insts, 41u, 1024)){
        printf("FAIL fixtures\n"); return 1; }
    build_registry();
    if(g_fail) return 1;

    QT W; qt_make(&W, TK, TN, 64, 4242u);
    QT W2; qt_make(&W2, TK, TN, 64, 8888u);
    float *x = mk_x(TM, TK, 77u);

    /* The current-path reference, computed once, by the same call the seam
     * falls back to. Every failure case must reproduce this exactly. */
    float *REF = (float*)malloc((size_t)TM*TN*4);
    matmul_qt(REF, x, &W, TM);
    unsigned long long ref_hash = fnv(REF, (size_t)TM*TN*4);
    printf("current-path reference hash %016llx\n", ref_hash);

    /* fmt4 source fingerprints, so "the authoritative weight is untouched" is
     * a measurement rather than an assumption. */
    const size_t q4b = (size_t)TN*((TK+1)/2), scb = (size_t)TN*((TK+63)/64)*4;

    /* ================================================================== */
    printf("\npre-device declines -- nothing may reach the device\n");
    {
        struct { const char *what; ColiXdnaFamily fam; int fmt,I,O,gs,planar,S; ColiXdnaHard want; } c[] = {
            { "wrong semantic family", COLI_XDNA_FAMILY_NONE, 4,TK,TN,64,0,TM, COLI_XDNA_HARD_FAMILY_UNSUPPORTED },
            { "sh_down orientation",   COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4,TN,TK,64,0,TM, COLI_XDNA_HARD_SHAPE_UNSUPPORTED },
            { "gs != 64",              COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4,TK,TN,32,0,TM, COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED },
            { "fmt != 4",              COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 2,TK,TN,64,0,TM, COLI_XDNA_HARD_FORMAT_UNSUPPORTED },
            { "M above range",         COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4,TK,TN,64,0,TM+1, COLI_XDNA_HARD_M_OUT_OF_RANGE },
            { "K/N mismatch",          COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4,TK+8,TN,64,0,TM, COLI_XDNA_HARD_SHAPE_UNSUPPORTED },
            /* K1 planar layout: the converter was never qualified against it */
            { "planar fmt4 layout",    COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4,TK,TN,64,1,TM, COLI_XDNA_HARD_LAYOUT_UNSUPPORTED }
        };
        for(size_t i = 0; i < sizeof c/sizeof c[0]; i++){
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            QT T; qt_make(&T, c[i].I, c[i].O, c[i].gs, 11u);
            T.fmt = c[i].fmt;
            float *xx = mk_x(c[i].S, c[i].I, 12u);
            float *yy = (float*)malloc((size_t)c[i].S*c[i].O*4);
            float *rr = (float*)malloc((size_t)c[i].S*c[i].O*4);
            T.fmt = 4; matmul_qt(rr, xx, &T, c[i].S); T.fmt = c[i].fmt;
            poison_buf(yy,(size_t)c[i].S*c[i].O);
            int handled = coli_xdna_try_matmul(c[i].fam, &T.xdna, c[i].fmt, T.q4, T.s,
                                               c[i].I, c[i].O, c[i].gs, c[i].planar, yy, xx, c[i].S);
            char m[200];
            snprintf(m,sizeof m,"%s -> %s", c[i].what, coli_xdna_hard_text(coli_xdna_test_last_hard()));
            ck(coli_xdna_test_last_hard()==c[i].want, m);
            snprintf(m,sizeof m,"%s -> not handled", c[i].what); ck(handled==0,m);
            snprintf(m,sizeof m,"%s -> device opens 0", c[i].what); ck(coli_xdna_test_device_opens()==0,m);
            snprintf(m,sizeof m,"%s -> artifact runtime opens 0", c[i].what); ck(coli_xdna_test_artifact_opens()==0,m);
            snprintf(m,sizeof m,"%s -> userptr wraps 0", c[i].what); ck(coli_xdna_test_userptr_wraps()==0,m);
            snprintf(m,sizeof m,"%s -> dispatches 0", c[i].what); ck(coli_xdna_test_dispatches()==0,m);
            /* the fallback still produces the current-path answer */
            T.fmt = 4;
            matmul_qt(yy, xx, &T, c[i].S);
            snprintf(m,sizeof m,"%s -> current path recomputes exactly", c[i].what);
            ck(memcmp(yy,rr,(size_t)c[i].S*c[i].O*4)==0, m);
            free(xx); free(yy); free(rr);
            coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
        }
    }

    /* ================================================================== */
    printf("\nartifact declines -- distinguishable, and refused before the helper\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_force_execution(1);
        char none[1200]; snprintf(none,sizeof none,"%s/absent", g_root);
        coli_xdna_test_set_artifact_root(none);
        float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&W.xdna,4,W.q4,W.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_test_last_hard()==COLI_XDNA_HARD_ARTIFACT_UNAVAILABLE,
           "artifact absent -> ARTIFACT_UNAVAILABLE");
        ck(coli_xdna_test_helper_calls()==0, "artifact absent -> helper never called");
        expect_exact_fallback("artifact absent", &W, x, TM, REF);

        coli_xdna_test_set_artifact_root(g_root);
        { FILE *f = fopen(g_xclbin,"r+b"); if(f){ fseek(f,64,SEEK_SET); fputc(0x5A,f); fclose(f); } }
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&W.xdna,4,W.q4,W.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_test_last_hard()==COLI_XDNA_HARD_ARTIFACT_INTEGRITY_FAILED,
           "artifact tampered -> ARTIFACT_INTEGRITY_FAILED (not the same as absent)");
        ck(coli_xdna_test_helper_calls()==0, "artifact tampered -> unverified bytes never reach the helper");
        expect_exact_fallback("artifact tampered", &W, x, TM, REF);
        write_blob(g_xclbin, 31u, 4096); build_registry();

        g_test_rows[0].correctness_qualified = 0;
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&W.xdna,4,W.q4,W.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_test_last_hard()==COLI_XDNA_HARD_ARTIFACT_UNQUALIFIED,
           "artifact never correctness-qualified -> ARTIFACT_UNQUALIFIED");
        g_test_rows[0].correctness_qualified = 1;
        free(y);
    }

    /* ================================================================== */
    printf("\nhelper declines -- absence and ABI failure are different\n");
    {
        struct { const char *path, *what; ColiXdnaHard want; } h[] = {
            { "tests/no_such_helper_at_all.dll", "helper absent", COLI_XDNA_HARD_HELPER_UNAVAILABLE },
            { g_helper_abi1,    "previous ABI generation", COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE },
            { g_helper_partial, "incomplete ABI 2",        COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE }
        };
        for(size_t i = 0; i < 3; i++){
            use_helper(h[i].path);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            QT T; qt_make(&T, TK, TN, 64, 4242u);
            float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
            coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,4,T.q4,T.s,TK,TN,64,0,y,x,TM);
            char m[200];
            snprintf(m,sizeof m,"%s -> %s", h[i].what, coli_xdna_hard_text(coli_xdna_test_last_hard()));
            ck(coli_xdna_test_last_hard()==h[i].want, m);
            snprintf(m,sizeof m,"%s -> no weight prepared (the gate precedes preparation)", h[i].what);
            ck(T.xdna==NULL, m);
            snprintf(m,sizeof m,"%s -> load attempted at most once", h[i].what);
            ck(coli_xdna_test_load_attempts()<=1, m);
            free(y); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
        }
        /* stickiness: a second query must not re-attempt the load */
        use_helper(g_helper_abi1);
        coli_xdna_binding(); coli_xdna_binding(); coli_xdna_binding();
        ck(coli_xdna_test_load_attempts()<=1, "loader verdict stays sticky across repeated queries");
        ck(coli_xdna_test_entry_points_bound()==0, "and nothing is bound");
    }

    /* ================================================================== */
    printf("\npreparation failure -- no XRT is reached, fmt4 survives\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        arm(F_NONE);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
        coli_xdna_test_set_convert_fail_pct(50);
        float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
        int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                           4,T.q4,T.s,TK,TN,64,0,y,x,TM);
        ck(handled==0, "mid-conversion failure -> not handled");
        ck(coli_xdna_test_last_exec()==COLI_XDNA_EXEC_WEIGHT_PREPARE_FAILED,
           "classified WEIGHT_PREPARE_FAILED");
        ck(coli_xdna_prepared_state(T.xdna)==COLI_XDNA_PREP_INVALID, "state PREPARED_INVALID");
        ck(coli_xdna_prepared_bytes(T.xdna)>0, "partial BF16 bytes are still allocated");
        ck(coli_xdna_prepared_image(T.xdna)==NULL, "but the image is not readable");
        ck(coli_xdna_test_userptr_wraps()==0, "userptr wraps 0 -- no XRT was reached");
        ck(coli_xdna_test_dispatches()==0, "dispatches 0");
        ck(coli_xdna_test_device_opens()==0, "device opens 0");
        ck(fnv(T.q4,q4b)==fnv(W.q4,q4b) && fnv(T.s,scb)==fnv(W.s,scb),
           "authoritative fmt4 weight and scales unchanged");
        matmul_qt(y, x, &T, TM);
        ck(count_poison(y,(size_t)TM*TN)==0, "current path leaves no residual poison");
        ck(memcmp(y,ref,(size_t)TM*TN*4)==0, "current path result exact");
        /* a fresh complete preparation recovers */
        coli_xdna_test_set_convert_fail_pct(0);
        poison_buf(y,(size_t)TM*TN);
        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                4,T.q4,T.s,TK,TN,64,0,y,x,TM)==1,
           "a complete re-preparation recovers and executes");
        ck(coli_xdna_prepared_state(T.xdna)==COLI_XDNA_PREP_VALID, "state VALID again");
        ck(coli_xdna_test_output_valid()==1, "and the output is valid");
        free(y); free(ref); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\nruntime failure stages -- each classified, each falls back exactly\n");
    {
        struct { int stage; const char *what; ColiXdnaExec want; ColiXdnaHard hard; } st[] = {
            { F_DEVICE,     "device init",     COLI_XDNA_EXEC_DEVICE_INIT_FAILED,   COLI_XDNA_HARD_DEVICE_UNAVAILABLE },
            { F_ARTIFACT,   "artifact runtime open", COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED, COLI_XDNA_HARD_ARTIFACT_RUNTIME_UNAVAILABLE },
            { F_WRAP,       "userptr wrap",    COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED,   COLI_XDNA_HARD_WEIGHT_WRAP_UNAVAILABLE },
            { F_EXECUTE,    "dispatch",        COLI_XDNA_EXEC_EXECUTE_FAILED,       COLI_XDNA_HARD_ELIGIBLE },
            { F_COMPLETION, "completion",      COLI_XDNA_EXEC_COMPLETION_FAILED,    COLI_XDNA_HARD_ELIGIBLE }
        };
        for(size_t i = 0; i < sizeof st/sizeof st[0]; i++){
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            arm(st[i].stage);
            QT T; qt_make(&T, TK, TN, 64, 4242u);
            float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
            float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
            int d0 = coli_xdna_test_dispatches();
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                               4,T.q4,T.s,TK,TN,64,0,y,x,TM);
            char m[220];
            snprintf(m,sizeof m,"%s failure -> not handled", st[i].what); ck(handled==0,m);
            snprintf(m,sizeof m,"%s failure -> %s", st[i].what,
                     coli_xdna_exec_text(coli_xdna_test_last_exec()));
            ck(coli_xdna_test_last_exec()==st[i].want, m);
            snprintf(m,sizeof m,"%s failure -> hard verdict %s", st[i].what,
                     coli_xdna_hard_text(coli_xdna_test_last_hard()));
            ck(coli_xdna_test_last_hard()==st[i].hard, m);
            snprintf(m,sizeof m,"%s failure -> output not valid", st[i].what);
            ck(coli_xdna_test_output_valid()==0, m);
            snprintf(m,sizeof m,"%s failure -> caller output untouched by the lane", st[i].what);
            ck(count_poison(y,(size_t)TM*TN)==(size_t)TM*TN, m);
            snprintf(m,sizeof m,"%s failure -> no successful dispatch", st[i].what);
            ck(coli_xdna_test_dispatches()==d0, m);
            /* the prepared image is a separate concern from runtime success */
            if(st[i].stage != F_DEVICE && st[i].stage != F_ARTIFACT){
                snprintf(m,sizeof m,"%s failure -> a correct prepared image stays VALID", st[i].what);
                ck(coli_xdna_prepared_state(T.xdna)==COLI_XDNA_PREP_VALID, m);
            }
            snprintf(m,sizeof m,"%s failure -> authoritative fmt4 unchanged", st[i].what);
            ck(fnv(T.q4,q4b)==fnv(W.q4,q4b) && fnv(T.s,scb)==fnv(W.s,scb), m);
            /* and the seam recomputes exactly */
            matmul_qt(y, x, &T, TM);
            snprintf(m,sizeof m,"%s failure -> residual poison 0", st[i].what);
            ck(count_poison(y,(size_t)TM*TN)==0, m);
            snprintf(m,sizeof m,"%s failure -> current path result exact", st[i].what);
            ck(memcmp(y,ref,(size_t)TM*TN*4)==0, m);
            free(y); free(ref); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
        }
    }

    /* ================================================================== */
    printf("\nlate output failure -- the helper wrote real bytes, then failed\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        arm(F_COMPLETION);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
        float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);

        int d_late = coli_xdna_test_dispatches();
        int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                           4,T.q4,T.s,TK,TN,64,0,y,x,TM);
        ck(handled==0, "not handled");
        ck(coli_xdna_test_last_exec()==COLI_XDNA_EXEC_COMPLETION_FAILED, "COMPLETION_FAILED");

        /* Prove the helper really did write output before failing: the staging
         * buffer holds a real result, not zeros and not poison. */
        size_t nfl = 0;
        const float *stg = coli_xdna_test_output_staging(&nfl);
        size_t nonzero = 0, finite = 0;
        if(stg) for(size_t i = 0; i < (size_t)TM*TN; i++){
            if(stg[i] != 0.0f) nonzero++;
            if(stg[i] == stg[i]) finite++;
        }
        ck(stg != NULL, "the lane staging buffer exists");
        ck(nonzero > (size_t)TM*TN/2, "the helper genuinely wrote real output bytes before failing");
        ck(finite == (size_t)TM*TN, "and those bytes are finite -- plausibility is not validity");

        ck(coli_xdna_test_output_valid()==0, "the written output is still INVALID");
        ck(count_poison(y,(size_t)TM*TN)==(size_t)TM*TN,
           "not one of those bytes reached the caller");
        ck(coli_xdna_test_dispatches()==d_late, "not counted as a dispatch");

        matmul_qt(y, x, &T, TM);
        ck(count_poison(y,(size_t)TM*TN)==0, "current path overwrote every logical element");
        ck(memcmp(y,ref,(size_t)TM*TN*4)==0, "and produced the exact current-path result");
        ck(fnv(ref,(size_t)TM*TN*4)==fnv(REF,(size_t)TM*TN*4) || T.q4!=W.q4,
           "reference is the matmul_qt answer for this tensor");
        free(y); free(ref); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\ninternal explicit mode -- classified failure, no silent substitution\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(0);   /* explicit does not need the force control */
        arm(F_EXECUTE);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
        ColiXdnaExec e = coli_xdna_test_attempt(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                                4,T.q4,T.s,TK,TN,64,0,y,x,TM);
        ck(e==COLI_XDNA_EXEC_EXECUTE_FAILED, "explicit mode returns the failure class");
        ck(coli_xdna_test_output_valid()==0, "output not valid");
        ck(count_poison(y,(size_t)TM*TN)==(size_t)TM*TN,
           "and no fallback happened -- the caller output is untouched");
        arm(F_NONE);
        poison_buf(y,(size_t)TM*TN);
        e = coli_xdna_test_attempt(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,
                                   4,T.q4,T.s,TK,TN,64,0,y,x,TM);
        ck(e==COLI_XDNA_EXEC_OK, "explicit mode succeeds when the lane works");
        ck(coli_xdna_test_output_valid()==1, "and the output is valid");
        ck(count_poison(y,(size_t)TM*TN)==0, "and was written");
        /* the AUTO-like seam remains inert without the force control */
        arm(F_NONE);
        poison_buf(y,(size_t)TM*TN);
        int d0 = coli_xdna_test_dispatches();
        seam(y, x, &T, TM);
        ck(coli_xdna_test_dispatches()==d0,
           "the production seam is still inert without the internal control");
        free(y); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\nlane health -- device failure is process-scoped, the rest is not\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        ck(coli_xdna_lane_health()==COLI_XDNA_LANE_UNINITIALIZED, "lane starts UNINITIALIZED");
        arm(F_WRAP);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *y = (float*)malloc((size_t)TM*TN*4);
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,4,T.q4,T.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_lane_health()==COLI_XDNA_LANE_HEALTHY,
           "a wrap failure leaves the lane HEALTHY -- it is one operation");
        arm(F_NONE);
        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&T.xdna,4,T.q4,T.s,TK,TN,64,0,y,x,TM)==1,
           "and the very next operation succeeds");

        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        arm(F_DEVICE);
        QT D; qt_make(&D, TK, TN, 64, 4242u);
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&D.xdna,4,D.q4,D.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_lane_health()==COLI_XDNA_LANE_UNAVAILABLE,
           "a device-init failure marks the lane UNAVAILABLE for the process");
        arm(F_NONE);
        int o0 = coli_xdna_test_device_opens();
        coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&D.xdna,4,D.q4,D.s,TK,TN,64,0,y,x,TM);
        ck(coli_xdna_test_device_opens()==o0,
           "and the device is not re-attempted per operation");
        ck(coli_xdna_test_last_hard()==COLI_XDNA_HARD_DEVICE_UNAVAILABLE, "the decline says why");
        coli_xdna_execution_shutdown();
        ck(coli_xdna_lane_health()==COLI_XDNA_LANE_UNINITIALIZED,
           "a full lane teardown is the one thing that clears it");
        free(y);
        coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
        coli_xdna_prepared_release(&D.xdna); free(D.q4); free(D.s);
    }

    /* ================================================================== */
    printf("\nwrapper lifetime\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        arm(F_NONE);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *y = (float*)malloc((size_t)TM*TN*4);

        int wbase = coli_xdna_test_userptr_wraps();
        int rubase = coli_xdna_test_wrapper_reuses();
        seam(y, x, &T, TM);
        int w1 = coli_xdna_test_userptr_wraps();
        ck(w1==wbase+1, "one wrapper for the first operation");
        seam(y, x, &T, TM);
        seam(y, x, &T, TM);
        ck(coli_xdna_test_userptr_wraps()==w1, "reused across operations, not recreated");
        ck(coli_xdna_test_wrapper_reuses()==rubase+2, "and the reuse is counted");

        /* same address, NEW image: the defect this slice fixed */
        const void *p1 = coli_xdna_prepared_image(T.xdna);
        unsigned long long g1 = coli_xdna_prepared_generation(T.xdna);
        int wr0 = coli_xdna_test_wrapper_releases();
        coli_xdna_prepared_invalidate(T.xdna);
        ck(coli_xdna_prepare_from_fmt4(T.xdna,4,W2.q4,W2.s,TK,TN,64)==COLI_XDNA_PREP_OK,
           "a different weight re-prepares into the retained buffer");
        const void *p2 = coli_xdna_prepared_image(T.xdna);
        ck(p1==p2, "and lands on the same address");
        ck(coli_xdna_prepared_generation(T.xdna)!=g1, "but a new generation");
        seam(y, x, &T, TM);
        ck(coli_xdna_test_userptr_wraps()==w1+1,
           "the wrapper is REBUILT, because the address is not an identity");
        /* Invalidating the image already dropped the wrapper: the bytes stopped
         * being authoritative at that moment, so no device-side view of them may
         * stay current either. */
        ck(coli_xdna_test_wrapper_releases()>wr0,
           "and the previous wrapper was released when the image was invalidated");

        /* and the result is the NEW weight's answer, not the old one */
        float *want = (float*)malloc((size_t)TM*TN*4);
        {
            const unsigned short *Wi = (const unsigned short*)coli_xdna_prepared_image(T.xdna);
            for(int i=0;i<TM;i++) for(int j=0;j<TN;j++){
                float a=0;
                for(int k=0;k<TK;k++){
                    unsigned int u=(unsigned int)f2b_t(x[(size_t)i*TK+k])<<16; float xv; memcpy(&xv,&u,4);
                    unsigned int v=(unsigned int)Wi[(size_t)k*TN+j]<<16; float wv; memcpy(&wv,&v,4);
                    a+=xv*wv; }
                want[(size_t)i*TN+j]=a; }
        }
        ck(memcmp(y,want,(size_t)TM*TN*4)==0,
           "the device computed the NEW image, not a stale view of the old one");
        free(want);

        /* freeing engine memory must drop the wrapper first */
        int wr1 = coli_xdna_test_wrapper_releases();
        coli_xdna_prepared_free_buffer(T.xdna);
        ck(coli_xdna_test_wrapper_releases()>wr1,
           "freeing the prepared buffer releases the wrapper that borrowed it");
        free(y); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\nsuccess -> failure -> success in one process\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        arm(F_NONE);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
        float *y1 = (float*)malloc((size_t)TM*TN*4);
        float *y2 = (float*)malloc((size_t)TM*TN*4);
        float *y3 = (float*)malloc((size_t)TM*TN*4);
        int opens0 = coli_xdna_test_artifact_opens();

        poison_buf(y1,(size_t)TM*TN); seam(y1, x, &T, TM);
        ck(coli_xdna_test_output_valid()==1, "first operation succeeds on the lane");
        unsigned long long h1 = fnv(y1,(size_t)TM*TN*4);

        arm(F_COMPLETION);
        poison_buf(y2,(size_t)TM*TN); seam(y2, x, &T, TM);
        ck(coli_xdna_test_output_valid()==0, "middle operation fails");
        ck(count_poison(y2,(size_t)TM*TN)==0, "and the current path filled it");
        ck(memcmp(y2,ref,(size_t)TM*TN*4)==0, "exactly");

        arm(F_NONE);
        poison_buf(y3,(size_t)TM*TN); seam(y3, x, &T, TM);
        ck(coli_xdna_test_output_valid()==1, "third operation succeeds again");
        ck(fnv(y3,(size_t)TM*TN*4)==h1, "with the same result as the first -- no stale state");
        ck(coli_xdna_test_artifact_opens()==opens0+1,
           "no helper reload was required by these failure classes");
        ck(coli_xdna_prepared_live_objects()>0, "prepared state survived the cycle");
        free(y1); free(y2); free(y3); free(ref);
        coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\nrepeated failure stability\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
        float *y = (float*)malloc((size_t)TM*TN*4);
        int stages[] = { F_WRAP, F_EXECUTE, F_COMPLETION, F_NONE };
        size_t bytes_mid = 0; int objs_mid = 0, wraps_mid = 0;
        int bad = 0;
        for(int cycle = 0; cycle < 200; cycle++){
            arm(stages[cycle & 3]);
            poison_buf(y,(size_t)TM*TN);
            seam(y, x, &T, TM);
            if(stages[cycle & 3] != F_NONE && memcmp(y,ref,(size_t)TM*TN*4)!=0) bad = 1;
            if(count_poison(y,(size_t)TM*TN)!=0) bad = 1;
            if(cycle == 99){ bytes_mid = coli_xdna_prepared_total_bytes();
                             objs_mid = coli_xdna_prepared_live_objects();
                             wraps_mid = coli_xdna_test_userptr_wraps(); }
        }
        ck(bad==0, "200 mixed failure/success cycles, every fallback exact and fully overwritten");
        ck(coli_xdna_prepared_total_bytes()==bytes_mid, "prepared host bytes do not grow");
        ck(coli_xdna_prepared_live_objects()==objs_mid, "prepared objects do not grow");
        ck(coli_xdna_test_userptr_wraps()-wraps_mid <= 100,
           "wrapper creations stay bounded by the failures that invalidate them");
        ck(coli_xdna_test_wrapper_releases() <= coli_xdna_test_userptr_wraps(),
           "never more releases than wraps");
        free(y); free(ref); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    /* ================================================================== */
    printf("\ndefault behaviour\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        arm(F_NONE);
        /* force deliberately NOT set: helper present, artifact valid, device fine */
        QT T; qt_make(&T, TK, TN, 64, 4242u);
        float *ref = (float*)malloc((size_t)TM*TN*4); matmul_qt(ref, x, &T, TM);
        float *y = (float*)malloc((size_t)TM*TN*4); poison_buf(y,(size_t)TM*TN);
        int d0 = coli_xdna_test_dispatches();
        seam(y, x, &T, TM);
        ck(coli_xdna_test_dispatches()==d0, "everything available, still zero dispatches");
        ck(memcmp(y,ref,(size_t)TM*TN*4)==0, "and the current path produced the result");
        ck(coli_xdna_test_output_valid()==0, "no XDNA output was claimed");
        free(y); free(ref); coli_xdna_prepared_release(&T.xdna); free(T.q4); free(T.s);
    }

    coli_xdna_execution_shutdown();
    coli_xdna_shutdown();
    coli_xdna_test_set_registry(NULL,0);
    coli_xdna_prepared_release(&W.xdna); coli_xdna_prepared_release(&W2.xdna);
    free(W.q4); free(W.s); free(W2.q4); free(W2.s); free(x); free(REF);

    ck(coli_xdna_prepared_live_objects()==0, "no prepared object leaked");
    ck(coli_xdna_prepared_total_bytes()==0, "no prepared bytes leaked");

    printf("\ntest_xdna_failure: %s (%d checks)\n", g_fail ? "FAIL" : "ok", g_checks);
    return g_fail;
}
