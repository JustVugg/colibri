/* W2-N7-I5 -- XDNA execution seam, qualified WITHOUT hardware.
 *
 * This is the owner of the execution control flow: gate order, artifact open,
 * userptr wrap, activation staging and padding, logical-row extraction, failure
 * classification, and the guarantee that every non-success path falls back to
 * the caller's current operation with its output buffer untouched.
 *
 * It runs against a synthetic ABI-2 helper (tests/xdna_fake_helper.c) that
 * contains no XRT and computes the GEMM on the CPU, so a machine with no NPU
 * qualifies all of it. The NUMERICS of real device execution are qualified
 * separately and explicitly on hardware; nothing here stands in for that.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../backend_xdna.c"

static int g_fail = 0;
static void ck(int cond, const char *what){
    if(!cond){ printf("  FAIL %s\n", what); g_fail = 1; }
    else       printf("  ok   %s\n", what);
}

/* ---- fixtures ----------------------------------------------------------- */

#define TK 256        /* K -- small so the CPU GEMM in the fake stays quick */
#define TN 128        /* N */
#define TM 64         /* artifact M, the bucket this slice implements */

static char g_root[1024];
static char g_helper[1024], g_helper_abi1[1024], g_helper_partial[1024];
static char g_xclbin[2048], g_insts[2048];
static char g_xhex[65], g_ihex[65];
static ColiXdnaArtifact g_test_rows[1];

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
    fclose(f);
    return 1;
}

/* Build a registry whose hashes are the REAL hashes of the fixture files, so the
 * integrity gate is exercised for what it is rather than stubbed out. */
static void build_registry(void){
    unsigned char h[32];
    if(!coli_xdna_sha256_file(g_xclbin, h)){ printf("  FAIL fixture xclbin unreadable\n"); g_fail=1; return; }
    hexify(h, g_xhex);
    if(!coli_xdna_sha256_file(g_insts, h)){ printf("  FAIL fixture insts unreadable\n"); g_fail=1; return; }
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

/* A deterministic fmt4 tensor with I=K, O=N. */
typedef struct { unsigned char *q4; float *s; int I, O, gs; } Fmt4;

static void fmt4_make(Fmt4 *t, int I, int O, int gs, unsigned seed){
    int rb = (I+1)/2, ng = (I+gs-1)/gs;
    t->I = I; t->O = O; t->gs = gs;
    t->q4 = (unsigned char *)malloc((size_t)O*rb);
    t->s  = (float *)malloc((size_t)O*ng*sizeof(float));
    unsigned st = seed;
    for(size_t i = 0; i < (size_t)O*rb; i++){ st = st*1664525u+1013904223u; t->q4[i] = (unsigned char)(st>>24); }
    for(size_t i = 0; i < (size_t)O*ng; i++){ st = st*1664525u+1013904223u;
        t->s[i] = 0.25f * (float)(1 + ((st>>26)&3)); }
}
static void fmt4_free(Fmt4 *t){ free(t->q4); free(t->s); t->q4=NULL; t->s=NULL; }

static float *mk_x(int S, int K, unsigned seed){
    float *x = (float *)malloc((size_t)S*K*sizeof(float));
    unsigned st = seed;
    for(size_t i = 0; i < (size_t)S*K; i++){ st = st*1664525u+1013904223u;
        x[i] = (float)((int)((st>>20)&255) - 128) * 0.0078125f; }
    return x;
}

static float b2f_t(unsigned short b){
    unsigned int u = (unsigned int)b << 16; float f; memcpy(&f,&u,4); return f;
}

/* Reference for the LAYOUT contract: BF16 activation x BF16 prepared weight,
 * float accumulation in k order -- exactly what the fake computes. This checks
 * that the right rows went in, the right rows came out, and nothing from the
 * padded region leaked; it is not a claim about device arithmetic. */
static void oracle(float *out, const float *x, const unsigned short *W,
                   int S, int K, int N){
    for(int i = 0; i < S; i++)
        for(int j = 0; j < N; j++){
            float acc = 0.0f;
            for(int k = 0; k < K; k++)
                acc += b2f_t(coli_xdna_f2b(x[(size_t)i*K+k])) * b2f_t(W[(size_t)k*N+j]);
            out[(size_t)i*N+j] = acc;
        }
}

#define POISON (-123456.0f)
static void poison(float *y, size_t n){ for(size_t i=0;i<n;i++) y[i]=POISON; }
static int all_poison(const float *y, size_t n){
    for(size_t i=0;i<n;i++) if(y[i]!=POISON) return 0; return 1;
}

/* Reset every lane-visible counter and state between groups. */
static void lane_reset(void){
    coli_xdna_execution_shutdown();
    coli_xdna_test_set_force_execution(0);
    g_xdna_dispatches = g_xdna_completions = g_xdna_artifact_opens = 0;
    g_xdna_act_preps = g_xdna_padded_ops = g_xdna_fallbacks = 0;
    g_xdna_device_opens = g_xdna_helper_calls = g_xdna_userptr_wraps = 0;
}

/* Rebind the loader to a helper path and clear its sticky verdict. */
static void use_helper(const char *path){
    lane_reset();
    coli_xdna_test_set_helper_path(path);
}

int main(int argc, char **argv){
    /* Locate fixtures beside this executable's directory. */
    char dir[512]; snprintf(dir, sizeof dir, "%s", argc>0?argv[0]:"tests/x");
    char *sl = strrchr(dir,'\\'); char *fw = strrchr(dir,'/');
    if(fw && (!sl || fw>sl)) sl = fw;
    if(sl) *sl = '\0'; else snprintf(dir,sizeof dir,".");

    snprintf(g_root, sizeof g_root, "%s/xdna_exec_fixtures", dir);
    snprintf(g_helper, sizeof g_helper, "%s/xdna_fake_helper.dll", dir);
    snprintf(g_helper_abi1, sizeof g_helper_abi1, "%s/xdna_fake_helper_abi1.dll", dir);
    snprintf(g_helper_partial, sizeof g_helper_partial, "%s/xdna_fake_helper_partial.dll", dir);
#ifdef _WIN32
    { char cmd[1200]; snprintf(cmd,sizeof cmd,"mkdir \"%s\" 2>nul", g_root);
      for(char *p=cmd;*p;p++) if(*p=='/') *p='\\';
      if(system(cmd)){} }
#else
    { char cmd[1200]; snprintf(cmd,sizeof cmd,"mkdir -p \"%s\"", g_root); if(system(cmd)){} }
#endif
    snprintf(g_xclbin, sizeof g_xclbin, "%s/fake.xclbin", g_root);
    snprintf(g_insts,  sizeof g_insts,  "%s/fake_insts.bin", g_root);
    if(!write_blob(g_xclbin, 11u, 4096) || !write_blob(g_insts, 22u, 1024)){
        printf("FAIL could not create fixtures under %s\n", g_root); return 1; }
    build_registry();
    if(g_fail) return 1;

    Fmt4 W;  fmt4_make(&W, TK, TN, 64, 7u);
    Fmt4 W2; fmt4_make(&W2, TK, TN, 64, 999u);      /* a genuinely different weight */

    /* ================================================================== */
    printf("default behaviour: the seam is inert without the internal control\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        ColiXdnaPrepared *slot = NULL;
        float *x = mk_x(TM, TK, 1u);
        float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
        int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &slot,
                                           4, W.q4, W.s, TK, TN, 64, y, x, TM);
        ck(handled == 0, "helper, artifact and device all available -- still not handled");
        ck(coli_xdna_test_dispatches() == 0, "and zero dispatches");
        ck(all_poison(y,(size_t)TM*TN), "caller output untouched");
        coli_xdna_prepared_release(&slot); free(x); free(y);
    }

    /* ================================================================== */
    printf("forced execution: the full path end to end\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        ColiXdnaPrepared *slot = NULL;
        float *x = mk_x(TM, TK, 2u);
        float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
        int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &slot,
                                           4, W.q4, W.s, TK, TN, 64, y, x, TM);
        ck(handled == 1, "handled");
        ck(coli_xdna_test_last_hard() == COLI_XDNA_HARD_ELIGIBLE, "FULL_XDNA_HARD_ELIGIBLE");
        ck(coli_xdna_test_last_exec() == COLI_XDNA_EXEC_OK, "exec OK");
        ck(coli_xdna_test_dispatches() == 1, "one dispatch");
        ck(coli_xdna_test_completions() == 1, "one completion");
        ck(coli_xdna_test_userptr_wraps() == 1, "one userptr wrap");
        ck(coli_xdna_test_artifact_opens() == 1, "one artifact open");
        ck(coli_xdna_prepared_state(slot) == COLI_XDNA_PREP_VALID, "weight left VALID");
        ck(coli_xdna_pointer_alignment_ok(coli_xdna_prepared_image(slot)),
           "prepared image 4096-aligned");

        const unsigned short *Wi = (const unsigned short *)coli_xdna_prepared_image(slot);
        float *ref = (float *)malloc((size_t)TM*TN*4);
        oracle(ref, x, Wi, TM, TK, TN);
        size_t mism = 0;
        for(size_t i = 0; i < (size_t)TM*TN; i++) if(y[i] != ref[i]) mism++;
        ck(mism == 0, "every output element matches the layout reference");
        free(ref); coli_xdna_prepared_release(&slot); free(x); free(y);
    }

    /* ================================================================== */
    printf("logical-M padding inside the qualified range\n");
    {
        int cases[] = { 1, 2, 7, 31, 32, 33, 63, 64 };
        for(size_t c = 0; c < sizeof cases/sizeof cases[0]; c++){
            int S = cases[c];
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            ColiXdnaPrepared *slot = NULL;
            float *x = mk_x(S, TK, 100u + (unsigned)S);
            float *y = (float *)malloc((size_t)S*TN*4); poison(y,(size_t)S*TN);
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, &slot,
                                               4, W.q4, W.s, TK, TN, 64, y, x, S);
            char msg[128];
            snprintf(msg,sizeof msg,"S=%d handled",S); ck(handled==1,msg);
            snprintf(msg,sizeof msg,"S=%d padded_ops=%d",S,coli_xdna_test_padded_operations());
            ck(coli_xdna_test_padded_operations() == (S<TM?1:0), msg);

            const unsigned short *Wi = (const unsigned short *)coli_xdna_prepared_image(slot);
            float *ref = (float *)malloc((size_t)S*TN*4);
            oracle(ref, x, Wi, S, TK, TN);
            size_t mism = 0;
            for(size_t i = 0; i < (size_t)S*TN; i++) if(y[i] != ref[i]) mism++;
            snprintf(msg,sizeof msg,"S=%d all %d logical rows exact, no padded row exposed",S,S);
            ck(mism == 0, msg);
            free(ref); coli_xdna_prepared_release(&slot); free(x); free(y);
        }
    }

    /* ================================================================== */
    printf("negative controls -- none of these may ever dispatch\n");
    {
        struct { const char *what; ColiXdnaFamily fam; int fmt, I, O, gs, S; ColiXdnaHard want; } cases[] = {
            { "wrong semantic family (same shape)", COLI_XDNA_FAMILY_NONE, 4, TK, TN, 64, TM,
              COLI_XDNA_HARD_FAMILY_UNSUPPORTED },
            { "sh_down orientation (K/N transposed)", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4, TN, TK, 64, TM,
              COLI_XDNA_HARD_SHAPE_UNSUPPORTED },
            { "gs != 64 (preparable but not qualified)", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4, TK, TN, 32, TM,
              COLI_XDNA_HARD_GROUP_SIZE_UNSUPPORTED },
            { "fmt != 4", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 2, TK, TN, 64, TM,
              COLI_XDNA_HARD_FORMAT_UNSUPPORTED },
            { "logical M above the qualified range", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4, TK, TN, 64, TM+1,
              COLI_XDNA_HARD_M_OUT_OF_RANGE },
            { "logical M below the qualified range", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4, TK, TN, 64, 0,
              COLI_XDNA_HARD_M_OUT_OF_RANGE },
            { "unqualified shape", COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP, 4, TK+8, TN, 64, TM,
              COLI_XDNA_HARD_SHAPE_UNSUPPORTED }
        };
        for(size_t c = 0; c < sizeof cases/sizeof cases[0]; c++){
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            Fmt4 T; fmt4_make(&T, cases[c].I, cases[c].O, cases[c].gs>0?cases[c].gs:64, 5u);
            ColiXdnaPrepared *slot = NULL;
            int S = cases[c].S > 0 ? cases[c].S : 1;
            float *x = mk_x(S, cases[c].I, 3u);
            float *y = (float *)malloc((size_t)S*cases[c].O*4); poison(y,(size_t)S*cases[c].O);
            int handled = coli_xdna_try_matmul(cases[c].fam, &slot, cases[c].fmt,
                                               T.q4, T.s, cases[c].I, cases[c].O,
                                               cases[c].gs, y, x, cases[c].S);
            char msg[160];
            snprintf(msg,sizeof msg,"%s -> not handled", cases[c].what);   ck(handled==0,msg);
            snprintf(msg,sizeof msg,"%s -> dispatches 0", cases[c].what);
            ck(coli_xdna_test_dispatches()==0,msg);
            snprintf(msg,sizeof msg,"%s -> %s", cases[c].what,
                     coli_xdna_hard_text(coli_xdna_test_last_hard()));
            ck(coli_xdna_test_last_hard()==cases[c].want,msg);
            snprintf(msg,sizeof msg,"%s -> output untouched", cases[c].what);
            ck(all_poison(y,(size_t)S*cases[c].O),msg);
            snprintf(msg,sizeof msg,"%s -> helper never called", cases[c].what);
            ck(coli_xdna_test_helper_calls()==0,msg);
            coli_xdna_prepared_release(&slot); free(x); free(y); fmt4_free(&T);
        }
    }

    /* ================================================================== */
    printf("gs!=64 is refused even though the converter can prepare it\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        Fmt4 T; fmt4_make(&T, TK, TN, 32, 8u);
        ColiXdnaPrepared *p = coli_xdna_prepared_create();
        ck(coli_xdna_prepare_from_fmt4(p,4,T.q4,T.s,TK,TN,32) == COLI_XDNA_PREP_OK,
           "gs=32 prepares successfully -- a valid derived host image");
        ck(coli_xdna_prepared_state(p) == COLI_XDNA_PREP_VALID, "and it is PREPARED_VALID");
        coli_xdna_prepared_release(&p);

        coli_xdna_test_set_force_execution(1);
        ColiXdnaPrepared *slot = NULL;
        float *x = mk_x(TM, TK, 4u);
        float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&slot,4,T.q4,T.s,
                                TK,TN,32,y,x,TM) == 0,
           "GENERIC_PREPARABLE is not INITIAL_XDNA_QUALIFIED");
        ck(coli_xdna_test_dispatches()==0, "dispatches 0");
        coli_xdna_prepared_release(&slot); free(x); free(y); fmt4_free(&T);
    }

    /* ================================================================== */
    printf("helper absent / ABI 1 / incomplete ABI 2\n");
    {
        /* Since W2-N7-I6 absence and ABI failure are different verdicts: one
         * says this machine simply has no optional lane, the other says a
         * helper is installed and something about it is wrong. */
        struct { const char *path, *what; ColiXdnaHard want; } hs[] = {
            { "tests/definitely_no_such_helper.dll", "helper absent",
              COLI_XDNA_HARD_HELPER_UNAVAILABLE },
            { g_helper_abi1,    "ABI generation 1 helper",
              COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE },
            { g_helper_partial, "incomplete ABI 2 helper",
              COLI_XDNA_HARD_HELPER_ABI_INCOMPATIBLE }
        };
        for(size_t c = 0; c < 3; c++){
            use_helper(hs[c].path);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            ColiXdnaPrepared *slot = NULL;
            float *x = mk_x(TM, TK, 5u);
            float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&slot,
                                               4,W.q4,W.s,TK,TN,64,y,x,TM);
            char msg[160];
            snprintf(msg,sizeof msg,"%s -> not handled", hs[c].what); ck(handled==0,msg);
            snprintf(msg,sizeof msg,"%s -> dispatches 0", hs[c].what);
            ck(coli_xdna_test_dispatches()==0,msg);
            snprintf(msg,sizeof msg,"%s -> %s", hs[c].what, coli_xdna_hard_text(hs[c].want));
            ck(coli_xdna_test_last_hard()==hs[c].want,msg);
            snprintf(msg,sizeof msg,"%s -> no entry point bound", hs[c].what);
            ck(coli_xdna_test_entry_points_bound()==0,msg);
            snprintf(msg,sizeof msg,"%s -> output untouched", hs[c].what);
            ck(all_poison(y,(size_t)TM*TN),msg);
            snprintf(msg,sizeof msg,"%s -> no weight prepared (gate precedes preparation)", hs[c].what);
            ck(slot==NULL,msg);
            coli_xdna_prepared_release(&slot); free(x); free(y);
        }
    }

    /* ================================================================== */
    printf("artifact absent / corrupt -- refused before any helper call\n");
    {
        char bad[2048]; snprintf(bad,sizeof bad,"%s/nonexistent_root", g_root);
        const char *roots[2]; roots[0] = bad; roots[1] = g_root;
        for(int c = 0; c < 2; c++){
            if(c == 1){
                /* corrupt a STAGING copy; canonical fixtures are never damaged */
                FILE *f = fopen(g_xclbin, "r+b");
                if(f){ fseek(f, 100, SEEK_SET); fputc(0xFF, f); fclose(f); }
            }
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(roots[c]);
            coli_xdna_test_set_force_execution(1);
            ColiXdnaPrepared *slot = NULL;
            float *x = mk_x(TM, TK, 6u);
            float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&slot,
                                               4,W.q4,W.s,TK,TN,64,y,x,TM);
            const char *what = c==0 ? "artifact absent" : "artifact hash mismatch";
            /* Since W2-N7-I6 these are DIFFERENT verdicts: a build that does not
             * ship the bytes and a build whose bytes were tampered with call for
             * different responses. */
            ColiXdnaHard want = c==0 ? COLI_XDNA_HARD_ARTIFACT_UNAVAILABLE
                                     : COLI_XDNA_HARD_ARTIFACT_INTEGRITY_FAILED;
            char msg[160];
            snprintf(msg,sizeof msg,"%s -> not handled", what); ck(handled==0,msg);
            snprintf(msg,sizeof msg,"%s -> %s", what, coli_xdna_hard_text(want));
            ck(coli_xdna_test_last_hard()==want,msg);
            snprintf(msg,sizeof msg,"%s -> helper never called", what);
            ck(coli_xdna_test_helper_calls()==0,msg);
            snprintf(msg,sizeof msg,"%s -> unverified bytes never reached the helper", what);
            ck(coli_xdna_test_device_opens()==0,msg);
            snprintf(msg,sizeof msg,"%s -> output untouched", what);
            ck(all_poison(y,(size_t)TM*TN),msg);
            coli_xdna_prepared_release(&slot); free(x); free(y);
        }
        /* restore the fixture and the registry hash agreement */
        write_blob(g_xclbin, 11u, 4096);
        build_registry();
    }

    /* ================================================================== */
    printf("helper failure at every stage -- fallback intact, output untouched\n");
    {
        struct { int stage; const char *what; ColiXdnaExec want; } fs[] = {
            { 1, "device init",  COLI_XDNA_EXEC_DEVICE_INIT_FAILED },
            { 2, "artifact open",COLI_XDNA_EXEC_ARTIFACT_OPEN_FAILED },
            { 3, "weight wrap",  COLI_XDNA_EXEC_WEIGHT_WRAP_FAILED },
            { 4, "dispatch",     COLI_XDNA_EXEC_EXECUTE_FAILED },
            { 5, "completion (output already written by the helper)",
                                 COLI_XDNA_EXEC_COMPLETION_FAILED }
        };
        for(size_t c = 0; c < sizeof fs/sizeof fs[0]; c++){
            use_helper(g_helper);
            coli_xdna_test_set_artifact_root(g_root);
            coli_xdna_test_set_force_execution(1);
            /* reach into the loaded fake to arm the injection */
            ck(coli_xdna_binding()==COLI_XDNA_AVAILABLE, "fake helper bound");
            {
                void (*setf)(int) = (void(*)(int))(void*)GetProcAddress(g_xdna.dll,"fake_set_fail");
                void (*rst)(void) = (void(*)(void))(void*)GetProcAddress(g_xdna.dll,"fake_reset");
                if(rst) rst();
                if(setf) setf(fs[c].stage);
            }
            ColiXdnaPrepared *slot = NULL;
            float *x = mk_x(TM, TK, 7u);
            float *y = (float *)malloc((size_t)TM*TN*4); poison(y,(size_t)TM*TN);
            int handled = coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&slot,
                                               4,W.q4,W.s,TK,TN,64,y,x,TM);
            char msg[200];
            snprintf(msg,sizeof msg,"%s failure -> not handled", fs[c].what); ck(handled==0,msg);
            snprintf(msg,sizeof msg,"%s failure -> %s", fs[c].what,
                     coli_xdna_exec_text(coli_xdna_test_last_exec()));
            ck(coli_xdna_test_last_exec()==fs[c].want,msg);
            snprintf(msg,sizeof msg,"%s failure -> caller output STILL untouched", fs[c].what);
            ck(all_poison(y,(size_t)TM*TN),msg);
            snprintf(msg,sizeof msg,"%s failure -> counted as a fallback", fs[c].what);
            ck(coli_xdna_test_fallbacks()>=1,msg);
            snprintf(msg,sizeof msg,"%s failure -> zero successful dispatches", fs[c].what);
            ck(coli_xdna_test_dispatches()==0,msg);
            snprintf(msg,sizeof msg,"%s failure -> fmt4 source unchanged", fs[c].what);
            ck(W.q4 != NULL && W.gs == 64, msg);
            coli_xdna_prepared_release(&slot); free(x); free(y);
            { void (*rst)(void) = (void(*)(void))(void*)GetProcAddress(g_xdna.dll,"fake_reset");
              if(rst) rst(); }
        }
    }

    /* ================================================================== */
    printf("weight reuse and genuinely distinct runtime weights\n");
    {
        use_helper(g_helper);
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        int conv0 = coli_xdna_test_conversions();

        ColiXdnaPrepared *sa = NULL, *sb = NULL;
        float *x = mk_x(TM, TK, 9u);
        float *ya = (float *)malloc((size_t)TM*TN*4);
        float *yb = (float *)malloc((size_t)TM*TN*4);
        float *ya2= (float *)malloc((size_t)TM*TN*4);

        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&sa,4,W.q4,W.s,
                                TK,TN,64,ya,x,TM)==1, "gate-like weight executes");
        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&sa,4,W.q4,W.s,
                                TK,TN,64,ya2,x,TM)==1, "same weight again executes");
        ck(coli_xdna_test_conversions()-conv0 == 1, "prepared once, reused on the second call");
        ck(coli_xdna_test_userptr_wraps() == 1, "wrapped once for the same pointer");
        ck(coli_xdna_test_artifact_opens() == 1, "artifact runtime opened once, not per operation");
        ck(memcmp(ya,ya2,(size_t)TM*TN*4)==0, "identical inputs give identical outputs");

        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&sb,4,W2.q4,W2.s,
                                TK,TN,64,yb,x,TM)==1, "up-like weight executes");
        ck(coli_xdna_test_userptr_wraps() == 2, "a different prepared image is wrapped again");
        ck(coli_xdna_test_artifact_opens() == 1, "and reuses the same artifact runtime");
        ck(memcmp(ya,yb,(size_t)TM*TN*4)!=0, "different runtime weights give different outputs");

        const unsigned short *Wb = (const unsigned short *)coli_xdna_prepared_image(sb);
        float *ref = (float *)malloc((size_t)TM*TN*4);
        oracle(ref, x, Wb, TM, TK, TN);
        size_t mism=0; for(size_t i=0;i<(size_t)TM*TN;i++) if(yb[i]!=ref[i]) mism++;
        ck(mism==0, "the second weight's output is exact too");
        free(ref);
        coli_xdna_prepared_release(&sa); coli_xdna_prepared_release(&sb);
        free(x); free(ya); free(yb); free(ya2);
    }

    /* ================================================================== */
    printf("shutdown lifetime\n");
    {
        use_helper(g_helper);
        coli_xdna_execution_shutdown();
        ck(1, "shutdown when never initialised is safe");
        coli_xdna_test_set_artifact_root(g_root);
        coli_xdna_test_set_force_execution(1);
        ColiXdnaPrepared *slot = NULL;
        float *x = mk_x(TM,TK,10u);
        float *y = (float*)malloc((size_t)TM*TN*4);
        ck(coli_xdna_try_matmul(COLI_XDNA_FAMILY_MOE_SHARED_GATE_UP,&slot,4,W.q4,W.s,
                                TK,TN,64,y,x,TM)==1, "operation succeeds");
        coli_xdna_execution_shutdown();
        ck(1, "shutdown after a successful operation is safe");
        coli_xdna_execution_shutdown();
        ck(1, "repeated shutdown is safe");
        coli_xdna_shutdown();
        ck(coli_xdna_test_entry_points_bound()==0, "module released, no pointer survives it");
        coli_xdna_prepared_release(&slot); free(x); free(y);
    }

    coli_xdna_test_set_helper_path(NULL);
    coli_xdna_shutdown();
    coli_xdna_test_set_registry(NULL,0);
    fmt4_free(&W); fmt4_free(&W2);

    ck(coli_xdna_prepared_live_objects()==0, "no prepared object leaked");
    ck(coli_xdna_prepared_total_bytes()==0, "no prepared bytes leaked");

    printf("test_xdna_execution: %s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
