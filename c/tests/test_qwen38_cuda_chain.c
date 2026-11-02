/* COLI_CUDA_CHAIN=1: Qwen3.8's layers as one chain on the card give the oracle's tokens.
 *
 * The tiny FP8 fixture (make qwen38-tiny-fp8-generate: 4 layers, two of them
 * DeltaNet, the PLE layer, QSA with its indexer, hidden 32) runs the real engine on
 * the fake CUDA tier with the whole dense trunk placed and computed by the fake
 * (fake_dense_compute), the chain's ops answered by the fake's host-side table
 * (tests/qwen36_fake_cuda.h, the kernels' arithmetic): once with the chain on, once
 * with a frame that fails mid-run (the state rebuilt on the CPU, the run finishing
 * there), once without the chain. Every run must stay within the oracle's limits
 * (ref.json), the chain runs must have set up and run every layer on the device.
 * Then the promotions' brake (q38cc_braked) on frames written by hand: the fixture has
 * no GGUF, so no host pool and no promotion runs above. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#undef setenv
#undef unsetenv
#define unsetenv(name) _putenv_s(name, "")
#endif

#include "qwen36_fake_cuda.h"

#include "../qwen36_tier.c"

static int t_fails;
static void tk(int ok, const char *what) { if (ok) { printf("  ok   %s\n", what); return; } printf("  FAIL %s\n", what); t_fails++; }
/* rows of ggml type t with random codes, a positive f16 scale at each block's start */
static unsigned t_seed = 7;
static unsigned t_rnd(void) { t_seed = t_seed * 1103515245u + 12345u; return t_seed >> 8; }
static uint16_t t_f16(float v) { uint32_t u; memcpy(&u, &v, 4); return (uint16_t)(((((u >> 23) & 255) - 112) << 10) | ((u >> 13) & 1023)); }
static void t_fill(int t, uint8_t *w, int rows, int cols) {
    const int bb = GQ_TYPES[t].bytes; const size_t nb = (size_t)rows * (cols / GQ_TYPES[t].blck);
    for (size_t k = 0; k < nb * bb; k++) w[k] = (uint8_t)t_rnd();
    for (size_t k = 0; k < nb; k++) { uint16_t h = t_f16(0.004f + (float)(t_rnd() % 1024) / 1024.f * 0.016f); memcpy(w + k * bb, &h, 2); }
}
/* a frame's books as q38cc_dm_books keeps them: its routes, miss of them read from the pool */
static int brake_frame(Q38CChain *b, unsigned routes, unsigned miss) { b->routes += routes; b->pool_reads += miss; return q38cc_braked(b); }

int main(void) {
    setenv("SNAP", "./qwen38_tiny_fp8", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
    setenv("NOSTREAM", "1", 1);
    setenv("USAGE_SAVE", "0", 1);
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPUS", "0", 1);
    setenv("QT_NO_WARMSTART", "1", 1);
    setenv("QT_UPLOAD_SYNC", "1", 1);
    setenv("HEAT_FILE", "", 1);
    setenv("Q38_TRUNK_GPU", "1", 1);
    setenv("Q38_TRUNK_MIN_KB", "0", 1);          /* every dense matrix of the fixture is offered */
    setenv("COLI_PLACE", "auto", 1);
    setenv("CUDA_EXPERT_GB", "0.5", 1);
    setenv("Q38_MTP", "0", 1);                     /* no draft: the run stays plain */
    fake_dense_compute = 1;
    char *argv[] = { (char *)"qwen38", (char *)"1", (char *)"8", (char *)"./qwen38_tiny_fp8/ref.json" };

    printf(" a prompt chunk's rows\n");
    unsetenv("COLI_CUDA_CHAIN_ROWS");
    tk(q38cc_rows_env() == 2048, "2048 rows a chunk unset: a 1628-token prompt's pooled experts over the bus once");
    setenv("COLI_CUDA_CHAIN_ROWS", "512", 1);
    tk(q38cc_rows_env() == 512, "COLI_CUDA_CHAIN_ROWS=512 the rows before");
    setenv("COLI_CUDA_CHAIN_ROWS", "0", 1);
    int lo = q38cc_rows_env();
    setenv("COLI_CUDA_CHAIN_ROWS", "99999", 1);
    tk(lo == 1 && q38cc_rows_env() == 8192, "within 1..8192");
    unsetenv("COLI_CUDA_CHAIN_ROWS");

    printf(" the chain on the fake card\n");
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0;
    int rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "engine with every layer on the chain stays within the oracle's limits");
    tk(fake_chain_frames > 8, "the forwards ran through the chain's frames");
    qt_shutdown();

    printf(" a frame that fails mid-run\n");
    fake_chain_frames = 0; fake_chain_fail_at = 30;   /* past the first forward's frames: a decode token's frame fails */
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the state rebuilt on the CPU, the run within the oracle's limits");
    tk(fake_chain_frames >= 30, "the fault fired");
    fake_chain_fail_at = 0;
    qt_shutdown();

    printf(" the same without the chain\n");
    setenv("COLI_CUDA_CHAIN", "0", 1);
    fake_chain_frames = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the per-matrix run stays within the oracle's limits");
    tk(fake_chain_frames == 0, "no frame reached the card");
    qt_shutdown();

    printf(" MTP draft rejection with the chain on the fake card\n");
    setenv("SNAP", "./qwen38_tiny_fp8_mtp", 1);
    setenv("Q38_MTP", "1", 1);
    setenv("Q38_MTP_FORCE", "reject", 1);
    argv[3] = (char *)"./qwen38_tiny_fp8_mtp/ref.json";
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0; fake_dn_snapshots = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "rejected MTP drafts roll back and remain within the oracle's limits");
    tk(fake_chain_frames > 8, "rejected MTP verification ran through chain frames");
    tk(fake_dn_snapshots > 0, "MTP verification wrote DeltaNet chain snapshots");
    unsetenv("Q38_MTP_FORCE");
    setenv("Q38_MTP", "0", 1);
    qt_shutdown();

    printf(" the promotions' brake\n");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    tk(q38cc_promote_hits() == Q38CC_PROMO_HITS, "unset: the default threshold");
    setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "150", 1);
    int h_hi = q38cc_promote_hits();
    setenv("COLI_CUDA_CHAIN_PROMOTE_HITS", "-3", 1);
    tk(h_hi == 100 && q38cc_promote_hits() == 0, "the threshold clamped to 0-100");
    unsetenv("COLI_CUDA_CHAIN_PROMOTE_HITS");
    static Q38CChain b0;
    Q38CChain b = b0; b.promo_hits = 90;
    tk(brake_frame(&b, 0, 0) == 0, "no routes yet: no brake");
    tk(brake_frame(&b, 1000, 100) == 1, "90 % of the routes on the card: the frame promotes nothing");
    tk(brake_frame(&b, 1000, 300) == 0, "the window at 80 %: promotions again");
    tk(brake_frame(&b, 100000, 2000) == 1, "a prompt chunk weighs its rows: 97.6 % over the window");
    tk(b.pr_braked == 2, "the braked frames counted");
    b = b0; b.promo_hits = 90;
    tk(brake_frame(&b, 1000, 101) == 0, "just under the threshold: no brake");
    b = b0; b.promo_hits = 90;
    int held = 0;
    brake_frame(&b, 100000, 100000);   /* a frame that found nothing on the card */
    for (int f = 1; f < Q38CC_PROMO_WIN; f++) held += brake_frame(&b, 1000, 0);
    tk(held == 0, "a frame of misses holds the window while it is in it");
    tk(brake_frame(&b, 1000, 0) == 1, "past the window it is forgotten");
    b = b0; b.promo_hits = 0;
    tk(brake_frame(&b, 1000, 0) == 0 && b.pr_braked == 0, "COLI_CUDA_CHAIN_PROMOTE_HITS=0: never");

    printf(" a prompt chunk's pooled slots on the CPU, by expert\n");
    unsetenv("COLI_CUDA_CHAIN_POOL_CPU_ROWS");
    tk(q38cc_pc_rows() == Q38CC_PC_ROWS, "unset: a prompt of up to Q38CC_PC_ROWS rows computes its pooled experts on the CPU");
    {   /* five experts as the host pool holds them (gate, up, down at 0, PAD(gate), twice that),
         * IQ2_S's eight-row kernel for gate and up, Q2_0's row kernel for down; 23 slots, expert 0
         * with 9 of them (past GQ_FUSED_ROWS: one unpack for all its rows) */
        enum { H = 512, I = 256, NE = 5, NS = 23 };
        const int tg = GQ_IQ2_S, td = GQ_Q2_0;
        const int64_t rg = gq_row_bytes(tg, H), rd = gq_row_bytes(td, I);
        const size_t pg = Q38CC_PAD(rg * I), st = 2 * pg + Q38CC_PAD(rd * H);
        uint8_t *pool = (uint8_t *)calloc(NE, st);
        for (int e = 0; e < NE && pool; e++) {
            t_fill(tg, pool + e * st, I, H); t_fill(tg, pool + e * st + pg, I, H); t_fill(td, pool + e * st + 2 * pg, H, I);
        }
        int ex[NS], cnt[NE + 1] = {0}, o[NE + 1], pos[NS], ng = 0;
        for (int q = 0; q < NS; q++) ex[q] = q < 9 ? 0 : q == 9 ? 1 : 2 + q % 3;
        float *x = falloc(NS * H), *xg = falloc(NS * H), *gu = falloc(NS * 2 * I), *hh = falloc(NS * I);
        float *dn1 = falloc(NS * H), *dn2 = falloc(NS * H);
        for (int k = 0; k < NS * H; k++) x[k] = (float)(t_rnd() % 2001) / 1000.f - 1.f;
        const uint8_t *gp[NS], *gg[NE]; const float *xp[NS];
        for (int q = 0; q < NS; q++) { gp[q] = pool + ex[q] * st; xp[q] = x + (size_t)q * H; cnt[ex[q] + 1]++; }
        int ok1 = pool && q38cc_pool_pairs(gp, xp, NS, H, I, tg, td, gu, hh, dn1);
        for (int e = 0; e < NE; e++) { if (cnt[e + 1]) { o[ng] = cnt[e]; gg[ng++] = pool + e * st; } cnt[e + 1] += cnt[e]; }
        o[ng] = NS;
        for (int q = 0; q < NS; q++) { pos[q] = cnt[ex[q]]++; memcpy(xg + (size_t)pos[q] * H, xp[q], H * sizeof(float)); }
        gq_act ax[NE], ah[NE]; memset(ax, 0, sizeof ax); memset(ah, 0, sizeof ah);
        int ok2 = pool && q38cc_pool_groups(gg, o, ng, xg, H, I, tg, td, gu, hh, dn2, ax, ah);
        int same = 1; double worst = 0;
        for (int q = 0; q < NS; q++) {
            double n2 = 0, d2 = 0;
            for (int d = 0; d < H; d++) {
                const float a = dn1[(size_t)q * H + d], b = dn2[(size_t)pos[q] * H + d];
                n2 += (double)a * a; d2 += (double)(a - b) * (a - b);
                if (ex[q] && a != b) same = 0;
            }
            if (n2 > 0 && sqrt(d2 / n2) > worst) worst = sqrt(d2 / n2);
        }
        tk(ok1 && ok2 && ng == NE, "the grouped kernel ran over every expert");
        tk(same, "an expert of up to four rows: each slot the decode step's bits");
        printf("       the nine-row expert against the decode step's kernel: %.2e of the row's norm\n", worst);
        tk(worst < 1e-3, "an expert of more rows: the unpack's arithmetic, within 1e-3 of the row's norm");
        for (int e = 0; e < NE; e++) { gq_act_free(&ax[e]); gq_act_free(&ah[e]); }
        free(pool); free(x); free(xg); free(gu); free(hh); free(dn1); free(dn2);
    }

    if (t_fails) { printf("test_qwen38_cuda_chain: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_cuda_chain: Qwen3.8's chain on the card gives the oracle's tokens\n");
    return 0;
}
