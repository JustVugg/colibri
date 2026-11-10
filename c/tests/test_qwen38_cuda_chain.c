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
 * Then a refusal of the card's memory on a chain written by hand (q38cc_forward gives
 * the K/V mirrors back), and the promotions' brake (q38cc_braked) on frames written by
 * hand: the fixture has no GGUF, so no host pool and no promotion runs above. */
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
struct SrvCtx { uint8_t *pool; size_t st; int ne; };
static const uint8_t *t_srv_where(void *ctx, int k) { const struct SrvCtx *c = ctx; return c->pool + (size_t)(k % c->ne) * c->st; }
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
    fake_chain_frames = 0; fake_chain_fail_at = 0; fake_dn_snapshots = 0; g_q38cc_mtp_rows = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "rejected MTP drafts roll back and remain within the oracle's limits");
    tk(fake_chain_frames > 8, "rejected MTP verification ran through chain frames");
    tk(fake_dn_snapshots > 0, "MTP verification wrote DeltaNet chain snapshots");
    tk(g_q38cc_mtp_rows > 0, "the MTP head's layer ran on the card");
    qt_shutdown();
    printf(" the MTP head on the CPU beside the chain (Q38_MTP_CHAIN=0)\n");
    setenv("Q38_MTP_CHAIN", "0", 1);
    fake_chain_frames = 0; g_q38cc_mtp_rows = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0 && g_q38cc_mtp_rows == 0, "the head on the CPU, the drafts within the oracle's limits");
    unsetenv("Q38_MTP_CHAIN");
    qt_shutdown();
    printf(" a greedy run's verify rows picked on the card\n");
    const size_t size0 = fake_chain_table.size;
    fake_chain_table.size = sizeof fake_chain_table;   /* pick_rows offered */
    setenv("Q38_MTP_FORCE", "accept", 1);
    fake_chain_frames = 0; fake_pick_rows = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0, "the drafts the card picked stand, within the oracle's limits");
    tk(fake_pick_rows > 1, "a verify's rows picked on the card, no logits down");
    qt_shutdown();
    setenv("COLI_CUDA_CHAIN_GREEDY_CARD", "0", 1);
    fake_pick_rows = 0;
    rc = qwen38_main_unused(4, argv);
    tk(rc == 0 && fake_pick_rows == 0, "COLI_CUDA_CHAIN_GREEDY_CARD=0: the logits down, the same tokens");
    unsetenv("COLI_CUDA_CHAIN_GREEDY_CARD");
    fake_chain_table.size = size0;
    unsetenv("Q38_MTP_FORCE");
    setenv("Q38_MTP", "0", 1);
    qt_shutdown();

    printf(" a refusal of the card's memory gives the K/V mirrors back\n");
    {   /* a chain written by hand: three layers, the middle one DeltaNet, and the MTP head's slot, the
         * mirrors in float16; the card's room (fake_chain_dev_cap) for one attention layer's mirrors and
         * half of the next, then for every mirror but not a row's scratch */
        static Model rm; static Q38CChain rc;
        static uint8_t att[3] = {1, 0, 1};
        static CcBuf *kc[4], *vc[4], *ik[4], *pk[4]; static int kvv[4], pkv[4];
        rm.c.layers = 3; rm.c.is_attn = att; rm.c.kv_heads = 2; rm.c.head_dim = 8; rm.c.idx_dim = 4; rm.c.idx_ratio = 4;
        rm.c.hc_width = 64; rm.kv_cap = 32; rm.cchain = &rc;
        rc.kc = kc; rc.vc = vc; rc.ik = ik; rc.pk = pk; rc.kv_valid = kvv; rc.pk_valid = pkv;
        rc.ok = 1; rc.mtp_on = 1; rc.kvh = 1;
        const size_t lay = 2 * (2 * 8 * 32 * 2) + 4 * 32 * 4 + 4 * (32 / 4 + 1) * 4;   /* kc and vc in halves, ik, pk */
        const int chain0 = g_cuda_chain; g_cuda_chain = 1;
        const size_t b0 = fake_chain_dev_bytes;
        int none = 1;
        fake_chain_dev_cap = b0 + lay + lay / 2;
        int r1 = q38cc_forward(&rm, NULL, 1, 0, 1, NULL, 0, NULL, NULL);
        for (int i = 0; i < 4; i++) none = none && !kc[i] && !vc[i] && !ik[i] && !pk[i];
        tk(r1 == 0 && rc.failed && rc.cap == 0, "the second layer's mirrors refused: the per-matrix path");
        tk(none && fake_chain_dev_bytes == b0, "the first layer's mirrors and the second's half given back");
        rc.failed = 0;
        fake_chain_dev_cap = b0 + 3 * lay + 16;   /* every mirror, not a row's 256-byte streams */
        int r2 = q38cc_forward(&rm, NULL, 1, 0, 1, NULL, 0, NULL, NULL);
        none = 1;
        for (int i = 0; i < 4; i++) none = none && !kc[i] && !vc[i] && !ik[i] && !pk[i];
        tk(r2 == 0 && rc.failed && none && fake_chain_dev_bytes == b0, "the scratch refused: every mirror given back");
        fake_chain_dev_cap = 0;
        tk(q38cc_forward(&rm, NULL, 1, 0, 1, NULL, 0, NULL, NULL) == 0 && fake_chain_dev_bytes == b0,
           "the chain stays off: a later forward takes nothing from the card");
        /* a growth to 64 positions with room for the first layer's new mirrors only: the last
         * layer's old ones, which dev kept, go too */
        const size_t lay64 = 2 * (2 * 8 * 64 * 2) + 4 * 64 * 4 + 4 * (64 / 4 + 1) * 4;
        int placed = q38cc_mirror(&rc, &rm) && rc.cap == 32 && fake_chain_dev_bytes == b0 + 3 * lay;
        rm.kv_cap = 64; fake_chain_dev_cap = b0 + 2 * lay + lay64;
        int r4 = q38cc_mirror(&rc, &rm);
        none = 1;
        for (int i = 0; i < 4; i++) none = none && !kc[i] && !vc[i] && !ik[i] && !pk[i];
        tk(placed && r4 == 0 && rc.cap == 0 && none && fake_chain_dev_bytes == b0, "a refused growth leaves no mirror on the card");
        fake_chain_dev_cap = 0; rm.kv_cap = 32;
        g_cuda_chain = chain0;
    }

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

    printf(" a decode step's pooled slots on the server's threads\n");
    {   /* six experts as the pool holds them, three of them pooled (res 2 and 3), routes of eight
         * slots a row with a slot off the table (-1); each job's rows against q38cc_dm_serve's: the
         * slots through q38cc_pool_pairs, a row's added in rank order from zero */
        enum { H = 512, I = 256, NE = 6, K = 8, L = 3, R = 5 };
        const int tg = GQ_IQ2_S, td = GQ_Q2_0;
        const int64_t rg = gq_row_bytes(tg, H), rd = gq_row_bytes(td, I);
        const size_t pg = Q38CC_PAD(rg * I), st = 2 * pg + Q38CC_PAD(rd * H);
        uint8_t *pool = (uint8_t *)calloc(NE, st);
        for (int e = 0; e < NE && pool; e++) {
            t_fill(tg, pool + e * st, I, H); t_fill(tg, pool + e * st + pg, I, H); t_fill(td, pool + e * st + 2 * pg, H, I);
        }
        struct SrvCtx sc = {pool, st, NE};
        uint8_t res[L * NE];
        for (int k = 0; k < L * NE; k++) res[k] = (uint8_t)(k % NE < 2 ? 1 : k % NE == 5 ? 3 : 2);
        int ids[L][R * K]; float w[L][R * K];
        float *x = falloc(R * H), *out = falloc(R * H), *want = falloc(R * H);
        for (int k = 0; k < R * H; k++) x[k] = (float)(t_rnd() % 2001) / 1000.f - 1.f;
        for (int i = 0; i < L; i++) for (int a = 0; a < R * K; a++) {
            ids[i][a] = a % 7 == 6 ? -1 : (int)(t_rnd() % NE); w[i][a] = (float)(t_rnd() % 1000) / 1000.f;
        }
        for (int nt = 1; nt <= 4; nt += 3) {
            Q38CCSrv *sv = pool ? q38cc_srv_start(nt, H, I, K, NE, t_srv_where, &sc) : NULL;
            tk(sv && sv->started == nt, nt == 1 ? "the server: one thread up" : "the server: four threads up");
            if (!sv) break;
            volatile unsigned dfl = 0, hfl = 0;
            int same = 1, slots = 0; unsigned v = 0;
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            for (int rep = 0; rep < 2; rep++) for (int i = 0; i < L; i++) {
                const int n = rep ? R : 1;   /* a decode step's row, a verify's five */
                q38cc_srv_post(sv, i, n, ++v, tg, td, ids[i], w[i], res + i * NE);
                /* q38cc_dm_serve's arithmetic for the same slots */
                const uint8_t *gp[R * K]; const float *xp[R * K]; int pa[R * K], np = 0;
                for (int a = 0; a < n * K; a++) {
                    const int e = ids[i][a];
                    if (e < 0 || res[i * NE + e] < 2) continue;
                    gp[np] = pool + e * st; xp[np] = x + (size_t)(a / K) * H; pa[np++] = a;
                }
                float *gu = falloc(R * K * 2 * I), *hh = falloc(R * K * I), *dn = falloc(R * K * H);
                memset(want, 0, (size_t)R * H * sizeof(float));
                if (!np || q38cc_pool_pairs(gp, xp, np, H, I, tg, td, gu, hh, dn))
                    for (int p = 0; p < np; p++) {
                        float *ys = want + (size_t)(pa[p] / K) * H; const float *o = dn + (size_t)p * H, wv = w[i][pa[p]];
                        for (int d = 0; d < H; d++) ys[d] += wv * o[d];
                    }
                else same = 0;
                free(gu); free(hh); free(dn);
                slots += np;
                dfl = v;   /* the device's routes */
                for (long spin = 0; hfl < v && spin < 2000000000L; spin++) q38cc_relax();
                if (hfl != v || memcmp(out, want, (size_t)n * H * sizeof(float))) {
                    if (same) {
                        double md = 0; int at = -1;
                        for (int k = 0; k < n * H; k++) if (fabs((double)out[k] - want[k]) > md) { md = fabs((double)out[k] - want[k]); at = k; }
                        printf("       layer %d rows %d: %d slots, word %u of %u, |d| %.3g at %d (%g against %g)\n", i, n, np, hfl, v, md, at,
                               at >= 0 ? out[at] : 0.f, at >= 0 ? want[at] : 0.f);
                    }
                    same = 0;
                }
            }
            tk(same, "each job's rows: q38cc_dm_serve's bits, decode and verify rows");
            tk(sv->groups < sv->slots, "a verify's slots of one expert went through its matrices together");
            int cl = q38cc_srv_close(sv, 1, v);
            if (!(cl && sv->jobs == 2 * L && (int)sv->slots == slots)) printf("       closed %d, jobs %llu, slots %llu of %d\n", cl, sv->jobs, sv->slots, slots);
            tk(cl && sv->jobs == 2 * L && (int)sv->slots == slots, "the forward closed, its jobs and slots in the books");
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            q38cc_srv_post(sv, 0, 1, ++v, tg, td, ids[0], w[0], res + 0 * NE);
            q38cc_srv_post(sv, 1, 1, ++v, tg, td, ids[1], w[1], res + 1 * NE);
            tk(q38cc_srv_close(sv, 0, v) && hfl == v, "a failed frame: the jobs skipped, the host's word past them");
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            q38cc_srv_post(sv, 2, 1, ++v, tg, td, ids[2], w[2], res + 2 * NE);
            dfl = v;
            for (long spin = 0; hfl < v && spin < 2000000000L; spin++) q38cc_relax();
            tk(q38cc_srv_close(sv, 1, v) && hfl == v, "and the next forward served");
            q38cc_srv_stop(sv);
        }
        free(pool); free(x); free(out); free(want);
    }

    printf(" the MTP head's experts in Q2_0 for the card (q38cc_q2_0_block)\n");
    {   /* blocks of weights as gq_blk_q2_0 decodes them back: within the 2-bit grid's error, which
         * on Gaussian weights is about 0.41 of their RMS; zeros stay zeros, a flat block is exact on
         * either sign (the grid's 2 or its -1) */
        float x[64], y[64]; uint8_t b[18];
        double err = 0, mag = 0, worst = 0;
        for (int t = 0; t < 400; t++) {
            const float sc = 0.001f + (float)(t_rnd() % 1000) / 1000.f * 0.05f, skew = (t & 1) ? 1.f : -1.f;
            double e = 0, m = 0;
            for (int j = 0; j < 64; j++) {
                float g = 0.f;
                for (int k = 0; k < 6; k++) g += (float)(t_rnd() % 10000) / 10000.f - 0.5f;
                x[j] = sc * g + (j == t % 64 ? skew * 3.f * sc : 0.f);   /* one weight well out on one side */
            }
            q38cc_q2_0_block(x, b);
            if (gq_dequant_row(GQ_Q2_0, b, y, 64)) { worst = 1e9; break; }
            for (int j = 0; j < 64; j++) { e += (double)(x[j] - y[j]) * (x[j] - y[j]); m += (double)x[j] * x[j]; }
            err += e; mag += m;
            if (e / m > worst) worst = e / m;
        }
        tk(sqrt(err / mag) < 0.45 && sqrt(worst) < 0.6, "random blocks within the grid's error");
        for (int j = 0; j < 64; j++) x[j] = 0.f;
        q38cc_q2_0_block(x, b);
        int zero = !gq_dequant_row(GQ_Q2_0, b, y, 64);
        for (int j = 0; j < 64; j++) zero = zero && y[j] == 0.f;
        tk(zero, "a block of zeros decodes to zeros");
        int flat = 1;
        for (int s = -1; s <= 1; s += 2) {
            for (int j = 0; j < 64; j++) x[j] = s * 0.0123f;
            q38cc_q2_0_block(x, b);
            flat = flat && !gq_dequant_row(GQ_Q2_0, b, y, 64);
            for (int j = 0; j < 64; j++) flat = flat && fabsf(y[j] - x[j]) <= 1e-3f * 0.0123f;
        }
        tk(flat, "a flat block, positive or negative, decodes to itself");
    }

    if (t_fails) { printf("test_qwen38_cuda_chain: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_cuda_chain: Qwen3.8's chain on the card gives the oracle's tokens\n");
    return 0;
}
