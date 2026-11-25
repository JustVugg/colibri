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
 * hand: the fixture has no GGUF, so no host pool and no promotion runs above.
 * Q38_CC_GRAPH=1 on a fake card whose graphs record the ops and replay them (fg_*): the
 * tokens and the last logits of the switch-off run bit for bit, with captures that break,
 * a stream that refuses to capture, and a verify's rejected rows rolled back. */
#define main qwen38_main_unused
#include "../qwen38.c"
#undef main

#ifdef _WIN32
#include <direct.h>
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

/* ---- Q38_CC_GRAPH on the fake card: graphs that record the ops and replay them ----
 * fg_install gives the fake's table the graph entries: between graph_begin and graph_end the
 * ops a segment may hold (copy, matmul, norm, rope, dnconv, dnrec, ew) are recorded, not run,
 * as a CUDA capture does; graph_launch runs them in their order with their arguments. Any
 * other op inside a capture breaks it (graph_end then says broke: nothing ran), as an op the
 * capture refuses would; fg_break_at = n breaks the n-th recorded op instead; fg_refuse makes
 * the stream refuse to capture. */
typedef struct { int op; ColiCudaTensor *t; CcBuf *b[8]; size_t z[2]; int iv;
                 union { CcNorm n; CcRope r; CcDnConv dc; CcDnRec dr; CcEw e; CcRoute rt; CcMoe mo; } p; } FgOp;
typedef struct { int n, cap; FgOp *ops; } FgGraph;
static FgGraph *fg_cur;
static int fg_broke, fg_break_at, fg_recorded, fg_refuse;
static unsigned long long fg_replays, fg_captures, fg_frees, fg_breaks, fg_refused, fg_pushed;
static unsigned long long fg_route_rec, fg_moe_rec, fg_waits;
static FgOp *fg_push(int op) {
    fg_pushed++;
    if (fg_break_at && ++fg_recorded == fg_break_at) { fg_broke = 1; return NULL; }
    if (fg_cur->n == fg_cur->cap) {
        fg_cur->cap = fg_cur->cap ? 2 * fg_cur->cap : 64;
        fg_cur->ops = (FgOp *)realloc(fg_cur->ops, (size_t)fg_cur->cap * sizeof(FgOp));
    }
    FgOp *o = &fg_cur->ops[fg_cur->n++];
    memset(o, 0, sizeof *o); o->op = op;
    return o;
}
static int fg_copy(CcBuf *dst, size_t doff, CcBuf *src, size_t soff, size_t n) {
    if (!fg_cur) return fcc_copy(dst, doff, src, soff, n);
    FgOp *o = fg_push(1); if (!o) return 0;
    o->b[0] = dst; o->z[0] = doff; o->b[1] = src; o->z[1] = soff; o->iv = (int)n; return 1;
}
static int fg_matmul(ColiCudaTensor *t, CcBuf *x, size_t xo, CcBuf *y, size_t yo, int S) {
    if (!fg_cur) return fcc_matmul(t, x, xo, y, yo, S);
    FgOp *o = fg_push(2); if (!o) return 0;
    o->t = t; o->b[0] = x; o->z[0] = xo; o->b[1] = y; o->z[1] = yo; o->iv = S; return 1;
}
static int fg_norm(CcBuf *x, CcBuf *w, CcBuf *y, const CcNorm *p) {
    if (!fg_cur) return fcc_norm(x, w, y, p);
    FgOp *o = fg_push(3); if (!o) return 0;
    o->b[0] = x; o->b[1] = w; o->b[2] = y; o->p.n = *p; return 1;
}
static int fg_rope(CcBuf *x, CcBuf *cs, const CcRope *p) {
    if (!fg_cur) return fcc_rope(x, cs, p);
    FgOp *o = fg_push(4); if (!o) return 0;
    o->b[0] = x; o->b[1] = cs; o->p.r = *p; return 1;
}
static int fg_dnconv(CcBuf *in, CcBuf *w, CcBuf *ring, CcBuf *out, CcBuf *snap, const CcDnConv *p) {
    if (!fg_cur) return fcc_dnconv(in, w, ring, out, snap, p);
    FgOp *o = fg_push(5); if (!o) return 0;
    o->b[0] = in; o->b[1] = w; o->b[2] = ring; o->b[3] = out; o->b[4] = snap; o->p.dc = *p; return 1;
}
static int fg_dnrec(int KD, CcBuf *cv, CcBuf *ab, CcBuf *z, CcBuf *st, CcBuf *prm, CcBuf *y, CcBuf *snap, const CcDnRec *p) {
    if (!fg_cur) return fcc_dnrec(KD, cv, ab, z, st, prm, y, snap, p);
    FgOp *o = fg_push(6); if (!o) return 0;
    o->iv = KD; o->b[0] = cv; o->b[1] = ab; o->b[2] = z; o->b[3] = st; o->b[4] = prm; o->b[5] = y; o->b[6] = snap; o->p.dr = *p; return 1;
}
static int fg_ew(CcBuf *y, CcBuf *a, CcBuf *b, CcBuf *c, CcBuf *e, const CcEw *p) {
    if (!fg_cur) return fcc_ew(y, a, b, c, e, p);
    FgOp *o = fg_push(7); if (!o) return 0;
    o->b[0] = y; o->b[1] = a; o->b[2] = b; o->b[3] = c; o->b[4] = e; o->p.e = *p; return 1;
}
static int fg_route(CcBuf *logits, CcBuf *bias, CcBuf *ids, CcBuf *w, const CcRoute *p) {
    if (!fg_cur) return fcc_route(logits, bias, ids, w, p);
    FgOp *o = fg_push(8); if (!o) return 0;
    fg_route_rec++;
    o->b[0] = logits; o->b[1] = bias; o->b[2] = ids; o->b[3] = w; o->p.rt = *p; return 1;
}
/* cc_moe on the fake card: fmt 1 via fcc_moe, GGUF block formats (>= 100) via gq_dequant_row */
static int fg_moe_exec(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) {
    if (!p || p->fmt < 100) return fcc_moe(tab, x, ids, w, hid, y, hy, p);
    if (!fcc_ready() || !tab || !ids || !w || !hid || !y) return 0;
    const int tg = p->fmt - 100, td = (p->fmt_d >= 100 ? p->fmt_d : p->fmt) - 100;
    const int64_t rg = gq_row_bytes(tg, p->D), rd = gq_row_bytes(td, p->I);
    if (rg <= 0 || rd <= 0 || p->D > 256 || p->I > 256) return 0;
    fake_chain_ops++;
    const CcExpert *tb = (const CcExpert *)tab->d + p->t_off;
    const int *id = (const int *)ids->d;
    const int LS = p->lay_S > 0 ? p->lay_S : p->S;
    if (p->lay_s0 < 0 || p->lay_s0 > LS - p->S || ((p->parts & 1) && (LS != p->S || p->lay_s0))) return 0;
    float *hd = hid->d + (size_t)LS * p->K * p->I;
    if (p->parts & 1) {
        if (!x) return 0;
        float rgw[256], ruw[256], rdw[256];
        for (int s = 0; s < p->S; s++) {
            const float *xr = x->d + p->x_off + s * p->x_row;
            for (int k = 0; k < p->K; k++) {
                int e = id[p->i_off + s * p->i_row + k];
                float *h = hid->d + ((size_t)s * p->K + k) * p->I, *dn = hd + ((size_t)s * p->K + k) * p->D;
                if (e < 0 || e >= p->E || !tb[e].g) { memset(dn, 0, (size_t)p->D * 4); continue; }
                const uint8_t *gw = (const uint8_t *)tb[e].g, *uw = (const uint8_t *)tb[e].u, *dw = (const uint8_t *)tb[e].d;
                for (int o = 0; o < p->I; o++) {
                    if (gq_dequant_row(tg, gw + (size_t)o * (size_t)rg, rgw, p->D) ||
                        gq_dequant_row(tg, uw + (size_t)o * (size_t)rg, ruw, p->D)) return 0;
                    float ga = 0.f, ua = 0.f;
                    for (int i = 0; i < p->D; i++) { ga += xr[i] * rgw[i]; ua += xr[i] * ruw[i]; }
                    h[o] = ga * fcc_sig(ga) * ua;
                }
                for (int o = 0; o < p->D; o++) {
                    if (gq_dequant_row(td, dw + (size_t)o * (size_t)rd, rdw, p->I)) return 0;
                    float a = 0.f; for (int i = 0; i < p->I; i++) a += h[i] * rdw[i];
                    dn[o] = a;
                }
            }
        }
    }
    if (p->parts & 2) {
        for (int s = 0; s < p->S; s++) {
            float *yr = y->d + p->y_off + s * p->y_row;
            if (p->h_off >= 0 && hy) memcpy(yr, hy->d + p->h_off + s * p->D, (size_t)p->D * 4);
            else memset(yr, 0, (size_t)p->D * 4);
            for (int k = 0; k < p->K; k++) {
                int e = id[p->i_off + s * p->i_row + k];
                if (e < 0 || e >= p->E || !tb[e].g) continue;
                float wk = w->d[p->w_off + s * p->w_row + k];
                const float *dn = hd + ((size_t)(p->lay_s0 + s) * p->K + k) * p->D;
                for (int d = 0; d < p->D; d++) yr[d] += wk * dn[d];
            }
        }
    }
    return 1;
}
static int fg_moe(CcBuf *tab, CcBuf *x, CcBuf *ids, CcBuf *w, CcBuf *hid, CcBuf *y, CcBuf *hy, const CcMoe *p) {
    if (!fg_cur) return fg_moe_exec(tab, x, ids, w, hid, y, hy, p);
    FgOp *o = fg_push(9); if (!o) return 0;
    fg_moe_rec++;
    o->b[0] = tab; o->b[1] = x; o->b[2] = ids; o->b[3] = w; o->b[4] = hid; o->b[5] = y; o->b[6] = hy; o->p.mo = *p; return 1;
}
/* the ops no segment holds: inside a capture they break it */
static int fg_signal(CcBuf *flag, size_t word, unsigned v) { if (fg_cur) { fg_broke = 1; return 0; } return fcc_signal(flag, word, v); }
static int fg_wait_host(CcBuf *flag, size_t word, unsigned v) { if (fg_cur) { fg_broke = 1; return 0; } fg_waits++; return fcc_wait_host(flag, word, v); }
static int fg_copy_regions(CcBuf *dst, CcBuf *src, const CcRegion *r, int n) { if (fg_cur) { fg_broke = 1; return 0; } return fcc_copy_regions(dst, src, r, n); }
static int fg_zero(CcBuf *dst, size_t off, size_t n) { if (fg_cur) { fg_broke = 1; return 0; } return fcc_zero(dst, off, n); }
static int fg_write(CcBuf *dst, size_t off, const void *src, size_t bytes) { if (fg_cur) { fg_broke = 1; return 0; } return fcc_write(dst, off, src, bytes); }
static int fg_attn(CcBuf *q, CcBuf *kc, CcBuf *vc, CcBuf *o, CcBuf *gate, CcBuf *sel, const CcAttn *p) {
    if (fg_cur) { fg_broke = 1; return 0; } return fcc_attn(q, kc, vc, o, gate, sel, p);
}
static int fg_qsa(CcBuf *src, CcBuf *w, CcBuf *pk, CcBuf *cs, CcBuf *sc, CcBuf *sel, const CcQsa *p) {
    if (fg_cur) { fg_broke = 1; return 0; } return fcc_qsa(src, w, pk, cs, sc, sel, p);
}
static int fg_ple(CcBuf *keys, CcBuf *hyp, CcBuf *val, CcBuf *prm, CcBuf *gated, CcBuf *normv, CcBuf *conv, CcBuf *ring, const CcPle *p) {
    if (fg_cur) { fg_broke = 1; return 0; } return fcc_ple(keys, hyp, val, prm, gated, normv, conv, ring, p);
}
static int fg_begin(void) {
    if (fg_cur || !fcc_ready()) return 0;
    if (fg_refuse) { fg_refused++; return 0; }   /* a stream that will not capture, the device fine */
    fg_cur = (FgGraph *)calloc(1, sizeof *fg_cur); fg_broke = 0;
    return 1;
}
/* where a GGUF's block matrices quantize (graph_scratch): fixed, or (fg_scratch_moves) moved before every segment */
static int fg_scratch_moves; static uintptr_t fg_scratch_at = 0x1000;
static const void *fg_scratch_of(void) { return (const void *)(fg_scratch_moves ? (fg_scratch_at += 64) : fg_scratch_at); }
static void fg_free(void *g) { FgGraph *x = (FgGraph *)g; if (x) { free(x->ops); free(x); fg_frees++; } }
static void *fg_end(int *broke) {
    FgGraph *g = fg_cur; fg_cur = NULL;
    *broke = fg_broke;
    if (fg_broke) fg_breaks++;
    if (!g || fg_broke) { if (g) { free(g->ops); free(g); } return NULL; }
    fg_captures++;
    return g;
}
static int fg_launch(void *gv) {
    FgGraph *g = (FgGraph *)gv; int ok = fcc_ready() && g && !fg_cur;
    for (int k = 0; ok && k < g->n; k++) {
        FgOp *o = &g->ops[k];
        switch (o->op) {
        case 1: ok = fcc_copy(o->b[0], o->z[0], o->b[1], o->z[1], (size_t)o->iv); break;
        case 2: ok = fcc_matmul(o->t, o->b[0], o->z[0], o->b[1], o->z[1], o->iv); break;
        case 3: ok = fcc_norm(o->b[0], o->b[1], o->b[2], &o->p.n); break;
        case 4: ok = fcc_rope(o->b[0], o->b[1], &o->p.r); break;
        case 5: ok = fcc_dnconv(o->b[0], o->b[1], o->b[2], o->b[3], o->b[4], &o->p.dc); break;
        case 6: ok = fcc_dnrec(o->iv, o->b[0], o->b[1], o->b[2], o->b[3], o->b[4], o->b[5], o->b[6], &o->p.dr); break;
        case 7: ok = fcc_ew(o->b[0], o->b[1], o->b[2], o->b[3], o->b[4], &o->p.e); break;
        case 8: ok = fcc_route(o->b[0], o->b[1], o->b[2], o->b[3], &o->p.rt); break;
        default: ok = fg_moe_exec(o->b[0], o->b[1], o->b[2], o->b[3], o->b[4], o->b[5], o->b[6], &o->p.mo); break;
        }
    }
    if (ok) fg_replays++;
    return ok;
}
static ColiCudaChainOps fg_saved;
static void fg_install(void) {
    fg_saved = fake_chain_table;
    ColiCudaChainOps *t = &fake_chain_table;
    t->size = sizeof *t;   /* the optional ops qwen38 asks for: these, tensor_dev and dev_addr (the fake's), pick_rows */
    t->copy = fg_copy; t->matmul = fg_matmul; t->norm = fg_norm; t->rope = fg_rope; t->dnconv = fg_dnconv; t->dnrec = fg_dnrec; t->ew = fg_ew;
    t->route = fg_route; t->moe = fg_moe; t->signal = fg_signal; t->wait_host = fg_wait_host;
    t->copy_regions = fg_copy_regions; t->zero = fg_zero; t->write = fg_write; t->attn = fg_attn; t->qsa = fg_qsa; t->ple = fg_ple;
    t->graph_begin = fg_begin; t->graph_end = fg_end; t->graph_launch = fg_launch; t->graph_free = fg_free;
    t->graph_scratch = fg_scratch_of;
}
static void fg_uninstall(void) { fake_chain_table = fg_saved; }
static char *fg_slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); char *b = NULL; long len = 0;
    if (f && !fseek(f, 0, SEEK_END) && (len = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET) && (b = (char *)malloc((size_t)len + 1)))
        len = (long)fread(b, 1, (size_t)len, f);
    if (f) fclose(f);
    if (b) b[len] = 0;
    *n = b ? (size_t)len : 0;
    return b;
}
/* the engine with its stdout and stderr in files, the last logits in DUMP */
static int fg_run(char **argv, const char *tag) {
    char out[64], err[64], dump[64];
    snprintf(out, sizeof out, "q38g_%s.out", tag); snprintf(err, sizeof err, "q38g_%s.err", tag); snprintf(dump, sizeof dump, "q38g_%s.f32", tag);
    setenv("DUMP", dump, 1);
    fflush(stdout); fflush(stderr);
    int so = dup(1), se = dup(2), rc = -1;
    if (freopen(out, "wb", stdout) && freopen(err, "wb", stderr)) rc = qwen38_main_unused(4, argv);
    fflush(stdout); fflush(stderr); dup2(so, 1); dup2(se, 2); close(so); close(se);
    unsetenv("DUMP");
    qt_shutdown();
    return rc;
}
/* Q38_CC_GRAPH off, then on: the same tokens (the engine's printout) and the same last logits bit
 * for bit; *took (a counter of the graphs' work, NULL: none) moved with the switch on and only then */
static void t_graph_switch(char **argv, const char *title, const unsigned long long *took) {
    printf(" %s\n", title);
    setenv("COLI_CUDA_CHAIN", "1", 1);
    fake_chain_frames = 0; fake_chain_fail_at = 0;
    unsetenv("Q38_CC_GRAPH");
    unsigned long long t0 = took ? *took : 0;
    int rc0 = fg_run(argv, "off");
    unsigned long long t1 = took ? *took : 0;
    setenv("Q38_CC_GRAPH", "1", 1);
    int rc1 = fg_run(argv, "on");
    unsigned long long t2 = took ? *took : 1;
    unsetenv("Q38_CC_GRAPH");
    size_t n0, n1, l0, l1;
    char *o0 = fg_slurp("q38g_off.out", &n0), *o1 = fg_slurp("q38g_on.out", &n1);
    char *d0 = fg_slurp("q38g_off.f32", &l0), *d1 = fg_slurp("q38g_on.f32", &l1);
    tk(rc0 == 0 && rc1 == 0 && fake_chain_frames > 8, "both runs on the chain within the oracle's limits");
    tk(t1 == t0 && t2 > t1, "the graphs' work ran with the switch on and only then");
    tk(o0 && o1 && n0 == n1 && n0 > 0 && !memcmp(o0, o1, n0), "the switch on prints the switch-off run's tokens and cosine");
    tk(d0 && d1 && l0 == l1 && l0 > 0 && !memcmp(d0, d1, l0), "the last logits are the switch-off run's bit for bit");
    free(o0); free(o1); free(d0); free(d1);
    static const char *const tmp[] = {"q38g_off.out", "q38g_on.out", "q38g_off.err", "q38g_on.err", "q38g_off.f32", "q38g_on.f32"};
    for (size_t i = 0; i < sizeof tmp / sizeof *tmp; i++) remove(tmp[i]);
}

/* ---- tiny block-typed qwen38 GGUF fixture + dm_on + Q38_CC_GRAPH=1 ----
 * A self-contained 4-layer qwen4exp GGUF (DeltaNet, QSA, DeltaNet, QSA; PLE on layer 0;
 * hidden 32, inter 32, 4 experts, top-2) whose routed experts are GQ_Q8_0 blocks, run on the
 * fake card with dm_on=1 and Q38_CC_GRAPH=1 against Q38_CC_GRAPH=0:
 *   1. all experts on the card: router, routes, shared expert and routed experts captured and
 *      replayed in the segment graphs, bit-for-bit parity with Q38_CC_GRAPH=0;
 *   2. graph_scratch invalidation: moving scratch before every segment (no capture/replay),
 *      moving scratch mid-run (old graphs invalidated, never replayed), and missing graph_scratch
 *      refused at setup;
 *   3. pend_v layer-after-hosted fallback: layer 1 with one expert on the host (hosted -> op by
 *      op, sets pend_v), layer 2 after it also op by op (pend_v != 0 -> no graph captured), layers
 *      0 and 3 captured and replayed, bit-for-bit parity with Q38_CC_GRAPH=0. */
typedef struct {
    char name[96];
    int type, ndims;
    uint64_t ne[4];
    uint8_t *data;
    size_t nbytes;
} TgTensor;
typedef struct { TgTensor *t; int n, cap; } TgList;

static void tg_add(TgList *g, const char *name, int type, int ndims, uint64_t d0, uint64_t d1, uint64_t d2, int is_norm) {
    if (g->n == g->cap) {
        g->cap = g->cap ? 2 * g->cap : 64;
        g->t = (TgTensor *)realloc(g->t, (size_t)g->cap * sizeof *g->t);
    }
    TgTensor *x = &g->t[g->n++];
    memset(x, 0, sizeof *x);
    snprintf(x->name, sizeof x->name, "%s", name);
    x->type = type; x->ndims = ndims;
    x->ne[0] = d0; x->ne[1] = ndims > 1 ? d1 : 1; x->ne[2] = ndims > 2 ? d2 : 1; x->ne[3] = 1;
    if (type == GQ_F32) {
        size_t nel = (size_t)(x->ne[0] * x->ne[1] * x->ne[2]);
        x->nbytes = nel * sizeof(float);
        float *f = (float *)malloc(x->nbytes);
        for (size_t k = 0; k < nel; k++) {
            float v = ((float)(t_rnd() % 2001) / 1000.f - 1.f) * 0.05f;
            f[k] = is_norm ? 1.f + v : v;
        }
        x->data = (uint8_t *)f;
    } else {
        int rows = (int)(x->ne[1] * x->ne[2]), cols = (int)x->ne[0];
        int64_t rb = gq_row_bytes(type, cols);
        if (rb <= 0) return;
        x->nbytes = (size_t)rows * (size_t)rb;
        x->data = (uint8_t *)calloc(1, x->nbytes);
        t_fill(type, x->data, rows, cols);
    }
}
static void tg_wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void tg_wr_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void tg_wr_str(FILE *f, const char *s) { uint64_t n = (uint64_t)strlen(s); tg_wr_u64(f, n); fwrite(s, 1, (size_t)n, f); }
static void tg_wr_kv_str(FILE *f, const char *k, const char *v) { tg_wr_str(f, k); tg_wr_u32(f, GGUF_T_STRING); tg_wr_str(f, v); }
static void tg_wr_kv_i64_arr(FILE *f, const char *k, const int64_t *a, uint64_t n) {
    tg_wr_str(f, k); tg_wr_u32(f, GGUF_T_ARRAY); tg_wr_u32(f, GGUF_T_I64); tg_wr_u64(f, n);
    fwrite(a, sizeof(int64_t), (size_t)n, f);
}
static void tg_pad32(FILE *f) {
    long pos = ftell(f), rem = pos & 31;
    if (rem) { uint8_t z[32] = {0}; fwrite(z, 1, (size_t)(32 - rem), f); }
}

static int t_make_tiny_gguf(const char *dir) {
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0755);
#endif
    char path[256];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fputs("{\n"
          "  \"model_type\": \"qwen4_exp_text\",\n"
          "  \"hidden_act\": \"silu\",\n"
          "  \"output_gate_type\": \"sigmoid\",\n"
          "  \"attention_bias\": false,\n"
          "  \"tie_word_embeddings\": false,\n"
          "  \"hidden_size\": 32,\n"
          "  \"num_hidden_layers\": 4,\n"
          "  \"vocab_size\": 64,\n"
          "  \"max_position_embeddings\": 128,\n"
          "  \"eos_token_id\": 0,\n"
          "  \"rms_norm_eps\": 1e-6,\n"
          "  \"rope_theta\": 10000.0,\n"
          "  \"hc_count\": 4,\n"
          "  \"hc_lowrank\": 8,\n"
          "  \"num_attention_heads\": 4,\n"
          "  \"num_key_value_heads\": 2,\n"
          "  \"head_dim\": 8,\n"
          "  \"partial_rotary_factor\": 1.0,\n"
          "  \"indexer_n_heads\": 2,\n"
          "  \"indexer_kv_heads\": 1,\n"
          "  \"indexer_head_dim\": 8,\n"
          "  \"indexer_budget\": 8,\n"
          "  \"indexer_compress_ratio\": 4,\n"
          "  \"num_experts\": 4,\n"
          "  \"num_experts_per_tok\": 2,\n"
          "  \"moe_intermediate_size\": 32,\n"
          "  \"shared_expert_intermediate_size\": 32,\n"
          "  \"norm_topk_prob\": true,\n"
          "  \"linear_num_key_heads\": 2,\n"
          "  \"linear_num_value_heads\": 4,\n"
          "  \"linear_key_head_dim\": 8,\n"
          "  \"linear_value_head_dim\": 8,\n"
          "  \"linear_conv_kernel_dim\": 4,\n"
          "  \"ple_embed_dim\": 16,\n"
          "  \"ple_conv_kernel_size\": 4,\n"
          "  \"ngram_size\": 3,\n"
          "  \"heads_per_ngram\": 1,\n"
          "  \"split_ngram_parts\": 1,\n"
          "  \"ple_layer_ids\": [1],\n"
          "  \"layer_types\": [\"linear_attention\", \"full_attention\", \"linear_attention\", \"full_attention\"]\n"
          "}\n", f);
    fclose(f);

    snprintf(path, sizeof path, "%s/gguf.json", dir);
    f = fopen(path, "wb");
    if (!f) return 0;
    fputs("{\"gguf\": \"tiny.gguf\"}\n", f);
    fclose(f);

    TgList g = {0};
    unsigned saved_seed = t_seed; t_seed = 424242u;
    /* top-level */
    tg_add(&g, "token_embd.weight", GQ_F32, 2, 32, 64, 1, 0);
    tg_add(&g, "output.weight", GQ_F32, 2, 32, 64, 1, 0);
    tg_add(&g, "output_hc_norm.weight", GQ_F32, 1, 128, 1, 1, 1);
    tg_add(&g, "output_hc_down.weight", GQ_F32, 2, 128, 8, 1, 0);
    tg_add(&g, "output_hc_up.weight", GQ_F32, 2, 8, 128, 1, 0);
    /* PLE on layer 0 */
    tg_add(&g, "blk.0.ple_key.weight", GQ_F32, 2, 16, 128, 1, 0);
    tg_add(&g, "blk.0.ple_value.weight", GQ_F32, 2, 16, 32, 1, 0);
    tg_add(&g, "blk.0.ple_norm_key.weight", GQ_F32, 1, 128, 1, 1, 1);
    tg_add(&g, "blk.0.ple_norm_query.weight", GQ_F32, 1, 128, 1, 1, 1);
    tg_add(&g, "blk.0.ple_norm_conv.weight", GQ_F32, 1, 128, 1, 1, 1);
    tg_add(&g, "blk.0.ple_conv1d.weight", GQ_F32, 2, 4, 128, 1, 0);
    tg_add(&g, "per_layer_token_embd.weight", GQ_F32, 2, 8, 64, 1, 0);
    for (int i = 0; i < 4; i++) {
        char nm[96];
        snprintf(nm, sizeof nm, "blk.%d.hc_attn_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 128, 1, 1, 1);
        snprintf(nm, sizeof nm, "blk.%d.hc_attn_down.weight", i); tg_add(&g, nm, GQ_F32, 2, 128, 8, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.hc_attn_up.weight", i); tg_add(&g, nm, GQ_F32, 2, 8, 128, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.hc_attn_inject.weight", i); tg_add(&g, nm, GQ_F32, 2, 128, 4, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.hc_ffn_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 128, 1, 1, 1);
        snprintf(nm, sizeof nm, "blk.%d.hc_ffn_down.weight", i); tg_add(&g, nm, GQ_F32, 2, 128, 8, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.hc_ffn_up.weight", i); tg_add(&g, nm, GQ_F32, 2, 8, 128, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.hc_ffn_inject.weight", i); tg_add(&g, nm, GQ_F32, 2, 128, 4, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 4, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_shexp.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up_shexp.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down_shexp.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_inp_shexp.weight", i); tg_add(&g, nm, GQ_F32, 1, 32, 1, 1, 0);
        /* block-typed routed experts (GQ_Q8_0) */
        snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", i); tg_add(&g, nm, GQ_Q8_0, 3, 32, 32, 4, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", i); tg_add(&g, nm, GQ_Q8_0, 3, 32, 32, 4, 0);
        snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", i); tg_add(&g, nm, GQ_Q8_0, 3, 32, 32, 4, 0);
        if (i & 1) {   /* QSA attention */
            snprintf(nm, sizeof nm, "blk.%d.attn_q.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 64, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.attn_k.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 16, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.attn_v.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 16, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.attn_output.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.attn_q_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 8, 1, 1, 1);
            snprintf(nm, sizeof nm, "blk.%d.attn_k_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 8, 1, 1, 1);
            snprintf(nm, sizeof nm, "blk.%d.indexer.q_proj.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 16, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.indexer.k_proj.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 8, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.indexer.q_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 8, 1, 1, 1);
            snprintf(nm, sizeof nm, "blk.%d.indexer.k_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 8, 1, 1, 1);
        } else {   /* DeltaNet */
            snprintf(nm, sizeof nm, "blk.%d.attn_qkv.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 64, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.attn_gate.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_beta.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 4, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_alpha.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 4, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_conv1d.weight", i); tg_add(&g, nm, GQ_F32, 2, 4, 64, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_dt.bias", i); tg_add(&g, nm, GQ_F32, 1, 4, 1, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_a", i); tg_add(&g, nm, GQ_F32, 1, 4, 1, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_norm.weight", i); tg_add(&g, nm, GQ_F32, 1, 8, 1, 1, 0);
            snprintf(nm, sizeof nm, "blk.%d.ssm_out.weight", i); tg_add(&g, nm, GQ_F32, 2, 32, 32, 1, 0);
        }
    }
    t_seed = saved_seed;

    snprintf(path, sizeof path, "%s/tiny.gguf", dir);
    f = fopen(path, "wb");
    if (!f) { for (int i = 0; i < g.n; i++) free(g.t[i].data); free(g.t); return 0; }
    fwrite("GGUF", 1, 4, f);
    tg_wr_u32(f, 3);
    tg_wr_u64(f, (uint64_t)g.n);
    tg_wr_u64(f, 4);
    static const int64_t ple_mul[3] = {1, 3, 5}, ple_voc[2] = {32, 32}, ple_off[2] = {0, 32};
    tg_wr_kv_str(f, "general.architecture", "qwen4exp");
    tg_wr_kv_i64_arr(f, "qwen4exp.ple.layer_multipliers", ple_mul, 3);
    tg_wr_kv_i64_arr(f, "qwen4exp.ple.head_vocab_sizes", ple_voc, 2);
    tg_wr_kv_i64_arr(f, "qwen4exp.ple.head_offsets", ple_off, 2);
    uint64_t off = 0;
    for (int i = 0; i < g.n; i++) {
        tg_wr_str(f, g.t[i].name);
        tg_wr_u32(f, (uint32_t)g.t[i].ndims);
        for (int d = 0; d < g.t[i].ndims; d++) tg_wr_u64(f, g.t[i].ne[d]);
        tg_wr_u32(f, (uint32_t)g.t[i].type);
        tg_wr_u64(f, off);
        off += (uint64_t)((g.t[i].nbytes + 31u) & ~(size_t)31u);
    }
    tg_pad32(f);
    for (int i = 0; i < g.n; i++) {
        fwrite(g.t[i].data, 1, g.t[i].nbytes, f);
        tg_pad32(f);
        free(g.t[i].data);
    }
    free(g.t);
    fclose(f);
    return 1;
}

static void t_gguf_upload_at(void *dev, size_t cap, const void *w, int fmt, int I, int O) {
    if (fmt < 100) return;
    int64_t rb = gq_row_bytes(fmt - 100, I);
    if (rb > 0 && (size_t)rb * (size_t)O <= cap) memcpy(dev, w, (size_t)rb * (size_t)O);
}
static Q38CChain *t_srv_ch;
static Model *t_srv_m;
static int t_srv_n;
static int t_serve_layer1(unsigned v) { return q38cc_dm_serve(t_srv_ch, t_srv_m, 1, t_srv_n, v); }
static int t_layer_has_graph(const Q38CChain *ch, int i) {
    if (!ch || !ch->gr) return 0;
    for (int s = 0; s < Q38G_SEGS; s++)
        for (int k = 0; k < Q38G_WAYS; k++)
            if (ch->gr[((size_t)i * Q38G_SEGS + s) * Q38G_WAYS + k].exec) return 1;
    return 0;
}

/* run the tiny block-typed GGUF through q38_forward (3-token prompt + 6 decode steps), returning
 * all 7 steps' logits (7 * 64 floats) and optional chain stats; host_l1=1 evicts layer 1's expert 3,
 * move_mid=1 bumps fg_scratch_at after decode step 2 */
static int t_gguf_dm_run(const char *dir, int graph_on, int host_l1, int move_mid, float *logits_out,
                         int *dm_ok, int *g_on_out, int *l_gr, unsigned long long *caps, unsigned long long *laps,
                         unsigned long long *eagers, int *part_out) {
    char gp[256]; snprintf(gp, sizeof gp, "%s/tiny.gguf", dir);
    if (gguf_open(&g_q38gg, gp)) return 0;
    g_q38_gguf = 1; g_q38_a_direct = 1; g_q38_dn_tiled = 1; g_q38_gguf_tier = 0;
    setenv("COLI_CUDA", "1", 1); setenv("COLI_CUDA_CHAIN", "1", 1);
    setenv("Q38_CC_GRAPH", graph_on ? "1" : "0", 1);
    if (host_l1) setenv("COLI_CUDA_CHAIN_PROMOTE", "0", 1); else unsetenv("COLI_CUDA_CHAIN_PROMOTE");
    Model m; model_init(&m, dir, 4, 8);
    m.max_t = 32; ensure_kv(&m);
    q38_tier_start(&m, 4);
    q38_trunk_cpu_int8(&m);
    if (g_q38_gguf_tier) q38_tier_warm(&m, 0);
    if (host_l1) {   /* evict layer 1's expert 3 before q38cc_setup and hold it on the host */
        pthread_mutex_lock(&G.mx);
        res_set(qs(1, 3), 0); qs(1, 3)->aslot = 0;
        for (int i = 0; i < 4; i++) for (int e = 0; e < 4; e++) if (qs(i, e)->resident) qs(i, e)->heat = 1u << 20;
        G.budget[0] = G.used[0];
        pthread_mutex_unlock(&G.mx);
    }
    Q38CChain *ch = q38cc_setup(&m, 1);
    g_cuda_chain = ch != NULL;
    if (dm_ok) *dm_ok = ch && ch->ok && ch->dm_on && ch->dm_fmt == QT_GGUF;
    if (g_on_out) *g_on_out = ch ? ch->g_on : 0;
    if (host_l1 && ch) { t_srv_ch = ch; t_srv_m = &m; fake_wait_hook = t_serve_layer1; }
    static const int prompt[3] = {1, 2, 3};
    int tok = 3, pos = 0;
    t_srv_n = 3;
    float *lg = q38_forward(&m, prompt, 3, pos, 1, NULL);
    pos += 3;
    if (lg && logits_out) memcpy(logits_out, lg, 64 * sizeof(float));
    int best = 0; for (int v = 1; lg && v < 64; v++) if (lg[v] > lg[best]) best = v;
    free(lg); tok = best;
    for (int step = 0; step < 6; step++) {
        if (move_mid && step == 3) fg_scratch_at += 64;   /* simulate a prompt growing bd_xq_scratch */
        t_srv_n = 1;
        lg = q38_forward(&m, &tok, 1, pos, 1, NULL);
        pos++;
        if (lg && logits_out) memcpy(logits_out + (size_t)(step + 1) * 64, lg, 64 * sizeof(float));
        best = 0; for (int v = 1; lg && v < 64; v++) if (lg[v] > lg[best]) best = v;
        free(lg); tok = best;
    }
    if (ch) {
        if (caps) *caps = ch->g_capture;
        if (laps) *laps = ch->g_launch;
        if (eagers) *eagers = ch->g_eager;
        if (l_gr) for (int i = 0; i < 4; i++) l_gr[i] = t_layer_has_graph(ch, i);
        if (part_out && ch->dm_part) for (int i = 0; i < 4; i++) part_out[i] = ch->dm_part[i];
    }
    fake_wait_hook = NULL; t_srv_ch = NULL; t_srv_m = NULL;
    g_cuda_chain = 0;
    q38_model_free(&m);
    qt_shutdown();
    cc_shutdown();
    gguf_close(&g_q38gg); memset(&g_q38gg, 0, sizeof g_q38gg);
    g_q38_gguf = 0; g_q38_gguf_tier = 0;
    unsetenv("Q38_CC_GRAPH"); unsetenv("COLI_CUDA_CHAIN_PROMOTE");
    return ch != NULL;
}

static void t_gguf_dm_graph(void) {
    printf(" tiny block-typed qwen38 GGUF: dm_on + Q38_CC_GRAPH=1\n");
    const char *dir = "./q38_tiny_gguf_tmp";
    if (!t_make_tiny_gguf(dir)) { tk(0, "created tiny block-typed qwen38 GGUF fixture"); return; }
    fg_install();
    const int be0 = fake_block_elems;
    fake_block_elems = 32;
    fake_upload_at_hook = t_gguf_upload_at;

    /* 1. all experts on the card: in-frame routed experts captured & replayed, 100% parity with Q38_CC_GRAPH=0 */
    float lg0[7 * 64], lg1[7 * 64];
    int dm0 = 0, dm1 = 0, go0 = 0, go1 = 0, lgr[4] = {0};
    unsigned long long c1 = 0, r1 = 0, e1 = 0;
    unsigned long long rt0 = fg_route_rec, mo0 = fg_moe_rec, br0 = fg_breaks;
    int ok0 = t_gguf_dm_run(dir, 0, 0, 0, lg0, &dm0, &go0, NULL, NULL, NULL, NULL, NULL);
    int ok1 = t_gguf_dm_run(dir, 1, 0, 0, lg1, &dm1, &go1, lgr, &c1, &r1, &e1, NULL);
    tk(ok0 && ok1 && dm0 && dm1 && !go0 && go1, "GGUF dm_on active for both runs, graphs enabled only with Q38_CC_GRAPH=1");
    tk(c1 > 0 && r1 > 0 && fg_route_rec > rt0 && fg_moe_rec > mo0 && fg_breaks == br0,
       "in-frame router and block-typed routed experts captured and replayed without breaking");
    tk(lgr[0] && lgr[1] && lgr[2] && lgr[3], "all four layers (DeltaNet, QSA, PLE) captured segment graphs");
    tk(!memcmp(lg0, lg1, sizeof lg0), "Q38_CC_GRAPH=1 matches Q38_CC_GRAPH=0 bit-for-bit across prompt and decode");

    /* 2. graph_scratch invalidation: continuous move, mid-run move, and missing graph_scratch refusal */
    unsigned long long c2 = 0, r2 = 0;
    fg_scratch_moves = 1;
    int ok2 = t_gguf_dm_run(dir, 1, 0, 0, lg1, &dm1, &go1, NULL, &c2, &r2, NULL, NULL);
    fg_scratch_moves = 0;
    tk(ok2 && dm1 && go1 && c2 == 0 && r2 == 0 && !memcmp(lg0, lg1, sizeof lg0),
       "graph_scratch moving before every segment: 0 captures/replays, bit-for-bit parity");
    unsigned long long c3 = 0, r3 = 0, e3 = 0;
    int ok3 = t_gguf_dm_run(dir, 1, 0, 1, lg1, &dm1, &go1, NULL, &c3, &r3, &e3, NULL);
    tk(ok3 && dm1 && go1 && c3 > c1 && e3 > e1 && r3 < r1 && !memcmp(lg0, lg1, sizeof lg0),
       "graph_scratch moved mid-run: stale graphs invalidated and re-captured, bit-for-bit parity");
    fake_chain_table.graph_scratch = NULL;
    int go_no_sc = 1;
    int ok_no_sc = t_gguf_dm_run(dir, 1, 0, 0, lg1, &dm1, &go_no_sc, NULL, NULL, NULL, NULL, NULL);
    fake_chain_table.graph_scratch = fg_scratch_of;
    tk(ok_no_sc && dm1 && !go_no_sc && !memcmp(lg0, lg1, sizeof lg0),
       "missing graph_scratch on GGUF: Q38_CC_GRAPH=1 refused cleanly, bit-for-bit parity");

    /* 3. pend_v layer-after-hosted fallback: layer 1 hosted -> layers 1 and 2 op-by-op, layers 0 and 3 graphed */
    int part[4] = {0}, lgr_h[4] = {0};
    unsigned long long ch1 = 0, rh1 = 0, w0 = fg_waits;
    br0 = fg_breaks;
    int okh0 = t_gguf_dm_run(dir, 0, 1, 0, lg0, &dm0, &go0, NULL, NULL, NULL, NULL, NULL);
    int okh1 = t_gguf_dm_run(dir, 1, 1, 0, lg1, &dm1, &go1, lgr_h, &ch1, &rh1, NULL, part);
    tk(okh0 && okh1 && dm0 && dm1 && go1 && part[1] && !part[0] && !part[2] && !part[3],
       "layer 1 hosted (expert 3 on host), layers 0, 2, 3 all on card");
    tk(fg_waits > w0 && fg_breaks == br0 && ch1 > 0 && rh1 > 0,
       "host share served via wait_host, graphs captured and replayed without breaking");
    tk(lgr_h[0] && !lgr_h[1] && !lgr_h[2] && lgr_h[3],
       "pend_v fallback: layers 0 and 3 graphed, hosted layer 1 and layer-after-hosted 2 ran op-by-op");
    tk(!memcmp(lg0, lg1, sizeof lg0), "hosted + pend_v fallback matches Q38_CC_GRAPH=0 bit-for-bit");

    fake_upload_at_hook = NULL;
    fake_block_elems = be0;
    fg_uninstall();

    char p[256];
    snprintf(p, sizeof p, "%s/tiny.gguf", dir); remove(p);
    snprintf(p, sizeof p, "%s/gguf.json", dir); remove(p);
    snprintf(p, sizeof p, "%s/config.json", dir); remove(p);
#ifdef _WIN32
    _rmdir(dir);
#else
    rmdir(dir);
#endif
}

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

    {   /* Q38_CC_GRAPH=1: the decode steps' segments captured on their second sight, replayed after */
        fg_install();
        t_graph_switch(argv, "Q38_CC_GRAPH=1 on the chain (a decode step's segments replayed as graphs)", &fg_replays);
        printf("       %llu segment graphs captured, %llu replays\n", fg_captures, fg_replays);
        tk(fg_captures > 0 && fg_breaks == 0 && fg_refused == 0, "segments captured, no capture broke");
        unsigned long long r0 = fg_replays;
        fg_break_at = 3; fg_recorded = 0;
        t_graph_switch(argv, "Q38_CC_GRAPH=1, a capture an op breaks: the ops one by one from there", &fg_breaks);
        tk(fg_replays == r0, "nothing replayed once the switch went off");
        fg_break_at = 0;
        fg_refuse = 1;
        t_graph_switch(argv, "Q38_CC_GRAPH=1, a capture the stream refuses: the ops one by one, the chain stays on", &fg_refused);
        tk(fg_replays == r0, "nothing replayed once the switch went off");
        fg_refuse = 0;
        {   /* a capture that breaks after others were made: the switch off frees every graph it holds */
            unsigned long long p0 = fg_pushed;
            t_graph_switch(argv, "Q38_CC_GRAPH=1 (counting the ops a run records)", &fg_replays);
            fg_break_at = (int)((fg_pushed - p0) / 2); fg_recorded = 0;
            unsigned long long c0 = fg_captures, f0 = fg_frees, b0 = fg_breaks;
            t_graph_switch(argv, "Q38_CC_GRAPH=1, a capture breaks halfway through the run", &fg_breaks);
            tk(fg_breaks == b0 + 1 && fg_captures > c0 && fg_frees - f0 == fg_captures - c0, "the graphs made before the break all freed");
            fg_break_at = 0;
        }
        {   /* the scratch a GGUF's block matrices quantize into moved (a prompt's rows grow it): a graph that
             * holds the old address is never replayed; here it moves before every segment */
            unsigned long long c0 = fg_captures, r0 = fg_replays;
            fg_scratch_moves = 1;
            t_graph_switch(argv, "Q38_CC_GRAPH=1, the block matrices' scratch moved before every segment", NULL);
            tk(fg_captures == c0 && fg_replays == r0, "every segment op by op: no graph captured or replayed");
            fg_scratch_moves = 0;
        }
        fg_uninstall();
    }

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
    {   /* verify rows (n > 1, the DeltaNet copies) replayed; a rejected draft's rollback copies into the live state */
        static const char *const force[] = {"reject", "mixed", "cycle"};
        fg_install();
        for (int k = 0; k < 3; k++) {
            char title[96];
            snprintf(title, sizeof title, "Q38_CC_GRAPH=1 with MTP verify rows, Q38_MTP_FORCE=%s", force[k]);
            setenv("Q38_MTP_FORCE", force[k], 1);
            fake_dn_snapshots = 0;
            t_graph_switch(argv, title, &fg_replays);
            tk(fake_dn_snapshots > 0, "the verifies wrote DeltaNet copies");
        }
        setenv("Q38_MTP_FORCE", "reject", 1);
        fg_uninstall();
    }
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
        {   /* a verify's table of copies (q38cc_snap_tab) over chunks of 2 rows (COLI_CUDA_CHAIN_ROWS
             * under the verify's 4): the second chunk's entries (slots 2, 3) beside the first's, not over
             * them, so a replayed graph of either chunk, which does not rewrite the table, reads its own */
            static Q38CChain tc; static CcBuf *rs[4][3], *gs[4][3];
            const size_t sz = fake_chain_table.size; fake_chain_table.size = sizeof fake_chain_table;   /* dev_addr offered */
            int ok = 1, own = 1, ro0 = 0, rg0 = 0, ro1 = 0, rg1 = 0;
            for (int s = 0; s < 4; s++) {
                tc.rec_snap[s] = rs[s]; tc.ring_snap[s] = gs[s];
                for (int i = 0; i < 3; i++) ok = ok && (rs[s][i] = cc_buf(16, CC_DEV)) && (gs[s][i] = cc_buf(16, CC_DEV));
            }
            ok = ok && q38cc_snap_tab(&tc, &rm, 1, 0, 2, &ro0, &rg0) && q38cc_snap_tab(&tc, &rm, 1, 2, 2, &ro1, &rg1);
            for (int r = 0; r < 2 && ok; r++)
                own = own && fcc_snap_at(tc.snap_tab, ro0, r) == rs[r][1]->d && fcc_snap_at(tc.snap_tab, rg0, r) == gs[r][1]->d &&
                      fcc_snap_at(tc.snap_tab, ro1, r) == rs[2 + r][1]->d && fcc_snap_at(tc.snap_tab, rg1, r) == gs[2 + r][1]->d;
            tk(ok && own, "a verify in two chunks: each chunk's table entries its own copies");
            for (int s = 0; s < 4; s++) for (int i = 0; i < 3; i++) { cc_free(rs[s][i]); cc_free(gs[s][i]); }
            cc_free(tc.snap_tab);
            fake_chain_table.size = sz;
        }
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
                q38cc_srv_post(sv, i, n, ++v, tg, td, ids[i], w[i], res + i * NE, 0);
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
            q38cc_srv_post(sv, 0, 1, ++v, tg, td, ids[0], w[0], res + 0 * NE, 0);
            q38cc_srv_post(sv, 1, 1, ++v, tg, td, ids[1], w[1], res + 1 * NE, 0);
            tk(q38cc_srv_close(sv, 0, v) && hfl == v, "a failed frame: the jobs skipped, the host's word past them");
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            q38cc_srv_post(sv, 2, 1, ++v, tg, td, ids[2], w[2], res + 2 * NE, 0);
            dfl = v;
            for (long spin = 0; hfl < v && spin < 2000000000L; spin++) q38cc_relax();
            tk(q38cc_srv_close(sv, 1, v) && hfl == v, "and the next forward served");
            /* a job the device does not wait for (q38cc_dm_skips): without a pooled slot (experts 0
             * and 1 are the card's, -1 off the table) it writes no rows and no word, a later job's
             * rows may be there; with one it is served as before */
            int idz[K]; float wz[K];
            for (int a = 0; a < K; a++) { idz[a] = a % 3 == 2 ? -1 : a % 2; wz[a] = 0.5f; }
            for (int k = 0; k < R * H; k++) out[k] = 7.f;
            const unsigned long long sk0 = sv->skips, jb0 = sv->jobs; const unsigned h0 = hfl;
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            q38cc_srv_post(sv, 0, 1, ++v, tg, td, idz, wz, res + 0 * NE, 1);
            dfl = v;
            int cl3 = q38cc_srv_close(sv, 1, v), kept = 1;
            for (int k = 0; k < R * H; k++) if (out[k] != 7.f) kept = 0;
            tk(cl3 && hfl == h0 && kept && sv->skips == sk0 + 1 && sv->jobs == jb0, "a job without a pooled slot the device went on from: no rows, no word");
            int np2 = 0;
            for (int a = 0; a < K; a++) if (ids[2][a] >= 0 && res[2 * NE + ids[2][a]] >= 2) np2++;
            q38cc_srv_open(sv, &dfl, &hfl, x, out);
            q38cc_srv_post(sv, 2, 1, ++v, tg, td, ids[2], w[2], res + 2 * NE, 1);
            dfl = v;
            for (long spin = 0; hfl < v && spin < 2000000000L; spin++) q38cc_relax();
            tk(np2 > 0 && q38cc_srv_close(sv, 1, v) && hfl == v && sv->skips == sk0 + 1 && sv->jobs == jb0 + 1,
               "one with pooled slots: served, its word written");
            q38cc_srv_stop(sv);
        }
        free(pool); free(x); free(out); free(want);
    }

    printf(" a decode step's pooled experts: the CPU's or the bus's, measured (q38cc_pd_*)\n");
    {   /* steps of one row at a time each path takes: the paths by turns in blocks of Q38CC_PD_BLK,
         * the faster once both have Q38CC_PD_N, the bus only past the margin, a new round after
         * Q38CC_PD_EVERY steps; q38cc_pc_step follows the forward's path */
        static const double ms[4][2] = {{10.0, 11.0}, {11.0, 10.0}, {10.0, 9.9}, {10.0, 9.0}};
        static const int want[4] = {0, 1, 0, 1};
        int32_t pof = 0; CcBuf *tabd = (CcBuf *)&pof;
        for (int c = 0; c < 4; c++) {
            Q38CChain *ch = (Q38CChain *)calloc(1, sizeof *ch);
            if (!ch) { tk(0, "the chain's struct"); break; }
            ch->pool_of = &pof; ch->pool_cpu = 1; ch->dm_tabd = tabd;
            int turns = 1, steps = 0, follows = 1;
            while (!ch->pd_dec && steps < 1000) {
                const int k = steps, pd = q38cc_pd_begin(ch, 1);
                if (pd != (k / Q38CC_PD_BLK) % 2) turns = 0;
                if (q38cc_pc_step(ch, 1) != !pd) follows = 0;
                q38cc_pd_end(ch, 1, pd, ms[c][pd] / 1e3);
                steps++;
            }
            tk(turns && follows, "the paths by turns, the decode table only on the CPU's");
            tk(ch->pd_dec && ch->pd_bus == want[c] && steps == 2 * Q38CC_PD_N,   /* Q38CC_PD_N a multiple of Q38CC_PD_BLK: the bus's last at step 2 N */
               c == 0 ? "the CPU faster: the CPU's" : c == 1 ? "the bus faster: the bus's" : c == 2 ? "the bus 1 % faster: still the CPU's" : "the bus 10 % faster: the bus's");
            if (c == 3) {
                int same = 1;
                while (ch->pd_k < ch->pd_next) { const int pd = q38cc_pd_begin(ch, 1); if (pd != 1) same = 0; q38cc_pd_end(ch, 1, pd, 0.01); }
                const int pd = q38cc_pd_begin(ch, 1);
                tk(same && !ch->pd_dec && pd == 0 && ch->pd_n[0][1] == 0, "the decided path until the next round, which measures again");
                q38cc_pd_end(ch, 1, pd, 0.01);
                tk(q38cc_pd_begin(ch, Q38CC_DM_ROWS + 1) < 0 && !ch->pd_now, "a prompt's rows: not measured, the CPU's");
                ch->pd_mode = 1;
                tk(q38cc_pd_begin(ch, 1) < 0, "COLI_CUDA_CHAIN_POOL_READ=0: not measured");
            }
            free(ch);
        }
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

    printf(" a prompt chunk's staging: the layers it runs in the frame\n");
    {   /* four layers of four experts, experts 0 to 2 of each in the host pool, layers 1 and 3 able
         * to miss (dm_part: a chunk takes the host's route there, which waits for no copy and starts
         * none): their pooled experts get no places and no copies, so the copies go from one framed
         * layer to the next (#2764) */
        enum { L = 4, E = 4, H = 64, IN_ = 64 };   /* IN_: windows.h defines IN */
        const size_t sz0 = fake_chain_table.size;
        fake_chain_table.size = sizeof fake_chain_table;   /* the staging's device addresses */
        fcc_init(0);
        Model mm; memset(&mm, 0, sizeof mm);
        mm.c.layers = L; mm.c.experts = E; mm.c.hidden = H; mm.c.inter = IN_;
        int gtg[L], gtd[L];
        for (int i = 0; i < L; i++) gtg[i] = gtd[i] = GQ_Q8_0;
        const int on0 = G.on, nl0 = G.nl, wf0 = G.wfmt; int *gtg0 = G_gtg, *gtd0 = G_gtd;
        G.on = 1; G.nl = L; G.wfmt = QT_GGUF; G_gtg = gtg; G_gtd = gtd;
        Q38CChain cc; memset(&cc, 0, sizeof cc);
        int64_t gb = 0, db = 0;
        const int typed = q38cc_pool_types(&mm, 0, &gb, &db);
        const size_t st = 2 * Q38CC_PAD(gb) + Q38CC_PAD(db);
        CcBuf *pool_b[1] = {cc_buf((size_t)L * E * st, CC_DOWN)};
        cc.pool_buf = pool_b; cc.pool_nb = 1; cc.pool_per = L * E; cc.pool_stride = st;
        cc.pool_of = (int32_t *)malloc((size_t)L * E * sizeof(int32_t)); cc.dm_host = (CcExpert *)calloc((size_t)L * E, sizeof(CcExpert));
        cc.dm_res = (uint8_t *)calloc((size_t)L * E, 1); cc.dm_part = (int *)calloc(L, sizeof(int));
        for (int j = 0; j < L * E; j++) { const int pooled = j % E < 3; cc.pool_of[j] = pooled ? j : -1; cc.dm_res[j] = pooled ? 2 : 1; }
        cc.dm_part[1] = cc.dm_part[3] = 1;
        cc.dm_stage = cc_buf(3 * st, CC_DEV); cc.dm_stage_size = 3 * st; cc.dm_stage_cap = 3;
        const int ok = typed && pool_b[0] && cc.dm_stage && q38cc_stage_table(&cc, &mm);
        tk(ok && cc.dm_frn[0] > 0 && cc.dm_frn[2] > 0, "the framed layers' pooled experts have their copies");
        tk(ok && cc.dm_frn[1] == 0 && cc.dm_frn[3] == 0, "a layer that can miss: no places, no copies");
        int kept = ok;
        for (int i = 1; i < L && kept; i += 2)
            for (int e = 0; e < E; e++) kept = kept && !memcmp(&cc.dm_hostb[i * E + e], &cc.dm_host[i * E + e], sizeof(CcExpert));
        tk(kept, "its entries in the chunk's table the same as the decode's");
        free(cc.pool_of); free(cc.dm_host); free(cc.dm_res); free(cc.dm_part);
        free(cc.dm_hostb); free(cc.dm_fro); free(cc.dm_frn); free(cc.dm_fr); free(cc.dm_iv);
        cc_free(cc.dm_tabb); cc_free(cc.dm_stage); cc_free(pool_b[0]);
        G.on = on0; G.nl = nl0; G.wfmt = wf0; G_gtg = gtg0; G_gtd = gtd0;
        fake_chain_table.size = sz0;
    }

    t_gguf_dm_graph();

    if (t_fails) { printf("test_qwen38_cuda_chain: %d failure(s)\n", t_fails); return 1; }
    printf("OK test_qwen38_cuda_chain: Qwen3.8's chain on the card gives the oracle's tokens\n");
    return 0;
}
