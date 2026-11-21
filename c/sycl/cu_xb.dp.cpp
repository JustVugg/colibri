/* cu_xb.cu -- the routed-expert tier's device side on CUDA (cu_tier_ops.h).
 *
 * The CUDA twin of what vk_tier.c asks of backend_vulkan.c: the tier's tensors in pools of
 * device memory (vk_alloc.h's sub-allocator over cudaMalloc blocks), a host pool of mapped
 * page-locked RAM the card reads over the bus, and the expert batch: a layer step's
 * resident experts (gate and up, the activation, down) on a stream of its own, issued and
 * joined. The tensors keep backend_vulkan.c's format numbers and layouts (each row padded
 * to whole words, the scales after): a GGUF's blocks go through cuda_bd.cuh's GEMVs, the
 * same decoders as the dense chain's, and colibri's formats through cu_gemv_k below, which
 * decodes as vk_tier.c's float reference (dev_dequant) does. The activations are
 * expert_act.comp's and expert_act_v4.comp's.
 *
 * Not here yet, declined as a Vulkan device without them declines: the dense chain's
 * expert table (xb_expert_entry 0), a second device, streaming's sub-batches and the
 * grouped step (gemm_rows 0), the low-bit formats 200-203 (no device form).
 *
 * On SYCL (c/sycl/MIGRATION.md): the device is coli_cuda_init's first (coli_sycl.h: its context,
 * its profile), the two streams are two in-order queues of this unit's own on that context (with
 * profiling, for the batch's elapsed time), the pools are USM device memory of that context, the
 * host pool is off (no mapped host memory the device reads: C.mapped 0), and cu_gemv_k runs at the
 * profile's sub-group through the SG dispatch. */
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>   /* dim3 */
#include "backend_cuda.h"
#define COLI_CUDA_CHAIN_NO_WRAPPERS
#include "backend_gpu_compat.h"
#include "coli_sycl.h"     /* the tier's device: coli_cuda_init's, through the profile module */
#include "backend_sycl.h"  /* the error protocol, the shuffles, the SG launch */
#include "cuda_bd.dp.hpp"
#include "cu_tier_ops.h"
#include "vk_alloc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <vector>

/* ---------------------------------------------------------------- formats and sizes */
/* backend_vulkan.c's rules, so the tier's conversions and budgets are the same */
static int cu_bd_fmt(int fmt) { return bd_fmt_ok(fmt); }
static size_t cu_row_bytes(int fmt, int I) {
    if (cu_bd_fmt(fmt)) return (size_t)I / BD_BLK[fmt - 100] * BD_BYTES[fmt - 100];
    return fmt == 1 || fmt == 12 || fmt == 13 ? (size_t)I
         : fmt == 5  ? ((size_t)I + 63) / 64 * 24
         : fmt == 10 ? (size_t)I * 4
         : fmt == 11 ? (size_t)I * 2
         : (size_t)(I + 1) / 2;
}
static int cu_row_words(int fmt, int I) { return (int)((cu_row_bytes(fmt, I) + 3) / 4); }
static int cu_grouped(int fmt) { return fmt == 4 || fmt == 7 || fmt == 12 || fmt == 13; }
static size_t cu_scale_floats(int fmt, int I, int O, int gs) {
    if (fmt == 10 || fmt == 11 || cu_bd_fmt(fmt)) return 1;
    if (fmt == 5) return (size_t)O * (((size_t)I + 63) / 64);
    if (cu_grouped(fmt)) return (size_t)O * (((size_t)I + gs - 1) / gs);
    return (size_t)O;
}
/* a format this backend computes: colibri's (grouped ones with a group of whole words of
 * codes) and the block types cuda_bd.cuh decodes */
static int cu_fmt_ok(int fmt, int I, int gs) {
    if (cu_bd_fmt(fmt)) return I % 32 == 0 && I % BD_BLK[fmt - 100] == 0;
    switch (fmt) {
    case 1: case 2: case 10: case 11: return 1;
    case 5: return I % 64 == 0;
    case 4: case 7: case 12: case 13: return gs >= 8 && gs % 8 == 0;
    default: return 0;
    }
}

/* ---------------------------------------------------------------- the device */
#define CU_POOLS 6        /* backend_vulkan.h's numbers: 1 the tier, 4 its extra layers, 5 the host pool */
#define CU_ALIGN 256
#define CU_BLOCK ((size_t)256 << 20)

struct CuPool {
    VkaPool p;
    std::vector<void *> mem;           /* by block index: device memory, or the host pool's mapping */
    int host, on;
    size_t tensors, payload;
};
struct ColiVkTensor {
    int fmt, I, O, gs, rowWords, pool;
    size_t wbytes, sbytes, wcap, scap;
    VkaRange wr, sr;
    uint8_t *w; float *s;              /* what the kernels read (the host pool's: device pointers of the mapping) */
    uint8_t *hw; float *hs;            /* the host pool's rows and scales as the CPU reads them */
    uint8_t *img;                      /* a device pool tensor's host image until it is committed */
};
struct ColiVkExpert { ColiVkTensor *g, *u, *d; };

static struct {
    std::mutex mx;                     /* the pools, the pending frees */
    int tried, ready, dev, integrated, mapped;
    char name[256];
    size_t total, smem_max;
    int sg;                            /* the sub-group cu_gemv_k and the block GEMVs run at (the profile's) */
    sycl::queue *xs, *cs;              /* the batch's queue, the copies': in order, on the device's context */
    sycl::event *e0, *e1;              /* the batch's bounds (barriers on xs), its elapsed time */
    float *e2m1;                       /* the e2m1 magnitudes (fmt 7's nibbles), a device array: CUDA's __constant__ cu_e2m1 */
    bd_u32 *tab;                       /* blockdec_host.h's table */
    CuPool pool[CU_POOLS];
    /* the expert batch */
    int D, I, act, xb;
    float limit, a, b;
    size_t cap;                        /* rows the scratch holds */
    float *dX, *dG, *dU, *dY, *dW, *hX, *hY, *hW;
    int inflight, failed, nrows;
    std::vector<ColiVkTensor *> pending;   /* freed while a batch was in flight */
    ColiVkXbStats st;
} C;

/* cudaSetDevice: nothing to select, the queues name the device */
static int cu_set(void) { return C.ready; }
/* the queue allocations and frees go through (its context is the device's) */
static sycl::queue &dq(void) { return *C.xs; }
static inline void sync_barrier(sycl::event *e, sycl::queue *q) { *e = q->ext_oneapi_submit_barrier(); }
/* this unit's books for coli_sycl_mem_info: the device pools' blocks (the host pool is off) */
static size_t xb_booked_bytes(int device) {
    if (!C.ready || device != C.dev) return 0;
    size_t b = 0;
    for (int k = 0; k < CU_POOLS; k++) if (!C.pool[k].host) b += (size_t)C.pool[k].p.total;
    return b;
}
static int cu_open(void) try {
    if (C.tried) return C.ready;
    C.tried = 1;
    if (coli_cuda_device_count() < 1) return 0;
    C.dev = coli_cuda_device_at(0);
    if (!coli_sycl_is_open(C.dev)) return 0;
    const coli_sycl_profile &pr = coli_sycl_profile_for(C.dev);
    snprintf(C.name, sizeof C.name, "%s", pr.device_name);
    C.integrated = pr.is_integrated;
    C.mapped = 0;   /* no host memory the device reads in place (cudaHostAllocMapped): the host pool is off */
    C.total = pr.global_mem_bytes;
    C.smem_max = pr.slm_bytes ? pr.slm_bytes : 48 * 1024;   /* the device's SLM: no opt-in past it */
    C.sg = pr.sg_default == 32 ? 32 : 16;
    coli_sycl_ledger_add(xb_booked_bytes);
    const sycl::property_list props = {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
    if (COLI_SYCL_TRY(C.xs = new sycl::queue(coli_sycl_context(C.dev), coli_sycl_device(C.dev), props)) != 0 ||
        COLI_SYCL_TRY(C.cs = new sycl::queue(coli_sycl_context(C.dev), coli_sycl_device(C.dev), props)) != 0) return 0;
    C.e0 = new sycl::event(); C.e1 = new sycl::event();
    std::vector<bd_u32> t(BDT_WORDS);
    bdtab_fill(t.data(), NULL);
    static const float e2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    if (COLI_SYCL_ALLOC(C.tab, (bd_u32 *)sycl::malloc_device(
                (size_t)BDT_WORDS * 4, dq()), (size_t)BDT_WORDS * 4) != 0 ||
        COLI_SYCL_TRY(dq()
                             .memcpy(C.tab, t.data(), (size_t)BDT_WORDS * 4)
                             .wait()) != 0 ||
        COLI_SYCL_ALLOC(C.e2m1, sycl::malloc_device<float>(8, dq()), 8) != 0 ||
        COLI_SYCL_TRY(dq().memcpy(C.e2m1, e2m1, sizeof e2m1).wait()) != 0) return 0;
    for (int k = 0; k < CU_POOLS; k++) { vka_pool_init(&C.pool[k].p, CU_BLOCK, 0); C.pool[k].host = k == 5; }
    C.pool[1].on = C.pool[4].on = 1;
    C.pool[5].on = C.mapped;
    C.ready = 1;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_open"); }

static int cu_available(void) { return cu_open(); }
static const char *cu_device_name(void) { return C.ready ? C.name : "no CUDA device"; }
static int cu_device_shares_ram(void) { return C.ready && C.integrated; }
static size_t cu_device_local_bytes(void) { return C.ready ? C.total : 0; }
/* the device's memory as the runtime reports it (cudaMemGetInfo): coli_sycl_mem_info's figure,
 * the device's own free count where it has the query (ext_intel_free_memory, ZES_ENABLE_SYSMAN=1),
 * else the total less the backend's books, this unit's pools among them (xb_booked_bytes) */
static int cu_mem_info(size_t *fr, size_t *tot) {
    if (!C.ready) return 0;
    return coli_sycl_mem_info(C.dev, fr, tot, nullptr);
}
/* the device's memory in GB as VK_EXT_memory_budget reports it: what is used, what there is */
static int cu_mem_budget(double *used_gb, double *budget_gb) try {
    size_t fr = 0, tot = 0;
    if (!cu_set() || !cu_mem_info(&fr, &tot)) return 0;
    *used_gb = (double)(tot - fr) / 1e9; *budget_gb = (double)tot / 1e9;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_mem_budget"); }
static int cu_dev_info(int dev, ColiVkDevInfo *out) try {
    if (dev != 0 || !C.ready) return 0;
    size_t fr = 0, tot = 0;
    memset(out, 0, sizeof *out);
    out->name = C.name; out->integrated = C.integrated; out->shares_ram = C.integrated;
    out->local_bytes = C.total; out->buf_align = CU_ALIGN;
    if (cu_set() && cu_mem_info(&fr, &tot)) {
        out->has_budget = 1; out->free_bytes = fr;
    }
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_dev_info"); }
static int cu_dev2_open_env(void) { return 0; }

/* ---------------------------------------------------------------- the pools */
static size_t cu_block_bytes(size_t def) { return def; }
static size_t cu_block_bytes_dev(int dev, size_t def) { (void)dev; return def; }
static size_t cu_buffer_alignment(void) { return CU_ALIGN; }
static int pool_of(int dev) { return dev == 0 ? 1 : dev == 2 ? 5 : -1; }

/* a range of `bytes` in pool k, a block added when none holds it. C.mx held. */
static int pool_take(int k, size_t bytes, VkaRange *r) try {
    CuPool *P = &C.pool[k];
    if (vka_alloc(&P->p, bytes, CU_ALIGN, r)) return 1;
    /* the usual size, or what is left under the limit when that is smaller but still holds
     * this request (a budget is rarely a block multiple): backend_vulkan.c's rule */
    uint64_t cap = vka_block_size_for(&P->p, bytes);
    if (P->p.limit && P->p.total + cap > P->p.limit && P->p.limit > P->p.total && P->p.limit - P->p.total >= bytes + CU_ALIGN)
        cap = (P->p.limit - P->p.total) & ~(uint64_t)4095;
    if (!vka_pool_can_grow(&P->p, cap) || cap < bytes) { P->p.refusals++; return 0; }
    void *m = NULL;
    if (P->host ? COLI_SYCL_ALLOC(m, (void *)sycl::malloc_host(
                                       cap, dq()), cap) != 0
                : COLI_SYCL_ALLOC(m, (void *)sycl::malloc_device(
                                       cap, dq()), cap) != 0) {
        P->p.refusals++;
        return 0;
    }
    int b = vka_pool_add_block(&P->p, cap, m);
    if (b < 0) {
        if (P->host) sycl::free(m, dq());
        else sycl::free(m, dq());
        return 0;
    }
    if ((int)P->mem.size() <= b) P->mem.resize(b + 1, NULL);
    P->mem[b] = m;
    return vka_alloc(&P->p, bytes, CU_ALIGN, r);
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "pool_take"); }
static void pool_give(int k, VkaRange r) {
    CuPool *P = &C.pool[k];
    if (vka_free(&P->p, r)) {
        void *m = P->mem[r.block];
        if (P->host) sycl::free(m, dq());
            else sycl::free(m, dq());
        P->mem[r.block] = NULL;
        vka_pool_drop_block(&P->p, r.block);
    }
}
static uint8_t *pool_at(int k, VkaRange r) { return (uint8_t *)C.pool[k].mem[r.block] + r.off; }
/* the device's pointer to a host pool byte: USM host memory has one address on both sides */
static uint8_t *host_dev_ptr(uint8_t *h) { return h; }

static void cu_pool_stats(int pool, ColiVkPoolStats *st) {
    memset(st, 0, sizeof *st);
    if (pool < 0 || pool >= CU_POOLS) return;
    std::lock_guard<std::mutex> lk(C.mx);
    CuPool *P = &C.pool[pool];
    VkaStats s; vka_stats(&P->p, &s);
    st->blocks = s.blocks; st->live = s.live; st->total = s.total; st->used = s.used; st->free = s.free;
    st->largest_free = s.largest_free; st->peak_used = s.peak_used; st->limit = P->p.limit;
    st->tensors = P->tensors; st->payload = P->payload;
    st->allocs = P->p.allocs; st->frees = P->p.frees; st->refusals = P->p.refusals; st->frag = s.frag;
}
static void cu_tier_pool_limit(size_t bytes) { std::lock_guard<std::mutex> lk(C.mx); C.pool[1].p.limit = bytes; }
static void cu_tier_pool_limit_dev(int dev, size_t bytes) {
    int k = pool_of(dev);
    if (k < 0) return;
    std::lock_guard<std::mutex> lk(C.mx);
    C.pool[k].p.limit = bytes;
}
static void cu_tier_extra_pool_limit(size_t bytes) { std::lock_guard<std::mutex> lk(C.mx); C.pool[4].p.limit = bytes; }
static int cu_tier_host_pool(void) { return C.ready && C.mapped; }

/* ---------------------------------------------------------------- the tensors */
static int cu_block_elems(int fmt) { return C.ready && cu_bd_fmt(fmt) ? (int)BD_BLK[fmt - 100] : 0; }
static size_t cu_tensor_row_bytes(int fmt, int I) { return cu_row_bytes(fmt, I); }
static size_t cu_tensor_scale_count(int fmt, int I, int O, int gs) { return cu_scale_floats(fmt, I, O, gs); }
static size_t cu_tensor_payload(int fmt, int I, int O, int gs) {
    if (cu_grouped(fmt)) { if (gs < 1) return 0; } else gs = 0;
    return (size_t)cu_row_words(fmt, I) * 4 * (size_t)O + cu_scale_floats(fmt, I, O, gs) * sizeof(float);
}
static size_t cu_tensor_alloc_bytes(int fmt, int I, int O, int gs, size_t *scale_bytes) {
    if (cu_grouped(fmt)) { if (gs < 1) return 0; } else gs = 0;
    if (scale_bytes) *scale_bytes = cu_scale_floats(fmt, I, O, gs) * sizeof(float);
    return (size_t)cu_row_words(fmt, I) * 4 * (size_t)O;
}

/* the rows and scales to fill: the host pool's mapping, else a host image (committed later) */
static int hand_out(ColiVkTensor *t, uint8_t **rows, size_t *stride, float **scales) {
    if (C.pool[t->pool].host) {
        memset(t->hw, 0, t->wbytes);
        *rows = t->hw; *scales = t->hs;
    } else {
        free(t->img);
        t->img = (uint8_t *)calloc(1, t->wbytes + t->sbytes);
        if (!t->img) return 0;
        *rows = t->img; *scales = (float *)(t->img + t->wbytes);
    }
    *stride = (size_t)t->rowWords * 4;
    return 1;
}
/* a tensor of fmt/I/O/gs in pool k with ranges of at least pad_w and pad_s bytes */
static ColiVkTensor *tensor_new(int k, int fmt, int I, int O, int gs, size_t pad_w, size_t pad_s) {
    if (k < 0 || !C.pool[k].on || !cu_fmt_ok(fmt, I, gs) || O < 1) return NULL;
    if (!cu_grouped(fmt)) gs = 0;
    ColiVkTensor *t = (ColiVkTensor *)calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fmt = fmt; t->I = I; t->O = O; t->gs = gs; t->pool = k;
    t->rowWords = cu_row_words(fmt, I);
    t->wbytes = (size_t)t->rowWords * 4 * (size_t)O;
    t->sbytes = cu_scale_floats(fmt, I, O, gs) * sizeof(float);
    t->wcap = t->wbytes > pad_w ? t->wbytes : pad_w;
    t->scap = t->sbytes > pad_s ? t->sbytes : pad_s;
    std::lock_guard<std::mutex> lk(C.mx);
    if (!pool_take(k, t->wcap, &t->wr)) { free(t); return NULL; }
    if (!pool_take(k, t->scap, &t->sr)) { pool_give(k, t->wr); free(t); return NULL; }
    uint8_t *w = pool_at(k, t->wr), *s = pool_at(k, t->sr);
    if (C.pool[k].host) {
        t->hw = w; t->hs = (float *)s;
        t->w = host_dev_ptr(w); t->s = (float *)host_dev_ptr(s);
        if (!t->w || !t->s) { pool_give(k, t->wr); pool_give(k, t->sr); free(t); return NULL; }
    } else { t->w = w; t->s = (float *)s; }
    C.pool[k].tensors++; C.pool[k].payload += t->wbytes + t->sbytes;
    return t;
}
static void tensor_release(ColiVkTensor *t) {   /* C.mx held */
    pool_give(t->pool, t->wr); pool_give(t->pool, t->sr);
    C.pool[t->pool].tensors--; C.pool[t->pool].payload -= t->wbytes + t->sbytes;
    free(t->img); free(t);
}
static int cu_tier_tensor_pad(int dev, ColiVkTensor **t, int fmt, int I, int O, int gs, size_t pad_w, size_t pad_s,
                              uint8_t **rows, size_t *stride, float **scales) {
    if (!cu_set()) return 0;
    ColiVkTensor *n = tensor_new(pool_of(dev), fmt, I, O, gs, pad_w, pad_s);
    if (!n) return 0;
    if (!hand_out(n, rows, stride, scales)) { std::lock_guard<std::mutex> lk(C.mx); tensor_release(n); return 0; }
    *t = n;
    return 1;
}
static int cu_tier_tensor_extra(ColiVkTensor **t, int fmt, int I, int O, int gs, uint8_t **rows, size_t *stride, float **scales) {
    if (!cu_set()) return 0;
    ColiVkTensor *n = tensor_new(4, fmt, I, O, gs, 0, 0);
    if (!n) return 0;
    if (!hand_out(n, rows, stride, scales)) { std::lock_guard<std::mutex> lk(C.mx); tensor_release(n); return 0; }
    *t = n;
    return 1;
}
static int cu_tier_tensor_like(int dev, ColiVkTensor **t, const ColiVkTensor *src, size_t pad_w, size_t pad_s) {
    if (!cu_set() || !src) return 0;
    ColiVkTensor *n = tensor_new(pool_of(dev), src->fmt, src->I, src->O, src->gs, pad_w, pad_s);
    if (!n) return 0;
    *t = n;
    return 1;
}
/* the host images up, all n before it returns */
static int cu_tensor_commit(ColiVkTensor *const *t, int n) try {
    if (!cu_set()) return 0;
    int ok = 1;
    for (int i = 0; i < n; i++) {
        ColiVkTensor *x = t[i];
        if (!x || !x->img) continue;
        ok &= COLI_SYCL_TRY(C.cs->memcpy(x->w, x->img, x->wbytes)) == 0 &&
              COLI_SYCL_TRY(
                  C.cs->memcpy(x->s, x->img + x->wbytes, x->sbytes)) == 0;
    }
    ok &= COLI_SYCL_TRY(C.cs->wait()) == 0;
    for (int i = 0; i < n; i++) if (t[i]) { free(t[i]->img); t[i]->img = NULL; }
    return ok;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_tensor_commit"); }
static int cu_tensor_refill(ColiVkTensor *t, uint8_t **rows, size_t *stride, float **scales) {
    return t && cu_set() && hand_out(t, rows, stride, scales);
}
static int cu_tensor_refill_as(ColiVkTensor *t, int fmt, int I, int O, int gs, uint8_t **rows, size_t *stride, float **scales) {
    if (!t || !cu_set() || !cu_fmt_ok(fmt, I, gs)) return 0;
    if (!cu_grouped(fmt)) gs = 0;
    int rw = cu_row_words(fmt, I);
    size_t wb = (size_t)rw * 4 * (size_t)O, sb = cu_scale_floats(fmt, I, O, gs) * sizeof(float);
    if (wb > t->wcap || sb > t->scap) return 0;
    {
        std::lock_guard<std::mutex> lk(C.mx);
        C.pool[t->pool].payload += wb + sb - t->wbytes - t->sbytes;
    }
    t->fmt = fmt; t->I = I; t->O = O; t->gs = gs; t->rowWords = rw; t->wbytes = wb; t->sbytes = sb;
    return hand_out(t, rows, stride, scales);
}
/* src (the host pool) into dst (device memory) by the copy engine, all n before it returns */
static int cu_tensor_copy(ColiVkTensor *const *dst, ColiVkTensor *const *src,
                          int n) try {
    if (!cu_set()) return 0;
    int ok = 1;
    for (int i = 0; i < n && ok; i++) {
        ColiVkTensor *d = dst[i]; const ColiVkTensor *s = src[i];
        if (!d || !s || d->wcap < s->wbytes || d->scap < s->sbytes) { ok = 0; break; }
        d->fmt = s->fmt; d->I = s->I; d->O = s->O; d->gs = s->gs; d->rowWords = s->rowWords;
        d->wbytes = s->wbytes; d->sbytes = s->sbytes;
        ok = COLI_SYCL_TRY(C.cs->memcpy(d->w, s->w, s->wbytes)) == 0 &&
             COLI_SYCL_TRY(C.cs->memcpy(d->s, s->s, s->sbytes)) == 0;
    }
    ok &= COLI_SYCL_TRY(C.cs->wait()) == 0;
    return ok;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_tensor_copy"); }
/* freed now, or when the batch in flight is joined */
static void cu_tensor_free(ColiVkTensor *t) {
    if (!t) return;
    std::lock_guard<std::mutex> lk(C.mx);
    if (C.inflight) { C.pending.push_back(t); return; }
    tensor_release(t);
}
static size_t cu_tensor_bytes(const ColiVkTensor *t) { return t ? t->wcap + t->scap : 0; }
static int cu_tensor_info(const ColiVkTensor *t, ColiVkTensorInfo *out) {
    if (!t) return 0;
    out->wbuf = t->w; out->sbuf = t->s; out->fmt = t->fmt; out->I = t->I; out->O = t->O;
    out->rowWords = t->rowWords; out->gs = t->gs;
    return 1;
}
static const void *cu_tensor_host_rows(const ColiVkTensor *t, const void **scales) {
    if (!t || !C.pool[t->pool].host) return NULL;
    if (scales) *scales = cu_bd_fmt(t->fmt) || t->fmt == 10 || t->fmt == 11 ? NULL : t->hs;
    return t->hw;
}

/* ---------------------------------------------------------------- colibri's formats */
#define CU_NR 8           /* activation rows one block of cu_gemv_k takes */
__dpct_inline__ float
cu_fp8(unsigned b) { /* e4m3 as backend_vulkan.c's shaders decode it */
    const int e = (b >> 3) & 15, m = b & 7;
    const float v =
        e == 0 ? (float)m / 512.f : sycl::ldexp(1.f + (float)m / 8.f, e - 7);
    return (b & 0x80u) ? -v : v;
}
template <int FMT> struct CuEpw { static const int v = FMT == 1 || FMT == 12 || FMT == 13 ? 4 : FMT == 10 ? 1 : FMT == 11 ? 2 : 8; };
/* a row's word as its values (before any scale): int8, e4m3, f32, bf16, e2m1, or int4 as v + 8;
 * element k of a 4-bit word is bits 4k..4k+3 (byte j's low nibble is element 2j) */
template <int FMT>
__dpct_inline__ void cu_word(unsigned v, float *w, float const *cu_e2m1) {
    if (FMT == 1 || FMT == 13) {
        for (int k = 0; k < 4; k++) w[k] = (float)(int)(signed char)(v >> (8 * k));
    } else if (FMT == 12) {
        for (int k = 0; k < 4; k++) w[k] = cu_fp8((v >> (8 * k)) & 255u);
    } else if (FMT == 10) {
        w[0] = sycl::bit_cast<float>(v);
    } else if (FMT == 11) {
        w[0] = sycl::bit_cast<float>(v << 16);
            w[1] = sycl::bit_cast<float>(v & 0xffff0000u);
    } else if (FMT == 7) {
        for (int k = 0; k < 8; k++) { unsigned n = (v >> (4 * k)) & 15u; w[k] = (n & 8u) ? -cu_e2m1[n & 7u] : cu_e2m1[n & 7u]; }
    } else {
        for (int k = 0; k < 8; k++) w[k] = (float)((int)((v >> (4 * k)) & 15u) - 8);
    }
}
/* y[s][o] = sum_i W[o][i] x[s][i], up to NR rows of x a block: a warp a row of W, its lanes on
 * whole words (int3-g64: on whole groups of 64), each value times its group's scale, the row's
 * scale at the end (fmt 1, 2) */
template <int FMT, int NR, int SG>
void cu_gemv_k(float *__restrict__ y, const float *__restrict__ x,
               const unsigned *__restrict__ W, const float *__restrict__ sc,
               int I, int O, int rowWords, int gs, int S,
               float const *cu_e2m1) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & (SG - 1),
              o = item_ct1.get_group(2) * (256 / SG) + (item_ct1.get_local_id(2) / SG),
              s0 = item_ct1.get_group(1) * NR;
    if (o >= O) return;
    const int ns = S - s0 < NR ? S - s0 : NR;
    const unsigned *row = W + (size_t)o * rowWords;
    const float *xb = x + (size_t)s0 * I;
    float acc[NR];
#pragma unroll
    for (int s = 0; s < NR; s++) acc[s] = 0.f;
    if (FMT == 5) {
        const int ng = I / 64;
        const unsigned char *rb = (const unsigned char *)row;
        for (int g = lane; g < ng; g += SG) {
            const unsigned char *lo = rb + (size_t)g * 24, *hi = lo + 16;
            const float scl = sc[(size_t)o * ng + g];
            for (int j = 0; j < 64; j++) {
                const float w = (float)((int)(((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2)) - 4) * scl;
#pragma unroll
                for (int s = 0; s < NR; s++) if (s < ns) acc[s] += w * xb[(size_t)s * I + g * 64 + j];
            }
        }
    } else {
        const int E = CuEpw<FMT>::v, nw = (I + E - 1) / E;
        const int ng = FMT == 4 || FMT == 7 || FMT == 12 || FMT == 13 ? (I + gs - 1) / gs : 1;
        for (int wi = lane; wi < nw; wi += SG) {
            float w[CuEpw<FMT>::v];
            cu_word<FMT>(*(row + wi), w, cu_e2m1);
            const int i0 = wi * E;
            float scl = 1.f;
            if (FMT == 4 || FMT == 7 || FMT == 12 || FMT == 13) scl = sc[(size_t)o * ng + i0 / gs];
#pragma unroll
            for (int k = 0; k < CuEpw<FMT>::v; k++) {
                if (i0 + k >= I) break;
                const float wk = w[k] * scl;
#pragma unroll
                for (int s = 0; s < NR; s++) if (s < ns) acc[s] += wk * xb[(size_t)s * I + i0 + k];
            }
        }
    }
#pragma unroll
    for (int s = 0; s < NR; s++)
        for (int off = SG / 2; off > 0; off >>= 1) acc[s] += shfl_xor(acc[s], off);
    if (lane == 0) {
        const float rs = FMT == 1 || FMT == 2 ? sc[o] : 1.f;
        for (int s = 0; s < ns; s++) y[(size_t)(s0 + s) * O + o] = acc[s] * rs;
    }
}
template <int FMT>
static void cu_gemv_t(float *y, const float *x, const ColiVkTensor *t, int S,
                      sycl::queue * st) {
    const int wpb = 256 / C.sg;   /* rows a block: a sub-group each */
    const unsigned bx = (unsigned)((t->O + wpb - 1) / wpb);
    const float *cu_e2m1_ptr_ct1 = C.e2m1;
    auto t_w_ct2 = (const unsigned *)t->w;
    auto t_s_ct3 = t->s;
    auto t_I_ct4 = t->I;
    auto t_O_ct5 = t->O;
    auto t_rowWords_ct6 = t->rowWords;
    auto t_gs_ct7 = t->gs;
    if (S == 1)
        COLI_SG_LAUNCH(*st, C.sg, sycl::range<3>(1, 1, bx), 256, 0,
                       cu_gemv_k<FMT, 1, SG>(y, x, t_w_ct2, t_s_ct3, t_I_ct4, t_O_ct5, t_rowWords_ct6, t_gs_ct7, 1, cu_e2m1_ptr_ct1));
    else
        COLI_SG_LAUNCH(*st, C.sg, sycl::range<3>(1, (unsigned)((S + CU_NR - 1) / CU_NR), bx), 256, 0,
                       cu_gemv_k<FMT, CU_NR, SG>(y, x, t_w_ct2, t_s_ct3, t_I_ct4, t_O_ct5, t_rowWords_ct6, t_gs_ct7, S, cu_e2m1_ptr_ct1));
}
static int g_exact;   /* GQ_EXACT=1: a GGUF's blocks in the float form, as the CPU's reference */
/* y [S x O] = x [S x I] W^T on the batch's stream; 0 = the format has no kernel here */
static int cu_gemv(float *y, const float *x, const ColiVkTensor *t, int S,
                   sycl::queue * st) try {
    if (cu_bd_fmt(t->fmt))
        return bd_gemv_launch(y, x, (const bd_u32 *)t->w, C.tab, t->fmt, t->I, t->O, t->rowWords, S, C.smem_max, g_exact, C.sg, st);
    switch (t->fmt) {
    case 1: cu_gemv_t<1>(y, x, t, S, st); break;
    case 2: cu_gemv_t<2>(y, x, t, S, st); break;
    case 4: cu_gemv_t<4>(y, x, t, S, st); break;
    case 5: cu_gemv_t<5>(y, x, t, S, st); break;
    case 7: cu_gemv_t<7>(y, x, t, S, st); break;
    case 10: cu_gemv_t<10>(y, x, t, S, st); break;
    case 11: cu_gemv_t<11>(y, x, t, S, st); break;
    case 12: cu_gemv_t<12>(y, x, t, S, st); break;
    case 13: cu_gemv_t<13>(y, x, t, S, st); break;
    default: return 0;
    }
    return 1;   /* a submit that fails throws (sycl_caught): cudaPeekAtLastError's check always passes here */
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_gemv"); }

/* ---------------------------------------------------------------- the activations */
__dpct_inline__ float cu_bf16r(float v) {
    unsigned b = sycl::bit_cast<unsigned int>(v);
    if ((b & 0x7f800000u) != 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
    return sycl::bit_cast<float>(b & 0xffff0000u);
}
__dpct_inline__ float cu_situ_tanh(float x) {
    return sycl::fabs(x) < 0.000244140625f ? x : sycl::tanh(x);
}
__dpct_inline__ float cu_pow2(int k) {
    return sycl::bit_cast<float>((unsigned)(k + 127) << 23);
}
/* |x| <= 448: the nearest E4M3 value, ties to even (expert_act_v4.comp's e4m3r) */
__dpct_inline__ float cu_e4m3r(float x) {
    float a = sycl::fabs(x), r;
    if (a == 0.f) return x;
    if (a >= 448.f) r = 448.f;
    else if (a < 0.015625f) {
        float s = a * 512.f, f = sycl::floor(s), fr = s - f;
        if (fr > 0.5f || (fr == 0.5f && ((int)f & 1) == 1)) f += 1.f;
        r = f * 0.001953125f;
    } else {
        unsigned b = sycl::bit_cast<unsigned int>(a);
        int E = (int)((b >> 23) & 0xffu) - 127;
        unsigned sig = 0x800000u | (b & 0x7fffffu), q = sig >> 20, rem = sig & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) == 1u)) q++;
        r = (float)q * cu_pow2(E - 3);
    }
    return x < 0.f ? -r : r;
}
/* expert_act.comp: h = act(g, u) in place of g */
void cu_act_k(float *__restrict__ h, const float *__restrict__ u, long n, int act, float limit, float a, float b) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long i = (long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                   item_ct1.get_local_id(2);
    if (i >= n) return;
    float gt = h[i], ut = u[i], r;
    if (act == 1) {
        r = a * cu_situ_tanh(gt / a) * (1.f / (1.f + sycl::native::exp(-gt))) *
            (b * cu_situ_tanh(ut / b));
    } else if (act == 2) {
        gt = cu_bf16r(gt); ut = cu_bf16r(ut);
        if (limit > 0.f) {
            gt = sycl::fmin(gt, limit);
            ut = sycl::fmax(-limit, sycl::fmin(ut, limit));
        }
        const float s =
            gt >= 0.f ? 1.f / (1.f + sycl::native::exp(-gt))
                      : sycl::native::exp(gt) / (1.f + sycl::native::exp(gt));
        r = gt * s * ut;
    } else {
        if (limit > 0.f) {
            if (gt > limit) gt = limit;
            if (ut > limit) ut = limit;
            if (ut < -limit) ut = -limit;
        }
        r = (gt / (1.f + sycl::native::exp(-gt))) * ut;
    }
    h[i] = r;
}
/* expert_act_v4.comp: each row times its route weight to bf16, then E4M3 and back with one
 * power-of-two scale per 128 values */
void cu_act_v4_k(float *__restrict__ h, const float *__restrict__ rw, int rows, int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &amax = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int t = item_ct1.get_local_id(2), r = item_ct1.get_group(1),
              i = item_ct1.get_group(2) * 128 + t;
    const bool live = r < rows && i < n;
    const size_t at = (size_t)r * n + (size_t)(i > 0 ? i : 0);
    const float v = live ? cu_bf16r(h[at] * rw[r]) : 0.f;
    amax[t] = sycl::fabs(v);
    item_ct1.barrier();
    for (int s = 64; s > 0; s >>= 1) {
        if (t < s) amax[t] = sycl::fmax(amax[t], amax[t + s]);
        item_ct1.barrier();
    }
    const float m = sycl::fmax(amax[0], 1e-4f);
    const unsigned mb = sycl::bit_cast<unsigned int>(m);
    const int E = (int)((mb >> 23) & 0xffu) - 127;
    int e = (mb & 0x7fffffu) <= 0x600000u ? E - 8 : E - 7;
    e = e < -126 ? -126 : e > 126 ? 126 : e;
    const float x = sycl::fmin(sycl::fmax(v * cu_pow2(-e), -448.f), 448.f);
    if (live) h[at] = cu_e4m3r(x) * cu_pow2(e);
}

/* ---------------------------------------------------------------- the expert batch */
static int cu_xb_init(int D, int I, int act, float limit, float a, float b) {
    if (!cu_open() || D < 1 || I < 1 || C.inflight) return 0;
    if (C.xb && (D != C.D || I != C.I)) {   /* another geometry (a tier started again): the scratch anew */
        sycl::free(C.dX, dq());
            sycl::free(C.dG, dq());
            sycl::free(C.dU, dq());
            sycl::free(C.dY, dq());
            sycl::free(C.dW, dq());
        sycl::free(C.hX, dq());
            sycl::free(C.hY, dq());
            sycl::free(C.hW, dq());
        C.dX = C.dG = C.dU = C.dY = C.dW = C.hX = C.hY = C.hW = NULL; C.cap = 0;
    }
    C.D = D; C.I = I; C.act = act; C.limit = limit; C.a = a; C.b = b; C.xb = 1;
    const char *e = getenv("GQ_EXACT");
    g_exact = e && *e == '1';
    return 1;
}
static int cu_xb_init_dev(int dev, int D, int I, int act, float limit, float a, float b) {
    return dev == 0 ? cu_xb_init(D, I, act, limit, a, b) : 0;
}
static int cu_xb_ready(void) { return C.xb; }
static int cu_xb_ready_dev(int dev) { return dev == 0 && C.xb; }
static int cu_xb_queue_shared(void) { return 0; }
static ColiVkExpert *cu_xb_expert(ColiVkTensor *g, ColiVkTensor *u, ColiVkTensor *d) {
    if (!C.xb || !g || !u || !d || g->fmt != u->fmt) return NULL;
    if (g->I != C.D || g->O != C.I || u->I != C.D || u->O != C.I || d->I != C.I || d->O != C.D) return NULL;
    ColiVkExpert *e = (ColiVkExpert *)calloc(1, sizeof *e);
    if (e) { e->g = g; e->u = u; e->d = d; }
    return e;
}
static void cu_xb_expert_free(ColiVkExpert *e) {
    if (!e) return;
    cu_tensor_free(e->g); cu_tensor_free(e->u); cu_tensor_free(e->d);
    free(e);
}
static void cu_xb_expert_tensors(const ColiVkExpert *e, ColiVkTensor **g, ColiVkTensor **u, ColiVkTensor **d) {
    *g = e ? e->g : NULL; *u = e ? e->u : NULL; *d = e ? e->d : NULL;
}
static int cu_xb_expert_entry(const ColiVkExpert *e, uint32_t *entry) { (void)e; (void)entry; return 0; }

/* the scratch for `rows` rows: x, gate (then the activation), up, y and the route weights on
 * the device, x, y and the weights in page-locked RAM. Not while a batch is in flight. */
static int cu_grow(int rows) try {
    if ((size_t)rows <= C.cap) return 1;
    size_t n = C.cap ? C.cap : 64;
    while (n < (size_t)rows) n *= 2;
    sycl::free(C.dX, dq());
        sycl::free(C.dG, dq());
        sycl::free(C.dU, dq());
        sycl::free(C.dY, dq());
        sycl::free(C.dW, dq());
    sycl::free(C.hX, dq());
        sycl::free(C.hY, dq());
        sycl::free(C.hW, dq());
    C.dX = C.dG = C.dU = C.dY = C.dW = C.hX = C.hY = C.hW = NULL; C.cap = 0;
    const size_t D = (size_t)C.D, I = (size_t)C.I;
    if (COLI_SYCL_ALLOC(C.dX, (float *)sycl::malloc_device(
                             n * D * 4, dq()), n * D * 4) != 0 ||
        COLI_SYCL_ALLOC(C.dG, (float *)sycl::malloc_device(
                             n * I * 4, dq()), n * I * 4) != 0 ||
        COLI_SYCL_ALLOC(C.dU, (float *)sycl::malloc_device(
                             n * I * 4, dq()), n * I * 4) != 0 ||
        COLI_SYCL_ALLOC(C.dY, (float *)sycl::malloc_device(
                             n * D * 4, dq()), n * D * 4) != 0 ||
        COLI_SYCL_ALLOC(C.dW, (float *)sycl::malloc_device(
                             n * 4, dq()), n * 4) != 0 ||
        COLI_SYCL_ALLOC(C.hX, (float *)sycl::malloc_host(
                             n * D * 4, dq()), n * D * 4) != 0 ||
        COLI_SYCL_ALLOC(C.hY, (float *)sycl::malloc_host(
                             n * D * 4, dq()), n * D * 4) != 0 ||
        COLI_SYCL_ALLOC(C.hW, (float *)sycl::malloc_host(
                             n * 4, dq()), n * 4) != 0) {
        return 0;
    }
    C.cap = n;
    C.st.scratch_bytes = n * (2 * D + 2 * I + 1) * 4;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_grow"); }
static int cu_xb_issue_w(ColiVkExpert *const *ex, const int *rows, int count,
                         const float *const *xrows, const float *wrows) try {
    if (!C.xb || C.inflight || count < 1 || !cu_set()) return 0;
    int tot = 0;
    for (int c = 0; c < count; c++) tot += rows[c];
    if (tot < 1 || !cu_grow(tot)) return 0;
    const size_t D = (size_t)C.D, I = (size_t)C.I;
    for (int j = 0; j < tot; j++) {
        memcpy(C.hX + (size_t)j * D, xrows[j], D * 4);
        C.hW[j] = wrows ? wrows[j] : 1.f;
    }
    sycl::queue * st = C.xs;
    int ok =
        COLI_SYCL_TRY(sync_barrier(C.e0, st)) == 0 &&
        COLI_SYCL_TRY(st->memcpy(C.dX, C.hX, (size_t)tot * D * 4)) == 0 &&
        (C.act != 2 ||
         COLI_SYCL_TRY(st->memcpy(C.dW, C.hW, (size_t)tot * 4)) == 0);
    for (int c = 0, r0 = 0; c < count && ok; r0 += rows[c], c++) {
        const int n = rows[c];
        if (!n) continue;
        const ColiVkExpert *e = ex[c];
        float *g = C.dG + (size_t)r0 * I, *u = C.dU + (size_t)r0 * I;
        ok = cu_gemv(g, C.dX + (size_t)r0 * D, e->g, n, st) && cu_gemv(u, C.dX + (size_t)r0 * D, e->u, n, st);
        if (!ok) break;
        const long ne = (long)n * (long)I;
        st->submit([&](sycl::handler &cgh) {
            auto C_act_ct3 = C.act;
            auto C_limit_ct4 = C.limit;
            auto C_a_ct5 = C.a;
            auto C_b_ct6 = C.b;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(1, 1, (unsigned)((ne + 255) / 256)) *
                        sycl::range<3>(1, 1, 256),
                    sycl::range<3>(1, 1, 256)),
                [=](sycl::nd_item<3> item_ct1) {
                    cu_act_k(g, u, ne, C_act_ct3, C_limit_ct4, C_a_ct5,
                             C_b_ct6);
                });
        });
            if (C.act == 2) st->submit([&](sycl::handler &cgh) {
                const float *C_dW_r0_ct1 = C.dW + r0;

                cgh.parallel_for(
                    sycl::nd_range<3>(
                        sycl::range<3>(1, (unsigned)n,
                                       (unsigned)((I + 127) / 128)) *
                            sycl::range<3>(1, 1, 128),
                        sycl::range<3>(1, 1, 128)),
                    [=](sycl::nd_item<3> item_ct1) {
                        cu_act_v4_k(g, C_dW_r0_ct1, n, (int)I);
                    });
            });
        ok = 0 == 0 && cu_gemv(C.dY + (size_t)r0 * D, g, e->d, n, st);
    }
    ok =
        ok &&
        COLI_SYCL_TRY(st->memcpy(C.hY, C.dY, (size_t)tot * D * 4)) == 0 &&
        COLI_SYCL_TRY(sync_barrier(C.e1, st)) == 0;
    if (!ok) {   /* nothing to join: the caller computes these experts */
        st->wait();
        return 0;
    }
    {
        std::lock_guard<std::mutex> lk(C.mx);
        C.inflight = 1;
    }
    C.nrows = tot;
    C.st.batches++; C.st.experts += (unsigned long long)count; C.st.rows += (unsigned long long)tot;
    C.st.gemv_batches++;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_xb_issue_w"); }
static int cu_xb_issue_dev(int dev, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows,
                           const float *wrows) {
    return dev == 0 ? cu_xb_issue_w(ex, rows, count, xrows, wrows) : 0;
}
static int cu_xb_join(const float **yrows, double *device_ms) try {
    if (!C.inflight) return 0;
    int e = COLI_SYCL_TRY(C.e1->wait_and_throw());
    float ms = 0.f;
    if (e == 0) try {   /* the batch's bounds on a profiling queue: cudaEventElapsedTime's figure */
        ms = (C.e1->get_profiling_info<sycl::info::event_profiling::command_end>() -
              C.e0->get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000000.0f;
    } catch (sycl::exception const &) { ms = 0.f; }
    {
        std::lock_guard<std::mutex> lk(C.mx);
        C.inflight = 0;
        for (ColiVkTensor *t : C.pending) tensor_release(t);
        C.pending.clear();
    }
    if (e != 0) return 0;
    for (int j = 0; j < C.nrows; j++) yrows[j] = C.hY + (size_t)j * C.D;
    if (device_ms) *device_ms = ms;
    C.st.device_ms += ms;
    return 1;
}
catch (sycl::exception const &exc) { return sycl_caught(exc, "cu_xb_join"); }
static int cu_xb_join_dev(int dev, const float **yrows, double *device_ms) { return dev == 0 ? cu_xb_join(yrows, device_ms) : 0; }
static int cu_xb_busy_dev(int dev) { return dev == 0 && C.inflight; }
static void cu_xb_stats(ColiVkXbStats *st) {
    *st = C.st;
    st->timestamps = 1; st->queue_shared = 0; st->gemm_rows = 0;   /* no streaming yet */
}

/* streaming's sub-batches and the grouped step: not here yet (gemm_rows 0 keeps the tier off them) */
static float *cu_xb_step_begin(int S, int K, const float *x) { (void)S; (void)K; (void)x; return NULL; }
static const float *cu_xb_step_sum(const float *w, const uint8_t *use, double *ms) { (void)w; (void)use; (void)ms; return NULL; }
static void cu_xb_step_end(void) {}
static int cu_xb_sub_fit(int rows, int experts, size_t extra) { (void)rows; (void)experts; (void)extra; return 0; }
static int cu_xb_sub_reserve(int rows, int experts) { (void)rows; (void)experts; return 0; }
static int cu_xb_sub_issue(int h, ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows, const float *wrows) {
    (void)h; (void)ex; (void)rows; (void)count; (void)xrows; (void)wrows; return 0;
}
static int cu_xb_sub_issue_step(int h, ColiVkExpert *const *ex, const int *rows, int count, const int *assign, const float *wrows) {
    (void)h; (void)ex; (void)rows; (void)count; (void)assign; (void)wrows; return 0;
}
static int cu_xb_sub_join(int h, float *const *yout, double *ms) { (void)h; (void)yout; (void)ms; return 0; }
static int cu_xb_sub_busy(int h) { (void)h; return 0; }
static int cu_xb_sub_copy(int h, ColiVkTensor *const *dst, ColiVkTensor *const *src, int n) { (void)h; (void)dst; (void)src; (void)n; return 0; }
static void cu_xb_sub_copy_drop(void) {}

/* ---------------------------------------------------------------- the table */
static const CuTierOps g_ops = {
    CU_TIER_OPS_VERSION,
    cu_available, cu_device_name, cu_device_shares_ram, cu_device_local_bytes, cu_mem_budget, cu_dev_info, cu_dev2_open_env,
    cu_block_bytes, cu_block_bytes_dev, cu_buffer_alignment, cu_pool_stats, cu_tier_pool_limit, cu_tier_pool_limit_dev,
    cu_tier_extra_pool_limit, cu_tier_host_pool,
    cu_block_elems, cu_tensor_row_bytes, cu_tensor_scale_count, cu_tensor_payload, cu_tensor_alloc_bytes,
    cu_tier_tensor_pad, cu_tier_tensor_extra, cu_tier_tensor_like, cu_tensor_commit, cu_tensor_refill, cu_tensor_refill_as,
    cu_tensor_copy, cu_tensor_free, cu_tensor_bytes, cu_tensor_info, cu_tensor_host_rows,
    cu_xb_init, cu_xb_init_dev, cu_xb_ready, cu_xb_ready_dev, cu_xb_queue_shared, cu_xb_expert, cu_xb_expert_free,
    cu_xb_expert_tensors, cu_xb_expert_entry, cu_xb_issue_w, cu_xb_issue_dev, cu_xb_join, cu_xb_join_dev, cu_xb_busy_dev,
    cu_xb_stats,
    cu_xb_step_begin, cu_xb_step_sum, cu_xb_step_end, cu_xb_sub_fit, cu_xb_sub_reserve, cu_xb_sub_issue, cu_xb_sub_issue_step,
    cu_xb_sub_join, cu_xb_sub_busy, cu_xb_sub_copy, cu_xb_sub_copy_drop,
};
extern "C" COLI_CUDA_DLLEXPORT const CuTierOps *coli_cuda_tier_ops(void) { return &g_ops; }
