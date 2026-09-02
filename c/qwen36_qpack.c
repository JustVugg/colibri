/* qwen36_qpack.c -- MLX-affine routed experts for the qwen36 engine, read
 * from a Swiftlet qpack v1 container: the routed projections match the CPU
 * affine reference, from a bounded pool of expert slots.
 *
 * Reading each expert blob once and keeping it resident for the life of
 * the store is the simplest correct design but unbounded, and naive
 * eviction is unsafe: coli_metal_matmul_affine keys its persistent handle
 * on stable host pointers, so evicting without invalidation would hand the
 * GPU stale copied weights.  The store is therefore a BOUNDED pool of
 * refillable slots:
 *
 *   - each slot owns one page-aligned expert_stride blob (the qpack layout
 *     guarantees expert_stride % 16384 == 0), registered with the Metal
 *     backend exactly ONCE as a whole-slot zero-copy buffer;
 *   - a miss refills the least-recently-used slot in place (one pread) under
 *     a NEW generation; every dispatch runs through a generation-checked
 *     QqExpertRef, so a ref held across an eviction is refused, never
 *     consumed;
 *   - Metal dispatch addresses the gate/up/down weight/scale/bias sections
 *     by byte offset inside the slot's registered buffer
 *     (coli_metal_matmul_affine_slot) -- nothing is cached by the refillable
 *     host pointers, and refills write through the one registration;
 *   - dispatch is synchronous: when a forward returns, the GPU is done
 *     reading the slot and the next miss may refill it.  The async
 *     coli_metal_moe_block_begin/end overlap cannot carry this format
 *     without a shader redesign -- its bindless moe_gemv kernel binds one
 *     weight and one scale address per expert in the QT fmt namespace, with
 *     no bias section and no affine group reconstruction -- so this store
 *     ships the synchronous slot path; a begin/end arm for the affine
 *     format is a separate change.
 *
 * The affine format boundary holds: these blobs are unsigned uint32-packed
 * MLX Q4/Q8 with scale AND bias, never Colibri's signed-int4 QT fmt=4, and
 * they never enter the QT fmt namespace.
 *
 * Single-threaded by contract: only the engine's MoE loop touches the store
 * (the PILOT prefetch worker services the QT slot cache, not this one). */
#include "qwen36_qpack.h"

#ifdef QWEN36_QPACK

#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <malloc.h>
#endif

#include "qpack.h"
#ifdef COLI_METAL
#include "backend_metal.h"
#else
/* Keep one dispatch site below instead of two #ifdef bodies. */
typedef struct ColiMetalSlotBuffer ColiMetalSlotBuffer;
#endif

/* Default pool: keeps the tiny test fixtures (<= 64 experts) fully resident
 * with headroom while bounding real containers; qq_open always clamps the
 * pool to the container's total expert count. */
enum { QQ_DEFAULT_SLOT_COUNT = 96 };

typedef struct {
    unsigned char *blob;                 /* page-aligned, expert_stride bytes */
    int layer, eid;                      /* -1 = empty */
    uint32_t generation;                 /* bumped on every (re)fill */
    uint64_t last_use;                   /* LRU clock; 0 = never used */
    ColiAffineQuantizedView gate, up, down;
    ColiMetalSlotBuffer *mbuf;           /* whole-slot buffer, registered once */
} QqSlot;

static struct {
    ColiQpackReader reader;
    QqSlot *slots;                       /* [n_slots] bounded, refillable */
    float *scratch_g, *scratch_u, *scratch_h;   /* [inter],[inter],[hidden] */
    size_t slot_len;                     /* expert_stride */
    int n_slots;
    int n_layers, n_experts, hidden, inter;
    int active, force_cpu;
    uint64_t tick;                       /* LRU clock source */
    uint64_t evictions, fills;
    uint64_t metal_projections, cpu_projections;
} g_qq;

/* Survives qq_close so tests and the engine configure the NEXT open. */
static int g_qq_slot_request;

static int qq_fail(char *error, size_t capacity, const char *message) {
    if (error && capacity) snprintf(error, capacity, "%s", message);
    return 0;
}

/* Slot blobs must satisfy the whole-slot registration contract in
 * backend_metal.h: 16384-aligned base, length a multiple of 16384 (the
 * qpack layout already guarantees the stride). */
static unsigned char *qq_blob_alloc(size_t stride) {
#ifdef _WIN32
    return (unsigned char *)_aligned_malloc(stride, 16384);
#else
    void *p = NULL;
    if (posix_memalign(&p, 16384, stride) != 0) return NULL;
    return (unsigned char *)p;
#endif
}

static void qq_blob_free(unsigned char *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Build and geometry-check the three projection views for one slot fill.
 * gate/up must be [inter, hidden], down [hidden, inter]. */
static int qq_views(QqSlot *sl, char *error, size_t error_capacity) {
    const size_t stride = g_qq.slot_len;
    if (coli_qpack_affine_view(&g_qq.reader, sl->blob, stride, "gate_proj",
                               &sl->gate, error, error_capacity) ||
        coli_qpack_affine_view(&g_qq.reader, sl->blob, stride, "up_proj",
                               &sl->up, error, error_capacity) ||
        coli_qpack_affine_view(&g_qq.reader, sl->blob, stride, "down_proj",
                               &sl->down, error, error_capacity))
        return 0;
    if (sl->gate.input_dim != (size_t)g_qq.hidden ||
        sl->gate.output_dim != (size_t)g_qq.inter ||
        sl->up.input_dim != (size_t)g_qq.hidden ||
        sl->up.output_dim != (size_t)g_qq.inter ||
        sl->down.input_dim != (size_t)g_qq.inter ||
        sl->down.output_dim != (size_t)g_qq.hidden)
        return qq_fail(error, error_capacity,
                       "projection shape does not match hidden/inter");
    return 1;
}

/* Hit: touch LRU state and return the slot.  Miss: refill the least-
 * recently-used slot in place.  The victim is invalidated FIRST -- from that
 * point every ref captured against the old fill is refused at dispatch,
 * even if the read below fails halfway. */
static QqSlot *qq_acquire_slot(int layer, int eid, char *error,
                               size_t error_capacity) {
    if (layer < 0 || layer >= g_qq.n_layers ||
        eid < 0 || eid >= g_qq.n_experts) {
        qq_fail(error, error_capacity, "layer or expert out of range");
        return NULL;
    }
    QqSlot *victim = &g_qq.slots[0];
    for (int i = 0; i < g_qq.n_slots; i++) {
        QqSlot *sl = &g_qq.slots[i];
        if (sl->layer == layer && sl->eid == eid) {
            sl->last_use = ++g_qq.tick;
            return sl;
        }
        if (sl->last_use < victim->last_use) victim = sl;
    }
    if (victim->layer >= 0) g_qq.evictions++;
    victim->generation++;
    victim->layer = -1;
    victim->eid = -1;
    if (coli_qpack_read_expert(&g_qq.reader, (size_t)layer, (size_t)eid,
                               victim->blob, g_qq.slot_len,
                               error, error_capacity))
        return NULL;
    if (!qq_views(victim, error, error_capacity)) return NULL;
    victim->layer = layer;
    victim->eid = eid;
    victim->last_use = ++g_qq.tick;
    g_qq.fills++;
    return victim;
}

/* y[view->output_dim] = x @ dequant(view)^T for one row.  Metal first when
 * the pipelines are ready and the oracle switch allows it -- through the
 * slot's whole-buffer registration, addressed by section byte offsets, never
 * through a handle cached on the refillable host pointers.  CPU reference
 * otherwise.  Counted per projection so tests can prove which path ran. */
static int qq_matmul(QqSlot *sl, const ColiAffineQuantizedView *view,
                     const float *x, float *y) {
#ifdef COLI_METAL
    if (!g_qq.force_cpu && coli_metal_affine_available() &&
        coli_metal_affine_dispatch_supported(view, 1)) {
        if (!sl->mbuf)
            sl->mbuf = coli_metal_slot_register(sl->blob, g_qq.slot_len);
        if (sl->mbuf) {
            const unsigned char *blob = sl->blob;
            size_t woff = (size_t)((const unsigned char *)view->weights - blob);
            size_t soff = (size_t)((const unsigned char *)view->scales - blob);
            size_t boff = (size_t)((const unsigned char *)view->biases - blob);
            if (coli_metal_matmul_affine_slot(sl->mbuf, woff, soff, boff,
                                              y, x, 1, view)) {
                g_qq.metal_projections++;
                return 1;
            }
        }
    }
#else
    (void)sl;
#endif
    if (coli_affine_matmul_ref(y, x, 1, view) != COLI_AFFINE_OK) return 0;
    g_qq.cpu_projections++;
    return 1;
}

/* Full teardown, shared by qq_close and the qq_open failure path (safe on a
 * partially built store: coli_qpack_close tolerates a zeroed reader). */
static void qq_teardown(void) {
    if (g_qq.slots) {
        for (int i = 0; i < g_qq.n_slots; i++) {
            QqSlot *sl = &g_qq.slots[i];
#ifdef COLI_METAL
            if (sl->mbuf) coli_metal_slot_unregister(sl->mbuf);
#endif
            qq_blob_free(sl->blob);
        }
        free(g_qq.slots);
    }
    free(g_qq.scratch_g); free(g_qq.scratch_u); free(g_qq.scratch_h);
    coli_qpack_close(&g_qq.reader);
    memset(&g_qq, 0, sizeof(g_qq));
}

int qq_open(const char *container_dir, int n_layers, int n_experts,
            int hidden, int inter, char *error, size_t error_capacity) {
    if (g_qq.active)
        return qq_fail(error, error_capacity, "qpack store is already open");
    if (!container_dir || n_layers <= 0 || n_experts <= 0 ||
        hidden <= 0 || inter <= 0)
        return qq_fail(error, error_capacity, "invalid qpack store geometry");
    if (coli_qpack_open(&g_qq.reader, container_dir, error, error_capacity))
        return 0;
    g_qq.n_layers = n_layers; g_qq.n_experts = n_experts;
    g_qq.hidden = hidden; g_qq.inter = inter;
    if (g_qq.reader.layout.layer_count != (size_t)n_layers ||
        g_qq.reader.layout.expert_count != (size_t)n_experts) {
        qq_fail(error, error_capacity,
                "container layer/expert count does not match the model");
        goto fail;
    }
    g_qq.slot_len = g_qq.reader.layout.expert_stride;
    {
        size_t total = (size_t)n_layers * (size_t)n_experts;
        int slots = g_qq_slot_request > 0 ? g_qq_slot_request
                                          : QQ_DEFAULT_SLOT_COUNT;
        if ((size_t)slots > total) slots = (int)total;
        g_qq.n_slots = slots;
    }
    g_qq.slots = (QqSlot *)calloc((size_t)g_qq.n_slots, sizeof(QqSlot));
    g_qq.scratch_g = (float *)malloc((size_t)inter * sizeof(float));
    g_qq.scratch_u = (float *)malloc((size_t)inter * sizeof(float));
    g_qq.scratch_h = (float *)malloc((size_t)hidden * sizeof(float));
    if (!g_qq.slots || !g_qq.scratch_g || !g_qq.scratch_u ||
        !g_qq.scratch_h) {
        qq_fail(error, error_capacity, "out of memory for qpack store");
        goto fail;
    }
    for (int i = 0; i < g_qq.n_slots; i++) {
        g_qq.slots[i].layer = g_qq.slots[i].eid = -1;
        g_qq.slots[i].blob = qq_blob_alloc(g_qq.slot_len);
        if (!g_qq.slots[i].blob) {
            qq_fail(error, error_capacity, "out of memory for expert slots");
            goto fail;
        }
    }
    /* Probe expert (0,0) now so a shape mismatch is an open-time refusal,
     * not a mid-decode one. */
    if (!qq_acquire_slot(0, 0, error, error_capacity)) goto fail;
    g_qq.active = 1;
    return 1;
fail:
    qq_teardown();
    return 0;
}

int qq_active(void) { return g_qq.active; }

void qq_set_slot_count(int n_slots) {
    g_qq_slot_request = n_slots > 0 ? n_slots : 0;
}

void qq_slot_stats(int *n_slots, uint64_t *evictions, uint64_t *fills) {
    if (n_slots) *n_slots = g_qq.n_slots;
    if (evictions) *evictions = g_qq.evictions;
    if (fills) *fills = g_qq.fills;
}

void qq_force_cpu(int force) { g_qq.force_cpu = force ? 1 : 0; }

int qq_expert_acquire(int layer, int eid, QqExpertRef *ref) {
    if (!g_qq.active || !ref) return 0;
    char error[256];
    QqSlot *sl = qq_acquire_slot(layer, eid, error, sizeof(error));
    if (!sl) {
        fprintf(stderr, "[qpack] expert layer %d expert %d: %s\n",
                layer, eid, error);
        return 0;
    }
    ref->slot = (int32_t)(sl - g_qq.slots);
    ref->generation = sl->generation;
    return 1;
}

int qq_ref_forward(QqExpertRef ref, const float *x, float weight,
                   float *out) {
    if (!g_qq.active || !x || !out) return 0;
    if (ref.slot < 0 || ref.slot >= g_qq.n_slots) return 0;
    QqSlot *sl = &g_qq.slots[ref.slot];
    /* The stale check IS the slot contract: an evicted slot must be impossible
     * to consume.  Checked here, at dispatch time, not at acquire. */
    if (sl->generation != ref.generation || sl->layer < 0) {
        fprintf(stderr, "[qpack] stale slot ref refused (slot %d generation"
                " %u, ref %u)\n", (int)ref.slot, sl->generation,
                ref.generation);
        return 0;
    }
    float *g = g_qq.scratch_g, *u = g_qq.scratch_u, *h = g_qq.scratch_h;
    if (!qq_matmul(sl, &sl->gate, x, g) || !qq_matmul(sl, &sl->up, x, u))
        return 0;
    for (int i = 0; i < g_qq.inter; i++) {
        float gv = g[i];
        g[i] = (gv / (1.f + expf(-gv))) * u[i];
    }
    if (!qq_matmul(sl, &sl->down, g, h)) return 0;
    for (int d = 0; d < g_qq.hidden; d++) out[d] += weight * h[d];
    return 1;
}

int qq_expert_forward(int layer, int eid, const float *x, float weight,
                      float *out) {
    if (!g_qq.active || !x || !out) return 0;
    QqExpertRef ref;
    if (!qq_expert_acquire(layer, eid, &ref)) return 0;
    return qq_ref_forward(ref, x, weight, out);
}

void qq_counts(uint64_t *metal_projections, uint64_t *cpu_projections) {
    if (metal_projections) *metal_projections = g_qq.metal_projections;
    if (cpu_projections) *cpu_projections = g_qq.cpu_projections;
}

void qq_close(void) { qq_teardown(); }

#endif /* QWEN36_QPACK */
