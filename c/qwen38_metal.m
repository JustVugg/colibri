/* Metal glue for the Qwen3.8 kernels.
 *
 * A second library, compiled from metal/qwen38.metal (built into this object),
 * on the device qwen36_metal.m opened. Its wrappers check sizes, bind
 * q36_tensors and dispatch through q36_metal_encoder, so they join qwen36's
 * current batch and encoder in call order; q36_gpu_flush and q36_gpu_sync
 * commit and wait for both. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "qwen38_gpu.h"
#include "qwen36_metal_internal.h"

static const char q38_source[] = {
#embed "metal/qwen38.metal"
    , 0
};

enum { K38_COPY, K38_DOT_BF16_EXACT, K38_DOT_BF16_FAST, K38_HC_LOW, K38_HC_MIX, K38_HC_INJ, K38_HC_APPLY, K38_HC_NORM,
       K38_PLE_GATE, K38_PLE_CONV, K38_QSA_POOL, K38_QSA_SCORE, K38_QSA_SELECT, K38_ATTN_SEL, K38_ATTN_SEL_HALF,
       K38_ROPE_HALF, K38_QGATE_SPLIT, K38_KV_APPEND, K38_KV_APPEND_HALF, K38_IDX_PREP, K38_COUNT };

static const char *const q38_kernel_names[K38_COUNT] = {
    "q38_copy",
    "q38_dot_bf16_exact",
    "q38_dot_bf16_fast",
    "q38_hc_low",
    "q38_hc_mix",
    "q38_hc_inj",
    "q38_hc_apply",
    "q38_hc_norm",
    "q38_ple_gate",
    "q38_ple_conv",
    "q38_qsa_pool",
    "q38_qsa_score",
    "q38_qsa_select",
    "q38_attn_sel",
    "q38_attn_sel_half",
    "q38_rope_half",
    "q38_qgate_split",
    "q38_kv_append",
    "q38_kv_append_half",
    "q38_idx_prep",
};

static id<MTLComputePipelineState> g_q38_pipe[K38_COUNT];
static id<MTLDevice> g_q38_device;              /* the device g_q38_pipe was built on */

/* The one place the qwen38 library's compile options are made: the options of
 * qwen36's library (IEEE float, no fast-math shortcuts). Test-only macros reach
 * the source from here (preprocessorMacros). */
static MTLCompileOptions *q38_compile_options(void) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
#ifdef Q38_COUNTER_TEST   /* test-only: tests/test_qwen38_metal_kernels counter builds */
    options.preprocessorMacros = @{@"Q38_COUNTER_TEST" : @(Q38_COUNTER_TEST)};
#endif
    return options;
}

int q38_gpu_init(void) {
    if (!q36_gpu_init()) return 0;
    id<MTLDevice> device = q36_metal_device();
    if (g_q38_device == device) return 1;
    @autoreleasepool {
        g_q38_device = nil;
        for (int i = 0; i < K38_COUNT; i++) g_q38_pipe[i] = nil;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:q38_source]
                                                      options:q38_compile_options() error:&error];
        if (!library) {
            fprintf(stderr, "[qwen38-metal] kernel compilation failed: %s\n",
                    error ? error.localizedDescription.UTF8String : "unknown error");
            return 0;
        }
        for (int i = 0; i < K38_COUNT; i++) {
            id<MTLFunction> fn = [library newFunctionWithName:@(q38_kernel_names[i])];
            g_q38_pipe[i] = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
            /* The SIMD reductions assume 32-lane groups, as qwen36's do. */
            if (!g_q38_pipe[i] || g_q38_pipe[i].threadExecutionWidth != 32) {
                fprintf(stderr, "[qwen38-metal] pipeline %s unavailable\n", q38_kernel_names[i]);
                for (int j = 0; j <= i; j++) g_q38_pipe[j] = nil;
                return 0;
            }
        }
        g_q38_device = device;
    }
    return 1;
}

/* ---- wrappers ------------------------------------------------------------- */

#define F4 (uint64_t)sizeof(float)

/* q38_gpu_init has built the pipelines, and q36_gpu_cleanup has not closed the device since. */
static int q38_ready(void) {
    return g_q38_device && g_q38_device == q36_metal_device();
}

/* n threads in groups of up to 256 (the pipeline's limit if lower). */
static void q38_threads(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> pso, uint64_t n) {
    uint64_t group = pso.maxTotalThreadsPerThreadgroup < 256 ? pso.maxTotalThreadsPerThreadgroup : 256;
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n < group ? n : group, 1, 1)];
}

int q38_copy(q36_tensor *y, q36_tensor *x, size_t n) {
    if (!q38_ready() || n < 1 || n > UINT32_MAX || !q36_metal_fits(x, (uint64_t)n * F4) ||
        !q36_metal_fits(y, (uint64_t)n * F4)) return 0;
    @autoreleasepool {
        uint32_t count = (uint32_t)n;
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_COPY];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, x, 0);
        q36_metal_bind(e, y, 1);
        [e setBytes:&count length:sizeof(count) atIndex:2];
        q38_threads(e, pso, (uint64_t)n);
    }
    return 1;
}

/* y (rows x out) = x (rows x in) . w^T, w raw BF16 out x in. One 32-lane threadgroup a cell. */
int q38_dot_bf16(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int in, int out) {
    if (!q38_ready() || rows < 1 || in < 1 || out < 1 || in % 8 != 0 || out > 65535 || rows > 65535 ||
        !q36_metal_fits(x, (uint64_t)rows * in * F4) || !q36_metal_fits(y, (uint64_t)rows * out * F4) ||
        !q36_metal_fits(w, (uint64_t)out * in * 2)) return 0;
    @autoreleasepool {
        uint32_t args[3] = {(uint32_t)rows, (uint32_t)in, (uint32_t)out};
        id<MTLComputePipelineState> pso = g_q38_pipe[q36_gpu_is_exact() ? K38_DOT_BF16_EXACT : K38_DOT_BF16_FAST];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, x, 0);
        q36_metal_bind(e, w, 1);
        q36_metal_bind(e, y, 2);
        [e setBytes:args length:sizeof(args) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)out, (NSUInteger)rows, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    return 1;
}

/* ---- hyper-connection ops ------------------------------------------------- */

typedef struct { uint32_t n, D, C; float fc; } q38_hc_args;

static int q38_hc_unary(int kernel, q36_tensor *y, q36_tensor *a, int n, float fc) {
    if (!q38_ready() || n < 1 || !(fc > 0.0f) || !q36_metal_fits(a, (uint64_t)n * F4) ||
        !q36_metal_fits(y, (uint64_t)n * F4)) return 0;
    @autoreleasepool {
        q38_hc_args p = {(uint32_t)n, 0, 0, fc};
        id<MTLComputePipelineState> pso = g_q38_pipe[kernel];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, a, 0);
        q36_metal_bind(e, y, 1);
        [e setBytes:&p length:sizeof(p) atIndex:2];
        q38_threads(e, pso, (uint64_t)n);
    }
    return 1;
}

int q38_hc_low(q36_tensor *y, q36_tensor *a, int n, float fc) { return q38_hc_unary(K38_HC_LOW, y, a, n, fc); }
int q38_hc_inj(q36_tensor *y, q36_tensor *a, int n, float fc) { return q38_hc_unary(K38_HC_INJ, y, a, n, fc); }

int q38_hc_mix(q36_tensor *y, q36_tensor *a, q36_tensor *b, int rows, int D, int C, float fc) {
    if (!q38_ready() || rows < 1 || D < 1 || C < 1 || !(fc > 0.0f) || (uint64_t)rows * D > UINT32_MAX ||
        (uint64_t)rows * D * C > UINT32_MAX || !q36_metal_fits(a, (uint64_t)rows * D * C * F4) ||
        !q36_metal_fits(b, (uint64_t)rows * D * C * F4) || !q36_metal_fits(y, (uint64_t)rows * D * F4)) return 0;
    @autoreleasepool {
        q38_hc_args p = {(uint32_t)((uint64_t)rows * D), (uint32_t)D, (uint32_t)C, fc};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_HC_MIX];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, a, 0);
        q36_metal_bind(e, b, 1);
        q36_metal_bind(e, y, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        q38_threads(e, pso, (uint64_t)rows * D);
    }
    return 1;
}

int q38_hc_apply(q36_tensor *hyper, q36_tensor *inject, q36_tensor *block, int rows, int D, int C) {
    if (!q38_ready() || rows < 1 || D < 1 || C < 1 || (uint64_t)rows * D * C > UINT32_MAX ||
        !q36_metal_fits(hyper, (uint64_t)rows * D * C * F4) || !q36_metal_fits(inject, (uint64_t)rows * C * F4) ||
        !q36_metal_fits(block, (uint64_t)rows * D * F4)) return 0;
    @autoreleasepool {
        q38_hc_args p = {(uint32_t)((uint64_t)rows * D * C), (uint32_t)D, (uint32_t)C, 1.0f};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_HC_APPLY];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, hyper, 0);
        q36_metal_bind(e, inject, 1);
        q36_metal_bind(e, block, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        q38_threads(e, pso, (uint64_t)rows * D * C);
    }
    return 1;
}

static int q38_tg256(int kernel) { return g_q38_pipe[kernel].maxTotalThreadsPerThreadgroup >= 256; }

int q38_hc_norm(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int D, int C, float eps) {
    if (!q38_ready() || rows < 1 || D < 1 || C < 1 || !(eps > 0.0f) || !q38_tg256(K38_HC_NORM) || (uint64_t)rows * C > UINT32_MAX ||
        (uint64_t)rows * D * C > UINT32_MAX || !q36_metal_fits(x, (uint64_t)rows * D * C * F4) ||
        !q36_metal_fits(y, (uint64_t)rows * D * C * F4) || !q36_metal_fits(w, (uint64_t)D * C * F4)) return 0;
    @autoreleasepool {
        struct { uint32_t D, C; float eps; } p = {(uint32_t)D, (uint32_t)C, eps};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_HC_NORM];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, x, 0);
        q36_metal_bind(e, w, 1);
        q36_metal_bind(e, y, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)rows * C, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}

/* ---- PLE gate and dilated conv -------------------------------------------- */

typedef struct { uint32_t S, C, W, H, CK, NG, snap_rows, accumulate; float eps; } q38_ple_args;

/* The shape both PLE kernels share: W = C*H, every count positive, sizes within 32 bits. */
static int q38_ple_shape_ok(const q38_ple_shape *s) {
    return s && s->S >= 1 && s->C >= 1 && s->H >= 1 && s->W >= 1 && (int64_t)s->C * s->H == s->W &&
           (uint64_t)s->S * s->W <= UINT32_MAX;
}

static q38_ple_args q38_ple_pack(const q38_ple_shape *s) {
    q38_ple_args p = {(uint32_t)s->S, (uint32_t)s->C, (uint32_t)s->W, (uint32_t)s->H, (uint32_t)s->CK,
                      (uint32_t)s->NG, (uint32_t)s->snap_rows, s->accumulate ? 1u : 0u, s->eps};
    return p;
}

int q38_ple_gate(q36_tensor *gated, q36_tensor *normv, q36_tensor *keys, q36_tensor *hyper,
                 q36_tensor *value, q36_tensor *prm, const q38_ple_shape *s) {
    if (!q38_ready() || !q38_ple_shape_ok(s) || !(s->eps > 0.0f) || !q38_tg256(K38_PLE_GATE)) return 0;
    uint64_t n = (uint64_t)s->S * s->W * F4;
    if (!q36_metal_fits(keys, n) || !q36_metal_fits(hyper, n) || !q36_metal_fits(gated, n) ||
        !q36_metal_fits(normv, n) || !q36_metal_fits(value, (uint64_t)s->S * s->H * F4) ||
        !q36_metal_fits(prm, (uint64_t)3 * s->W * F4)) return 0;
    @autoreleasepool {
        q38_ple_args p = q38_ple_pack(s);
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_PLE_GATE];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, keys, 0);
        q36_metal_bind(e, hyper, 1);
        q36_metal_bind(e, value, 2);
        q36_metal_bind(e, prm, 3);
        q36_metal_bind(e, gated, 4);
        q36_metal_bind(e, normv, 5);
        [e setBytes:&p length:sizeof(p) atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)s->S * s->C, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}

int q38_ple_conv(q36_tensor *out, q36_tensor *ring, q36_tensor *snap, q36_tensor *gated,
                 q36_tensor *normv, q36_tensor *conv, const q38_ple_shape *s) {
    if (!q38_ready() || !q38_ple_shape_ok(s) || s->CK < 2 || s->NG < 1 || (int64_t)(s->CK - 1) * s->NG > 32 ||
        s->snap_rows < 0 || s->snap_rows > s->S) return 0;
    uint64_t SL = (uint64_t)(s->CK - 1) * s->NG, n = (uint64_t)s->S * s->W * F4, nr = (uint64_t)s->W * SL * F4;
    if (!q36_metal_fits(out, n) || !q36_metal_fits(gated, n) || !q36_metal_fits(normv, n) ||
        !q36_metal_fits(ring, nr) || !q36_metal_fits(conv, (uint64_t)s->W * s->CK * F4) ||
        (s->snap_rows > 0 && !q36_metal_fits(snap, (uint64_t)s->snap_rows * nr))) return 0;
    @autoreleasepool {
        q38_ple_args p = q38_ple_pack(s);
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_PLE_CONV];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, out, 0);
        q36_metal_bind(e, ring, 1);
        q36_metal_bind(e, s->snap_rows > 0 ? snap : ring, 2);   /* not written without snap rows */
        q36_metal_bind(e, gated, 3);
        q36_metal_bind(e, normv, 4);
        q36_metal_bind(e, conv, 5);
        [e setBytes:&p length:sizeof(p) atIndex:6];
        q38_threads(e, pso, (uint64_t)s->W);
    }
    return 1;
}

/* ---- QSA: block keys, scores, selection, sparse attention ----------------- */

#define Q38_SEL_MAX 2048   /* qwen38.metal's: the most blocks a row takes (budget / R) */

/* A shape the QSA kernels take: positive counts, ID <= 256, an even rotary <= ID. */
static int q38_qsa_shape_ok(const q38_qsa_shape *s) {
    return s && s->IQ >= 1 && s->ID >= 1 && s->ID <= 256 && s->R >= 1 && s->budget >= 0 && s->rotary >= 0 &&
           s->rotary % 2 == 0 && s->rotary <= s->ID && s->eps > 0.0f && s->theta > 0.0f;
}

/* q38_rope's bases, the CPU's powf in float, and their reciprocals (qwen38.metal's
 * q38_rope_table); returns rotary / 2. rotary <= 256. */
typedef struct { float base[128], inv[128]; } q38_rope_table;
static int q38_rope_fill(q38_rope_table *rt, float theta, int rotary) {
    int half = rotary / 2;
    for (int i = 0; i < 128; i++) {
        rt->base[i] = i < half ? powf(theta, (float)(2 * i) / rotary) : 1.0f;
        rt->inv[i] = 1.0f / rt->base[i];
    }
    return half;
}


int q38_qsa_pool(q36_tensor *pk, q36_tensor *ikraw, q36_tensor *idx_kn, int b0, int nb, const q38_qsa_shape *s) {
    if (!q38_ready() || !q38_qsa_shape_ok(s) || b0 < 0 || nb < 1 || !q38_tg256(K38_QSA_POOL)) return 0;
    uint64_t blocks = (uint64_t)b0 + (uint64_t)nb;
    if (blocks * (uint64_t)s->R > UINT32_MAX || !q36_metal_fits(ikraw, blocks * s->R * s->ID * F4) ||
        !q36_metal_fits(pk, blocks * s->ID * F4) || !q36_metal_fits(idx_kn, (uint64_t)s->ID * F4)) return 0;
    @autoreleasepool {
        q38_rope_table rt;
        int half = q38_rope_fill(&rt, s->theta, s->rotary);
        struct { uint32_t ID, R, b0, half_; float eps; } p = {(uint32_t)s->ID, (uint32_t)s->R, (uint32_t)b0,
                                                               (uint32_t)half, s->eps};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_QSA_POOL];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, ikraw, 0);
        q36_metal_bind(e, idx_kn, 1);
        q36_metal_bind(e, pk, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        [e setBytes:&rt length:sizeof(rt) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}

int q38_qsa_score(q36_tensor *sc, q36_tensor *q, q36_tensor *pk, int S, int pos_base, int nbmax, const q38_qsa_shape *s) {
    if (!q38_ready() || !q38_qsa_shape_ok(s) || S < 1 || pos_base < 0 || nbmax < 0 ||
        (uint64_t)pos_base + S > INT32_MAX) return 0;
    uint64_t most = ((uint64_t)pos_base + S) / s->R;   /* the last row's blocks */
    if (most > (uint64_t)nbmax || !q36_metal_fits(q, (uint64_t)S * s->IQ * s->ID * F4) ||
        !q36_metal_fits(pk, most * s->ID * F4) || !q36_metal_fits(sc, (uint64_t)S * nbmax * F4)) return 0;
    if (!most) return 1;   /* no row has a block yet: nothing to score */
    @autoreleasepool {
        struct { uint32_t ID, R, S, pos_base, IQ, nbmax; } p = {(uint32_t)s->ID, (uint32_t)s->R, (uint32_t)S,
                                                                (uint32_t)pos_base, (uint32_t)s->IQ, (uint32_t)nbmax};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_QSA_SCORE];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, q, 0);
        q36_metal_bind(e, pk, 1);
        q36_metal_bind(e, sc, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        uint64_t group = pso.maxTotalThreadsPerThreadgroup < 256 ? pso.maxTotalThreadsPerThreadgroup : 256;
        [e dispatchThreads:MTLSizeMake((NSUInteger)most, (NSUInteger)S, 1)
            threadsPerThreadgroup:MTLSizeMake(most < group ? (NSUInteger)most : (NSUInteger)group, 1, 1)];
    }
    return 1;
}

int q38_qsa_select(q36_tensor *sel, q36_tensor *sc, int S, int pos_base, int nbmax, int sel_row, const q38_qsa_shape *s) {
    if (!q38_ready() || !q38_qsa_shape_ok(s) || S < 1 || pos_base < 0 || nbmax < 0 ||
        (uint64_t)pos_base + S > INT32_MAX || s->budget / s->R > Q38_SEL_MAX || !q38_tg256(K38_QSA_SELECT)) return 0;
    uint64_t most = ((uint64_t)pos_base + S) / s->R;
    if (most > (uint64_t)nbmax || sel_row < s->budget / s->R * s->R + s->R ||
        !q36_metal_fits(sc, (uint64_t)S * nbmax * F4) || !q36_metal_fits(sel, (uint64_t)S * sel_row * F4)) return 0;
    @autoreleasepool {
        struct { uint32_t R, S, pos_base, budget, nbmax, sel_row; } p = {
            (uint32_t)s->R, (uint32_t)S, (uint32_t)pos_base, (uint32_t)s->budget, (uint32_t)nbmax, (uint32_t)sel_row};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_QSA_SELECT];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, sc, 0);
        q36_metal_bind(e, sel, 1);
        [e setBytes:&p length:sizeof(p) atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)S, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}

int q38_attn_sel(q36_tensor *out, q36_tensor *q, q36_tensor *gate, q36_tensor *kc, q36_tensor *vc,
                 q36_tensor *sel, int S, int pos_base, int heads, int kv_heads, int D, int cap, int sel_row,
                 int kv_half) {
    int kernel = kv_half ? K38_ATTN_SEL_HALF : K38_ATTN_SEL;
    if (!q38_ready() || S < 1 || S > 65535 || pos_base < 0 || heads < 1 || kv_heads < 1 || heads % kv_heads ||
        D < 1 || D > 256 || cap < 1 || (int64_t)pos_base + S > cap || sel_row < 1 || !q38_tg256(kernel)) return 0;
    uint64_t qn = (uint64_t)S * heads * D * F4, cache = (uint64_t)kv_heads * cap * D * (kv_half ? 2 : F4);
    if (!q36_metal_fits(q, qn) || !q36_metal_fits(gate, qn) || !q36_metal_fits(out, qn) ||
        !q36_metal_fits(kc, cache) || !q36_metal_fits(vc, cache) ||
        !q36_metal_fits(sel, (uint64_t)S * sel_row * F4)) return 0;
    @autoreleasepool {
        struct { uint32_t S, pos_base, heads, kv_heads, D, cap, sel_row; } p = {
            (uint32_t)S, (uint32_t)pos_base, (uint32_t)heads, (uint32_t)kv_heads, (uint32_t)D, (uint32_t)cap,
            (uint32_t)sel_row};
        id<MTLComputePipelineState> pso = g_q38_pipe[kernel];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, q, 0);
        q36_metal_bind(e, gate, 1);
        q36_metal_bind(e, kc, 2);
        q36_metal_bind(e, vc, 3);
        q36_metal_bind(e, sel, 4);
        q36_metal_bind(e, out, 5);
        [e setBytes:&p length:sizeof(p) atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)heads, (NSUInteger)S, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}

/* ---- attention glue: RoPE, q/gate split, KV and indexer append ------------ */

int q38_rope_half(q36_tensor *x, int rows, int heads, int D, int rotary, int pos_base, float theta) {
    if (!q38_ready() || rows < 1 || heads < 1 || D < 1 || rotary < 0 || rotary % 2 || rotary > D || rotary > 256 ||
        pos_base < 0 || (int64_t)pos_base + rows > INT32_MAX || !(theta > 0.0f)) return 0;
    uint64_t n = (uint64_t)rows * heads * (rotary / 2);
    if ((uint64_t)rows * heads * D > UINT32_MAX || !q36_metal_fits(x, (uint64_t)rows * heads * D * F4)) return 0;
    if (!n) return 1;   /* rotary 0: nothing turns */
    @autoreleasepool {
        q38_rope_table rt;
        int half = q38_rope_fill(&rt, theta, rotary);
        struct { uint32_t n, heads, D, half_, pos_base; } p = {(uint32_t)n, (uint32_t)heads, (uint32_t)D,
                                                               (uint32_t)half, (uint32_t)pos_base};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_ROPE_HALF];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, x, 0);
        [e setBytes:&p length:sizeof(p) atIndex:1];
        [e setBytes:&rt length:sizeof(rt) atIndex:2];
        q38_threads(e, pso, n);
    }
    return 1;
}

int q38_qgate_split(q36_tensor *q, q36_tensor *gate, q36_tensor *qraw, int rows, int heads, int D) {
    if (!q38_ready() || rows < 1 || heads < 1 || D < 1) return 0;
    uint64_t n = (uint64_t)rows * heads * D;
    if (2 * n > UINT32_MAX || !q36_metal_fits(qraw, 2 * n * F4) || !q36_metal_fits(q, n * F4) ||
        !q36_metal_fits(gate, n * F4)) return 0;
    @autoreleasepool {
        struct { uint32_t n, D; } p = {(uint32_t)n, (uint32_t)D};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_QGATE_SPLIT];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, qraw, 0);
        q36_metal_bind(e, q, 1);
        q36_metal_bind(e, gate, 2);
        [e setBytes:&p length:sizeof(p) atIndex:3];
        q38_threads(e, pso, n);
    }
    return 1;
}

int q38_kv_append(q36_tensor *kc, q36_tensor *vc, q36_tensor *k, q36_tensor *v, int rows, int pos_base,
                  int kv_heads, int D, int cap, int half) {
    if (!q38_ready() || rows < 1 || pos_base < 0 || kv_heads < 1 || D < 1 || cap < 1 ||
        (int64_t)pos_base + rows > cap) return 0;
    uint64_t n = (uint64_t)rows * kv_heads * D, cache = (uint64_t)kv_heads * cap * D * (half ? 2 : F4);
    if (n > UINT32_MAX || !q36_metal_fits(k, n * F4) || !q36_metal_fits(v, n * F4) ||
        !q36_metal_fits(kc, cache) || !q36_metal_fits(vc, cache)) return 0;
    @autoreleasepool {
        struct { uint32_t n, kv_heads, D, cap, pos_base; } p = {(uint32_t)n, (uint32_t)kv_heads, (uint32_t)D,
                                                                (uint32_t)cap, (uint32_t)pos_base};
        id<MTLComputePipelineState> pso = g_q38_pipe[half ? K38_KV_APPEND_HALF : K38_KV_APPEND];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, kc, 0);
        q36_metal_bind(e, vc, 1);
        q36_metal_bind(e, k, 2);
        q36_metal_bind(e, v, 3);
        [e setBytes:&p length:sizeof(p) atIndex:4];
        q38_threads(e, pso, n);
    }
    return 1;
}

int q38_idx_prep(q36_tensor *q, q36_tensor *ikc, q36_tensor *ip, q36_tensor *idx_qn, int rows, int pos_base,
                 int cap, const q38_qsa_shape *s) {
    if (!q38_ready() || !q38_qsa_shape_ok(s) || rows < 1 || rows > 65535 || pos_base < 0 || cap < 1 ||
        (int64_t)pos_base + rows > cap || !q38_tg256(K38_IDX_PREP)) return 0;
    uint64_t ID = (uint64_t)s->ID;
    if (!q36_metal_fits(ip, (uint64_t)rows * (s->IQ + 1) * ID * F4) ||
        !q36_metal_fits(q, (uint64_t)rows * s->IQ * ID * F4) || !q36_metal_fits(ikc, (uint64_t)cap * ID * F4) ||
        !q36_metal_fits(idx_qn, ID * F4)) return 0;
    @autoreleasepool {
        q38_rope_table rt;
        int half = q38_rope_fill(&rt, s->theta, s->rotary);
        struct { uint32_t IQ, ID, half_, pos_base; float eps; } p = {(uint32_t)s->IQ, (uint32_t)s->ID, (uint32_t)half,
                                                                     (uint32_t)pos_base, s->eps};
        id<MTLComputePipelineState> pso = g_q38_pipe[K38_IDX_PREP];
        id<MTLComputeCommandEncoder> e = q36_metal_encoder(pso);
        q36_metal_bind(e, ip, 0);
        q36_metal_bind(e, idx_qn, 1);
        q36_metal_bind(e, q, 2);
        q36_metal_bind(e, ikc, 3);
        [e setBytes:&p length:sizeof(p) atIndex:4];
        [e setBytes:&rt length:sizeof(rt) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)s->IQ + 1, (NSUInteger)rows, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return 1;
}
