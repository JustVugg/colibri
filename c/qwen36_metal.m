/* Metal glue for the Qwen3.6 engine.
 *
 * One device, one queue, one library compiled from metal/qwen36.metal (built
 * into the executable), tensors, file-mapping windows, the command batch and
 * one thin wrapper per kernel. A wrapper checks sizes, binds buffers and
 * dispatches; it knows nothing about the model. qwen36.c decides what runs
 * and in which order. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "qwen36_gpu.h"

static const char q36_source[] = {
#embed "metal/qwen36.metal"
    , 0
};

/* Twins of the argument structs in metal/qwen36.metal: same fields, same order. */
typedef struct { uint32_t n, rows, in, out, mode; float eps; uint32_t group, stride, ld; } q36_dense_args;
typedef struct {
    uint32_t rows, hidden, qh, kv, hd, qd, rotary, heads;
    uint32_t vh, kh, kd, vd, convk, convd, pos, cap;
    uint32_t snap;
    uint32_t low;                               /* DeltaNet: the state has its low plane */
} q36_state_args;
typedef struct { uint32_t rows, H, E, K, G, T, shared_gate; } q36_router_args;
typedef struct { uint32_t H, I, K, rows, first, n; uint64_t scales; } q36_expert_args;

enum {
    K_RMSNORM, K_RMSNORM_FAST, K_DOT_I8, K_GATE, K_ADD,
    K_DN_AUX, K_DN_CONV, K_DN_L2, K_DN_DELTA,
    K_ATTN_SPLIT, K_ATTN_ROPE, K_ATTN_KV, K_ATTN, K_ATTN_SCORES, K_ATTN_VALUES,
    K_ROUTER_DOT, K_ROUTER_SOFTMAX, K_ROUTER_SELECT, K_SILU_MUL, K_SHARED_GATE, K_SCALE_ROWS,
    K_EXPERT_GATE_UP, K_EXPERT_DOWN, K_EXPERT_SUM, K_KEEPALIVE, K_DOT_I8_MM,
    K_ATTN_QK, K_ATTN_SOFTMAX, K_ATTN_PV, K_EXPERT_GATE_UP_MM, K_EXPERT_DOWN_MM, K_DN_SCAN, K_ATTN_WEIGHTS,
    K_RMSNORM_ROW, K_ROUTER_SOFTMAX_ROW, K_ROUTER_SELECT_ROW, K_SHARED_GATE_ROW,
    K_DN_DELTA_ROW, K_ATTN_GQA_QK, K_ATTN_GQA_SOFTMAX, K_ATTN_GQA_PV, K_ATTN_GQA_REDUCE,
    K_DOT_I8_VEC4,
    K_ATTN_KV_HALF, K_ATTN_QK_HALF, K_ATTN_PV_HALF, K_ATTN_GQA_QK_HALF, K_ATTN_GQA_PV_HALF,
    K_ATTN_QK8, K_ATTN_QK8_HALF, K_ATTN_PV8, K_ATTN_PV8_HALF, K_RMSNORM_WIDE,
    K_ROUTER_SOFTMAX_FAST, K_DN_CHUNK_GRAM, K_DN_CHUNK_INV, K_DN_CHUNK_UW, K_DN_CHUNK_STEP, K_DN_FOLD, K_DN_AUX4,
    K_EXPERT_GATE_UP_MM4, K_EXPERT_DOWN_MM4, K_DOT_I8_R4, K_ROUTER_DOT2, K_COUNT
};
static const char *const q36_kernel_names[K_COUNT] = {
    "q36_rmsnorm", "q36_rmsnorm_fast", "q36_dot_i8", "q36_gate", "q36_add",
    "q36_dn_aux", "q36_dn_conv", "q36_dn_l2", "q36_dn_delta",
    "q36_attn_split", "q36_attn_rope", "q36_attn_kv", "q36_attn", "q36_attn_scores", "q36_attn_values",
    "q36_router_dot", "q36_router_softmax", "q36_router_select", "q36_silu_mul",
    "q36_shared_gate", "q36_scale_rows", "q36_expert_gate_up", "q36_expert_down", "q36_expert_sum",
    "q36_keepalive", "q36_dot_i8_mm", "q36_attn_qk", "q36_attn_softmax", "q36_attn_pv",
    "q36_expert_gate_up_mm", "q36_expert_down_mm", "q36_dn_scan", "q36_attn_weights",
    "q36_rmsnorm_row", "q36_router_softmax_row", "q36_router_select_row",
    "q36_shared_gate_row",
    "q36_dn_delta_row",
    "q36_attn_gqa_qk", "q36_attn_gqa_softmax", "q36_attn_gqa_pv", "q36_attn_gqa_reduce",
    "q36_dot_i8_vec4",
    "q36_attn_kv_half", "q36_attn_qk_half", "q36_attn_pv_half", "q36_attn_gqa_qk_half", "q36_attn_gqa_pv_half",
    "q36_attn_qk8", "q36_attn_qk8_half", "q36_attn_pv8", "q36_attn_pv8_half",
    "q36_rmsnorm_wide", "q36_router_softmax_fast", "q36_dn_chunk_gram", "q36_dn_chunk_inv", "q36_dn_chunk_uw",
    "q36_dn_chunk_step", "q36_dn_fold", "q36_dn_aux4", "q36_expert_gate_up_mm4", "q36_expert_down_mm4", "q36_dot_i8_r4",
    "q36_router_dot2",
};

static id<MTLDevice> g_device;
static id<MTLBuffer> g_dn_chunk_scratch;       /* q36_dn_chunk: eight prepared chunks */
static id<MTLCommandQueue> g_queue;
static id<MTLComputePipelineState> g_pipe[K_COUNT];
static id<MTLCommandBuffer> g_batch;           /* open, not committed yet */
static id<MTLComputeCommandEncoder> g_encoder;
static id<MTLCommandBuffer> g_last;            /* last committed batch */
static id<MTLCommandBuffer> g_marks[4];        /* q36_gpu_mark's tickets */
static unsigned g_mark_next;
static atomic_int g_failed;                    /* a committed batch ended in error */
static int g_exact;                            /* QWEN36_METAL_EXACT=1: exact kernels everywhere */
/* Batches whose GPU time goes to a timer, up to the next sync. A fast prompt
 * layer commits up to about 260 (its windows, the shared expert, each group
 * of eight experts); past the limit a batch is not timed, and sync says so. */
#define Q36_TIMED_CAPACITY 512
static id<MTLCommandBuffer> g_timed[Q36_TIMED_CAPACITY];
static double *g_timed_ms[Q36_TIMED_CAPACITY];
static uint64_t g_timed_dropped;
static int g_ntimed;

struct q36_tensor {
    __strong id<MTLBuffer> buffer;
    size_t offset, bytes;
};

static char g_device_name[256] = "none";

/* The kernel library on device, compiled from the source built into this
 * object; nil after printing why it failed. */
static id<MTLLibrary> q36_compile(id<MTLDevice> device) {
    /* The kernels are written against IEEE float without fast-math
     * shortcuts; the fixed results of the exact kernels depend on it. */
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
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:q36_source]
                                                  options:options error:&error];
    if (!library)
        fprintf(stderr, "[qwen36-metal] kernel compilation failed: %s\n",
                error ? error.localizedDescription.UTF8String : "unknown error");
    return library;
}

const char *q36_gpu_default_device_name(void) {
    static char name[256];
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        snprintf(name, sizeof name, "%s", device ? device.name.UTF8String : "");
    }
    return name;
}

int q36_gpu_compile_check(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = device ? q36_compile(device) : nil;
        if (!library) return 0;
        for (int i = 0; i < K_COUNT; i++)
            if (![library newFunctionWithName:@(q36_kernel_names[i])]) {
                fprintf(stderr, "[qwen36-metal] kernel %s missing from the library\n", q36_kernel_names[i]);
                return 0;
            }
    }
    return 1;
}

int q36_gpu_init(void) {
    if (g_device) return 1;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) return 0;
        NSError *error = nil;
        id<MTLLibrary> library = q36_compile(device);
        if (!library) return 0;
        for (int i = 0; i < K_COUNT; i++) {
            id<MTLFunction> fn = [library newFunctionWithName:@(q36_kernel_names[i])];
            g_pipe[i] = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
            /* The SIMD reductions assume 32-lane groups. */
            if (!g_pipe[i] || g_pipe[i].threadExecutionWidth != 32) {
                fprintf(stderr, "[qwen36-metal] pipeline %s unavailable\n", q36_kernel_names[i]);
                for (int j = 0; j <= i; j++) g_pipe[j] = nil;
                return 0;
            }
        }
        g_queue = [device newCommandQueue];
        if (!g_queue) return 0;
        g_device = device;
        snprintf(g_device_name, sizeof g_device_name, "%s", device.name.UTF8String);
        const char *exact = getenv("QWEN36_METAL_EXACT");
        g_exact = exact && !strcmp(exact, "1");
    }
    return 1;
}

static void q36_keepalive_stop(void);

void q36_gpu_cleanup(void) {
    q36_gpu_sync();
    q36_keepalive_stop();
    @autoreleasepool {
        for (int i = 0; i < K_COUNT; i++) g_pipe[i] = nil;
        g_dn_chunk_scratch = nil;
        g_queue = nil;
        g_device = nil;
    }
    snprintf(g_device_name, sizeof g_device_name, "none");
}

void q36_gpu_set_exact(int on) { g_exact = on; }
int q36_gpu_is_exact(void) { return g_exact != 0; }

const char *q36_gpu_device_name(void) {
    return g_device_name;   /* copied at init: no autoreleased string for a C caller */
}

/* ---- tensors ---------------------------------------------------------- */

static q36_tensor *q36_tensor_make(id<MTLBuffer> buffer, size_t offset, size_t bytes) {
    q36_tensor *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->buffer = buffer;
    t->offset = offset;
    t->bytes = bytes;
    return t;
}

q36_tensor *q36_tensor_new(size_t bytes) {
    if (!g_device || !bytes) return NULL;
    /* C callers need not have an autorelease pool. The returned descriptor
     * takes its own strong buffer reference before framework temporaries are
     * drained; failed descriptor allocation releases the buffer in this pool. */
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) return NULL;
        memset(buffer.contents, 0, bytes);
        return q36_tensor_make(buffer, 0, bytes);
    }
}

q36_tensor *q36_tensor_upload(const void *data, size_t bytes) {
    q36_tensor *t = q36_tensor_new(bytes);
    if (t && data) memcpy(q36_tensor_data(t), data, bytes);
    return t;
}

q36_tensor *q36_tensor_view(q36_tensor *base, size_t offset, size_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    return q36_tensor_make(base->buffer, base->offset + offset, bytes);
}

void q36_tensor_free(q36_tensor *t) {
    if (!t) return;
    @autoreleasepool { t->buffer = nil; }   /* C callers need not have an autorelease pool */
    free(t);
}

void *q36_tensor_data(q36_tensor *t) {
    return t ? (char *)t->buffer.contents + t->offset : NULL;
}

size_t q36_tensor_bytes(const q36_tensor *t) {
    return t ? t->bytes : 0;
}

/* ---- file mappings ------------------------------------------------------ */

#define Q36_MAX_WINDOWS 1024
#define Q36_WINDOW_BYTES ((size_t)64 << 20)

struct q36_map {
    const char *base;
    size_t bytes, extent, page;
    int count;
    size_t offset[Q36_MAX_WINDOWS], length[Q36_MAX_WINDOWS];
    __strong id<MTLBuffer> window[Q36_MAX_WINDOWS];
};

q36_map *q36_map_new(const void *base, size_t bytes) {
    long page = sysconf(_SC_PAGESIZE);
    if (!g_device || !base || !bytes || page <= 0 || ((uintptr_t)base % (size_t)page)) return NULL;
    q36_map *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->base = base;
    m->bytes = bytes;
    m->page = (size_t)page;
    m->extent = (bytes + m->page - 1) & ~(m->page - 1);
    return m;
}

/* Windows start on multiples of half their size, so any tensor up to 32 MiB
 * fits in one of them; a larger tensor gets a window of its own. */
q36_tensor *q36_map_view(q36_map *m, size_t offset, size_t bytes) {
    if (!m || !bytes || offset > m->bytes || bytes > m->bytes - offset) return NULL;
    for (int i = 0; i < m->count; i++) {
        if (offset >= m->offset[i] && offset - m->offset[i] <= m->length[i] &&
            bytes <= m->length[i] - (offset - m->offset[i]))
            return q36_tensor_make(m->window[i], offset - m->offset[i], bytes);
    }
    if (m->count == Q36_MAX_WINDOWS) return NULL;
    size_t cap = ((size_t)g_device.maxBufferLength < Q36_WINDOW_BYTES ?
                  (size_t)g_device.maxBufferLength : Q36_WINDOW_BYTES) & ~(m->page - 1);
    size_t stride = (cap / 2) & ~(m->page - 1);
    if (stride < m->page) stride = m->page;
    size_t begin = offset / stride * stride;
    size_t end = begin + cap < m->extent ? begin + cap : m->extent;
    if (bytes > end - offset) {
        begin = offset & ~(m->page - 1);
        end = (offset + bytes + m->page - 1) & ~(m->page - 1);
    }
    if (end - begin > (size_t)g_device.maxBufferLength) return NULL;
    id<MTLBuffer> window = [g_device newBufferWithBytesNoCopy:(void *)(m->base + begin) length:end - begin
                                                      options:MTLResourceStorageModeShared deallocator:nil];
    if (!window) return NULL;
    m->offset[m->count] = begin;
    m->length[m->count] = end - begin;
    m->window[m->count] = window;
    m->count++;
    return q36_tensor_make(window, offset - begin, bytes);
}

void q36_map_free(q36_map *m) {
    if (!m) return;
    for (int i = 0; i < m->count; i++) m->window[i] = nil;
    free(m);
}

/* ---- the command batch -------------------------------------------------- */

/* One serial encoder per batch: each dispatch sees the results of the ones
 * encoded before it, so no explicit barriers are needed. */
static int g_concurrent;                        /* q36_gpu_concurrent */

static id<MTLComputeCommandEncoder> q36_encoder(int kernel) {
    if (!g_encoder) {
        if (!g_batch) g_batch = [g_queue commandBuffer];
        g_encoder = [g_batch computeCommandEncoderWithDispatchType:g_concurrent ? MTLDispatchTypeConcurrent
                                                                                : MTLDispatchTypeSerial];
    }
    [g_encoder setComputePipelineState:g_pipe[kernel]];
    return g_encoder;
}

void q36_gpu_concurrent(int on) {
    on = on != 0;
    if (on == g_concurrent) return;
    if (g_encoder) {                            /* encoders of one batch run in order */
        [g_encoder endEncoding];
        g_encoder = nil;
    }
    g_concurrent = on;
}

void q36_gpu_barrier(void) {
    if (g_encoder && g_concurrent) [g_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

int q36_gpu_flush(void) {
    if (g_encoder) {
        [g_encoder endEncoding];
        g_encoder = nil;
    }
    if (g_batch) {
        [g_batch addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            if (cb.status == MTLCommandBufferStatusError) atomic_store(&g_failed, 1);
        }];
        [g_batch commit];
        g_last = g_batch;
        g_batch = nil;
    }
    return !atomic_load(&g_failed);
}

/* The GPU reports when it started and finished each batch: the time is
 * measured without waiting for anything, and read at the next sync. */
int q36_gpu_flush_timed(double *ms) {
    if (g_batch) {
        if (g_ntimed < Q36_TIMED_CAPACITY) {
            g_timed[g_ntimed] = g_batch;
            g_timed_ms[g_ntimed++] = ms;
        } else g_timed_dropped++;
    }
    return q36_gpu_flush();
}

int q36_gpu_mark(void) {
    if (!q36_gpu_flush()) return 0;
    unsigned t = g_mark_next++ % 4;
    g_marks[t] = g_last;
    return (int)t + 1;
}

int q36_gpu_wait_mark(int mark) {
    if (mark < 1 || mark > 4) return 0;
    @autoreleasepool {
        if (g_marks[mark - 1]) {
            [g_marks[mark - 1] waitUntilCompleted];
            g_marks[mark - 1] = nil;
        }
    }
    return !atomic_load(&g_failed);
}

/* A queue runs its batches in order: waiting for the last one waits for all. */
int q36_gpu_sync(void) {
    @autoreleasepool {
        q36_gpu_flush();
        if (g_last) {
            [g_last waitUntilCompleted];
            g_last = nil;
        }
        for (int i = 0; i < g_ntimed; i++) {
            [g_timed[i] waitUntilCompleted];
            *g_timed_ms[i] += (g_timed[i].GPUEndTime - g_timed[i].GPUStartTime) * 1e3;
            g_timed[i] = nil;
        }
        g_ntimed = 0;
        if (g_timed_dropped) {
            fprintf(stderr, "[qwen36-metal] COLI_TIMERS: %llu batches not timed (more than %d before a sync)\n",
                    (unsigned long long)g_timed_dropped, Q36_TIMED_CAPACITY);
            g_timed_dropped = 0;
        }
    }
    return !atomic_load(&g_failed);
}

/* ---- keep-alive ----------------------------------------------------------- */

/* Decode stops the GPU at every layer (the choices go to the CPU, experts
 * come from disk) and the power manager takes those gaps for idleness: the
 * clocks drop and the same kernels run up to half as fast as alone. While
 * decode runs, a thread keeps one threadgroup of q36_keepalive going on a
 * second queue, as ds4 does for its tensor-parallel decode; it sleeps once
 * the deadline passes. QWEN36_METAL_KEEPALIVE=0 turns it off, to measure. */
/* About 0.1 ms per dispatch on an M1. The engine's batches can wait behind a
 * dispatch of the other queue: with 3 ms dispatches that cost 95 ms per token,
 * at 0.1 ms nothing measurable, and the clocks stay up as well. */
#define Q36_KEEPALIVE_ITERS 3000u
static pthread_mutex_t g_ka_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_ka_wake = PTHREAD_COND_INITIALIZER;
static pthread_t g_ka_thread;
static id<MTLCommandQueue> g_ka_queue;
static id<MTLBuffer> g_ka_buffer;
static int g_ka_state, g_ka_stop;               /* state: 0 not started, 1 running, -1 off */
static uint64_t g_ka_runs;                      /* dispatches completed, under g_ka_mutex */
static double g_ka_until;                       /* monotonic seconds */

static double q36_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *q36_keepalive_thread(void *arg) {
    (void)arg;
    const uint32_t iters = Q36_KEEPALIVE_ITERS;
    pthread_mutex_lock(&g_ka_mutex);
    for (;;) {
        while (!g_ka_stop && q36_now() >= g_ka_until) pthread_cond_wait(&g_ka_wake, &g_ka_mutex);
        if (g_ka_stop) break;
        pthread_mutex_unlock(&g_ka_mutex);
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [g_ka_queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:g_pipe[K_KEEPALIVE]];
            [e setBuffer:g_ka_buffer offset:0 atIndex:0];
            [e setBytes:&iters length:sizeof(iters) atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        pthread_mutex_lock(&g_ka_mutex);
        g_ka_runs++;
    }
    pthread_mutex_unlock(&g_ka_mutex);
    return NULL;
}

void q36_gpu_keepalive(double seconds) {
    if (!g_device) return;
    pthread_mutex_lock(&g_ka_mutex);
    if (!g_ka_state) {
        const char *off = getenv("QWEN36_METAL_KEEPALIVE");
        g_ka_state = -1;
        if (!off || strcmp(off, "0")) {
            @autoreleasepool {
                g_ka_queue = [g_device newCommandQueue];
                g_ka_buffer = [g_device newBufferWithLength:256 * sizeof(float) options:MTLResourceStorageModeShared];
            }
            if (g_ka_queue && g_ka_buffer && !pthread_create(&g_ka_thread, NULL, q36_keepalive_thread, NULL))
                g_ka_state = 1;
        }
    }
    g_ka_until = seconds > 0 ? q36_now() + seconds : 0;
    pthread_cond_signal(&g_ka_wake);
    pthread_mutex_unlock(&g_ka_mutex);
}

uint64_t q36_gpu_keepalive_runs(void) {
    pthread_mutex_lock(&g_ka_mutex);
    uint64_t n = g_ka_runs;
    pthread_mutex_unlock(&g_ka_mutex);
    return n;
}

static void q36_keepalive_stop(void) {
    pthread_mutex_lock(&g_ka_mutex);
    int running = g_ka_state == 1;
    g_ka_stop = 1;
    pthread_cond_signal(&g_ka_wake);
    pthread_mutex_unlock(&g_ka_mutex);
    if (running) pthread_join(g_ka_thread, NULL);
    g_ka_state = g_ka_stop = 0;
    @autoreleasepool {
        g_ka_queue = nil;
        g_ka_buffer = nil;
    }
}

/* ---- wrappers ------------------------------------------------------------- */

static int q36_fits(const q36_tensor *t, uint64_t bytes) {
    return t && bytes <= t->bytes;
}

static void q36_bind(id<MTLComputeCommandEncoder> e, q36_tensor *t, NSUInteger index) {
    [e setBuffer:t->buffer offset:t->offset atIndex:index];
}

static void q36_bind_at(id<MTLComputeCommandEncoder> e, q36_tensor *t, uint64_t skip, NSUInteger index) {
    [e setBuffer:t->buffer offset:t->offset + skip atIndex:index];
}

static void q36_threads(id<MTLComputeCommandEncoder> e, uint64_t n) {
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(n < 64 ? n : 64, 1, 1)];
}

static void q36_groups(id<MTLComputeCommandEncoder> e, uint64_t x, uint64_t y) {
    [e dispatchThreadgroups:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static q36_state_args q36_state(const q36_shape *s) {
    return (q36_state_args){
        (uint32_t)s->rows, (uint32_t)s->hidden, (uint32_t)s->q_heads, (uint32_t)s->kv_heads,
        (uint32_t)s->head_dim, (uint32_t)s->q_head_dim, (uint32_t)s->rotary, 0,
        (uint32_t)s->v_heads, (uint32_t)s->k_heads, (uint32_t)s->k_dim, (uint32_t)s->v_dim,
        (uint32_t)s->conv_kernel, (uint32_t)s->conv_dim, (uint32_t)s->pos, (uint32_t)s->cap,
        s->snap > 0 ? (uint32_t)s->snap - 1 : ~0u, 1};
}

#define F4 (uint64_t)sizeof(float)
/* Batches of at most Q36_ROW_ROWS rows (generation) run the norm, the router's
 * softmax and top-k and the shared gate as one threadgroup per row (the *_row
 * kernels: the same bits, loads in parallel); vectors up to Q36_ROW_MAX floats. */
#define Q36_ROW_ROWS 8
#define Q36_ROW_MAX 4096
static void q36_rows(id<MTLComputeCommandEncoder> e, uint64_t rows) {
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int q36_rmsnorm(q36_tensor *y, q36_tensor *x, q36_tensor *w, int rows, int dim, float eps, int plain) {
    if (rows < 1 || dim < 1 || !q36_fits(x, (uint64_t)rows * dim * F4) ||
        !q36_fits(y, (uint64_t)rows * dim * F4) || !q36_fits(w, (uint64_t)dim * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {0, (uint32_t)rows, (uint32_t)dim, 0, plain ? 1u : 0u, eps, 0, 0, 0};
        /* The FAST kernel supports every positive dimension, including tails.
         * Keep the original exact selection as a fallback and for exact mode. */
        int fast = !g_exact && g_pipe[K_RMSNORM_FAST].maxTotalThreadsPerThreadgroup >= 32;
        int row = rows <= Q36_ROW_ROWS && dim <= Q36_ROW_MAX;
        /* a few long rows (decode): a threadgroup per row; many rows: a SIMD group each */
        int wide = fast && rows <= Q36_ROW_ROWS && dim >= 1024 && dim % 4 == 0 && (x->offset | y->offset) % 16 == 0 &&
                   g_pipe[K_RMSNORM_WIDE].maxTotalThreadsPerThreadgroup >= 256;
        id<MTLComputeCommandEncoder> e = q36_encoder(wide ? K_RMSNORM_WIDE : fast ? K_RMSNORM_FAST :
                                                     (row ? K_RMSNORM_ROW : K_RMSNORM));
        q36_bind(e, x, 0); q36_bind(e, w, 1); q36_bind(e, y, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        if (wide) q36_rows(e, (uint64_t)rows);
        else if (fast) q36_groups(e, (uint64_t)rows, 1);
        else if (row) q36_rows(e, (uint64_t)rows);
        else q36_threads(e, (uint64_t)rows);
    }
    return 1;
}

int q36_dot_i8(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out) {
    if (rows < 1 || in < 1 || out < 1 || !q36_fits(q, (uint64_t)out * in) || !q36_fits(sc, (uint64_t)out * F4) ||
        !q36_fits(x, (uint64_t)rows * in * F4) || !q36_fits(y, (uint64_t)rows * out * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {0, (uint32_t)rows, (uint32_t)in, (uint32_t)out, 0, 0, 0, 0, 0};
        if (!g_exact && rows > 8 && out % 32 == 0 && in % 32 == 0 && x->offset % 16 == 0 && y->offset % 8 == 0) {
            /* prompt batches: the matrix units, not bit-identical (q36_dot_i8_mm) */
            id<MTLComputeCommandEncoder> e = q36_encoder(K_DOT_I8_MM);
            q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
            [e setBytes:&a length:sizeof(a) atIndex:4];
            [e setThreadgroupMemoryLength:32 * 32 * F4 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 31) / 32, (NSUInteger)(out + 127) / 128, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return 1;
        }
        int vector = !g_exact && in % 4 == 0, four = !vector && out % 4 == 0;
        id<MTLComputeCommandEncoder> e = q36_encoder(vector ? K_DOT_I8_VEC4 : four ? K_DOT_I8_R4 : K_DOT_I8);
        q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
        [e setBytes:&a length:sizeof(a) atIndex:4];
        q36_groups(e, (uint64_t)rows * out / (four ? 4 : 1), 1);
    }
    return 1;
}

int q36_dot_i8_heads(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out, int group,
                     int stride) {
    /* the weight rows the outputs take: up to the last head's part */
    uint64_t wrows = group > 0 && out % group == 0 ? (uint64_t)(out / group - 1) * stride + group : 0;
    if (g_exact || rows <= 8 || in < 32 || in % 32 || out < 32 || out % 32 || group < 32 || group % 32 || out % group ||
        stride < group || x->offset % 16 || y->offset % 8 || !q36_fits(q, wrows * in) || !q36_fits(sc, wrows * F4) ||
        !q36_fits(x, (uint64_t)rows * in * F4) || !q36_fits(y, (uint64_t)rows * out * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {0, (uint32_t)rows, (uint32_t)in, (uint32_t)out, 0, 0, (uint32_t)group, (uint32_t)stride, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_DOT_I8_MM);
        q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
        [e setBytes:&a length:sizeof(a) atIndex:4];
        [e setThreadgroupMemoryLength:32 * 32 * F4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 31) / 32, (NSUInteger)(out + 127) / 128, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    return 1;
}

int q36_dot_i8_cols(q36_tensor *y, int ldy, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out) {
    if (g_exact || rows <= 8 || in < 32 || in % 32 || out < 32 || out % 32 || ldy < out || ldy % 2 || x->offset % 16 ||
        y->offset % 8 || !q36_fits(q, (uint64_t)out * in) || !q36_fits(sc, (uint64_t)out * F4) ||
        !q36_fits(x, (uint64_t)rows * in * F4) || !q36_fits(y, ((uint64_t)(rows - 1) * ldy + out) * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {0, (uint32_t)rows, (uint32_t)in, (uint32_t)out, 0, 0, 0, 0, (uint32_t)ldy};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_DOT_I8_MM);
        q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
        [e setBytes:&a length:sizeof(a) atIndex:4];
        [e setThreadgroupMemoryLength:32 * 32 * F4 atIndex:0];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(rows + 31) / 32, (NSUInteger)(out + 127) / 128, 1)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    return 1;
}

int q36_gate(q36_tensor *y, q36_tensor *x, q36_tensor *z, int n, int silu) {
    if (n < 1 || !q36_fits(x, n * F4) || !q36_fits(z, n * F4) || !q36_fits(y, n * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {(uint32_t)n, 0, 0, 0, silu ? 1u : 0u, 0, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_GATE);
        q36_bind(e, x, 0); q36_bind(e, z, 1); q36_bind(e, y, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        q36_threads(e, (uint64_t)n);
    }
    return 1;
}

int q36_add(q36_tensor *y, q36_tensor *x, q36_tensor *r, int n) {
    if (n < 1 || !q36_fits(x, n * F4) || !q36_fits(r, n * F4) || !q36_fits(y, n * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {(uint32_t)n, 0, 0, 0, 0, 0, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ADD);
        q36_bind(e, x, 0); q36_bind(e, r, 1); q36_bind(e, y, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        q36_threads(e, (uint64_t)n);
    }
    return 1;
}

int q36_dn_aux(q36_tensor *decay, q36_tensor *beta, q36_tensor *x, q36_tensor *a, q36_tensor *b,
               q36_tensor *alog, q36_tensor *dt, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, vh = (uint64_t)s->v_heads, H = (uint64_t)s->hidden;
    if (!rows || !vh || !q36_fits(x, rows * H * F4) || !q36_fits(a, vh * H * F4) || !q36_fits(b, vh * H * F4) ||
        !q36_fits(alog, vh * F4) || !q36_fits(dt, vh * F4) || !q36_fits(decay, rows * vh * F4) ||
        !q36_fits(beta, rows * vh * F4)) return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        /* batches of 4 rows or more: 4 rows per SIMD group, the same bits (q36_dn_aux4) */
        int four = rows >= 4;
        id<MTLComputeCommandEncoder> e = q36_encoder(four ? K_DN_AUX4 : K_DN_AUX);
        q36_bind(e, x, 0); q36_bind(e, a, 1); q36_bind(e, b, 2); q36_bind(e, alog, 3);
        q36_bind(e, dt, 4); q36_bind(e, decay, 5); q36_bind(e, beta, 6);
        [e setBytes:&p length:sizeof(p) atIndex:7];
        q36_groups(e, four ? (rows + 3) / 4 * vh : rows * vh, 1);
    }
    return 1;
}

int q36_dn_conv(q36_tensor *y, q36_tensor *ring, q36_tensor *x, q36_tensor *w, q36_tensor *snap,
                const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, cd = (uint64_t)s->conv_dim, ck = (uint64_t)s->conv_kernel;
    if (!rows || !cd || ck < 2 || !q36_fits(x, rows * cd * F4) || !q36_fits(w, cd * ck * F4) ||
        !q36_fits(ring, cd * (ck - 1) * F4) || !q36_fits(y, rows * cd * F4) || s->snap < 0 ||
        (uint64_t)s->snap > rows || (s->snap && !q36_fits(snap, cd * (ck - 1) * F4))) return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        id<MTLComputeCommandEncoder> e = q36_encoder(K_DN_CONV);
        q36_bind(e, x, 0); q36_bind(e, w, 1); q36_bind(e, ring, 2); q36_bind(e, y, 3);
        [e setBytes:&p length:sizeof(p) atIndex:4];
        q36_bind(e, s->snap ? snap : ring, 5);
        q36_threads(e, cd);
    }
    return 1;
}

int q36_dn_l2(q36_tensor *y, q36_tensor *x, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, cd = (uint64_t)s->conv_dim;
    if (!rows || !s->k_heads || !q36_fits(x, rows * cd * F4) || !q36_fits(y, rows * cd * F4)) return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        id<MTLComputeCommandEncoder> e = q36_encoder(K_DN_L2);
        q36_bind(e, x, 0); q36_bind(e, y, 1);
        [e setBytes:&p length:sizeof(p) atIndex:2];
        q36_groups(e, rows * 2 * (uint64_t)s->k_heads, 1);
    }
    return 1;
}

/* Twin of q36_dn_chunk_args in metal/qwen36.metal. */
typedef struct { q36_state_args s; uint32_t first[8], n[8]; uint32_t chunk, capture; } q36_dn_chunk_args;

/* A segment of at most 8 rows of the prompt's DeltaNet chunks: the scan. */
static void q36_dn_short(q36_tensor *out, q36_tensor *state, q36_tensor *qk, q36_tensor *conv, q36_tensor *decay,
                         q36_tensor *beta, q36_tensor *snap, const q36_shape *s, int low, uint32_t first,
                         uint32_t n, int capture) {
    q36_state_args p = q36_state(s);
    p.rows = n;
    p.low = (uint32_t)low;
    p.snap = capture ? n - 1 : ~0u;
    id<MTLComputeCommandEncoder> e = q36_encoder(K_DN_SCAN);
    uint64_t vector_skip = (uint64_t)first * s->conv_dim * F4, scalar_skip = (uint64_t)first * s->v_heads * F4;
    q36_bind_at(e, qk, vector_skip, 0); q36_bind_at(e, conv, vector_skip, 1);
    q36_bind_at(e, decay, scalar_skip, 2); q36_bind_at(e, beta, scalar_skip, 3);
    q36_bind(e, state, 4);
    q36_bind_at(e, out, (uint64_t)first * s->v_heads * 128 * F4, 5);
    [e setBytes:&p length:sizeof(p) atIndex:6];
    q36_bind(e, snap, 7);
    [e dispatchThreadgroups:MTLSizeMake(8, (NSUInteger)s->v_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

/* The prompt's DeltaNet in chunks of 32 rows (metal/qwen36.metal, q36_dn_chunk_*), cut at the
 * snapshot; segments of at most 8 rows run the scan. The state-independent parts of eight
 * chunks take three dispatches, then each chunk one. A state with its low plane is folded into
 * the high one first. The scratch holds eight prepared chunks: 10 MiB for 32 value heads. The
 * private scratch belongs to the one serial Metal backend, not to an expert cache. Encoded
 * commands retain an older buffer if this one grows. */
static int q36_dn_chunk(q36_tensor *out, q36_tensor *state, q36_tensor *qk, q36_tensor *conv,
                        q36_tensor *decay, q36_tensor *beta, q36_tensor *snap, const q36_shape *s, int low) {
    enum { SEGS = 96, STRIDE = 10304 };
    uint32_t first_of[SEGS], n_of[SEGS], capture_of[SEGS];
    int nseg = 0;
    for (uint32_t first = 0; first < (uint32_t)s->rows;) {
        uint32_t n = (uint32_t)s->rows - first;
        if (n > 32) n = 32;
        if (s->snap > 0 && (uint32_t)s->snap > first && (uint32_t)s->snap < first + n) n = (uint32_t)s->snap - first;
        if (nseg == SEGS) return 0;
        first_of[nseg] = first; n_of[nseg] = n;
        capture_of[nseg++] = s->snap > 0 && first + n == (uint32_t)s->snap;
        first += n;
    }
    const int kernels[5] = {K_DN_CHUNK_GRAM, K_DN_CHUNK_INV, K_DN_CHUNK_UW, K_DN_CHUNK_STEP, K_DN_FOLD};
    const NSUInteger shared[5] = {2112 * F4, 2048 * F4, 0, 1024 * F4, 0};
    for (int k = 0; k < 5; k++) {
        id<MTLComputePipelineState> pipe = g_pipe[kernels[k]];
        if (!pipe || pipe.threadExecutionWidth != 32 || pipe.maxTotalThreadsPerThreadgroup < 128 ||
            shared[k] + pipe.staticThreadgroupMemoryLength > g_device.maxThreadgroupMemoryLength) return 0;
    }
    uint64_t bytes = (uint64_t)8 * s->v_heads * STRIDE * F4;
    @autoreleasepool {
        if (!g_dn_chunk_scratch || g_dn_chunk_scratch.length < bytes) {
            id<MTLBuffer> next = [g_device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModePrivate];
            if (!next) return 0;                /* nothing encoded: the caller runs the scan */
            g_dn_chunk_scratch = next;
        }
        if (low) {
            uint32_t cells = (uint32_t)(s->v_heads * 128 * 128);
            id<MTLComputeCommandEncoder> e = q36_encoder(K_DN_FOLD);
            q36_bind(e, state, 0);
            [e setBytes:&cells length:sizeof(cells) atIndex:1];
            q36_threads(e, cells);
        }
        q36_dn_chunk_args a = {q36_state(s), {0}, {0}, 0, 0};
        a.s.low = (uint32_t)low;                /* the update keeps writing the low planes zero */
        for (int i = 0; i < nseg;) {
            if (n_of[i] <= 8) {
                q36_dn_short(out, state, qk, conv, decay, beta, snap, s, low, first_of[i], n_of[i], capture_of[i]);
                i++;
                continue;
            }
            /* the next eight chunks of at least 9 rows: prepared at once */
            int last = i, count = 0;
            for (int j = i; j < nseg && count < 8; j++)
                if (n_of[j] > 8) { a.first[count] = first_of[j]; a.n[count++] = n_of[j]; last = j; }
            id<MTLComputeCommandEncoder> e = q36_encoder(K_DN_CHUNK_GRAM);
            q36_bind(e, qk, 0); q36_bind(e, decay, 1); q36_bind(e, beta, 2);
            [e setBuffer:g_dn_chunk_scratch offset:0 atIndex:3];
            [e setBytes:&a length:sizeof(a) atIndex:4];
            [e setThreadgroupMemoryLength:shared[0] atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)count, (NSUInteger)s->v_heads, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            e = q36_encoder(K_DN_CHUNK_INV);
            [e setBuffer:g_dn_chunk_scratch offset:0 atIndex:0];
            [e setBytes:&a length:sizeof(a) atIndex:1];
            [e setThreadgroupMemoryLength:shared[1] atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)count, (NSUInteger)s->v_heads, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            e = q36_encoder(K_DN_CHUNK_UW);
            q36_bind(e, qk, 0); q36_bind(e, conv, 1); q36_bind(e, beta, 2);
            [e setBuffer:g_dn_chunk_scratch offset:0 atIndex:3];
            [e setBytes:&a length:sizeof(a) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)count, (NSUInteger)s->v_heads, 2)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            for (int j = i, slot = 0; j <= last; j++) {
                if (n_of[j] <= 8) {             /* a short segment between them */
                    q36_dn_short(out, state, qk, conv, decay, beta, snap, s, low, first_of[j], n_of[j], capture_of[j]);
                    continue;
                }
                a.chunk = (uint32_t)slot++;
                a.capture = capture_of[j];
                e = q36_encoder(K_DN_CHUNK_STEP);
                q36_bind(e, qk, 0); q36_bind(e, state, 1); q36_bind(e, out, 2); q36_bind(e, snap, 3);
                [e setBuffer:g_dn_chunk_scratch offset:0 atIndex:4];
                [e setBytes:&a length:sizeof(a) atIndex:5];
                [e setThreadgroupMemoryLength:shared[3] atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake(4, (NSUInteger)s->v_heads, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            i = last + 1;
        }
    }
    return 1;
}

int q36_dn_high_only(const q36_shape *s) {
    return !g_exact && s->k_dim % 32 == 0 && s->k_dim <= 128 && s->v_dim % 4 == 0 && s->conv_dim % 4 == 0;
}

int q36_dn_delta(q36_tensor *out, q36_tensor *state, q36_tensor *qk, q36_tensor *conv,
                 q36_tensor *decay, q36_tensor *beta, q36_tensor *snap, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, vh = (uint64_t)s->v_heads, kd = (uint64_t)s->k_dim, vd = (uint64_t)s->v_dim;
    uint64_t cd = (uint64_t)s->conv_dim;
    if (!rows || !vh || !s->k_heads || vh % (uint64_t)s->k_heads || kd < 1 || kd > 128 || !vd ||
        !q36_fits(qk, rows * cd * F4) || !q36_fits(conv, rows * cd * F4) || !q36_fits(decay, rows * vh * F4) ||
        !q36_fits(beta, rows * vh * F4) || !q36_fits(state, vh * kd * vd * F4) ||
        !q36_fits(out, rows * vh * vd * F4) || s->snap < 0 || (uint64_t)s->snap > rows) return 0;
    /* The low plane of the state: the exact kernels need it; in fast mode the
     * state may have the high plane only, which the fast kernels keep. */
    int low = q36_fits(state, 2 * vh * kd * vd * F4);
    if (s->snap && !q36_fits(snap, (low ? 2 : 1) * vh * kd * vd * F4)) return 0;
    if (!s->snap) snap = state;                 /* bound, never written */
    /* fast mode: q36_dn_scan, not bit-identical; its float4 accesses need
     * 16-byte alignment */
    int fast = q36_dn_high_only(s) &&
               (qk->offset | conv->offset | state->offset | out->offset | snap->offset) % 16 == 0;
    /* Prompt batches in fast mode: chunks of 32 rows on the matrix units
     * (q36_dn_chunk). */
    if (!fast && !low) return 0;
    if (fast && rows > 8 && kd == 128 && vd == 128 && vh <= 64 &&
        q36_dn_chunk(out, state, qk, conv, decay, beta, snap, s, low)) return 1;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        p.low = (uint32_t)low;
        id<MTLComputeCommandEncoder> e = q36_encoder(fast ? K_DN_SCAN : rows == 1 ? K_DN_DELTA_ROW : K_DN_DELTA);
        q36_bind(e, qk, 0); q36_bind(e, conv, 1); q36_bind(e, decay, 2); q36_bind(e, beta, 3);
        q36_bind(e, state, 4); q36_bind(e, out, 5);
        [e setBytes:&p length:sizeof(p) atIndex:6];
        q36_bind(e, snap, 7);
        if (fast)
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(vd + 15) / 16, (NSUInteger)vh, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        else
            q36_groups(e, (vd + 31) / 32, vh);
    }
    return 1;
}

int q36_attn_split(q36_tensor *query, q36_tensor *gate, q36_tensor *key, q36_tensor *q, q36_tensor *k,
                   const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, qh = (uint64_t)s->q_heads, kv = (uint64_t)s->kv_heads;
    uint64_t hd = (uint64_t)s->head_dim, qd = (uint64_t)s->q_head_dim;
    if (!rows || !qh || !kv || kv > qh || !q36_fits(q, rows * qh * qd * F4) || !q36_fits(k, rows * kv * hd * F4) ||
        !q36_fits(query, rows * qh * hd * F4) || !q36_fits(gate, rows * qh * hd * F4) ||
        !q36_fits(key, rows * kv * hd * F4)) return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ATTN_SPLIT);
        q36_bind(e, q, 0); q36_bind(e, query, 1); q36_bind(e, gate, 2); q36_bind(e, k, 3); q36_bind(e, key, 4);
        [e setBytes:&p length:sizeof(p) atIndex:5];
        q36_threads(e, rows * qh * hd);
    }
    return 1;
}

int q36_attn_rope(q36_tensor *x, q36_tensor *rot, int heads, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, half = (uint64_t)s->rotary / 2;
    if (!rows || heads < 1 || !half || s->rotary > s->head_dim ||
        !q36_fits(x, rows * (uint64_t)heads * (uint64_t)s->head_dim * F4) || !q36_fits(rot, rows * half * 2 * F4))
        return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        p.heads = (uint32_t)heads;
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ATTN_ROPE);
        q36_bind(e, x, 0); q36_bind(e, rot, 1);
        [e setBytes:&p length:sizeof(p) atIndex:2];
        q36_threads(e, rows * (uint64_t)heads * half);
    }
    return 1;
}

/* The KV cache holds floats, or halves where q36_kv_half: its bytes tell. */
static uint64_t q36_kv_bytes(const q36_tensor *kc, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->kv_heads * (uint64_t)s->cap * (uint64_t)s->head_dim;
    return q36_fits(kc, rows * F4) ? F4 : 2;
}

int q36_kv_half(const q36_shape *s) {
    return !g_exact && s->kv_heads > 0 && s->q_heads == 8 * s->kv_heads && s->head_dim % 64 == 0 &&
           s->head_dim <= 256 &&
           g_pipe[K_ATTN_GQA_QK_HALF].maxTotalThreadsPerThreadgroup >= 128 &&
           g_pipe[K_ATTN_GQA_SOFTMAX].maxTotalThreadsPerThreadgroup >= 256 &&
           g_pipe[K_ATTN_GQA_PV_HALF].maxTotalThreadsPerThreadgroup >= 128;
}

int q36_attn_kv(q36_tensor *kc, q36_tensor *vc, q36_tensor *k, q36_tensor *v, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, kv = (uint64_t)s->kv_heads, hd = (uint64_t)s->head_dim;
    uint64_t es = q36_kv_bytes(kc, s), cache = kv * (uint64_t)s->cap * hd * es;
    if (!rows || s->pos < 0 || (uint64_t)s->pos + rows > (uint64_t)s->cap || !q36_fits(k, rows * kv * hd * F4) ||
        !q36_fits(v, rows * kv * hd * F4) || !q36_fits(kc, cache) || !q36_fits(vc, cache)) return 0;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        id<MTLComputeCommandEncoder> e = q36_encoder(es == F4 ? K_ATTN_KV : K_ATTN_KV_HALF);
        q36_bind(e, k, 0); q36_bind(e, v, 1); q36_bind(e, kc, 2); q36_bind(e, vc, 3);
        [e setBytes:&p length:sizeof(p) atIndex:4];
        q36_threads(e, rows * kv * hd);
    }
    return 1;
}

int q36_attn(q36_tensor *out, q36_tensor *q, q36_tensor *kc, q36_tensor *vc, q36_tensor *scores, const q36_shape *s) {
    uint64_t rows = (uint64_t)s->rows, qh = (uint64_t)s->q_heads, kv = (uint64_t)s->kv_heads;
    uint64_t hd = (uint64_t)s->head_dim, es = kv ? q36_kv_bytes(kc, s) : F4, cache = kv * (uint64_t)s->cap * hd * es;
    if (!rows || !kv || qh % kv || hd < 1 || hd > 256 || (uint64_t)s->pos + rows > (uint64_t)s->cap ||
        !q36_fits(q, rows * qh * hd * F4) || !q36_fits(out, rows * qh * hd * F4) ||
        !q36_fits(kc, cache) || !q36_fits(vc, cache)) return 0;
    /* A half cache is read by the fast kernels only: every decode step takes
     * the GQA path and every batch of two rows or more the matrix path. */
    int half = es != F4;
    if (half && !q36_kv_half(s)) return 0;
    uint64_t chunk = rows < Q36_ATTN_CHUNK ? rows : Q36_ATTN_CHUNK;
    @autoreleasepool {
        q36_state_args p = q36_state(s);
        /* Decode GQA on matrix units. The first plane is live probabilities;
         * partial PV starts strictly after it, sized by the actual frontier.
         * A 257-token full cache needs more than two planes at hd=256. */
        uint64_t parts = ((uint64_t)s->pos + 256) / 256; /* Q36_GQA_PART */
        if (!g_exact && rows == 1 && qh / kv == 8 && hd % 64 == 0 && (s->pos >= 127 || half) && scores &&
            g_pipe[K_ATTN_GQA_QK].maxTotalThreadsPerThreadgroup >= 128 &&
            g_pipe[K_ATTN_GQA_SOFTMAX].maxTotalThreadsPerThreadgroup >= 256 &&
            g_pipe[K_ATTN_GQA_PV].maxTotalThreadsPerThreadgroup >= 128 &&
            q36_fits(scores, qh * ((uint64_t)s->cap + parts * hd) * F4)) {
            id<MTLComputeCommandEncoder> e = q36_encoder(half ? K_ATTN_GQA_QK_HALF : K_ATTN_GQA_QK);
            q36_bind(e, q, 0); q36_bind(e, kc, 1); q36_bind(e, scores, 2);
            [e setBytes:&p length:sizeof(p) atIndex:3];
            [e setThreadgroupMemoryLength:(64 + 8) * 32 * F4 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(s->pos + 64) / 64, kv, 1)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            e = q36_encoder(K_ATTN_GQA_SOFTMAX);
            q36_bind(e, scores, 0);
            [e setBytes:&p length:sizeof(p) atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(qh, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            e = q36_encoder(half ? K_ATTN_GQA_PV_HALF : K_ATTN_GQA_PV);
            q36_bind(e, scores, 0); q36_bind(e, vc, 1);
            [e setBytes:&p length:sizeof(p) atIndex:2];
            [e setThreadgroupMemoryLength:(64 + 8) * 32 * F4 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)hd / 64, (NSUInteger)parts, kv)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            e = q36_encoder(K_ATTN_GQA_REDUCE);
            q36_bind(e, scores, 0); q36_bind(e, out, 1);
            [e setBytes:&p length:sizeof(p) atIndex:2];
            q36_threads(e, qh * hd);
            return 1;
        }
        /* the maxima of every 32 positions after the scores */
        uint64_t tail = chunk * qh * (uint64_t)s->cap * F4, tiles = ((uint64_t)s->cap + 31) / 32;
        if (!g_exact && (rows > 8 || (half && rows > 1)) && qh == 8 * kv && hd % 256 == 0 && scores &&
            q36_fits(scores, tail + chunk * qh * tiles * F4)) {
            /* eight query heads per KV head: the matrices of 32 query vectors
             * (q36_attn_qk8, q36_attn_pv8), not bit-identical */
            for (uint64_t r0 = 0; r0 < rows; r0 += Q36_ATTN_CHUNK) {
                q36_state_args c = p;
                c.rows = (uint32_t)(rows - r0 < Q36_ATTN_CHUNK ? rows - r0 : Q36_ATTN_CHUNK);
                c.pos = p.pos + (uint32_t)r0;
                uint64_t skip = r0 * qh * hd * F4, n = (uint64_t)c.pos + c.rows;
                id<MTLComputeCommandEncoder> e = q36_encoder(half ? K_ATTN_QK8_HALF : K_ATTN_QK8);
                q36_bind_at(e, q, skip, 0); q36_bind(e, kc, 1); q36_bind(e, scores, 2);
                [e setBytes:&c length:sizeof(c) atIndex:3];
                q36_bind_at(e, scores, tail, 4);
                [e setThreadgroupMemoryLength:32 * 32 * F4 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((c.rows + 3) / 4, (NSUInteger)(n + 127) / 128, kv)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                e = q36_encoder(half ? K_ATTN_PV8_HALF : K_ATTN_PV8);
                q36_bind(e, scores, 0); q36_bind(e, vc, 1); q36_bind_at(e, out, skip, 2);
                [e setBytes:&c length:sizeof(c) atIndex:3];
                q36_bind_at(e, scores, tail, 4);
                [e setThreadgroupMemoryLength:32 * 32 * F4 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((c.rows + 3) / 4, (NSUInteger)hd / 256, kv)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            return 1;
        }
        if (!g_exact && (rows > 8 || (half && rows > 1)) && hd % 64 == 0 && scores &&
            q36_fits(scores, chunk * qh * (uint64_t)s->cap * F4)) {
            /* prompt rows, Q36_ATTN_CHUNK at a time: scores, softmax and values on
             * the matrix units (q36_attn_qk), not bit-identical */
            for (uint64_t r0 = 0; r0 < rows; r0 += Q36_ATTN_CHUNK) {
                q36_state_args c = p;
                c.rows = (uint32_t)(rows - r0 < Q36_ATTN_CHUNK ? rows - r0 : Q36_ATTN_CHUNK);
                c.pos = p.pos + (uint32_t)r0;
                uint64_t skip = r0 * qh * hd * F4, n = (uint64_t)c.pos + c.rows;
                id<MTLComputeCommandEncoder> e = q36_encoder(half ? K_ATTN_QK_HALF : K_ATTN_QK);
                q36_bind_at(e, q, skip, 0); q36_bind(e, kc, 1); q36_bind(e, scores, 2);
                [e setBytes:&c length:sizeof(c) atIndex:3];
                [e setThreadgroupMemoryLength:(32 + 16) * 32 * F4 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((c.rows + 15) / 16, (NSUInteger)(n + 31) / 32, qh)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                e = q36_encoder(K_ATTN_SOFTMAX);
                q36_bind(e, scores, 0);
                [e setBytes:&c length:sizeof(c) atIndex:1];
                q36_groups(e, (uint64_t)c.rows * qh, 1);
                e = q36_encoder(half ? K_ATTN_PV_HALF : K_ATTN_PV);
                q36_bind(e, scores, 0); q36_bind(e, vc, 1); q36_bind_at(e, out, skip, 2);
                [e setBytes:&c length:sizeof(c) atIndex:3];
                [e setThreadgroupMemoryLength:(32 + 16) * 32 * F4 atIndex:0];
                [e dispatchThreadgroups:MTLSizeMake((c.rows + 15) / 16, (NSUInteger)hd / 32, qh)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            return 1;
        }
        if (half) return 0;                     /* no exact kernel reads a half cache */
        if (rows == 1 && scores && qh / kv <= 8 && q36_fits(scores, 3 * qh * (uint64_t)s->cap * F4)) {
            /* decode: the same result from three parallel steps (q36_attn_scores) */
            id<MTLComputeCommandEncoder> e = q36_encoder(K_ATTN_SCORES);
            q36_bind(e, q, 0); q36_bind(e, kc, 1); q36_bind(e, scores, 2);
            [e setBytes:&p length:sizeof(p) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(s->pos + 32) / 32, kv, 1)
              threadsPerThreadgroup:MTLSizeMake(32 * (NSUInteger)(qh / kv), 1, 1)];
            e = q36_encoder(K_ATTN_WEIGHTS);
            q36_bind(e, scores, 0);
            [e setBytes:&p length:sizeof(p) atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(qh, 1, 1)                   /* Q36_ATTN_THREADS */
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            e = q36_encoder(K_ATTN_VALUES);
            q36_bind(e, scores, 0); q36_bind(e, vc, 1); q36_bind(e, out, 2);
            [e setBytes:&p length:sizeof(p) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake((NSUInteger)(hd + 31) / 32, (NSUInteger)(qh + 1) / 2, 1)
              threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return 1;
        }
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ATTN);
        q36_bind(e, q, 0); q36_bind(e, kc, 1); q36_bind(e, vc, 2); q36_bind(e, out, 3);
        [e setBytes:&p length:sizeof(p) atIndex:4];
        q36_groups(e, rows * qh, 1);
    }
    return 1;
}

int q36_router_dot(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out) {
    /* FAST uses the dense vector/matrix reductions for actual and lookahead
     * routing. Exact mode retains the ordered router arithmetic below. */
    if (!g_exact) return q36_dot_i8(y, x, q, sc, rows, in, out);
    if (rows < 1 || in < 1 || out < 1 || !q36_fits(q, (uint64_t)out * in) || !q36_fits(sc, (uint64_t)out * F4) ||
        !q36_fits(x, (uint64_t)rows * in * F4) || !q36_fits(y, (uint64_t)rows * out * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {0, (uint32_t)rows, (uint32_t)in, (uint32_t)out, 0, 0, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ROUTER_DOT);
        q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
        [e setBytes:&a length:sizeof(a) atIndex:4];
        q36_threads(e, (uint64_t)rows * out);
    }
    return 1;
}

int q36_router_dot2(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc,
                    q36_tensor *y2, q36_tensor *x2, q36_tensor *q2, q36_tensor *sc2, int in, int out) {
    if (!g_exact) return q36_dot_i8(y, x, q, sc, 1, in, out) && q36_dot_i8(y2, x2, q2, sc2, 1, in, out);
    if (in < 1 || out < 1 || !q36_fits(q, (uint64_t)out * in) || !q36_fits(sc, (uint64_t)out * F4) ||
        !q36_fits(x, (uint64_t)in * F4) || !q36_fits(y, (uint64_t)out * F4) || !q36_fits(q2, (uint64_t)out * in) ||
        !q36_fits(sc2, (uint64_t)out * F4) || !q36_fits(x2, (uint64_t)in * F4) || !q36_fits(y2, (uint64_t)out * F4))
        return 0;
    @autoreleasepool {
        q36_dense_args a = {0, 1, (uint32_t)in, (uint32_t)out, 0, 0, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_ROUTER_DOT2);
        q36_bind(e, q, 0); q36_bind(e, sc, 1); q36_bind(e, x, 2); q36_bind(e, y, 3);
        [e setBytes:&a length:sizeof(a) atIndex:4];
        q36_bind(e, q2, 5); q36_bind(e, sc2, 6); q36_bind(e, x2, 7); q36_bind(e, y2, 8);
        q36_threads(e, 2 * (uint64_t)out);
    }
    return 1;
}

int q36_shared_dot(q36_tensor *y, q36_tensor *x, q36_tensor *q, q36_tensor *sc, int rows, int in, int out) {
    if (!g_exact) return q36_dot_i8(y, x, q, sc, rows, in, out);
    return q36_router_dot(y, x, q, sc, rows, in, out);
}

int q36_router_softmax(q36_tensor *prob, q36_tensor *logits, q36_tensor *bias, int rows, int experts) {
    if (rows < 1 || experts < 1 || !q36_fits(logits, (uint64_t)rows * experts * F4) ||
        !q36_fits(prob, (uint64_t)rows * experts * F4) || !q36_fits(bias, (uint64_t)experts * F4)) return 0;
    @autoreleasepool {
        q36_router_args a = {(uint32_t)rows, 0, (uint32_t)experts, 0, 1, 0, 0};
        int row = rows <= Q36_ROW_ROWS && experts <= Q36_ROW_MAX;
        /* fast mode: a threadgroup per row and the sum in parallel, not in expert
         * order on one thread, for every batch size (the same bits in any batch) */
        int fast = !g_exact && g_pipe[K_ROUTER_SOFTMAX_FAST].maxTotalThreadsPerThreadgroup >= 256;
        id<MTLComputeCommandEncoder> e = q36_encoder(fast ? K_ROUTER_SOFTMAX_FAST : row ? K_ROUTER_SOFTMAX_ROW
                                                                                      : K_ROUTER_SOFTMAX);
        q36_bind(e, logits, 0); q36_bind(e, bias, 1); q36_bind(e, prob, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        if (fast || row) q36_rows(e, (uint64_t)rows); else q36_threads(e, (uint64_t)rows);
    }
    return 1;
}

int q36_router_select(q36_tensor *ids, q36_tensor *weights, q36_tensor *prob, int rows, int experts,
                      int topk, int groups, int topk_groups) {
    if (rows < 1 || experts < 1 || experts > 256 || topk < 1 || topk > 8 || topk > experts ||
        groups < 1 || experts % groups || !q36_fits(prob, (uint64_t)rows * experts * F4) ||
        !q36_fits(ids, (uint64_t)rows * topk * 4) || !q36_fits(weights, (uint64_t)rows * topk * F4)) return 0;
    @autoreleasepool {
        q36_router_args a = {(uint32_t)rows, 0, (uint32_t)experts, (uint32_t)topk, (uint32_t)groups,
                             (uint32_t)topk_groups, 0};
        int row = rows <= Q36_ROW_ROWS && groups == 1;
        id<MTLComputeCommandEncoder> e = q36_encoder(row ? K_ROUTER_SELECT_ROW : K_ROUTER_SELECT);
        q36_bind(e, prob, 0); q36_bind(e, ids, 1); q36_bind(e, weights, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        if (row) q36_rows(e, (uint64_t)rows); else q36_threads(e, (uint64_t)rows);
    }
    return 1;
}

int q36_silu_mul(q36_tensor *h, q36_tensor *g, q36_tensor *u, int n) {
    if (n < 1 || !q36_fits(g, n * F4) || !q36_fits(u, n * F4) || !q36_fits(h, n * F4)) return 0;
    @autoreleasepool {
        q36_dense_args a = {(uint32_t)n, 0, 0, 0, 0, 0, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_SILU_MUL);
        q36_bind(e, u, 0); q36_bind(e, g, 1); q36_bind(e, h, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        q36_threads(e, (uint64_t)n);
    }
    return 1;
}

int q36_shared_gate(q36_tensor *gate, q36_tensor *x, q36_tensor *w, int rows, int hidden) {
    if (rows < 1 || hidden < 1 || !q36_fits(x, (uint64_t)rows * hidden * F4) ||
        !q36_fits(gate, (uint64_t)rows * F4) || (w && !q36_fits(w, (uint64_t)hidden * F4))) return 0;
    @autoreleasepool {
        q36_router_args a = {(uint32_t)rows, (uint32_t)hidden, 0, 0, 1, 0, w ? 1u : 0u};
        int row = rows <= Q36_ROW_ROWS && w && hidden <= Q36_ROW_MAX;
        id<MTLComputeCommandEncoder> e = q36_encoder(row ? K_SHARED_GATE_ROW : K_SHARED_GATE);
        q36_bind(e, x, 0); q36_bind(e, w ? w : x, 1); q36_bind(e, gate, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        if (row) q36_rows(e, (uint64_t)rows); else q36_threads(e, (uint64_t)rows);
    }
    return 1;
}

int q36_scale_rows(q36_tensor *x, q36_tensor *s, int rows, int hidden) {
    if (rows < 1 || hidden < 1 || !q36_fits(x, (uint64_t)rows * hidden * F4) || !q36_fits(s, (uint64_t)rows * F4))
        return 0;
    @autoreleasepool {
        q36_router_args a = {(uint32_t)rows, (uint32_t)hidden, 0, 0, 1, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_SCALE_ROWS);
        q36_bind(e, x, 0); q36_bind(e, s, 1);
        [e setBytes:&a length:sizeof(a) atIndex:2];
        q36_threads(e, (uint64_t)rows * hidden);
    }
    return 1;
}

/* Bytes of one expert slot: three planar int4 matrices (3n/2 bytes), then,
 * from byte `scales`, their 3n/64 float scales. */
static uint64_t q36_slot_bytes(int hidden, int inter, size_t scales) {
    uint64_t n = (uint64_t)hidden * inter;
    if (scales < 3 * n / 2) return UINT64_MAX;
    return (uint64_t)scales + 3 * (n / 64) * F4;
}

/* Every pair names a bound slot and a choice inside the batch. The CPU wrote
 * the pairs, so reading them back here costs nothing. */
static int q36_pairs_ok(q36_tensor *pairs, int first, int n, q36_tensor *const slots[8], int rows, int topk,
                        uint64_t slot_bytes) {
    uint64_t choices = (uint64_t)rows * topk;
    if (first < 0 || n < 1 || choices > 0x10000000u || !q36_fits(pairs, ((uint64_t)first + n) * 4)) return 0;
    const uint32_t *p = (const uint32_t *)q36_tensor_data(pairs) + first;
    for (int i = 0; i < n; i++) {
        uint32_t slot = p[i] >> 28;
        if (slot > 7 || (p[i] & 0x0fffffffu) >= choices || !q36_fits(slots[slot], slot_bytes)) return 0;
    }
    return 1;
}

/* Where each slot's choices sit in pairs[first, first + n); the largest count,
 * or 0 when a slot's choices are not contiguous. */
static uint32_t q36_expert_ranges(q36_tensor *pairs, int first, int n, uint32_t off[8], uint32_t cnt[8]) {
    const uint32_t *pp = (const uint32_t *)q36_tensor_data(pairs) + first;
    uint32_t most = 0;
    memset(off, 0, 8 * sizeof(uint32_t));
    memset(cnt, 0, 8 * sizeof(uint32_t));
    for (int i = 0; i < n; i++) {
        uint32_t slot = pp[i] >> 28;
        if (slot > 7) return 0;
        if (!cnt[slot]) off[slot] = (uint32_t)i;
        else if (off[slot] + cnt[slot] != (uint32_t)i) return 0;
        if (++cnt[slot] > most) most = cnt[slot];
    }
    return most;
}

static int q36_experts(int kernel, q36_tensor *out, uint64_t out_bytes, q36_tensor *in, uint64_t in_bytes,
                       q36_tensor *pairs, int first, int n, q36_tensor *const slots[8], int rows, int hidden,
                       int inter, int topk, size_t scales, int width) {
    if (rows < 1 || topk < 1 || topk > 8 || hidden < 64 || inter < 64 || hidden % 64 || inter % 64 ||
        scales % 4 || !q36_fits(out, out_bytes) || !q36_fits(in, in_bytes) ||
        !q36_pairs_ok(pairs, first, n, slots, rows, topk, q36_slot_bytes(hidden, inter, scales))) return 0;
    @autoreleasepool {
        q36_expert_args a = {(uint32_t)hidden, (uint32_t)inter, (uint32_t)topk, (uint32_t)rows,
                             (uint32_t)first, (uint32_t)n, (uint64_t)scales};
        struct { uint32_t off[8], cnt[8]; } rg;
        uint32_t tile = 0;
        if (!g_exact && rows > 8 && (kernel == K_EXPERT_GATE_UP || kernel == K_EXPERT_DOWN))
            tile = q36_expert_ranges(pairs, first, n, rg.off, rg.cnt);
        if (tile > 0 && in->offset % 16 == 0 && out->offset % 8 == 0) {
            /* prompt batches on the matrix units (q36_expert_*_mm): up to 64
             * choices of a slot and 64 columns per threadgroup, the last one as
             * many row blocks of 8 as the slot has left */
            int gate_up = kernel == K_EXPERT_GATE_UP;
            NSUInteger cols = gate_up ? (NSUInteger)inter / 32 : (NSUInteger)hidden / 64;
            /* the last tile of each slot with 1 to 4 row blocks: q36_expert_*_mm4 */
            id<MTLComputeCommandEncoder> e = q36_encoder(gate_up ? K_EXPERT_GATE_UP_MM4 : K_EXPERT_DOWN_MM4);
            q36_bind(e, in, 0); q36_bind(e, pairs, 1); q36_bind(e, out, 2);
            [e setBytes:&a length:sizeof(a) atIndex:3];
            for (int k = 0; k < 8; k++) q36_bind(e, slots[k] ? slots[k] : pairs, (NSUInteger)(4 + k));
            [e setBytes:&rg length:sizeof(rg) atIndex:12];
            [e dispatchThreadgroups:MTLSizeMake(1, cols, 8) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            if (tile <= 32) return 1;           /* no slot has a tile of 5 row blocks or more */
            e = q36_encoder(gate_up ? K_EXPERT_GATE_UP_MM : K_EXPERT_DOWN_MM);
            [e setThreadgroupMemoryLength:64 * 32 * F4 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake((tile + 63) / 64, cols, 8) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return 1;
        }
        id<MTLComputeCommandEncoder> e = q36_encoder(kernel);
        q36_bind(e, in, 0); q36_bind(e, pairs, 1); q36_bind(e, out, 2);
        [e setBytes:&a length:sizeof(a) atIndex:3];
        for (int k = 0; k < 8; k++) q36_bind(e, slots[k] ? slots[k] : pairs, (NSUInteger)(4 + k));
        NSUInteger most = g_pipe[kernel].maxTotalThreadsPerThreadgroup;
        [e dispatchThreads:MTLSizeMake((NSUInteger)width, (NSUInteger)n, 1)
            threadsPerThreadgroup:MTLSizeMake(most < 64 ? most : 64, 1, 1)];
    }
    return 1;
}

int q36_expert_gate_up(q36_tensor *h, q36_tensor *x, q36_tensor *pairs, int first, int n,
                       q36_tensor *const slots[8], int rows, int hidden, int inter, int topk, size_t scales) {
    uint64_t choices = (uint64_t)rows * topk;
    return q36_experts(K_EXPERT_GATE_UP, h, choices * inter * F4, x, (uint64_t)rows * hidden * F4, pairs, first, n,
                       slots, rows, hidden, inter, topk, scales, inter);
}

int q36_expert_down(q36_tensor *y, q36_tensor *h, q36_tensor *pairs, int first, int n,
                    q36_tensor *const slots[8], int rows, int hidden, int inter, int topk, size_t scales) {
    uint64_t choices = (uint64_t)rows * topk;
    return q36_experts(K_EXPERT_DOWN, y, choices * hidden * F4, h, choices * inter * F4, pairs, first, n,
                       slots, rows, hidden, inter, topk, scales, hidden);
}

int q36_expert_sum(q36_tensor *out, q36_tensor *y, q36_tensor *weights, q36_tensor *shared,
                   q36_tensor *residual, int rows, int hidden, int topk) {
    uint64_t cells = (uint64_t)rows * hidden;
    if (rows < 1 || hidden < 1 || topk < 1 || topk > 8 || cells > UINT32_MAX ||
        !q36_fits(out, cells * F4) || !q36_fits(y, cells * topk * F4) ||
        !q36_fits(weights, (uint64_t)rows * topk * F4) || !q36_fits(shared, cells * F4) ||
        !q36_fits(residual, cells * F4)) return 0;
    @autoreleasepool {
        q36_expert_args a = {(uint32_t)hidden, 0, (uint32_t)topk, (uint32_t)rows, 0, 0, 0};
        id<MTLComputeCommandEncoder> e = q36_encoder(K_EXPERT_SUM);
        q36_bind(e, y, 0); q36_bind(e, weights, 1); q36_bind(e, shared, 2);
        q36_bind(e, residual, 3); q36_bind(e, out, 4);
        [e setBytes:&a length:sizeof(a) atIndex:5];
        q36_threads(e, cells);
    }
    return 1;
}
