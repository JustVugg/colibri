/* V4 GPU placement plan: one decision for every VRAM tier.
 *
 * The native V4 engine used to let each tier pick its own place: dense
 * mirrors, the shared experts, the prefill bank and the decode mirror cache
 * each read a different environment variable and reserved VRAM on its own.
 * That is why a budget could not be enforced and why the same setting meant
 * different things on 6, 8 and 16 GB cards.
 *
 * This header is the single decision point. It is a pure function: no GPU,
 * no environment, no I/O. The engine passes the model geometry, the real
 * device budget and the workload; the returned plan says which tiers live on
 * the device and how many mirror slots remain. Every allocation is charged
 * against the same budget, so `sum(plan) <= budget` holds by construction.
 *
 * Bytes are bytes here (not MiB/GiB); the caller converts once.
 */
#ifndef COLI_V4_GPU_PLAN_H
#define COLI_V4_GPU_PLAN_H

#include <stdint.h>

typedef enum {
    V4_GPU_WORKLOAD_AUTO = 0,   /* decode-first: a server answers tokens */
    V4_GPU_WORKLOAD_DECODE = 1,
    V4_GPU_WORKLOAD_PREFILL = 2, /* one long prompt, little generation */
} V4GpuWorkload;

typedef struct {
    int backend_ok;          /* a usable CUDA/HIP device was opened */
    int workload;            /* V4GpuWorkload */
    long long budget_bytes;  /* user budget; 0 = use free_bytes */
    long long free_bytes;    /* device free at open */
    long long safety_bytes;  /* system reserve taken from an automatic budget */
    long long dense_bytes;   /* all dense/attention mirrors, shared included */
    long long shared_bytes;  /* shared experts alone, all layers */
    long long bank_bytes;    /* full routed-expert bank, all slots */
    long long kv_bytes;      /* context/KV/work estimate */
    long long mirror_bytes;  /* one decode expert mirror (gate+up+down) */
    int n_experts;           /* routed experts (132, 256, ...) */
} V4GpuPlanRequest;

typedef struct {
    int backend;             /* 0 = CPU, 1 = device opened */
    int dense;               /* dense/attention mirrors on the device */
    int shared;              /* shared-expert mirrors on the device */
    int bank;                /* prefill bank enabled */
    int attn;                /* batched prefill attention on the device */
    int mirror_slots;        /* decode mirror cache capacity */
    long long reserve_bytes; /* KV + allocator headroom, kept out of tiers */
    long long used_bytes;    /* dense + shared + bank + mirrors + reserve */
    int phase;               /* the workload the plan was built for */
} V4GpuPlan;

#define V4_GPU_PLAN_MIN_MIRRORS 8
#define V4_GPU_PLAN_HEADROOM_BYTES (256LL << 20)
#define V4_GPU_PLAN_DEFAULT_SAFETY_BYTES (512LL << 20)
#define V4_GPU_PLAN_DENSE_DECODE_MIN_BUDGET (10LL << 30)
#define V4_GPU_PLAN_PREFILL_MIN_TOKENS 64

static inline long long v4_gpu_plan_mirror_bytes(const V4GpuPlanRequest *r) {
    return r->mirror_bytes > 0 ? r->mirror_bytes : (12LL << 20) + (3LL << 19);
}

static inline V4GpuPlan v4_gpu_plan_decide(const V4GpuPlanRequest *request) {
    V4GpuPlan plan;
    long long budget, reserve, avail, mirror;
    int phase;

    plan.backend = 0;
    plan.dense = 0;
    plan.shared = 0;
    plan.bank = 0;
    plan.attn = 0;
    plan.mirror_slots = 0;
    plan.reserve_bytes = 0;
    plan.used_bytes = 0;
    plan.phase = V4_GPU_WORKLOAD_DECODE;

    if (!request || !request->backend_ok) return plan;
    plan.backend = 1;
    phase = request->workload == V4_GPU_WORKLOAD_PREFILL
                ? V4_GPU_WORKLOAD_PREFILL
                : V4_GPU_WORKLOAD_DECODE;
    plan.phase = phase;

    /* The physical card is a hard ceiling: a user budget above free VRAM
     * would make the allocator promise memory that is not there. An
     * automatic budget also leaves the system safety reserve: the driver,
     * the compositor and other processes are not part of the model plan.
     * vLLM ships the same idea as gpu_memory_utilization=0.9; colibri's
     * resource planner keeps 2 GiB for the general CUDA path. The native
     * engine keeps 512 MiB here, enough for the driver/context without
     * throwing away a mirror-heavy 8 GB plan. */
    budget = request->budget_bytes > 0 ? request->budget_bytes
                                       : request->free_bytes;
    if (request->budget_bytes <= 0 && request->safety_bytes > 0) {
        budget -= request->safety_bytes;
        if (budget < 0) budget = 0;
    }
    if (request->free_bytes > 0 && budget > request->free_bytes)
        budget = request->free_bytes;
    reserve = (request->kv_bytes > 0 ? request->kv_bytes : 0) +
              V4_GPU_PLAN_HEADROOM_BYTES;
    if (budget <= reserve) {
        plan.reserve_bytes = budget > 0 ? budget : 0;
        plan.used_bytes = plan.reserve_bytes;
        return plan;
    }
    avail = budget - reserve;
    mirror = v4_gpu_plan_mirror_bytes(request);

    if (phase == V4_GPU_WORKLOAD_PREFILL) {
        /* Prefill is attention-bound once the dense mirrors fit: measured
         * 222 s for 1955 tokens with dense+attn vs 390 s with dense in RAM
         * and the bank on an 8 GB card. The bank only wins when dense does
         * not fit beside a minimal mirror set. */
        if (request->dense_bytes > 0 &&
            request->dense_bytes +
                    (long long)V4_GPU_PLAN_MIN_MIRRORS * mirror <= avail) {
            long long remaining;
            plan.dense = 1;
            plan.attn = 1;
            remaining = avail - request->dense_bytes;
            /* The bank is a prefill tier: keep it when the dense mirrors
             * still leave room for a full layer plus the mirror minimum
             * (12 GB/256 experts, 16 GB/132 experts, ...). */
            if (request->bank_bytes > 0 &&
                remaining >= request->bank_bytes +
                                 (long long)V4_GPU_PLAN_MIN_MIRRORS * mirror) {
                plan.bank = 1;
                remaining -= request->bank_bytes;
            }
            plan.mirror_slots = (int)(remaining / mirror);
        } else if (request->shared_bytes > 0 && request->bank_bytes > 0 &&
                   request->shared_bytes + request->bank_bytes +
                           (long long)V4_GPU_PLAN_MIN_MIRRORS * mirror <= avail) {
            plan.shared = 1;
            plan.bank = 1;
            plan.mirror_slots =
                (int)((avail - request->shared_bytes - request->bank_bytes) /
                      mirror);
        } else {
            plan.mirror_slots = (int)(avail / mirror);
        }
    } else {
        /* Decode: the mirror cache is the throughput. On 8 GB the measured
         * 0.65 tok/s came from dense in RAM and ~600 mirrors, against
         * 0.42 tok/s from dense on the card and ~25 mirrors. Keep the dense
         * mirrors on the device only when the card is large enough to afford
         * both them and a working mirror set (10 GiB rule, measured). */
        if (budget >= V4_GPU_PLAN_DENSE_DECODE_MIN_BUDGET &&
            request->dense_bytes > 0 &&
            request->dense_bytes +
                    (long long)V4_GPU_PLAN_MIN_MIRRORS * mirror <= avail) {
            plan.dense = 1;
            plan.attn = 1;
            plan.mirror_slots =
                (int)((avail - request->dense_bytes) / mirror);
        } else {
            /* dense in RAM; shared experts stay in RAM too unless the bank
             * is wanted, which decode does not use. */
            plan.mirror_slots = (int)(avail / mirror);
        }
    }

    /* The current mirror cache refuses fewer than 8 live entries: a smaller
     * set can recycle a slot a view of the same token still references. A
     * plan that cannot hold the minimum runs on the CPU rather than produce
     * a cache that silently returns garbage. */
    if (plan.mirror_slots < V4_GPU_PLAN_MIN_MIRRORS)
        plan.mirror_slots = 0;

    plan.reserve_bytes = reserve;
    plan.used_bytes = reserve +
                      (plan.dense ? request->dense_bytes : 0) +
                      (plan.shared ? request->shared_bytes : 0) +
                      (plan.bank ? request->bank_bytes : 0) +
                      (long long)plan.mirror_slots * mirror;
    if (plan.used_bytes > budget) {
        /* Defensive: the branches above already fit; a rounded input that
         * broke the invariant must not reach the allocator. */
        plan.dense = plan.shared = plan.bank = plan.attn = 0;
        plan.mirror_slots = (int)((budget > reserve ? budget - reserve : 0) /
                                  mirror);
        if (plan.mirror_slots < V4_GPU_PLAN_MIN_MIRRORS) plan.mirror_slots = 0;
        plan.used_bytes = reserve + (long long)plan.mirror_slots * mirror;
    }
    return plan;
}

#endif /* COLI_V4_GPU_PLAN_H */
