/* Table tests for the pure V4 GPU planner. No GPU, no model, no env. */
#include <stdio.h>
#include <string.h>
#include "deepseek_v4_gpu_plan.h"

#define GIB (1LL << 30)
#define MIB (1LL << 20)

static int failures;

static V4GpuPlanRequest request(long long budget, int workload, int experts) {
    V4GpuPlanRequest r;
    memset(&r, 0, sizeof(r));
    r.backend_ok = 1;
    r.workload = workload;
    r.budget_bytes = budget;
    r.free_bytes = budget;
    r.dense_bytes = (long long)(6.226 * GIB);
    r.shared_bytes = (long long)(1.010 * GIB);
    r.bank_bytes = experts == 256 ? (long long)(3.188 * GIB)
                                  : (long long)(1.683 * GIB);
    r.kv_bytes = (long long)(0.700 * GIB);
    r.mirror_bytes = (long long)(12.75 * MIB);
    r.n_experts = experts;
    return r;
}

static void check(int condition, const char *what, int budget_gib,
                  int workload, int experts) {
    if (!condition) {
        failures++;
        fprintf(stderr, "FAIL %s (%d GiB, workload=%d, experts=%d)\n",
                what, budget_gib, workload, experts);
    }
}

static void check_plan(const V4GpuPlanRequest *r, const V4GpuPlan *p,
                       const char *label) {
    char what[256];
    snprintf(what, sizeof(what), "%s: used <= budget", label);
    check(p->used_bytes <= r->budget_bytes, what,
          (int)(r->budget_bytes / GIB), r->workload, r->n_experts);
    snprintf(what, sizeof(what), "%s: backend on", label);
    check(p->backend == 1, what, (int)(r->budget_bytes / GIB),
          r->workload, r->n_experts);
    snprintf(what, sizeof(what), "%s: attn only with dense", label);
    check(!p->attn || p->dense, what, (int)(r->budget_bytes / GIB),
          r->workload, r->n_experts);
    snprintf(what, sizeof(what), "%s: bank needs shared", label);
    check(!p->bank || p->shared || p->dense, what,
          (int)(r->budget_bytes / GIB), r->workload, r->n_experts);
    snprintf(what, sizeof(what), "%s: mirrors zero or >= 8", label);
    check(p->mirror_slots == 0 || p->mirror_slots >= V4_GPU_PLAN_MIN_MIRRORS,
          what, (int)(r->budget_bytes / GIB), r->workload, r->n_experts);
}

int main(void) {
    static const int budgets[] = {4, 6, 8, 12, 16, 24};
    for (size_t i = 0; i < sizeof(budgets) / sizeof(budgets[0]); i++) {
        for (int workload = V4_GPU_WORKLOAD_DECODE;
             workload <= V4_GPU_WORKLOAD_PREFILL; workload++) {
            for (int experts = 132; experts <= 256; experts += 124) {
                V4GpuPlanRequest r = request((long long)budgets[i] * GIB,
                                             workload, experts);
                V4GpuPlan p = v4_gpu_plan_decide(&r);
                char label[64];
                snprintf(label, sizeof(label), "%d GiB w%d e%d", budgets[i],
                         workload, experts);
                check_plan(&r, &p, label);

                if (workload == V4_GPU_WORKLOAD_PREFILL && budgets[i] >= 8) {
                    snprintf(label, sizeof(label), "%d GiB prefill e%d dense",
                             budgets[i], experts);
                    check(p.dense == 1, label, budgets[i], workload, experts);
                    check(p.attn == 1, label, budgets[i], workload, experts);
                }
                if (workload == V4_GPU_WORKLOAD_PREFILL && budgets[i] <= 6 &&
                    (budgets[i] == 6 || experts == 132)) {
                    snprintf(label, sizeof(label), "%d GiB prefill e%d bank",
                             budgets[i], experts);
                    check(p.bank == 1, label, budgets[i], workload, experts);
                    check(p.dense == 0, label, budgets[i], workload, experts);
                }
                if (workload == V4_GPU_WORKLOAD_DECODE && budgets[i] == 8) {
                    snprintf(label, sizeof(label), "8 GiB decode e%d dense-ram",
                             experts);
                    check(p.dense == 0, label, budgets[i], workload, experts);
                    check(p.mirror_slots > 100, label, budgets[i], workload,
                          experts);
                }
                if (workload == V4_GPU_WORKLOAD_DECODE && budgets[i] >= 12) {
                    snprintf(label, sizeof(label),
                             "%d GiB decode e%d dense-gpu", budgets[i], experts);
                    check(p.dense == 1, label, budgets[i], workload, experts);
                }
            }
        }
    }

    {   /* 256 experts do not fit beside dense on 8 GB, but they do on 12. */
        V4GpuPlanRequest r = request(8 * GIB, V4_GPU_WORKLOAD_PREFILL, 256);
        V4GpuPlan p = v4_gpu_plan_decide(&r);
        check(p.dense == 1 && p.bank == 0, "8 GiB/256 prefill dense", 8,
              r.workload, 256);
        r = request(12 * GIB, V4_GPU_WORKLOAD_PREFILL, 256);
        p = v4_gpu_plan_decide(&r);
        check(p.dense == 1 && p.bank == 1, "12 GiB/256 prefill dense+bank", 12,
              r.workload, 256);
    }

    {   /* A budget above free VRAM is clamped to the card. */
        V4GpuPlanRequest r = request(8 * GIB, V4_GPU_WORKLOAD_DECODE, 132);
        r.budget_bytes = 64 * GIB;
        V4GpuPlan p = v4_gpu_plan_decide(&r);
        check(p.used_bytes <= r.free_bytes, "budget clamped to free", 8,
              r.workload, 132);
    }

    {   /* No room for the minimum mirror set: no cache, not a broken cache. */
        V4GpuPlanRequest r = request(2 * GIB, V4_GPU_WORKLOAD_DECODE, 132);
        r.kv_bytes = 2 * GIB;
        V4GpuPlan p = v4_gpu_plan_decide(&r);
        check(p.mirror_slots == 0, "below reserve: no mirrors", 2, r.workload,
              132);
        check(p.used_bytes <= r.budget_bytes, "below reserve: budget", 2,
              r.workload, 132);
    }

    {   /* Backend off is a real off: no tier, no mirrors. */
        V4GpuPlanRequest r = request(24 * GIB, V4_GPU_WORKLOAD_DECODE, 132);
        r.backend_ok = 0;
        V4GpuPlan p = v4_gpu_plan_decide(&r);
        check(p.backend == 0 && p.dense == 0 && p.bank == 0 &&
              p.mirror_slots == 0, "backend off", 24, r.workload, 132);
    }

    if (failures) {
        fprintf(stderr, "v4 gpu plan: %d failure(s)\n", failures);
        return 1;
    }
    printf("PASS v4 gpu plan: 4/6/8/12/16/24 GiB, 132/256 experts, "
           "decode/prefill, used <= budget\n");
    return 0;
}
