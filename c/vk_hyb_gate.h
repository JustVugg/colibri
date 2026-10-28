/* The gate on the dense chain's hybrid steps (vk_dmoe.h): whether a decode step's host pool
 * experts are computed by the CPU beside the device's (the hybrid steps) or read by the
 * device over the bus is measured, not assumed. The decode forwards are timed with the
 * device's reads for VKC_HYB_PROBE forwards, then with the hybrid steps for as many, and
 * the faster is kept (the hybrid where it is not slower by a twentieth); the first forward
 * of a phase warms it and is not counted; after VKC_HYB_REPROBE forwards both are measured
 * again. Whether the CPU's pair beats the device's read depends on the bus, the CPU, the
 * format and how many pairs a step has: on an RTX 3070 at PCIe 4.0 x16 with Qwen3.6
 * int4 (1.4 pool pairs a step) the device's reads won, 18.5 ms a token against 33 to 50
 * with the hybrid; on an RTX PRO 4000 at x8 the hybrid won (vulkan.md). The gate waits with
 * the hybrid on (phase 3) until an engine reports its first forward (qwen36's chain does):
 * an engine that does not report keeps the hybrid steps as before. Its forwards have no fixed
 * row count yet (qwen38's MTP verifies one to four rows a forward), and a probe of two such
 * forwards a phase kept the device's reads on Qwen3.8-Flash-Next IQ2_XS on an RTX PRO 4000
 * (21.6 against 18.6 ms) where the hybrid decoded faster, 42.2-42.7 tok/s against 40.5-41.4.
 * Pure state, no device: tests/test_vk_chain.c runs it. */
#ifndef VK_HYB_GATE_H
#define VK_HYB_GATE_H
#define VKC_HYB_PROBE 2
#define VKC_HYB_REPROBE 1024
typedef struct {
    int use;                 /* this forward's steps take the hybrid */
    int phase, n;            /* 0: the device's reads measured, 1: the hybrid's, 2: settled, 3: waiting for a report; the phase's forwards */
    int probes;              /* times both were measured */
    double ms[2], kept[2];   /* the phases' summed forward times; the last decision's, a forward each (hybrid, device) */
    unsigned long long fwd, gate_off;   /* forwards since the decision; decisions against the hybrid */
} VkcHybGate;
static inline void vkc_hyb_gate_init(VkcHybGate *g, int gated) {
    g->use = 1; g->phase = gated ? 3 : 2; g->n = 0; g->probes = 0;
    g->ms[0] = g->ms[1] = g->kept[0] = g->kept[1] = 0; g->fwd = 0; g->gate_off = 0;
}
/* a decode forward of `ms` done; returns 1 when a decision was just made */
static inline int vkc_hyb_gate_forward(VkcHybGate *g, double ms) {
    if (g->phase == 3) { g->phase = 0; g->n = 0; g->use = 0; return 0; }   /* the first report: the probe starts */
    if (g->phase == 2) {
        if (g->probes == 0) return 0;              /* not gated */
        if (++g->fwd < VKC_HYB_REPROBE) return 0;  /* settled: measure again after a while */
        g->fwd = 0; g->phase = 0; g->n = 0; g->ms[0] = g->ms[1] = 0; g->use = 0;
        return 0;
    }
    if (g->n++ == 0) return 0;                     /* the phase's first forward warms it */
    g->ms[g->phase] += ms;
    if (g->n <= VKC_HYB_PROBE) return 0;
    if (g->phase == 0) { g->phase = 1; g->n = 0; g->use = 1; return 0; }
    g->use = g->ms[1] <= g->ms[0] * 1.05;
    g->kept[0] = g->ms[1] / VKC_HYB_PROBE; g->kept[1] = g->ms[0] / VKC_HYB_PROBE;
    g->phase = 2; g->n = 0; g->fwd = 0; g->probes++;
    if (!g->use) g->gate_off++;
    return 1;
}
#endif
