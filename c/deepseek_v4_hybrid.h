/* deepseek_v4_hybrid.h — the q* fill/execute split for GPU expert misses.
 *
 * On a decode step some routed experts are resident in VRAM and some are not.
 * Today the engine is all-or-nothing: one missing expert sends the WHOLE
 * token's routed MoE to the CPU, and every miss is uploaded over PCIe
 * unconditionally. The hybrid split partitions the m missing experts into a
 * fill set (upload to VRAM, compute on GPU) and a host set (compute on CPU),
 * sized so the two branches take the same time:
 *
 *     T_fill(q)  ~= q * S / B_P          (PCIe transfer of q experts)
 *     T_host(m-q)~= (m-q) * S / (B_H-B_P) (host compute of the rest)
 *
 * Balancing the two gives the closed form
 *
 *     q* = m * B_P / B_H
 *
 * with B_P the measured fill bandwidth and B_H the measured host execution
 * bandwidth. As B_H approaches B_P, q* approaches m and the split degenerates
 * into plain upload-everything — which is also the answer while the
 * bandwidths are still unmeasured. Both bandwidths are live EMAs of the
 * engine's own uploads and expert forwards, never assumptions.
 *
 * Everything in this header is pure arithmetic so the CI can test the policy
 * on machines with no GPU at all; the CUDA wiring lives in deepseek_v4.c
 * behind DSV4_HYBRID=1. */
#ifndef DEEPSEEK_V4_HYBRID_H
#define DEEPSEEK_V4_HYBRID_H

/* How many of `missing` experts to upload-and-run-on-GPU; the caller computes
 * the rest on the CPU. Bandwidths are in experts/second (only their RATIO
 * matters, so any consistent unit works). Unmeasured or nonsensical
 * bandwidths fall back to `missing` — the engine's historical
 * upload-everything behaviour, never something new. */
static inline int coli_v4_hybrid_fill_count(int missing, double fill_bw,
                                            double host_bw) {
    if (missing <= 0) return 0;
    if (fill_bw <= 0.0 || host_bw <= 0.0) return missing;
    double ideal = (double)missing * fill_bw / host_bw;   /* q* = m*B_P/B_H */
    int fill = (int)(ideal + 0.5);
    if (fill < 0) fill = 0;
    if (fill > missing) fill = missing;
    return fill;
}

/* One-pole EMA for the live bandwidth estimates. A non-positive sample is
 * ignored (a failed or zero-length measurement must not poison the state);
 * the first valid sample seeds the average directly. */
static inline double coli_v4_hybrid_ema(double previous, double sample) {
    if (sample <= 0.0) return previous;
    if (previous <= 0.0) return sample;
    return previous * 0.8 + sample * 0.2;
}

#endif /* DEEPSEEK_V4_HYBRID_H */
