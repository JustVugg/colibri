/* hybrid_split.h — pure split/budget policies for GPU-tier work, shared by
 * the engines. Everything here is arithmetic on measured rates so the CI can
 * test the decisions on machines with no GPU at all; each engine's wiring
 * stays in its own file behind its own opt-in switch.
 *
 * DSV4 hybrid decode split (DSV4_HYBRID=1): of m experts missing from the
 * VRAM mirrors, upload-and-run q* = m * B_P / B_H on the GPU and compute the
 * rest on the host, so the transfer branch and the host branch finish
 * together (derivation in the function comment below).
 *
 * K3 adaptive tier fill (K3_VK_UP=auto): the fill-once routed-expert tier
 * uploads from freshly-read RAM slots inline on the decode thread, so every
 * upload is decode latency. The budget bounds that investment to a fraction
 * of the measured step time instead of a hardcoded count. */
#ifndef HYBRID_SPLIT_H
#define HYBRID_SPLIT_H

/* How many of `missing` experts to upload-and-run-on-GPU; the caller computes
 * the rest on the host. Balancing the two concurrent branches,
 *     T_fill(q) ~= q*S/B_P   against   T_host(m-q) ~= (m-q)*S/(B_H-B_P),
 * gives the closed form q* = m * B_P / B_H, clamped to [0, m]. Bandwidths
 * are in experts/second (only their ratio matters). Unmeasured or
 * nonsensical bandwidths fall back to `missing` — the historical
 * upload-everything behaviour, never something new. */
static inline int coli_v4_hybrid_fill_count(int missing, double fill_bw,
                                            double host_bw) {
    if (missing <= 0) return 0;
    if (fill_bw <= 0.0 || host_bw <= 0.0) return missing;
    double ideal = (double)missing * fill_bw / host_bw;
    int fill = (int)(ideal + 0.5);
    if (fill < 0) fill = 0;
    if (fill > missing) fill = missing;
    return fill;
}

/* One-pole EMA for the live rate estimates. A non-positive sample is ignored
 * (a failed or zero-length measurement must not poison the state); the first
 * valid sample seeds the average directly. */
static inline double coli_v4_hybrid_ema(double previous, double sample) {
    if (sample <= 0.0) return previous;
    if (previous <= 0.0) return sample;
    return previous * 0.8 + sample * 0.2;
}

/* Per-step upload budget for a fill-once tier whose uploads run INLINE on
 * the decode thread: spend at most `fraction` of the measured step time on
 * uploads, i.e. budget = fraction * step_seconds / upload_seconds. While
 * either rate is unmeasured the caller's legacy fixed cap applies, so
 * enabling the adaptive mode can never behave worse than the default before
 * the first measurements exist. Clamped to [1, 64]: at least one upload per
 * step keeps the tier filling even on a slow bus, and 64 bounds the latency
 * spike a mismeasured step could buy. */
static inline int coli_k3_fill_budget(double step_seconds,
                                      double upload_seconds,
                                      double fraction, int legacy_cap) {
    if (step_seconds <= 0.0 || upload_seconds <= 0.0) return legacy_cap;
    if (fraction <= 0.0) return 1;
    double ideal = fraction * step_seconds / upload_seconds;
    int budget = (int)(ideal + 0.5);
    if (budget < 1) budget = 1;
    if (budget > 64) budget = 64;
    return budget;
}

#endif /* HYBRID_SPLIT_H */
