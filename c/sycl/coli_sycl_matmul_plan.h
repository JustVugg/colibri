/* Host-only planning for the legacy 256-thread-per-output matmul. */
#ifndef COLI_SYCL_MATMUL_PLAN_H
#define COLI_SYCL_MATMUL_PLAN_H
#include <cerrno>
#include <cstdlib>
#include <limits>

/* Unset/invalid/auto: automatic safe bound. Zero: legacy unsplit diagnostic.
 * Positive values can force smaller batches, but never enlarge the safe bound. */
inline int coli_sycl_mm_override(const char *text) {
    if (!text || !*text) return -1;
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (errno || *end || v < 0 || v > (std::numeric_limits<int>::max)()) return -1;
    return (int)v;
}
inline int coli_sycl_mm_rows(int rows, int outputs, int requested = -1) {
    if (rows < 1 || outputs < 1) return 0;
    if (requested == 0) return rows;
    // Divide first: no product can overflow, including for rejected dimensions.
    int safe = (std::numeric_limits<int>::max)() / 256 / outputs;
    if (requested > 0 && requested < safe) safe = requested;
    return rows < safe ? rows : safe;
}
#endif
