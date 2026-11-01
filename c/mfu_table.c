/* MFU popularity table + persistence — PLAN_cache Task 1 (§5.1).
 *
 * A flat uint32 counter per expert, incremented O(1) per ROUTED selection
 * (decode + MTP/spec drafts). Load decay is OFF by default (see
 * MFU_DECAY_SHIFT_DEFAULT): the persisted counts seed the next boot's
 * warm fill in full; COLI_DSV4_VK_MFU_DECAY_SHIFT=N>0 opts back into
 * >>N-at-load recency. Corrupt/truncated/bad-magic/missing file ->
 * cold start (mfu_load returns -1, caller's zeroed table untouched).
 *
 * See mfu_table.h for the full contract. */
#include "mfu_table.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- little-endian serialization (portable across the Makefile's
 * Linux/macOS/Windows hosts; the single-host deployment reads and writes
 * with the same helpers, but a fixed byte order keeps files portable and
 * testable byte-for-byte) ---- */
static void wr_u32le(FILE *f, uint32_t v) {
    unsigned char b[4];
    b[0] = (unsigned char)(v);
    b[1] = (unsigned char)(v >> 8);
    b[2] = (unsigned char)(v >> 16);
    b[3] = (unsigned char)(v >> 24);
    fwrite(b, 1, 4, f);
}
static void wr_u64le(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; i++) b[i] = (unsigned char)(v >> (8 * i));
    fwrite(b, 1, 8, f);
}
static int rd_u32le(FILE *f, uint32_t *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return -1;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int rd_u64le(FILE *f, uint64_t *out) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) return -1;
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    *out = v;
    return 0;
}

void mfu_init(MfuTable *m) {
    memset(m, 0, sizeof(*m));
    m->decay_shift = MFU_DECAY_SHIFT_DEFAULT; /* 0 = no load decay (default) */
    const char *e = getenv("COLI_DSV4_VK_MFU_DECAY_SHIFT");
    if (e && *e) {
        unsigned long v = strtoul(e, NULL, 10);
        if (v <= 31) m->decay_shift = (uint32_t)v; /* 0 = no load decay */
    }
}

void mfu_scale_down(MfuTable *m) {
    /* Overflow guard: a correctness invariant, never a no-op — always
     * shifts by at least 1, whatever the (possibly zero) load-decay
     * setting. */
    unsigned s = m->decay_shift;
    if (s == 0 || s >= 32) s = 1;
    for (size_t i = 0; i < MFU_NUM_EXPERTS; i++) {
        m->counts[i] >>= s;
        m->delta_counts[i] >>= s; /* lockstep: the decomposition survives */
    }
    m->overflow_shift += s; /* cumulative scale, persisted for comparability */
}

void mfu_reset_deltas(MfuTable *m) {
    memset(m->delta_counts, 0, sizeof(m->delta_counts));
}

int mfu_save(const MfuTable *m, const char *path) {
    if (!m || !path || !*path) return -1;

    char tmp[4200];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) return -1;

    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int rc = -1;
    do {
        if (fwrite(MFU_FILE_MAGIC, 1, 4, f) != 4) break;
        wr_u32le(f, MFU_FILE_VERSION);
        wr_u32le(f, m->overflow_shift);
        wr_u64le(f, m->total_accesses);
        for (size_t i = 0; i < MFU_NUM_EXPERTS; i++) wr_u32le(f, m->counts[i]);
        if (ferror(f)) break;
        if (fclose(f) != 0) { f = NULL; break; }
        f = NULL;
        if (rename(tmp, path) != 0) break;
        rc = 0;
    } while (0);
    if (f) fclose(f);
    if (rc != 0) remove(tmp); /* leave no half-written .tmp behind */
    return rc;
}

int mfu_load(MfuTable *m, const char *path) {
    if (!m || !path || !*path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    /* Parse into locals first: the table is touched only on full success, so
     * a corrupt/truncated file leaves the caller's zeroed cold-start table
     * exactly as mfu_init made it. */
    uint32_t version = 0, overflow_shift = 0;
    uint64_t total = 0;
    uint32_t counts[MFU_NUM_EXPERTS];
    int ok = 0;

    char magic[4];
    if (fread(magic, 1, 4, f) == 4 && memcmp(magic, MFU_FILE_MAGIC, 4) == 0 &&
        rd_u32le(f, &version) == 0 && version == MFU_FILE_VERSION &&
        rd_u32le(f, &overflow_shift) == 0 && rd_u64le(f, &total) == 0) {
        ok = 1;
        for (size_t i = 0; i < MFU_NUM_EXPERTS; i++) {
            if (rd_u32le(f, &counts[i]) != 0) { ok = 0; break; } /* truncated */
        }
    }
    fclose(f);
    if (!ok) return -1;

    /* Load decay — OFF by default (MFU_DECAY_SHIFT_DEFAULT = 0): the
     * persisted counts are the next boot's warm-fill seed in full (GPU
     * fill-to-free prefill + CPU pool warm both rank this table); only an
     * explicit COLI_DSV4_VK_MFU_DECAY_SHIFT=N>0 opts back into the old
     * recency semantics (last request's traffic dominates, old history only
     * biases). */
    unsigned s = m->decay_shift; /* 0 = no decay; >= 32 clamped to 0 */
    if (s >= 32) s = 0;
    if (s) {
        for (size_t i = 0; i < MFU_NUM_EXPERTS; i++) counts[i] >>= s;
        m->overflow_shift = overflow_shift + s; /* this decay is part of the
                                                   scale */
    } else {
        m->overflow_shift = overflow_shift;
    }
    m->total_accesses = total;
    memcpy(m->counts, counts, sizeof(counts));
    memset(m->delta_counts, 0, sizeof(m->delta_counts)); /* fresh request */
    return 0;
}
