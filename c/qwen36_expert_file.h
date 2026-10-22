/* Persistent, uniform Qwen expert records: planar INT4 gs64 followed by the
 * scales, in (layer, expert) order. Header integers and F32 use the host ABI.
 * The header is exactly 128 bytes with no padding.
 *
 * Layout 1 stores the scales as F32. Layout 2 stores each as a 16-bit code
 * when the quantizer's scale is max|w| / 7 with a bfloat16 max|w|: the code is
 * that bfloat16, 0 the 1e-12 / 7 of an all-zero group, and the reader computes
 * the scale back with the same F32 division, so a record reads into the same
 * bytes either way, 5.6% fewer from disk on the 35B model. The converter
 * writes layout 2 only when every scale encodes exactly.
 *
 * Validation binds the geometry/layout to a caller-supplied checkpoint
 * fingerprint and rejects truncated or overlong files by their exact size.
 * It is not payload authentication: opening does not scan weights or scales
 * and cannot detect unwritten/corrupted payload with the expected file size.
 *
 * Initialize QefFile to zero before its first use. Calls return 1 on success,
 * 0 on failure. A failed read may have modified the destination; publish a
 * cache slot only after success. Concurrent reads are safe while the caller
 * keeps the file open and its contents immutable; close requires drained I/O.
 */
#ifndef QWEN36_EXPERT_FILE_H
#define QWEN36_EXPERT_FILE_H

#include <stdio.h>
#include "compat.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define QEF_MAGIC "QWEN36E4"
#define QEF_VERSION 1u
#define QEF_LAYOUT 1u                   /* F32 scales */
#define QEF_LAYOUT_CODES 2u             /* 16-bit scale codes (see above) */
#define QEF_ALIGN UINT64_C(16384)

typedef struct {
    uint32_t layers, experts, hidden, inter, group;
} QefGeometry;

typedef struct {
    char magic[8];
    uint32_t version, layout, layers, experts, hidden, inter, group, reserved0;
    uint64_t fingerprint, count, packed_bytes, scale_count;
    uint64_t record_bytes, stride, data_offset, bytes;
    uint64_t reserved[3];
} QefHeader;

typedef struct {
    int fd, opened;
    QefHeader h;
    float *scales;                      /* layout 2: the scale of every code */
} QefFile;

typedef char qef_header_size_is_128[(sizeof(QefHeader) == 128) ? 1 : -1];
typedef char qef_header_fingerprint_at_40[
    (offsetof(QefHeader, fingerprint) == 40) ? 1 : -1];
typedef char qef_header_reserved_at_104[
    (offsetof(QefHeader, reserved) == 104) ? 1 : -1];
typedef char qef_float_size_is_4[(sizeof(float) == 4) ? 1 : -1];

static inline int qef_error(char *error, size_t cap, const char *message) {
    if (error && cap) snprintf(error, cap, "%s", message);
    return 0;
}

static inline int qef_io_error(char *error, size_t cap, const char *operation) {
    int saved_errno = errno;
    if (error && cap)
        snprintf(error, cap, "%s: %s", operation, strerror(saved_errno));
    return 0;
}

static inline int qef_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b && a > UINT64_MAX / b) return 0;
    *out = a * b;
    return 1;
}

static inline int qef_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 0;
    *out = a + b;
    return 1;
}

/* The scale of a layout-2 code, and the code of a scale (0 when it has none). */
static inline float qef_eps_scale(void) { return 1e-12f / 7.0f; }

static inline float qef_code_scale(uint16_t code) {
    uint32_t bits = (uint32_t)code << 16;
    float b;
    if (!code) return qef_eps_scale();
    memcpy(&b, &bits, sizeof b);
    return b / 7.0f;
}

static inline int qef_scale_code(float s, uint16_t *code) {
    float b = s * 7.0f;
    uint32_t bits;
    memcpy(&bits, &s, sizeof bits);
    if ((bits & 0x7f800000u) == 0x7f800000u) return 0;      /* infinities and NaNs have none */
    if (s == qef_eps_scale()) {
        *code = 0;
        return 1;
    }
    memcpy(&bits, &b, sizeof bits);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;   /* nearest bfloat16 */
    if (!bits || qef_code_scale((uint16_t)bits) != s) return 0;
    *code = (uint16_t)bits;
    return 1;
}

static inline int qef_layout_as(const QefGeometry *g, uint64_t fingerprint, uint32_t layout,
                                QefHeader *out, char *error, size_t cap) {
    QefHeader h;
    uint64_t elements, scale_bytes, payload_bytes, rounded;
    if (error && cap) error[0] = '\0';
    if (!g || !out) return qef_error(error, cap, "null layout argument");
    if (layout != QEF_LAYOUT && layout != QEF_LAYOUT_CODES) return qef_error(error, cap, "unknown expert file layout");
    if (!g->layers || !g->experts || !g->hidden || !g->inter ||
        g->group != 64 || g->hidden % 64 || g->inter % 64)
        return qef_error(error, cap, "geometry requires positive dimensions and gs64 multiples");
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QEF_MAGIC, sizeof(h.magic));
    h.version = QEF_VERSION;
    h.layout = layout;
    h.layers = g->layers;
    h.experts = g->experts;
    h.hidden = g->hidden;
    h.inter = g->inter;
    h.group = g->group;
    h.fingerprint = fingerprint;
    if (!qef_mul_u64(g->layers, g->experts, &h.count) || h.count > INT32_MAX)
        return qef_error(error, cap, "expert count exceeds INT32_MAX");
    if (!qef_mul_u64(g->hidden, g->inter, &elements) ||
        !qef_mul_u64(elements, 3, &elements))
        return qef_error(error, cap, "expert element count overflow");
    h.packed_bytes = elements / 2;
    h.scale_count = elements / 64;
    if (!qef_mul_u64(h.scale_count, layout == QEF_LAYOUT_CODES ? 2 : 4, &scale_bytes) ||
        !qef_add_u64(h.packed_bytes, scale_bytes, &h.record_bytes) ||
        !qef_add_u64(h.record_bytes, QEF_ALIGN - 1, &rounded))
        return qef_error(error, cap, "expert record size overflow");
    h.stride = rounded & ~(QEF_ALIGN - 1);
    h.data_offset = QEF_ALIGN;
    if (!qef_mul_u64(h.count, h.stride, &payload_bytes) ||
        !qef_add_u64(h.data_offset, payload_bytes, &h.bytes))
        return qef_error(error, cap, "expert file size overflow");
    if (h.bytes > (uint64_t)INT64_MAX || h.bytes > (uint64_t)SIZE_MAX ||
        (off_t)h.bytes < 0 || (uint64_t)(off_t)h.bytes != h.bytes ||
        h.record_bytes > (uint64_t)(SIZE_MAX >> 1))
        return qef_error(error, cap, "expert file or read size exceeds platform limits");
    *out = h;
    return 1;
}

static inline int qef_layout(const QefGeometry *g, uint64_t fingerprint,
                             QefHeader *out, char *error, size_t cap) {
    return qef_layout_as(g, fingerprint, QEF_LAYOUT, out, error, cap);
}

/* What qef_read writes: the weights and the F32 scales, in either layout. */
static inline uint64_t qef_payload_bytes(const QefHeader *h) {
    return h->packed_bytes + h->scale_count * 4;
}

/* The normal path is one pread. A signal or a short read is the only reason
 * to issue another request; do not split the record into weights and scales. */
static inline int qef_pread_all(int fd, void *out, size_t bytes,
                                uint64_t offset, char *error, size_t cap) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, (unsigned char *)out + done, bytes - done,
                            (off_t)(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            return qef_io_error(error, cap, "read expert file");
        }
        if (!got) return qef_error(error, cap, "unexpected EOF in expert file");
        done += (size_t)got;
    }
    return 1;
}

static inline int qef_open(QefFile *file, const char *path,
                           const QefGeometry *geometry, uint64_t fingerprint,
                           char *error, size_t cap) {
    QefHeader expected, actual;
    struct stat sb;
    int fd;
    if (error && cap) error[0] = '\0';
    if (!file || !path || !geometry)
        return qef_error(error, cap, "null open argument");
    if (file->opened) return qef_error(error, cap, "expert file is already open");
    if (!qef_layout(geometry, fingerprint, &expected, error, cap)) return 0;
    fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) return qef_io_error(error, cap, "open expert file");
    if (fstat(fd, &sb)) {
        qef_io_error(error, cap, "stat expert file");
        close(fd);
        return 0;
    }
    if (!S_ISREG(sb.st_mode) || sb.st_size < (off_t)sizeof(actual)) {
        close(fd);
        return qef_error(error, cap, "expert file size or type mismatch");
    }
    if (!qef_pread_all(fd, &actual, sizeof(actual), 0, error, cap)) {
        close(fd);
        return 0;
    }
    /* The layout field picks which header to rebuild; any other value
     * rebuilds layout 1's and fails the comparison below. */
    if (actual.layout == QEF_LAYOUT_CODES &&
        !qef_layout_as(geometry, fingerprint, QEF_LAYOUT_CODES, &expected, error, cap)) {
        close(fd);
        return 0;
    }
    if ((uint64_t)sb.st_size != expected.bytes) {
        close(fd);
        return qef_error(error, cap, "expert file size or type mismatch");
    }
    /* Rebuilding also checks all reserved fields, stride/alignment and every
     * redundant size; there are no unchecked on-disk offsets to trust. */
    if (memcmp(&actual, &expected, sizeof(actual))) {
        close(fd);
        return qef_error(error, cap, "expert file header, geometry or fingerprint mismatch");
    }
    if (expected.layout == QEF_LAYOUT_CODES) {
        file->scales = (float *)malloc(65536 * sizeof(float));
        if (!file->scales) {
            close(fd);
            return qef_error(error, cap, "cannot allocate the scale table");
        }
        for (uint32_t c = 0; c < 65536; c++) file->scales[c] = qef_code_scale((uint16_t)c);
    }
    file->fd = fd;
    file->h = expected;
    file->opened = 1;
    return 1;
}

static inline int qef_read(const QefFile *file, uint32_t layer, uint32_t expert,
                           void *out, size_t capacity, char *error, size_t cap) {
    uint64_t key, offset;
    if (error && cap) error[0] = '\0';
    if (!file || !file->opened || !out)
        return qef_error(error, cap, "expert file is closed or destination is null");
    if (layer >= file->h.layers || expert >= file->h.experts)
        return qef_error(error, cap, "expert key is outside the geometry");
    if (capacity < qef_payload_bytes(&file->h))
        return qef_error(error, cap, "expert destination capacity is too small");
    /* The immutable, reconstructed header proved these products and the full
     * final record fit uint64_t, off_t and size_t during open. */
    key = (uint64_t)layer * file->h.experts + expert;
    offset = file->h.data_offset + key * file->h.stride;
    if (!qef_pread_all(file->fd, out, (size_t)file->h.record_bytes, offset, error, cap)) return 0;
    if (file->h.layout == QEF_LAYOUT_CODES) {
        /* Codes to scales in place, from the last: scale i is written over
         * codes 2i and 2i + 1, which are read already. */
        unsigned char *base = (unsigned char *)out + file->h.packed_bytes;
        for (uint64_t i = file->h.scale_count; i-- > 0;) {
            uint16_t code;
            memcpy(&code, base + 2 * i, sizeof code);
            memcpy(base + 4 * i, &file->scales[code], sizeof(float));
        }
    }
    return 1;
}

static inline void qef_close(QefFile *file) {
    if (!file) return;
    if (file->opened) close(file->fd);
    free(file->scales);
    memset(file, 0, sizeof(*file));
    file->fd = -1;
}

#endif /* QWEN36_EXPERT_FILE_H */
