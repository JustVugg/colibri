#ifndef COLIBRI_SEGMENT_ADAPTER_INTERNAL_H
#define COLIBRI_SEGMENT_ADAPTER_INTERNAL_H

/* Shared implementation details for the engine-owned Segment adapters.
 * This is not a public ABI.  Snapshot payloads remain adapter-private and are
 * accepted only after the engine/range/context identity has been checked. */

#include "segment_runtime.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_SEGMENT_SNAPSHOT_VERSION 1u
#define COLI_SEGMENT_HASH_INIT UINT64_C(1469598103934665603)

typedef struct {
    unsigned char magic[8];
    uint32_t format_version;
    uint32_t header_size;
    char engine_id[COLI_SEGMENT_ENGINE_ID_CAP];
    uint32_t layer_begin;
    uint32_t layer_end;
    uint32_t context_tokens;
    uint32_t position;
    uint64_t payload_bytes;
    uint64_t payload_hash;
    uint64_t reserved[4];
} ColiSegmentSnapshotHeader;

typedef struct {
    void *data;
    size_t size;
} ColiSegmentStateSpan;

static int coli_segment_adapter_error(char *error, size_t error_size,
                                      const char *format, ...) {
    if (error && error_size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_size, format, args);
        va_end(args);
    }
    return -1;
}

static int coli_segment_size_mul(size_t left, size_t right, size_t *result) {
    if (!result || (right && left > SIZE_MAX / right)) return -1;
    *result = left * right;
    return 0;
}

static uint64_t coli_segment_hash_update(uint64_t hash, const void *data,
                                         size_t size) {
    const unsigned char *bytes = (const unsigned char *)data;
    if (!hash) hash = COLI_SEGMENT_HASH_INIT;
    for (size_t index = 0; index < size; index++) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int coli_segment_stream_write(ColiSegmentWriteFn write_fn,
                                     void *user_data, const void *data,
                                     size_t size, char *error,
                                     size_t error_size) {
    if (size && write_fn(user_data, data, size))
        return coli_segment_adapter_error(error, error_size,
                                          "segment snapshot write failed");
    return 0;
}

static int coli_segment_stream_read(ColiSegmentReadFn read_fn,
                                    void *user_data, void *data, size_t size,
                                    char *error, size_t error_size) {
    if (size && read_fn(user_data, data, size))
        return coli_segment_adapter_error(error, error_size,
                                          "segment snapshot read failed");
    return 0;
}

static void coli_segment_snapshot_header_init(
    ColiSegmentSnapshotHeader *header, const char *engine_id,
    uint32_t layer_begin, uint32_t layer_end, uint32_t context_tokens,
    uint32_t position, uint64_t payload_bytes, uint64_t payload_hash) {
    static const unsigned char magic[8] = {'C','O','L','I','S','E','G','1'};
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, magic, sizeof(magic));
    header->format_version = COLI_SEGMENT_SNAPSHOT_VERSION;
    header->header_size = (uint32_t)sizeof(*header);
    snprintf(header->engine_id, sizeof(header->engine_id), "%s", engine_id);
    header->layer_begin = layer_begin;
    header->layer_end = layer_end;
    header->context_tokens = context_tokens;
    header->position = position;
    header->payload_bytes = payload_bytes;
    header->payload_hash = payload_hash;
}

static int coli_segment_snapshot_header_valid(
    const ColiSegmentSnapshotHeader *header, const char *engine_id,
    uint32_t layer_begin, uint32_t layer_end, uint32_t context_tokens,
    uint64_t payload_bytes, char *error, size_t error_size) {
    static const unsigned char magic[8] = {'C','O','L','I','S','E','G','1'};
    if (!header || memcmp(header->magic, magic, sizeof(magic)) ||
        header->format_version != COLI_SEGMENT_SNAPSHOT_VERSION ||
        header->header_size != sizeof(*header) ||
        strncmp(header->engine_id, engine_id,
                sizeof(header->engine_id)) ||
        header->layer_begin != layer_begin || header->layer_end != layer_end ||
        header->context_tokens != context_tokens ||
        header->position > context_tokens ||
        header->payload_bytes != payload_bytes)
        return coli_segment_adapter_error(
            error, error_size, "segment snapshot identity is incompatible");
    return 0;
}

static void coli_segment_capability_string(char *output, size_t capacity,
                                           const char *value) {
    if (!output || !capacity) return;
    snprintf(output, capacity, "%s", value ? value : "");
}

static int coli_segment_spans_size(const ColiSegmentStateSpan *spans,
                                   size_t count, size_t *bytes) {
    if (!bytes || (count && !spans)) return -1;
    size_t total = 0;
    for (size_t index = 0; index < count; index++) {
        if (spans[index].size > SIZE_MAX - total) return -1;
        total += spans[index].size;
    }
    *bytes = total;
    return 0;
}

static uint64_t coli_segment_spans_hash(const ColiSegmentStateSpan *spans,
                                        size_t count) {
    uint64_t hash = COLI_SEGMENT_HASH_INIT;
    for (size_t index = 0; index < count; index++)
        hash = coli_segment_hash_update(hash, spans[index].data,
                                        spans[index].size);
    return hash;
}

static int coli_segment_spans_write(const ColiSegmentStateSpan *spans,
                                    size_t count, ColiSegmentWriteFn write_fn,
                                    void *user_data, char *error,
                                    size_t error_size) {
    for (size_t index = 0; index < count; index++)
        if (coli_segment_stream_write(write_fn, user_data, spans[index].data,
                                      spans[index].size, error, error_size))
            return -1;
    return 0;
}

typedef int (*ColiSegmentPayloadValidateFn)(
    const ColiSegmentStateSpan *spans, size_t count,
    const unsigned char *payload, size_t payload_bytes,
    void *user_data, char *error, size_t error_size);

/* Read into a compact temporary buffer, validate the complete payload, then
 * commit to live state.  A failed or truncated restore therefore leaves the
 * conversation byte-for-byte unchanged.  The optional validator lets an
 * adapter reject semantically impossible state while it is still staged. */
static int coli_segment_spans_restore_checked(
    const ColiSegmentStateSpan *spans, size_t count, uint64_t expected_hash,
    ColiSegmentReadFn read_fn, void *read_user_data,
    ColiSegmentPayloadValidateFn validate_fn, void *validate_user_data,
    char *error, size_t error_size) {
    size_t bytes;
    if (coli_segment_spans_size(spans, count, &bytes))
        return coli_segment_adapter_error(error, error_size,
                                           "segment snapshot size overflow");
    unsigned char *payload = bytes ? (unsigned char *)malloc(bytes) : NULL;
    if (bytes && !payload)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory restoring segment state");
    if (coli_segment_stream_read(read_fn, read_user_data, payload, bytes,
                                 error, error_size)) {
        free(payload);
        return -1;
    }
    if (coli_segment_hash_update(COLI_SEGMENT_HASH_INIT, payload, bytes) !=
        expected_hash) {
        free(payload);
        return coli_segment_adapter_error(error, error_size,
                                           "segment snapshot checksum mismatch");
    }
    if (validate_fn && validate_fn(spans, count, payload, bytes,
                                   validate_user_data, error, error_size)) {
        free(payload);
        return -1;
    }
    const unsigned char *cursor = payload;
    for (size_t index = 0; index < count; index++) {
        if (spans[index].size) {
            memcpy(spans[index].data, cursor, spans[index].size);
            cursor += spans[index].size;
        }
    }
    free(payload);
    return 0;
}

static int coli_segment_spans_restore(const ColiSegmentStateSpan *spans,
                                      size_t count, uint64_t expected_hash,
                                      ColiSegmentReadFn read_fn,
                                      void *user_data, char *error,
                                      size_t error_size) {
    return coli_segment_spans_restore_checked(
        spans, count, expected_hash, read_fn, user_data, NULL, NULL,
        error, error_size);
}

#endif /* COLIBRI_SEGMENT_ADAPTER_INTERNAL_H */
