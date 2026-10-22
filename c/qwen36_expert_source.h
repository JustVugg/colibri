/* Checkpoint stamp for experts.qc. This is cache coherence, not authentication.
 * All indexed expert descriptors enter the stamp. Each distinct source shard
 * contributes its inode, size, mtime and ctime (qes_stat_stamp) and two 128-byte
 * samples from its first expert tensor (weight or scale). No full-model scan or
 * per-expert sampling occurs. Replacing a byte-identical shard also invalidates the
 * cache intentionally. Payload corruption that preserves this metadata and samples is
 * not detected. The device number stays out: macOS can give a volume another one at a
 * restart, and the file it stamped would then be stale for no change at all. */
#ifndef COLI_QWEN36_EXPERT_SOURCE_H
#define COLI_QWEN36_EXPERT_SOURCE_H
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "st.h"
#include "qwen36_expert_file.h"

static inline int qes_error(char *error, size_t cap, const char *format, ...) {
    if (error && cap) {
        va_list args; va_start(args, format); vsnprintf(error, cap, format, args); va_end(args);
    }
    return 0;
}

static inline uint64_t qes_hash(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *p = data;
    for (size_t i = 0; i < bytes; i++) hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

static inline int qes_read_at(int fd, void *output, size_t bytes, uint64_t offset,
                              char *error, size_t cap) {
    if (fd < 0 || (!output && bytes) || offset > INT64_MAX || bytes > (uint64_t)INT64_MAX - offset)
        return qes_error(error, cap, "invalid source read extent");
    if ((uint64_t)(off_t)offset != offset ||
        (bytes && (uint64_t)(off_t)(offset + bytes - 1) != offset + bytes - 1))
        return qes_error(error, cap, "source read extent does not fit off_t");
    size_t done = 0;
    while (done < bytes) {
        size_t part = bytes - done;
        if (part > (size_t)SSIZE_MAX) part = (size_t)SSIZE_MAX;
        ssize_t got = pread(fd, (char *)output + done, part, (off_t)(offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return qes_error(error, cap, "source read at %llu: %s",
                (unsigned long long)(offset + done), got < 0 ? strerror(errno) : "unexpected end of file");
        done += (size_t)got;
    }
    return 1;
}

static inline int qes_sources(shards *source, const QefGeometry *geometry, uint32_t layer, uint32_t expert,
                              const st_tensor **weights, const st_tensor **scales, char *error, size_t cap) {
    QefHeader layout;
    if (!source || !weights || !scales || !qef_layout(geometry, 1, &layout, error, cap)) return 0;
    if (layer >= geometry->layers || expert >= geometry->experts)
        return qes_error(error, cap, "expert source key is outside the model");
    char name[160];
    snprintf(name, sizeof name, "model.layers.%u.mlp.experts.%u.merged_weight", layer, expert);
    const st_tensor *w = st_find(source, name);
    snprintf(name, sizeof name, "model.layers.%u.mlp.experts.%u.qs", layer, expert);
    const st_tensor *s = st_find(source, name);
    if (!w || w->dtype != 3 || w->nbytes != (int64_t)layout.packed_bytes ||
        w->numel != (int64_t)layout.packed_bytes || w->rank != 1 || w->shape[0] != w->numel || w->off < 0)
        return qes_error(error, cap, "layer %u expert %u requires a packed INT4 U8 vector", layer, expert);
    if (!s || s->dtype != 2 || s->numel != (int64_t)layout.scale_count ||
        s->nbytes != (int64_t)(layout.scale_count * sizeof(float)) ||
        s->rank != 1 || s->shape[0] != s->numel || s->off < 0)
        return qes_error(error, cap, "layer %u expert %u requires native F32 group scales", layer, expert);
    *weights = w; *scales = s;
    return 1;
}

/* What a source shard's stat contributes to the stamp: what a restart leaves alone. */
static inline void qes_stat_stamp(const struct stat *info, uint64_t stamp[6]) {
    int64_t mt_sec = info->st_mtime, ct_sec = info->st_ctime, mt_ns = 0, ct_ns = 0;
#if defined(__APPLE__)
    mt_ns = info->st_mtimespec.tv_nsec; ct_ns = info->st_ctimespec.tv_nsec;
#elif !defined(_WIN32)
    mt_ns = info->st_mtim.tv_nsec; ct_ns = info->st_ctim.tv_nsec;
#endif
    stamp[0] = (uint64_t)info->st_ino; stamp[1] = (uint64_t)info->st_size;
    stamp[2] = (uint64_t)mt_sec; stamp[3] = (uint64_t)mt_ns; stamp[4] = (uint64_t)ct_sec; stamp[5] = (uint64_t)ct_ns;
}

static inline uint64_t qes_fingerprint(shards *source, const QefGeometry *geometry, char *error, size_t cap) {
    QefHeader layout;
    if (!source || !qef_layout(geometry, 1, &layout, error, cap)) return 0;
    static const char scheme[] = "qwen36-expert-planar-source-stat-sample-v1";
    uint64_t hash = qes_hash(UINT64_C(14695981039346656037), scheme, sizeof scheme);
    uint32_t shape[] = {geometry->layers, geometry->experts, geometry->hidden, geometry->inter, geometry->group};
    hash = qes_hash(hash, shape, sizeof shape);
    int seen_fd[ST_MAX_SHARDS]; uint64_t seen_size[ST_MAX_SHARDS]; unsigned seen = 0;
    for (uint32_t layer = 0; layer < geometry->layers; layer++) {
        for (uint32_t expert = 0; expert < geometry->experts; expert++) {
            const st_tensor *weights, *scales;
            if (!qes_sources(source, geometry, layer, expert, &weights, &scales, error, cap)) return 0;
            const st_tensor *tensors[] = {weights, scales};
            for (unsigned part = 0; part < 2; part++) {
                const st_tensor *t = tensors[part];
                unsigned shard = 0;
                while (shard < seen && seen_fd[shard] != t->fd) shard++;
                if (shard == seen) {
                    struct stat info;
                    if (seen == ST_MAX_SHARDS || t->fd < 0 || fstat(t->fd, &info) || info.st_size < 0) {
                        qes_error(error, cap, "cannot identify source shard for %s", t->name); return 0;
                    }
                    uint64_t stamp[6];
                    qes_stat_stamp(&info, stamp);
                    hash = qes_hash(hash, stamp, sizeof stamp);
                    seen_fd[seen] = t->fd; seen_size[seen++] = (uint64_t)info.st_size;
                    if ((uint64_t)t->off > seen_size[shard] || (uint64_t)t->nbytes > seen_size[shard] - (uint64_t)t->off) {
                        qes_error(error, cap, "%s exceeds its source shard", t->name); return 0;
                    }
                    unsigned char sample[256];
                    size_t n = (uint64_t)t->nbytes < 128 ? (size_t)t->nbytes : 128;
                    if (!qes_read_at(t->fd, sample, n, (uint64_t)t->off, error, cap) ||
                        !qes_read_at(t->fd, sample + n, n, (uint64_t)t->off + (uint64_t)t->nbytes - n, error, cap)) return 0;
                    hash = qes_hash(hash, sample, 2 * n);
                }
                if ((uint64_t)t->off > seen_size[shard] || (uint64_t)t->nbytes > seen_size[shard] - (uint64_t)t->off) {
                    qes_error(error, cap, "%s exceeds its source shard", t->name); return 0;
                }
                uint64_t descriptor[] = {shard, (uint64_t)t->dtype, (uint64_t)t->rank, (uint64_t)t->numel,
                                         (uint64_t)t->off, (uint64_t)t->nbytes};
                hash = qes_hash(hash, t->name, strlen(t->name) + 1);
                hash = qes_hash(hash, descriptor, sizeof descriptor);
                hash = qes_hash(hash, t->shape, (size_t)t->rank * sizeof(t->shape[0]));
            }
        }
    }
    return hash ? hash : 1;
}
#endif
